//! LQL Executor — dispatches parsed AST statements to backend operations
//!
//! The base vindex is always readonly. All mutations go through a patch overlay.
//! INSERT/DELETE/UPDATE auto-start an anonymous patch session if none is active.

mod helpers;
mod introspection;
mod lifecycle;
mod mutation;
mod query;
mod remote;
mod trace;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::error::LqlError;
use crate::relations::RelationClassifier;

/// The active backend for the session.
/// The base vindex is always loaded readonly. A PatchedVindex overlay
/// handles all mutations without modifying base files on disk.
pub(crate) enum Backend {
    Vindex {
        path: PathBuf,
        config: larql_vindex::VindexConfig,
        /// Patched overlay on the readonly base. All queries and mutations
        /// go through this. The base files on disk are never modified.
        patched: larql_vindex::PatchedVindex,
        relation_classifier: Option<RelationClassifier>,
        /// MoE router index (if available). Used for MoE-aware DESCRIBE.
        router: Option<larql_vindex::RouterIndex>,
    },
    /// Direct model weight access — no vindex extraction needed.
    /// Supports INFER, EXPLAIN INFER, and STATS. Browse/mutation ops
    /// require extraction to a vindex first.
    Weight {
        model_id: String,
        weights: larql_inference::ModelWeights,
        tokenizer: larql_inference::tokenizers::Tokenizer,
    },
    /// Remote server backend — queries forwarded via HTTP.
    /// Local patches can be applied for client-side overlay.
    Remote {
        url: String,
        client: reqwest::blocking::Client,
        local_patches: Vec<larql_vindex::VindexPatch>,
        session_id: String,
    },
    None,
}

/// Session state for the REPL / batch executor.
pub struct Session {
    pub(crate) backend: Backend,
    /// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
    /// If None and a mutation happens, an anonymous patch is auto-started.
    pub(crate) patch_recording: Option<PatchRecording>,
    /// Whether the current patch was auto-started (anonymous).
    pub(crate) auto_patch: bool,
    /// Architecture A: per-layer cached decoy residuals. Kept for backward compat.
    #[allow(dead_code)]
    pub(crate) decoy_residual_cache:
        std::collections::HashMap<usize, Vec<larql_vindex::ndarray::Array1<f32>>>,
    /// Raw captured residuals per installed slot, indexed by
    /// `(layer, feature)`. Each entry is the unscaled residual the
    /// model produced for the install prompt at that layer's FFN
    /// entry (last token), captured on the clean base index before
    /// any installs were applied. Used by INSERT's online refine
    /// pass to rebuild the full constellation from raw inputs each
    /// time a new slot lands — if we refined against the currently
    /// stored (already-refined) peers, compound drift across
    /// iterations would leave the latest insert dominating (see
    /// `refine_demo` 10-fact run where every prompt returned the
    /// last-installed target before this cache existed).
    #[allow(dead_code)]
    pub(crate) raw_install_residuals: std::collections::HashMap<
        (usize, usize),
        larql_vindex::ndarray::Array1<f32>,
    >,
}

/// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
pub(crate) struct PatchRecording {
    pub path: String,
    pub operations: Vec<larql_vindex::PatchOp>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
            patch_recording: None,
            auto_patch: false,
            decoy_residual_cache: std::collections::HashMap::new(),
            raw_install_residuals: std::collections::HashMap::new(),
        }
    }

    /// Ensure a patch session is active. If not, auto-start an anonymous one.
    /// Returns messages about auto-patch start (empty if already active).
    pub(crate) fn ensure_patch_session(&mut self) -> Vec<String> {
        if self.patch_recording.is_some() {
            return vec![];
        }
        self.patch_recording = Some(PatchRecording {
            path: String::new(), // anonymous
            operations: Vec::new(),
        });
        self.auto_patch = true;
        vec!["Auto-patch started (use SAVE PATCH \"file.vlp\" to persist, or edits are lost on exit)".into()]
    }

    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        // Remote backend: forward supported queries via HTTP.
        if self.is_remote() {
            return self.execute_remote(stmt);
        }

        match stmt {
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            Statement::Use { target } => self.exec_use(target),
            Statement::Stats { vindex } => self.exec_stats(vindex.as_deref()),
            Statement::Walk { prompt, top, layers, mode, compare } => {
                self.exec_walk(prompt, *top, layers.as_ref(), *mode, *compare)
            }
            Statement::Describe { entity, band, layer, relations_only, mode } => {
                self.exec_describe(entity, *band, *layer, *relations_only, *mode)
            }
            Statement::Select { source, fields, conditions, nearest, order, limit } => {
                match source {
                    SelectSource::Edges => self.exec_select(fields, conditions, nearest.as_ref(), order.as_ref(), *limit),
                    SelectSource::Features => self.exec_select_features(conditions, *limit),
                    SelectSource::Entities => self.exec_select_entities(conditions, *limit),
                }
            }
            Statement::Explain { prompt, mode, layers, band, verbose, top, relations_only, with_attention } => {
                match mode {
                    ExplainMode::Walk => self.exec_explain(prompt, layers.as_ref(), *verbose),
                    ExplainMode::Infer => self.exec_infer_trace(prompt, *top, *band, *relations_only, *with_attention),
                }
            }
            Statement::ShowRelations { layer, with_examples, mode } => {
                self.exec_show_relations(*layer, *with_examples, *mode)
            }
            Statement::ShowLayers { range } => self.exec_show_layers(range.as_ref()),
            Statement::ShowFeatures { layer, conditions, limit } => {
                self.exec_show_features(*layer, conditions, *limit)
            }
            Statement::ShowEntities { layer, limit } => {
                self.exec_show_entities(*layer, *limit)
            }
            Statement::ShowModels => self.exec_show_models(),
            Statement::Extract { model, output, components, layers, extract_level } => {
                self.exec_extract(model, output, components.as_deref(), layers.as_ref(), *extract_level)
            }
            Statement::Compile { vindex, output, format, target, on_conflict } => {
                self.exec_compile(
                    vindex, output, *format, *target, *on_conflict,
                )
            }
            Statement::Diff { a, b, layer, relation, limit, into_patch } => {
                self.exec_diff(a, b, *layer, relation.as_deref(), *limit, into_patch.as_deref())
            }
            Statement::Insert { entity, relation, target, layer, confidence, alpha } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_insert(
                    entity, relation, target,
                    *layer, *confidence, *alpha,
                )?);
                Ok(out)
            }
            Statement::Infer { prompt, top, compare } => {
                self.exec_infer(prompt, *top, *compare)
            }
            Statement::Delete { conditions } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_delete(conditions)?);
                Ok(out)
            }
            Statement::Update { set, conditions } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_update(set, conditions)?);
                Ok(out)
            }
            Statement::Merge { source, target, conflict } => {
                self.exec_merge(source, target.as_deref(), *conflict)
            }
            // ── Patch commands ──
            Statement::BeginPatch { path } => self.exec_begin_patch(path),
            Statement::SavePatch => self.exec_save_patch(),
            Statement::ApplyPatch { path } => self.exec_apply_patch(path),
            Statement::ShowPatches => self.exec_show_patches(),
            Statement::RemovePatch { path } => self.exec_remove_patch(path),
            // ── Trace commands ──
            Statement::Trace { prompt, answer, decompose, layers, positions, save } => {
                self.exec_trace(prompt, answer.as_deref(), *decompose, layers.as_ref(), *positions, save.as_deref())
            }
        }
    }

    /// Execute a statement against a remote backend.
    fn execute_remote(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Use { target } => self.exec_use(target),
            Statement::Describe { entity, band, mode, .. } => {
                self.remote_describe(entity, *band, *mode)
            }
            Statement::Walk { prompt, top, layers, .. } => {
                self.remote_walk(prompt, *top, layers.as_ref())
            }
            Statement::Infer { prompt, top, compare } => {
                self.remote_infer(prompt, *top, *compare)
            }
            Statement::Stats { .. } => self.remote_stats(),
            Statement::ShowRelations { mode, with_examples, .. } => self.remote_show_relations(*mode, *with_examples),
            Statement::Insert { entity, relation, target, layer, confidence, alpha: _ } => {
                // Remote backend doesn't forward ALPHA — the HTTP
                // protocol doesn't have a schema for it yet. Local
                // backend honours alpha via `exec_insert`.
                self.remote_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Delete { conditions } => self.remote_delete(conditions),
            Statement::Update { set, conditions } => self.remote_update(set, conditions),
            Statement::Select { source: _, fields: _, conditions, nearest: _, order: _, limit } => {
                self.remote_select(conditions, *limit)
            }
            Statement::Explain { prompt, mode, layers, band, verbose: _, top, relations_only, with_attention } => {
                match mode {
                    ExplainMode::Infer => self.remote_explain_infer(prompt, *top, *band, *relations_only, *with_attention),
                    ExplainMode::Walk => self.remote_walk(prompt, *top, layers.as_ref()),
                }
            }
            Statement::ApplyPatch { path } => self.remote_apply_local_patch(path),
            Statement::ShowPatches => self.remote_show_patches(),
            Statement::RemovePatch { path } => self.remote_remove_local_patch(path),
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            _ => Err(LqlError::Execution(
                "this statement is not supported on a remote backend. \
                 Supported: DESCRIBE, WALK, INFER, EXPLAIN INFER, EXPLAIN WALK, SELECT, STATS, \
                 SHOW RELATIONS, INSERT, DELETE, UPDATE, APPLY PATCH, SHOW PATCHES, REMOVE PATCH, USE. \
                 TRACE requires a local vindex (USE \"path.vindex\")."
                    .into(),
            )),
        }
    }

    // ── Patch execution ──

    fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        if self.patch_recording.is_some() && !self.auto_patch {
            return Err(LqlError::Execution(
                "patch session already active. Run SAVE PATCH or discard first.".into(),
            ));
        }
        // If there was an auto-patch, upgrade it to a named one
        self.patch_recording = Some(PatchRecording {
            path: path.to_string(),
            operations: if self.auto_patch {
                // Keep existing operations from auto-patch
                self.patch_recording.take().map(|r| r.operations).unwrap_or_default()
            } else {
                Vec::new()
            },
        });
        self.auto_patch = false;
        Ok(vec![format!("Patch session started: {path}")])
    }

    fn exec_save_patch(&mut self) -> Result<Vec<String>, LqlError> {
        let recording = self.patch_recording.take().ok_or_else(|| {
            LqlError::Execution("no active patch session. Run BEGIN PATCH first.".into())
        })?;

        if recording.path.is_empty() {
            return Err(LqlError::Execution(
                "anonymous patch session — use SAVE PATCH \"filename.vlp\" or BEGIN PATCH \"filename.vlp\" first.".into(),
            ));
        }

        let model_name = match &self.backend {
            Backend::Vindex { config, .. } => config.model.clone(),
            Backend::Weight { model_id, .. } => model_id.clone(),
            _ => "unknown".into(),
        };

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: model_name,
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: vec![],
            operations: recording.operations,
        };

        let (ins, upd, del) = patch.counts();
        let path = PathBuf::from(&recording.path);
        patch.save(&path)
            .map_err(|e| LqlError::exec("failed to save patch", e))?;

        self.auto_patch = false;

        Ok(vec![format!(
            "Saved: {} ({} inserts, {} updates, {} deletes)",
            path.display(), ins, upd, del,
        )])
    }

    fn exec_apply_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::exec("failed to load patch", e))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        // Apply through the PatchedVindex overlay (base files untouched)
        match &mut self.backend {
            Backend::Vindex { patched, .. } => {
                patched.apply_patch(patch);
            }
            _ => return Err(LqlError::NoBackend),
        }

        Ok(vec![format!(
            "Applied: {path} ({total} operations: {ins} inserts, {upd} updates, {del} deletes)"
        )])
    }

    fn exec_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let patched = self.require_patched()?;
        let mut out = Vec::new();

        if patched.patches.is_empty() && patched.num_overrides() == 0 {
            out.push("  (no patches applied)".into());
        } else {
            for (i, patch) in patched.patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1, name, patch.len(), ins, upd, del,
                ));
            }
            if patched.num_overrides() > 0 && patched.patches.is_empty() {
                out.push(format!("  (anonymous session: {} overrides)", patched.num_overrides()));
            }
            let file_total: usize = patched.patches.iter().map(|p| p.len()).sum();
            let overlay_total = patched.num_overrides();
            if file_total > 0 || overlay_total > 0 {
                out.push(format!("  Total: {} from files, {} in session", file_total, overlay_total));
            }
        }

        if let Some(ref recording) = self.patch_recording {
            let label = if recording.path.is_empty() { "(anonymous)" } else { &recording.path };
            out.push(format!("  Recording: {} ({} ops pending)", label, recording.operations.len()));
        }

        Ok(out)
    }

    fn exec_remove_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patched = match &mut self.backend {
            Backend::Vindex { patched, .. } => patched,
            _ => return Err(LqlError::NoBackend),
        };

        let pos = patched.patches.iter().position(|p| {
            p.description.as_deref() == Some(path)
        });
        match pos {
            Some(i) => {
                patched.remove_patch(i);
                Ok(vec![format!("Removed patch #{}", i + 1)])
            }
            None => Err(LqlError::Execution(format!("patch not found: {path}"))),
        }
    }

    // ── Backend accessors ──

    /// Get readonly access to the patched vindex (base + overlay).
    pub(crate) fn require_patched(
        &self,
    ) -> Result<&larql_vindex::PatchedVindex, LqlError> {
        match &self.backend {
            Backend::Vindex { patched, .. } => Ok(patched),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "this operation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    /// Get mutable access to the patched overlay.
    pub(crate) fn require_patched_mut(
        &mut self,
    ) -> Result<(&Path, &larql_vindex::VindexConfig, &mut larql_vindex::PatchedVindex), LqlError> {
        match &mut self.backend {
            Backend::Vindex { path, config, patched, .. } => Ok((path, config, patched)),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "mutation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    /// Get readonly access to path + config + base index.
    pub(crate) fn require_vindex(
        &self,
    ) -> Result<(&Path, &larql_vindex::VindexConfig, &larql_vindex::PatchedVindex), LqlError>
    {
        match &self.backend {
            Backend::Vindex { path, config, patched, .. } => Ok((path, config, patched)),
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "this operation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }

    pub(crate) fn relation_classifier(&self) -> Option<&RelationClassifier> {
        match &self.backend {
            Backend::Vindex { relation_classifier, .. } => relation_classifier.as_ref(),
            _ => None,
        }
    }

    /// Mutable access to the patch overlay of the current vindex backend,
    /// for tests and benchmarks that need to inject patches without going
    /// through the full INSERT pipeline (which would require a real
    /// tokenizer + relation classifier the synthetic test fixtures don't
    /// carry). Returns `None` if no vindex is loaded. Production code
    /// should go through `INSERT`/`DELETE`/`UPDATE` statements instead.
    pub fn patched_overlay_mut(&mut self) -> Option<&mut larql_vindex::PatchedVindex> {
        match &mut self.backend {
            Backend::Vindex { patched, .. } => Some(patched),
            _ => None,
        }
    }
}

#[allow(dead_code)]
/// Architecture A: canonical decoy prompt set. Kept for backward compat.
///
/// Same set as `experiments/14_vindex_compilation/experiment_vindex_compilation.py`.
/// These prompts span literary, philosophical, poetic, and common
/// completion templates — the canonical bleed targets for a
/// fact-install slot operating at `gate_scale=30`. Capturing residuals
/// at the install layer through the clean base index and
/// orthogonalising the installed gate against those residuals
/// prevents the slot from firing on unrelated prompts.
///
/// The set is hardcoded so every session gets the same decoy
/// defense without user configuration. A future refinement could
/// move this to `EXTRACT ... WITH DECOYS` for per-vindex canonical
/// sets, or let the user override via `INSERT ... WITH DECOYS`, but
/// v0 ships a fixed list that covers the validated reference cases.
pub(crate) const CANONICAL_DECOY_PROMPTS: &[&str] = &[
    "Once upon a time",
    "The quick brown fox",
    "To be or not to be",
    "Water is a",
    "A long time ago",
    "In the beginning",
    "The weather today is",
    "She opened the door and",
    "He looked at the sky",
    "The children played in the",
];

