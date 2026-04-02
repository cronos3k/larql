/// LQL Executor — dispatches parsed AST statements to backend operations.
///
/// The base vindex is always readonly. All mutations go through a patch overlay.
/// INSERT/DELETE/UPDATE auto-start an anonymous patch session if none is active.

mod helpers;
mod introspection;
mod lifecycle;
mod mutation;
mod query;
mod remote;

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
}

/// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
pub(crate) struct PatchRecording {
    pub path: String,
    pub operations: Vec<larql_vindex::PatchOp>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
            patch_recording: None,
            auto_patch: false,
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
            Statement::Describe { entity, band, layer, relations_only, verbose } => {
                self.exec_describe(entity, *band, *layer, *relations_only, *verbose)
            }
            Statement::Select { fields, conditions, nearest, order, limit } => {
                self.exec_select(fields, conditions, nearest.as_ref(), order.as_ref(), *limit)
            }
            Statement::Explain { prompt, mode, layers, verbose, top } => {
                match mode {
                    ExplainMode::Walk => self.exec_explain(prompt, layers.as_ref(), *verbose),
                    ExplainMode::Infer => self.exec_infer_trace(prompt, *top),
                }
            }
            Statement::ShowRelations { layer, with_examples } => {
                self.exec_show_relations(*layer, *with_examples)
            }
            Statement::ShowLayers { range } => self.exec_show_layers(range.as_ref()),
            Statement::ShowFeatures { layer, conditions, limit } => {
                self.exec_show_features(*layer, conditions, *limit)
            }
            Statement::ShowModels => self.exec_show_models(),
            Statement::Extract { model, output, components, layers, extract_level } => {
                self.exec_extract(model, output, components.as_deref(), layers.as_ref(), *extract_level)
            }
            Statement::Compile { vindex, output, format, target } => {
                self.exec_compile(vindex, output, *format, *target)
            }
            Statement::Diff { a, b, layer, relation, limit, into_patch } => {
                self.exec_diff(a, b, *layer, relation.as_deref(), *limit, into_patch.as_deref())
            }
            Statement::Insert { entity, relation, target, layer, confidence } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_insert(entity, relation, target, *layer, *confidence)?);
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
        }
    }

    /// Execute a statement against a remote backend.
    fn execute_remote(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Use { target } => self.exec_use(target),
            Statement::Describe { entity, band, verbose, .. } => {
                self.remote_describe(entity, *band, *verbose)
            }
            Statement::Walk { prompt, top, layers, .. } => {
                self.remote_walk(prompt, *top, layers.as_ref())
            }
            Statement::Infer { prompt, top, compare } => {
                self.remote_infer(prompt, *top, *compare)
            }
            Statement::Stats { .. } => self.remote_stats(),
            Statement::ShowRelations { .. } => self.remote_show_relations(),
            Statement::Insert { entity, relation, target, layer, confidence } => {
                self.remote_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Delete { conditions } => self.remote_delete(conditions),
            Statement::Update { set, conditions } => self.remote_update(set, conditions),
            Statement::Select { fields: _, conditions, nearest: _, order: _, limit } => {
                self.remote_select(conditions, *limit)
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
                 Supported: DESCRIBE, WALK, INFER, SELECT, STATS, SHOW RELATIONS, \
                 INSERT, DELETE, UPDATE, APPLY PATCH, SHOW PATCHES, REMOVE PATCH, USE"
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
            .map_err(|e| LqlError::Execution(format!("failed to save patch: {e}")))?;

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
            .map_err(|e| LqlError::Execution(format!("failed to load patch: {e}")))?;

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
                model_id.split('/').last().unwrap_or(model_id),
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
                model_id.split('/').last().unwrap_or(model_id),
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
                model_id.split('/').last().unwrap_or(model_id),
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
}
