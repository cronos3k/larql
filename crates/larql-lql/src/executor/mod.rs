/// LQL Executor — dispatches parsed AST statements to backend operations.
///
/// Manages session state (active vindex/model) and formats output.

mod helpers;
mod introspection;
mod lifecycle;
mod mutation;
mod query;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::error::LqlError;
use crate::relations::RelationClassifier;

/// The active backend for the session.
pub(crate) enum Backend {
    /// Pre-extracted vindex — fast, supports mutation.
    Vindex {
        path: PathBuf,
        config: larql_vindex::VindexConfig,
        index: larql_vindex::VectorIndex,
        relation_classifier: Option<RelationClassifier>,
    },
    /// No backend loaded.
    None,
}

/// Session state for the REPL / batch executor.
pub struct Session {
    pub(crate) backend: Backend,
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
        }
    }

    /// Execute a single LQL statement, returning formatted output lines.
    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
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
            Statement::Compile { vindex, output, format } => {
                self.exec_compile(vindex, output, *format)
            }
            Statement::Diff { a, b, layer, relation, limit } => {
                self.exec_diff(a, b, *layer, relation.as_deref(), *limit)
            }
            Statement::Insert { entity, relation, target, layer, confidence } => {
                self.exec_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Infer { prompt, top, compare } => {
                self.exec_infer(prompt, *top, *compare)
            }
            Statement::Delete { conditions } => self.exec_delete(conditions),
            Statement::Update { set, conditions } => self.exec_update(set, conditions),
            Statement::Merge { source, target, conflict } => {
                self.exec_merge(source, target.as_deref(), *conflict)
            }
        }
    }

    // ── Backend accessors ──

    pub(crate) fn require_vindex(
        &self,
    ) -> Result<(&Path, &larql_vindex::VindexConfig, &larql_vindex::VectorIndex), LqlError>
    {
        match &self.backend {
            Backend::Vindex {
                path,
                config,
                index,
                ..
            } => Ok((path, config, index)),
            Backend::None => Err(LqlError::NoBackend),
        }
    }

    pub(crate) fn relation_classifier(&self) -> Option<&RelationClassifier> {
        match &self.backend {
            Backend::Vindex { relation_classifier, .. } => relation_classifier.as_ref(),
            Backend::None => None,
        }
    }
}
