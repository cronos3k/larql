//! POST /v1/infer — full forward pass with attention.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    #[serde(default = "default_top")]
    pub top: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_top() -> usize { 5 }
fn default_mode() -> String { "walk".into() }

fn run_infer(
    model: &LoadedModel,
    req: &InferRequest,
) -> Result<serde_json::Value, ServerError> {
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer)".into(),
        ));
    }

    if !model.config.has_model_weights
        && model.config.extract_level != larql_vindex::ExtractLevel::Inference
        && model.config.extract_level != larql_vindex::ExtractLevel::All
    {
        return Err(ServerError::InferenceUnavailable(
            "vindex does not contain model weights. Rebuild with --include-weights".into(),
        ));
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();

    let is_compare = req.mode == "compare";
    let use_walk = req.mode == "walk" || is_compare;
    let use_dense = req.mode == "dense" || is_compare;

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));

    if use_walk {
        let patched = model.patched.blocking_read();
        let walk_ffn = larql_inference::WalkFfn::new(weights, &*patched, 8092);
        let walk_start = std::time::Instant::now();
        let pred = larql_inference::predict_with_ffn(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
            &walk_ffn,
        );
        let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

        let predictions: Vec<serde_json::Value> = pred
            .predictions
            .iter()
            .map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            })
            .collect();

        if is_compare {
            result.insert("walk".into(), serde_json::json!(predictions));
            result.insert("walk_ms".into(), serde_json::json!((walk_ms * 10.0).round() / 10.0));
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!("walk"));
        }
    }

    if use_dense {
        let dense_start = std::time::Instant::now();
        let pred = larql_inference::predict(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
        );
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

        let predictions: Vec<serde_json::Value> = pred
            .predictions
            .iter()
            .map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            })
            .collect();

        if is_compare {
            result.insert("dense".into(), serde_json::json!(predictions));
            result.insert("dense_ms".into(), serde_json::json!((dense_ms * 10.0).round() / 10.0));
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!("dense"));
        }
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    result.insert("latency_ms".into(), serde_json::json!((latency_ms * 10.0).round() / 10.0));

    Ok(serde_json::Value::Object(result))
}

pub async fn handle_infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || run_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_infer_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || run_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
