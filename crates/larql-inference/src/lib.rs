extern crate blas_src;

pub mod attention;
pub mod backend;
pub mod capture;
pub mod error;
pub mod ffn;
pub mod forward;
pub mod graph_ffn;
pub mod layer_graph;
pub mod model;
pub mod route_ffn;
pub mod residual;
pub mod tokenizer;
pub mod trace;
pub mod vindex;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports.
pub use backend::{MatMulBackend, MatMulOp, default_backend, cpu_backend, dot_proj_gpu, matmul_gpu};
pub use backend::cpu::CpuBackend;
#[cfg(feature = "metal")]
pub use backend::metal::MetalBackend;

// Re-export essentials at crate root.
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use error::InferenceError;
pub use ffn::{FfnBackend, HighwayFfn, LayerFfnRouter, SparseFfn, WeightFfn};
pub use attention::AttentionWeights;
pub use forward::{
    calibrate_scalar_gains, capture_residuals, forward_to_layer, logit_lens_top1, predict,
    predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
    predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router,
    predict_with_strategy, trace_forward, trace_forward_full, trace_forward_with_ffn,
    LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention,
    PredictResultWithResiduals, TraceResult,
};
pub use graph_ffn::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
#[allow(deprecated)]
pub use ffn::experimental::cached::CachedFfn;
#[allow(deprecated)]
pub use ffn::experimental::clustered::{ClusteredFfn, ClusteredGateIndex};
#[allow(deprecated)]
pub use ffn::experimental::down_clustered::{DownClusteredFfn, DownClusteredIndex};
#[allow(deprecated)]
pub use ffn::experimental::entity_routed::EntityRoutedFfn;
#[allow(deprecated)]
pub use ffn::experimental::feature_list::FeatureListFfn;
#[allow(deprecated)]
pub use ffn::experimental::graph::GraphFfn;
pub use route_ffn::{RouteFfn, RouteGuidedFfn, RouteTable};
pub use trace::{
    trace_residuals, trace as trace_decomposed, AnswerWaypoint, LayerSummary,
    ResidualTrace, TraceNode, TracePositions, TraceStore, TraceWriter,
    BoundaryStore, BoundaryWriter,
    ContextStore, ContextWriter, ContextTier,
};
pub use layer_graph::{
    // Production
    LayerGraph, LayerOutput, DenseLayerGraph, WalkLayerGraph,
    CachedLayerGraph, PerLayerGraph,
    predict_with_graph, predict_with_graph_vindex_logits,
    trace_with_graph, build_adaptive_graph,
    // Analysis/validation
    TemplatePattern, TemplateUniverse, GuidedWalkLayerGraph,
    detect_template,
};
pub use vindex::WalkFfn;
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, load_tokenizer};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
