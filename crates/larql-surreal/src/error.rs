use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SurrealError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unknown component: {0}")]
    UnknownComponent(String),
    #[error("no vector files found in {0}")]
    NoVectorFiles(PathBuf),
    #[error("surreal error: {0}")]
    Surreal(String),
}
