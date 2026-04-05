/// LQL error types.

#[derive(Debug, thiserror::Error)]
pub enum LqlError {
    #[error("No backend loaded. Run USE \"path.vindex\" first.")]
    NoBackend,

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Mutation requires a vindex. Run EXTRACT first.")]
    MutationRequiresVindex,
}
