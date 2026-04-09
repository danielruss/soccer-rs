use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError{

    #[error("Model Access Error: {0}")]
    APIError(#[from]hf_hub::api::sync::ApiError),
    #[error("Cacheing Error: {0}")]
    CacheError(String),
    #[error("Classification System Error: {0}")]
    ClassificationSystem(String),
    #[error("Crosswalk Error: {0}")]
    Crosswalk(String),
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Building Error: {0}")]
    BuilderError(String),
    #[error("Preprocessing Error: {0}")]
    PreprocessingError(String),
    #[error("Embedding Error: {0}")]
    EmbeddingError(String),
    #[error("SOCcer Error: {0}")]
    SoccerError(String),
}
