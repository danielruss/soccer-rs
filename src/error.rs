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
    #[error("CSV Syntax error at line {line}: {source}")]
    CSVSyntaxError { source: csv::Error, line: u64 },
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
    #[error("Output Error: {0}")]
    OutputError(String),
}

impl From<csv::Error> for MyError {
    fn from(e: csv::Error) -> Self {
        let line = e.position().map(|p| p.line()).unwrap_or(0);
        MyError::CSVSyntaxError { line, source: e }
    }
}