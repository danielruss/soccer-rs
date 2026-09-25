use thiserror::Error;

#[derive(Error, Debug)]
/// Errors produced while loading data, constructing models, or running inference.
pub enum SoccerError {
    /// A model or data file could not be cached or located.
    #[error("Caching Error: {0}")]
    CacheError(#[from] soccer_cache::CacheError),
    /// A classification-system name or definition was invalid.
    #[error("Classification System Error: {0}")]
    ClassificationSystem(String),
    /// A requested crosswalk was unavailable or invalid.
    #[error("Crosswalk Error: {0}")]
    Crosswalk(String),
    /// A CSV record could not be parsed.
    #[error("CSV Syntax error at line {line}: {source}")]
    CSVSyntaxError {
        /// Underlying CSV parser error.
        source: csv::Error,
        /// One-based CSV line number, or zero when unavailable.
        line: u64,
    },
    /// A filesystem or stream operation failed.
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    /// A model pipeline or input could not be constructed.
    #[error("Building Error: {0}")]
    BuilderError(String),
    /// Input text preprocessing failed.
    #[error("Preprocessing Error: {0}")]
    PreprocessingError(String),
    /// Tokenization or embedding failed.
    #[error("Embedding Error: {0}")]
    EmbeddingError(String),
    /// ONNX model inference or output extraction failed.
    #[error("Inference Error: {0}")]
    InferenceError(String),
    /// Model output could not be converted into the requested result.
    #[error("Output Error: {0}")]
    OutputError(String),
}

impl From<csv::Error> for SoccerError {
    fn from(e: csv::Error) -> Self {
        let line = e.position().map(|p| p.line()).unwrap_or(0);
        SoccerError::CSVSyntaxError { line, source: e }
    }
}
