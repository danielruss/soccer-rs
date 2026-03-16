use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError{
    #[error("Cacheing Error: ${0}")]
    CacheError(String),
    #[error("IO Error: ${0}")]
    IOError(#[from] std::io::Error),
}
