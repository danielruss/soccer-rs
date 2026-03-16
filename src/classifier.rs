#![allow(dead_code)]

use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde::Deserialize;


#[derive(Deserialize,Debug,PartialEq, Eq,Hash)]
struct EmbeddingConfig{
    pooling:String,
    normalize:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash)]
struct PipeLineConfig{
    dtype:String,
    quantized:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash)]
pub struct ModelConfig{
    embedding_model:String,
    model_url:String,
    embedding_config:EmbeddingConfig,
    pipeline_config:PipeLineConfig,

}
#[derive(Deserialize,Debug,PartialEq, Eq)]
pub struct VersionedModel{
    #[serde(alias = "default")]
    default_version:String,

    versions: HashMap<String, ModelConfig>,
}

#[derive(Debug, Eq, PartialEq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    SOCcerNET,
    CLIPS,
}

pub struct Classifier{
    embedding_model:String,
    model_url:String,
}

#[derive(Debug,Deserialize)]
pub struct StartupConfig(HashMap<ModelType, VersionedModel>);
impl StartupConfig{
    pub fn get_config(&self, name: &ModelType, version: &str) -> Option<&ModelConfig>{
        self.0
            .get(name)
            .and_then(|models|models.versions.get(version) )
    }
    pub fn get_default_version(&self, name:&ModelType) ->  Option<&ModelConfig>{
        let versioned_model = self.0.get(name)?;
        let default_version = versioned_model.default_version.as_str();
        versioned_model.versions.get(default_version)
    }
}

// This is the singleton... Only run at startup...
pub static MODEL_CONFIG: Lazy<StartupConfig> = Lazy::new(||{
    let json_content = include_str!("../data/classifier.json");

    serde_json::from_str(json_content)
        .expect("Crate Internal Configuration Error: Failed to parse classifier configuration file.")
});




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load(){
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.0");
        assert!(config.is_some());
        let config=MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.1");
        assert!(config.is_none());
        let config = MODEL_CONFIG.get_config(&ModelType::CLIPS, "1.0.0");
        assert!(config.is_some());

        let config = MODEL_CONFIG.get_default_version(&ModelType::SOCcerNET);
        assert!(config.is_some());
        println!("{:?}",config.unwrap())
    }

    #[test]
    fn test_soccer(){
        
    }
}