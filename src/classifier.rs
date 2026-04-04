#![allow(dead_code)]

use std::{collections::HashMap, sync::Arc};

use once_cell::sync::Lazy;
use ort::{session::Session, value::Value};
use serde::Deserialize;
use hf_hub::api::sync::Api;
use tokenizers::{Encoding, PaddingParams, Tokenizer};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::{cache::Cache, crosswalk::ClassificationSystem, error::MyError, preprocessing::clean_free_text};


// Private functions that takes a Vec<Encoding> (more generally as a slice) and returns parts
// of the input as a Value such.  the Fn selects the part of the encodings (e.g., |e| e.get_ids() )
fn stack_field(encodings:&[Encoding],f:impl Fn(&Encoding)->&[u32]) -> Result<Value,MyError>{
    if encodings.is_empty() {
        return Err(MyError::BuilderError("Empty batch".to_string()));
    }
    let batch_size = encodings.len();
    let seq_len = f(&encodings[0]).len();

    let flattened:Vec<i64> = encodings.iter().flat_map(|enc| f(enc).iter().map(|&i| i as i64)).collect();
    let array:Array2<i64> = Array2::from_shape_vec((batch_size,seq_len), flattened)
        .map_err(|e| MyError::BuilderError(e.to_string()))?;
    Value::from_array(array)
        .map(|v|v.into_dyn())
        .map_err(|e| MyError::BuilderError(e.to_string()))
}


#[derive(Debug, Deserialize,Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all="lowercase")]
pub enum PoolingStrategy {
    Cls,
    Mean,
}


#[derive(Deserialize,Debug,PartialEq, Eq,Hash)]
struct EmbeddingConfig{
    pooling:PoolingStrategy,
    normalize:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash)]
struct PipeLineConfig{
    dtype:String,
    quantized:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash)]
pub struct ModelConfig{
    model_type:ModelType,
    embedding_model:String,
    model_url:String,
    embedding_config:EmbeddingConfig,
    pipeline_config:PipeLineConfig,
}

pub trait SoccerBuilder {
    fn build(config:&ModelConfig) -> Result<SoccerRuntime,MyError>;
}

#[derive(Debug)]
pub struct Scored<T>(pub T, pub f32);

#[derive(Debug)]
pub struct SoccerRuntime {
    pub tokenizer: Tokenizer,
    pub embedding_session: Session,
    pub soccer_session: Session,

    pub model_type: ModelType,
    pub pooling:PoolingStrategy,
    pub normalize:bool,
}

impl SoccerRuntime {
    pub fn run<T:AsRef<str>>(&mut self,text:&[T]) -> Result<Vec<Vec<Scored<usize>>>,MyError>{
        // Run all the embedding in a local block
        // this FORCES rust to drop the borrowed mutable reference to self so I
        // can use borrow the reference as unmutable.

        let array:Array3<f32> = {
            let text_vec = text.iter().map(|t| t.as_ref()).collect();
            let encodings = self.tokenizer.encode_batch(text_vec, true).unwrap();
            let input_ids:Value = stack_field(&encodings, |e| e.get_ids())?;
            let attention_mask:Value = stack_field(&encodings, |e| e.get_attention_mask())?;
            let token_type_ids:Value = stack_field(&encodings, |e| e.get_type_ids())?;
    
            let inputs = ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ];
    
            let embeddings = self.embedding_session.run(inputs)
                .map_err(|e| MyError::EmbeddingError(e.to_string()))?;
            let (shape,slice) = embeddings["last_hidden_state"].try_extract_tensor::<f32>()
                .map_err(|e| MyError::EmbeddingError(e.to_string()))?;
            Array3::from_shape_vec((shape[0] as usize,shape[1] as usize,shape[2] as usize),slice.to_vec())
                .map_err(|e| MyError::EmbeddingError(e.to_string()))?
        };
        let mut embeddings:Array2<f32> = self.apply_pooling(array.view());
        if self.normalize {
            embeddings.axis_iter_mut(Axis(0))
                .for_each( |mut row|{
                    let norm = row.mapv(|x| x.powi(2)).sum().sqrt();
                    if norm > 1e-12 {
                        row.map_inplace(|x| *x /= norm)
                    }
                })
        }
        dbg!(&embeddings);

        // run soccer....
        let results = {
            let multihot:Array2<f32> = Array2::zeros( (text.len(),840usize) );
            let embedding_values = Value::from_array(embeddings)
                .map(|v|v.into_dyn())
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
            let multihot_values = Value::from_array(multihot)
                .map(|v|v.into_dyn())
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
            let input = ort::inputs![
                "embedded_input" => embedding_values,
                "crosswalked_inp" => multihot_values,
            ];
            let soccer_results = self.soccer_session.run(input)
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
            let (shape,slice) = soccer_results["soc2010_out"]
                .try_extract_tensor::<f32>()
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
            println!("DEBUG: Raw Output Shape: {:?}", shape);
            println!("DEBUG: First 10 values: {:?}", &slice[..10]);
            let view = ArrayView2::from_shape( (shape[0] as usize,shape[1] as usize), slice)
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
            let sorted:Vec<Vec<Scored<usize>>> = view.axis_iter(Axis(0))
                .map(|row| SoccerRuntime::argsort(row))
                .collect();
            
            sorted.iter()
                .enumerate()
                .for_each( |(line,row)| {
                    println!("{line}: {:?} {:?} {:?} {:?} {:?}", row[0],row[1],row[2],row[3],row[4]);
                });

            sorted
        };

        Ok(results)
    }

    pub fn apply_pooling(&self,view:ArrayView3<f32>) -> Array2<f32>{
        // note Axis 0 == seq in batch (seq 0, 1, 2, 3,...)
        //      Axis 1 == token in seq (token 0, 1, 2, 3,...)
        //      Axis 2 == embedding entry (vec element 0, 1, 2, ...)
        match self.pooling {
            PoolingStrategy::Cls => {
                // get the 0th token (i.e., the cls token)
                view.index_axis(ndarray::Axis(1), 0).to_owned()
            },
            PoolingStrategy::Mean => {
                // get the mead across the sequences (for each element)
                // this is infallable so no worries about unwrapping.
                view.mean_axis(ndarray::Axis(1)).unwrap()
            },
        }
    }

    pub fn argsort(row:ArrayView1<f32>) -> Vec<Scored<usize>> {

        let mut zipped:Vec<Scored<usize>> = row.iter()
            .enumerate()
            .map(|(i,&f)| Scored(i,f))
            .collect();

        zipped.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        zipped
    }
}

impl SoccerBuilder for SoccerRuntime {
    fn build(config:&ModelConfig) -> Result<SoccerRuntime,MyError> {
        println!("in SoccerRuntime::build() => {:?}",config.embedding_model);
        println!("in SoccerRuntime::build() => {:?}",config.model_url);
        println!("in SoccerRuntime::build() => {:?}",config.embedding_config.pooling);
        let client = Api::new().unwrap();
        let repo = client.model(config.embedding_model.clone());

        // WARNING: This may change...
        let model_path = repo.get("onnx/model.onnx").unwrap();
        let tokenizer_path = repo.get("tokenizer.json").unwrap();
        println!("{:?}\n{:?}",model_path,tokenizer_path);

        let mut tokenizer:Tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MyError::CacheError(format!("Problem getting the Tokenizer cached at {:?}.\n\t{}",&tokenizer_path.to_str(),e.to_string())))?;

        tokenizer.with_padding(Some( PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

        let embedding_session = Session::builder().map_err(|e| MyError::BuilderError(format!("Embedding session E1: {}",e.to_string())))?
            .with_memory_pattern(true).map_err(|e| MyError::BuilderError(format!("Embedding session E2: {}",e.to_string())))?
            .commit_from_file(model_path).map_err(|e| MyError::BuilderError(format!("Emedding session E3: {}",e.to_string())))?;

        let soccer_model_path = Cache::get_onnx_from(&config.model_url)?;
        let soccer_session = Session::builder().map_err(|e| MyError::BuilderError(format!("SOCcer session E1: {}",e.to_string())))?
            .with_memory_pattern(true).map_err(|e| MyError::BuilderError(format!("SOCcer session E2: {}",e.to_string())))?
            .commit_from_file(soccer_model_path).map_err(|e| MyError::BuilderError(format!("SOCcer session E3: {}",e.to_string())))?;

        let model_type = config.model_type;
        let pooling = config.embedding_config.pooling;
        let normalize = config.embedding_config.normalize;
        Ok( SoccerRuntime { tokenizer, embedding_session, soccer_session, model_type, pooling, normalize } )
    }
}



#[derive(Deserialize,Debug,PartialEq, Eq)]
pub struct VersionedModel{
    #[serde(alias = "default")]
    default_version:String,

    versions: HashMap<String, ModelConfig>,
}

pub trait PreprocessStrategy{
    fn preprocess<T:AsRef<str>>(&self,text1:T,text2:Option<T>)->String;
    fn preprocess_batch<T:AsRef<str>>(&self,text1:&[T],text2:Option<&[T]>) -> Result<Vec<String>,MyError>;
}


#[derive(Debug, Eq, PartialEq, Hash, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    SOCcerNET,
    CLIPS,
}

impl PreprocessStrategy for ModelType {
    fn preprocess<T:AsRef<str>>(&self,text1:T,text2:Option<T>)->String {
        // for SOCcerNET text1 is the JobTitle text2 is the JobTasks
        // for CLIPS text1 is the PS, text2 is None...
        let cleaned_text1 = clean_free_text(text1);
        let cleaned_text2 = text2.map(clean_free_text)
            .filter(|s| !s.is_empty()); // If it's "", it becomes None

        match self {
            ModelType::SOCcerNET => {
                cleaned_text2.map(|ct2| format!("{} {}",cleaned_text1,ct2))
                    .unwrap_or(cleaned_text1)
            },
            ModelType::CLIPS => cleaned_text1
        }
    }

    fn preprocess_batch<T:AsRef<str>>(&self,text1:&[T],text2:Option<&[T]>) -> Result<Vec<String>,MyError> {
        if let Some(t2_array) = text2 {
            if t2_array.len() != text1.len() {
                return Err(MyError::PreprocessingError("The two text inputs do not have the same size".to_string()));
            }
        }

        Ok( text1.iter().enumerate().map(|(i, s1)| {
            // if text2 is None, s2 will be None... otherwise text2[i]
            let s2 = text2.map(|array| &array[i]);
            self.preprocess(s1, s2)
        }).collect() )
    }
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

pub fn display_soccer_results(results:Vec<Vec<Scored<usize>>>,n:usize,results_coding_system:Arc<ClassificationSystem>){
    results.iter().for_each( |job| {
        job.iter().take(n).for_each(|scored_index| {
            let (cde,title) = results_coding_system.get_code_title(scored_index.0 as u32).unwrap();
            print!("{}  {:10.10}  {:0.4}\t",cde,title,scored_index.1);
        });
        println!("");
    });
}


#[cfg(test)]
mod tests {
    use crate::crosswalk::{ClassificationSystem, ClassificationSystemRegistry, KnownClassificationSystem};

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
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.0").unwrap();
        let mut runtime = SoccerRuntime::build(config).unwrap();
        let expected_values:Vec<Vec<f32>> = vec![
            vec![0.9931, 0.1799, 0.0282, 0.0248, 0.0222], // plumber (from JS version)
            vec![0.9900, 0.6386, 0.6347, 0.5575, 0.3101], // doctor
        ];
        let expected_codes:Vec<Vec<&str>> = vec![
            vec!["47-2152","47-3015","51-9199","47-2151","47-2061"],
            vec!["29-1069","29-1067","29-1063","29-1199","29-1011"]
        ];

        let text = &["plumber","doctor"];
        let res = runtime.run(text).unwrap();
        let soc2010 = ClassificationSystem::try_from(KnownClassificationSystem::SOC2010).unwrap();
        
        res.iter().enumerate().for_each(|(row,prediction)| {
            print!("{}:  ",row);
            prediction.iter().take(7)
                .for_each(|scored_index| print!(" {} {:.4}   ",soc2010.get_code(scored_index.0 as u32).unwrap(),scored_index.1));
            println!();
        });
        
        res.iter()
            .zip(expected_values.iter())
            .zip(expected_codes.iter())
            .enumerate()
            .for_each(|(batch_idx, ((actual_row, exp_scores), exp_codes))| {
                actual_row.iter()
                    .take(5)
                    .zip(exp_scores.iter())
                    .zip(exp_codes.iter())
                    .for_each(|((scored_index, &exp_score), &exp_code)| {
                        let acutal_index = scored_index.0;
                        let actual_score=scored_index.1;
                        // 1. Check the Score (Tolerance 0.0001)
                        let diff = (actual_score - exp_score).abs();
                        assert!(
                            diff < 0.0001,
                            "Batch {batch_idx} score mismatch! Actual: {actual_score}, Expected: {exp_score} (diff: {diff})"
                        );

                        // 2. Check the SOC Code
                        // Assuming you have a way to map 'id' back to 'code' (e.g., a lookup table)
                        let actual_code = soc2010.get_code(acutal_index as u32).unwrap(); 
                        assert_eq!(
                            actual_code, exp_code,
                            "Batch {batch_idx} code mismatch! Actual: {actual_code}, Expected: {exp_code}"
                        );
                    });
            });
    }

     #[test]
    fn test_soccer_2(){
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.0").unwrap();
        let mut runtime = SoccerRuntime::build(config).unwrap();
        //let x= runtime.model_type.preprocess(" ceo ", Some("  "));
        let x=runtime.model_type.preprocess(" ceo ",None);
        dbg!(&x);

        let titles = ["--ceo--","  doctor  "];
        let tasks = ["run company","         treat  patients  "];
        let x = runtime.model_type.preprocess_batch(&titles, Some(&tasks));
        let x = x.unwrap();
        let res = runtime.run(&x).unwrap();
        println!();
        display_soccer_results(res,3,ClassificationSystemRegistry::get_classification_system(KnownClassificationSystem::SOC2010));
        println!();
    }
}