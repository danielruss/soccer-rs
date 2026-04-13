#![allow(dead_code)]

use std::{collections::HashMap, str::FromStr, sync::Arc};

use once_cell::sync::Lazy;
use ort::{session::Session, value::Value};
use serde::Deserialize;
use hf_hub::api::sync::Api;
use tokenizers::{Encoding, PaddingParams, Tokenizer};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::{cache::Cache, crosswalk::{CLASSIFICATION_SYSTEM_REGISTRY, ClassificationSystem, KnownClassificationSystem}, error::MyError, preprocessing::clean_free_text};


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


#[derive(Deserialize,Debug,PartialEq, Eq,Hash,Clone)]
struct EmbeddingConfig{
    pooling:PoolingStrategy,
    normalize:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash,Clone)]
struct PipeLineConfig{
    dtype:String,
    quantized:bool,
}

#[derive(Deserialize,Debug,PartialEq,Eq,Hash,Clone)]
pub struct ModelConfig{
    pub(crate) model_type:ModelType,
    pub(crate) output_classification_system:KnownClassificationSystem,
    embedding_model:String,
    model_url:String,
    embedding_config:EmbeddingConfig,
    pipeline_config:PipeLineConfig,
}
impl ModelConfig{
    pub fn output_system(&self) -> Arc<ClassificationSystem>{
        CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(self.output_classification_system)
    }
    pub fn output_dim(&self) -> usize{
        self.output_system().len()
    }
}

pub trait SoccerBuilder {
    fn build(config:&ModelConfig) -> Result<SoccerPipeline,MyError>;
}

#[derive(Debug)]
pub struct Scored<T>(pub T, pub f32);

#[derive(Debug)]
pub struct Embedder{
    config: EmbeddingConfig,
    tokenizer: Tokenizer,
    embedding_session: Session,
}
impl Embedder {
    fn new(config:&ModelConfig) ->Result<Self,MyError>{
        println!("in Embedder::new() => {:?}\nCreating API and getting model...",config.embedding_config);
        let embedding_config = config.embedding_config.clone();

        println!("in Embedder::new() => {:?}\nCreating API and getting model...",config.embedding_model);
        let client = Api::new().unwrap();
        let repo = client.model(config.embedding_model.clone());

        // WARNING: This may change...
        let model_path = repo.get("onnx/model.onnx")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
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

        
        Ok( Self { tokenizer, embedding_session, config: embedding_config } )
    }

    fn embed_text<T:AsRef<str>>(&mut self,text:&[T]) -> Result<Array2<f32>,MyError>{
        let text_vec = text.iter().map(|t| t.as_ref()).collect();
        let encodings = self.tokenizer.encode_batch(text_vec, true)
            .map_err(|e| MyError::EmbeddingError(e.to_string()))?;

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

        let array:Array3<f32> = Array3::from_shape_vec((shape[0] as usize,shape[1] as usize,shape[2] as usize),slice.to_vec())
            .map_err(|e| MyError::EmbeddingError(e.to_string()))?;

        let mut pooled_array:Array2<f32> = Embedder::apply_pooling(self.config.pooling,array.view());
        if self.config.normalize {
            pooled_array.axis_iter_mut(Axis(0))
                .for_each( |mut row|{
                    let norm = row.mapv(|x| x.powi(2)).sum().sqrt();
                    if norm > 1e-12 {
                        row.map_inplace(|x| *x /= norm)
                    }
                })
        }
        
        Ok(pooled_array)
    }
    fn embed_job_descriptions(&mut self,job_descriptions:Vec<PreprocessedJobDescription>) -> Result<EmbeddedJobDescriptions,MyError>{
        let (ids,text):(Vec<String>,Vec<String>) = job_descriptions.into_iter()
            .map(|job| {
                (job.id,job.cleaned_text)
            }).unzip();
        let embeddings = self.embed_text(&text)?;

        Ok( EmbeddedJobDescriptions {ids,embeddings } )
    }

    pub fn apply_pooling(strategy:PoolingStrategy,view:ArrayView3<f32>) -> Array2<f32>{
        // note Axis 0 == seq in batch (seq 0, 1, 2, 3,...)
        //      Axis 1 == token in seq (token 0, 1, 2, 3,...)
        //      Axis 2 == embedding entry (vec element 0, 1, 2, ...)
        match strategy {
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
}

#[derive(Debug)]
pub struct JobDescription{
    pub id:String,
    pub text1:String, // For CLIPS -> PS, SOCcerNET -> JobTitle
    pub text2:Option<String>, //For CLIPS -> None, SOCcerNET -> JobTasks
    pub multihot_prior:Box<[u16]>
}
impl From<(String,String)> for JobDescription{
    fn from(value: (String,String)) -> Self {
        Self { id: value.0, text1: value.1, text2:  None, multihot_prior:Box::default() }
    }
}
impl From<(String,String,String)> for JobDescription{
    fn from(value: (String,String,String)) -> Self {
        Self { id: value.0, text1: value.1, text2:  Some(value.2), multihot_prior:Box::default()  }
    }
}
impl From<(&str,&str)> for JobDescription{
    fn from(value: (&str,&str)) -> Self {
        Self { id: value.0.to_string(), text1: value.1.to_string(), text2:  None, multihot_prior:Box::default()  }
    }
}
impl From<(&str,&str,&str)> for JobDescription{
    fn from(value: (&str,&str,&str)) -> Self {
        Self { id: value.0.to_string(), text1: value.1.to_string(), text2:  Some(value.2.to_string()), multihot_prior:Box::default()  }
    }
}

#[derive(Debug)]
pub struct PreprocessedJobDescription{
    pub id:String,
    pub cleaned_text:String
}

#[derive(Debug)]
pub struct EmbeddedJobDescriptions{
    pub ids:Vec<String>,
    pub embeddings:Array2<f32>,
}
impl EmbeddedJobDescriptions{
    pub fn len(&self)->usize{
        self.ids.len()
    }
}

#[derive(Debug)]
pub struct CodedJobDescription{
    pub id:String,
    pub scored_code_index:Vec<Scored<usize>>
}

#[derive(Debug)]
pub struct SoccerPipeline {
    pub embedder:Embedder,
    pub soccer_session: Session,

    pub config: ModelConfig,
}

impl SoccerPipeline {
    pub fn run(&mut self,job_descriptions:&[&JobDescription]) -> Result<Vec<CodedJobDescription>,MyError>{

        // create the crosswalked input
        let multihot_array = self.create_multihot_array2d(job_descriptions);

        // preprocess the job descriptions
        let preprocessed_job_descriptions = self.config.model_type.preprocess_batch(job_descriptions);

        // Embed the text...
        let embedded_jobs = self.embedder.embed_job_descriptions(preprocessed_job_descriptions)?;

        // run soccer....
        self.run_soccer(embedded_jobs, multihot_array)
    }

    fn create_multihot_array2d(&self,job_descriptions:&[&JobDescription]) -> Array2<f32> {
        let mut multihot:Array2<f32> = Array2::zeros( (job_descriptions.len(),self.config.output_dim()) );
        job_descriptions.iter().enumerate()
            .for_each(|(row_indx,job)|{
                job.multihot_prior.iter().for_each(|&col_indx| multihot[[row_indx, col_indx as usize]]=1.0)
            });
        multihot
    }

    fn run_soccer(&mut self,embedded_jobs: EmbeddedJobDescriptions,xw_input:Array2<f32> ) -> Result<Vec<CodedJobDescription>,MyError> {
        let embedding_values = Value::from_array(embedded_jobs.embeddings)
            .map(|v| v.into_dyn())
            .map_err(|e| MyError::SoccerError(e.to_string()))?;
        let xw_values = Value::from_array(xw_input)
                .map(|v|v.into_dyn())
                .map_err(|e| MyError::SoccerError(e.to_string()))?;
        let input = ort::inputs![
                "embedded_input" => embedding_values,
                "crosswalked_inp" => xw_values,
        ];        
        let soccer_results = self.soccer_session.run(input)
            .map_err(|e| MyError::SoccerError(e.to_string()))?;

        let (shape, slice) = soccer_results["soc2010_out"]
            .try_extract_tensor::<f32>()
            .map_err(|e| MyError::SoccerError(e.to_string()))?;
        let view = ArrayView2::from_shape((shape[0] as usize, shape[1] as usize), slice)
            .map_err(|e| MyError::SoccerError(e.to_string()))?;

        let sorted: Vec<CodedJobDescription> = view.axis_iter(Axis(0))
            .zip(embedded_jobs.ids)
            .map(|(row, id)| CodedJobDescription {
                id,
                scored_code_index: Self::argsort(row)
            })
            .collect();

        Ok(sorted)
    }

    pub fn argsort(row:ArrayView1<f32>) -> Vec<Scored<usize>> {

        let mut zipped:Vec<Scored<usize>> = row.iter()
            .enumerate()
            .map(|(i,&f)| Scored(i,f))
            .collect();

        zipped.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
        });

        zipped
    }
}

impl SoccerBuilder for SoccerPipeline {
    fn build(config:&ModelConfig) -> Result<SoccerPipeline,MyError> {
        let embedder = Embedder::new(config)?;

        println!("in SoccerRuntime::build() => {:?}",config.model_url);
        let soccer_model_path = Cache::get_onnx_from(&config.model_url)?;
        let soccer_session = Session::builder().map_err(|e| MyError::BuilderError(format!("SOCcer session E1: {}",e.to_string())))?
            .with_memory_pattern(true).map_err(|e| MyError::BuilderError(format!("SOCcer session E2: {}",e.to_string())))?
            .commit_from_file(soccer_model_path).map_err(|e| MyError::BuilderError(format!("SOCcer session E3: {}",e.to_string())))?;
    
        println!("in SoccerRuntime::build() => {:?}",config.embedding_config.pooling);

        Ok( SoccerPipeline { embedder, soccer_session, config:config.clone()} )
    }
}



#[derive(Deserialize,Debug,PartialEq, Eq)]
pub struct VersionedModel{
    #[serde(alias = "default")]
    default_version:String,

    versions: HashMap<String, ModelConfig>,
}

pub trait PreprocessStrategy{
    fn preprocess(&self,job_description:&JobDescription)->PreprocessedJobDescription;
    fn preprocess_batch(&self,job_descriptions:&[&JobDescription]) -> Vec<PreprocessedJobDescription>;
}


#[derive(Debug, Eq, PartialEq, Hash, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    SOCcerNET,
    CLIPS,
}
impl FromStr for ModelType{
    type Err=MyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "soccernet" => Ok(ModelType::SOCcerNET),
            "clips" => Ok(ModelType::CLIPS),
            sys => Err(MyError::BuilderError(format!("Unknown Model Type {}",sys)))
        }
    }
}
impl PreprocessStrategy for ModelType {
    fn preprocess(&self,job_description:&JobDescription)->PreprocessedJobDescription {
        let cloned_id = job_description.id.clone();

        // for SOCcerNET text1 is the JobTitle text2 is the JobTasks
        // for CLIPS text1 is the PS, text2 is None...
        let cleaned_text1 = clean_free_text(&job_description.text1);
        let cleaned_text2 = job_description.text2.as_ref().map(clean_free_text)
            .filter(|s| !s.is_empty()); // If it's "", it becomes None

        let cleaned_text = match self {
            ModelType::SOCcerNET => {
                cleaned_text2.map(|ct2| format!("{} {}",cleaned_text1,ct2))
                    .unwrap_or(cleaned_text1)
            },
            ModelType::CLIPS => cleaned_text1
        };

        PreprocessedJobDescription { id: cloned_id, cleaned_text }
    }

    fn preprocess_batch(&self,job_descriptions:&[&JobDescription]) -> Vec<PreprocessedJobDescription>{
        job_descriptions.iter()
            .map(|job| self.preprocess(job) )
            .collect()
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

pub fn display_soccer_results(results:Vec<CodedJobDescription>,n:usize,results_coding_system:Arc<ClassificationSystem>){
    results.iter().for_each( |job| {
        print!("{}",job.id);
        job.scored_code_index.iter().take(n).for_each(|scored_index| {
            let (cde,title) = results_coding_system.get_code_title(scored_index.0 as u32).unwrap();
            print!("{}  {:10.10}  {:0.4}\t",cde,title,scored_index.1);
        });
        println!("");
    });
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::crosswalk::{CLASSIFICATION_SYSTEM_REGISTRY, KnownClassificationSystem};


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
        let mut runtime = SoccerPipeline::build(config).unwrap();
        let expected_values:Vec<Vec<f32>> = vec![
            vec![0.9931, 0.1799, 0.0282, 0.0248, 0.0222], // plumber (from JS version)
            vec![0.9900, 0.6386, 0.6347, 0.5575, 0.3101], // doctor
        ];
        let expected_codes:Vec<Vec<&str>> = vec![
            vec!["47-2152","47-3015","51-9199","47-2151","47-2061"],
            vec!["29-1069","29-1067","29-1063","29-1199","29-1011"]
        ];

        let job_descriptions: Vec<JobDescription> = vec![
            ("test01-1","plumber").into(),
            ("test01-2","doctor").into()
        ];
        let refs:Vec<&JobDescription> = job_descriptions
            .iter()
            .collect();

        let res = runtime.run(&refs).unwrap();
        let soc2010= CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(KnownClassificationSystem::SOC2010);
        
        res.iter().enumerate().for_each(|(row,prediction)| {
            print!("{} {}:  ",row,prediction.id);
            prediction.scored_code_index.iter().take(7)
                .for_each(|scored_index| print!(" {} {:.4}   ",soc2010.get_code(scored_index.0 as u32).unwrap(),scored_index.1));
            println!();
        });
        
        res.iter()
            .zip(expected_values.iter())
            .zip(expected_codes.iter())
            .enumerate()
            .for_each(|(batch_idx, ((actual_row, exp_scores), exp_codes))| {
                actual_row.scored_code_index.iter()
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
        let output_system = config.output_system();
        assert_eq!(840,output_system.len());

        let mut runtime = SoccerPipeline::build(config).unwrap();
        let job_descriptions:[JobDescription;2] = [
            ("testjob-1","--ceo--","run company ").into(),
            ("testjob-2","doctor. ","    treat patients.  ").into()
        ];
        let job_descriptions:Vec<&JobDescription> = job_descriptions.iter().collect();

        let res: Vec<CodedJobDescription> = runtime.run(&job_descriptions).unwrap();
        println!();
        display_soccer_results(res,3,CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(KnownClassificationSystem::SOC2010));
        println!();
    }
}