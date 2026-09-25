#![allow(dead_code)]

use std::{borrow::Cow, collections::HashMap, str::FromStr, sync::Arc};

use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use once_cell::sync::Lazy;
use ort::{session::Session, value::Value};
use serde::Deserialize;
use soccer_cache::Cache;
use tokenizers::{Encoding, PaddingParams, Tokenizer};

use crate::{
    crosswalk::{CLASSIFICATION_SYSTEM_REGISTRY, ClassificationSystem, KnownClassificationSystem},
    error::SoccerError,
    preprocessing::clean_free_text,
};

// Private functions that takes a Vec<Encoding> (more generally as a slice) and returns parts
// of the input as a Value such.  the Fn selects the part of the encodings (e.g., |e| e.get_ids() )
fn stack_field(
    encodings: &[Encoding],
    f: impl Fn(&Encoding) -> &[u32],
) -> Result<Value, SoccerError> {
    if encodings.is_empty() {
        return Err(SoccerError::BuilderError("Empty batch".to_string()));
    }
    let batch_size = encodings.len();
    let seq_len = f(&encodings[0]).len();

    let flattened: Vec<i64> = encodings
        .iter()
        .flat_map(|enc| f(enc).iter().map(|&i| i as i64))
        .collect();
    let array: Array2<i64> = Array2::from_shape_vec((batch_size, seq_len), flattened)
        .map_err(|e| SoccerError::BuilderError(e.to_string()))?;
    Value::from_array(array)
        .map(|v| v.into_dyn())
        .map_err(|e| SoccerError::BuilderError(e.to_string()))
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
/// Pooling operation used to reduce token embeddings to one vector per input.
pub enum PoolingStrategy {
    /// Uses the embedding of the first (classification) token.
    Cls,
    /// Uses the mean embedding across all tokens.
    Mean,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash, Clone)]
struct EmbeddingConfig {
    pooling: PoolingStrategy,
    normalize: bool,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash, Clone)]
struct PipeLineConfig {
    dtype: String,
    quantized: bool,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash, Clone)]
/// Configuration for an embedding and classification model pipeline.
pub struct ModelConfig {
    pub(crate) model_type: ModelType,
    pub(crate) output_classification_system: KnownClassificationSystem,
    embedding_model: String,
    model_url: String,
    model_output_name: String,
    embedding_config: EmbeddingConfig,
    pipeline_config: PipeLineConfig,
}
impl ModelConfig {
    /// Returns the classification system produced by this model.
    pub fn output_system(&self) -> Arc<ClassificationSystem> {
        CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(self.output_classification_system)
    }
    /// Returns the number of classifications produced by this model.
    pub fn output_dim(&self) -> usize {
        self.output_system().len()
    }
}

/// Constructs a SOCcer model pipeline from configuration.
pub trait SoccerBuilder {
    /// Downloads or locates required models and creates the pipeline.
    fn build(config: &ModelConfig) -> Result<SoccerPipeline, SoccerError>;
}

#[derive(Debug)]
/// A value paired with its model score.
pub struct Scored<T>(pub T, pub f32);

#[derive(Debug)]
/// Tokenizer and ONNX session used to create text embeddings.
pub struct Embedder {
    config: EmbeddingConfig,
    tokenizer: Tokenizer,
    embedding_session: Session,
}
impl Embedder {
    fn new(config: &ModelConfig, cache: &Cache) -> Result<Self, SoccerError> {
        let model_name = config.embedding_model.as_str();
        let embedding_config = config.embedding_config.clone();

        let model_path = cache.get_embedding_model_path(model_name, None)?;
        let tokenizer_path = cache.get_tokenizer_path(model_name, None)?;

        let mut tokenizer: Tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            SoccerError::EmbeddingError(format!(
                "Problem getting the Tokenizer cached at {:?}.\n\t{}",
                &tokenizer_path.to_str(),
                e
            ))
        })?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let embedding_session = Session::builder()
            .map_err(|e| SoccerError::BuilderError(format!("Embedding session E1: {}", e)))?
            .with_memory_pattern(true)
            .map_err(|e| SoccerError::BuilderError(format!("Embedding session E2: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| SoccerError::BuilderError(format!("Emedding session E3: {}", e)))?;

        Ok(Self {
            tokenizer,
            embedding_session,
            config: embedding_config,
        })
    }

    fn embed_text<T: AsRef<str>>(&mut self, text: &[T]) -> Result<Array2<f32>, SoccerError> {
        let text_vec = text.iter().map(|t| t.as_ref()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(text_vec, true)
            .map_err(|e| SoccerError::EmbeddingError(e.to_string()))?;

        let input_ids: Value = stack_field(&encodings, |e| e.get_ids())?;
        let attention_mask: Value = stack_field(&encodings, |e| e.get_attention_mask())?;
        let token_type_ids: Value = stack_field(&encodings, |e| e.get_type_ids())?;

        let inputs = ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        ];

        let embeddings = self
            .embedding_session
            .run(inputs)
            .map_err(|e| SoccerError::EmbeddingError(e.to_string()))?;
        let (shape, slice) = embeddings["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| SoccerError::EmbeddingError(e.to_string()))?;

        let array: Array3<f32> = Array3::from_shape_vec(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            slice.to_vec(),
        )
        .map_err(|e| SoccerError::EmbeddingError(e.to_string()))?;

        let mut pooled_array: Array2<f32> =
            Embedder::apply_pooling(self.config.pooling, array.view());
        if self.config.normalize {
            pooled_array.axis_iter_mut(Axis(0)).for_each(|mut row| {
                let norm = row.mapv(|x| x.powi(2)).sum().sqrt();
                if norm > 1e-12 {
                    row.map_inplace(|x| *x /= norm)
                }
            })
        }

        Ok(pooled_array)
    }
    fn embed_job_descriptions(
        &mut self,
        job_descriptions: Vec<PreprocessedJobDescription>,
    ) -> Result<EmbeddedJobDescriptions<'static>, SoccerError> {
        let (ids, text): (Vec<Cow<'_, str>>, Vec<String>) = job_descriptions
            .into_iter()
            .map(|job| (Cow::Owned(job.id), job.cleaned_text))
            .unzip();
        let embeddings = self.embed_text(&text)?;

        Ok(EmbeddedJobDescriptions { ids, embeddings })
    }

    /// Pools batched token embeddings according to `strategy`.
    pub fn apply_pooling(strategy: PoolingStrategy, view: ArrayView3<f32>) -> Array2<f32> {
        // note Axis 0 == seq in batch (seq 0, 1, 2, 3,...)
        //      Axis 1 == token in seq (token 0, 1, 2, 3,...)
        //      Axis 2 == embedding entry (vec element 0, 1, 2, ...)
        match strategy {
            PoolingStrategy::Cls => {
                // get the 0th token (i.e., the cls token)
                view.index_axis(ndarray::Axis(1), 0).to_owned()
            }
            PoolingStrategy::Mean => {
                // get the mead across the sequences (for each element)
                // this is infallable so no worries about unwrapping.
                view.mean_axis(ndarray::Axis(1)).unwrap()
            }
        }
    }
}

#[derive(Debug)]
/// Input job text and optional prior classification indices.
pub struct JobDescription {
    /// Caller-provided identifier retained in the result.
    pub id: String,
    /// Primary text: job title for SOCcerNET or products/services for CLIPS.
    pub text1: String, // For CLIPS -> PS, SOCcerNET -> JobTitle
    /// Optional secondary text, normally SOCcerNET job tasks.
    pub text2: Option<String>, //For CLIPS -> None, SOCcerNET -> JobTasks
    /// Indices of prior classifications supplied to the model.
    pub multihot_prior: Box<[u16]>,
}
impl From<(String, String)> for JobDescription {
    fn from(value: (String, String)) -> Self {
        Self {
            id: value.0,
            text1: value.1,
            text2: None,
            multihot_prior: Box::default(),
        }
    }
}
impl From<(String, String, String)> for JobDescription {
    fn from(value: (String, String, String)) -> Self {
        Self {
            id: value.0,
            text1: value.1,
            text2: Some(value.2),
            multihot_prior: Box::default(),
        }
    }
}
impl From<(&str, &str)> for JobDescription {
    fn from(value: (&str, &str)) -> Self {
        Self {
            id: value.0.to_string(),
            text1: value.1.to_string(),
            text2: None,
            multihot_prior: Box::default(),
        }
    }
}
impl From<(&str, &str, &str)> for JobDescription {
    fn from(value: (&str, &str, &str)) -> Self {
        Self {
            id: value.0.to_string(),
            text1: value.1.to_string(),
            text2: Some(value.2.to_string()),
            multihot_prior: Box::default(),
        }
    }
}

#[derive(Debug)]
/// A job description after model-specific text preprocessing.
pub struct PreprocessedJobDescription {
    /// Caller-provided job identifier.
    pub id: String,
    /// Cleaned text ready for tokenization.
    pub cleaned_text: String,
}

#[derive(Debug)]
/// Batched job identifiers and their embedding matrix.
pub struct EmbeddedJobDescriptions<'a> {
    /// Identifiers corresponding to embedding rows.
    pub ids: Vec<Cow<'a, str>>,
    /// Matrix with one embedding vector per row.
    pub embeddings: Array2<f32>,
}
impl<'a> EmbeddedJobDescriptions<'a> {
    /// Returns the number of embedded jobs.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

#[derive(Debug)]
/// Ranked classification scores for one job description.
pub struct CodedJobDescription<'a> {
    /// Caller-provided job identifier.
    pub id: Cow<'a, str>,
    /// Classification indices ordered from highest to lowest score.
    pub scored_code_index: Vec<Scored<usize>>,
}

#[derive(Debug)]
/// Loaded embedding and classifier sessions for one configured model.
pub struct SoccerPipeline {
    /// Text embedding pipeline.
    pub embedder: Embedder,
    /// SOCcerNET or CLIPS classifier session.
    pub soccer_session: Session,

    /// Configuration used to build this pipeline.
    pub config: ModelConfig,
}

impl SoccerPipeline {
    /// Embeds a batch of jobs without running classification.
    pub fn embed_only(
        &mut self,
        job_descriptions: &[&JobDescription],
    ) -> Result<EmbeddedJobDescriptions<'static>, SoccerError> {
        // still need to clean the job description...
        let preprocessed_job_descriptions =
            self.config.model_type.preprocess_batch(job_descriptions);

        self.embedder
            .embed_job_descriptions(preprocessed_job_descriptions)
    }

    /// Runs prediction using parallel input columns.
    ///
    /// All supplied columns must have identical lengths.
    pub fn predict_from_columns<'a>(
        &mut self,
        id: &[&'a str],
        text1: &[&'a str],
        text2: Option<&[&'a str]>,
        prior: &[Box<[u16]>],
    ) -> Result<Vec<CodedJobDescription<'a>>, SoccerError> {
        let row_count = id.len();
        let text2_len = text2.map_or(row_count, |values| values.len());
        if text1.len() != row_count || text2_len != row_count || prior.len() != row_count {
            return Err(SoccerError::BuilderError(format!(
                "Input column lengths do not match: id={}, text1={}, text2={}, prior={}",
                row_count,
                text1.len(),
                text2_len,
                prior.len()
            )));
        }

        // create the crosswalked input
        let multihot_array = self.create_multihot_array2d_col(prior);

        // preprocess the job descriptions
        let clean_text: Vec<String> = match text2 {
            Some(t2) => text1
                .iter()
                .zip(t2.iter())
                .map(|(txt1, txt2)| self.config.model_type.preprocess(txt1, Some(txt2)))
                .collect(),
            None => text1
                .iter()
                .map(|txt1| self.config.model_type.preprocess(txt1, None))
                .collect(),
        };

        let embeded_text: Array2<f32> = self.embedder.embed_text(&clean_text)?;
        let embedded_jobs: EmbeddedJobDescriptions<'a> = EmbeddedJobDescriptions {
            ids: id.iter().map(|&i| Cow::Borrowed(i)).collect(),
            embeddings: embeded_text,
        };
        self.run_soccer(embedded_jobs, multihot_array)
    }

    fn create_multihot_array2d_col(&self, prior: &[Box<[u16]>]) -> Array2<f32> {
        let mut multihot: Array2<f32> = Array2::zeros((prior.len(), self.config.output_dim()));
        prior.iter().enumerate().for_each(|(row_indx, job)| {
            job.iter()
                .for_each(|&col_indx| multihot[[row_indx, col_indx as usize]] = 1.0)
        });
        multihot
    }

    /// Runs prediction for a single job description.
    pub fn run1(
        &mut self,
        job_description: &JobDescription,
    ) -> Result<CodedJobDescription<'static>, SoccerError> {
        let jobs = &[job_description];
        let multihot_array = self.create_multihot_array2d(jobs);
        let preprocessed_job_descriptions = self.config.model_type.preprocess_batch(jobs);
        let embedded_jobs = self
            .embedder
            .embed_job_descriptions(preprocessed_job_descriptions)?;

        self.run_soccer(embedded_jobs, multihot_array)
            .and_then(|vec| {
                vec.into_iter()
                    .next()
                    .ok_or(SoccerError::InferenceError("No Value returned".to_string()))
            })
    }
    /// Runs prediction for a batch of job descriptions.
    pub fn run(
        &mut self,
        job_descriptions: &[&JobDescription],
    ) -> Result<Vec<CodedJobDescription<'static>>, SoccerError> {
        // create the crosswalked input
        let multihot_array = self.create_multihot_array2d(job_descriptions);

        // preprocess the job descriptions
        let preprocessed_job_descriptions =
            self.config.model_type.preprocess_batch(job_descriptions);

        // Embed the text...
        let embedded_jobs = self
            .embedder
            .embed_job_descriptions(preprocessed_job_descriptions)?;

        // run soccer....
        self.run_soccer(embedded_jobs, multihot_array)
    }

    fn create_multihot_array2d(&self, job_descriptions: &[&JobDescription]) -> Array2<f32> {
        let mut multihot: Array2<f32> =
            Array2::zeros((job_descriptions.len(), self.config.output_dim()));
        job_descriptions
            .iter()
            .enumerate()
            .for_each(|(row_indx, job)| {
                job.multihot_prior
                    .iter()
                    .for_each(|&col_indx| multihot[[row_indx, col_indx as usize]] = 1.0)
            });
        multihot
    }

    fn run_soccer<'a>(
        &mut self,
        embedded_jobs: EmbeddedJobDescriptions<'a>,
        xw_input: Array2<f32>,
    ) -> Result<Vec<CodedJobDescription<'a>>, SoccerError> {
        let embedding_values = Value::from_array(embedded_jobs.embeddings)
            .map(|v| v.into_dyn())
            .map_err(|e| SoccerError::InferenceError(e.to_string()))?;
        let xw_values = Value::from_array(xw_input)
            .map(|v| v.into_dyn())
            .map_err(|e| SoccerError::InferenceError(e.to_string()))?;
        let input = ort::inputs![
                "embedded_input" => embedding_values,
                "crosswalked_inp" => xw_values,
        ];
        let soccer_results = self
            .soccer_session
            .run(input)
            .map_err(|e| SoccerError::InferenceError(e.to_string()))?;

        let (shape, slice) = soccer_results[self.config.model_output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| SoccerError::InferenceError(e.to_string()))?;
        let view = ArrayView2::from_shape((shape[0] as usize, shape[1] as usize), slice)
            .map_err(|e| SoccerError::InferenceError(e.to_string()))?;

        let sorted: Vec<CodedJobDescription> = view
            .axis_iter(Axis(0))
            .zip(embedded_jobs.ids)
            .map(|(row, id)| CodedJobDescription {
                id,
                scored_code_index: Self::argsort(row),
            })
            .collect();

        Ok(sorted)
    }

    /// Returns row indices and scores sorted from highest to lowest score.
    pub fn argsort(row: ArrayView1<f32>) -> Vec<Scored<usize>> {
        let mut zipped: Vec<Scored<usize>> =
            row.iter().enumerate().map(|(i, &f)| Scored(i, f)).collect();

        zipped.sort_by(|a, b| b.1.total_cmp(&a.1));

        zipped
    }
}

impl SoccerBuilder for SoccerPipeline {
    fn build(config: &ModelConfig) -> Result<SoccerPipeline, SoccerError> {
        let cache = Cache::new()?;

        let embedder = Embedder::new(config, &cache)?;
        let soccer_model_path = cache.get_from_url(&config.model_url)?;
        let soccer_session = Session::builder()
            .map_err(|e| SoccerError::BuilderError(format!("SOCcer session E1: {}", e)))?
            .with_memory_pattern(true)
            .map_err(|e| SoccerError::BuilderError(format!("SOCcer session E2: {}", e)))?
            .commit_from_file(soccer_model_path)
            .map_err(|e| SoccerError::BuilderError(format!("SOCcer session E3: {}", e)))?;

        Ok(SoccerPipeline {
            embedder,
            soccer_session,
            config: config.clone(),
        })
    }
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
/// Available configurations and default version for one model family.
pub struct VersionedModel {
    #[serde(alias = "default")]
    default_version: String,

    versions: HashMap<String, ModelConfig>,
}

/// Model-specific text preprocessing behavior.
pub trait PreprocessStrategy {
    /// Cleans and combines one model input.
    fn preprocess<T: AsRef<str>>(&self, text1: T, text2: Option<T>) -> String;
    /// Preprocesses a batch of job descriptions.
    fn preprocess_batch(
        &self,
        job_descriptions: &[&JobDescription],
    ) -> Vec<PreprocessedJobDescription>;
}

#[derive(Debug, Eq, PartialEq, Hash, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
/// Supported SOCcer model families.
pub enum ModelType {
    /// Occupational coding from job title and task text.
    SOCcerNET,
    /// Industry coding from products and services text.
    CLIPS,
}
impl FromStr for ModelType {
    type Err = SoccerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "soccernet" => Ok(ModelType::SOCcerNET),
            "clips" => Ok(ModelType::CLIPS),
            sys => Err(SoccerError::BuilderError(format!(
                "Unknown Model Type {}",
                sys
            ))),
        }
    }
}
impl PreprocessStrategy for ModelType {
    fn preprocess<T: AsRef<str>>(&self, text1: T, text2: Option<T>) -> String {
        let cleaned_text1 = clean_free_text(&text1);
        let cleaned_text2 = text2
            .as_ref()
            .map(clean_free_text)
            .filter(|s| !s.is_empty()); // If it's "", it becomes None

        match self {
            ModelType::SOCcerNET => cleaned_text2
                .map(|ct2| format!("{} {}", cleaned_text1, ct2))
                .unwrap_or(cleaned_text1),
            ModelType::CLIPS => cleaned_text1,
        }
    }

    fn preprocess_batch(
        &self,
        job_descriptions: &[&JobDescription],
    ) -> Vec<PreprocessedJobDescription> {
        job_descriptions
            .iter()
            .map(|job| {
                let cleaned_text = self.preprocess(&job.text1, job.text2.as_ref());
                PreprocessedJobDescription {
                    id: job.id.clone(),
                    cleaned_text,
                }
            })
            .collect()
    }
}

pub struct Classifier {
    embedding_model: String,
    model_url: String,
}

#[derive(Debug, Deserialize)]
/// Registry of model configurations embedded in the crate.
pub struct StartupConfig(HashMap<ModelType, VersionedModel>);
impl StartupConfig {
    /// Returns an exact model version when configured.
    pub fn get_config(&self, name: &ModelType, version: &str) -> Option<&ModelConfig> {
        self.0
            .get(name)
            .and_then(|models| models.versions.get(version))
    }
    /// Returns the configured default version for a model family.
    pub fn get_default_version(&self, name: &ModelType) -> Option<&ModelConfig> {
        let versioned_model = self.0.get(name)?;
        let default_version = versioned_model.default_version.as_str();
        versioned_model.versions.get(default_version)
    }
}

// This is the singleton... Only run at startup...
/// Model configurations embedded from `data/classifier.json`.
pub static MODEL_CONFIG: Lazy<StartupConfig> = Lazy::new(|| {
    let json_content = include_str!("../data/classifier.json");

    serde_json::from_str(json_content).expect(
        "Crate Internal Configuration Error: Failed to parse classifier configuration file.",
    )
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crosswalk::{CLASSIFICATION_SYSTEM_REGISTRY, KnownClassificationSystem};

    fn display_soccer_results(
        results: Vec<CodedJobDescription>,
        n: usize,
        results_coding_system: Arc<ClassificationSystem>,
    ) {
        results.iter().for_each(|job| {
            print!("{}", job.id);
            job.scored_code_index
                .iter()
                .take(n)
                .for_each(|scored_index| {
                    let (cde, title) = results_coding_system
                        .get_code_title(scored_index.0 as u32)
                        .unwrap();
                    print!("{}  {:10.10}  {:0.4}\t", cde, title, scored_index.1);
                });
            println!("");
        });
    }

    #[test]
    fn test_load() {
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.0");
        assert!(config.is_some());
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.1");
        assert!(config.is_none());
        let config = MODEL_CONFIG.get_config(&ModelType::CLIPS, "1.0.0");
        assert!(config.is_some());

        let config = MODEL_CONFIG.get_default_version(&ModelType::SOCcerNET);
        assert!(config.is_some());
        println!("{:?}", config.unwrap())
    }

    #[test]
    fn test_soccer() {
        let config = MODEL_CONFIG
            .get_config(&ModelType::SOCcerNET, "1.0.0")
            .unwrap();
        let mut runtime = SoccerPipeline::build(config).unwrap();
        let expected_values: Vec<Vec<f32>> = vec![
            vec![0.9931, 0.1799, 0.0282, 0.0248, 0.0222], // plumber (from JS version)
            vec![0.9900, 0.6386, 0.6347, 0.5575, 0.3101], // doctor
        ];
        let expected_codes: Vec<Vec<&str>> = vec![
            vec!["47-2152", "47-3015", "51-9199", "47-2151", "47-2061"],
            vec!["29-1069", "29-1067", "29-1063", "29-1199", "29-1011"],
        ];

        let job_descriptions: Vec<JobDescription> = vec![
            ("test01-1", "plumber").into(),
            ("test01-2", "doctor").into(),
        ];
        let refs: Vec<&JobDescription> = job_descriptions.iter().collect();

        let res = runtime.run(&refs).unwrap();
        let soc2010 = CLASSIFICATION_SYSTEM_REGISTRY
            .get_classification_system(KnownClassificationSystem::SOC2010);

        res.iter().enumerate().for_each(|(row, prediction)| {
            print!("{} {}:  ", row, prediction.id);
            prediction
                .scored_code_index
                .iter()
                .take(7)
                .for_each(|scored_index| {
                    print!(
                        " {} {:.4}   ",
                        soc2010.get_code(scored_index.0 as u32).unwrap(),
                        scored_index.1
                    )
                });
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
    fn test_soccer_2() {
        let config = MODEL_CONFIG
            .get_config(&ModelType::SOCcerNET, "1.0.0")
            .unwrap();
        let output_system = config.output_system();
        assert_eq!(840, output_system.len());

        let mut runtime = SoccerPipeline::build(config).unwrap();
        let job_descriptions: [JobDescription; 2] = [
            ("testjob-1", "--ceo--", "run company ").into(),
            ("testjob-2", "doctor. ", "    treat patients.  ").into(),
        ];
        let job_descriptions: Vec<&JobDescription> = job_descriptions.iter().collect();

        let res: Vec<CodedJobDescription> = runtime.run(&job_descriptions).unwrap();
        println!();
        display_soccer_results(
            res,
            3,
            CLASSIFICATION_SYSTEM_REGISTRY
                .get_classification_system(KnownClassificationSystem::SOC2010),
        );
        println!();
    }
}
