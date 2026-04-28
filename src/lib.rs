#![doc = include_str!("../README.md")]
#![allow(dead_code)]
use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    str::FromStr,
    sync::Arc,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{from_reader, from_str};

use crate::{
    classifier::ModelConfig,
    crosswalk::{
        CLASSIFICATION_SYSTEM_REGISTRY, ClassificationSystem, KnownClassificationSystem,
        KnownCrosswalk,
    },
    io::{CSVSchema, JobStream},
};
mod cache;
mod classifier;
mod crosswalk;
mod error;
mod io;
mod preprocessing;

pub use crate::classifier::{
    CodedJobDescription, JobDescription, MODEL_CONFIG, ModelType, SoccerBuilder, SoccerPipeline,
};
pub use crate::crosswalk::Crosswalk;
pub use crate::error::MyError;

#[derive(Debug, Deserialize)]
pub struct SOCcerJobDescription {
    #[serde(rename = "Id")]
    id: String,
    #[serde(rename = "JobTitle")]
    job_title: String,
    #[serde(rename = "JobTask", default)]
    job_task: String,

    #[serde(default)]
    soc1980: Vec<String>,
    #[serde(default)]
    soc2010: Vec<String>,
    #[serde(default)]
    soc2018: Vec<String>,
    #[serde(default)]
    isco1988: Vec<String>,
    #[serde(default)]
    noc2011: Vec<String>,
}

impl Display for SOCcerJobDescription {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:\t{}\t{}\n\t{}\t{}\t{}\t{}\t-->soc 2010:\t{}",
            self.id,
            self.job_title,
            self.job_task,
            self.soc1980.join(", "),
            self.noc2011.join(", "),
            self.isco1988.join(", "),
            self.soc2018.join(", "),
            self.soc2010.join(", ")
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CLIPSJobDescription {
    id: String,
    products_services: String,

    #[serde(default)]
    job_task: String,
    #[serde(default)]
    sic1987: Vec<String>,
}

pub fn load_json<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<Vec<T>, String> {
    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    from_reader(reader).map_err(|e| e.to_string())
}

pub fn load_jsonl<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<Vec<T>, String> {
    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open file: {}", e))?;

    let reader = BufReader::new(file);
    let res: Vec<T> = reader
        .lines()
        .enumerate()
        .filter_map(|(line, row)| {
            let line_string = row
                .map_err(|e| eprintln!("JSON error line:{} {}", line, e.to_string()))
                .ok()?;
            if line_string.trim().is_empty() {
                return None;
            }
            from_str(&line_string)
                .map_err(|e| eprintln!("Deserialization error line:{} {}", line, e.to_string()))
                .ok()?
        })
        .collect();

    Ok(res)
}

pub fn load_csv_str<'a, T: AsRef<str> + ?Sized>(
    content: &'a T,
    model_config: &ModelConfig,
) -> Result<
    io::JobStream<
        Box<dyn Iterator<Item = Result<csv::StringRecord, MyError>> + 'a>,
        CSVSchema,
        csv::StringRecord,
    >,
    MyError,
> {
    let mut reader = csv::Reader::from_reader(content.as_ref().as_bytes());
    let headers = reader
        .headers()
        .map_err(|e| MyError::BuilderError(e.to_string()))?
        .clone();

    let mapper = CSVSchema::from_headers(&headers, model_config)?;
    let records_iter = Box::new(
        reader
            .into_records()
            .map(|r| r.map_err(|e| MyError::BuilderError(e.to_string()))),
    );

    Ok(JobStream::new(records_iter, mapper))
}

pub fn get_classification_system<T: AsRef<str>>(
    system: T,
) -> Result<Arc<ClassificationSystem>, MyError> {
    let classification_system = KnownClassificationSystem::from_str(system.as_ref())?;
    Ok(CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(classification_system))
}
pub fn get_crosswalk<T1: AsRef<str>, T2: AsRef<str>>(
    system1: T1,
    system2: T2,
) -> Result<Arc<Crosswalk>, MyError> {
    let known_xw = KnownCrosswalk::find(
        KnownClassificationSystem::from_str(system1.as_ref())?,
        KnownClassificationSystem::from_str(system2.as_ref())?,
    )?;
    Ok(CLASSIFICATION_SYSTEM_REGISTRY.get_crosswalk(known_xw))
}

#[derive(Serialize, Debug)]
pub struct SOCcerResult {
    pub code: String,
    pub title: String,
    pub score: f32,
}
impl From<(&str, &str, f32)> for SOCcerResult {
    fn from(value: (&str, &str, f32)) -> Self {
        SOCcerResult {
            code: value.0.to_string(),
            title: value.1.to_string(),
            score: value.2,
        }
    }
}

pub fn run_soccer_job(
    job_description: &JobDescription,
    version: &str,
    n: usize,
) -> Result<Box<[SOCcerResult]>, MyError> {
    let config =  match MODEL_CONFIG.get_config(&ModelType::SOCcerNET, version){
        Some(c) => c,
        None => MODEL_CONFIG.get_default_version(&ModelType::SOCcerNET).ok_or_else(||MyError::SoccerError(format!("Unable to get either the requested or default version of SOCcer: \n\trequested version: {}",version)))?
    };

    run_job(job_description, n, config)
}

pub fn run_clips_job(
    job_description: &JobDescription,
    version: &str,
    n: usize,
) -> Result<Box<[SOCcerResult]>, MyError> {
    let config =  match MODEL_CONFIG.get_config(&ModelType::CLIPS, version){
        Some(c) => c,
        None => MODEL_CONFIG.get_default_version(&ModelType::CLIPS).ok_or_else(||MyError::SoccerError(format!("Unable to get either the requested or default version of SOCcer: \n\trequested version: {}",version)))?
    };

    run_job(job_description, n, config)
}
fn run_job(
    job_description: &JobDescription,
    n: usize,
    config: &ModelConfig,
) -> Result<Box<[SOCcerResult]>, MyError> {
    let mut soccer = SoccerPipeline::build(config)?;
    let results = soccer.run1(&job_description)?;
    let classification_system = config.output_system();

    let result: Vec<SOCcerResult> = results
        .scored_code_index
        .iter()
        .take(n)
        .map(|si| {
            let (code, title) = classification_system.get_code_title(si.0 as u32).unwrap();
            println!("\t{}\t{}\t{:4}", code, title, si.1);
            SOCcerResult::from((code, title, si.1))
        })
        .collect();
    Ok(result.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::{
        JobDescription, MODEL_CONFIG, ModelType, SoccerBuilder, SoccerPipeline,
    };
    use itertools::{Either, Itertools};

    #[test]
    fn test_csv() {
        let config = MODEL_CONFIG
            .get_config(&ModelType::SOCcerNET, "1.0.0")
            .unwrap();
        let classification_system = config.output_system();
        let mut soccer = SoccerPipeline::build(config).expect("Problem building SOCcerNET");
        let csv_data = "id,JobTitle,jobtask,soc1980\n\
                        libtestcsv-1,Software Engineer,Writes code,111\n\
                        libtestcsv-2,Broken Row,\"Forgot to close quotes\",\n\
                        libtestcsv-3,Data Scientist,Analyzes data,\n
                        libtestcsv-4,lawyer,save clients from jail,";
        let n = 5;
        let job_descriptions =
            load_csv_str(csv_data, config).expect("The file structure should be valid");

        for chunk in &job_descriptions.into_iter().chunks(2) {
            let (successes, errors): (Vec<_>, Vec<_>) =
                chunk.into_iter().partition_map(|(res, rec)| match res {
                    Ok(jd) => Either::Left((jd, rec)),
                    Err(e) => Either::Right((e, rec)),
                });

            if !successes.is_empty() {
                let (job_batch, og_input): (Vec<JobDescription>, Vec<csv::StringRecord>) =
                    successes.into_iter().unzip();
                let refs: Vec<&JobDescription> = job_batch.iter().collect();
                let results = soccer.run(&refs).unwrap();

                results.iter().zip(og_input.iter()).for_each(|(job, sr)| {
                    println!("{:?}:", sr);
                    job.scored_code_index.iter().take(n).for_each(|si| {
                        let (code, title) =
                            classification_system.get_code_title(si.0 as u32).unwrap();
                        println!("\t{}\t{}\t{:4}", code, title, si.1)
                    });
                });
            }
            errors.iter().for_each(|(e, record)| {
                // Here you have both the error and the record that caused it!
                eprintln!("Skipping Record due to Error: {}\nRecord: {:?}", e, record);
            });
        }
    }
}
