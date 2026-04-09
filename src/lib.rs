#![allow(dead_code)]
use std::{fmt::{Display, Formatter}, fs::File, io::{BufRead, BufReader}, path::Path, str::FromStr, sync::Arc};
use csv::StringRecord;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{from_reader, from_str};

use crate::{classifier::{JobDescription, ModelConfig, ModelType}, crosswalk::{Crosswalk, KnownClassificationSystem, KnownCrosswalk, Resolve}, error::MyError};
mod cache;
mod classifier;
mod crosswalk;
mod error;
mod preprocessing;

#[derive(Debug,Deserialize)]
pub struct SOCcerJobDescription{
    #[serde(rename = "Id")]
    id:String,
    #[serde(rename = "JobTitle")]
    job_title:String,
    #[serde(rename = "JobTask", default)]
    job_task:String,

    #[serde(default)]
    soc1980:Vec<String>,
    #[serde(default)]
    soc2010:Vec<String>,
    #[serde(default)]
    soc2018:Vec<String>,
    #[serde(default)]
    isco1988:Vec<String>,
    #[serde(default)]
    noc2011:Vec<String>,
}


impl Display for SOCcerJobDescription{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:\t{}\t{}\n\t{}\t{}\t{}\t{}\t-->soc 2010:\t{}", self.id, self.job_title, self.job_task,
            self.soc1980.join(", "),self.noc2011.join(", "),self.isco1988.join(", "),self.soc2018.join(", "),self.soc2010.join(", "))
    }
}


#[derive(Debug,Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CLIPSJobDescription{
    id:String,
    products_services:String,

    #[serde(default)]
    job_task:String,
    #[serde(default)]
    sic1987:Vec<String>,
}

pub fn load_json<T:DeserializeOwned, P:AsRef<Path>>(path: P) -> Result<Vec<T>,String>{
     let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    from_reader(reader).map_err(|e| e.to_string())
}

pub fn load_jsonl<T:DeserializeOwned, P:AsRef<Path>>(path: P) -> Result<Vec<T>,String>{
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open file: {}", e))?;

    let reader = BufReader::new(file);
    let res:Vec<T> = reader.lines()
        .enumerate()
        .filter_map( |(line,row)| {
            let line_string = row.map_err(|e| eprintln!("JSON error line:{} {}",line,e.to_string())).ok()?;
            if line_string.trim().is_empty(){
                return None;
            }
            from_str(&line_string).map_err(|e| eprintln!("Deserialization error line:{} {}",line,e.to_string())).ok()?
        })
        .collect();

    Ok(res)
}



struct CSVSchema {
    id_col: Option<usize>,
    text1_col: usize,
    text2_col: Option<usize>,
    prior_cols: Box<[(usize,Arc<Crosswalk>)]>
}

impl CSVSchema {
    fn from_headers(headers: &StringRecord, config:&ModelConfig) -> Result<Self,MyError>{
        let model_type = config.model_type;
        let output_classfication = config.output_classification_system;
        let id_column = "id";
        let (text1_column,text2_column) = match model_type{
            ModelType::SOCcerNET => ("jobtitle",Some("jobtask")),
            ModelType::CLIPS => ("products_services",None),
        };
        
        let mut id_col: Option<usize>=None;
        let mut text1_col: Option<usize>=None;
        let mut text2_col: Option<usize>=None;
        let mut prior_cols: Vec<(usize, Arc<Crosswalk>)> = Vec::new();

        headers.iter().enumerate().for_each(|(col,col_header)|{
            let c = col_header.to_lowercase();
            match c.as_ref() {
                s if s==id_column => {id_col = Some(col);},
                s if s == text1_column => {text1_col = Some(col);},
                s if text2_column.map_or(false, |t2|t2==s) => { text2_col = Some(col);},
                s => {
                    KnownClassificationSystem::from_str(s)
                        .and_then(|source| KnownCrosswalk::find(source,output_classfication) )
                        .map(|kxw| { 
                            let xw = kxw.resolve();
                            prior_cols.push( (col,xw) ) 
                        })
                        .ok();
                }
            }
        });

        let text1_col = text1_col.ok_or_else(|| {
            MyError::BuilderError(format!("CSV missing required column '{}'", text1_column))
        })?;

        let prior_cols = prior_cols.into_boxed_slice();
        Ok(CSVSchema {id_col,text1_col,text2_col,prior_cols} )

    }
}

pub fn load_csv<T:AsRef<str>>(content:T,model_config:&ModelConfig) -> Result<Vec<Result<(JobDescription,StringRecord),MyError>>,MyError>{
    let mut reader = csv::Reader::from_reader(content.as_ref().as_bytes());
    let headers = reader.headers().map_err(|e|MyError::BuilderError(e.to_string()))?.clone();

    let schema = CSVSchema::from_headers(&headers, &model_config)?;
    Ok( reader.records().enumerate()
        .map( |(row_number, rec)|{
            let record = rec.map_err(|e| MyError::BuilderError(e.to_string()))?;

            let id = schema.id_col
                .and_then( |col| record.get(col))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("row_{}",row_number + 1));

            let text1 = record.get(schema.text1_col).unwrap_or_default().to_string();
            let text2= schema.text2_col
                .and_then(|col| record.get(col))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());

            let mut multihot_prior:Vec<u16> = Vec::new();
            for (col_idx, xw) in schema.prior_cols.iter() {
                if let Some(value) = record.get(*col_idx).filter(|s| !s.is_empty()) {
                    xw.crosswalk_into(&[value], &mut multihot_prior);
                }
            }
            let multihot_prior = multihot_prior.into_boxed_slice();

            let job = JobDescription { id, text1, text2, multihot_prior };
            Ok((job, record))
        }).collect())


}

#[cfg(test)]
mod tests {
    use crate::classifier::MODEL_CONFIG;

    use super::*;

    #[test]
    fn test_it(){
        let config = MODEL_CONFIG.get_config(&ModelType::SOCcerNET, "1.0.0").unwrap();
        let csv_data = "id,JobTitle,description,soc1980\n\
                        1,Software Engineer,Writes code,111\n\
                        2,Broken Row,\"Forgot to close quotes\",\n\
                        3,Data Scientist,Analyzes data,";
        let results = load_csv(csv_data, config).expect("The file structure should be valid");
        assert_eq!(results.len(),3);
        dbg!(results);
    }
}