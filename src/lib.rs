#![allow(dead_code)]
use std::{fmt::{Display, Formatter}, fs::File, io::{BufRead, BufReader}, path::Path};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{from_reader, from_str};
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