use std::{marker::PhantomData, str::FromStr, sync::Arc};

use csv::StringRecord;

use crate::{classifier::{JobDescription, ModelConfig, ModelType}, crosswalk::{Crosswalk, KnownClassificationSystem, KnownCrosswalk, Resolve}, error::MyError};

pub trait  JobMapper<S> {
    fn map(&self, source:&S, row_number:usize) -> JobDescription;
}

pub trait ErrorRecord{
    fn from_error_str(msg:&str) -> Self;
}
impl ErrorRecord for csv::StringRecord{
    fn from_error_str(msg:&str) -> Self {
        let mut r=Self::new();
        r.push_field(msg);
        r
    }
}
pub struct JobStream<I,M,S>
where
    I: Iterator<Item = Result<S,MyError>>,
    M: JobMapper<S>,
    S: ErrorRecord
{
    records: I,
    mapper:M,
    row_count:usize,
    _marker: PhantomData<S>
}

impl<I, M, S> JobStream<I, M, S>
where 
    I: Iterator<Item = Result<S, MyError>>,
    M: JobMapper<S>,
    S: ErrorRecord
{
    pub fn new(records:I, mapper:M) -> Self{
        Self{
            records,
            mapper,
            row_count: 0,
            _marker: PhantomData,
        }
    }
}

impl<I,M,S> Iterator for JobStream<I,M,S>
where
    I: Iterator<Item = Result<S, MyError>>,
    M: JobMapper<S>,
    S: ErrorRecord
{
    type Item = (Result<JobDescription,MyError>,S);

    fn next(&mut self) -> Option<Self::Item> {
        let raw_record = match self.records.next()? {
            Ok(rec) => rec,
            Err(e) => {
                let msg = match &e {
                    MyError::CSVSyntaxError{source,line} => 
                        format!("CSV Syntax error on line {}: {}", line, source),
                    _ => format!("CSV Syntax error on unknown line {}",e.to_string()),
                };
                let problem_child = S::from_error_str(&msg);
                
                return Some( (Err(e),problem_child) )
            }
        };
        self.row_count += 1;
        let job = self.mapper.map(&raw_record, self.row_count);
        Some((Ok(job),raw_record))
    }
}



pub struct CSVSchema {
    id_col: Option<usize>,
    text1_col: usize,
    text2_col: Option<usize>,
    prior_cols: Box<[(usize,Arc<Crosswalk>)]>
}

impl CSVSchema {
    pub(crate) fn from_headers(headers: &StringRecord, config:&ModelConfig) -> Result<Self,MyError>{
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

impl JobMapper<csv::StringRecord> for CSVSchema{
    fn map(&self, source:&csv::StringRecord, row_number:usize) -> JobDescription {
        let id = self.id_col
            .and_then( |col| source.get(col))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("row_{}",row_number + 1));

        let text1 = source.get(self.text1_col).unwrap_or_default().to_string(); 

        let text2= self.text2_col
            .and_then(|col| source.get(col))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let mut multihot_prior:Vec<u16> = Vec::new();
        for (col_idx, xw) in self.prior_cols.iter() {
            if let Some(value) = source.get(*col_idx).filter(|s| !s.is_empty()) {
                xw.crosswalk_into(&[value], &mut multihot_prior);
            }
        }
        JobDescription {
            id,
            text1,
            text2,
            multihot_prior: multihot_prior.into_boxed_slice(),
        }
    }
}




