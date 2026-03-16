
#![allow(dead_code)]
use std::{collections::HashMap, fs::File, path::PathBuf, sync::{Arc, RwLock}};

use csv::{Reader, StringRecord};
use once_cell::sync::Lazy;
use serde::Deserialize;
use crate::{cache::Cache,error::MyError};

#[derive(Debug)]
pub struct ClassificationSystemRegistry{
    classification_systems: HashMap<KnownClassificationSystem,Arc<ClassificationSystem>>,
    crosswalks: HashMap<KnownCrosswalk,Arc<Crosswalk>>
}
impl ClassificationSystemRegistry{
    fn get_or_load_classification_system(&mut self,system:KnownClassificationSystem) -> Result<Arc<ClassificationSystem>,MyError>{
        if let Some(cs) = self.classification_systems.get(&system) {
            Ok(Arc::clone(cs))
        } else {
            let cs= Arc::new(ClassificationSystem::try_from(system)?);
            self.classification_systems.insert(system, cs.clone());
            Ok(cs)
        }
    }

    fn get_or_load_crosswalk(&mut self,crosswalk:KnownCrosswalk,source_cs:Arc<ClassificationSystem>,target_cs:Arc<ClassificationSystem>) -> Result<Arc<Crosswalk>,MyError>{
        if let Some(xw) = self.crosswalks.get(&crosswalk) {
            Ok(Arc::clone(xw))
        } else {
            let xw= Arc::new(Crosswalk::new(crosswalk,source_cs,target_cs)?);   
            self.crosswalks.insert(crosswalk, xw.clone());
            
            Ok(xw)
        }
    }

    pub fn get_classification_system(system:KnownClassificationSystem) -> Result<Arc<ClassificationSystem>,MyError>{
        let mut reg = CLASSIFICATION_SYSTEM_REGISTRY.write().expect("Lock poisoned");
        reg.get_or_load_classification_system(system)
    }

    pub fn get_crosswalk(crosswalk:KnownCrosswalk) -> Result<Arc<Crosswalk>,MyError>{
        // if I already have the crosswalk short cut 
        {
            let reg = CLASSIFICATION_SYSTEM_REGISTRY.write().expect("Lock poisoned");
            if  let Some(xw) = reg.crosswalks.get(&crosswalk){
                return Ok(Arc::clone(xw));
            }
        }
        let (source,target) = match crosswalk {
            KnownCrosswalk::SOC1980SOC2010 => (KnownClassificationSystem::SOC1980,KnownClassificationSystem::SOC2010),
        };
        let source_cs = Self::get_classification_system(source)?;
        let target_cs = Self::get_classification_system(target)?;

        let mut reg = CLASSIFICATION_SYSTEM_REGISTRY.write().expect("Lock poisoned");
        reg.get_or_load_crosswalk(crosswalk,source_cs,target_cs)
    }
}

pub static CLASSIFICATION_SYSTEM_REGISTRY: Lazy<RwLock<ClassificationSystemRegistry>> = Lazy::new(||{
    RwLock::new(ClassificationSystemRegistry{
        classification_systems: HashMap::new(),
        crosswalks:HashMap::new(),
    })
});


#[derive(Debug,PartialEq, Eq,Hash, Clone, Copy)]
pub enum KnownClassificationSystem{
    SOC1980,
    SOC2010
}

impl KnownClassificationSystem{
    pub fn load_with<F>(&self, filter:F) -> ClassificationSystem
    where F:Fn(&StringRecord) -> bool {

        let text_str = match &self {
            KnownClassificationSystem::SOC1980 => include_str!("../data/soc1980.csv"),
            KnownClassificationSystem::SOC2010 => include_str!("../data/soc2010.csv")             
        };

        let mut reader = Reader::from_reader(text_str.as_bytes());
        let rows:Vec<StringRecord> = reader.records()
            .filter_map(|r| r.ok())
            .filter(filter)
            .collect();

        ClassificationSystem::from_stringrecords(rows)
    }
}



#[derive(Debug,Default)]
pub struct ClassificationSystem{
    codes: Box<[u8]>,
    titles: Box<[u8]>,
    code_offsets:Box<[u32]>,
    title_offsets:Box<[u32]>,

    lookup: HashMap<String, u32>,
}

/// Classification system use the following format:
///     code,title,Level,<...>
impl ClassificationSystem {
    pub fn lookup_index(&self,code:&str) -> Option<u32>{
        self.lookup.get(code).copied()
    }

    pub fn get_title<'a>(&'a self,index:u32) -> Option<&'a str>{
        let start = if index==0 {0} else {self.title_offsets.get((index-1) as usize).copied()?} as usize;
        let end = self.title_offsets.get(index as usize).copied()? as usize;

        Some(str::from_utf8( &self.titles[start..end] ).unwrap_or_default() )
    }
    pub fn get_code<'a>(&'a self,index:u32) -> Option<&'a str>{
        let start = if index==0 {0} else {self.code_offsets.get((index-1) as usize).copied()?} as usize;
        let end = self.code_offsets.get(index as usize).copied()? as usize;

        Some(str::from_utf8( &self.codes[start..end] ).unwrap_or_default() )
    }
    pub fn get_code_title<'a>(&'a self,index:u32) -> Option<(&'a str,&'a str)>{
        let code_start = if index==0 {0} else {self.code_offsets.get((index-1) as usize).copied()?} as usize;
        let code_end = self.code_offsets.get(index as usize).copied()? as usize;
        
        let title_start = if index==0 {0} else {self.title_offsets.get((index-1) as usize).copied()?} as usize;
        let title_end = self.title_offsets.get(index as usize).copied()? as usize;

        Some((
            str::from_utf8( &self.codes[code_start..code_end] ).unwrap_or_default(),
            str::from_utf8( &self.titles[title_start..title_end] ).unwrap_or_default()
        ))
    }

    fn len(&self) -> usize{
        self.lookup.len()
    }

    fn from_raw_rows(rows:Vec<ClassificationSystemRow>) -> Self{
        let mut code_bytes = Vec::with_capacity(rows.iter().map(|r| r.code.len()).sum());
        let mut title_bytes = Vec::with_capacity(rows.iter().map(|r| r.title.len()).sum());

        let mut code_offsets = Vec::with_capacity(rows.len());
        let mut title_offsets = Vec::with_capacity(rows.len());

        let mut lookup = std::collections::HashMap::with_capacity(rows.len());

        rows.into_iter().enumerate().for_each(|(index,row)|{
            code_bytes.extend_from_slice(row.code.as_bytes());
            let codes_end = code_bytes.len() as u32;
            code_offsets.push(codes_end);

            lookup.insert(row.code, index as u32);

            title_bytes.extend_from_slice(row.title.as_bytes());
            let title_end = title_bytes.len() as u32;
            title_offsets.push(title_end);

        });

        Self { 
            codes: code_bytes.into_boxed_slice(), 
            titles: title_bytes.into_boxed_slice(), 
            code_offsets: code_offsets.into_boxed_slice(), 
            title_offsets: title_offsets.into_boxed_slice(),
            lookup,
        }
    }

    fn from_stringrecords(rows:Vec<StringRecord>) -> Self{
        let mut code_bytes = Vec::with_capacity(rows.iter().map(|r| r.get(0).unwrap_or_default().len()).sum());
        let mut title_bytes = Vec::with_capacity(rows.iter().map(|r| r.get(0).unwrap_or_default().len()).sum());

        let mut code_offsets = Vec::with_capacity(rows.len());
        let mut title_offsets = Vec::with_capacity(rows.len());

        let mut lookup = std::collections::HashMap::with_capacity(rows.len());

        rows.into_iter().enumerate().for_each(|(index,row)|{
            let current_code = row.get(0).unwrap_or_default();
            code_bytes.extend_from_slice(current_code.as_bytes());
            let codes_end = code_bytes.len() as u32;
            code_offsets.push(codes_end);

            lookup.insert(current_code.to_owned(), index as u32);

            let current_title = row.get(1).unwrap_or_default();
            title_bytes.extend_from_slice(current_title.as_bytes());
            let title_end = title_bytes.len() as u32;
            title_offsets.push(title_end);

        });

        Self { 
            codes: code_bytes.into_boxed_slice(), 
            titles: title_bytes.into_boxed_slice(), 
            code_offsets: code_offsets.into_boxed_slice(), 
            title_offsets: title_offsets.into_boxed_slice(),
            lookup,
        }
    }

    fn get_csv_reader<S:AsRef<str>>(url_or_path:S) -> Result<Reader<File>,MyError>{
        let location = url_or_path.as_ref();
        let path = if Cache::is_url(location) {
            Cache::cache_text_from(location)?
        } else {
            PathBuf::from(location)
        };
        let file = File::open(path)?;

        Ok(Reader::from_reader(file))
    }

    fn from_csv<S:AsRef<str>>(url_or_path:S) ->Result<Self,MyError>{
        let mut reader=Self::get_csv_reader(url_or_path)?;

        let rows:Vec<StringRecord> = reader.records()
            .filter_map(|r| r.ok())
            .collect();

        Ok(Self::from_stringrecords(rows))
    }

    fn from_csv_filtered<S,F>(url_or_path:S, filter:F) ->Result<Self,MyError>
    where 
        S:AsRef<str>,
        F:Fn(&StringRecord) -> bool
    {
        let mut reader = Self::get_csv_reader(url_or_path)?;
        let rows:Vec<StringRecord> = reader.records()
            .filter_map(|r| r.ok())
            .filter(filter)
            .collect();

        Ok(Self::from_stringrecords(rows))
    }

}


impl TryFrom<KnownClassificationSystem> for ClassificationSystem{
    type Error=MyError;

    fn try_from(value: KnownClassificationSystem) -> Result<Self, Self::Error> {
        let text_str = match value {
            KnownClassificationSystem::SOC1980 => include_str!("../data/soc1980.csv"),
            KnownClassificationSystem::SOC2010 => include_str!("../data/soc2010.csv")             
        };

        let mut reader = Reader::from_reader(text_str.as_bytes());
        let rows:Vec<StringRecord> = reader.records()
            .filter_map(|r| r.ok())
            .collect();

        Ok(Self::from_stringrecords(rows))
    }
}

#[derive(Debug,Deserialize)]
struct ClassificationSystemRow{
    code:String,
    title:String,
    #[serde(rename="Level")]
    level:String,   
}

#[derive(Debug)]
pub struct Crosswalk{
    source_cs:Arc<ClassificationSystem>,
    target_cs:Arc<ClassificationSystem>,
    index_mapping:HashMap<u32,Vec<u32>>
}

impl Crosswalk {

    fn new(value: KnownCrosswalk,source_cs:Arc<ClassificationSystem>,target_cs:Arc<ClassificationSystem>) -> Result<Self, MyError> {                    
        let index_bytes= match value {
            KnownCrosswalk::SOC1980SOC2010 => include_bytes!("../data/soc1980_soc2010.csv"),
        };
        let source_len = source_cs.len();

        let mut reader=Reader::from_reader(&index_bytes[..]);
        let index_mapping = reader.records()
            .into_iter()
            .try_fold(
                HashMap::with_capacity(source_len),
                 |mut acc: HashMap<u32, Vec<u32>>, rec: Result<StringRecord, csv::Error>| -> Result<HashMap<u32, Vec<u32>>, MyError> {
                    let record = rec.map_err(|_|MyError::CacheError(format!("Problem reading Crosswalk: {:?}",value)))?;
                    
                    let source_code = &record[0];
                    let target_code = &record[1];

                    let source_index = source_cs.lookup_index(source_code).ok_or(MyError::CacheError(format!("Code {}: does not exist ",source_code)))?;
                    let target_index = target_cs.lookup_index(target_code).ok_or(MyError::CacheError(format!("Code {}: does not exist ",target_code)))?;
                    acc.entry(source_index).or_default().push(target_index);

                    Ok(acc)
            })?;
        
        Ok(Crosswalk{
            source_cs,
            target_cs,
            index_mapping,
        })
    }

    fn crosswalk(&self,codes:&[&str]) -> Vec<&[u32]>{
        codes.iter()
            .map(|&code| {
                self.source_cs.lookup_index(code)
                    .and_then(|src_indx| self.index_mapping.get(&src_indx))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            })
            .collect()
    }
}



#[derive(Debug,PartialEq, Eq,Hash,Clone,Copy)]
pub enum KnownCrosswalk{
    SOC1980SOC2010,
}

#[derive(Debug,Deserialize)]
struct CrosswalkRow{
    source_code:String,
    target_code:String,
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_download(){
        let x: Result<ClassificationSystem, MyError> = ClassificationSystem::from_csv("https://danielruss.github.io/codingsystems/soc1980_all.csv");
        assert!(x.is_ok());
        let x=x.unwrap();
        println!("{}",x.codes[0]);
        println!("{:?}",x.lookup.get("9911"));

        let x: Result<ClassificationSystem, MyError> = ClassificationSystem::from_csv_filtered(
            "https://danielruss.github.io/codingsystems/soc1980_all.csv",
            |row| row.get(2).map_or(false, |v|v=="unit")
        );
        assert!(x.is_ok());
    }

    #[test]
    fn test_load(){
        let soc1980 = ClassificationSystemRegistry::get_classification_system(KnownClassificationSystem::SOC1980).unwrap_or_default();
        let soc2010 = ClassificationSystemRegistry::get_classification_system(KnownClassificationSystem::SOC2010).unwrap_or_default();
        assert_eq!(soc1980.len(),665);
        assert_eq!(soc2010.len(),840);

        let ind_1131 = soc1980.lookup_index("1131");
        let ind_11_2022 = soc2010.lookup_index("11-2022");
        assert_eq!(ind_1131,Some(2));
        assert_eq!(ind_11_2022,Some(5));
    }

    #[test]
    fn test_file(){
        let path = PathBuf::from("/Users/druss/Downloads/soc1980_all.csv");

        let x = ClassificationSystem::from_csv_filtered(
            path.to_string_lossy(),
             |row| row.get(8).map_or(false, |v|v=="TRUE")
        );
        assert!(x.is_ok());
        let x=x.unwrap();
        println!("{:?}",x.lookup.get("9911"));
        println!("{:?}",x.lookup.get("112"));

        let c9991_index = x.lookup.get("9911");
        let c112_index = x.lookup.get("112");
        assert_eq!(c9991_index.copied(),Some(628));
        assert_eq!(c112_index.copied(),Some(11));
        let test_code = x.get_code_title(c9991_index.copied().unwrap());
        println!("{:?}",test_code);

        assert_eq!(Some(("9911", "GRADUATE ASSISTANT")),test_code);
        let test_code = x.get_code_title(c112_index.copied().unwrap());
        println!("{:?}",test_code);
    }

    #[test]
    fn test_crosswalk(){
        let soc1980_soc2010 = ClassificationSystemRegistry::get_crosswalk(KnownCrosswalk::SOC1980SOC2010);
        assert!(soc1980_soc2010.is_ok());

        let soc1980_soc2010=soc1980_soc2010.unwrap();
        let r1 = soc1980_soc2010.crosswalk(&["111","1131"]);
        let r2 = soc1980_soc2010.crosswalk(&["110"]);
        println!("{:?}\n{:?}",r1,r2);

        r1.iter().for_each(|&res|{
            println!("=== {:?}",res);
            res.iter().for_each(|&index| println!("\t\t{:?}",soc1980_soc2010.target_cs.get_code_title(index)));
        });
        r2.iter().for_each(|&res|{
            println!("=== {:?}",res);
            res.iter().for_each(|&index| println!("\t\t{:?}",soc1980_soc2010.target_cs.get_code_title(index)));
        });
    }
}