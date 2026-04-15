
#![allow(dead_code)]
use std::{collections::HashMap, fs::File, path::PathBuf, str::FromStr, sync::Arc};

use csv::{Reader, StringRecord};
use once_cell::sync::Lazy;
use serde::Deserialize;
use crate::{cache::Cache,error::MyError};

pub trait Resolve{
    type Output;
    fn resolve(&self) -> Self::Output;
}

#[derive(Debug)]
pub struct ClassificationSystemRegistry{
    pub soc1980: Arc<ClassificationSystem>,
    pub soc2010: Arc<ClassificationSystem>,
    pub sic1987: Arc<ClassificationSystem>,
    pub naics2022: Arc<ClassificationSystem>,

    pub soc1980_soc2010: Arc<Crosswalk>,
    pub sic1987_naics2022: Arc<Crosswalk>,
}

impl ClassificationSystemRegistry{
    pub fn get_classification_system(&self, classification_system:KnownClassificationSystem) -> Arc<ClassificationSystem>{
        match classification_system {
            KnownClassificationSystem::SOC1980 => self.soc1980.clone(),
            KnownClassificationSystem::SOC2010 => self.soc2010.clone(),
            KnownClassificationSystem::SIC1987 => self.sic1987.clone(),
            KnownClassificationSystem::NAICS2022 => self.naics2022.clone(),
        }
    }
    pub fn get_crosswalk(&self, crosswalk:KnownCrosswalk) -> Arc<Crosswalk> {
        match crosswalk {
            KnownCrosswalk::SOC1980SOC2010 => self.soc1980_soc2010.clone(),
            KnownCrosswalk::SIC1987NAICS2022 => self.sic1987_naics2022.clone(),
        }
    }
}


pub static CLASSIFICATION_SYSTEM_REGISTRY: Lazy<ClassificationSystemRegistry> = Lazy::new(||{

    let soc1980 = Arc::new(ClassificationSystem::from(KnownClassificationSystem::SOC1980));
    let soc2010 = Arc::new(ClassificationSystem::from(KnownClassificationSystem::SOC2010));
    let sic1987 = Arc::new(ClassificationSystem::from(KnownClassificationSystem::SIC1987));
    let naics2022 = Arc::new(ClassificationSystem::from(KnownClassificationSystem::NAICS2022));

    // Note: This should only panic in developement.  
    let soc1980_soc2010 = Arc::new(
        Crosswalk::new(KnownCrosswalk::SOC1980SOC2010,soc1980.clone(),soc2010.clone())
            .unwrap_or_else(|e| panic!("soc1980 -> soc2010 crosswalk is invalid: {e}"))
    );
    let sic1987_naics2022 = Arc::new(
        Crosswalk::new(KnownCrosswalk::SIC1987NAICS2022,sic1987.clone(),naics2022.clone())
            .unwrap_or_else(|e| panic!("sic1987 -> naics2022 crosswalk is invalid: {e}"))
    );


    ClassificationSystemRegistry { soc1980,soc2010,sic1987,naics2022,soc1980_soc2010,sic1987_naics2022 }
});


#[derive(Debug,PartialEq, Eq,Hash, Clone, Copy, Deserialize)]
pub enum KnownClassificationSystem{
    #[serde(rename = "soc1980")]
    SOC1980,
    #[serde(rename = "soc2010")]
    SOC2010,
    #[serde(rename = "sic1987")]
    SIC1987,
    #[serde(rename = "naics2022")]
    NAICS2022
}

impl Resolve for KnownClassificationSystem{
    type Output = Arc<ClassificationSystem>;

    fn resolve(&self) -> Self::Output{
        CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(*self)
    }
}

impl FromStr for KnownClassificationSystem {
    type Err = MyError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "soc1980" => Ok(Self::SOC1980),
            "soc2010" => Ok(Self::SOC2010),
            "sic1987" => Ok(Self::SIC1987),
            "naics2022" => Ok(Self::NAICS2022),
            _ => Err(MyError::ClassificationSystem(format!("Unknown classification system {}",s))),
        }
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

    pub fn len(&self) -> usize{
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


impl From<KnownClassificationSystem> for ClassificationSystem{
    fn from(value: KnownClassificationSystem) -> Self {
        let text_bytes: &[u8] = match value {
            KnownClassificationSystem::SOC1980 => include_bytes!("../data/soc1980.csv"),
            KnownClassificationSystem::SOC2010 => include_bytes!("../data/soc2010.csv"),
            KnownClassificationSystem::SIC1987 => include_bytes!("../data/sic1987.csv"),
            KnownClassificationSystem::NAICS2022 => include_bytes!("../data/naics2022.csv"),
        };

        // This will only panic at compile time IF there is a problem
        // with the classification system.
        let mut reader = Reader::from_reader(text_bytes);
        let rows:Vec<StringRecord> = reader.records()
            .enumerate()
            .map(|(line,r)| r.unwrap_or_else(|e| panic!("Classfication Setup: malformed record on line {} in {:?} CSV {}",line+1,value,e.to_string())))
//            .filter_map(|r| r.ok())
            .collect();

        Self::from_stringrecords(rows)
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
        let index_bytes:&[u8]= match value {
            KnownCrosswalk::SOC1980SOC2010 => include_bytes!("../data/soc1980_soc2010.csv"),
            KnownCrosswalk::SIC1987NAICS2022 => include_bytes!("../data/sic1987_naics2022.csv"),
        };
        let source_len = source_cs.len();

        let mut reader=Reader::from_reader(&index_bytes[..]);
        let index_mapping = reader.records()
            .try_fold(
                HashMap::with_capacity(source_len),
                 |mut acc: HashMap<u32, Vec<u32>>, rec: Result<StringRecord, csv::Error>| -> Result<HashMap<u32, Vec<u32>>, MyError> {
                    let record = rec.map_err(|_|MyError::CacheError(format!("Problem reading Crosswalk: {:?}",value)))?;
                    
                    let source_code = &record[0];
                    let target_code = &record[1];

                    let source_index = source_cs.lookup_index(source_code).ok_or(MyError::CacheError(format!("Crosswalk {:?}: Code {}: does not exist ",value,source_code)))?;
                    let target_index = target_cs.lookup_index(target_code).ok_or(MyError::CacheError(format!("Crosswalk {:?}: Code {}: does not exist ",value,target_code)))?;
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

    pub fn crosswalk_into(&self, codes: &[&str], out: &mut Vec<u16>) {
        codes.iter().for_each(|&code| {
            self.source_cs.lookup_index(code)
                .and_then(|src_idx| self.index_mapping.get(&src_idx))
                .map(|v| v.iter())
                .into_iter()
                .flatten()
                .for_each(|&i| out.push(i as u16));
        });
    }
}

#[derive(Debug,PartialEq, Eq,Hash,Clone,Copy)]
pub enum KnownCrosswalk{
    SOC1980SOC2010,
    SIC1987NAICS2022,
}
impl KnownCrosswalk {
    pub fn find(from:KnownClassificationSystem,to:KnownClassificationSystem) -> Result<KnownCrosswalk,MyError>{
        match (from,to) {
            (KnownClassificationSystem::SOC1980,KnownClassificationSystem::SOC2010) => Ok(KnownCrosswalk::SOC1980SOC2010),
            (KnownClassificationSystem::SIC1987,KnownClassificationSystem::NAICS2022) => Ok(KnownCrosswalk::SIC1987NAICS2022),
            (a,b) => Err(MyError::Crosswalk(format!("Unknown Crosswalk {:?} to {:?}",a,b))),
        }
    }
}

impl Resolve for KnownCrosswalk{
    type Output = Arc<Crosswalk>;

    fn resolve(&self) -> Self::Output{
        CLASSIFICATION_SYSTEM_REGISTRY.get_crosswalk(*self)
    }
}

#[derive(Debug,Deserialize)]
struct CrosswalkRow{
    source_code:String,
    target_code:String,
}


#[cfg(test)]
mod tests {
    use std::u32;

    use crate::{get_classification_system, get_crosswalk};

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
        let soc1980 = CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(KnownClassificationSystem::SOC1980);
        let soc2010 = CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system(KnownClassificationSystem::SOC2010);
        assert_eq!(soc1980.len(),665);
        assert_eq!(soc2010.len(),840);

        let ind_1131 = soc1980.lookup_index("1131");
        let ind_11_2022 = soc2010.lookup_index("11-2022");
        assert_eq!(ind_1131,Some(2));
        assert_eq!(ind_11_2022,Some(5));
    }

    #[test]
    fn test_kcs(){
        let soc1980 = CLASSIFICATION_SYSTEM_REGISTRY.get_classification_system (KnownClassificationSystem::SOC1980);
        let code = soc1980.get_code(0).unwrap();
        println!("{:?}",code);
        let title = soc1980.get_title(0).unwrap();
        println!("{:?}",title);
        let ct = soc1980.get_code_title(0).unwrap();
        println!("{:?}",ct);
    }

    #[test]
    fn test_crosswalk(){
        let soc1980_soc2010 = CLASSIFICATION_SYSTEM_REGISTRY.get_crosswalk(KnownCrosswalk::SOC1980SOC2010);

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

        println!("===== Testing SIC-> NAICS");
        let sic1987_naics2022 = get_crosswalk("sic1987", "naics2022").unwrap();
        dbg!(&sic1987_naics2022);
        let r1 =sic1987_naics2022.crosswalk(&["7372"]);
        r1.iter().for_each(|&res|{
            println!("=== {:?}",res);
            res.iter().for_each(|&index| println!("\t\t{:?}",soc1980_soc2010.target_cs.get_code_title(index)));
        });

        let mut my_vec: Vec<u16> = Vec::new();
        sic1987_naics2022.crosswalk_into(&["7372"], &mut my_vec);
        assert!(!my_vec.is_empty());
    }

    #[test]
    fn test_xw2(){

        let sic1987 = get_classification_system("sic1987").unwrap();
        //let naics2022 = get_classification_system("naics2022").unwrap();
        let sic1987_naics2022 = get_crosswalk("sic1987", "naics2022").unwrap();

        
        // the code 7372 exists in sic1987 is there a problem?
        let s7372 = sic1987.lookup_index("7372");
        assert!(s7372.is_some());
        let s_indx = s7372.unwrap_or(u32::MAX);
        assert_eq!(s_indx,866);
        let xw_indx = sic1987_naics2022.index_mapping.get(&s_indx);
        assert!(xw_indx.is_some());
        let xw_indx = xw_indx.unwrap();
        assert_eq!(xw_indx.len(),2);
    }

    #[test]
    fn test_len() {
        assert_eq!(840usize,CLASSIFICATION_SYSTEM_REGISTRY.soc2010.len());
        assert_eq!(689usize,KnownClassificationSystem::NAICS2022.resolve().len());
    }
}