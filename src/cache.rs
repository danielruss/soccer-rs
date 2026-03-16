#![allow(dead_code)]
use std::{fs::File, io::Write, path::PathBuf};
use dirs;

use hf_hub::api::sync::Api;
use reqwest::blocking::get;

use crate::error::MyError;


pub struct Cache();

impl Cache {
    pub fn is_url<S:AsRef<str>>(url:S) -> bool{
        let url_str = url.as_ref();
        url_str.starts_with("https://") || url_str.starts_with("http://")
    }

    fn make_cache_error<E:ToString>(err: E) -> MyError{
        MyError::CacheError(err.to_string())
    }


    // A private function for getting the cache dir.
    fn get_cache_dir() -> Result<PathBuf,MyError>{
        let mut cache_dir = dirs::cache_dir()
            .ok_or_else(|| MyError::CacheError("Cache Dir is not defined".to_string()) )?;
        cache_dir.push("soccernet");
        std::fs::create_dir_all(&cache_dir)
            .map_err(Cache::make_cache_error)?;

        Ok(cache_dir)
    }

    fn get_cached_filename(url:&str) -> Result<PathBuf,MyError>{
        let filename = url.split("/").last()
            .ok_or_else(||Cache::make_cache_error("malformed URL"))?;
        Ok(Cache::get_cache_dir()?.join(filename))
    }

    // download using GET in one block...
    pub(crate) fn cache_text_from(url:&str) -> Result<PathBuf,MyError>{
        println!("\n\n");

        let cached_file = Cache::get_cached_filename(url)?;
        if !cached_file.exists() {
            println!("... downloading file from {}",url);
            let mut file = File::create(&cached_file).map_err(|e| MyError::CacheError(e.to_string()))?;
            let text = get(url)
                .map_err(|e|MyError::CacheError(e.to_string()))?
                .text()
                .map_err(|e| MyError::CacheError(e.to_string()))?;

            // write the text to the file...
            file.write_all(text.as_bytes())
                .map_err(|e| MyError::CacheError(e.to_string()))?;
        } else {
            println!("... using cached file {:?}",cached_file);
        }
        
        Ok(cached_file)
    }

    pub(crate) fn get_onnx_from(url:&str) -> Result<PathBuf,MyError>{
        let cached_file = Cache::get_cached_filename(url)?;
        if !cached_file.exists() {
            println!("... downloading file from {}",url);
            let mut file = File::create(&cached_file).map_err(Cache::make_cache_error)?;
            let client = reqwest::blocking::Client::new();
            let mut response = client.get(url).send().map_err(Cache::make_cache_error)?;
            std::io::copy(&mut response, &mut file).map_err(Cache::make_cache_error)?;
        } else {
            println!("... using cached file {:?}",cached_file);
        }

        Ok(cached_file)
    }

    pub(crate) fn get_from_hf_hub(model_id:&str) ->  Result<PathBuf,MyError>{
        let api=Api::new().map_err(Self::make_cache_error)?;
        let repo = api.model(model_id.to_string());
        repo.get("config.json").map_err(Self::make_cache_error)
    }

    pub(crate) fn get_from_data_dir(fname:&str) -> Result<PathBuf,MyError>{
        let mut file_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        file_path.push("data");
        file_path.push(fname);

        match file_path.exists() {
            true => Ok(file_path),
            false => Err(MyError::CacheError(format!("{} does not exist in the data directory.",fname)))
        }
    }

    pub fn list_cached_files(){
        if let Some(path) = Self::get_cache_dir()
            .ok()
            .filter(|p| p.is_dir()) {
                std::fs::read_dir(path)
                    .ok().into_iter()
                    // convert the Some(iter) into the iter..
                    .flatten()
                    .filter_map(|f| f.ok())
                    .for_each(|f| println!("{}",f.path().to_string_lossy() ));
            };
    }
}




#[cfg(test)]
mod tests {
    use crate::cache::Cache;


    #[test]
    fn test_cache(){
        let r=Cache::cache_text_from("https://danielruss.github.com/codingsystems/soc2010_all.csv");
        assert!(r.is_ok());
        let r=Cache::get_onnx_from("https://danielruss.github.io/soccer-models/SOCcer_v3.0.0.onnx");
        assert!(r.is_ok());
        Cache::list_cached_files();
        let p = Cache::get_from_hf_hub("Xenova/GIST-small-Embedding-v0");
        assert!(p.is_ok());
        println!("model path: {:?}",p.unwrap_or_default());
    }
}