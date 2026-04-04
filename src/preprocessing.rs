use std::collections::HashMap;
use once_cell::sync::Lazy;

pub static ABBEVIATIONS: Lazy<HashMap<String,String>> = Lazy::new(||{
    let json_content = include_str!("../data/abbrev.json");

    serde_json::from_str(json_content)
        .expect("Crate Internal Configuration Error: Failed to parse abbrevition file.")
});

pub fn clean_free_text<T:AsRef<str>>(text:T) -> String {
    let s = text.as_ref();
    let trimmed = s.trim_matches(|c: char| c.is_whitespace() || c == '-' || c == '.');

    let lowercase = trimmed.to_lowercase();
    if let Some(expanded) = ABBEVIATIONS.get(&lowercase) {
        // got to clone it...
        expanded.clone()
    } else {
        lowercase
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_it(){
        assert_eq!("disc jockey",clean_free_text("--dj---"));
    }
}