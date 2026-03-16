use playserde::{SOCcerJobDescription, load_json};

fn main() -> Result<(),String>{
    let x:Vec<SOCcerJobDescription> = load_json("/Users/druss/OneDrive/SOCcer/data/onet.json")?;
    x.iter().for_each(|job| {
        dbg!(&job);
        println!("{}",job)
    });
    Ok(())   
}
