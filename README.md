# soccer-rs

`soccer-rs` provides occupational and industry coding for epidemiologic and
survey data. It contains the shared Rust implementation used to preprocess
text, run the SOCcerNET and CLIPS ONNX models, and crosswalk between supported
classification systems.

- **SOCcerNET** predicts 2010 Standard Occupational Classification (SOC 2010)
  codes from a job title and optional job-task text.
- **CLIPS** predicts 2022 North American Industry Classification System
  (NAICS 2022) codes from products-and-services text.

## Installation

Add the crate to a Rust project:

```console
cargo add soccer-rs
```

The crate uses ONNX Runtime through `ort`. Building it therefore requires a
platform supported by the configured ONNX Runtime distribution.

## Model cache

Classification data and crosswalks are embedded in the crate. The larger model
files are downloaded on demand the first time a pipeline is constructed:

- the GIST embedding model and tokenizer come from Hugging Face;
- the SOCcerNET and CLIPS classifiers come from the SOCcer model repository.

Downloads are managed by `soccer-cache`. Hugging Face files use the Hugging
Face cache, while SOCcer model files use the platform-specific user cache under
`soccernet`. Cached files are safe to remove because they can be downloaded
again when needed.

The companion `cache-models` command-line application can pre-download or clear
these files before running an application that embeds `soccer-rs`.

## SOCcerNET example

```no_run
use soccer_rs::{JobDescription, SoccerError, run_soccer_job};

fn main() -> Result<(), SoccerError> {
    let job: JobDescription = (
        "job-1",
        "registered nurse",
        "provides direct patient care",
    )
        .into();

    for result in run_soccer_job(&job, "1.0.0", 5)? {
        println!("{}\t{}\t{:.4}", result.code, result.title, result.score);
    }

    Ok(())
}
```

If the requested model version is unavailable, the configured default version
is used.

## CLIPS example

```no_run
use soccer_rs::{JobDescription, SoccerError, run_clips_job};

fn main() -> Result<(), SoccerError> {
    let job: JobDescription = (
        "business-1",
        "residential plumbing installation and repair",
    )
        .into();

    for result in run_clips_job(&job, "1.0.0", 5)? {
        println!("{}\t{}\t{:.4}", result.code, result.title, result.score);
    }

    Ok(())
}
```

## CSV input

`load_csv_str` maps CSV rows into job descriptions for a selected model. A
SOCcerNET CSV requires a `jobtitle` column and may include `id`, `jobtask`, and
recognized prior-classification columns. A CLIPS CSV requires a
`products_services` column.

Malformed records are returned individually through the stream, allowing the
caller to report or skip them without discarding valid rows.

```no_run
use soccer_rs::{MODEL_CONFIG, ModelType, SoccerError, load_csv_str};

fn main() -> Result<(), SoccerError> {
    let config = MODEL_CONFIG
        .get_default_version(&ModelType::SOCcerNET)
        .ok_or_else(|| SoccerError::BuilderError("missing SOCcerNET configuration".into()))?;

    let csv = "id,jobtitle,jobtask\n1,plumber,repairs water pipes\n";
    for (job, original_record) in load_csv_str(&csv, config)? {
        match job {
            Ok(job) => println!("{}: {}", job.id, job.text1),
            Err(error) => eprintln!("{error}: {original_record:?}"),
        }
    }

    Ok(())
}
```

## Classification systems and crosswalks

The crate includes SOC 1980, NOC 2011, ISCO 1988, SOC 2010, SIC 1987, and
NAICS 2022 classification data. Use `get_classification_system` to look up
codes and titles and `get_crosswalk` to obtain a supported mapping.

```no_run
use soccer_rs::{SoccerError, get_crosswalk};

fn main() -> Result<(), SoccerError> {
    let crosswalk = get_crosswalk("soc1980", "soc2010")?;
    let mut target_indices = Vec::new();
    crosswalk.crosswalk_into(&["111", "1131"], &mut target_indices);
    Ok(())
}
```

Supported mappings are:

- SOC 1980 to SOC 2010
- NOC 2011 to SOC 2010
- ISCO 1988 to SOC 2010
- SIC 1987 to NAICS 2022

## Error handling

Public model, CSV, cache, and crosswalk operations return `SoccerError`.
Callers can match its variants when they need to distinguish caching, input,
embedding, inference, and output failures.

`load_jsonl` is intentionally lossy: unreadable or malformed lines are logged
and skipped, while valid records are returned.

## Development

```console
cargo fmt --check
cargo clippy --all-targets
cargo test --lib
```

The external-download test is ignored by default because it requires network
access. Run it explicitly with:

```console
cargo test test_download -- --ignored
```

## License

MIT
