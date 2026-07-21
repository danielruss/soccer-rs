#![cfg(any(
    all(target_os = "windows", target_env = "gnu"),
    all(target_os = "macos", target_arch = "x86_64")
))]

use soccer_rs::{
    JobDescription, MODEL_CONFIG, ModelType, SoccerBuilder, SoccerPipeline,
    initialize_onnx_runtime,
};

#[test]
#[ignore = "requires ORT_DYLIB_PATH to point to a packaged ONNX Runtime library"]
fn loads_supplied_runtime_and_runs_inference() {
    let runtime_path = std::env::var_os("ORT_DYLIB_PATH")
        .expect("ORT_DYLIB_PATH must point to libonnxruntime.dylib or onnxruntime.dll");

    initialize_onnx_runtime(&runtime_path).expect("dynamic ONNX Runtime should initialize");
    initialize_onnx_runtime(&runtime_path).expect("initialization should be idempotent");

    let config = MODEL_CONFIG
        .get_config(&ModelType::SOCcerNET, "1.0.0")
        .expect("SOCcerNET configuration should exist");
    let mut pipeline = SoccerPipeline::build(config).expect("pipeline should use dynamic ORT");
    let job: JobDescription = ("dynamic-ort-test", "plumber").into();
    let results = pipeline
        .run(&[&job])
        .expect("inference through dynamic ORT should succeed");

    assert_eq!(results.len(), 1);
}
