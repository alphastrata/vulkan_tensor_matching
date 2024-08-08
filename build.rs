use std::env;
use std::fs;
use std::path::Path;

/// Build script for Vulkan Tensor Matching Library
/// Implements the compilation pipeline for tensorial template matching shaders
fn compile_shader(compiler: &shaderc::Compiler, shader_path: &str, output_name: &str) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join(output_name);
    let shader_source = fs::read_to_string(shader_path).unwrap_or_else(|_| panic!("Failed to read shader source: {}", shader_path));
    let compile_options = shaderc::CompileOptions::new().expect("Failed to init compile options");
    let binary_result = compiler
        .compile_into_spirv(
            &shader_source,
            shaderc::ShaderKind::Compute,
            shader_path,
            "main",
            Some(&compile_options),
        )
        .unwrap_or_else(|err| panic!("Failed to compile shader {}: {}", shader_path, err));
    fs::write(&out_path, binary_result.as_binary_u8()).unwrap_or_else(|_| panic!("Failed to write SPIR-V: {}", output_name));
    println!("cargo:rerun-if-changed={}", shader_path);
}

fn main() {
    println!("Running build script...");

    // Try to compile shaders, but don't fail if shaderc is not available
    match shaderc::Compiler::new() {
        Ok(compiler) => {
            println!("Compiling shaders...");
            // Compile the main correlation shader
            compile_shader(&compiler, "src/shader/corr.comp", "corr.spv");

            // Compile the tensorial shaders
            compile_shader(&compiler, "src/shader/tensor_generation_full.comp", "tensor_generation_full.spv");
            compile_shader(&compiler, "src/shader/tensorial_correlation.comp", "tensorial_correlation.spv");
            compile_shader(&compiler, "src/shader/tensorial_peak_detection.comp", "tensorial_peak_detection.spv");
        }
        Err(_) => {
            println!("cargo:warning=shaderc not available, creating dummy shader files");
            // Create dummy files so the build doesn't fail
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let out_path = Path::new(&out_dir);

            let corr_spv = out_path.join("corr.spv");
            fs::write(&corr_spv, b"").unwrap();

            let tensor_gen_spv = out_path.join("tensor_generation_full.spv");
            fs::write(&tensor_gen_spv, b"").unwrap();

            let tensor_corr_spv = out_path.join("tensorial_correlation.spv");
            fs::write(&tensor_corr_spv, b"").unwrap();

            let tensor_peak_spv = out_path.join("tensorial_peak_detection.spv");
            fs::write(&tensor_peak_spv, b"").unwrap();
        }
    }

    // Don't try to link shaderc_shared if it's not available
    // println!("cargo:rustc-link-lib=dylib=shaderc_shared");
}