use std::env;
use std::fs;
use std::path::Path;

fn compile_shader(compiler: &shaderc::Compiler, shader_path: &str, output_name: &str) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join(output_name);
    let shader_source = fs::read_to_string(shader_path)
        .unwrap_or_else(|_| panic!("Failed to read shader source: {}", shader_path));
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
    fs::write(&out_path, binary_result.as_binary_u8())
        .unwrap_or_else(|_| panic!("Failed to write SPIR-V: {}", output_name));
    println!("cargo:rerun-if-changed={}", shader_path);
}

fn main() {
    println!("Running build script...");

    match shaderc::Compiler::new() {
        Ok(compiler) => {
            println!("Compiling shaders...");
            compile_shader(&compiler, "src/shader/corr.comp", "corr.spv");
        }
        Err(_) => {
            println!("cargo:warning=shaderc not available, creating dummy shader file");
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let out_path = Path::new(&out_dir);
            fs::write(out_path.join("corr.spv"), b"").unwrap();
        }
    }
}
