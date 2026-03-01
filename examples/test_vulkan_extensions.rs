use ash::Entry;

fn main() {
    let entry = unsafe { Entry::load_from("/opt/homebrew/lib/libMoltenVK.dylib") }
        .expect("Failed to load MoltenVK");

    let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
    println!("Available Vulkan extensions:");
    for ext in &extensions {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        println!("  - {}", name.to_string_lossy());
    }

    // Check specifically for portability
    let has_portability = extensions.iter().any(|ext| {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        name.to_string_lossy().contains("PORTABILITY")
    });
    println!("\nHas portability extension: {}", has_portability);
}
