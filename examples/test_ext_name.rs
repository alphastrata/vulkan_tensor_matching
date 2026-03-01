fn main() {
    println!(
        "ash debug_utils NAME: {}",
        ash::ext::debug_utils::NAME.to_string_lossy()
    );
    println!("Expected: VK_EXT_debug_utils");

    // Check if they match
    let expected = std::ffi::CStr::from_bytes_with_nul(b"VK_EXT_debug_utils\0").unwrap();
    println!("Match: {}", ash::ext::debug_utils::NAME == expected);
}
