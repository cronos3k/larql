fn main() {
    // Tell Cargo to rerun this script if the C source changes.
    println!("cargo:rerun-if-changed=csrc/q4_dot.c");

    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    // Use CARGO_CFG_TARGET_* env vars so this works correctly for cross-compilation.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env  = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    match target_arch.as_str() {
        "aarch64" => {
            // ARM: enable the dot-product extension (vdotq_s32 in q4_dot.c)
            build.flag("-march=armv8.2-a+dotprod");
        }
        "x86_64" => {
            // x86_64: enable AVX2 for potential future SIMD paths.
            // MSVC uses /arch:AVX2; GCC/Clang use -mavx2.
            if target_env == "msvc" {
                build.flag("/arch:AVX2");
            } else {
                build.flag("-mavx2");
            }
        }
        _ => {}
    }

    build.compile("q4_dot");
}
