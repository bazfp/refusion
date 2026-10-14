//! Compile the C++ shim and link the static OpenCASCADE toolkits.
//! OCCT location: $OCCT_DIR, default `<workspace>/third_party/occt`.
use std::path::PathBuf;

// Dependents before dependencies (topological order from OCCT's EXTERNLIB files).
const TOOLKITS: &[&str] = &[
    "TKDEIGES",
    "TKFeat",
    "TKOffset",
    "TKFillet",
    "TKBool",
    "TKDESTL",
    "TKDESTEP",
    "TKXCAF",
    "TKVCAF",
    "TKV3d",
    "TKHLR",
    "TKMesh",
    "TKService",
    "TKCAF",
    "TKBO",
    "TKPrim",
    "TKLCAF",
    "TKCDF",
    "TKXSBase",
    "TKShHealing",
    "TKTopAlgo",
    "TKGeomAlgo",
    "TKBRep",
    "TKGeomBase",
    "TKG3d",
    "TKG2d",
    "TKMath",
    "TKDE",
    "TKernel",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let occt = std::env::var("OCCT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../third_party/occt"));
    let inc = occt.join("include/opencascade");
    let lib = occt.join("lib");
    assert!(
        inc.exists(),
        "OpenCASCADE headers not found at {} (set OCCT_DIR)",
        inc.display()
    );

    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-env-changed=OCCT_DIR");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .include(&inc)
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter")
        .opt_level(2)
        .compile("refusion_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    for tk in TOOLKITS {
        println!("cargo:rustc-link-lib=static:-bundle={tk}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");
}
