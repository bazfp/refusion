//! Real-file check. The sample designs are user files and are not committed:
//! point REFUSION_F3D_SAMPLES at .f3d files or directories (':'-separated).
//!     REFUSION_F3D_SAMPLES=~/designs cargo test -p refusion-f3d --release -- --nocapture

fn collect(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if p.is_dir() {
        for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
            collect(&e.path(), out);
        }
    } else if p.extension().is_some_and(|e| e == "f3d") {
        out.push(p.to_path_buf());
    }
}

#[test]
fn sample_designs_import_as_valid_solids() {
    let Ok(spec) = std::env::var("REFUSION_F3D_SAMPLES") else {
        eprintln!("REFUSION_F3D_SAMPLES not set; skipping");
        return;
    };
    let mut files = Vec::new();
    for s in spec.split(':') {
        collect(std::path::Path::new(s), &mut files);
    }
    assert!(!files.is_empty(), "no .f3d files found in {spec}");
    for f in files {
        let r = refusion_f3d::import(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(!r.parts.is_empty(), "{}: no bodies", f.display());
        for (name, s) in &r.parts {
            assert!(s.is_valid(), "{}: {name} invalid", f.display());
        }
        eprintln!(
            "{}: {} part(s), {} faces",
            f.display(),
            r.parts.len(),
            r.stats.faces
        );
    }
}
