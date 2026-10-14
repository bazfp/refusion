//! Diagnose .f3d import:  cargo run -p refusion-f3d --example f3dinfo -- a.f3d [b.f3d …]
fn main() {
    for p in std::env::args().skip(1) {
        let t = std::time::Instant::now();
        match refusion_f3d::import(std::path::Path::new(&p)) {
            Ok(r) => {
                let s = &r.stats;
                for warning in &r.warnings {
                    println!("  WARNING: {warning}");
                }
                println!(
                    "{p}\n  blobs {} bodies {} faces {} (failed {}, filled {}) approx-edges {} sheets-skipped {} duplicates {} — {:.0} ms",
                    r.blobs,
                    s.bodies,
                    s.faces,
                    s.failed_faces,
                    s.filled_faces,
                    s.approx_edges,
                    r.sheets_skipped,
                    r.duplicates,
                    t.elapsed().as_secs_f64() * 1e3
                );
                for sketch in &r.sketches {
                    println!(
                        "  {}: {} curves, {} points, {} unsupported entities, plane {:?}",
                        sketch.name,
                        sketch.curves.len(),
                        sketch.points.len(),
                        sketch.unsupported,
                        sketch.frame
                    );
                }
                for (name, shape) in &r.parts {
                    let (lo, hi) = shape.bbox().unwrap_or(([0.0; 3], [0.0; 3]));
                    println!(
                        "  {name}: faces {} volume {:.2} mm³ valid {} size {:.1}×{:.1}×{:.1} mm",
                        shape.face_count(),
                        shape.volume(),
                        shape.is_valid(),
                        hi[0] - lo[0],
                        hi[1] - lo[1],
                        hi[2] - lo[2]
                    );
                }
            }
            Err(e) => println!("{p}\n  ERROR {e}"),
        }
    }
}
