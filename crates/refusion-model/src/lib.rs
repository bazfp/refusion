pub mod doc;
pub mod expr;

pub use doc::*;
pub use expr::Unit;
pub use refusion_kernel as kernel;
pub use refusion_sketch as sketch;

pub fn f3d_import(path: &std::path::Path) -> Result<Vec<(String, kernel::Shape)>, String> {
    refusion_f3d::import(path).map(|r| r.parts)
}

pub fn add_f3d_sketches(
    doc: &mut Document,
    sketches: Vec<refusion_f3d::sketches::ImportedSketch>,
) -> Vec<FeatureId> {
    use refusion_f3d::sketches::Curve;
    let mut ids = Vec::new();
    for imported in sketches {
        let mut sketch = sketch::Sketch::new();
        let mut points: std::collections::HashMap<(i64, i64), sketch::EntityId> =
            std::collections::HashMap::new();
        let point =
            |sk: &mut sketch::Sketch, pts: &mut std::collections::HashMap<_, _>, p: [f64; 2]| {
                let key = ((p[0] * 1e7).round() as i64, (p[1] * 1e7).round() as i64);
                *pts.entry(key).or_insert_with(|| sk.add_point(p))
            };
        for p in imported.points {
            point(&mut sketch, &mut points, p);
        }
        for curve in imported.curves {
            match curve {
                Curve::Line(a, b) => {
                    let a = point(&mut sketch, &mut points, a);
                    let b = point(&mut sketch, &mut points, b);
                    sketch.add_line(a, b);
                }
                Curve::Circle { centre, radius } => {
                    let c = point(&mut sketch, &mut points, centre);
                    sketch.add_circle(c, radius);
                }
                Curve::Arc {
                    centre,
                    radius,
                    start,
                    sweep,
                } => {
                    let c = point(&mut sketch, &mut points, centre);
                    let a = point(
                        &mut sketch,
                        &mut points,
                        [
                            centre[0] + radius * start.cos(),
                            centre[1] + radius * start.sin(),
                        ],
                    );
                    let t = start + sweep;
                    let b = point(
                        &mut sketch,
                        &mut points,
                        [centre[0] + radius * t.cos(), centre[1] + radius * t.sin()],
                    );
                    sketch.add_arc(c, a, b);
                }
            }
        }
        let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::Fixed {
                origin: imported.frame.origin,
                x: imported.frame.x,
                y: imported.frame.y,
            },
            sketch,
            projections: vec![],
        }));
        let original = imported.name;
        let mut name = original.clone();
        let mut suffix = 2;
        while doc.features.iter().any(|f| f.id != id && f.name == name) {
            name = format!("{original} ({suffix})");
            suffix += 1;
        }
        if let Some(feature) = doc.feature_mut(id) {
            feature.name = name;
        }
        ids.push(id);
    }
    ids
}
pub fn f3d_design_import(path: &std::path::Path) -> Result<refusion_f3d::Report, String> {
    refusion_f3d::import(path)
}
