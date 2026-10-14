use refusion_f3d::sketches::{Curve, ImportedSketch};
use refusion_model::*;
#[test]
fn imported_sketch_plane_profiles_extrusion_and_save_reload() {
    let mut doc = Document::new();
    let ids = add_f3d_sketches(
        &mut doc,
        vec![ImportedSketch {
            name: "Imported rings".into(),
            frame: kernel::Frame {
                origin: [3., 5., 7.],
                x: [1., 0., 0.],
                y: [0., 0., -1.],
            },
            curves: vec![
                Curve::Circle {
                    centre: [0., 0.],
                    radius: 5.,
                },
                Curve::Circle {
                    centre: [0., 0.],
                    radius: 2.,
                },
            ],
            points: vec![[0., 0.]],
            unsupported: 0,
        }],
    );
    assert_eq!(ids.len(), 1);
    let id = ids[0];
    assert!(doc.errors.is_empty());
    assert_eq!(doc.feature(id).unwrap().name, "Imported rings");
    let sf = match &doc.feature(id).unwrap().kind {
        FeatureKind::Sketch(sf) => sf,
        _ => panic!(),
    };
    assert_eq!(
        sf.sketch
            .entities
            .values()
            .filter(|e| matches!(e.geom, sketch::Geom::Point(_)))
            .count(),
        1
    );
    let state = &doc.state().sketches[&id];
    assert_eq!(state.frame.origin, [3., 5., 7.]);
    let profile = state.profiles.iter().find(|p| !p.holes.is_empty()).unwrap();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: id,
        profiles: vec![profile_ref(profile)],
        faces: vec![],
        extent: Extent::OneSide { d: "3 mm".into() },
        op: Operation::NewBody,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let shape = &doc.state().bodies[0].shape;
    assert!(shape.is_valid());
    assert!((shape.volume() - std::f64::consts::PI * 21. * 3.).abs() < 1e-5);
    let restored = Document::from_json(&doc.to_json()).unwrap();
    assert!(restored.errors.is_empty());
    assert!((restored.state().bodies[0].shape.volume() - shape.volume()).abs() < 1e-6);
    assert_eq!(restored.state().sketches[&id].frame.origin, [3., 5., 7.]);
}
#[test]
fn invalid_imported_plane_is_reported_without_building_geometry() {
    let mut doc = Document::new();
    let mut sketch = sketch::Sketch::new();
    sketch.add_rectangle([0., 0.], [2., 2.]);
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Fixed {
            origin: [0.; 3],
            x: [0.; 3],
            y: [0., 1., 0.],
        },
        sketch,
        projections: vec![],
    }));
    assert!(doc.errors[&id].contains("orthonormal"));
    assert!(!doc.state().sketches.contains_key(&id));
}

#[test]
fn repeated_imports_keep_distinct_sketch_names() {
    let mut doc = Document::new();
    for _ in 0..3 {
        add_f3d_sketches(
            &mut doc,
            vec![ImportedSketch {
                name: "F3D Sketch 1".into(),
                frame: kernel::Frame {
                    origin: [0.; 3],
                    x: [1., 0., 0.],
                    y: [0., 1., 0.],
                },
                curves: vec![Curve::Circle {
                    centre: [0., 0.],
                    radius: 5.,
                }],
                points: vec![],
                unsupported: 0,
            }],
        );
    }
    assert!(doc.errors.is_empty());
    let names: Vec<_> = doc.features.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["F3D Sketch 1", "F3D Sketch 1 (2)", "F3D Sketch 1 (3)"]
    );
}
