use refusion_model::sketch::Sketch;
use refusion_model::*;

fn offset(doc: &mut Document, base: PlaneRef, distance: &str) -> FeatureId {
    doc.add_feature(FeatureKind::OffsetPlane(OffsetPlaneFeature {
        base,
        distance: distance.into(),
    }))
}

#[test]
fn signed_offset_follows_each_origin_plane_normal() {
    for (base, normal) in [
        (PlaneRef::XY, [0.0, 0.0, 1.0]),
        (PlaneRef::XZ, [0.0, 1.0, 0.0]),
        (PlaneRef::YZ, [1.0, 0.0, 0.0]),
    ] {
        for distance in [-12.0, 0.0, 12.0] {
            let mut doc = Document::new();
            let id = offset(&mut doc, base.clone(), &distance.to_string());
            assert!(doc.errors.is_empty(), "{:?}", doc.errors);
            let frame = doc.state().planes[&id];
            assert_eq!(frame.normal(), normal);
            assert_eq!(frame.origin, normal.map(|v| v * distance));
        }
    }
}

#[test]
fn nested_parametric_planes_drive_sketch_and_solid_after_save_and_replay() {
    let mut doc = Document::new();
    doc.params.push(Param {
        name: "height".into(),
        expr: "10 mm".into(),
        comment: String::new(),
    });
    let first = offset(&mut doc, PlaneRef::XZ, "height");
    let second = offset(&mut doc, PlaneRef::Construction(first), "-2 mm");
    let mut sketch = Sketch::new();
    sketch.add_rectangle([0.0, 0.0], [20.0, 10.0]);
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Construction(second),
        sketch,
        projections: Vec::new(),
    }));
    let profile = profile_ref(&doc.state().sketches[&sk].profiles[0]);
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![profile],
        faces: Vec::new(),
        extent: Extent::OneSide { d: "5 mm".into() },
        op: Operation::NewBody,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert_eq!(doc.state().sketches[&sk].frame.origin, [0.0, 8.0, 0.0]);
    assert!((doc.state().bodies[0].shape.volume() - 1000.0).abs() < 1e-6);
    doc.params[0].expr = "25 mm".into();
    doc.recompute_all();
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let loaded = Document::from_json(&doc.to_json()).unwrap();
    assert_eq!(loaded.state().sketches[&sk].frame.origin, [0.0, 23.0, 0.0]);
    let mesh = loaded.state().bodies[0].shape.mesh(0.1, 0.2).unwrap();
    let low = mesh
        .positions
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min);
    let high = mesh
        .positions
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!((low - 23.0).abs() < 1e-5 && (high - 28.0).abs() < 1e-5);
    doc.set_marker(0);
    assert!(doc.state().planes.is_empty());
    doc.set_marker(doc.features.len());
    assert!(doc.errors.is_empty());
    assert_eq!(doc.state().planes[&second].origin, [0.0, 23.0, 0.0]);
    doc.feature_mut(first).unwrap().suppressed = true;
    doc.changed(first);
    assert!(!doc.state().planes.contains_key(&first));
    assert!(doc.errors.contains_key(&second));
    assert!(doc.errors.contains_key(&sk));
    doc.feature_mut(first).unwrap().suppressed = false;
    doc.changed(first);
    assert!(doc.errors.is_empty());
    doc.remove_feature(first);
    assert!(doc.errors.contains_key(&second));
}

#[test]
fn invalid_offset_does_not_register_a_plane() {
    let mut doc = Document::new();
    let missing = offset(&mut doc, PlaneRef::Construction(1000), "5");
    let invalid = offset(&mut doc, PlaneRef::XY, "unknown_parameter");
    assert!(doc.errors.contains_key(&missing));
    assert!(doc.errors.contains_key(&invalid));
    assert!(doc.state().planes.is_empty());
}

#[test]
fn offset_from_body_face_tracks_the_face_reference() {
    let mut doc = Document::new();
    let mut sketch = Sketch::new();
    sketch.add_rectangle([0.0, 0.0], [20.0, 10.0]);
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch,
        projections: vec![],
    }));
    let profile = profile_ref(&doc.state().sketches[&sk].profiles[0]);
    let extrusion = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![profile],
        faces: vec![],
        extent: Extent::OneSide { d: "5".into() },
        op: Operation::NewBody,
    }));
    let body = &doc.state().bodies[0];
    let face = (1..=body.shape.face_count() as u32)
        .find(|&f| {
            body.shape
                .face_info(f)
                .and_then(|i| i.plane)
                .is_some_and(|p| (p.origin[1] - 5.0).abs() < 1e-6 && p.normal()[1].abs() > 0.9)
        })
        .unwrap();
    let base = PlaneRef::Face(doc.state().face_ref(body.id, face).unwrap());
    let original = doc.state().plane_frame(&base).unwrap();
    let plane = offset(&mut doc, base, "3");
    let expected = original.origin[1] + original.normal()[1] * 3.0;
    assert!((doc.state().planes[&plane].origin[1] - expected).abs() < 1e-6);
    if let FeatureKind::Extrude(e) = &mut doc.feature_mut(extrusion).unwrap().kind {
        e.extent = Extent::OneSide { d: "8".into() };
    }
    doc.changed(extrusion);
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().planes[&plane].origin[1] - expected - 3.0).abs() < 1e-6);
}
