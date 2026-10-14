use refusion_model::kernel::Shape;
use refusion_model::*;

fn revolve(direction: RevolveDirection, angle: &str, angle2: &str, full: bool) -> Document {
    let mut doc = Document::new();
    let mut sketch = sketch::Sketch::new();
    sketch.add_rectangle([10., 0.], [20., 5.]);
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch,
        projections: vec![],
    }));
    let profile = profile_ref(&doc.state().sketches[&id].profiles[0]);
    let feature = doc.add_feature(FeatureKind::Revolve(RevolveFeature {
        sketch: id,
        profiles: vec![profile],
        faces: vec![],
        axis: Axis::Y,
        direction,
        angle: angle.into(),
        angle2: angle2.into(),
        full,
        project_axis: true,
        op: Operation::NewBody,
    }));
    assert!(!doc.errors.contains_key(&feature), "{:?}", doc.errors);
    doc
}

#[test]
fn revolve_full_partial_symmetric_and_two_sides_have_cad_angle_semantics() {
    let quarter = std::f64::consts::PI * (400. - 100.) * 5. / 4.;
    for (direction, angle, second, full, expected) in [
        (RevolveDirection::OneSide, "90 deg", "0 deg", false, quarter),
        (
            RevolveDirection::Symmetric,
            "45 deg",
            "0 deg",
            false,
            quarter,
        ),
        (
            RevolveDirection::TwoSides,
            "30 deg",
            "60 deg",
            false,
            quarter,
        ),
        (
            RevolveDirection::OneSide,
            "unused",
            "unused",
            true,
            quarter * 4.,
        ),
    ] {
        let doc = revolve(direction, angle, second, full);
        let shape = &doc.state().bodies[0].shape;
        assert!((shape.volume() - expected).abs() < 1e-5);
        if direction == RevolveDirection::Symmetric {
            let (lo, hi) = shape.bbox().unwrap();
            assert!((lo[2] + hi[2]).abs() < 1e-6);
        }
        let replay = Document::from_json(&doc.to_json()).unwrap();
        assert!(replay.errors.is_empty());
        assert!((replay.state().bodies[0].shape.volume() - expected).abs() < 1e-5);
    }
}

#[test]
fn revolve_accepts_planar_faces_and_rejects_projected_zero_length_axis() {
    let mut doc = Document::new();
    doc.add_import(
        "source",
        &Shape::boxed(10., 5., 2.)
            .unwrap()
            .translated([10., 0., 0.], 1)
            .unwrap(),
    )
    .unwrap();
    let body = &doc.state().bodies[0];
    let top = (1..=body.shape.face_count() as u32)
        .find(|&f| {
            body.shape
                .face_info(f)
                .unwrap()
                .plane
                .is_some_and(|p| p.normal()[2] > 0.99)
        })
        .unwrap();
    let face = doc.state().face_ref(body.id, top).unwrap();
    let mut feature = RevolveFeature {
        sketch: 0,
        profiles: vec![],
        faces: vec![face],
        axis: Axis::Y,
        direction: RevolveDirection::OneSide,
        angle: "90 deg".into(),
        angle2: "0 deg".into(),
        full: false,
        project_axis: true,
        op: Operation::NewBody,
    };
    let id = doc.add_feature(FeatureKind::Revolve(feature.clone()));
    assert!(!doc.errors.contains_key(&id), "{:?}", doc.errors);
    let volume = std::f64::consts::PI * 300. * 5. / 4.;
    assert!((doc.state().bodies.last().unwrap().shape.volume() - volume).abs() < 1e-5);
    feature.axis = Axis::Z;
    let invalid = doc.add_feature(FeatureKind::Revolve(feature));
    assert!(doc.errors[&invalid].contains("zero length"));
}

#[test]
fn extrusion_half_and_whole_measurement_persist_independently() {
    for (extent, height) in [
        (Extent::Symmetric { d: "10 mm".into() }, 10.),
        (Extent::SymmetricHalf { d: "10 mm".into() }, 20.),
    ] {
        let mut doc = Document::new();
        let mut sketch = sketch::Sketch::new();
        sketch.add_rectangle([0., 0.], [10., 10.]);
        let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch,
            projections: vec![],
        }));
        let profile = profile_ref(&doc.state().sketches[&id].profiles[0]);
        doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch: id,
            profiles: vec![profile],
            faces: vec![],
            extent,
            op: Operation::NewBody,
        }));
        let replay = Document::from_json(&doc.to_json()).unwrap();
        assert!(replay.errors.is_empty());
        let body = &replay.state().bodies[0];
        assert!((body.shape.volume() - height * 100.).abs() < 1e-6);
        let (lo, hi) = body.shape.bbox().unwrap();
        assert!((lo[2] + height / 2.).abs() < 1e-5 && (hi[2] - height / 2.).abs() < 1e-5);
    }
}

#[test]
fn mirror_join_keeps_a_single_body_and_new_body_retains_copies() {
    for join in [false, true] {
        let mut doc = Document::new();
        doc.add_import("box", &Shape::boxed(10., 10., 10.).unwrap())
            .unwrap();
        let body = doc.state().bodies[0].id;
        let id = doc.add_feature(FeatureKind::Mirror(MirrorFeature {
            bodies: vec![body],
            plane: PlaneRef::YZ,
            join,
        }));
        assert!(!doc.errors.contains_key(&id), "{:?}", doc.errors);
        assert_eq!(doc.state().bodies.len(), if join { 1 } else { 2 });
        let volume: f64 = doc.state().bodies.iter().map(|b| b.shape.volume()).sum();
        assert!((volume - 2000.).abs() < 1e-6);
        let replay = Document::from_json(&doc.to_json()).unwrap();
        assert!(replay.errors.is_empty());
        assert_eq!(replay.state().bodies.len(), doc.state().bodies.len());
    }
}
