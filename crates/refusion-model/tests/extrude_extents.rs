use refusion_model::*;
fn profile(doc: &mut Document, size: f64) -> (FeatureId, ProfileRef) {
    let mut sketch = sketch::Sketch::new();
    sketch.add_rectangle([-size / 2., -size / 2.], [size / 2., size / 2.]);
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch,
        projections: vec![],
    }));
    (id, profile_ref(&doc.state().sketches[&id].profiles[0]))
}
fn extrude(doc: &mut Document, sk: FeatureId, p: ProfileRef, extent: Extent) -> FeatureId {
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![p],
        faces: vec![],
        extent,
        op: Operation::NewBody,
    }))
}
#[test]
fn tapered_rectangle_signed_and_symmetric_volumes_and_replay() {
    for (distance, angle) in [(10f64, 10f64), (-10., 10.), (10., -10.)] {
        let mut doc = Document::new();
        let (sk, p) = profile(&mut doc, 20.);
        let id = extrude(
            &mut doc,
            sk,
            p,
            Extent::Tapered {
                extent: Box::new(Extent::OneSide {
                    d: distance.to_string(),
                }),
                angle: format!("{angle} deg"),
            },
        );
        assert!(doc.errors.is_empty(), "{:?}", doc.errors);
        let a = 20.;
        let b = a - 2. * distance.abs() * angle.to_radians().tan();
        let expected = distance.abs() / 3. * (a * a + a * b + b * b);
        assert!((doc.state().bodies[0].shape.volume() - expected).abs() < 1e-5);
        assert!(doc.state().bodies[0].shape.is_valid());
        let tags = doc.state().bodies[0].shape.face_tags();
        assert!(tags.iter().all(|t| *t != 0));
        let reload = Document::from_json(&doc.to_json()).unwrap();
        assert!(reload.errors.is_empty());
        assert_eq!(tags, reload.state().bodies[0].shape.face_tags());
        assert!(doc.feature(id).is_some());
    }
    let mut doc = Document::new();
    let (sk, p) = profile(&mut doc, 20.);
    extrude(
        &mut doc,
        sk,
        p,
        Extent::Tapered {
            extent: Box::new(Extent::Symmetric { d: "20".into() }),
            angle: "10 deg".into(),
        },
    );
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let (lo, hi) = doc.state().bodies[0].shape.bbox().unwrap();
    assert!((lo[2] + 10.).abs() < 1e-5 && (hi[2] - 10.).abs() < 1e-5);
}
#[test]
fn to_face_tracks_changed_target_and_offset_after_save_reload() {
    let mut doc = Document::new();
    doc.params.push(Param {
        name: "height".into(),
        expr: "12 mm".into(),
        comment: String::new(),
    });
    let (sk, p) = profile(&mut doc, 40.);
    extrude(&mut doc, sk, p, Extent::OneSide { d: "height".into() });
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
    let target = doc.state().face_ref(body.id, top).unwrap();
    let (sk, p) = profile(&mut doc, 4.);
    extrude(
        &mut doc,
        sk,
        p,
        Extent::ToFace {
            face: target,
            offset: "2 mm".into(),
        },
    );
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().bodies[1].shape.volume() - 16. * 14.).abs() < 1e-5);
    doc.params[0].expr = "25 mm".into();
    doc.recompute_all();
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let reload = Document::from_json(&doc.to_json()).unwrap();
    assert!(reload.errors.is_empty());
    assert!((reload.state().bodies[1].shape.volume() - 16. * 27.).abs() < 1e-5);
}
#[test]
fn invalid_taper_preserves_previous_body() {
    let mut doc = Document::new();
    let shape = kernel::Shape::boxed(10., 10., 10.).unwrap();
    doc.add_import("Existing", &shape).unwrap();
    let before = doc.state().bodies[0].shape.to_brep().unwrap();
    let (sk, p) = profile(&mut doc, 2.);
    let id = extrude(
        &mut doc,
        sk,
        p,
        Extent::Tapered {
            extent: Box::new(Extent::OneSide { d: "20".into() }),
            angle: "45 deg".into(),
        },
    );
    assert!(doc.errors.contains_key(&id));
    assert_eq!(doc.state().bodies.len(), 1);
    assert_eq!(doc.state().bodies[0].shape.to_brep().unwrap(), before);
}
#[test]
fn to_angled_face_terminates_on_actual_plane_and_taper_handles_holes() {
    let mut doc = Document::new();
    let angle = 30f64.to_radians();
    let mut sketch = sketch::Sketch::new();
    sketch.add_rectangle([-20., -20.], [20., 20.]);
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Fixed {
            origin: [0., 0., 15.],
            x: [angle.cos(), 0., angle.sin()],
            y: [0., 1., 0.],
        },
        sketch,
        projections: vec![],
    }));
    let p = profile_ref(&doc.state().sketches[&id].profiles[0]);
    extrude(&mut doc, id, p, Extent::OneSide { d: "1".into() });
    let body = &doc.state().bodies[0];
    let top = (1..=body.shape.face_count() as u32)
        .find(|&f| {
            body.shape
                .face_info(f)
                .unwrap()
                .plane
                .is_some_and(|p| p.normal()[2] > 0.8)
        })
        .unwrap();
    let target = doc.state().face_ref(body.id, top).unwrap();
    let plane = body.shape.face_info(top).unwrap().plane.unwrap();
    let (sk, p) = profile(&mut doc, 4.);
    extrude(
        &mut doc,
        sk,
        p,
        Extent::ToFace {
            face: target,
            offset: "0".into(),
        },
    );
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let result = &doc.state().bodies[1].shape;
    assert!(result.is_valid());
    assert!((result.volume() - 16. * (15. + 1. / angle.cos())).abs() < 1e-5);
    let cap = (1..=result.face_count() as u32)
        .find(|&f| {
            result
                .face_info(f)
                .unwrap()
                .plane
                .is_some_and(|p| p.normal()[2] > 0.8)
        })
        .unwrap();
    let centre = result.face_info(cap).unwrap().centroid;
    let n = plane.normal();
    let dot: f64 = (0..3).map(|i| (centre[i] - plane.origin[i]) * n[i]).sum();
    assert!(dot.abs() < 1e-6);
    let mut doc = Document::new();
    let mut sketch = sketch::Sketch::new();
    let c = sketch.add_point([0., 0.]);
    sketch.add_circle(c, 5.);
    sketch.add_circle(c, 2.);
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch,
        projections: vec![],
    }));
    let p = profile_ref(
        doc.state().sketches[&sk]
            .profiles
            .iter()
            .find(|p| !p.holes.is_empty())
            .unwrap(),
    );
    extrude(
        &mut doc,
        sk,
        p,
        Extent::Tapered {
            extent: Box::new(Extent::OneSide { d: "3".into() }),
            angle: "10 deg".into(),
        },
    );
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!(
        (doc.state().bodies[0].shape.volume()
            - std::f64::consts::PI * 63. * (1. - 10f64.to_radians().tan()))
        .abs()
            < 1e-5
    );
}
#[test]
fn to_face_with_taper_and_missing_target_are_transactional() {
    let mut doc = Document::new();
    let (sk, p) = profile(&mut doc, 30.);
    extrude(&mut doc, sk, p, Extent::OneSide { d: "10".into() });
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
    let target = doc.state().face_ref(body.id, top).unwrap();
    let (sk, p) = profile(&mut doc, 10.);
    extrude(
        &mut doc,
        sk,
        p.clone(),
        Extent::Tapered {
            extent: Box::new(Extent::ToFace {
                face: target.clone(),
                offset: "0".into(),
            }),
            angle: "5 deg".into(),
        },
    );
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let a = 10.;
    let b = a - 20. * 5f64.to_radians().tan();
    let expected = 10. / 3. * (a * a + a * b + b * b);
    assert!((doc.state().bodies[1].shape.volume() - expected).abs() < 1e-5);
    let volumes: Vec<_> = doc
        .state()
        .bodies
        .iter()
        .map(|b| b.shape.volume())
        .collect();
    let mut missing = target;
    missing.body = 9999;
    missing.tag = 999999999;
    missing.hint = [1e6; 3];
    let id = extrude(
        &mut doc,
        sk,
        p,
        Extent::ToFace {
            face: missing,
            offset: "0".into(),
        },
    );
    assert!(doc.errors[&id].contains("no longer exists"));
    assert_eq!(
        volumes,
        doc.state()
            .bodies
            .iter()
            .map(|b| b.shape.volume())
            .collect::<Vec<_>>()
    );
}
