use refusion_model::{kernel::Shape, sketch::Sketch, *};
fn rectangular(features: Vec<FeatureId>, distance: &str) -> PatternFeature {
    PatternFeature {
        objects: PatternObjects::Features,
        features,
        bodies: vec![],
        kind: PatternKind::Rectangular {
            axis1: LinearRef::X,
            axis2: LinearRef::Y,
            count1: "3".into(),
            count2: "1".into(),
            distance1: distance.into(),
            distance2: "1 mm".into(),
            spacing: true,
        },
    }
}
fn stock() -> Document {
    let mut d = Document::new();
    d.add_import("stock", &Shape::boxed(40., 30., 10.).unwrap())
        .unwrap();
    d
}
fn hole(d: &mut Document, center: [f64; 2]) -> FeatureId {
    let mut f = HoleFeature::new(d.state().bodies[0].id, PlaneRef::XY);
    f.centers = vec![[format!("{} mm", center[0]), format!("{} mm", center[1])]];
    f.diameter = "4 mm".into();
    f.extent = HoleExtent::ThroughAll;
    f.tip_angle = None;
    f.flip = true;
    d.add_feature(FeatureKind::Hole(f))
}
fn healthy(d: &Document, expected: f64) {
    assert!(d.errors.is_empty(), "{:?}", d.errors);
    let volume: f64 = d
        .state()
        .bodies
        .iter()
        .map(|b| {
            assert!(b.shape.is_valid() && b.shape.is_solid());
            assert!(!b.shape.mesh(0.1, 0.2).unwrap().indices.is_empty());
            b.shape.volume()
        })
        .sum();
    assert!((volume - expected).abs() < 1e-4, "{volume} != {expected}");
}
#[test]
fn rectangular_holes_replay_parameter_edits_roundtrip_and_rollback() {
    let mut d = stock();
    d.params.push(Param {
        name: "pitch".into(),
        expr: "10 mm".into(),
        comment: "".into(),
    });
    let source = hole(&mut d, [5., 5.]);
    let pattern = d.add_feature(FeatureKind::Pattern(rectangular(vec![source], "pitch")));
    let expected = 12000. - 3. * std::f64::consts::PI * 4. * 10.;
    healthy(&d, expected);
    assert_eq!(d.state().bodies.len(), 1);
    healthy(&Document::from_json(&d.to_json()).unwrap(), expected);
    d.params[0].expr = "12 mm".into();
    d.recompute_all();
    healthy(&d, expected);
    if let FeatureKind::Hole(f) = &mut d.feature_mut(source).unwrap().kind {
        f.diameter = "6 mm".into();
    }
    d.changed(source);
    healthy(&d, 12000. - 3. * std::f64::consts::PI * 9. * 10.);
    d.set_marker(2);
    healthy(&d, 12000. - std::f64::consts::PI * 9. * 10.);
    d.set_marker(3);
    healthy(&d, 12000. - 3. * std::f64::consts::PI * 9. * 10.);
    d.feature_mut(source).unwrap().suppressed = true;
    d.changed(source);
    assert!(d.errors.contains_key(&pattern));
    assert_eq!(d.state().bodies[0].shape.volume(), 12000.);
}
#[test]
fn circular_holes_full_partial_and_invalid_instances_are_atomic() {
    for angle in ["360 deg", "180 deg"] {
        let mut d = Document::new();
        d.add_import(
            "stock",
            &Shape::boxed(40., 40., 10.)
                .unwrap()
                .translated([-20., -20., 0.], 0)
                .unwrap(),
        )
        .unwrap();
        let source = hole(&mut d, [10., 0.]);
        d.add_feature(FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Features,
            features: vec![source],
            bodies: vec![],
            kind: PatternKind::Circular {
                axis: LinearRef::Z,
                count: "4".into(),
                angle: angle.into(),
            },
        }));
        healthy(&d, 16000. - 4. * std::f64::consts::PI * 4. * 10.);
    }
    let mut d = stock();
    let source = hole(&mut d, [5., 5.]);
    let before = d.state().bodies[0].shape.uid();
    let pattern = d.add_feature(FeatureKind::Pattern(rectangular(vec![source], "25 mm")));
    assert!(d.errors[&pattern].contains("no effect"));
    assert_eq!(
        d.state().bodies[0].shape.uid(),
        before,
        "first successful copy must also roll back"
    );
    if let FeatureKind::Pattern(f) = &mut d.feature_mut(pattern).unwrap().kind {
        f.features = vec![999];
    }
    d.changed(pattern);
    assert!(d.errors[&pattern].contains("earlier"));
    assert_eq!(d.state().bodies[0].shape.uid(), before);
}
fn boss(d: &mut Document, op: Operation) -> FeatureId {
    let mut sketch = Sketch::new();
    sketch.add_rectangle([2., 2.], [4., 4.]);
    let id = d.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Fixed {
            origin: [0., 0., 10.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        },
        sketch,
        projections: vec![],
    }));
    let profiles = vec![profile_ref(&d.state().sketches[&id].profiles[0])];
    d.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: id,
        profiles,
        faces: vec![],
        extent: Extent::OneSide { d: "5 mm".into() },
        op,
    }))
}
#[test]
fn joined_and_new_body_features_keep_their_operations_and_named_faces() {
    for op in [Operation::Join, Operation::NewBody] {
        let mut d = stock();
        let source = boss(&mut d, op);
        d.add_feature(FeatureKind::Pattern(rectangular(vec![source], "10 mm")));
        healthy(&d, 12060.);
        assert_eq!(
            d.state().bodies.len(),
            if op == Operation::Join { 1 } else { 4 }
        );
        let tags: Vec<_> = d
            .state()
            .bodies
            .iter()
            .flat_map(|b| b.shape.face_tags())
            .collect();
        assert!(tags.iter().all(|t| *t != 0));
        assert_eq!(
            tags.len(),
            tags.iter().collect::<std::collections::HashSet<_>>().len()
        );
        healthy(&Document::from_json(&d.to_json()).unwrap(), 12060.);
    }
}
#[test]
fn multiple_features_are_ordered_and_invalid_sources_are_diagnosed() {
    let mut d = stock();
    let a = hole(&mut d, [5., 5.]);
    let b = hole(&mut d, [5., 20.]);
    let pattern = d.add_feature(FeatureKind::Pattern(rectangular(vec![b, a, a], "10 mm")));
    healthy(&d, 12000. - 6. * std::f64::consts::PI * 4. * 10.);
    let import = d.features[0].id;
    if let FeatureKind::Pattern(f) = &mut d.feature_mut(pattern).unwrap().kind {
        f.features = vec![import];
    }
    d.changed(pattern);
    assert!(d.errors[&pattern].contains("not supported"));
}
#[test]
fn legacy_body_patterns_default_to_bodies() {
    let mut d = stock();
    let mut pattern = rectangular(vec![], "50 mm");
    pattern.objects = PatternObjects::Bodies;
    pattern.bodies = vec![1];
    d.add_feature(FeatureKind::Pattern(pattern));
    let mut data: serde_json::Value = serde_json::from_str(&d.to_json()).unwrap();
    let p = data["features"][1]["kind"]["Pattern"]
        .as_object_mut()
        .unwrap();
    p.remove("objects");
    p.remove("features");
    let loaded = Document::from_json(&data.to_string()).unwrap();
    healthy(&loaded, 36000.);
}
#[test]
fn boss_with_fillet_or_chamfer_patterns_selected_features_in_timeline_order() {
    for fillet in [true, false] {
        let mut d = stock();
        let source = boss(&mut d, Operation::Join);
        let body = &d.state().bodies[0];
        let edge = (1..=body.shape.edge_count() as u32)
            .find(|&i| {
                let e = body.shape.edge_info(i).unwrap();
                (e.mid[0] - 2.).abs() < 1e-6
                    && (e.mid[1] - 2.).abs() < 1e-6
                    && (e.mid[2] - 12.5).abs() < 1e-6
            })
            .unwrap();
        let reference = d.state().edge_ref(body.id, edge).unwrap();
        let blend = d.add_feature(if fillet {
            FeatureKind::Fillet(FilletFeature {
                edges: vec![reference],
                radius: "0.5 mm".into(),
            })
        } else {
            FeatureKind::Chamfer(ChamferFeature {
                edges: vec![reference],
                kind: ChamferKind::Equal { d: "0.5 mm".into() },
            })
        });
        assert!(d.errors.is_empty(), "{:?}", d.errors);
        let effect = d.state().bodies[0].shape.volume() - 12000.;
        d.add_feature(FeatureKind::Pattern(rectangular(
            vec![blend, source],
            "10 mm",
        )));
        healthy(&d, 12000. + 3. * effect);
        healthy(
            &Document::from_json(&d.to_json()).unwrap(),
            12000. + 3. * effect,
        );
    }
}
#[test]
fn revolve_results_pattern_as_new_bodies() {
    let mut d = Document::new();
    let mut s = Sketch::new();
    s.add_rectangle([2., 0.], [4., 5.]);
    let sketch = d.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Fixed {
            origin: [0.; 3],
            x: [1., 0., 0.],
            y: [0., 0., 1.],
        },
        sketch: s,
        projections: vec![],
    }));
    let profiles = vec![profile_ref(&d.state().sketches[&sketch].profiles[0])];
    let source = d.add_feature(FeatureKind::Revolve(RevolveFeature {
        sketch,
        profiles,
        faces: vec![],
        axis: Axis::Z,
        angle: "360 deg".into(),
        angle2: "90 deg".into(),
        full: true,
        direction: RevolveDirection::OneSide,
        project_axis: false,
        op: Operation::NewBody,
    }));
    d.add_feature(FeatureKind::Pattern(rectangular(vec![source], "12 mm")));
    healthy(&d, 3. * std::f64::consts::PI * 12. * 5.);
    assert_eq!(d.state().bodies.len(), 3);
}
