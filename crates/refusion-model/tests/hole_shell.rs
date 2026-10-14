use refusion_model::{kernel::Shape, *};
fn stock() -> Document {
    let mut d = Document::new();
    d.add_import("stock", &Shape::boxed(20., 20., 20.).unwrap())
        .unwrap();
    d
}
fn face(d: &Document, normal: [f64; 3]) -> FaceRef {
    let b = &d.state().bodies[0];
    let id = (1..=b.shape.face_count() as u32)
        .find(|&i| {
            b.shape
                .face_info(i)
                .and_then(|f| f.plane)
                .is_some_and(|p| kernel::dot(p.normal(), normal) > 0.99)
        })
        .unwrap();
    d.state().face_ref(b.id, id).unwrap()
}
fn hole(d: &Document) -> HoleFeature {
    let top = face(d, [0., 0., 1.]);
    let frame = d.state().plane_frame(&PlaneRef::Face(top.clone())).unwrap();
    let xy = frame.to_local([10., 10., 20.]);
    let mut f = HoleFeature::new(d.state().bodies[0].id, PlaneRef::Face(top));
    f.centers = vec![[format!("{} mm", xy[0]), format!("{} mm", xy[1])]];
    f.diameter = "4 mm".into();
    f.tip_angle = None;
    f
}
fn healthy(d: &Document) {
    assert!(d.errors.is_empty(), "{:?}", d.errors);
    for b in &d.state().bodies {
        assert!(b.shape.is_valid() && b.shape.is_solid());
        assert!(!b.shape.mesh(0.1, 0.2).unwrap().indices.is_empty());
    }
}
fn volume(d: &Document, expected: f64) {
    healthy(d);
    assert!(
        (d.state().bodies[0].shape.volume() - expected).abs() < 1e-4,
        "{} != {expected}",
        d.state().bodies[0].shape.volume()
    );
}
#[test]
fn hole_types_match_analytic_volumes_and_replay() {
    let pi = std::f64::consts::PI;
    for (kind, extra) in [
        (HoleKind::Simple, 0.),
        (
            HoleKind::Counterbore {
                diameter: "8 mm".into(),
                depth: "2 mm".into(),
            },
            pi * (16. - 4.) * 2.,
        ),
        (
            HoleKind::Countersink {
                diameter: "8 mm".into(),
                angle: "90 deg".into(),
            },
            pi * 2. / 3. * (16. + 8. + 4.) - pi * 4. * 2.,
        ),
    ] {
        let mut d = stock();
        let mut f = hole(&d);
        f.kind = kind;
        let id = d.add_feature(FeatureKind::Hole(f));
        let expected = 8000. - pi * 4. * 5. - extra;
        volume(&d, expected);
        volume(&Document::from_json(&d.to_json()).unwrap(), expected);
        d.set_marker(1);
        volume(&d, 8000.);
        d.set_marker(2);
        volume(&d, expected);
        d.feature_mut(id).unwrap().suppressed = true;
        d.changed(id);
        volume(&d, 8000.);
    }
}
#[test]
fn angled_tip_all_to_face_and_flipped_holes() {
    let pi = std::f64::consts::PI;
    let mut d = stock();
    let mut f = hole(&d);
    f.tip_angle = Some("90 deg".into());
    d.add_feature(FeatureKind::Hole(f));
    volume(&d, 8000. - pi * 4. * 5. - pi * 4. * 2. / 3.);
    for extent in [
        HoleExtent::ThroughAll,
        HoleExtent::ToFace {
            face: face(&stock(), [0., 0., -1.]),
            offset: "-2 mm".into(),
        },
    ] {
        // Build refs from this document (same stable import names, but do not rely on ID coincidence).
        let mut d = stock();
        let mut f = hole(&d);
        f.extent = if matches!(extent, HoleExtent::ThroughAll) {
            extent
        } else {
            HoleExtent::ToFace {
                face: face(&d, [0., 0., -1.]),
                offset: "-2 mm".into(),
            }
        };
        let depth = if matches!(f.extent, HoleExtent::ThroughAll) {
            20.
        } else {
            18.
        };
        d.add_feature(FeatureKind::Hole(f));
        volume(&d, 8000. - pi * 4. * depth);
    }
    let mut d = stock();
    let mut f = hole(&d);
    f.plane = PlaneRef::XY;
    f.centers = vec![["10 mm".into(), "10 mm".into()]];
    f.flip = true;
    d.add_feature(FeatureKind::Hole(f));
    volume(&d, 8000. - pi * 4. * 5.);
}
#[test]
fn hole_expressions_multiple_centers_and_invalid_later_center_are_atomic() {
    let mut d = stock();
    d.params.push(Param {
        name: "bore".into(),
        expr: "4 mm".into(),
        comment: "".into(),
    });
    let mut f = hole(&d);
    f.diameter = "bore".into();
    let frame = d.state().plane_frame(&f.plane).unwrap();
    let xy = frame.to_local([5., 5., 20.]);
    f.centers
        .push([format!("{} mm", xy[0]), format!("{} mm", xy[1])]);
    let id = d.add_feature(FeatureKind::Hole(f));
    volume(&d, 8000. - 2. * std::f64::consts::PI * 4. * 5.);
    d.params[0].expr = "6 mm".into();
    d.recompute_all();
    volume(&d, 8000. - 2. * std::f64::consts::PI * 9. * 5.);
    let FeatureKind::Hole(f) = &mut d.feature_mut(id).unwrap().kind else {
        panic!()
    };
    f.centers.push(["1000 mm".into(), "1000 mm".into()]);
    d.changed(id);
    assert!(d.errors.contains_key(&id));
    assert!((d.state().bodies[0].shape.volume() - 8000.).abs() < 1e-6);
}
#[test]
fn invalid_hole_dimensions_and_targets_preserve_stock() {
    for case in 0..7 {
        let mut d = stock();
        let mut f = hole(&d);
        match case {
            0 => f.diameter = "-1 mm".into(),
            1 => f.tip_angle = Some("180 deg".into()),
            2 => {
                f.kind = HoleKind::Counterbore {
                    diameter: "2 mm".into(),
                    depth: "2 mm".into(),
                }
            }
            3 => {
                f.kind = HoleKind::Counterbore {
                    diameter: "8 mm".into(),
                    depth: "6 mm".into(),
                }
            }
            4 => {
                f.extent = HoleExtent::ToFace {
                    face: face(&d, [1., 0., 0.]),
                    offset: "0 mm".into(),
                }
            }
            5 => f.centers.clear(),
            _ => f.flip = true,
        };
        let id = d.add_feature(FeatureKind::Hole(f));
        assert!(d.errors.contains_key(&id), "case {case}");
        assert!((d.state().bodies[0].shape.volume() - 8000.).abs() < 1e-6);
    }
}
#[test]
fn both_shell_open_closed_independent_offsets_and_backward_compatibility() {
    for open in [false, true] {
        let mut d = stock();
        let body = d.state().bodies[0].id;
        let faces = if open {
            vec![face(&d, [0., 0., 1.])]
        } else {
            vec![]
        };
        let id = d.add_feature(FeatureKind::Shell(ShellFeature {
            body,
            body_tag: None,
            faces,
            thickness: "1 mm".into(),
            outward: false,
            outside_thickness: Some("2 mm".into()),
        }));
        let expected = if open {
            24. * 24. * 22. - 18. * 18. * 19.
        } else {
            24f64.powi(3) - 18f64.powi(3)
        };
        volume(&d, expected);
        volume(&Document::from_json(&d.to_json()).unwrap(), expected);
        let tags = d.state().bodies[0].shape.face_tags();
        assert!(tags.iter().all(|t| *t != 0));
        assert_eq!(
            tags.len(),
            tags.iter().collect::<std::collections::HashSet<_>>().len()
        );
        let FeatureKind::Shell(f) = &mut d.feature_mut(id).unwrap().kind else {
            panic!()
        };
        f.outside_thickness = Some("-1 mm".into());
        d.changed(id);
        assert!(d.errors.contains_key(&id));
        assert!((d.state().bodies[0].shape.volume() - 8000.).abs() < 1e-6);
    }
    let mut d = stock();
    let body = d.state().bodies[0].id;
    d.add_feature(FeatureKind::Shell(ShellFeature {
        body,
        body_tag: None,
        faces: vec![],
        thickness: "1 mm".into(),
        outward: false,
        outside_thickness: None,
    }));
    let mut json: serde_json::Value = serde_json::from_str(&d.to_json()).unwrap();
    json["features"][1]["kind"]["Shell"]
        .as_object_mut()
        .unwrap()
        .remove("outside_thickness");
    volume(
        &Document::from_json(&json.to_string()).unwrap(),
        8000. - 18f64.powi(3),
    );
}
#[test]
fn feature_body_anchors_survive_upstream_pattern_count_changes() {
    for shell in [false, true] {
        let mut d = stock();
        let first = d.state().bodies[0].id;
        d.params.push(Param {
            name: "copies".into(),
            expr: "2".into(),
            comment: "".into(),
        });
        d.add_feature(FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![first],
            kind: PatternKind::Rectangular {
                axis1: LinearRef::X,
                axis2: LinearRef::Y,
                count1: "copies".into(),
                count2: "1".into(),
                distance1: "40 mm".into(),
                distance2: "0 mm".into(),
                spacing: true,
            },
        }));
        d.add_import(
            "target",
            &Shape::boxed(20., 20., 20.)
                .unwrap()
                .translated([100., 0., 0.], 123)
                .unwrap(),
        )
        .unwrap();
        let b = d.state().bodies.last().unwrap();
        let body = b.id;
        let tag = b.shape.face_tags()[0];
        if shell {
            d.add_feature(FeatureKind::Shell(ShellFeature {
                body,
                body_tag: Some(tag),
                faces: vec![],
                thickness: "1 mm".into(),
                outward: false,
                outside_thickness: Some("2 mm".into()),
            }));
        } else {
            let mut f = HoleFeature::new(
                body,
                PlaneRef::Fixed {
                    origin: [100., 0., 20.],
                    x: [1., 0., 0.],
                    y: [0., 1., 0.],
                },
            );
            f.body_tag = Some(tag);
            f.centers = vec![["10 mm".into(), "10 mm".into()]];
            f.tip_angle = None;
            d.add_feature(FeatureKind::Hole(f));
        }
        healthy(&d);
        let expected = d.state().bodies.last().unwrap().shape.volume();
        d.params[0].expr = "4".into();
        d.recompute_all();
        healthy(&d);
        let target = d
            .state()
            .bodies
            .iter()
            .find(|b| b.name == "target")
            .unwrap();
        assert_ne!(target.id, body);
        assert!((target.shape.volume() - expected).abs() < 1e-5);
        for other in d.state().bodies.iter().filter(|b| b.id != target.id) {
            assert!((other.shape.volume() - 8000.).abs() < 1e-6);
        }
    }
}
#[test]
fn both_shell_multiple_openings_parameter_replay_and_excessive_offsets() {
    let mut d = stock();
    let body = d.state().bodies[0].id;
    for (name, expr) in [("inner", "1 mm"), ("outer", "2 mm")] {
        d.params.push(Param {
            name: name.into(),
            expr: expr.into(),
            comment: "".into(),
        });
    }
    let faces = vec![face(&d, [0., 0., 1.]), face(&d, [0., 0., -1.])];
    let id = d.add_feature(FeatureKind::Shell(ShellFeature {
        body,
        body_tag: None,
        faces,
        thickness: "inner".into(),
        outward: false,
        outside_thickness: Some("outer".into()),
    }));
    volume(&d, (24. * 24. - 18. * 18.) * 20.);
    d.params[0].expr = "2 mm".into();
    d.params[1].expr = "3 mm".into();
    d.recompute_all();
    volume(&d, (26. * 26. - 16. * 16.) * 20.);
    d.params[0].expr = "11 mm".into();
    d.recompute_all();
    assert!(d.errors.contains_key(&id));
    assert!((d.state().bodies[0].shape.volume() - 8000.).abs() < 1e-6);
}
