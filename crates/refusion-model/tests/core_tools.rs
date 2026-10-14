use refusion_model::kernel::Shape;
use refusion_model::sketch::Sketch;
use refusion_model::*;
fn cube() -> Document {
    let mut d = Document::new();
    d.add_import("cube.step", &Shape::boxed(20.0, 20.0, 20.0).unwrap())
        .unwrap();
    d
}
fn good(d: &Document) {
    assert!(d.errors.is_empty(), "{:?}", d.errors);
    for b in &d.state().bodies {
        assert!(b.shape.volume() > 0.0);
        assert!(!b.shape.mesh(0.1, 0.2).unwrap().indices.is_empty());
    }
}
#[test]
fn shell_closed_open_invalid_and_persistent_faces() {
    for open in [false, true] {
        let mut d = cube();
        let body = d.state().bodies[0].id;
        let face = (1..=6)
            .find(|&i| d.state().bodies[0].shape.face_info(i).unwrap().centroid[2] > 19.0)
            .unwrap();
        let faces = if open {
            vec![d.state().face_ref(body, face).unwrap()]
        } else {
            vec![]
        };
        let id = d.add_feature(FeatureKind::Shell(ShellFeature {
            body,
            body_tag: None,
            faces,
            thickness: "1 mm".into(),
            outward: false,
            outside_thickness: None,
        }));
        good(&d);
        let expected = if open { 1844.0 } else { 2168.0 };
        let tags = d.state().bodies[0].shape.face_tags();
        assert!(tags.iter().all(|t| *t != 0));
        assert_eq!(
            tags.iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            tags.len(),
            "shell face names must be unique"
        );
        assert!(
            (d.state().bodies[0].shape.volume() - expected).abs() < 1e-5,
            "open={open}, volume={}",
            d.state().bodies[0].shape.volume()
        );
        let loaded = Document::from_json(&d.to_json()).unwrap();
        good(&loaded);
        assert!((loaded.state().bodies[0].shape.volume() - expected).abs() < 1e-5);
        if let FeatureKind::Shell(f) = &mut d.feature_mut(id).unwrap().kind {
            f.thickness = "-1 mm".into();
        }
        d.changed(id);
        assert!(d.errors.contains_key(&id));
        assert!((d.state().bodies[0].shape.volume() - 8000.0).abs() < 1e-6);
    }
}
#[test]
fn mirror_reflects_bounds_without_modifying_original_and_names_copies() {
    let mut d = cube();
    let body = d.state().bodies[0].id;
    d.add_feature(FeatureKind::Mirror(MirrorFeature {
        join: false,
        bodies: vec![body],
        plane: PlaneRef::YZ,
    }));
    good(&d);
    assert_eq!(d.state().bodies.len(), 2);
    let (lo, hi) = d.state().bodies[1].shape.bbox().unwrap();
    assert!((lo[0] + 20.0).abs() < 1e-5 && hi[0].abs() < 1e-5);
    let tags = d.state().bodies[0].shape.face_tags();
    assert!(d.state().bodies[1]
        .shape
        .face_tags()
        .iter()
        .all(|t| !tags.contains(t)));
    good(&Document::from_json(&d.to_json()).unwrap());
}
#[test]
fn rectangular_pattern_spacing_extent_and_transactional_limits() {
    for spacing in [false, true] {
        let mut d = cube();
        let body = d.state().bodies[0].id;
        let id = d.add_feature(FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![body],
            kind: PatternKind::Rectangular {
                axis1: LinearRef::X,
                axis2: LinearRef::Y,
                count1: "3".into(),
                count2: "2".into(),
                distance1: "60 mm".into(),
                distance2: "30 mm".into(),
                spacing,
            },
        }));
        good(&d);
        assert_eq!(d.state().bodies.len(), 6);
        let max = d
            .state()
            .bodies
            .iter()
            .map(|b| b.shape.bbox().unwrap().1[0])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max - if spacing { 140.0 } else { 80.0 }).abs() < 1e-5);
        good(&Document::from_json(&d.to_json()).unwrap());
        if let FeatureKind::Pattern(PatternFeature {
            kind: PatternKind::Rectangular { count1, .. },
            ..
        }) = &mut d.feature_mut(id).unwrap().kind
        {
            *count1 = "1001".into();
        }
        d.changed(id);
        assert!(d.errors.contains_key(&id));
        assert_eq!(d.state().bodies.len(), 1);
    }
}
#[test]
fn circular_pattern_full_and_partial_angles() {
    for angle in ["360 deg", "180 deg"] {
        let mut d = cube();
        let body = d.state().bodies[0].id;
        d.add_feature(FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![body],
            kind: PatternKind::Circular {
                axis: LinearRef::Z,
                count: "4".into(),
                angle: angle.into(),
            },
        }));
        good(&d);
        assert_eq!(d.state().bodies.len(), 4);
        let centers: Vec<_> = d
            .state()
            .bodies
            .iter()
            .map(|b| {
                let (lo, hi) = b.shape.bbox().unwrap();
                [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]
            })
            .collect();
        for i in 0..centers.len() {
            for j in i + 1..centers.len() {
                assert!((centers[i][0] - centers[j][0]).hypot(centers[i][1] - centers[j][1]) > 1.0);
            }
        }
        good(&Document::from_json(&d.to_json()).unwrap());
    }
}
#[test]
fn angled_plane_hosts_sketch_and_rejects_axis_outside_base() {
    let mut d = Document::new();
    let id = d.add_feature(FeatureKind::AngledPlane(AngledPlaneFeature {
        base: PlaneRef::XY,
        axis: LinearRef::X,
        angle: "90 deg".into(),
    }));
    good(&d);
    let n = d.state().planes[&id].normal();
    assert!((n[1] + 1.0).abs() < 1e-9 && n[2].abs() < 1e-9);
    let sk = d.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Construction(id),
        sketch: Sketch::new(),
        projections: vec![],
    }));
    good(&d);
    assert_eq!(d.state().sketches[&sk].frame, d.state().planes[&id]);
    good(&Document::from_json(&d.to_json()).unwrap());
    if let FeatureKind::AngledPlane(f) = &mut d.feature_mut(id).unwrap().kind {
        f.axis = LinearRef::Z;
    }
    d.changed(id);
    assert!(d.errors.contains_key(&id));
    assert!(!d.state().planes.contains_key(&id));
}
#[test]
fn through_all_cuts_in_every_direction_and_tracks_target_size() {
    for (direction, z) in [
        (AllDirection::Positive, 0.0),
        (AllDirection::Negative, 30.0),
        (AllDirection::Both, 10.0),
    ] {
        let mut d = cube();
        let plane = d.add_feature(FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::XY,
            distance: z.to_string(),
        }));
        let mut sketch = Sketch::new();
        sketch.add_rectangle([5.0, 5.0], [7.0, 7.0]);
        let sk = d.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::Construction(plane),
            sketch,
            projections: vec![],
        }));
        let profile = profile_ref(&d.state().sketches[&sk].profiles[0]);
        d.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch: sk,
            profiles: vec![profile],
            faces: vec![],
            extent: Extent::ThroughAll { direction },
            op: Operation::Cut,
        }));
        good(&d);
        assert!((d.state().bodies[0].shape.volume() - 7920.0).abs() < 1e-5);
        good(&Document::from_json(&d.to_json()).unwrap());
    }
}
#[test]
fn through_all_adapts_after_target_height_parameter_changes() {
    let mut d = Document::new();
    d.params.push(Param {
        name: "height".into(),
        expr: "20 mm".into(),
        comment: String::new(),
    });
    let mut s = Sketch::new();
    s.add_rectangle([0.0, 0.0], [20.0, 20.0]);
    let sk = d.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch: s,
        projections: vec![],
    }));
    let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
    d.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![p],
        faces: vec![],
        extent: Extent::OneSide { d: "height".into() },
        op: Operation::NewBody,
    }));
    let mut s = Sketch::new();
    s.add_rectangle([5.0, 5.0], [7.0, 7.0]);
    let sk = d.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch: s,
        projections: vec![],
    }));
    let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
    d.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![p],
        faces: vec![],
        extent: Extent::ThroughAll {
            direction: AllDirection::Positive,
        },
        op: Operation::Cut,
    }));
    for h in [20.0, 45.0, 120.0] {
        d.params[0].expr = format!("{h} mm");
        d.recompute_all();
        good(&d);
        assert!((d.state().bodies[0].shape.volume() - 396.0 * h).abs() < 1e-5);
    }
}
#[test]
fn outward_shell_has_correct_material_volume() {
    let mut d = cube();
    let body = d.state().bodies[0].id;
    d.add_feature(FeatureKind::Shell(ShellFeature {
        body,
        body_tag: None,
        faces: vec![],
        thickness: "1 mm".into(),
        outward: true,
        outside_thickness: None,
    }));
    good(&d);
    assert!((d.state().bodies[0].shape.volume() - (22.0f64.powi(3) - 8000.0)).abs() < 1e-5);
}

#[test]
fn excessive_shell_thickness_is_rejected_without_replacing_body() {
    let mut d = cube();
    let body = d.state().bodies[0].id;
    let id = d.add_feature(FeatureKind::Shell(ShellFeature {
        body,
        body_tag: None,
        faces: vec![],
        thickness: "11 mm".into(),
        outward: false,
        outside_thickness: None,
    }));
    assert!(d.errors.contains_key(&id));
    assert!((d.state().bodies[0].shape.volume() - 8000.0).abs() < 1e-6);
}
