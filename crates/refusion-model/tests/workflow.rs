//! End-to-end parametric workflow.
use refusion_model::sketch::{Constraint, Sketch};
use refusion_model::*;
use std::f64::consts::PI;

fn rect_sketch(w_expr: &str, h: f64) -> Sketch {
    let mut s = Sketch::new();
    let l = s.add_rectangle([0.0, 0.0], [30.0, 15.0]);
    let (a, _) = s.line_pts(l[0]).unwrap();
    s.add_constraint(Constraint::Fix(a)).unwrap();
    let w = s
        .add_constraint(Constraint::Length {
            line: l[0],
            value: 30.0,
        })
        .unwrap()
        .unwrap();
    s.constraints.get_mut(&w).unwrap().expr = Some(w_expr.into());
    s.add_constraint(Constraint::Length {
        line: l[1],
        value: h,
    })
    .unwrap();
    s
}

fn top_edges(doc: &Document, y: f64) -> Vec<EdgeRef> {
    let st = doc.state();
    let b = &st.bodies[0];
    (1..=b.shape.edge_count() as u32)
        .filter(|&e| {
            let i = b.shape.edge_info(e).unwrap();
            (i.start[1] - y).abs() < 1e-6 && (i.end[1] - y).abs() < 1e-6
        })
        .map(|e| st.edge_ref(b.id, e).unwrap())
        .collect()
}

#[test]
fn sketch_extrude_fillet_hole_then_change_parameter() {
    let mut doc = Document::new();
    doc.params.push(Param {
        name: "width".into(),
        expr: "40 mm".into(),
        comment: String::new(),
    });

    // Sketch on the ground plane (XZ), extrude 10 mm up (+Y).
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch: rect_sketch("width", 20.0),
        projections: vec![],
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let prof = doc.state().sketches[&sk].profiles[0].clone();
    assert!((prof.area() - 800.0).abs() < 1e-6, "{}", prof.area());
    let ex = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![profile_ref(&prof)],
        faces: vec![],
        extent: Extent::OneSide { d: "10".into() },
        op: Operation::NewBody,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().bodies[0].shape.volume() - 8000.0).abs() < 1e-6);

    // Fillet the four top edges, r = 2.
    let edges = top_edges(&doc, 10.0);
    assert_eq!(edges.len(), 4);
    doc.add_feature(FeatureKind::Fillet(FilletFeature {
        edges,
        radius: "2 mm".into(),
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let filleted = doc.state().bodies[0].shape.volume();
    let perimeter = 2.0 * (40.0 + 20.0);
    let expect = 8000.0 - perimeter * 4.0 * (1.0 - PI / 4.0);
    // corners make it slightly different from the straight-run estimate
    assert!((filleted - expect).abs() < 5.0, "{filleted} vs {expect}");

    // Sketch on the bottom face, a Ø6 circle, cut through-all upward.
    let st = doc.state();
    let b = &st.bodies[0];
    let bottom = (1..=b.shape.face_count() as u32)
        .find(|&f| {
            b.shape
                .face_info(f)
                .and_then(|i| i.plane)
                .is_some_and(|p| p.normal()[1] < -0.999)
        })
        .unwrap();
    let face_ref = st.face_ref(b.id, bottom).unwrap();
    let mut s2 = Sketch::new();
    let frame = b.shape.face_info(bottom).unwrap().plane.unwrap();
    let c = s2.add_point(frame.to_local([10.0, 0.0, -10.0]));
    let circ = s2.add_circle(c, 3.0);
    s2.add_constraint(Constraint::Diameter {
        e: circ,
        value: 6.0,
    })
    .unwrap();
    let sk2 = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Face(face_ref),
        sketch: s2,
        projections: vec![],
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let disk = doc.state().sketches[&sk2]
        .profiles
        .iter()
        .find(|p| p.holes.is_empty())
        .unwrap()
        .clone();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk2,
        profiles: vec![profile_ref(&disk)],
        faces: vec![],
        extent: Extent::OneSide { d: "-50".into() }, // into the body (face normal points down)
        op: Operation::Cut,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let with_hole = doc.state().bodies[0].shape.volume();
    assert!(
        (filleted - with_hole - PI * 9.0 * 10.0).abs() < 1e-3,
        "{}",
        filleted - with_hole
    );

    // Change the parameter: everything downstream replays, fillet included.
    doc.params[0].expr = "60 mm".into();
    doc.recompute_all();
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let v = doc.state().bodies[0].shape.volume();
    let perimeter = 2.0 * (60.0 + 20.0);
    let expect = 60.0 * 20.0 * 10.0 - perimeter * 4.0 * (1.0 - PI / 4.0) - PI * 9.0 * 10.0;
    assert!((v - expect).abs() < 5.0, "{v} vs {expect}");
    // Still exactly one fillet per top edge: 6 original faces + 4 fillet faces + corner patches + hole wall
    assert!(doc.state().bodies[0].shape.face_count() >= 11);

    // Edit the extrude height via the feature definition.
    if let Some(Feature {
        kind: FeatureKind::Extrude(x),
        ..
    }) = doc.feature_mut(ex)
    {
        x.extent = Extent::OneSide { d: "15".into() };
    }
    doc.changed(ex);
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let v15 = doc.state().bodies[0].shape.volume();
    assert!(
        (v15 - v - (60.0 * 20.0 - PI * 9.0) * 5.0).abs() < 1e-3,
        "{}",
        v15 - v
    );

    // Save / load round trip reproduces the model.
    let json = doc.to_json();
    let doc2 = Document::from_json(&json).unwrap();
    assert!(doc2.errors.is_empty(), "{:?}", doc2.errors);
    assert!((doc2.state().bodies[0].shape.volume() - v15).abs() < 1e-6);

    // Roll the timeline back to just the first extrude.
    doc.set_marker(2);
    assert!((doc.state().bodies[0].shape.volume() - 60.0 * 20.0 * 15.0).abs() < 1e-6);
}

#[test]
fn revolve_chamfer_and_projection() {
    let mut doc = Document::new();
    // Profile on XY: rectangle x∈[10,20], y∈[0,30]; axis = the Y axis line drawn in the sketch.
    let mut s = Sketch::new();
    s.add_rectangle([10.0, 0.0], [20.0, 30.0]);
    let axis = s.add_line_pts([0.0, -5.0], [0.0, 40.0]);
    s.entities.get_mut(&axis).unwrap().construction = true;
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch: s,
        projections: vec![],
    }));
    let prof = doc.state().sketches[&sk].profiles[0].clone();
    doc.add_feature(FeatureKind::Revolve(RevolveFeature {
        faces: vec![],
        direction: RevolveDirection::OneSide,
        angle2: "90 deg".into(),
        full: false,
        project_axis: false,
        sketch: sk,
        profiles: vec![profile_ref(&prof)],
        axis: Axis::SketchLine(axis),
        angle: "360".into(),
        op: Operation::NewBody,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let v = doc.state().bodies[0].shape.volume().abs();
    assert!((v - PI * (400.0 - 100.0) * 30.0).abs() < 1e-3, "{v}");

    // Chamfer the outer top circular edge, 2 mm.
    let st = doc.state();
    let b = &st.bodies[0];
    let e = (1..=b.shape.edge_count() as u32)
        .find(|&e| matches!(b.shape.edge_info(e).unwrap().kind, kernel::EdgeKind::Circle { radius, .. } if (radius - 20.0).abs() < 1e-6) && (b.shape.edge_info(e).unwrap().mid[1] - 30.0).abs() < 1e-6)
        .unwrap();
    let r = st.edge_ref(b.id, e).unwrap();
    doc.add_feature(FeatureKind::Chamfer(ChamferFeature {
        edges: vec![r.clone()],
        kind: ChamferKind::Equal { d: "2".into() },
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let vc = doc.state().bodies[0].shape.volume().abs();
    // ring of triangular section 2×2/2 at radius ≈ 20 - 2/3
    let removed = 2.0 * PI * (20.0 - 2.0 / 3.0) * 2.0;
    assert!((v - vc - removed).abs() < 1e-2, "{}", v - vc);

    // Project the (pre-chamfer) top outer circle into a sketch on XZ offset-free plane.
    let sk3 = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch: Sketch::new(),
        projections: vec![],
    }));
    let st = doc.state_before(sk3).clone();
    let b = &st.bodies[0];
    let bottom_outer = (1..=b.shape.edge_count() as u32)
        .find(|&e| matches!(b.shape.edge_info(e).unwrap().kind, kernel::EdgeKind::Circle { radius, .. } if (radius - 20.0).abs() < 1e-6) && b.shape.edge_info(e).unwrap().mid[1].abs() < 1e-6)
        .unwrap();
    let ents = doc
        .project_edge(sk3, st.edge_ref(b.id, bottom_outer).unwrap())
        .unwrap();
    doc.changed(sk3);
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let Some(Feature {
        kind: FeatureKind::Sketch(sf),
        ..
    }) = doc.feature(sk3)
    else {
        panic!()
    };
    let circle = *ents.last().unwrap();
    assert!((sf.sketch.radius(circle).unwrap() - 20.0).abs() < 1e-9);
    assert!(sf.sketch.entity(circle).unwrap().projected);
    // A projected circle is a closed profile on its own.
    assert_eq!(doc.state().sketches[&sk3].profiles.len(), 1);
}

#[test]
fn extrude_body_face_join_then_cut() {
    let mut doc = Document::new();
    let sk = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch: rect_sketch("40", 20.0),
        projections: vec![],
    }));
    let prof = doc.state().sketches[&sk].profiles[0].clone();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk,
        profiles: vec![profile_ref(&prof)],
        faces: vec![],
        extent: Extent::OneSide { d: "10".into() },
        op: Operation::NewBody,
    }));
    let top = |doc: &Document| {
        let st = doc.state();
        let b = &st.bodies[0];
        let f = (1..=b.shape.face_count() as u32)
            .find(|&f| {
                b.shape
                    .face_info(f)
                    .and_then(|i| i.plane)
                    .is_some_and(|p| p.normal()[1] > 0.999)
            })
            .unwrap();
        st.face_ref(b.id, f).unwrap()
    };
    // Pull the top face up 5 mm (join)…
    let up = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: 0,
        profiles: vec![],
        faces: vec![top(&doc)],
        extent: Extent::OneSide { d: "5".into() },
        op: Operation::Join,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().bodies[0].shape.volume() - 40.0 * 20.0 * 15.0).abs() < 1e-6);
    // …then cut 3 mm back into the new top face.
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: 0,
        profiles: vec![],
        faces: vec![top(&doc)],
        extent: Extent::OneSide { d: "-3".into() },
        op: Operation::Cut,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().bodies[0].shape.volume() - 40.0 * 20.0 * 12.0).abs() < 1e-6);
    // The face reference survives an upstream edit of the first pull.
    if let Some(Feature {
        kind: FeatureKind::Extrude(x),
        ..
    }) = doc.feature_mut(up)
    {
        x.extent = Extent::OneSide { d: "8".into() };
    }
    doc.changed(up);
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!((doc.state().bodies[0].shape.volume() - 40.0 * 20.0 * 15.0).abs() < 1e-6);
}

#[test]
fn import_step_then_model_on_it() {
    // Export a filleted body to STEP, import it as a base feature into a new
    // document, then extrude one of its faces — references must work.
    let dir = std::env::temp_dir().join(format!("refusion-import-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("part.step");
    let b = kernel::Shape::boxed(20.0, 10.0, 30.0).unwrap();
    let b = b.fillet(&[1], 2.0, 1).unwrap();
    let v0 = b.volume();
    kernel::write_step(&[b], &path).unwrap();

    let mut doc = Document::new();
    let shape = kernel::read_step(&path).unwrap();
    doc.add_import("part.step", &shape).unwrap();
    let snapshot = doc.clone();
    let (FeatureKind::Import(a), FeatureKind::Import(b)) =
        (&doc.features[0].kind, &snapshot.features[0].kind)
    else {
        panic!("expected import")
    };
    assert!(std::sync::Arc::ptr_eq(&a.brep, &b.brep));
    assert_eq!(doc.state().bodies.len(), 1);
    assert_eq!(doc.state().bodies[0].name, "part");
    assert!((doc.state().bodies[0].shape.volume() - v0).abs() < 1e-6);
    let st = doc.state();
    let body = &st.bodies[0];
    let top = (1..=body.shape.face_count() as u32)
        .find(|&f| {
            body.shape
                .face_info(f)
                .and_then(|i| i.plane)
                .is_some_and(|p| p.normal()[2] > 0.999)
        })
        .unwrap();
    let r = st.face_ref(body.id, top).unwrap();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: 0,
        profiles: vec![],
        faces: vec![r],
        extent: Extent::OneSide { d: "5".into() },
        op: Operation::Join,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert!(doc.state().bodies[0].shape.volume() > v0);
    // Embedded geometry survives save/load.
    let again = Document::from_json(&doc.to_json()).unwrap();
    assert!(again.errors.is_empty(), "{:?}", again.errors);
    assert!(
        (again.state().bodies[0].shape.volume() - doc.state().bodies[0].shape.volume()).abs()
            < 1e-6
    );
    std::fs::remove_dir_all(&dir).ok();
}
