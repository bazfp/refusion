//! Build a demo part exercising the original core workflow and save it as a .rfz file:
//! parameters → sketch with constraints & dimensions → extrude → fillet →
//! sketch on a face with projection → cut → revolve → chamfer.
//!
//!     cargo run -p refusion-model --example demo -- examples/bracket.rfz

use refusion_model::sketch::{Constraint, DimKind, Sketch};
use refusion_model::*;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bracket.rfz".into());
    let mut doc = Document::new();
    for (n, e) in [
        ("width", "80 mm"),
        ("depth", "50 mm"),
        ("thick", "8 mm"),
        ("hole", "width / 8"),
    ] {
        doc.params.push(Param {
            name: n.into(),
            expr: e.into(),
            comment: String::new(),
        });
    }

    // 1. Base plate on the ground plane, centred on the origin via symmetry.
    let mut s = Sketch::new();
    let origin = s.add_point([0.0, 0.0]);
    s.entities.get_mut(&origin).unwrap().projected = true;
    let l = s.add_rectangle([-40.0, -25.0], [40.0, 25.0]);
    let (c0, _) = s.line_pts(l[0]).unwrap();
    let (c2, _) = s.line_pts(l[2]).unwrap();
    let diag = s.add_line(c0, c2);
    s.entities.get_mut(&diag).unwrap().construction = true;
    s.add_constraint(Constraint::Midpoint(origin, diag))
        .unwrap();
    let w = s
        .add_constraint(Constraint::Length {
            line: l[0],
            value: 80.0,
        })
        .unwrap()
        .unwrap();
    s.constraints.get_mut(&w).unwrap().expr = Some("width".into());
    let d = s
        .add_constraint(Constraint::Length {
            line: l[1],
            value: 50.0,
        })
        .unwrap()
        .unwrap();
    s.constraints.get_mut(&d).unwrap().expr = Some("depth".into());
    s.constraints.get_mut(&w).unwrap().label = [0.0, -8.0];
    s.constraints.get_mut(&d).unwrap().label = [8.0, 0.0];
    assert_eq!(
        s.analyze_now().dof,
        0,
        "base sketch should be fully constrained"
    );
    let sk1 = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch: s,
        projections: vec![],
    }));
    let prof = doc.state().sketches[&sk1].profiles[0].clone();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk1,
        profiles: vec![profile_ref(&prof)],
        faces: vec![],
        extent: Extent::OneSide { d: "thick".into() },
        op: Operation::NewBody,
    }));

    // 2. Fillet the four vertical corner edges.
    let st = doc.state();
    let b = &st.bodies[0];
    let verticals: Vec<EdgeRef> = (1..=b.shape.edge_count() as u32)
        .filter(|&e| {
            let i = b.shape.edge_info(e).unwrap();
            (i.start[0] - i.end[0]).abs() < 1e-9 && (i.start[2] - i.end[2]).abs() < 1e-9
        })
        .filter_map(|e| st.edge_ref(b.id, e))
        .collect();
    doc.add_feature(FeatureKind::Fillet(FilletFeature {
        edges: verticals,
        radius: "6 mm".into(),
    }));

    // 3. Sketch on the top face: project the outline, two holes tied to it.
    let st = doc.state().clone();
    let b = &st.bodies[0];
    let top = (1..=b.shape.face_count() as u32)
        .find(|&f| {
            b.shape
                .face_info(f)
                .and_then(|i| i.plane)
                .is_some_and(|p| p.normal()[1] > 0.999)
        })
        .unwrap();
    let frame = b.shape.face_info(top).unwrap().plane.unwrap();
    let mut s2 = Sketch::new();
    let o2 = s2.add_point(frame.to_local([0.0, 0.0, 0.0]));
    s2.entities.get_mut(&o2).unwrap().projected = true;
    let mut holes = Vec::new();
    for x in [-25.0, 25.0] {
        let c = s2.add_point(frame.to_local([x, 8.0, 0.0]));
        holes.push((c, s2.add_circle(c, 5.0)));
    }
    s2.add_constraint(Constraint::Equal(holes[0].1, holes[1].1))
        .unwrap();
    let dia = s2
        .add_constraint(Constraint::Diameter {
            e: holes[0].1,
            value: 10.0,
        })
        .unwrap()
        .unwrap();
    s2.constraints.get_mut(&dia).unwrap().expr = Some("hole".into());
    s2.add_constraint(Constraint::HorizontalPts(holes[0].0, holes[1].0))
        .unwrap();
    s2.add_constraint(Constraint::Distance {
        a: holes[0].0,
        b: holes[1].0,
        kind: DimKind::Horizontal,
        value: 50.0,
    })
    .unwrap();
    let link = s2.add_line(holes[0].0, holes[1].0);
    s2.entities.get_mut(&link).unwrap().construction = true;
    s2.add_constraint(Constraint::Midpoint(o2, link)).unwrap();
    assert_eq!(
        s2.analyze_now().dof,
        0,
        "hole sketch should be fully constrained"
    );
    let sk2 = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Face(st.face_ref(b.id, top).unwrap()),
        sketch: s2,
        projections: vec![],
    }));
    // project the top face's outer edges (they follow the fillets and parameters)
    let top_edges: Vec<EdgeRef> = (1..=b.shape.edge_count() as u32)
        .filter(|&e| b.shape.edge_faces(e).contains(&top))
        .filter_map(|e| st.edge_ref(b.id, e))
        .collect();
    for e in top_edges {
        doc.project_edge(sk2, e).unwrap();
    }
    doc.changed(sk2);
    let disks: Vec<ProfileRef> = doc.state().sketches[&sk2]
        .profiles
        .iter()
        .filter(|p| p.holes.is_empty() && p.area().abs() < 500.0)
        .map(profile_ref)
        .collect();
    assert_eq!(disks.len(), 2, "two hole profiles");
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: sk2,
        profiles: disks,
        faces: vec![],
        extent: Extent::OneSide { d: "-thick".into() },
        op: Operation::Cut,
    }));

    // 4. A revolved boss on the XY plane (axis: a construction line), joined.
    let mut s3 = Sketch::new();
    let axis = s3.add_line_pts([0.0, 0.0], [0.0, 40.0]);
    s3.entities.get_mut(&axis).unwrap().construction = true;
    s3.add_constraint(Constraint::Fix(axis)).unwrap();
    let pts = [
        [0.0, 8.0],
        [14.0, 8.0],
        [14.0, 14.0],
        [8.0, 22.0],
        [0.0, 22.0],
    ];
    let ids: Vec<_> = pts.iter().map(|p| s3.add_point(*p)).collect();
    for i in 0..ids.len() {
        s3.add_line(ids[i], ids[(i + 1) % ids.len()]);
    }
    let sk3 = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XY,
        sketch: s3,
        projections: vec![],
    }));
    let prof3 = doc.state().sketches[&sk3].profiles[0].clone();
    doc.add_feature(FeatureKind::Revolve(RevolveFeature {
        faces: vec![],
        direction: RevolveDirection::OneSide,
        angle2: "90 deg".into(),
        full: false,
        project_axis: false,
        sketch: sk3,
        profiles: vec![profile_ref(&prof3)],
        axis: Axis::SketchLine(axis),
        angle: "360".into(),
        op: Operation::Join,
    }));

    // 5. Chamfer the top circular edge of the boss.
    let st = doc.state();
    let b = &st.bodies[0];
    let rim = (1..=b.shape.edge_count() as u32)
        .find(|&e| matches!(b.shape.edge_info(e).unwrap().kind, kernel::EdgeKind::Circle { radius, .. } if (radius - 8.0).abs() < 1e-6) && (b.shape.edge_info(e).unwrap().mid[1] - 22.0).abs() < 1e-6)
        .unwrap();
    doc.add_feature(FeatureKind::Chamfer(ChamferFeature {
        edges: vec![st.edge_ref(b.id, rim).unwrap()],
        kind: ChamferKind::Equal { d: "1.5".into() },
    }));

    assert!(doc.errors.is_empty(), "errors: {:?}", doc.errors);
    println!(
        "bodies: {}, volume {:.1} mm³",
        doc.state().bodies.len(),
        doc.state().bodies[0].shape.volume()
    );
    std::fs::write(&out, doc.to_json()).unwrap();
    println!("wrote {out}");

    // Show parametric replay: widen the plate.
    doc.params[0].expr = "120 mm".into();
    doc.recompute_all();
    assert!(
        doc.errors.is_empty(),
        "after width change: {:?}",
        doc.errors
    );
    println!(
        "width=120 → volume {:.1} mm³ (all features replayed)",
        doc.state().bodies[0].shape.volume()
    );
}
