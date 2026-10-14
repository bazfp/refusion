//! Regenerate the complete supported-feature gallery:
//! cargo run -p refusion-model --example all_features -- examples/all-features.rfz
use refusion_model::kernel::{EdgeKind, Shape};
use refusion_model::sketch::{Constraint as C, DimKind, EntityId, Geom, Sketch};
use refusion_model::*;

fn add(doc: &mut Document, name: &str, kind: FeatureKind) -> FeatureId {
    let id = doc.add_feature(kind);
    doc.feature_mut(id).unwrap().name = name.into();
    assert!(doc.errors.is_empty(), "{name}: {:?}", doc.errors);
    id
}
fn sketch(doc: &mut Document, name: &str, plane: PlaneRef, geometry: Sketch) -> FeatureId {
    assert!(
        geometry.clone().solve().converged,
        "{name}: sketch did not solve"
    );
    add(
        doc,
        name,
        FeatureKind::Sketch(SketchFeature {
            plane,
            sketch: geometry,
            projections: vec![],
        }),
    )
}
fn rect(
    doc: &mut Document,
    name: &str,
    plane: PlaneRef,
    at: [f64; 2],
    size: [f64; 2],
) -> (FeatureId, ProfileRef) {
    let mut s = Sketch::new();
    s.add_rectangle(at, [at[0] + size[0], at[1] + size[1]]);
    let id = sketch(doc, name, plane, s);
    (id, profile_ref(&doc.state().sketches[&id].profiles[0]))
}
fn extrude(
    doc: &mut Document,
    name: &str,
    sk: FeatureId,
    profiles: Vec<ProfileRef>,
    extent: Extent,
    op: Operation,
) -> FeatureId {
    add(
        doc,
        name,
        FeatureKind::Extrude(ExtrudeFeature {
            sketch: sk,
            profiles,
            faces: vec![],
            extent,
            op,
        }),
    )
}
fn imported_box(doc: &mut Document, name: &str, at: [f64; 3], size: [f64; 3]) -> BodyId {
    let shape = Shape::boxed(size[0], size[1], size[2])
        .unwrap()
        .translated(at, doc.features.len() as u64 + 900)
        .unwrap();
    let id = doc.add_import(name, &shape).unwrap();
    doc.feature_mut(id).unwrap().name = name.into();
    assert!(doc.errors.is_empty());
    doc.state().bodies.last().unwrap().id
}
fn face(doc: &Document, body: BodyId, normal: [f64; 3]) -> FaceRef {
    let b = doc.state().body(body).unwrap();
    let id = (1..=b.shape.face_count() as u32)
        .find(|&id| {
            b.shape
                .face_info(id)
                .and_then(|f| f.plane)
                .is_some_and(|f| kernel::dot(f.normal(), normal) > 0.99)
        })
        .unwrap();
    doc.state().face_ref(body, id).unwrap()
}
fn edge(doc: &Document, body: BodyId) -> EdgeRef {
    let b = doc.state().body(body).unwrap();
    let id = (1..=b.shape.edge_count() as u32)
        .find(|&id| {
            b.shape
                .edge_info(id)
                .is_some_and(|e| e.kind == EdgeKind::Line)
        })
        .unwrap();
    doc.state().edge_ref(body, id).unwrap()
}
fn circle(s: &mut Sketch, at: [f64; 2], radius: f64) -> EntityId {
    let c = s.add_point(at);
    s.add_circle(c, radius)
}
fn constrain(s: &mut Sketch, constraint: C) {
    s.add_constraint(constraint).unwrap();
}

/// A single self-contained design. Stations are separated so Boolean tests do
/// not accidentally intersect unrelated solids; feature names identify coverage.
pub fn build() -> Document {
    let mut d = Document::new();
    for (name, expr, comment) in [
        (
            "plate_width",
            "60 mm",
            "Edit to exercise constrained sketch replay",
        ),
        (
            "plate_height",
            "8 mm",
            "Drives face sketches and associative projections",
        ),
        ("hole_diameter", "6 mm", "Driving diameter dimension"),
        ("wall", "1 mm", "Open and closed shell wall thickness"),
        ("station_depth", "8 mm", "Extrude distance expression"),
        ("plane_lift", "12 mm", "Nested construction-plane distance"),
        ("copy_count", "3", "Rectangular pattern quantity"),
    ] {
        d.params.push(Param {
            name: name.into(),
            expr: expr.into(),
            comment: comment.into(),
        });
    }

    // Parametric plate, fillet, face-hosted linked projection, driven dimension,
    // a through cut, and a joined boss.
    let mut s = Sketch::new();
    let origin = s.add_point([0., 0.]);
    s.entities.get_mut(&origin).unwrap().projected = true;
    let lines = s.add_rectangle([-30., -20.], [30., 20.]);
    let a = s.line_pts(lines[0]).unwrap().0;
    let b = s.line_pts(lines[2]).unwrap().0;
    let diagonal = s.add_line(a, b);
    s.entities.get_mut(&diagonal).unwrap().construction = true;
    constrain(&mut s, C::Midpoint(origin, diagonal));
    let width = s
        .add_constraint(C::Length {
            line: lines[0],
            value: 60.,
        })
        .unwrap()
        .unwrap();
    s.constraints.get_mut(&width).unwrap().expr = Some("plate_width".into());
    constrain(
        &mut s,
        C::Length {
            line: lines[1],
            value: 40.,
        },
    );
    let sk = sketch(&mut d, "01 Plate / constrained rectangle", PlaneRef::XY, s);
    let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
    extrude(
        &mut d,
        "02 Plate / parametric new body",
        sk,
        vec![p],
        Extent::OneSide {
            d: "plate_height".into(),
        },
        Operation::NewBody,
    );
    let plate = d.state().bodies.last().unwrap().id;
    let edges: Vec<_> = {
        let body = d.state().body(plate).unwrap();
        (1..=body.shape.edge_count() as u32)
            .filter(|&e| {
                let e = body.shape.edge_info(e).unwrap();
                e.kind == EdgeKind::Line && (e.start[2] - e.end[2]).abs() > 7.
            })
            .map(|e| d.state().edge_ref(plate, e).unwrap())
            .collect()
    };
    add(
        &mut d,
        "03 Plate / vertical edge fillets",
        FeatureKind::Fillet(FilletFeature {
            edges,
            radius: "3 mm".into(),
        }),
    );
    let top = face(&d, plate, [0., 0., 1.]);
    let (bi, fi) = d.state().resolve_face(&top).unwrap();
    let frame = d.state().bodies[bi]
        .shape
        .face_info(fi)
        .unwrap()
        .plane
        .unwrap();
    let boundary: Vec<_> = (1..=d.state().bodies[bi].shape.edge_count() as u32)
        .filter(|&e| d.state().bodies[bi].shape.edge_faces(e).contains(&fi))
        .map(|e| d.state().edge_ref(plate, e).unwrap())
        .collect();
    let mut holes = Sketch::new();
    let disks: Vec<_> = [-15., 15.]
        .into_iter()
        .map(|x| circle(&mut holes, frame.to_local([x, 0., 8.]), 3.))
        .collect();
    constrain(&mut holes, C::Equal(disks[0], disks[1]));
    let dim = holes
        .add_constraint(C::Diameter {
            e: disks[0],
            value: 6.,
        })
        .unwrap()
        .unwrap();
    holes.constraints.get_mut(&dim).unwrap().expr = Some("hole_diameter".into());
    holes.add_driven(C::Radius {
        e: disks[1],
        value: 3.,
    });
    let sk = sketch(
        &mut d,
        "04 Plate / face sketch with projection and holes",
        PlaneRef::Face(top.clone()),
        holes,
    );
    for edge in boundary {
        d.project_edge(sk, edge).unwrap();
    }
    d.changed(sk);
    let unlinked = match &d.feature(sk).unwrap().kind {
        FeatureKind::Sketch(s) => s.sketch.clone(),
        _ => unreachable!(),
    };
    sketch(
        &mut d,
        "Projection / unlinked fixed reference geometry",
        PlaneRef::Fixed {
            origin: [80., 80., 40.],
            x: frame.x,
            y: frame.y,
        },
        unlinked,
    );
    let profiles = d.state().sketches[&sk]
        .profiles
        .iter()
        .filter(|p| disks.iter().any(|id| p.signature() == vec![*id]))
        .map(profile_ref)
        .collect();
    extrude(
        &mut d,
        "05 Plate / Through All negative cut",
        sk,
        profiles,
        Extent::ThroughAll {
            direction: AllDirection::Negative,
        },
        Operation::Cut,
    );
    let (sk, p) = rect(
        &mut d,
        "06 Plate / boss profile",
        PlaneRef::XY,
        [-4., -4.],
        [8., 8.],
    );
    extrude(
        &mut d,
        "07 Plate / Join",
        sk,
        vec![p],
        Extent::OneSide {
            d: "plate_height + 5 mm".into(),
        },
        Operation::Join,
    );

    // Every extrusion extent, signed direction and taper sense.
    let extents = [
        (
            "Negative distance",
            Extent::OneSide {
                d: "-station_depth".into(),
            },
        ),
        (
            "Symmetric Whole Length",
            Extent::Symmetric {
                d: "station_depth".into(),
            },
        ),
        (
            "Symmetric Half Length",
            Extent::SymmetricHalf {
                d: "station_depth".into(),
            },
        ),
        (
            "Two Sides",
            Extent::TwoSides {
                d1: "station_depth".into(),
                d2: "3 mm".into(),
            },
        ),
        (
            "Positive taper",
            Extent::Tapered {
                extent: Box::new(Extent::OneSide {
                    d: "station_depth".into(),
                }),
                angle: "5 deg".into(),
            },
        ),
        (
            "Negative taper",
            Extent::Tapered {
                extent: Box::new(Extent::OneSide {
                    d: "station_depth".into(),
                }),
                angle: "-5 deg".into(),
            },
        ),
        (
            "Two-sided taper",
            Extent::Tapered {
                extent: Box::new(Extent::TwoSides {
                    d1: "6 mm".into(),
                    d2: "3 mm".into(),
                }),
                angle: "4 deg".into(),
            },
        ),
    ];
    for (i, (name, extent)) in extents.into_iter().enumerate() {
        let at = [-130. + i as f64 * 40., -90.];
        let (sk, p) = rect(
            &mut d,
            &format!("Extent / {name} profile"),
            PlaneRef::XY,
            at,
            [18., 16.],
        );
        extrude(
            &mut d,
            &format!("Extent / {name}"),
            sk,
            vec![p],
            extent,
            Operation::NewBody,
        );
    }
    for (i, direction) in [AllDirection::Positive, AllDirection::Both]
        .into_iter()
        .enumerate()
    {
        let at = [-125. + i as f64 * 40., -40.];
        imported_box(
            &mut d,
            "Through All / embedded target",
            [at[0], at[1], 0.],
            [18., 16., 10.],
        );
        let plane = add(
            &mut d,
            "Through All / start plane",
            FeatureKind::OffsetPlane(OffsetPlaneFeature {
                base: PlaneRef::XY,
                distance: if direction == AllDirection::Both {
                    "5 mm"
                } else {
                    "0 mm"
                }
                .into(),
            }),
        );
        let (sk, p) = rect(
            &mut d,
            "Through All / aperture",
            PlaneRef::Construction(plane),
            [at[0] + 6., at[1] + 6.],
            [4., 4.],
        );
        extrude(
            &mut d,
            &format!("Through All / {direction:?}"),
            sk,
            vec![p],
            Extent::ThroughAll { direction },
            Operation::Cut,
        );
    }
    let target = imported_box(
        &mut d,
        "To Object / embedded target",
        [80., -40., 0.],
        [22., 18., 12.],
    );
    let target_face = face(&d, target, [0., 0., 1.]);
    let (sk, p) = rect(
        &mut d,
        "To Object / profile",
        PlaneRef::XY,
        [84., -36.],
        [6., 6.],
    );
    extrude(
        &mut d,
        "To Object / planar target with offset",
        sk,
        vec![p],
        Extent::ToFace {
            face: target_face,
            offset: "-2 mm".into(),
        },
        Operation::Cut,
    );
    let source_face = face(&d, target, [1., 0., 0.]);
    add(
        &mut d,
        "Face Extrude / New Body",
        FeatureKind::Extrude(ExtrudeFeature {
            sketch: 0,
            profiles: vec![],
            faces: vec![source_face],
            extent: Extent::OneSide { d: "4 mm".into() },
            op: Operation::NewBody,
        }),
    );

    // Circle/rectangle overlap: independently chosen intersection region.
    let mut overlap = Sketch::new();
    overlap.add_rectangle([70., 25.], [90., 45.]);
    circle(&mut overlap, [88., 35.], 8.);
    let sk = sketch(
        &mut d,
        "Overlap / circle crossing rectangle",
        PlaneRef::XY,
        overlap,
    );
    let profiles = &d.state().sketches[&sk].profiles;
    assert_eq!(profiles.len(), 3);
    let p = profile_ref(profiles.iter().find(|p| p.contains([85., 35.])).unwrap());
    extrude(
        &mut d,
        "Overlap / intersection profile",
        sk,
        vec![p],
        Extent::OneSide { d: "6 mm".into() },
        Operation::NewBody,
    );
    imported_box(
        &mut d,
        "Intersect / Boolean target",
        [115., 25., 0.],
        [18., 18., 8.],
    );
    let mut circle_profile = Sketch::new();
    circle(&mut circle_profile, [133., 34.], 10.);
    let sk = sketch(
        &mut d,
        "Intersect / circular tool profile",
        PlaneRef::XY,
        circle_profile,
    );
    let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
    extrude(
        &mut d,
        "Intersect / Boolean operation",
        sk,
        vec![p],
        Extent::OneSide { d: "8 mm".into() },
        Operation::Intersect,
    );

    // Three chamfer parameterisations and inward/outward, open/closed shells.
    for (i, kind) in [
        ChamferKind::Equal { d: "0.8 mm".into() },
        ChamferKind::TwoDistances {
            d1: "0.8 mm".into(),
            d2: "1.2 mm".into(),
        },
        ChamferKind::DistanceAngle {
            d: "1 mm".into(),
            angle: "30 deg".into(),
        },
    ]
    .into_iter()
    .enumerate()
    {
        let body = imported_box(
            &mut d,
            "Chamfer / embedded stock",
            [-130. + i as f64 * 40., 25., 0.],
            [18., 18., 12.],
        );
        let edge = edge(&d, body);
        add(
            &mut d,
            &format!("Chamfer / mode {}", i + 1),
            FeatureKind::Chamfer(ChamferFeature {
                edges: vec![edge],
                kind,
            }),
        );
    }
    for (i, (outward, open)) in [(false, true), (false, false), (true, true), (true, false)]
        .into_iter()
        .enumerate()
    {
        let body = imported_box(
            &mut d,
            "Shell / embedded stock",
            [-125. + i as f64 * 40., 70., 0.],
            [18., 18., 16.],
        );
        let faces = if open {
            vec![face(&d, body, [0., 0., 1.])]
        } else {
            vec![]
        };
        add(
            &mut d,
            &format!(
                "Shell / {} {}",
                if outward { "Outside" } else { "Inside" },
                if open { "open" } else { "closed" }
            ),
            FeatureKind::Shell(ShellFeature {
                body,
                body_tag: None,
                faces,
                thickness: "wall".into(),
                outward,
                outside_thickness: None,
            }),
        );
    }

    // All plane reference forms, nested signed offsets and angled sketch hosts.
    let lifted = add(
        &mut d,
        "Plane / XY offset",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::XY,
            distance: "plane_lift".into(),
        }),
    );
    let nested = add(
        &mut d,
        "Plane / nested signed offset",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::Construction(lifted),
            distance: "-2 mm".into(),
        }),
    );
    add(
        &mut d,
        "Plane / face offset",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::Face(top),
            distance: "3 mm".into(),
        }),
    );
    add(
        &mut d,
        "Plane / XZ offset",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::XZ,
            distance: "-5 mm".into(),
        }),
    );
    add(
        &mut d,
        "Plane / YZ offset",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::YZ,
            distance: "5 mm".into(),
        }),
    );
    let (sk, p) = rect(
        &mut d,
        "Plane / nested-plane sketch",
        PlaneRef::Construction(nested),
        [45., -10.],
        [12., 12.],
    );
    extrude(
        &mut d,
        "Plane / nested-plane solid",
        sk,
        vec![p],
        Extent::OneSide { d: "5 mm".into() },
        Operation::NewBody,
    );
    let angle_plane = add(
        &mut d,
        "Plane / at angle",
        FeatureKind::AngledPlane(AngledPlaneFeature {
            base: PlaneRef::XY,
            axis: LinearRef::X,
            angle: "30 deg".into(),
        }),
    );
    let (sk, p) = rect(
        &mut d,
        "Plane / angled-plane sketch",
        PlaneRef::Construction(angle_plane),
        [45., 45.],
        [12., 12.],
    );
    extrude(
        &mut d,
        "Plane / angled solid",
        sk,
        vec![p],
        Extent::OneSide { d: "4 mm".into() },
        Operation::NewBody,
    );

    // Revolve directions and all axis types. Symmetric angles are per side.
    for (i, (direction, full)) in [
        (RevolveDirection::OneSide, true),
        (RevolveDirection::OneSide, false),
        (RevolveDirection::TwoSides, false),
        (RevolveDirection::Symmetric, false),
    ]
    .into_iter()
    .enumerate()
    {
        let mut s = Sketch::new();
        let axis = s.add_line_pts([0., 0.], [0., 10.]);
        s.entities.get_mut(&axis).unwrap().construction = true;
        s.add_rectangle([5., 0.], [9., 8.]);
        let plane = PlaneRef::Fixed {
            origin: [-120. + i as f64 * 65., -140., 20.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        };
        let sk = sketch(&mut d, "Revolve / local-axis profile", plane, s);
        let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
        add(
            &mut d,
            &format!(
                "Revolve / {direction:?} {}",
                if full { "Full" } else { "Partial" }
            ),
            FeatureKind::Revolve(RevolveFeature {
                sketch: sk,
                faces: vec![],
                profiles: vec![p],
                axis: Axis::SketchLine(axis),
                direction,
                angle: "45 deg".into(),
                angle2: "30 deg".into(),
                full,
                project_axis: true,
                op: Operation::NewBody,
            }),
        );
    }
    for (axis, plane, at) in [
        (Axis::X, PlaneRef::XY, [-140., 5.]),
        (Axis::Y, PlaneRef::XY, [5., -125.]),
        (
            Axis::Z,
            PlaneRef::Fixed {
                origin: [0., 0., 45.],
                x: [1., 0., 0.],
                y: [0., 0., 1.],
            },
            [5., 0.],
        ),
    ] {
        let (sk, p) = rect(&mut d, "Revolve / origin-axis profile", plane, at, [4., 8.]);
        add(
            &mut d,
            "Revolve / origin axis",
            FeatureKind::Revolve(RevolveFeature {
                sketch: sk,
                faces: vec![],
                profiles: vec![p],
                axis,
                direction: RevolveDirection::OneSide,
                angle: "90 deg".into(),
                angle2: "30 deg".into(),
                full: false,
                project_axis: true,
                op: Operation::NewBody,
            }),
        );
    }
    let stock = imported_box(
        &mut d,
        "Revolve / face source",
        [110., 100., 0.],
        [10., 8., 8.],
    );
    let reference = face(&d, stock, [0., -1., 0.]);
    let b = d.state().body(stock).unwrap();
    let axis = (1..=b.shape.edge_count() as u32)
        .find(|&e| {
            let e = b.shape.edge_info(e).unwrap();
            e.kind == EdgeKind::Line
                && (e.start[0] - 110.).abs() < 1e-6
                && (e.start[1] - 100.).abs() < 1e-6
                && (e.end[2] - e.start[2]).abs() > 7.
        })
        .unwrap();
    let axis = d.state().edge_ref(stock, axis).unwrap();
    add(
        &mut d,
        "Revolve / planar face and linear model edge",
        FeatureKind::Revolve(RevolveFeature {
            sketch: 0,
            faces: vec![reference],
            profiles: vec![],
            axis: Axis::Edge(axis),
            direction: RevolveDirection::OneSide,
            angle: "60 deg".into(),
            angle2: "30 deg".into(),
            full: false,
            project_axis: false,
            op: Operation::NewBody,
        }),
    );
    let circular_body = d.state().bodies.last().unwrap();
    let circular = (1..=circular_body.shape.edge_count() as u32)
        .find(|&e| {
            matches!(
                circular_body.shape.edge_info(e).unwrap().kind,
                EdgeKind::Circle { .. }
            )
        })
        .unwrap();
    let axis = d.state().edge_ref(circular_body.id, circular).unwrap();
    let (sk, p) = rect(
        &mut d,
        "Revolve / circular-axis profile",
        PlaneRef::Fixed {
            origin: [110., 100., 20.],
            x: [1., 0., 0.],
            y: [0., 0., 1.],
        },
        [12., 0.],
        [3., 5.],
    );
    add(
        &mut d,
        "Revolve / circular model edge",
        FeatureKind::Revolve(RevolveFeature {
            sketch: sk,
            faces: vec![],
            profiles: vec![p],
            axis: Axis::Edge(axis),
            direction: RevolveDirection::OneSide,
            angle: "90 deg".into(),
            angle2: "30 deg".into(),
            full: false,
            project_axis: true,
            op: Operation::NewBody,
        }),
    );

    let seed = imported_box(&mut d, "Mirror / seed", [145., 65., 0.], [10., 10., 10.]);
    add(
        &mut d,
        "Mirror / New Body",
        FeatureKind::Mirror(MirrorFeature {
            bodies: vec![seed],
            plane: PlaneRef::YZ,
            join: false,
        }),
    );
    let seed = imported_box(&mut d, "Mirror Join / seed", [0., 105., 0.], [8., 10., 10.]);
    add(
        &mut d,
        "Mirror / Join",
        FeatureKind::Mirror(MirrorFeature {
            bodies: vec![seed],
            plane: PlaneRef::YZ,
            join: true,
        }),
    );
    for (i, spacing) in [true, false].into_iter().enumerate() {
        let seed = imported_box(
            &mut d,
            "Rectangular pattern / seed",
            [-140. + i as f64 * 210., -180., 0.],
            [8., 8., 6.],
        );
        add(
            &mut d,
            &format!(
                "Rectangular pattern / {}",
                if spacing { "Spacing" } else { "Extent" }
            ),
            FeatureKind::Pattern(PatternFeature {
                objects: PatternObjects::Bodies,
                features: vec![],
                bodies: vec![seed],
                kind: PatternKind::Rectangular {
                    axis1: LinearRef::X,
                    axis2: LinearRef::Y,
                    count1: "copy_count".into(),
                    count2: "2".into(),
                    distance1: "30 mm".into(),
                    distance2: "18 mm".into(),
                    spacing,
                },
            }),
        );
    }
    for (i, angle) in ["360 deg", "150 deg"].into_iter().enumerate() {
        let seed = imported_box(
            &mut d,
            "Circular pattern / seed",
            [175., 0., 25. + i as f64 * 20.],
            [8., 8., 6.],
        );
        add(
            &mut d,
            "Circular pattern / distribution",
            FeatureKind::Pattern(PatternFeature {
                objects: PatternObjects::Bodies,
                features: vec![],
                bodies: vec![seed],
                kind: PatternKind::Circular {
                    axis: LinearRef::Z,
                    count: "5".into(),
                    angle: angle.into(),
                },
            }),
        );
    }
    // Feature patterns repeat drilled material on its source body, rather than stock copies.
    for (name, at, size, center, kind) in [
        (
            "Rectangular feature pattern",
            [240., -180., 0.],
            [40., 20., 10.],
            [245., -175.],
            PatternKind::Rectangular {
                axis1: LinearRef::X,
                axis2: LinearRef::Y,
                count1: "3".into(),
                count2: "1".into(),
                distance1: "10 mm".into(),
                distance2: "1 mm".into(),
                spacing: true,
            },
        ),
        (
            "Circular feature pattern",
            [-25., -25., -60.],
            [50., 50., 10.],
            [10., 0.],
            PatternKind::Circular {
                axis: LinearRef::Z,
                count: "4".into(),
                angle: "360 deg".into(),
            },
        ),
    ] {
        let body = imported_box(&mut d, &format!("{name} / stock"), at, size);
        let mut hole = HoleFeature::new(
            body,
            PlaneRef::Fixed {
                origin: [0., 0., at[2]],
                x: [1., 0., 0.],
                y: [0., 1., 0.],
            },
        );
        hole.body_tag = d
            .state()
            .body(body)
            .and_then(|b| b.shape.face_tags().first().copied());
        hole.centers = vec![[format!("{} mm", center[0]), format!("{} mm", center[1])]];
        hole.diameter = "4 mm".into();
        hole.extent = HoleExtent::ThroughAll;
        hole.tip_angle = None;
        hole.flip = true;
        let source = add(
            &mut d,
            &format!("{name} / seed hole"),
            FeatureKind::Hole(hole),
        );
        add(
            &mut d,
            name,
            FeatureKind::Pattern(PatternFeature {
                objects: PatternObjects::Features,
                features: vec![source],
                bodies: vec![],
                kind,
            }),
        );
    }

    let stock = imported_box(
        &mut d,
        "Edge reference / seed",
        [-145., 110., 0.],
        [8., 8., 8.],
    );
    let reference = edge(&d, stock);
    add(
        &mut d,
        "Pattern / linear model edge",
        FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![stock],
            kind: PatternKind::Circular {
                axis: LinearRef::Edge(reference),
                count: "2".into(),
                angle: "90 deg".into(),
            },
        }),
    );

    sketch(
        &mut d,
        "Constraint gallery / every constraint and dimension",
        PlaneRef::Fixed {
            origin: [-105., 125., 35.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        },
        constraint_gallery(),
    );
    sketch(
        &mut d,
        "Sketch editing / Trim Extend Offset Arc",
        PlaneRef::Fixed {
            origin: [40., 125., 35.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        },
        edited_sketch(),
    );
    // Empty origin-plane sketches expose XZ and YZ hosts without adding clutter.
    sketch(&mut d, "Sketch host / XZ", PlaneRef::XZ, Sketch::new());
    sketch(&mut d, "Sketch host / YZ", PlaneRef::YZ, Sketch::new());

    for (i, kind) in [
        HoleKind::Simple,
        HoleKind::Counterbore {
            diameter: "10 mm".into(),
            depth: "2 mm".into(),
        },
        HoleKind::Countersink {
            diameter: "10 mm".into(),
            angle: "90 deg".into(),
        },
    ]
    .into_iter()
    .enumerate()
    {
        let at = [210. + i as f64 * 30., 80., 0.];
        let body = imported_box(&mut d, "Hole / embedded stock", at, [20., 20., 12.]);
        let top = face(&d, body, [0., 0., 1.]);
        let plane = if i == 0 {
            PlaneRef::XY
        } else {
            PlaneRef::Face(top)
        };
        let xy = d
            .state()
            .plane_frame(&plane)
            .unwrap()
            .to_local([at[0] + 10., at[1] + 10., 12.]);
        let body_tag = d
            .state()
            .body(body)
            .and_then(|b| b.shape.face_tags().first().copied());
        let extent = match i {
            0 => HoleExtent::Distance {
                depth: "6 mm".into(),
            },
            1 => HoleExtent::ThroughAll,
            _ => HoleExtent::ToFace {
                face: face(&d, body, [0., 0., -1.]),
                offset: "-2 mm".into(),
            },
        };
        add(
            &mut d,
            "Hole / drilled feature",
            FeatureKind::Hole(HoleFeature {
                body,
                body_tag,
                plane,
                centers: vec![[format!("{} mm", xy[0]), format!("{} mm", xy[1])]],
                diameter: "hole_diameter".into(),
                kind,
                extent,
                tip_angle: if i == 0 { Some("118 deg".into()) } else { None },
                flip: i == 0,
            }),
        );
    }
    for open in [false, true] {
        let body = imported_box(
            &mut d,
            "Both Shell / embedded stock",
            [210. + if open { 40. } else { 0. }, 120., 0.],
            [20., 20., 16.],
        );
        let faces = if open {
            vec![face(&d, body, [0., 0., 1.])]
        } else {
            vec![]
        };
        let body_tag = d
            .state()
            .body(body)
            .and_then(|b| b.shape.face_tags().first().copied());
        add(
            &mut d,
            "Shell / Both independent offsets",
            FeatureKind::Shell(ShellFeature {
                body,
                body_tag,
                faces,
                thickness: "wall".into(),
                outward: false,
                outside_thickness: Some("0.5 mm".into()),
            }),
        );
    }
    assert!(d.errors.is_empty());
    d
}

/// One evolving enclosure; patterns necessarily create additional bodies.
pub fn build_combined() -> Document {
    let mut d = Document::new();
    for (name, expr) in [
        ("wall", "1 mm"),
        ("boss_height", "16 mm"),
        ("hole_diameter", "3 mm"),
        ("copy_count", "2"),
    ] {
        d.params.push(Param {
            name: name.into(),
            expr: expr.into(),
            comment: "Combined workflow replay parameter".into(),
        });
    }
    let body = imported_box(
        &mut d,
        "01 Enclosure / embedded half stock",
        [0., -20., 0.],
        [30., 40., 12.],
    );
    let top = face(&d, body, [0., 0., 1.]);
    add(
        &mut d,
        "02 Enclosure / open two-sided shell",
        FeatureKind::Shell(ShellFeature {
            body,
            body_tag: None,
            faces: vec![top],
            thickness: "wall".into(),
            outward: false,
            outside_thickness: Some("0.5 mm".into()),
        }),
    );
    add(
        &mut d,
        "03 Enclosure / joined mirror",
        FeatureKind::Mirror(MirrorFeature {
            bodies: vec![body],
            plane: PlaneRef::YZ,
            join: true,
        }),
    );
    let (sk, p) = rect(
        &mut d,
        "04 Boss / rectangle",
        PlaneRef::XY,
        [-10., -10.],
        [20., 20.],
    );
    extrude(
        &mut d,
        "05 Boss / joined parametric extrusion",
        sk,
        vec![p],
        Extent::OneSide {
            d: "boss_height".into(),
        },
        Operation::Join,
    );
    let e = edge(&d, body);
    add(
        &mut d,
        "06 Enclosure / fillet",
        FeatureKind::Fillet(FilletFeature {
            edges: vec![e],
            radius: "0.3 mm".into(),
        }),
    );
    let e = edge(&d, body);
    add(
        &mut d,
        "07 Enclosure / chamfer",
        FeatureKind::Chamfer(ChamferFeature {
            edges: vec![e],
            kind: ChamferKind::Equal { d: "0.2 mm".into() },
        }),
    );
    let mut s = Sketch::new();
    s.add_rectangle([4., -20.], [8., -12.]);
    let sk = sketch(&mut d, "08 Collar / revolve profile", PlaneRef::XZ, s);
    let p = profile_ref(&d.state().sketches[&sk].profiles[0]);
    add(
        &mut d,
        "09 Collar / joined full revolve",
        FeatureKind::Revolve(RevolveFeature {
            sketch: sk,
            profiles: vec![p],
            faces: vec![],
            axis: Axis::Z,
            direction: RevolveDirection::OneSide,
            angle: "360 deg".into(),
            angle2: "0 deg".into(),
            full: true,
            project_axis: true,
            op: Operation::Join,
        }),
    );
    let top = face(&d, body, [0., 0., 1.]);
    let (bi, fi) = d.state().resolve_face(&top).unwrap();
    let frame = d.state().bodies[bi]
        .shape
        .face_info(fi)
        .unwrap()
        .plane
        .unwrap();
    let reference = edge(&d, body);
    let mut s = Sketch::new();
    let c = circle(&mut s, frame.to_local([0., 0., frame.origin[2]]), 1.5);
    let dim = s
        .add_constraint(C::Diameter { e: c, value: 3. })
        .unwrap()
        .unwrap();
    s.constraints.get_mut(&dim).unwrap().expr = Some("hole_diameter".into());
    s.add_driven(C::Radius { e: c, value: 1.5 });
    let sk = sketch(
        &mut d,
        "10 Bore / linked face sketch",
        PlaneRef::Face(top.clone()),
        s,
    );
    d.project_edge(sk, reference).unwrap();
    d.changed(sk);
    let xy = frame.to_local([0., 0., frame.origin[2]]);
    add(
        &mut d,
        "11 Bore / through Hole feature",
        FeatureKind::Hole(HoleFeature {
            body,
            body_tag: None,
            plane: PlaneRef::Face(top),
            centers: vec![[format!("{} mm", xy[0]), format!("{} mm", xy[1])]],
            diameter: "hole_diameter".into(),
            kind: HoleKind::Simple,
            extent: HoleExtent::ThroughAll,
            tip_angle: None,
            flip: false,
        }),
    );
    let lift = add(
        &mut d,
        "12 Plane / raised rim",
        FeatureKind::OffsetPlane(OffsetPlaneFeature {
            base: PlaneRef::XY,
            distance: "10 mm".into(),
        }),
    );
    let (sk, p) = rect(
        &mut d,
        "13 Rim / plane hosted notch",
        PlaneRef::Construction(lift),
        [25., -4.],
        [8., 8.],
    );
    extrude(
        &mut d,
        "14 Rim / two-sided notch",
        sk,
        vec![p],
        Extent::TwoSides {
            d1: "4 mm".into(),
            d2: "2 mm".into(),
        },
        Operation::Cut,
    );
    let tilt = add(
        &mut d,
        "15 Plane / tilted notch",
        FeatureKind::AngledPlane(AngledPlaneFeature {
            base: PlaneRef::XY,
            axis: LinearRef::X,
            angle: "15 deg".into(),
        }),
    );
    let (sk, p) = rect(
        &mut d,
        "16 Floor / angled notch",
        PlaneRef::Construction(tilt),
        [15., -3.],
        [3., 6.],
    );
    extrude(
        &mut d,
        "17 Floor / symmetric angled cut",
        sk,
        vec![p],
        Extent::Symmetric { d: "8 mm".into() },
        Operation::Cut,
    );
    let (sk, p) = rect(
        &mut d,
        "18 Envelope / intersection tool",
        PlaneRef::XY,
        [-35., -25.],
        [70., 50.],
    );
    extrude(
        &mut d,
        "19 Envelope / intersect collar height",
        sk,
        vec![p],
        Extent::OneSide { d: "19 mm".into() },
        Operation::Intersect,
    );
    // Constraint and edit results remain sketches: open geometry cannot all be solid profiles.
    sketch(
        &mut d,
        "20 Reference / all constraints and dimensions",
        PlaneRef::Fixed {
            origin: [-45., 30., 0.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        },
        constraint_gallery(),
    );
    sketch(
        &mut d,
        "21 Reference / trim extend offset arcs",
        PlaneRef::Fixed {
            origin: [-30., -50., 0.],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        },
        edited_sketch(),
    );
    assert_eq!(d.state().bodies.len(), 1);
    add(
        &mut d,
        "22 Assembly / rectangular body pattern",
        FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![body],
            kind: PatternKind::Rectangular {
                axis1: LinearRef::X,
                axis2: LinearRef::Y,
                count1: "copy_count".into(),
                count2: "1".into(),
                distance1: "85 mm".into(),
                distance2: "0 mm".into(),
                spacing: true,
            },
        }),
    );
    let rectangular_copy = d.state().bodies[1].id;
    add(
        &mut d,
        "23 Assembly / circular body pattern",
        FeatureKind::Pattern(PatternFeature {
            objects: PatternObjects::Bodies,
            features: vec![],
            bodies: vec![rectangular_copy],
            kind: PatternKind::Circular {
                axis: LinearRef::Z,
                count: "2".into(),
                angle: "90 deg".into(),
            },
        }),
    );
    d
}

fn constraint_gallery() -> Sketch {
    let mut s = Sketch::new();
    for index in 0..22 {
        let x = (index % 6) as f64 * 18.;
        let y = (index / 6) as f64 * 18.;
        let p = |a: f64, b: f64| [x + a, y + b];
        match index {
            0 => {
                let line = s.add_line_pts(p(0., 0.), p(8., 0.));
                let point = s.add_point(p(4., 0.));
                constrain(&mut s, C::Coincident(point, line));
            }
            1 => {
                let line = s.add_line_pts(p(0., 0.), p(8., 0.));
                constrain(&mut s, C::Horizontal(line));
            }
            2 => {
                let line = s.add_line_pts(p(0., 0.), p(0., 8.));
                constrain(&mut s, C::Vertical(line));
            }
            3 => {
                let a = s.add_point(p(0., 0.));
                let b = s.add_point(p(8., 0.));
                constrain(&mut s, C::HorizontalPts(a, b));
            }
            4 => {
                let a = s.add_point(p(0., 0.));
                let b = s.add_point(p(0., 8.));
                constrain(&mut s, C::VerticalPts(a, b));
            }
            5 | 6 | 8 | 10 | 21 => {
                let a = s.add_line_pts(p(0., 0.), p(8., 0.));
                let b = match index {
                    6 => s.add_line_pts(p(4., 0.), p(4., 8.)),
                    10 => s.add_line_pts(p(9., 0.), p(14., 0.)),
                    21 => s.add_line_pts(p(0., 3.), p(6., 9.)),
                    _ => s.add_line_pts(p(0., 3.), p(8., 3.)),
                };
                constrain(
                    &mut s,
                    match index {
                        5 => C::Parallel(a, b),
                        6 => C::Perpendicular(a, b),
                        8 => C::Equal(a, b),
                        10 => C::Collinear(a, b),
                        _ => C::Angle {
                            a,
                            b,
                            value: std::f64::consts::FRAC_PI_4,
                        },
                    },
                );
            }
            7 => {
                let line = s.add_line_pts(p(0., 0.), p(8., 0.));
                let c = circle(&mut s, p(4., 2.), 2.);
                constrain(&mut s, C::Tangent(line, c));
            }
            9 => {
                let a = circle(&mut s, p(5., 5.), 2.);
                let b = circle(&mut s, p(5., 5.), 4.);
                constrain(&mut s, C::Concentric(a, b));
            }
            11 => {
                let line = s.add_line_pts(p(0., 0.), p(8., 0.));
                let point = s.add_point(p(4., 0.));
                constrain(&mut s, C::Midpoint(point, line));
            }
            12 => {
                let axis = s.add_line_pts(p(5., 0.), p(5., 10.));
                let a = s.add_point(p(2., 5.));
                let b = s.add_point(p(8., 5.));
                constrain(&mut s, C::Symmetric(a, b, axis));
            }
            13 => {
                let c = circle(&mut s, p(5., 5.), 3.);
                constrain(&mut s, C::Fix(c));
            }
            14 => {
                let a = s.add_line_pts(p(0., 0.), p(8., 0.));
                let b = s.offset(a, 3.).unwrap();
                assert!(s
                    .constraints
                    .values()
                    .any(|c| matches!(c.c, C::Offset { a: u, b: v, .. } if u == a && v == b)));
            }
            15..=17 => {
                let a = s.add_point(p(0., 0.));
                let b = s.add_point(if index == 15 { p(3., 4.) } else { p(6., 4.) });
                constrain(
                    &mut s,
                    C::Distance {
                        a,
                        b,
                        kind: match index {
                            15 => DimKind::Aligned,
                            16 => DimKind::Horizontal,
                            _ => DimKind::Vertical,
                        },
                        value: match index {
                            15 => 5.,
                            16 => 6.,
                            _ => 4.,
                        },
                    },
                );
            }
            18 => {
                let line = s.add_line_pts(p(0., 0.), p(8., 0.));
                constrain(&mut s, C::Length { line, value: 8. });
            }
            19 | 20 => {
                let c = circle(&mut s, p(5., 5.), 3.);
                constrain(
                    &mut s,
                    if index == 19 {
                        C::Radius { e: c, value: 3. }
                    } else {
                        C::Diameter { e: c, value: 6. }
                    },
                );
            }
            _ => unreachable!(),
        }
    }
    for (index, constraint) in s.constraints.values_mut().enumerate() {
        constraint.label = [2., 2. + (index % 3) as f64];
    }
    assert!(s.solve().converged);
    s
}
fn edited_sketch() -> Sketch {
    let mut s = Sketch::new();
    let trim = s.add_line_pts([0., 0.], [12., 0.]);
    s.add_line_pts([4., -3.], [4., 3.]);
    s.add_line_pts([8., -3.], [8., 3.]);
    s.trim(trim, [6., 0.]).unwrap();
    let extend = s.add_line_pts([0., 8.], [6., 8.]);
    s.add_line_pts([10., 5.], [10., 11.]);
    s.extend(extend, [6., 8.]).unwrap();
    let extended = s
        .entities
        .keys()
        .copied()
        .find(|&id| {
            s.line_pts(id).is_some_and(|(a, b)| {
                (s.pos(a)[1] - 8.).abs() < 1e-6
                    && (s.pos(b)[1] - 8.).abs() < 1e-6
                    && (s.pos(a)[0] - s.pos(b)[0]).abs() > 9.
            })
        })
        .unwrap();
    s.offset(extended, 2.).unwrap();
    let center = s.add_point([20., 5.]);
    let start = s.add_point([20., 0.]);
    let end = s.add_point([25., 5.]);
    let arc = s.add_arc(center, start, end);
    let tangent_start = s.add_point([12., 0.]);
    let tangent = s.add_line(tangent_start, start);
    constrain(&mut s, C::Tangent(tangent, arc));
    let first = [14., 14.];
    let last = [24., 14.];
    let (center, ccw) =
        refusion_model::sketch::edit::three_point_arc(first, last, [19., 20.]).unwrap();
    let c = s.add_point(center);
    let a = s.add_point(first);
    let b = s.add_point(last);
    let three_point = if ccw {
        s.add_arc(c, a, b)
    } else {
        s.add_arc(c, b, a)
    };
    constrain(
        &mut s,
        C::Radius {
            e: three_point,
            value: (first[0] - center[0]).hypot(first[1] - center[1]),
        },
    );
    let offset_circle = circle(&mut s, [52., 5.], 3.);
    s.offset(offset_circle, 2.).unwrap();
    let center_rectangle = s.add_point([36., 5.]);
    s.add_center_rectangle(center_rectangle, [40., 9.]);
    assert!(s.solve().converged);
    assert!(s
        .entities
        .values()
        .any(|e| matches!(e.geom, Geom::Arc { .. })));
    s
}
fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/all-features.rfz".into());
    let doc = if std::env::args().any(|arg| arg == "--combined") {
        build_combined()
    } else {
        build()
    };
    for body in &doc.state().bodies {
        assert!(
            body.shape.is_valid() && body.shape.is_solid() && body.shape.volume() > 0.,
            "invalid body {}",
            body.id
        );
    }
    std::fs::write(&out, doc.to_json()).unwrap();
    println!(
        "Wrote {out}: {} features, {} bodies, {} planes, {} sketches",
        doc.features.len(),
        doc.state().bodies.len(),
        doc.state().planes.len(),
        doc.state().sketches.len()
    );
}
