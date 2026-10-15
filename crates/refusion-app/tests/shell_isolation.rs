#[allow(dead_code)]
#[path = "../src/shell_worker.rs"]
mod shell_worker;

use refusion_model::{
    kernel::{self, EdgeKind, Frame, Seg2, Shape},
    *,
};

fn isolated(
    shape: &Shape,
    faces: &[u32],
    thickness: f64,
    feature: u64,
    round: bool,
) -> kernel::Result<Shape> {
    let executable = std::env::var_os("REFUSION_TEST_EXECUTABLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_refusion").into());
    shell_worker::shell_at(&executable, shape, faces, thickness, feature, round)
}

fn plate() -> Shape {
    let frame = Frame {
        origin: [0.; 3],
        x: [1., 0., 0.],
        y: [0., 1., 0.],
    };
    let mut loops = vec![vec![
        Seg2::Line {
            a: [0., 0.],
            b: [60., 0.],
        },
        Seg2::Line {
            a: [60., 0.],
            b: [60., 40.],
        },
        Seg2::Line {
            a: [60., 40.],
            b: [0., 40.],
        },
        Seg2::Line {
            a: [0., 40.],
            b: [0., 0.],
        },
    ]];
    for c in [[10., 10.], [50., 10.], [10., 30.], [50., 30.]] {
        loops.push(vec![
            Seg2::Arc {
                c,
                a: [c[0] + 3., c[1]],
                b: [c[0] - 3., c[1]],
                ccw: false,
            },
            Seg2::Arc {
                c,
                a: [c[0] - 3., c[1]],
                b: [c[0] + 3., c[1]],
                ccw: false,
            },
        ]);
    }
    let shape = Shape::face_from_loops(&frame, &loops, &[])
        .unwrap()
        .extrude([0., 0., 1.], 0., 10., 7)
        .unwrap();
    let edges: Vec<_> = (1..=shape.edge_count() as u32)
        .filter(|&i| {
            let e = shape.edge_info(i).unwrap();
            matches!(e.kind, EdgeKind::Line)
                && (e.start[2] - 10.).abs() < 1e-6
                && (e.end[2] - 10.).abs() < 1e-6
        })
        .collect();
    assert_eq!(edges.len(), 4);
    let result = shape.chamfer(&edges, 1., None, 8).unwrap();
    assert!(result.is_valid() && result.is_solid());
    result
}

fn overlapping_chamfered_plate() -> Shape {
    let mut doc = Document::new();
    let mut sketch = sketch::Sketch::new();
    sketch.add_rectangle([-70., -55.], [73., 26.]);
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::XZ,
        sketch,
        projections: vec![],
    }));
    let profiles = doc.state().sketches[&id]
        .profiles
        .iter()
        .map(profile_ref)
        .collect();
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: id,
        profiles,
        faces: vec![],
        extent: Extent::OneSide { d: "20 mm".into() },
        op: Operation::NewBody,
    }));
    let mut sketch = sketch::Sketch::new();
    for (x, z, r) in [
        (-45., -13., 18.5),
        (10., 31., 42.5),
        (44., -12., 21.),
        (-50., 21., 16.5),
    ] {
        let center = sketch.add_point([x, -z]);
        sketch.add_circle(center, r);
    }
    let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
        plane: PlaneRef::Fixed {
            origin: [0., 20., 0.],
            x: [1., 0., 0.],
            y: [0., 0., -1.],
        },
        sketch,
        projections: vec![],
    }));
    let profiles = doc.state().sketches[&id]
        .profiles
        .iter()
        .filter(|p| p.contains([-47., -5.2]) || p.contains([38., 0.]))
        .map(profile_ref)
        .collect::<Vec<_>>();
    assert_eq!(profiles.len(), 2);
    doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: id,
        profiles,
        faces: vec![],
        extent: Extent::OneSide { d: "10 mm".into() },
        op: Operation::Join,
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let body = &doc.state().bodies[0];
    let edges = (1..=body.shape.edge_count() as u32)
        .filter(|&i| {
            let e = body.shape.edge_info(i).unwrap();
            matches!(e.kind, EdgeKind::Line)
                && (e.start[1] - 20.).abs() < 1e-6
                && (e.end[1] - 20.).abs() < 1e-6
                && ((e.mid[0] + 70.).abs() < 1e-6 || e.mid[2].abs() > 25.)
        })
        .map(|i| doc.state().edge_ref(body.id, i).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(edges.len(), 3);
    doc.add_feature(FeatureKind::Chamfer(ChamferFeature {
        edges,
        kind: ChamferKind::Equal { d: "3 mm".into() },
    }));
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    let shape = doc.state().bodies[0].shape.clone();
    assert!(shape.is_valid() && shape.is_solid());
    shape
}

#[test]
fn production_shell_worker_preserves_names_and_handles_chamfered_four_circle_plate() {
    kernel::set_shell_executor(isolated).unwrap();
    let cube = Shape::boxed(20., 20., 20.)
        .unwrap()
        .with_face_tags(&[101, 102, 103, 104, 105, 106])
        .unwrap();
    let before = cube.to_brep().unwrap();
    for (faces, thickness) in [(vec![], -1.), (vec![1], -1.), (vec![], 1.), (vec![1], 1.)] {
        let result = cube.shell(&faces, thickness, 77).unwrap();
        assert!(result.is_valid() && result.is_solid());
        assert!(result.face_tags().iter().all(|&t| t != 0));
        if faces.is_empty() {
            let expected = if thickness < 0. {
                8000. - 18f64.powi(3)
            } else {
                22f64.powi(3) - 8000.
            };
            assert!((result.volume() - expected).abs() < 1e-5);
        }
    }
    let repeat = cube.shell(&[], -1., 77).unwrap();
    assert_eq!(
        repeat.face_tags(),
        cube.shell(&[], -1., 77).unwrap().face_tags()
    );
    assert_eq!(cube.to_brep().unwrap(), before);

    for source in [plate(), overlapping_chamfered_plate()] {
        let mut saved = Document::new();
        saved
            .add_import("chamfered plate with four circles", &source)
            .unwrap();
        let original = saved.to_json();
        let mut preview = saved.clone();
        let body = preview.state().bodies[0].id;
        let id = preview.add_feature(FeatureKind::Shell(ShellFeature {
            body,
            body_tag: None,
            faces: vec![],
            thickness: "1 mm".into(),
            outward: false,
            outside_thickness: None,
        }));
        // Unsupported offsets are a recoverable feature error, never a process
        // crash or a modification to the original document used by the desktop.
        if preview.errors.contains_key(&id) {
            assert!((preview.state().bodies[0].shape.volume() - source.volume()).abs() < 1e-6);
        } else {
            assert!(preview.state().bodies[0].shape.is_valid());
        }
        assert_eq!(saved.to_json(), original);
        assert!(cube.shell(&[], -1., 78).unwrap().is_valid());
    }
    for faces in [vec![], vec![1]] {
        let result = cube.shell_both(&faces, 1., 2., 79).unwrap();
        assert!(result.is_valid() && result.is_solid());
    }
    let mut document = Document::new();
    document.add_import("stock", &cube).unwrap();
    document.add_feature(FeatureKind::Shell(ShellFeature {
        body: document.state().bodies[0].id,
        body_tag: None,
        faces: vec![],
        thickness: "1 mm".into(),
        outward: false,
        outside_thickness: None,
    }));
    assert!(document.errors.is_empty());
    let loaded = Document::from_json(&document.to_json()).unwrap();
    assert!(loaded.errors.is_empty());
    assert!((loaded.state().bodies[0].shape.volume() - (8000. - 18f64.powi(3))).abs() < 1e-5);
    assert_eq!(
        loaded.state().bodies[0].shape.face_tags(),
        document.state().bodies[0].shape.face_tags()
    );
}
