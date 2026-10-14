//! Assert coverage of the currently supported saved feature and input variants.
use refusion_model::*;
use serde_json::Value;
use std::collections::BTreeSet;
const FIXTURE: &str = include_str!("../../../examples/all-features.rfz");
fn enum_name(value: &Value) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| value.as_object().unwrap().keys().next().unwrap())
}
fn assert_covers(actual: &BTreeSet<String>, expected: &[&str]) {
    for name in expected {
        assert!(actual.contains(*name), "missing {name}; found {actual:?}");
    }
}
fn extent_modes(extent: &Value, modes: &mut BTreeSet<String>, directions: &mut BTreeSet<String>) {
    let kind = enum_name(extent);
    modes.insert(kind.into());
    if kind == "Tapered" {
        extent_modes(&extent[kind]["extent"], modes, directions);
    }
    if kind == "ThroughAll" {
        directions.insert(extent[kind]["direction"].as_str().unwrap().into());
    }
}
#[test]
fn gallery_contains_every_saved_feature_geometry_constraint_and_input_variant() {
    let value: Value = serde_json::from_str(FIXTURE).unwrap();
    let holes: Vec<_> = value["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["kind"].get("Hole"))
        .collect();
    assert_covers(
        &holes.iter().map(|h| enum_name(&h["kind"]).into()).collect(),
        &["Simple", "Counterbore", "Countersink"],
    );
    assert_covers(
        &holes
            .iter()
            .map(|h| enum_name(&h["extent"]).into())
            .collect(),
        &["Distance", "ThroughAll", "ToFace"],
    );
    assert!(
        holes.iter().any(|h| h["tip_angle"].is_null())
            && holes.iter().any(|h| h["tip_angle"].is_string())
    );
    assert!(holes.iter().any(|h| h["flip"] == true) && holes.iter().any(|h| h["flip"] == false));
    let both: Vec<_> = value["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["kind"].get("Shell"))
        .filter(|s| s["outside_thickness"].is_string())
        .collect();
    assert!(
        both.iter()
            .any(|s| s["faces"].as_array().unwrap().is_empty())
            && both
                .iter()
                .any(|s| !s["faces"].as_array().unwrap().is_empty())
    );

    let mut features = BTreeSet::new();
    let mut extents = BTreeSet::new();
    let mut all_directions = BTreeSet::new();
    let mut operations = BTreeSet::new();
    let mut chamfers = BTreeSet::new();
    let mut axes = BTreeSet::new();
    let mut planes = BTreeSet::new();
    let mut geometry = BTreeSet::new();
    let mut constraints = BTreeSet::new();
    let mut dimension_directions = BTreeSet::new();
    let mut revolve_directions = BTreeSet::new();
    let mut shell_modes = BTreeSet::new();
    let mut pattern_modes = BTreeSet::new();
    let mut pattern_objects = BTreeSet::new();
    let mut mirror_modes = BTreeSet::new();
    let mut full_revolve = BTreeSet::new();
    let mut project_axis = BTreeSet::new();
    let mut has_linked = false;
    let mut has_unlinked = false;
    let mut has_driven = false;
    let mut has_expression = false;
    let mut has_construction = false;
    let mut has_face_extrude = false;
    let mut has_face_revolve = false;
    for feature in value["features"].as_array().unwrap() {
        assert_eq!(feature["suppressed"], false);
        let kind = enum_name(&feature["kind"]);
        features.insert(kind.into());
        let data = &feature["kind"][kind];
        if let Some(op) = data["op"].as_str() {
            operations.insert(op.into());
        }
        match kind {
            "Sketch" => {
                planes.insert(enum_name(&data["plane"]).into());
                has_linked |= !data["projections"].as_array().unwrap().is_empty();
                let unlinked = data["projections"].as_array().unwrap().is_empty();
                for entity in data["sketch"]["entities"].as_object().unwrap().values() {
                    geometry.insert(enum_name(&entity["geom"]).into());
                    has_construction |= entity["construction"] == true;
                    has_unlinked |= unlinked && entity["projected"] == true;
                }
                for rec in data["sketch"]["constraints"].as_object().unwrap().values() {
                    let kind = enum_name(&rec["c"]);
                    constraints.insert(kind.into());
                    if kind == "Distance" {
                        dimension_directions
                            .insert(rec["c"][kind]["kind"].as_str().unwrap().into());
                    }
                    has_driven |= rec["driven"] == true;
                    has_expression |= rec["expr"].is_string();
                }
            }
            "Extrude" => {
                extent_modes(&data["extent"], &mut extents, &mut all_directions);
                has_face_extrude |= !data["faces"].as_array().unwrap().is_empty();
            }
            "Revolve" => {
                axes.insert(enum_name(&data["axis"]).into());
                revolve_directions.insert(data["direction"].as_str().unwrap().into());
                full_revolve.insert(data["full"].as_bool().unwrap());
                project_axis.insert(data["project_axis"].as_bool().unwrap());
                has_face_revolve |= !data["faces"].as_array().unwrap().is_empty();
            }
            "Chamfer" => {
                chamfers.insert(enum_name(&data["kind"]).into());
            }
            "Shell" => {
                shell_modes.insert((
                    data["outward"].as_bool().unwrap(),
                    data["faces"].as_array().unwrap().is_empty(),
                ));
            }
            "Mirror" => {
                mirror_modes.insert(data["join"].as_bool().unwrap());
            }
            "Pattern" => {
                pattern_objects.insert(data["objects"].as_str().unwrap().to_string());
                let kind = enum_name(&data["kind"]);
                let data = &data["kind"][kind];
                pattern_modes.insert(format!(
                    "{kind}:{}",
                    if kind == "Rectangular" {
                        data["spacing"].as_bool().unwrap().to_string()
                    } else {
                        data["angle"].as_str().unwrap().into()
                    }
                ));
            }
            _ => {}
        }
    }
    assert_covers(
        &features,
        &[
            "Sketch",
            "Extrude",
            "Revolve",
            "Fillet",
            "Chamfer",
            "Hole",
            "Import",
            "OffsetPlane",
            "AngledPlane",
            "Shell",
            "Mirror",
            "Pattern",
        ],
    );
    assert_covers(
        &extents,
        &[
            "OneSide",
            "Symmetric",
            "SymmetricHalf",
            "TwoSides",
            "ThroughAll",
            "ToFace",
            "Tapered",
        ],
    );
    assert_covers(&all_directions, &["Positive", "Negative", "Both"]);
    assert_covers(&operations, &["NewBody", "Join", "Cut", "Intersect"]);
    assert_covers(&chamfers, &["Equal", "TwoDistances", "DistanceAngle"]);
    assert_covers(&axes, &["X", "Y", "Z", "SketchLine", "Edge"]);
    assert_covers(
        &planes,
        &["XY", "XZ", "YZ", "Face", "Construction", "Fixed"],
    );
    assert_covers(&geometry, &["Point", "Line", "Circle", "Arc"]);
    assert_covers(
        &constraints,
        &[
            "Coincident",
            "Horizontal",
            "Vertical",
            "HorizontalPts",
            "VerticalPts",
            "Parallel",
            "Perpendicular",
            "Tangent",
            "Equal",
            "Concentric",
            "Collinear",
            "Midpoint",
            "Symmetric",
            "Fix",
            "Offset",
            "Distance",
            "Length",
            "Radius",
            "Diameter",
            "Angle",
        ],
    );
    assert_covers(
        &dimension_directions,
        &["Aligned", "Horizontal", "Vertical"],
    );
    assert_covers(&revolve_directions, &["OneSide", "TwoSides", "Symmetric"]);
    assert_covers(
        &pattern_modes,
        &[
            "Rectangular:true",
            "Rectangular:false",
            "Circular:360 deg",
            "Circular:150 deg",
        ],
    );
    assert_covers(&pattern_objects, &["Bodies", "Features"]);
    assert_eq!(shell_modes.len(), 4);
    assert_eq!(mirror_modes.len(), 2);
    assert_eq!(full_revolve.len(), 2);
    assert_eq!(project_axis.len(), 2);
    assert!(
        has_linked
            && has_unlinked
            && has_driven
            && has_expression
            && has_construction
            && has_face_extrude
            && has_face_revolve
    );
}
fn healthy(doc: &Document) {
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    for body in &doc.state().bodies {
        assert!(
            body.shape.is_valid() && body.shape.is_solid(),
            "invalid body {}",
            body.id
        );
        assert!(body.shape.volume().is_finite() && body.shape.volume() > 0.);
    }
    for feature in &doc.features {
        if let FeatureKind::Sketch(sf) = &feature.kind {
            let report = sf.sketch.clone().solve();
            assert!(
                report.converged && report.max_residual < 1e-7,
                "{}: {report:?}",
                feature.name
            );
        }
    }
}
#[test]
fn gallery_loads_replays_rolls_timeline_and_meshes_every_body() {
    let mut doc = Document::from_json(FIXTURE).unwrap();
    healthy(&doc);
    assert_eq!(doc.features.len(), 115);
    assert_eq!(doc.state().bodies.len(), 67);
    assert_eq!(doc.state().planes.len(), 8);
    assert_eq!(doc.state().sketches.len(), 30);
    assert_eq!(doc.marker, doc.features.len());
    let volumes: Vec<_> = doc
        .state()
        .bodies
        .iter()
        .map(|b| b.shape.volume())
        .collect();
    for body in &doc.state().bodies {
        let mesh = body.shape.mesh(0.15, 0.25).unwrap();
        assert!(
            !mesh.indices.is_empty(),
            "body {} has no render triangles",
            body.id
        );
    }
    let roundtrip = Document::from_json(&doc.to_json()).unwrap();
    healthy(&roundtrip);
    for (body, volume) in roundtrip.state().bodies.iter().zip(&volumes) {
        assert!((body.shape.volume() - volume).abs() < 1e-6);
    }
    for marker in 0..=doc.features.len() {
        doc.set_marker(marker);
        assert!(doc.errors.is_empty(), "marker {marker}: {:?}", doc.errors);
        assert_eq!(doc.marker, marker);
        healthy(&doc);
    }
    healthy(&doc);
    assert_eq!(doc.state().bodies.len(), volumes.len());
    for (body, volume) in doc.state().bodies.iter().zip(volumes) {
        assert!((body.shape.volume() - volume).abs() < 1e-6);
    }
}
fn projected_points(doc: &Document, name: &str) -> Vec<[f64; 2]> {
    let feature = doc.features.iter().find(|f| f.name == name).unwrap();
    let FeatureKind::Sketch(sf) = &feature.kind else {
        panic!()
    };
    sf.sketch
        .entities
        .iter()
        .filter(|(_, e)| e.projected && matches!(e.geom, sketch::Geom::Point(_)))
        .map(|(&id, _)| sf.sketch.pos(id))
        .collect()
}
#[test]
fn gallery_parameter_edits_preserve_history_and_projection_associativity() {
    let mut doc = Document::from_json(FIXTURE).unwrap();
    let linked = "04 Plate / face sketch with projection and holes";
    let unlinked = "Projection / unlinked fixed reference geometry";
    let original_linked = projected_points(&doc, linked);
    let original_unlinked = projected_points(&doc, unlinked);
    let bodies = doc.state().bodies.len();
    for (name, expr) in [
        ("plate_width", "72 mm"),
        ("plate_height", "10 mm"),
        ("hole_diameter", "8 mm"),
        ("wall", "1.2 mm"),
        ("station_depth", "10 mm"),
        ("plane_lift", "16 mm"),
        ("copy_count", "4"),
    ] {
        doc.params.iter_mut().find(|p| p.name == name).unwrap().expr = expr.into();
    }
    doc.recompute_all();
    healthy(&doc);
    assert_eq!(doc.state().bodies.len(), bodies + 4);
    assert_ne!(projected_points(&doc, linked), original_linked);
    assert_eq!(projected_points(&doc, unlinked), original_unlinked);
    healthy(&Document::from_json(&doc.to_json()).unwrap());
}

const COMBINED: &str = include_str!("../../../examples/all-features-combined.rfz");
fn mesh_all(doc: &Document) {
    for body in &doc.state().bodies {
        let mesh = body.shape.mesh(0.15, 0.25).unwrap();
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len(), mesh.tri_face.len() * 3);
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert!(mesh
            .positions
            .iter()
            .chain(&mesh.normals)
            .flatten()
            .all(|v| v.is_finite()));
        assert!(mesh
            .indices
            .iter()
            .all(|&i| (i as usize) < mesh.positions.len()));
        assert!(mesh
            .tri_face
            .iter()
            .all(|&i| i > 0 && i <= body.shape.face_count() as u32));
    }
}
#[test]
fn both_fixtures_mesh_at_every_timeline_step() {
    for fixture in [FIXTURE, COMBINED] {
        let mut doc = Document::from_json(fixture).unwrap();
        for marker in 0..=doc.features.len() {
            doc.set_marker(marker);
            healthy(&doc);
            mesh_all(&doc);
        }
    }
}
#[test]
fn combined_covers_feature_families_and_preserves_one_body_until_patterns() {
    let json: Value = serde_json::from_str(COMBINED).unwrap();
    let kinds = json["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| enum_name(&f["kind"]).to_owned())
        .collect();
    assert_covers(
        &kinds,
        &[
            "Sketch",
            "Extrude",
            "Revolve",
            "Fillet",
            "Chamfer",
            "Hole",
            "Import",
            "OffsetPlane",
            "AngledPlane",
            "Shell",
            "Mirror",
            "Pattern",
        ],
    );
    let mut doc = Document::from_json(COMBINED).unwrap();
    let body_id = doc.state().bodies[0].id;
    let mut previous = 0.;
    for marker in 1..=doc.features.len() {
        doc.set_marker(marker);
        healthy(&doc);
        if marker <= 21 {
            assert_eq!(doc.state().bodies.len(), 1, "marker {marker}");
            assert_eq!(doc.state().bodies[0].id, body_id);
        }
        let volume = doc.state().body(body_id).unwrap().shape.volume();
        if matches!(marker, 2 | 3 | 5 | 6 | 7 | 9 | 11 | 14 | 17 | 19) {
            assert!(
                (volume - previous).abs() > 1e-5,
                "feature {marker} must change the evolving body"
            );
        }
        previous = volume;
    }
    assert_eq!(doc.state().bodies.len(), 3);
    let volumes: Vec<_> = doc
        .state()
        .bodies
        .iter()
        .map(|b| b.shape.volume())
        .collect();
    let roundtrip = Document::from_json(&doc.to_json()).unwrap();
    healthy(&roundtrip);
    for (body, volume) in roundtrip.state().bodies.iter().zip(&volumes) {
        assert!((body.shape.volume() - volume).abs() < 1e-6);
    }
    for (name, expr) in [
        ("wall", "1.2 mm"),
        ("boss_height", "17 mm"),
        ("hole_diameter", "4 mm"),
        ("copy_count", "3"),
    ] {
        doc.params.iter_mut().find(|p| p.name == name).unwrap().expr = expr.into();
    }
    doc.recompute_all();
    healthy(&doc);
    mesh_all(&doc);
    assert_eq!(doc.state().bodies.len(), 4);
    assert!((doc.state().body(body_id).unwrap().shape.volume() - volumes[0]).abs() > 1.);
    healthy(&Document::from_json(&doc.to_json()).unwrap());
}
