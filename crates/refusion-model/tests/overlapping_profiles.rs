use refusion_model::*;

#[test]
fn overlapping_fixture_regions_reidentify_and_extrude_independently() {
    let original = Document::from_json(include_str!("fixtures/intersect.rfz")).unwrap();
    let profiles = &original.state().sketches[&1].profiles;
    assert!(profiles.len() >= 3);
    for (index, profile) in profiles.iter().enumerate() {
        let reference = profile_ref(profile);
        let matched = match_profile(profiles, &reference).unwrap();

        assert!(
            (matched.area() - profile.area()).abs() < 1e-6,
            "region {index} resolves to a different area"
        );
        let mut doc = original.clone();
        let id = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch: 1,
            profiles: vec![reference],
            faces: vec![],
            extent: Extent::OneSide { d: "10 mm".into() },
            op: Operation::NewBody,
        }));
        assert!(!doc.errors.contains_key(&id), "{:?}", doc.errors);
        let body = &doc.state().bodies[0];
        assert!((body.shape.volume().abs() - exact_area(profile).abs() * 10.).abs() < 1e-5);
        let saved = Document::from_json(&doc.to_json()).unwrap();
        assert!(saved.errors.is_empty());
        assert!((saved.state().bodies[0].shape.volume() - body.shape.volume()).abs() < 1e-5);
    }
}

fn exact_area(profile: &sketch::Profile) -> f64 {
    std::iter::once(&profile.outer)
        .chain(&profile.holes)
        .flat_map(|lp| &lp.segs)
        .map(|segment| {
            let (a, b) = (segment.start(), segment.end());
            let chord = a[0] * b[1] - a[1] * b[0];
            let curved = match *segment {
                sketch::Seg::Arc { r, ccw, .. } => {
                    let angle = segment.sweep() * if ccw { 1. } else { -1. };
                    r * r * (angle - angle.sin())
                }
                _ => 0.,
            };
            (chord + curved) / 2.
        })
        .sum()
}

#[test]
fn overlapping_circle_regions_can_be_extruded_together() {
    let mut doc = Document::from_json(include_str!("fixtures/intersect.rfz")).unwrap();
    let profiles = &doc.state().sketches[&1].profiles;
    let selected: Vec<_> = profiles
        .iter()
        .filter(|p| p.signature().len() == 3)
        .map(profile_ref)
        .collect();
    assert_eq!(selected.len(), 2);
    let id = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
        sketch: 1,
        profiles: selected,
        faces: vec![],
        extent: Extent::OneSide { d: "10 mm".into() },
        op: Operation::NewBody,
    }));
    assert!(!doc.errors.contains_key(&id), "{:?}", doc.errors);
    let expected = std::f64::consts::PI * 44.88766545486154_f64.powi(2) * 10.;
    assert!((doc.state().bodies[0].shape.volume() - expected).abs() < 1e-5);
}
