//! Structural validation at the persistence boundary, before native replay.
use super::*;
use std::collections::HashSet;

fn finite(values: &[f64]) -> Result<(), String> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err("Non-finite reference coordinates".into())
    }
}
fn face(r: &FaceRef) -> Result<(), String> {
    finite(&r.hint)?;
    if r.body == 0 {
        Err("Invalid body reference".into())
    } else {
        Ok(())
    }
}
fn edge(r: &EdgeRef) -> Result<(), String> {
    finite(&r.hint)?;
    if r.body == 0 {
        Err("Invalid body reference".into())
    } else {
        Ok(())
    }
}
fn linear(r: &LinearRef) -> Result<(), String> {
    if let LinearRef::Edge(r) = r {
        edge(r)?;
    }
    Ok(())
}
fn plane(p: &PlaneRef, prior: &HashMap<FeatureId, &FeatureKind>) -> Result<(), String> {
    match p {
        PlaneRef::Face(r) => face(r)?,
        PlaneRef::Construction(id) => {
            if prior.get(id).is_some_and(|kind| {
                !matches!(
                    kind,
                    FeatureKind::OffsetPlane(_) | FeatureKind::AngledPlane(_)
                )
            }) {
                return Err("Construction plane must reference an earlier plane feature".into());
            }
        }
        PlaneRef::Fixed { origin, x, y } => {
            finite(origin)?;
            finite(x)?;
            finite(y)?;
            if (k::dot(*x, *x) - 1.).abs() > 1e-7
                || (k::dot(*y, *y) - 1.).abs() > 1e-7
                || k::dot(*x, *y).abs() > 1e-7
            {
                return Err("Sketch plane axes must be orthonormal".into());
            }
        }
        _ => {}
    }
    Ok(())
}
fn profiles(
    id: FeatureId,
    refs: &[ProfileRef],
    prior: &HashMap<FeatureId, &FeatureKind>,
) -> Result<(), String> {
    if refs.is_empty() {
        return Ok(());
    }
    if prior
        .get(&id)
        .is_some_and(|kind| !matches!(kind, FeatureKind::Sketch(_)))
    {
        return Err("Profiles must reference a sketch".into());
    }
    for r in refs {
        finite(&r.hint)?;
        // Persistent IDs may be stale after edits; matching deliberately uses
        // the interior hint/curve overlap. Missing upstream features are replay
        // diagnostics, whereas invalid internal sketch indices are unsafe.
    }
    Ok(())
}
fn extent(x: &Extent) -> Result<(), String> {
    match x {
        Extent::ToFace { face: r, .. } => face(r),
        Extent::Tapered { extent: x, .. } => extent(x),
        _ => Ok(()),
    }
}

impl Document {
    pub(super) fn validate_loaded(&self) -> Result<(), String> {
        if self.format_version > 1 {
            return Err(format!(
                "Unsupported design format version {}",
                self.format_version
            ));
        }
        if self.marker > self.features.len() {
            return Err("Timeline marker exceeds the feature count".into());
        }
        let mut ids = HashSet::new();
        for f in &self.features {
            if f.id == 0 || !ids.insert(f.id) {
                return Err("Feature IDs must be nonzero and unique".into());
            }
        }
        if self.next_feature < self.features.iter().map(|f| f.id).max().unwrap_or(0)
            || self.next_feature == u32::MAX
        {
            return Err("Invalid feature ID allocation counter".into());
        }
        let mut names = HashSet::new();
        for p in &self.params {
            if p.name.is_empty()
                || !names.insert(&p.name)
                || !p.name.chars().enumerate().all(|(i, c)| {
                    c == '_'
                        || if i == 0 {
                            c.is_alphabetic()
                        } else {
                            c.is_alphanumeric()
                        }
                })
            {
                return Err("Parameter names must be valid, unique identifiers".into());
            }
        }
        let mut prior = HashMap::new();
        for f in &self.features {
            let validate = || -> Result<(), String> {
                match &f.kind {
                    FeatureKind::Sketch(sf) => {
                        sf.sketch.validate_data()?;
                        plane(&sf.plane, &prior)?;
                        for p in &sf.projections {
                            edge(&p.edge)?;
                            if p.entities.is_empty()
                                || p.entities.iter().any(|id| sf.sketch.entity(*id).is_none())
                            {
                                return Err("Projection references missing sketch entities".into());
                            }
                        }
                    }
                    FeatureKind::OffsetPlane(p) => plane(&p.base, &prior)?,
                    FeatureKind::AngledPlane(p) => {
                        plane(&p.base, &prior)?;
                        linear(&p.axis)?;
                    }
                    FeatureKind::Extrude(x) => {
                        profiles(x.sketch, &x.profiles, &prior)?;
                        for r in &x.faces {
                            face(r)?;
                        }
                        extent(&x.extent)?;
                    }
                    FeatureKind::Revolve(x) => {
                        profiles(x.sketch, &x.profiles, &prior)?;
                        for r in &x.faces {
                            face(r)?;
                        }
                        match &x.axis {
                            Axis::Edge(r) => edge(r)?,
                            Axis::SketchLine(id) => {
                                if let Some(FeatureKind::Sketch(sf)) = prior.get(&x.sketch) {
                                    if sf.sketch.entity(*id).is_some()
                                        && sf.sketch.line_pts(*id).is_none()
                                    {
                                        return Err(
                                            "Revolve axis references a non-line entity".into()
                                        );
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    FeatureKind::Fillet(x) => {
                        for r in &x.edges {
                            edge(r)?;
                        }
                    }
                    FeatureKind::Chamfer(x) => {
                        for r in &x.edges {
                            edge(r)?;
                        }
                    }
                    FeatureKind::Shell(x) => {
                        if x.body == 0 {
                            return Err("Invalid shell body reference".into());
                        }
                        for r in &x.faces {
                            face(r)?;
                        }
                    }
                    FeatureKind::Hole(x) => {
                        if x.body == 0 {
                            return Err("Invalid hole body reference".into());
                        }
                        plane(&x.plane, &prior)?;
                        if let HoleExtent::ToFace { face: r, .. } = &x.extent {
                            face(r)?;
                        }
                    }
                    FeatureKind::Mirror(x) => {
                        plane(&x.plane, &prior)?;
                        if x.bodies.contains(&0) {
                            return Err("Invalid mirror body reference".into());
                        }
                    }
                    FeatureKind::Pattern(x) => {
                        if x.features.contains(&0) || x.features.contains(&f.id) {
                            return Err("Invalid pattern feature reference".into());
                        }
                        if x.objects == PatternObjects::Features && !x.bodies.is_empty()
                            || x.objects == PatternObjects::Bodies && !x.features.is_empty()
                        {
                            return Err(
                                "Pattern objects must match the selected object type".into()
                            );
                        }
                        if x.bodies.contains(&0) {
                            return Err("Invalid pattern body reference".into());
                        }
                        match &x.kind {
                            PatternKind::Rectangular { axis1, axis2, .. } => {
                                linear(axis1)?;
                                linear(axis2)?;
                            }
                            PatternKind::Circular { axis, .. } => linear(axis)?,
                        }
                    }
                    FeatureKind::Import(_) => {}
                }
                Ok(())
            };
            validate().map_err(|e| format!("Feature {} ({}): {e}", f.id, f.name))?;
            prior.insert(f.id, &f.kind);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn saved() -> serde_json::Value {
        let mut doc = Document::new();
        let mut sketch = Sketch::new();
        sketch.add_rectangle([0., 0.], [10., 10.]);
        doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch,
            projections: vec![],
        }));
        serde_json::from_str(&doc.to_json()).unwrap()
    }
    #[test]
    fn malformed_documents_fail_before_replay() {
        for (field, value) in [("marker", 999), ("next_feature", 0), ("format_version", 2)] {
            let mut data = saved();
            data[field] = value.into();
            assert!(Document::from_json(&data.to_string()).is_err(), "{field}");
        }
        let mut data = saved();
        let duplicate = data["features"][0].clone();
        data["features"].as_array_mut().unwrap().push(duplicate);
        assert!(Document::from_json(&data.to_string()).is_err());
        let mut data = saved();
        data["features"][0]["kind"]["Sketch"]["sketch"]["entities"]["0"]["geom"]["Point"] =
            serde_json::json!([99999, 1]);
        assert!(Document::from_json(&data.to_string()).is_err());
        let mut data = saved();
        data["features"][0]["kind"]["Sketch"]["plane"] =
            serde_json::json!({"Fixed":{"origin":[0,0,0],"x":[2,0,0],"y":[0,1,0]}});
        assert!(Document::from_json(&data.to_string()).is_err());
    }
    #[test]
    fn legacy_files_migrate_and_current_files_roundtrip() {
        let mut data = saved();
        data.as_object_mut().unwrap().remove("format_version");
        let doc = Document::from_json(&data.to_string()).unwrap();
        assert_eq!(doc.format_version, 1);
        let loaded = Document::from_json(&doc.to_json()).unwrap();
        assert_eq!(loaded.features[0].id, doc.features[0].id);
    }

    #[test]
    fn parameter_units_survive_dependencies_and_invalid_parameters_are_reported() {
        let mut doc = Document::new();
        doc.params = vec![
            Param {
                name: "angle".into(),
                expr: "30 deg".into(),
                comment: "".into(),
            },
            Param {
                name: "double".into(),
                expr: "angle * 2".into(),
                comment: "".into(),
            },
        ];
        assert!((doc.eval("double", Unit::Angle).unwrap().to_degrees() - 60.).abs() < 1e-10);
        assert!(doc.eval("double", Unit::Length).is_err());
        doc.add_import("cube", &Shape::boxed(1., 1., 1.).unwrap())
            .unwrap();
        let uid = doc.state().bodies[0].shape.uid();
        doc.params[0].expr = "unknown + 1".into();
        doc.recompute_all();
        assert!(doc.parameter_error.as_ref().unwrap().contains("angle"));
        assert!(!doc.errors.is_empty());
        assert_eq!(doc.state().bodies[0].shape.uid(), uid);
    }

    #[test]
    fn persistent_reference_fallback_and_broken_timeline_diagnostics_survive_loading() {
        let mut doc = Document::from_json(&saved().to_string()).unwrap();
        let sketch = doc.features[0].id;
        let mut reference = profile_ref(&doc.state().sketches[&sketch].profiles[0]);
        reference.curves[0] = EntityId(9999); // a stale persistent name after editing
        let extrude = doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch,
            profiles: vec![reference],
            faces: vec![],
            extent: Extent::OneSide { d: "5 mm".into() },
            op: Operation::NewBody,
        }));
        assert!(Document::from_json(&doc.to_json())
            .unwrap()
            .errors
            .is_empty());
        doc.remove_feature(sketch);
        let loaded = Document::from_json(&doc.to_json()).unwrap();
        assert!(loaded.errors.contains_key(&extrude));
    }
}
