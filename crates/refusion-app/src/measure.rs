use crate::scene::Pick;
use refusion_model::kernel::{self as k, EdgeKind, Shape, Sub};
use refusion_model::sketch::Geom;
use refusion_model::{Document, FeatureKind, State};

#[derive(Clone, Default)]
pub struct MeasureCmd {
    pub a: Option<Pick>,
    pub b: Option<Pick>,
    pub precision: usize,
    pub result: Option<MeasureResult>,
}

#[derive(Clone, Debug, Default)]
pub struct MeasureResult {
    pub name_a: String,
    pub name_b: String,
    /// distance, closest point on a, closest point on b
    pub distance: Option<(f64, [f64; 3], [f64; 3])>,
    /// Acute angle in radians between planar/linear selections.
    pub angle: Option<f64>,
    pub info: Vec<(String, String)>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum Geo {
    Plane([f64; 3]),
    Line([f64; 3]),
    Point,
    Other,
}

struct Target {
    shape: Shape,
    sub: Sub,
    geo: Geo,
    name: String,
    info: Vec<(String, f64, &'static str)>,
}

fn target(doc: &Document, state: &State, p: Pick) -> Option<Target> {
    match p {
        Pick::Face { body, face } => {
            let b = state.body(body)?;
            let fi = b.shape.face_info(face)?;
            Some(Target {
                shape: b.shape.clone(),
                sub: Sub::Face(face),
                geo: fi
                    .plane
                    .map(|pl| Geo::Plane(pl.normal()))
                    .unwrap_or(Geo::Other),
                name: format!("{} face", b.name),
                info: vec![("Area".into(), fi.area, "mm²")],
            })
        }
        Pick::Edge { body, edge } => {
            let b = state.body(body)?;
            let ei = b.shape.edge_info(edge)?;
            let mut info = vec![("Length".into(), ei.length, "mm")];
            if let EdgeKind::Circle { radius, .. } = ei.kind {
                info.push(("Radius".into(), radius, "mm"));
            }
            Some(Target {
                shape: b.shape.clone(),
                sub: Sub::Edge(edge),
                geo: if ei.kind == EdgeKind::Line {
                    Geo::Line(ei.tangent)
                } else {
                    Geo::Other
                },
                name: format!("{} edge", b.name),
                info,
            })
        }
        Pick::Vertex { body, vertex } => {
            let b = state.body(body)?;
            let pos = b.shape.vertex_pos(vertex)?;
            Some(Target {
                shape: b.shape.clone(),
                sub: Sub::Vertex(vertex),
                geo: Geo::Point,
                name: format!("{} vertex", b.name),
                info: pos_info(pos),
            })
        }
        Pick::Ent { sketch, ent } => {
            let FeatureKind::Sketch(sf) = &doc.feature(sketch)?.kind else {
                return None;
            };
            let frame = state.sketches.get(&sketch)?.frame;
            let sk = &sf.sketch;
            match sk.entity(ent)?.geom {
                Geom::Point(_) => {
                    let w = frame.to_world(sk.pos(ent));
                    Some(Target {
                        shape: Shape::vertex(w).ok()?,
                        sub: Sub::Whole,
                        geo: Geo::Point,
                        name: "sketch point".into(),
                        info: pos_info(w),
                    })
                }
                Geom::Line { a, b } => {
                    let (pa, pb) = (frame.to_world(sk.pos(a)), frame.to_world(sk.pos(b)));
                    let d = k::sub(pb, pa);
                    let len = k::dot(d, d).sqrt();
                    Some(Target {
                        shape: Shape::segment(pa, pb).ok()?,
                        sub: Sub::Whole,
                        geo: Geo::Line([d[0] / len, d[1] / len, d[2] / len]),
                        name: "sketch line".into(),
                        info: vec![("Length".into(), len, "mm")],
                    })
                }
                _ => {
                    // circles / arcs: measure to the centre point
                    let c = frame.to_world(sk.pos(sk.center(ent)?));
                    let mut info = pos_info(c);
                    info.insert(0, ("Radius".into(), sk.radius(ent)?, "mm"));
                    Some(Target {
                        shape: Shape::vertex(c).ok()?,
                        sub: Sub::Whole,
                        geo: Geo::Point,
                        name: "sketch circle centre".into(),
                        info,
                    })
                }
            }
        }
        _ => None,
    }
}

fn pos_info(p: [f64; 3]) -> Vec<(String, f64, &'static str)> {
    vec![
        ("X".into(), p[0], "mm"),
        ("Y".into(), p[1], "mm"),
        ("Z".into(), p[2], "mm"),
    ]
}

fn angle(a: Geo, b: Geo) -> Option<f64> {
    let acute = |c: f64| c.abs().min(1.0).acos();
    match (a, b) {
        (Geo::Plane(n1), Geo::Plane(n2)) | (Geo::Line(n1), Geo::Line(n2)) => {
            Some(acute(k::dot(n1, n2)))
        }
        (Geo::Line(d), Geo::Plane(n)) | (Geo::Plane(n), Geo::Line(d)) => {
            Some(std::f64::consts::FRAC_PI_2 - acute(k::dot(d, n)))
        }
        _ => None,
    }
}

pub fn compute(
    doc: &Document,
    state: &State,
    a: Option<Pick>,
    b: Option<Pick>,
    precision: usize,
) -> MeasureResult {
    let fmt = |v: f64| format!("{v:.precision$}");
    let mut r = MeasureResult::default();
    let ta = a.and_then(|p| target(doc, state, p));
    let tb = b.and_then(|p| target(doc, state, p));
    match (ta, tb) {
        (Some(t), None) => {
            r.name_a = t.name;
            r.info = t
                .info
                .into_iter()
                .map(|(n, v, u)| (n, format!("{} {u}", fmt(v))))
                .collect();
        }
        (Some(ta), Some(tb)) => {
            r.name_a = ta.name.clone();
            r.name_b = tb.name.clone();
            match k::min_distance(&ta.shape, ta.sub, &tb.shape, tb.sub) {
                Ok(d) => r.distance = Some(d),
                Err(e) => r.error = Some(e.0),
            }
            r.angle = angle(ta.geo, tb.geo);
            if let Some((d, p1, p2)) = r.distance {
                r.info.push(("Distance".into(), format!("{} mm", fmt(d))));
                if let Some(a) = r.angle {
                    r.info
                        .push(("Angle".into(), format!("{}°", fmt(a.to_degrees()))));
                }
                for (i, axis) in ["ΔX", "ΔY", "ΔZ"].iter().enumerate() {
                    r.info.push((
                        axis.to_string(),
                        format!("{} mm", fmt((p2[i] - p1[i]).abs())),
                    ));
                }
                r.info.push((
                    "Point 1".into(),
                    format!("{}, {}, {}", fmt(p1[0]), fmt(p1[1]), fmt(p1[2])),
                ));
                r.info.push((
                    "Point 2".into(),
                    format!("{}, {}, {}", fmt(p2[0]), fmt(p2[1]), fmt(p2[2])),
                ));
            }
        }
        _ => {}
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn angle_rules() {
        let z = Geo::Plane([0.0, 0.0, 1.0]);
        let x = Geo::Plane([1.0, 0.0, 0.0]);
        let lz = Geo::Line([0.0, 0.0, -1.0]);
        assert!((angle(z, x).unwrap().to_degrees() - 90.0).abs() < 1e-9);
        assert!(angle(z, Geo::Plane([0.0, 0.0, -1.0])).unwrap().abs() < 1e-9);
        assert!((angle(lz, z).unwrap().to_degrees() - 90.0).abs() < 1e-9);
        assert!(angle(lz, x).unwrap().abs() < 1e-9);
    }
}
