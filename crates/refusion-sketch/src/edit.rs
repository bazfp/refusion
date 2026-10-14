//! Transactional sketch modifications. Boundary curves and shared points remain intact.
use crate::profile::{carrier_intersections, intersect, Curve};
use crate::{Constraint, EntityId, Geom, Sketch};
use std::f64::consts::TAU;
const EPS: f64 = 1e-7;
fn curve(sk: &Sketch, id: EntityId) -> Option<Curve> {
    Some(match sk.entity(id)?.geom {
        Geom::Line { a, b } => Curve::Line {
            a: sk.pos(a),
            b: sk.pos(b),
        },
        Geom::Circle { c, .. } => Curve::Arc {
            c: sk.pos(c),
            r: sk.radius(id)?,
            t0: 0.0,
            sweep: TAU,
        },
        Geom::Arc { c, s, e } => {
            let c = sk.pos(c);
            let s = sk.pos(s);
            let e = sk.pos(e);
            let t0 = (s[1] - c[1]).atan2(s[0] - c[0]);
            let sweep = ((e[1] - c[1]).atan2(e[0] - c[0]) - t0).rem_euclid(TAU);
            Curve::Arc {
                c,
                r: sk.radius(id)?,
                t0,
                sweep,
            }
        }
        _ => return None,
    })
}
fn parameter(c: Curve, p: [f64; 2]) -> f64 {
    match c {
        Curve::Line { a, b } => {
            ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1]))
                / ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2))
        }
        Curve::Arc { c, t0, sweep, .. } => {
            ((p[1] - c[1]).atan2(p[0] - c[0]) - t0).rem_euclid(TAU) / sweep
        }
    }
}
fn piece(sk: &mut Sketch, c: Curve, a: f64, b: f64, construction: bool) -> EntityId {
    let point = |sk: &mut Sketch, p: [f64; 2]| {
        sk.entities
            .iter()
            .find_map(|(&id, e)| match e.geom {
                Geom::Point(_) => {
                    let q = sk.pos(id);
                    ((p[0] - q[0]).hypot(p[1] - q[1]) < EPS).then_some(id)
                }
                _ => None,
            })
            .unwrap_or_else(|| sk.add_point(p))
    };
    let p = point(sk, c.at(a));
    let q = point(sk, c.at(b));
    let id = match c {
        Curve::Line { .. } => sk.add_line(p, q),
        Curve::Arc { c, .. } => {
            let center = point(sk, c);
            sk.add_arc(center, p, q)
        }
    };
    sk.entities.get_mut(&id).unwrap().construction = construction;
    id
}
fn trim_bounds(
    sk: &Sketch,
    id: EntityId,
    c: Curve,
    cursor: [f64; 2],
) -> Result<(f64, f64), String> {
    let mut ts: Vec<_> = sk
        .entities
        .keys()
        .filter(|&&e| e != id)
        .filter_map(|&e| curve(sk, e))
        .flat_map(|other| intersect(&c, &other))
        .map(|p| parameter(c, p))
        .collect();
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() < EPS);
    let t = parameter(c, cursor);
    if c.is_closed() {
        if ts.len() == 1 {
            return Err("A circle needs two distinct trim intersections".into());
        }
        if ts.is_empty() {
            return Ok((0.0, 1.0));
        }
        let upper = ts.iter().position(|&v| v > t).unwrap_or(0);
        let lo = ts[(upper + ts.len() - 1) % ts.len()];
        let hi = ts[upper];
        Ok((lo, if hi <= lo { hi + 1.0 } else { hi }))
    } else {
        ts.retain(|v| *v > EPS && *v < 1.0 - EPS);
        ts.insert(0, 0.0);
        ts.push(1.0);
        let v = ts
            .windows(2)
            .find(|v| t >= v[0] - EPS && t <= v[1] + EPS)
            .ok_or("Cursor is outside the curve")?;
        Ok((v[0], v[1]))
    }
}
fn extend_bounds(
    sk: &Sketch,
    id: EntityId,
    c: Curve,
    cursor: [f64; 2],
) -> Result<(f64, f64), String> {
    if c.is_closed() {
        return Err("A full circle has no endpoint to extend".into());
    }
    let d = |p: [f64; 2]| (p[0] - cursor[0]).hypot(p[1] - cursor[1]);
    let start = d(c.at(0.0)) < d(c.at(1.0));
    let mut candidates: Vec<_> = sk
        .entities
        .keys()
        .filter(|&&e| e != id)
        .filter_map(|&e| curve(sk, e))
        .flat_map(|other| {
            carrier_intersections(&c, &other)
                .into_iter()
                .filter(move |&p| other.param_of(p).is_some())
        })
        .map(|p| parameter(c, p))
        .collect();
    if start {
        if let Curve::Arc { sweep, .. } = c {
            for v in &mut candidates {
                *v -= TAU / sweep;
            }
        }
    }
    candidates.retain(|v| if start { *v < -EPS } else { *v > 1.0 + EPS });
    let t = candidates
        .into_iter()
        .min_by(|a, b| {
            if start {
                b.total_cmp(a)
            } else {
                a.total_cmp(b)
            }
        })
        .ok_or("No curve intersects beyond that endpoint")?;
    Ok(if start { (t, 1.0) } else { (0.0, t) })
}
/// Preview exactly the removed or added segment, without solving or modifying a sketch.
pub fn edit_preview(
    sk: &Sketch,
    id: EntityId,
    cursor: [f64; 2],
    extend: bool,
) -> Result<crate::Seg, String> {
    if sk.entity(id).ok_or("Select a curve")?.projected {
        return Err("Projected geometry is read-only".into());
    }
    let c = curve(sk, id).ok_or("Select a curve")?;
    let (lo, hi) = if extend {
        let (lo, hi) = extend_bounds(sk, id, c, cursor)?;
        if lo < 0.0 {
            (lo, 0.0)
        } else {
            (1.0, hi)
        }
    } else {
        trim_bounds(sk, id, c, cursor)?
    };
    Ok(match c {
        Curve::Line { .. } => crate::Seg::Line {
            a: c.at(lo),
            b: c.at(hi),
        },
        Curve::Arc { c: origin, r, .. } => crate::Seg::Arc {
            c: origin,
            r,
            a: c.at(lo),
            b: c.at(hi),
            ccw: true,
        },
    })
}
impl Sketch {
    /// Trim the interval under the cursor, bounded by actual curve intersections.
    pub fn trim(&mut self, id: EntityId, cursor: [f64; 2]) -> Result<(), String> {
        let entity = self.entity(id).ok_or("Select a curve")?;
        if entity.projected {
            return Err("Projected geometry cannot be trimmed".into());
        }
        let construction = entity.construction;
        let c = curve(self, id).ok_or("Select a curve")?;
        let (lo, hi) = trim_bounds(self, id, c, cursor)?;
        let mut next = self.clone();
        if c.is_closed() {
            if hi - lo < 1.0 - EPS {
                piece(&mut next, c, hi, lo + 1.0, construction);
            }
        } else {
            if lo > EPS {
                piece(&mut next, c, 0.0, lo, construction);
            }
            if hi < 1.0 - EPS {
                piece(&mut next, c, hi, 1.0, construction);
            }
        }
        next.remove_entity(id);
        let points: Vec<_> = next
            .entities
            .iter()
            .filter(|(id, _)| !self.entities.contains_key(id))
            .flat_map(|(_, e)| match e.geom {
                Geom::Line { a, b } => vec![a, b],
                Geom::Arc { s, e, .. } => vec![s, e],
                _ => vec![],
            })
            .collect();
        for p in points {
            let position = next.pos(p);
            for (&other, _) in &self.entities {
                if other != id {
                    if let Some(c) = curve(self, other) {
                        if let Some(t) = c.param_of(position) {
                            let q = c.at(t);
                            if (q[0] - position[0]).hypot(q[1] - position[1]) < EPS
                                && !next.defining_points(other).contains(&p)
                            {
                                let _ = next.add_constraint(Constraint::Coincident(p, other));
                            }
                        }
                    }
                }
            }
        }
        *self = next;
        Ok(())
    }
    /// Extend the closer endpoint to the nearest intersection with another curve.
    pub fn extend(&mut self, id: EntityId, cursor: [f64; 2]) -> Result<(), String> {
        let entity = self.entity(id).ok_or("Select a curve")?;
        if entity.projected {
            return Err("Projected geometry cannot be extended".into());
        }
        let construction = entity.construction;
        let c = curve(self, id).ok_or("Select a curve")?;
        let (lo, hi) = extend_bounds(self, id, c, cursor)?;
        let mut next = self.clone();
        piece(&mut next, c, lo, hi, construction);
        next.remove_entity(id);
        *self = next;
        Ok(())
    }
    /// Offset one curve with a persistent relation to its source geometry.
    pub fn offset(&mut self, id: EntityId, distance: f64) -> Result<EntityId, String> {
        if !distance.is_finite() || distance.abs() < EPS {
            return Err("Offset must be finite and nonzero".into());
        }
        let geom = self.entity(id).ok_or("Select a curve")?.geom;
        let mut next = self.clone();
        let result = match geom {
            Geom::Line { a, b } => {
                let p = self.pos(a);
                let q = self.pos(b);
                let l = (q[0] - p[0]).hypot(q[1] - p[1]);
                if l < EPS {
                    return Err("Cannot offset a zero-length line".into());
                }
                let n = [-(q[1] - p[1]) / l * distance, (q[0] - p[0]) / l * distance];
                next.add_line_pts([p[0] + n[0], p[1] + n[1]], [q[0] + n[0], q[1] + n[1]])
            }
            Geom::Circle { c, .. } => {
                let r = self.radius(id).unwrap() + distance;
                if r < EPS {
                    return Err("Offset would collapse the circle".into());
                }
                next.add_circle(c, r)
            }
            Geom::Arc { c, s, e } => {
                let r = self.radius(id).unwrap();
                if r + distance < EPS {
                    return Err("Offset would collapse the arc".into());
                }
                let center = self.pos(c);
                let scale = (r + distance) / r;
                let point = |p: [f64; 2]| {
                    [
                        center[0] + (p[0] - center[0]) * scale,
                        center[1] + (p[1] - center[1]) * scale,
                    ]
                };
                let s = next.add_point(point(self.pos(s)));
                let e = next.add_point(point(self.pos(e)));
                next.add_arc(c, s, e)
            }
            _ => return Err("Select a line, circle or arc".into()),
        };
        next.add_constraint(Constraint::Offset {
            a: id,
            b: result,
            value: distance,
        })
        .map_err(|e| e.to_string())?;
        *self = next;
        Ok(result)
    }
}
/// Circumcircle and orientation for a start/end/through-point arc.
pub fn three_point_arc(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> Result<([f64; 2], bool), String> {
    if a.into_iter().chain(b).chain(p).any(|v| !v.is_finite()) {
        return Err("Arc points must be finite".into());
    }
    let u = [b[0] - a[0], b[1] - a[1]];
    let v = [p[0] - a[0], p[1] - a[1]];
    let den = 2.0 * (u[0] * v[1] - u[1] * v[0]);
    let scale = u[0].hypot(u[1]) * v[0].hypot(v[1]);
    if scale < EPS || den.abs() < 1e-10 * scale {
        return Err("Arc points must be distinct and non-collinear".into());
    }
    let uu = u[0] * u[0] + u[1] * u[1];
    let vv = v[0] * v[0] + v[1] * v[1];
    let c = [
        a[0] + (uu * v[1] - vv * u[1]) / den,
        a[1] + (u[0] * vv - v[0] * uu) / den,
    ];
    let ang = |q: [f64; 2]| (q[1] - c[1]).atan2(q[0] - c[0]);
    let ccw = (ang(p) - ang(a)).rem_euclid(TAU) < (ang(b) - ang(a)).rem_euclid(TAU);
    Ok((c, ccw))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trim_middle_keeps_boundaries_and_drops_only_target_constraints() {
        let mut s = Sketch::new();
        let l = s.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let a = s.add_line_pts([3.0, -5.0], [3.0, 5.0]);
        let b = s.add_line_pts([7.0, -5.0], [7.0, 5.0]);
        s.add_constraint(Constraint::Horizontal(l)).unwrap();
        s.trim(l, [5.0, 0.0]).unwrap();
        assert!(s.entity(a).is_some() && s.entity(b).is_some());
        let lines: Vec<_> = s
            .entities
            .values()
            .filter(|e| matches!(e.geom, Geom::Line { .. }))
            .collect();
        assert_eq!(lines.len(), 4);
        assert!(s.constraints.values().all(|r| !r.c.entities().contains(&l)));
        assert!(!s.entities.contains_key(&l));
    }
    #[test]
    fn extend_uses_nearest_finite_boundary_and_preserves_shared_point() {
        let mut s = Sketch::new();
        let a = s.add_point([0.0, 0.0]);
        let b = s.add_point([2.0, 0.0]);
        let l = s.add_line(a, b);
        let other = s.add_line_pts([5.0, -1.0], [5.0, 1.0]);
        s.add_line_pts([8.0, -1.0], [8.0, 1.0]);
        let shared = s.add_line_pts([2.0, 0.0], [2.0, 2.0]);
        s.extend(l, [2.0, 0.0]).unwrap();
        assert!(s.entity(other).is_some() && s.entity(shared).is_some());
        assert!(s.entities.values().any(|e| match e.geom {
            Geom::Line { a, b } => s.pos(a) == [0.0, 0.0] && s.pos(b) == [5.0, 0.0],
            _ => false,
        }));
    }
    #[test]
    fn modifications_fail_atomically_on_projected_or_no_extension() {
        let mut s = Sketch::new();
        let id = s.add_line_pts([0.0, 0.0], [2.0, 0.0]);
        let before = serde_json::to_string(&s).unwrap();
        assert!(s.extend(id, [2.0, 0.0]).is_err());
        assert_eq!(serde_json::to_string(&s).unwrap(), before);
        s.entities.get_mut(&id).unwrap().projected = true;
        assert!(s.trim(id, [1.0, 0.0]).is_err());
    }
    #[test]
    fn circle_trim_leaves_the_opposite_arc() {
        let mut s = Sketch::new();
        let c = s.add_point([0.0, 0.0]);
        let id = s.add_circle(c, 5.0);
        s.add_line_pts([-10.0, 0.0], [10.0, 0.0]);
        s.trim(id, [0.0, 5.0]).unwrap();
        let arc = s
            .entities
            .iter()
            .find_map(|(&id, e)| matches!(e.geom, Geom::Arc { .. }).then_some(id))
            .unwrap();
        let cv = curve(&s, arc).unwrap();
        assert!(cv.at(0.5)[1] < -4.9);
    }
    #[test]
    fn associative_offsets_follow_fixed_source_edits_and_roundtrip() {
        for round in [false, true] {
            let mut s = Sketch::new();
            let source = if round {
                let c = s.add_point([0.0, 0.0]);
                s.add_circle(c, 5.0)
            } else {
                s.add_line_pts([0.0, 0.0], [10.0, 0.0])
            };
            s.add_constraint(Constraint::Fix(source)).unwrap();
            let id = s.offset(source, 2.0).unwrap();
            if round {
                if let Geom::Circle { r, .. } = s.entities[&source].geom {
                    s.x[r] = 10.0;
                }
            } else {
                let (a, b) = s.line_pts(source).unwrap();
                s.set_pos(a, [0.0, 3.0]);
                s.set_pos(b, [20.0, 3.0]);
            }
            assert!(s.solve().converged);
            if round {
                assert!((s.radius(id).unwrap() - 12.0).abs() < 1e-6);
            } else {
                let (a, b) = s.line_pts(id).unwrap();
                assert!((s.pos(a)[1] - 5.0).abs() < 1e-6);
                assert!((s.pos(b)[0] - 20.0).abs() < 1e-6);
            }
            let mut loaded: Sketch =
                serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
            assert!(loaded.solve().converged);
        }
    }
    #[test]
    fn invalid_offset_and_degenerate_arc_are_rejected() {
        let mut s = Sketch::new();
        let c = s.add_point([0.0, 0.0]);
        let id = s.add_circle(c, 5.0);
        let count = s.entities.len();
        assert!(s.offset(id, -6.0).is_err());
        assert!(s.offset(id, f64::NAN).is_err());
        assert_eq!(s.entities.len(), count);
        assert!(three_point_arc([0.0, 0.0], [2.0, 0.0], [1.0, 0.0]).is_err());
        let (c, ccw) = three_point_arc([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]).unwrap();
        assert!(c[0].abs() < 1e-9 && c[1].abs() < 1e-9 && ccw);
    }
}
#[cfg(test)]
mod extended_tests {
    use super::*;
    #[test]
    fn arc_extension_works_at_either_end() {
        for start in [false, true] {
            let mut s = Sketch::new();
            let c = s.add_point([0.0, 0.0]);
            let a = s.add_point([1.0, 0.0]);
            let b = s.add_point([0.0, 1.0]);
            let arc = s.add_arc(c, a, b);
            if start {
                s.add_line_pts([0.0, -2.0], [0.0, -0.5]);
            } else {
                s.add_line_pts([-2.0, 0.0], [-0.5, 0.0]);
            }
            s.extend(arc, if start { [1.0, 0.0] } else { [0.0, 1.0] })
                .unwrap();
            let arc = s
                .entities
                .iter()
                .find_map(|(&id, e)| matches!(e.geom, Geom::Arc { .. }).then_some(id))
                .unwrap();
            if let Curve::Arc { sweep, .. } = curve(&s, arc).unwrap() {
                assert!((sweep - std::f64::consts::PI).abs() < 1e-6);
            }
            assert!(s.solve().converged);
        }
    }
    #[test]
    fn line_offset_follows_source_rotation() {
        let mut s = Sketch::new();
        let source = s.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        s.add_constraint(Constraint::Fix(source)).unwrap();
        let new = s.offset(source, 2.0).unwrap();
        let (a, b) = s.line_pts(source).unwrap();
        s.set_pos(a, [0.0, 0.0]);
        s.set_pos(b, [0.0, 10.0]);
        assert!(s.solve().converged);
        let (a, b) = s.line_pts(new).unwrap();
        assert!((s.pos(a)[0] + 2.0).abs() < 1e-6 && (s.pos(a)[1]).abs() < 1e-6);
        assert!((s.pos(b)[0] + 2.0).abs() < 1e-6 && (s.pos(b)[1] - 10.0).abs() < 1e-6);
    }
    #[test]
    fn arc_offset_retains_start_end_angles_and_signed_radius() {
        let mut s = Sketch::new();
        let c = s.add_point([0.0, 0.0]);
        let a = s.add_point([5.0, 0.0]);
        let b = s.add_point([0.0, 5.0]);
        let source = s.add_arc(c, a, b);
        s.add_constraint(Constraint::Fix(source)).unwrap();
        let new = s.offset(source, -1.0).unwrap();
        s.set_pos(a, [10.0, 0.0]);
        s.set_pos(b, [0.0, 10.0]);
        assert!(s.solve().converged);
        assert!((s.radius(new).unwrap() - 9.0).abs() < 1e-6);
        if let Geom::Arc { s: a, e: b, .. } = s.entities[&new].geom {
            assert!((s.pos(a)[0] - 9.0).abs() < 1e-6 && s.pos(a)[1].abs() < 1e-6);
            assert!((s.pos(b)[1] - 9.0).abs() < 1e-6 && s.pos(b)[0].abs() < 1e-6);
        }
    }
}
#[cfg(test)]
mod preview_tests {
    use super::*;
    #[test]
    fn preview_is_exact_and_read_only() {
        let mut s = Sketch::new();
        let id = s.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        s.add_line_pts([3.0, -1.0], [3.0, 1.0]);
        s.add_line_pts([7.0, -1.0], [7.0, 1.0]);
        let before = serde_json::to_string(&s).unwrap();
        assert_eq!(
            edit_preview(&s, id, [5.0, 0.0], false).unwrap(),
            crate::Seg::Line {
                a: [3.0, 0.0],
                b: [7.0, 0.0]
            }
        );
        assert_eq!(serde_json::to_string(&s).unwrap(), before);
        let id = s.add_line_pts([-3.0, 0.5], [-1.0, 0.5]);
        assert_eq!(
            edit_preview(&s, id, [-1.0, 0.5], true).unwrap(),
            crate::Seg::Line {
                a: [-1.0, 0.5],
                b: [3.0, 0.5]
            }
        );
    }
}
