use eframe::egui::{Pos2, Vec2};
use refusion_model::sketch::{Constraint, DimKind, Geom, Sketch};

pub const ARROW_LEN: f32 = 10.0;
pub const ARROW_HALF_W: f32 = 3.5;
const EXT_GAP: f32 = 2.0;
const EXT_OVERSHOOT: f32 = 5.0;

type P2 = [f64; 2];

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}
fn mul(a: P2, k: f64) -> P2 {
    [a[0] * k, a[1] * k]
}
fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len(a: P2) -> f64 {
    a[0].hypot(a[1])
}
fn unit(a: P2) -> P2 {
    let l = len(a).max(1e-300);
    [a[0] / l, a[1] / l]
}

/// The construction of a dimension in sketch coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum DimSpec {
    /// Measured points, measuring direction, and extra reference-line extensions.
    Linear {
        p1: P2,
        p2: P2,
        d: P2,
        t: P2,
        ref_ext: Vec<(P2, P2)>,
    },
    /// Centre, radius, text, and for arcs (start angle, CCW sweep).
    Radius {
        c: P2,
        r: f64,
        t: P2,
        arc: Option<(f64, f64)>,
    },
    Diameter {
        c: P2,
        r: f64,
        t: P2,
        arc: Option<(f64, f64)>,
    },
    /// Vertex, sector start/end unit directions (CCW from e1 to e2), radius, text, extensions.
    Angular {
        v: P2,
        e1: P2,
        e2: P2,
        r: f64,
        t: P2,
        ext: Vec<(P2, P2)>,
    },
}

/// Foot of the perpendicular from p onto the infinite line (a, b).
fn foot(p: P2, a: P2, b: P2) -> P2 {
    let u = unit(sub(b, a));
    add(a, mul(u, dot(sub(p, a), u)))
}

/// Extension along a reference segment when `q` (on its carrier) lies outside it.
fn seg_extension(q: P2, a: P2, b: P2) -> Option<(P2, P2)> {
    let u = sub(b, a);
    let t = dot(sub(q, a), u) / dot(u, u).max(1e-300);
    if t < 0.0 {
        Some((a, q))
    } else if t > 1.0 {
        Some((b, q))
    } else {
        None
    }
}

fn arc_of(sk: &Sketch, e: refusion_model::sketch::EntityId) -> Option<(f64, f64)> {
    if let Geom::Arc { c, s, e: end } = sk.entity(e)?.geom {
        let (c, s, end) = (sk.pos(c), sk.pos(s), sk.pos(end));
        let a0 = (s[1] - c[1]).atan2(s[0] - c[0]);
        let mut sw = ((end[1] - c[1]).atan2(end[0] - c[0]) - a0).rem_euclid(std::f64::consts::TAU);
        if sw < 1e-9 {
            sw = std::f64::consts::TAU;
        }
        return Some((a0, sw));
    }
    None
}

/// Build the construction for a dimension constraint with text at `t`.
pub fn spec(sk: &Sketch, c: &Constraint, t: P2) -> Option<DimSpec> {
    use Constraint::*;
    Some(match *c {
        Offset { a, b, value } => {
            if sk.line_pts(a).is_some() {
                return spec(
                    sk,
                    &Distance {
                        a,
                        b,
                        kind: DimKind::Aligned,
                        value,
                    },
                    t,
                );
            }
            let center = sk.pos(sk.center(a)?);
            let direction = unit(sub(t, center));
            let p1 = add(center, mul(direction, sk.radius(a)?));
            let p2 = add(center, mul(direction, sk.radius(b)?));
            DimSpec::Linear {
                p1,
                p2,
                d: direction,
                t,
                ref_ext: vec![],
            }
        }
        Length { line, .. } => {
            let (a, b) = sk.line_pts(line)?;
            let (p1, p2) = (sk.pos(a), sk.pos(b));
            DimSpec::Linear {
                p1,
                p2,
                d: unit(sub(p2, p1)),
                t,
                ref_ext: vec![],
            }
        }
        Distance { a, b, kind, .. } => {
            let (pa, pb) = (sk.is_point(a), sk.is_point(b));
            if pa && pb {
                let (p1, p2) = (sk.pos(a), sk.pos(b));
                let d = match kind {
                    DimKind::Aligned => unit(sub(p2, p1)),
                    DimKind::Horizontal => [1.0, 0.0],
                    DimKind::Vertical => [0.0, 1.0],
                };
                DimSpec::Linear {
                    p1,
                    p2,
                    d,
                    t,
                    ref_ext: vec![],
                }
            } else {
                // point–line or parallel line–line: perpendicular to the reference line
                let (p, l) = if pa {
                    (sk.pos(a), b)
                } else if pb {
                    (sk.pos(b), a)
                } else {
                    (sk.pos(sk.line_pts(b)?.0), a)
                };
                let (la, lb) = sk.line_pts(l)?;
                let (la, lb) = (sk.pos(la), sk.pos(lb));
                let f = foot(p, la, lb);
                let d = if len(sub(f, p)) > 1e-12 {
                    unit(sub(f, p))
                } else {
                    [-(lb[1] - la[1]), lb[0] - la[0]]
                };
                DimSpec::Linear {
                    p1: p,
                    p2: f,
                    d: unit(d),
                    t,
                    ref_ext: seg_extension(f, la, lb).into_iter().collect(),
                }
            }
        }
        Radius { e, .. } => DimSpec::Radius {
            c: sk.pos(sk.center(e)?),
            r: sk.radius(e)?,
            t,
            arc: arc_of(sk, e),
        },
        Diameter { e, .. } => DimSpec::Diameter {
            c: sk.pos(sk.center(e)?),
            r: sk.radius(e)?,
            t,
            arc: arc_of(sk, e),
        },
        Angle { a, b, value } => angular(sk, a, b, value, t)?,
        _ => return None,
    })
}

fn angular(
    sk: &Sketch,
    a: refusion_model::sketch::EntityId,
    b: refusion_model::sketch::EntityId,
    value: f64,
    t: P2,
) -> Option<DimSpec> {
    let (a1, a2) = sk.line_pts(a)?;
    let (b1, b2) = sk.line_pts(b)?;
    let (a1, a2, b1, b2) = (sk.pos(a1), sk.pos(a2), sk.pos(b1), sk.pos(b2));
    let (u, w) = (unit(sub(a2, a1)), unit(sub(b2, b1)));
    let den = u[0] * w[1] - u[1] * w[0];
    if den.abs() < 1e-9 {
        return None;
    }
    let s = ((b1[0] - a1[0]) * w[1] - (b1[1] - a1[1]) * w[0]) / den;
    let v = add(a1, mul(u, s));
    let tdir = unit(sub(t, v));
    let ang = |x: P2| x[1].atan2(x[0]);
    // the four quadrants (±u, ±w), each as a CCW sector of less than 180°
    let mut cands = Vec::new();
    for su in [1.0, -1.0] {
        for sw in [1.0, -1.0] {
            let (mut e1, mut e2) = (mul(u, su), mul(w, sw));
            if e1[0] * e2[1] - e1[1] * e2[0] < 0.0 {
                std::mem::swap(&mut e1, &mut e2);
            }
            let open = (ang(e2) - ang(e1)).rem_euclid(std::f64::consts::TAU);
            let rel = (ang(tdir) - ang(e1)).rem_euclid(std::f64::consts::TAU);
            let bis = unit(add(e1, e2));
            cands.push((e1, e2, open, rel <= open, dot(bis, tdir)));
        }
    }
    let matches_value = |o: f64| (o - value).abs() < 1e-3;
    let pick = cands
        .iter()
        .filter(|c| matches_value(c.2))
        .max_by(|x, y| (x.3, x.4).partial_cmp(&(y.3, y.4)).unwrap())
        .or_else(|| {
            cands
                .iter()
                .max_by(|x, y| (x.3, x.4).partial_cmp(&(y.3, y.4)).unwrap())
        })?;
    let (e1, e2) = (pick.0, pick.1);
    let r = len(sub(t, v)).max(1e-9);
    let mut ext = Vec::new();
    for (e, (p, q)) in [(e1, (a1, a2)), (e2, (b1, b2))] {
        let at = add(v, mul(e, r));
        // the arc point lies on line (p,q)'s carrier unless the sector uses the other line
        let on_line = |x: P2| {
            let n = unit(sub(q, p));
            ((x[0] - p[0]) * n[1] - (x[1] - p[1]) * n[0]).abs() < 1e-6 * (1.0 + r)
        };
        if on_line(at) {
            ext.extend(seg_extension(at, p, q));
        } else {
            let other = if (p, q) == (a1, a2) {
                (b1, b2)
            } else {
                (a1, a2)
            };
            ext.extend(seg_extension(at, other.0, other.1));
        }
    }
    Some(DimSpec::Angular {
        v,
        e1,
        e2,
        r,
        t,
        ext,
    })
}

/// Screen-space primitives for one dimension.
#[derive(Clone, Debug, Default)]
pub struct Prims {
    pub lines: Vec<(Pos2, Pos2)>,
    /// Arrowheads: (tip, unit direction the arrow points to).
    pub arrows: Vec<(Pos2, Vec2)>,
    /// Polylines (angular arc, arc extension).
    pub polys: Vec<Vec<Pos2>>,
    pub text: Pos2,
    /// Text rotation in radians (screen, right-reading).
    pub angle: f32,
}

fn right_reading(mut a: f32) -> f32 {
    use std::f32::consts::PI;
    while a >= PI / 2.0 - 1e-4 {
        a -= PI;
    }
    while a < -PI / 2.0 - 1e-4 {
        a += PI;
    }
    a
}

/// Project and size the construction. `text_w` is the label width in px.
pub fn layout(
    spec: &DimSpec,
    to_screen: &dyn Fn(P2) -> Option<Pos2>,
    text_w: f32,
) -> Option<Prims> {
    let mut pr = Prims::default();
    let dir = |a: Pos2, b: Pos2| (b - a).normalized();
    match spec {
        DimSpec::Linear {
            p1,
            p2,
            d,
            t,
            ref_ext,
        } => {
            let proj = |p: P2| add(*t, mul(*d, dot(sub(p, *t), *d)));
            let (d1, d2) = (proj(*p1), proj(*p2));
            let (s1, s2, sd1, sd2, st) = (
                to_screen(*p1)?,
                to_screen(*p2)?,
                to_screen(d1)?,
                to_screen(d2)?,
                to_screen(*t)?,
            );
            // extension lines (skipped when the point is on the dimension line)
            for (s, sd) in [(s1, sd1), (s2, sd2)] {
                let v = sd - s;
                if v.length() > EXT_GAP + 0.5 {
                    let u = v.normalized();
                    pr.lines.push((s + u * EXT_GAP, sd + u * EXT_OVERSHOOT));
                }
            }
            for (a, b) in ref_ext {
                pr.lines.push((to_screen(*a)?, to_screen(*b)?));
            }
            let span = (sd2 - sd1).length();
            let along = if span > 1e-3 { dir(sd1, sd2) } else { Vec2::X };
            // dimension line, extended to the text when it is dragged outside
            let tt = (st - sd1).dot(along);
            let (lo, hi) = (tt.min(0.0), tt.max(span));
            let inside = span >= 2.0 * ARROW_LEN + text_w + 8.0;
            if inside {
                pr.lines.push((sd1 + along * lo, sd1 + along * hi));
                pr.arrows.push((sd1, -along));
                pr.arrows.push((sd2, along));
            } else {
                // arrows outside pointing in, with tails
                let tail = ARROW_LEN + 12.0;
                pr.lines.push((
                    sd1 + along * (lo.min(0.0) - tail),
                    sd1 + along * (hi.max(span) + tail),
                ));
                pr.arrows.push((sd1, along));
                pr.arrows.push((sd2, -along));
            }
            pr.text = st;
            pr.angle = right_reading(along.y.atan2(along.x));
        }
        DimSpec::Radius { c, r, t, arc } | DimSpec::Diameter { c, r, t, arc } => {
            let diameter = matches!(spec, DimSpec::Diameter { .. });
            let u = if len(sub(*t, *c)) > 1e-12 {
                unit(sub(*t, *c))
            } else {
                [1.0, 0.0]
            };
            let q = add(*c, mul(u, *r));
            let (sc, sq, st) = (to_screen(*c)?, to_screen(q)?, to_screen(*t)?);
            let outside = len(sub(*t, *c)) > *r;
            if diameter {
                let q1 = to_screen(sub(*c, mul(u, *r)))?;
                pr.lines.push((q1, sq));
                pr.arrows.push((q1, dir(sq, q1)));
                pr.arrows.push((sq, dir(q1, sq)));
                if outside {
                    pr.lines.push((sq, st));
                }
            } else if outside {
                pr.lines.push((sq, st));
                pr.arrows.push((sq, dir(st, sq)));
            } else {
                pr.lines.push((sc, sq));
                pr.arrows.push((sq, dir(sc, sq)));
            }
            // arc extension when the chord point is outside the arc's span
            if let Some((a0, sw)) = arc {
                let ta = u[1].atan2(u[0]);
                let rel = (ta - a0).rem_euclid(std::f64::consts::TAU);
                if rel > *sw + 1e-9 {
                    // go from the nearer arc end to the chord point
                    let (from, sweep) = if rel - sw < std::f64::consts::TAU - rel {
                        (a0 + sw, rel - sw)
                    } else {
                        (ta, std::f64::consts::TAU - rel)
                    };
                    let n = 24;
                    let pts: Option<Vec<Pos2>> = (0..=n)
                        .map(|i| {
                            let a = from + sweep * i as f64 / n as f64;
                            to_screen([c[0] + r * a.cos(), c[1] + r * a.sin()])
                        })
                        .collect();
                    pr.polys.push(pts?);
                }
            }
            pr.text = st;
            pr.angle = 0.0;
        }
        DimSpec::Angular {
            v,
            e1,
            e2,
            r,
            t,
            ext,
        } => {
            let a1 = e1[1].atan2(e1[0]);
            let open = (e2[1].atan2(e2[0]) - a1).rem_euclid(std::f64::consts::TAU);
            let n = 48;
            let pts: Option<Vec<Pos2>> = (0..=n)
                .map(|i| {
                    let a = a1 + open * i as f64 / n as f64;
                    to_screen([v[0] + r * a.cos(), v[1] + r * a.sin()])
                })
                .collect();
            let pts = pts?;
            if pts.len() >= 2 {
                let (s, e) = (pts[0], *pts.last().unwrap());
                pr.arrows.push((s, dir(pts[1], s)));
                pr.arrows.push((e, dir(pts[pts.len() - 2], e)));
            }
            pr.polys.push(pts);
            for (a, b) in ext {
                let (sa, sb) = (to_screen(*a)?, to_screen(*b)?);
                let u = dir(sa, sb);
                pr.lines.push((sa + u * EXT_GAP, sb + u * EXT_OVERSHOOT));
            }
            pr.text = to_screen(*t)?;
            pr.angle = 0.0;
        }
    }
    Some(pr)
}

/// Filled arrowhead triangle (tip, base corners).
pub fn arrow_triangle(tip: Pos2, d: Vec2) -> [Pos2; 3] {
    let n = Vec2::new(-d.y, d.x);
    let base = tip - d * ARROW_LEN;
    [tip, base + n * ARROW_HALF_W, base - n * ARROW_HALF_W]
}

#[cfg(test)]
mod tests {
    use super::*;
    use refusion_model::sketch::{Constraint, DimKind, Sketch};

    fn id(p: P2) -> Option<Pos2> {
        Some(Pos2::new(p[0] as f32, p[1] as f32))
    }

    #[test]
    fn horizontal_dimension_projects_to_text_line() {
        let mut sk = Sketch::new();
        let a = sk.add_point([0.0, 0.0]);
        let b = sk.add_point([100.0, 50.0]);
        let c = Constraint::Distance {
            a,
            b,
            kind: DimKind::Horizontal,
            value: 100.0,
        };
        let s = spec(&sk, &c, [50.0, 120.0]).unwrap();
        let pr = layout(&s, &id, 30.0).unwrap();
        // two extension lines + dimension line at y = 120
        assert_eq!(pr.lines.len(), 3);
        let dl = pr.lines[2];
        assert!((dl.0.y - 120.0).abs() < 1e-4 && (dl.1.y - 120.0).abs() < 1e-4);
        assert!((dl.0.x - 0.0).abs() < 1e-4 && (dl.1.x - 100.0).abs() < 1e-4);
        // arrows at the ends, pointing outward
        assert_eq!(pr.arrows.len(), 2);
        assert!(pr.arrows[0].1.x < 0.0 && pr.arrows[1].1.x > 0.0);
        assert!(pr.angle.abs() < 1e-6);
    }

    #[test]
    fn short_dimension_flips_arrows_outside() {
        let mut sk = Sketch::new();
        let l = sk.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let s = spec(
            &sk,
            &Constraint::Length {
                line: l,
                value: 10.0,
            },
            [5.0, -20.0],
        )
        .unwrap();
        let pr = layout(&s, &id, 30.0).unwrap();
        assert!(
            pr.arrows[0].1.x > 0.0 && pr.arrows[1].1.x < 0.0,
            "{:?}",
            pr.arrows
        );
    }

    #[test]
    fn vertical_text_reads_upright() {
        // screen y points down: -90° is bottom-to-top
        assert!(
            (right_reading(std::f32::consts::FRAC_PI_2) + std::f32::consts::FRAC_PI_2).abs() < 1e-3
        );
        assert!(
            (right_reading(-std::f32::consts::FRAC_PI_2) + std::f32::consts::FRAC_PI_2).abs()
                < 1e-3
        );
        assert!(right_reading(std::f32::consts::PI).abs() < 1e-3);
    }

    #[test]
    fn radius_and_diameter_touch_the_circle() {
        let mut sk = Sketch::new();
        let c = sk.add_point([0.0, 0.0]);
        let circ = sk.add_circle(c, 10.0);
        let s = spec(
            &sk,
            &Constraint::Diameter {
                e: circ,
                value: 20.0,
            },
            [30.0, 0.0],
        )
        .unwrap();
        let pr = layout(&s, &id, 20.0).unwrap();
        // diameter across the circle + leader to the text
        assert_eq!(pr.lines[0], (Pos2::new(-10.0, 0.0), Pos2::new(10.0, 0.0)));
        assert_eq!(pr.lines[1], (Pos2::new(10.0, 0.0), Pos2::new(30.0, 0.0)));
        assert_eq!(pr.arrows.len(), 2);
        let s = spec(
            &sk,
            &Constraint::Radius {
                e: circ,
                value: 10.0,
            },
            [0.0, 4.0],
        )
        .unwrap();
        let pr = layout(&s, &id, 20.0).unwrap();
        // text inside: centre → chord point, single arrow at the circle pointing out
        assert_eq!(pr.lines, vec![(Pos2::new(0.0, 0.0), Pos2::new(0.0, 10.0))]);
        assert_eq!(pr.arrows.len(), 1);
        assert!((pr.arrows[0].0.y - 10.0).abs() < 1e-5 && pr.arrows[0].1.y > 0.0);
    }

    #[test]
    fn angular_quadrant_follows_text_and_value() {
        let mut sk = Sketch::new();
        let l1 = sk.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let l2 = sk.add_line_pts([0.0, 0.0], [10.0, 10.0]);
        // 45° between the lines; text in the 45° wedge
        let s = spec(
            &sk,
            &Constraint::Angle {
                a: l1,
                b: l2,
                value: 45f64.to_radians(),
            },
            [8.0, 3.0],
        )
        .unwrap();
        let DimSpec::Angular { e1, e2, r, .. } = s else {
            panic!()
        };
        assert!(
            (e1[1].atan2(e1[0])).abs() < 1e-9
                && (e2[1].atan2(e2[0]) - 45f64.to_radians()).abs() < 1e-9
        );
        assert!((r - 73f64.sqrt()).abs() < 1e-9);
        // text placed in the opposite 45° wedge → that wedge (still 45°)
        let s = spec(
            &sk,
            &Constraint::Angle {
                a: l1,
                b: l2,
                value: 45f64.to_radians(),
            },
            [-8.0, -3.0],
        )
        .unwrap();
        let DimSpec::Angular { e1, e2, ext, .. } = s else {
            panic!()
        };
        let open = (e2[1].atan2(e2[0]) - e1[1].atan2(e1[0])).rem_euclid(std::f64::consts::TAU);
        assert!((open - 45f64.to_radians()).abs() < 1e-9);
        assert_eq!(
            ext.len(),
            2,
            "extension lines from the line ends to the arc"
        );
    }

    #[test]
    fn point_line_distance_extends_reference_line() {
        let mut sk = Sketch::new();
        let l = sk.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let p = sk.add_point([20.0, 5.0]);
        let s = spec(
            &sk,
            &Constraint::Distance {
                a: p,
                b: l,
                kind: DimKind::Aligned,
                value: 5.0,
            },
            [25.0, 2.0],
        )
        .unwrap();
        let DimSpec::Linear { p2, ref_ext, .. } = s else {
            panic!()
        };
        assert_eq!(p2, [20.0, 0.0]);
        assert_eq!(ref_ext, vec![([10.0, 0.0], [20.0, 0.0])]);
    }
}
