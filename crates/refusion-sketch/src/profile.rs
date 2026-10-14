use crate::sketch::{EntityId, Geom, Sketch};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    f64::consts::{PI, TAU},
};

const TOL: f64 = 1e-7;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Seg {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    /// Arc from `a` to `b` around `c`; `ccw` gives the travel direction.
    Arc {
        c: [f64; 2],
        r: f64,
        a: [f64; 2],
        b: [f64; 2],
        ccw: bool,
    },
}

impl Seg {
    pub fn start(&self) -> [f64; 2] {
        match *self {
            Seg::Line { a, .. } | Seg::Arc { a, .. } => a,
        }
    }
    pub fn end(&self) -> [f64; 2] {
        match *self {
            Seg::Line { b, .. } | Seg::Arc { b, .. } => b,
        }
    }
    /// Sweep angle of an arc segment (always positive).
    pub fn sweep(&self) -> f64 {
        match *self {
            Seg::Line { .. } => 0.0,
            Seg::Arc { c, a, b, ccw, .. } => {
                let ta = (a[1] - c[1]).atan2(a[0] - c[0]);
                let tb = (b[1] - c[1]).atan2(b[0] - c[0]);
                let mut s = if ccw { tb - ta } else { ta - tb };
                while s <= 1e-12 {
                    s += TAU;
                }
                while s > TAU + 1e-12 {
                    s -= TAU;
                }
                s
            }
        }
    }
    /// Polyline approximation (excluding the end point).
    pub fn sample(&self, chord_tol: f64, out: &mut Vec<[f64; 2]>) {
        match *self {
            Seg::Line { a, .. } => out.push(a),
            Seg::Arc { c, r, a, ccw, .. } => {
                let sweep = self.sweep();
                let step = 2.0 * (1.0 - (chord_tol / r).min(1.0)).acos().max(1e-3);
                let n = ((sweep / step).ceil() as usize).clamp(2, 512);
                let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
                let dir = if ccw { 1.0 } else { -1.0 };
                for i in 0..n {
                    let t = t0 + dir * sweep * i as f64 / n as f64;
                    out.push([c[0] + r * t.cos(), c[1] + r * t.sin()]);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Loop {
    pub segs: Vec<Seg>,
    /// Sketch entities this loop is made of.
    pub curves: Vec<EntityId>,
}

impl Loop {
    pub fn polygon(&self, chord_tol: f64) -> Vec<[f64; 2]> {
        let mut out = Vec::new();
        for s in &self.segs {
            s.sample(chord_tol, &mut out);
        }
        out
    }
    pub fn area(&self) -> f64 {
        polygon_area(&self.polygon(1e-3))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// Counter-clockwise outer boundary.
    pub outer: Loop,
    /// Clockwise inner boundaries.
    pub holes: Vec<Loop>,
}

impl Profile {
    pub fn area(&self) -> f64 {
        self.outer.area() + self.holes.iter().map(|h| h.area()).sum::<f64>()
    }

    /// Sorted, de-duplicated entity ids bounding this profile — used to find
    /// "the same" profile again after the sketch changes.
    pub fn signature(&self) -> Vec<EntityId> {
        let mut v: Vec<EntityId> = self
            .outer
            .curves
            .iter()
            .chain(self.holes.iter().flat_map(|h| h.curves.iter()))
            .copied()
            .collect();
        v.sort();
        v.dedup();
        v
    }

    pub fn contains(&self, p: [f64; 2]) -> bool {
        point_in_polygon(p, &self.outer.polygon(1e-3))
            && !self
                .holes
                .iter()
                .any(|h| point_in_polygon(p, &h.polygon(1e-3)))
    }
}

pub fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let (u, v) = (p[i], p[(i + 1) % n]);
        a += u[0] * v[1] - v[0] * u[1];
    }
    0.5 * a
}

pub fn point_in_polygon(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ----------------------------------------------------------------- curves

#[derive(Clone, Copy, Debug)]
pub(crate) enum Curve {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    /// CCW from t0 by `sweep` (2π for full circles).
    Arc {
        c: [f64; 2],
        r: f64,
        t0: f64,
        sweep: f64,
    },
}

impl Curve {
    pub(crate) fn at(&self, t: f64) -> [f64; 2] {
        match *self {
            Curve::Line { a, b } => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])],
            Curve::Arc { c, r, t0, sweep } => {
                let ang = t0 + t * sweep;
                [c[0] + r * ang.cos(), c[1] + r * ang.sin()]
            }
        }
    }
    pub(crate) fn is_closed(&self) -> bool {
        matches!(*self, Curve::Arc { sweep, .. } if sweep >= TAU - 1e-12)
    }
    /// Parameter (0..1) of a point known to lie on the curve's carrier, if inside the span.
    pub(crate) fn param_of(&self, p: [f64; 2]) -> Option<f64> {
        match *self {
            Curve::Line { a, b } => {
                let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
                let l2 = ux * ux + uy * uy;
                let t = ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / l2;
                let tol = TOL / l2.sqrt();
                (t >= -tol && t <= 1.0 + tol).then_some(t.clamp(0.0, 1.0))
            }
            Curve::Arc { c, r, t0, sweep } => {
                let ang = (p[1] - c[1]).atan2(p[0] - c[0]);
                let mut d = (ang - t0).rem_euclid(TAU);
                let tol = TOL / r;
                if d > sweep + tol && d > TAU - tol {
                    d -= TAU; // just before the start
                }
                if d < -tol || d > sweep + tol {
                    return None;
                }
                Some((d / sweep).clamp(0.0, 1.0))
            }
        }
    }
}

pub(crate) fn carrier_intersections(a: &Curve, b: &Curve) -> Vec<[f64; 2]> {
    use Curve::*;
    let carrier_hits: Vec<[f64; 2]> = match (*a, *b) {
        (Line { a: p, b: q }, Line { a: r, b: s }) => {
            let (d1, d2) = ([q[0] - p[0], q[1] - p[1]], [s[0] - r[0], s[1] - r[1]]);
            let den = d1[0] * d2[1] - d1[1] * d2[0];
            if den.abs() < 1e-14 * (d1[0].hypot(d1[1]) * d2[0].hypot(d2[1])) {
                // Parallel/collinear: only shared endpoints matter.
                [p, q]
                    .into_iter()
                    .filter(|&e| b.param_of(e).is_some() && on_carrier(b, e))
                    .chain(
                        [r, s]
                            .into_iter()
                            .filter(|&e| a.param_of(e).is_some() && on_carrier(a, e)),
                    )
                    .collect()
            } else {
                let t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den;
                vec![[p[0] + t * d1[0], p[1] + t * d1[1]]]
            }
        }
        (Line { a: p, b: q }, Arc { c, r, .. }) | (Arc { c, r, .. }, Line { a: p, b: q }) => {
            let d = [q[0] - p[0], q[1] - p[1]];
            let f = [p[0] - c[0], p[1] - c[1]];
            let (aa, bb, cc) = (
                d[0] * d[0] + d[1] * d[1],
                2.0 * (f[0] * d[0] + f[1] * d[1]),
                f[0] * f[0] + f[1] * f[1] - r * r,
            );
            let disc = bb * bb - 4.0 * aa * cc;
            let tol = 2.0 * TOL * r * aa.sqrt() * 2.0;
            if disc < -tol * aa.sqrt() {
                vec![]
            } else {
                let sq = disc.max(0.0).sqrt();
                let ts = if sq < 1e-12 {
                    vec![-bb / (2.0 * aa)]
                } else {
                    vec![(-bb - sq) / (2.0 * aa), (-bb + sq) / (2.0 * aa)]
                };
                ts.into_iter()
                    .map(|t| [p[0] + t * d[0], p[1] + t * d[1]])
                    .collect()
            }
        }
        (Arc { c: c1, r: r1, .. }, Arc { c: c2, r: r2, .. }) => {
            let d = (c2[0] - c1[0]).hypot(c2[1] - c1[1]);
            if d < TOL || d > r1 + r2 + TOL || d < (r1 - r2).abs() - TOL {
                vec![]
            } else {
                let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
                let h = (r1 * r1 - a * a).max(0.0).sqrt();
                let (ux, uy) = ((c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d);
                let m = [c1[0] + a * ux, c1[1] + a * uy];
                if h < 1e-9 {
                    vec![m]
                } else {
                    vec![
                        [m[0] - h * uy, m[1] + h * ux],
                        [m[0] + h * uy, m[1] - h * ux],
                    ]
                }
            }
        }
    };
    carrier_hits
}

pub(crate) fn intersect(a: &Curve, b: &Curve) -> Vec<[f64; 2]> {
    carrier_intersections(a, b)
        .into_iter()
        .filter(|&p| a.param_of(p).is_some() && b.param_of(p).is_some())
        .collect()
}

fn on_carrier(c: &Curve, p: [f64; 2]) -> bool {
    match *c {
        Curve::Line { a, b } => {
            let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
            ((ux * (p[1] - a[1]) - uy * (p[0] - a[0])) / ux.hypot(uy)).abs() < TOL
        }
        Curve::Arc { c, r, .. } => ((p[0] - c[0]).hypot(p[1] - c[1]) - r).abs() < TOL,
    }
}

// ----------------------------------------------------------------- graph

struct Edge {
    v: [usize; 2],
    seg: Seg, // oriented v0 → v1
    entity: EntityId,
}

fn reversed(s: Seg) -> Seg {
    match s {
        Seg::Line { a, b } => Seg::Line { a: b, b: a },
        Seg::Arc { c, r, a, b, ccw } => Seg::Arc {
            c,
            r,
            a: b,
            b: a,
            ccw: !ccw,
        },
    }
}

/// Direction leaving the start of a segment, sampled a short way along it so
/// that a line and an arc sharing a tangent are still ordered correctly.
fn leave_angle(s: &Seg) -> f64 {
    match *s {
        Seg::Line { a, b } => (b[1] - a[1]).atan2(b[0] - a[0]),
        Seg::Arc { c, r, a, ccw, .. } => {
            let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
            let dt = (s.sweep() * 1e-4).min(1e-3 / r.max(1e-9));
            let t = if ccw { t0 + dt } else { t0 - dt };
            let p = [c[0] + r * t.cos(), c[1] + r * t.sin()];
            (p[1] - a[1]).atan2(p[0] - a[0])
        }
    }
}

pub fn find_profiles(sk: &Sketch) -> Vec<Profile> {
    if sk.validate_data().is_err() {
        return vec![];
    }
    // 1. Curves
    let mut curves: Vec<(EntityId, Curve)> = Vec::new();
    for (&id, e) in &sk.entities {
        if e.construction {
            continue;
        }
        match e.geom {
            Geom::Point(_) => {}
            Geom::Line { a, b } => {
                let (a, b) = (sk.pos(a), sk.pos(b));
                if (a[0] - b[0]).hypot(a[1] - b[1]) > TOL {
                    curves.push((id, Curve::Line { a, b }));
                }
            }
            Geom::Circle { c, r } => {
                let r = sk.x[r];
                if r > TOL {
                    curves.push((
                        id,
                        Curve::Arc {
                            c: sk.pos(c),
                            r,
                            t0: 0.0,
                            sweep: TAU,
                        },
                    ));
                }
            }
            Geom::Arc { c, s, e } => {
                let (c, s, e) = (sk.pos(c), sk.pos(s), sk.pos(e));
                let r = (s[0] - c[0]).hypot(s[1] - c[1]);
                let t0 = (s[1] - c[1]).atan2(s[0] - c[0]);
                let t1 = (e[1] - c[1]).atan2(e[0] - c[0]);
                let mut sweep = (t1 - t0).rem_euclid(TAU);
                if sweep < 1e-12 {
                    sweep = TAU;
                }
                if r > TOL {
                    curves.push((id, Curve::Arc { c, r, t0, sweep }));
                }
            }
        }
    }

    // 2. Split parameters
    let mut splits: Vec<Vec<f64>> = curves
        .iter()
        .map(|(_, c)| {
            if c.is_closed() {
                vec![0.0, 0.5]
            } else {
                vec![0.0, 1.0]
            }
        })
        .collect();
    // Conservative sweep broad phase. Arc bounds cover the full carrier
    // circle, so tangencies/endpoints are never excluded by approximation.
    let bounds: Vec<_> = curves
        .iter()
        .map(|(_, c)| match *c {
            Curve::Line { a, b } => (
                [a[0].min(b[0]) - TOL, a[1].min(b[1]) - TOL],
                [a[0].max(b[0]) + TOL, a[1].max(b[1]) + TOL],
            ),
            Curve::Arc { c, r, .. } => (
                [c[0] - r - TOL, c[1] - r - TOL],
                [c[0] + r + TOL, c[1] + r + TOL],
            ),
        })
        .collect();
    let mut order: Vec<_> = (0..curves.len()).collect();
    order.sort_by(|&a, &b| bounds[a].0[0].total_cmp(&bounds[b].0[0]));
    for (at, &i) in order.iter().enumerate() {
        for &j in &order[at + 1..] {
            if bounds[j].0[0] > bounds[i].1[0] {
                break;
            }
            if bounds[j].0[1] > bounds[i].1[1] || bounds[i].0[1] > bounds[j].1[1] {
                continue;
            }
            for p in intersect(&curves[i].1, &curves[j].1) {
                if let Some(t) = curves[i].1.param_of(p) {
                    splits[i].push(t);
                }
                if let Some(t) = curves[j].1.param_of(p) {
                    splits[j].push(t);
                }
            }
        }
    }

    // 3. Vertices and edges
    let mut verts: Vec<[f64; 2]> = Vec::new();
    let weld = 10. * TOL;
    let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut vid = |p: [f64; 2], verts: &mut Vec<[f64; 2]>| -> usize {
        let key = ((p[0] / weld).floor() as i64, (p[1] / weld).floor() as i64);
        let mut found = None;
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                if let Some(bucket) =
                    cells.get(&(key.0.saturating_add(dx), key.1.saturating_add(dy)))
                {
                    for &i in bucket {
                        let v = verts[i];
                        if (v[0] - p[0]).hypot(v[1] - p[1]) < weld
                            && found.is_none_or(|old| i < old)
                        {
                            found = Some(i);
                        }
                    }
                }
            }
        }
        if let Some(i) = found {
            i
        } else {
            let i = verts.len();
            verts.push(p);
            cells.entry(key).or_default().push(i);
            i
        }
    };
    let mut edges: Vec<Edge> = Vec::new();
    for (ci, (id, c)) in curves.iter().enumerate() {
        let ts = &mut splits[ci];
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let closed = c.is_closed();
        let n = ts.len();
        let count = if closed { n } else { n - 1 };
        for k in 0..count {
            let (t0, t1) = (
                ts[k],
                if closed && k + 1 == n {
                    ts[0] + 1.0
                } else {
                    ts[k + 1]
                },
            );
            let (pa, pb) = (c.at(t0), c.at(t1));
            let (va, vb) = (vid(pa, &mut verts), vid(pb, &mut verts));
            let seg = match *c {
                Curve::Line { .. } => Seg::Line {
                    a: verts[va],
                    b: verts[vb],
                },
                Curve::Arc { c, r, .. } => Seg::Arc {
                    c,
                    r,
                    a: verts[va],
                    b: verts[vb],
                    ccw: true,
                },
            };
            if va == vb && matches!(seg, Seg::Line { .. }) {
                continue;
            }
            edges.push(Edge {
                v: [va, vb],
                seg,
                entity: *id,
            });
        }
    }

    // 4. Prune dangling edges
    let mut alive = vec![true; edges.len()];
    let mut incident = vec![Vec::new(); verts.len()];
    for (i, e) in edges.iter().enumerate() {
        for &v in &e.v {
            incident[v].push(i);
        }
    }
    let mut degree: Vec<_> = incident.iter().map(Vec::len).collect();
    let mut dangling: VecDeque<_> = degree
        .iter()
        .enumerate()
        .filter_map(|(v, &d)| (d == 1).then_some(v))
        .collect();
    while let Some(v) = dangling.pop_front() {
        for &i in &incident[v] {
            if !alive[i] {
                continue;
            }
            alive[i] = false;
            for &endpoint in &edges[i].v {
                degree[endpoint] -= 1;
                if degree[endpoint] == 1 {
                    dangling.push_back(endpoint);
                }
            }
        }
    }

    // 5. Half-edges and angular order around vertices
    // half-edge h: edge h/2, forward if h even.
    let he_seg = |h: usize| {
        if h % 2 == 0 {
            edges[h / 2].seg
        } else {
            reversed(edges[h / 2].seg)
        }
    };
    let he_from = |h: usize| edges[h / 2].v[h % 2];
    let he_to = |h: usize| edges[h / 2].v[1 - h % 2];
    let mut out_of: Vec<Vec<(f64, usize)>> = vec![Vec::new(); verts.len()];
    for (i, _) in edges.iter().enumerate().filter(|(i, _)| alive[*i]) {
        for h in [2 * i, 2 * i + 1] {
            out_of[he_from(h)].push((leave_angle(&he_seg(h)), h));
        }
    }
    for l in &mut out_of {
        l.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let next = |h: usize| -> usize {
        let v = he_to(h);
        let twin = h ^ 1;
        let ring = &out_of[v];
        let k = ring.iter().position(|&(_, x)| x == twin).unwrap();
        // previous in CCW order = turn as far right as possible
        ring[(k + ring.len() - 1) % ring.len()].1
    };

    // 6. Trace cycles
    let mut used = vec![false; edges.len() * 2];
    let mut cycles: Vec<(Loop, f64, usize)> = Vec::new(); // loop, area, component vertex
    for start in 0..edges.len() * 2 {
        if used[start] || !alive[start / 2] {
            continue;
        }
        let mut lp = Loop {
            segs: Vec::new(),
            curves: Vec::new(),
        };
        let mut h = start;
        let mut guard = 0;
        loop {
            used[h] = true;
            lp.segs.push(he_seg(h));
            lp.curves.push(edges[h / 2].entity);
            h = next(h);
            guard += 1;
            if h == start || used[h] || guard > 100_000 {
                break;
            }
        }
        if h != start {
            continue; // malformed; skip
        }
        let area = lp.area();
        cycles.push((lp, area, he_from(start)));
    }

    // 7. Connected components (for hole assignment)
    let mut comp: Vec<usize> = (0..verts.len()).collect();
    fn find(c: &mut [usize], mut i: usize) -> usize {
        while c[i] != i {
            c[i] = c[c[i]];
            i = c[i];
        }
        i
    }
    for (e, _) in edges.iter().zip(&alive).filter(|(_, &a)| a) {
        let (a, b) = (find(&mut comp, e.v[0]), find(&mut comp, e.v[1]));
        comp[a] = b;
    }

    let mut profiles: Vec<(Profile, f64, usize, Vec<[f64; 2]>)> = Vec::new();
    for (lp, area, v) in cycles.iter() {
        if *area > 1e-12 {
            let poly = lp.polygon(1e-3);
            profiles.push((
                Profile {
                    outer: lp.clone(),
                    holes: vec![],
                },
                *area,
                find(&mut comp, *v),
                poly,
            ));
        }
    }
    for (lp, area, v) in cycles.iter() {
        if *area < -1e-12 {
            let c = find(&mut comp, *v);
            let probe = lp.segs[0].start();
            let host = profiles
                .iter()
                .enumerate()
                .filter(|(_, (_, _, pc, poly))| *pc != c && point_in_polygon(probe, poly))
                .min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
                .map(|(i, _)| i);
            if let Some(i) = host {
                profiles[i].0.holes.push(lp.clone());
            }
        }
    }
    // A hole can only belong to the innermost face; faces of a component nested
    // inside another face already exclude it, so nothing else to do.
    let _ = PI;
    profiles.into_iter().map(|p| p.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn areas(sk: &Sketch) -> Vec<f64> {
        let mut a: Vec<f64> = find_profiles(sk)
            .iter()
            .map(|p| (p.area() * 1000.0).round() / 1000.0)
            .collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        a
    }

    #[test]
    fn single_rectangle() {
        let mut s = Sketch::new();
        s.add_rectangle([0.0, 0.0], [4.0, 3.0]);
        assert_eq!(areas(&s), vec![12.0]);
    }

    #[test]
    fn circle_in_rectangle_makes_washer_and_disk() {
        let mut s = Sketch::new();
        s.add_rectangle([0.0, 0.0], [10.0, 10.0]);
        let c = s.add_point([5.0, 5.0]);
        s.add_circle(c, 2.0);
        let a = areas(&s);
        assert_eq!(a.len(), 2);
        let disk = PI * 4.0;
        assert!((a[0] - disk).abs() < 0.01, "{a:?}");
        assert!((a[1] - (100.0 - disk)).abs() < 0.01, "{a:?}");
        let profs = find_profiles(&s);
        let washer = profs.iter().find(|p| !p.holes.is_empty()).unwrap();
        assert!(!washer.contains([5.0, 5.0]) && washer.contains([1.0, 1.0]));
    }

    #[test]
    fn overlapping_circles_give_three_regions() {
        let mut s = Sketch::new();
        let c1 = s.add_point([0.0, 0.0]);
        let c2 = s.add_point([1.0, 0.0]);
        s.add_circle(c1, 1.0);
        s.add_circle(c2, 1.0);
        assert_eq!(areas(&s).len(), 3);
    }

    #[test]
    fn line_through_rectangle_splits_it_and_spurs_are_ignored() {
        let mut s = Sketch::new();
        s.add_rectangle([0.0, 0.0], [4.0, 2.0]);
        s.add_line_pts([2.0, -1.0], [2.0, 3.0]); // crosses, with spurs outside
        assert_eq!(areas(&s), vec![4.0, 4.0]);
    }

    #[test]
    fn slot_from_lines_and_arcs() {
        let mut s = Sketch::new();
        let p = [
            s.add_point([0.0, 0.0]),
            s.add_point([10.0, 0.0]),
            s.add_point([10.0, 4.0]),
            s.add_point([0.0, 4.0]),
        ];
        let c = [s.add_point([10.0, 2.0]), s.add_point([0.0, 2.0])];
        s.add_line(p[0], p[1]);
        s.add_arc(c[0], p[1], p[2]);
        s.add_line(p[2], p[3]);
        s.add_arc(c[1], p[3], p[0]);
        let a = areas(&s);
        assert_eq!(a.len(), 1);
        assert!((a[0] - (40.0 + PI * 4.0)).abs() < 0.01, "{a:?}");
    }
}
