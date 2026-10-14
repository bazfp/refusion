use crate::sab::{Sab, Tok};
use refusion_kernel::builder::{BSplineCurve, BSplineSurface, Builder};
use refusion_kernel::Shape;
use std::collections::HashMap;

struct Cur<'a> {
    t: &'a [Tok],
    i: usize,
}

impl<'a> Cur<'a> {
    fn new(t: &'a [Tok]) -> Self {
        Cur { t, i: 0 }
    }
    /// Skip the common record prefix: attrib pointer, history int, pointer.
    fn entity(t: &'a [Tok]) -> Self {
        let mut c = Cur::new(t);
        if matches!(c.peek(), Some(Tok::Ptr(_))) {
            c.i += 1;
        }
        if matches!(c.peek(), Some(Tok::Int(_))) {
            c.i += 1;
        }
        if matches!(c.peek(), Some(Tok::Ptr(_))) {
            c.i += 1;
        }
        c
    }
    fn peek(&self) -> Option<&'a Tok> {
        self.t.get(self.i)
    }
    fn next(&mut self) -> Option<&'a Tok> {
        let t = self.t.get(self.i);
        self.i += 1;
        t
    }
    fn ptr(&mut self) -> Option<i64> {
        while let Some(t) = self.next() {
            if let Tok::Ptr(p) = t {
                return Some(*p);
            }
        }
        None
    }
    fn dbl(&mut self) -> Option<f64> {
        while let Some(t) = self.next() {
            match t {
                Tok::Dbl(v) => return Some(*v),
                Tok::Int(v) => return Some(*v as f64),
                _ => {}
            }
        }
        None
    }
    fn boolean(&mut self) -> Option<bool> {
        while let Some(t) = self.next() {
            if let Tok::Bool(b) = t {
                return Some(*b);
            }
        }
        None
    }
    fn vec3(&mut self) -> Option<[f64; 3]> {
        while let Some(t) = self.next() {
            if let Tok::Pos(v) | Tok::Vec3(v) = t {
                return Some(*v);
            }
        }
        None
    }
    /// An interval: two ends, each `F` (unbounded) or `T value`.
    fn interval(&mut self) {
        for _ in 0..2 {
            if let Some(Tok::Bool(true)) = self.next() {
                self.i += 1;
            }
        }
    }
    /// Next exact double (no skipping) — for B-spline data.
    fn exact_dbl(&mut self) -> Option<f64> {
        match self.next()? {
            Tok::Dbl(v) => Some(*v),
            Tok::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    fn exact_int(&mut self) -> Option<i64> {
        match self.next()? {
            Tok::Int(v) => Some(*v),
            _ => None,
        }
    }
}

fn scale3(v: [f64; 3], k: f64) -> [f64; 3] {
    [v[0] * k, v[1] * k, v[2] * k]
}

/// Resolve `{ ref n }` and return the tokens of the sub-type that follows
/// the cursor's `Open`.
fn subtype_body<'a>(sab: &'a Sab, toks: &'a [Tok], open: usize) -> Option<&'a [Tok]> {
    let body = crate::sab::block_body(toks, open)?;
    if let (Some(Tok::Id(r)), Some(Tok::Int(n))) = (body.first(), body.get(1)) {
        if r == "ref" {
            return sab.subtype(usize::try_from(*n).ok()?);
        }
    }
    Some(body)
}

fn find_bs3(t: &[Tok]) -> Option<usize> {
    let mut depth = 0usize;
    for (i, tok) in t.iter().enumerate() {
        match tok {
            Tok::Open => depth += 1,
            Tok::Close => depth = depth.checked_sub(1)?,
            Tok::Str(s) | Tok::Id(s) if depth == 0 && (s == "nubs" || s == "nurbs") => {
                return Some(i)
            }
            _ => {}
        }
    }
    None
}

/// Counts and pole dimensions must fit the actual remaining token stream before
/// allocating. Invalid splines take the existing reported approximation path.
fn spline_knots(c: &mut Cur, n: usize, degree: i32) -> Option<(Vec<f64>, Vec<i32>)> {
    if n < 2 || n > c.t.len().saturating_sub(c.i) / 2 || degree <= 0 {
        return None;
    }
    let mut knots = Vec::with_capacity(n);
    let mut mults = Vec::with_capacity(n);
    for _ in 0..n {
        let knot = c.exact_dbl()?;
        let mult = i32::try_from(c.exact_int()?).ok()?;
        if !knot.is_finite()
            || knots.last().is_some_and(|last| knot <= *last)
            || mult <= 0
            || mult > degree
        {
            return None;
        }
        knots.push(knot);
        mults.push(mult);
    }
    Some((knots, mults))
}

fn spline_pole_count(mults: &[i32], degree: i32) -> Option<usize> {
    let sum = mults
        .iter()
        .try_fold(0i64, |sum, m| sum.checked_add(*m as i64))?;
    let n = usize::try_from(sum.checked_sub(degree as i64)?.checked_add(1)?).ok()?;
    (n > 0).then_some(n)
}

/// ACIS bs3 curve → OCCT B-spline (scaled to mm).
fn bs3_curve(t: &[Tok], unit: f64) -> Option<BSplineCurve> {
    let at = find_bs3(t)?;
    let rational = matches!(&t[at], Tok::Str(s) | Tok::Id(s) if s == "nurbs");
    let mut c = Cur::new(&t[at + 1..]);
    let degree = i32::try_from(c.exact_int()?).ok()?;
    let (closure, nk) = match c.next()? {
        Tok::Enum(closure) => (*closure, c.exact_int()?),
        Tok::Int(first) => {
            if matches!(c.peek(), Some(Tok::Int(_))) {
                (*first, c.exact_int()?)
            } else {
                (0, *first)
            }
        }
        _ => return None,
    };
    if !(0..=2).contains(&closure) {
        return None;
    }
    let (knots, mut mults) = spline_knots(&mut c, usize::try_from(nk).ok()?, degree)?;
    let np = spline_pole_count(&mults, degree)?;
    let dim = if rational { 4 } else { 3 };
    if np > c.t.len().saturating_sub(c.i) / dim {
        return None;
    }
    let mut poles = Vec::with_capacity(np);
    let mut weights = Vec::new();
    for _ in 0..np {
        let mut p = [0.0; 4];
        for x in p.iter_mut().take(dim) {
            *x = c.exact_dbl()?;
            if !x.is_finite() {
                return None;
            }
        }
        let pole = scale3([p[0], p[1], p[2]], unit);
        if !pole.iter().all(|x| x.is_finite()) || (rational && p[3] <= 0.) {
            return None;
        }
        poles.push(pole);
        if rational {
            weights.push(p[3]);
        }
    }
    // ACIS stores end multiplicity = degree; OCCT clamped ends need degree + 1.
    let periodic = closure == 2;
    if !periodic {
        if let Some(m) = mults.first_mut() {
            *m = m.checked_add(1)?;
        }
        if let Some(m) = mults.last_mut() {
            *m = m.checked_add(1)?;
        }
    }
    Some(BSplineCurve {
        degree,
        knots,
        mults,
        poles,
        weights: rational.then_some(weights),
        periodic,
    })
}

/// ACIS bs3 surface → OCCT B-spline surface (scaled to mm).
fn bs3_surface(t: &[Tok], unit: f64) -> Option<BSplineSurface> {
    let at = find_bs3(t)?;
    let rational = matches!(&t[at], Tok::Str(s) | Tok::Id(s) if s == "nurbs");
    let mut c = Cur::new(&t[at + 1..]);
    let du = i32::try_from(c.exact_int()?).ok()?;
    let dv = i32::try_from(c.exact_int()?).ok()?;
    // optional form strings, then closure/singularity ints, then the two knot counts
    let mut ints = Vec::new();
    while let Some(tok) = c.peek() {
        match tok {
            Tok::Int(v) => {
                ints.push(*v);
                c.i += 1;
            }
            Tok::Str(_) | Tok::Id(_) | Tok::Enum(_) => c.i += 1,
            _ => break,
        }
    }
    if ints.len() < 2 {
        return None;
    }
    let (nku, nkv) = (
        usize::try_from(ints[ints.len() - 2]).ok()?,
        usize::try_from(ints[ints.len() - 1]).ok()?,
    );
    let (cu, cv) = if ints.len() >= 4 {
        (ints[0], ints[1])
    } else {
        (0, 0)
    };
    let (ku, mut mu) = spline_knots(&mut c, nku, du)?;
    let (kv, mut mv) = spline_knots(&mut c, nkv, dv)?;
    let nu = spline_pole_count(&mu, du)?;
    let nv = spline_pole_count(&mv, dv)?;
    let dim = if rational { 4 } else { 3 };
    let np = nu.checked_mul(nv)?;
    if np > c.t.len().saturating_sub(c.i) / dim {
        return None;
    }
    let mut poles = Vec::with_capacity(np);
    let mut weights = Vec::new();
    for _ in 0..np {
        let mut p = [0.0; 4];
        for x in p.iter_mut().take(dim) {
            *x = c.exact_dbl()?;
            if !x.is_finite() {
                return None;
            }
        }
        let pole = scale3([p[0], p[1], p[2]], unit);
        if !pole.iter().all(|x| x.is_finite()) || (rational && p[3] <= 0.) {
            return None;
        }
        poles.push(pole);
        if rational {
            weights.push(p[3]);
        }
    }
    let (uper, vper) = (cu == 2, cv == 2);
    for (m, per) in [(&mut mu, uper), (&mut mv, vper)] {
        if !per {
            if let Some(x) = m.first_mut() {
                *x = x.checked_add(1)?;
            }
            if let Some(x) = m.last_mut() {
                *x = x.checked_add(1)?;
            }
        }
    }
    // ACIS stores control points with v varying fastest for each u row.
    Some(BSplineSurface {
        du,
        dv,
        ku,
        mu,
        kv,
        mv,
        nu,
        nv,
        poles,
        weights: rational.then_some(weights),
        uper,
        vper,
    })
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub bodies: usize,
    pub faces: usize,
    pub failed_faces: usize,
    pub filled_faces: usize,
    pub approx_edges: usize,
}

pub struct Decoder<'a> {
    sab: &'a Sab,
    b: Builder,
    unit: f64,
    verts: HashMap<i64, i32>,
    edges: HashMap<i64, i32>,
    curves: HashMap<i64, Option<(i32, bool)>>,
    surfs: HashMap<i64, Option<(i32, bool)>>,
    pub stats: Stats,
}

impl<'a> Decoder<'a> {
    pub fn new(sab: &'a Sab) -> Self {
        Decoder {
            sab,
            b: Builder::new(),
            unit: sab.unit_mm,
            verts: HashMap::new(),
            edges: HashMap::new(),
            curves: HashMap::new(),
            surfs: HashMap::new(),
            stats: Stats::default(),
        }
    }

    fn rec(&self, p: i64) -> Option<&'a crate::sab::Record> {
        usize::try_from(p)
            .ok()
            .and_then(|p| self.sab.records.get(p))
    }

    fn vertex(&mut self, v: i64) -> Option<i32> {
        if let Some(&id) = self.verts.get(&v) {
            return Some(id);
        }
        let r = self.rec(v)?;
        let mut c = Cur::entity(&r.toks);
        let mut last = None;
        while let Some(p) = c.ptr() {
            last = Some(p);
        }
        let pt = self.rec(last?)?;
        let pos = Cur::entity(&pt.toks).vec3()?;
        let id = self.b.vertex(scale3(pos, self.unit));
        self.verts.insert(v, id);
        Some(id)
    }

    /// Builder curve id and whether the ACIS curve runs against its own data.
    fn curve(&mut self, c: i64) -> Option<(i32, bool)> {
        if let Some(v) = self.curves.get(&c) {
            return *v;
        }
        let r = self.rec(c)?;
        let u = self.unit;
        let res = (|| -> Option<(i32, bool)> {
            if r.name.ends_with("straight-curve") {
                let mut k = Cur::entity(&r.toks);
                let (p, d) = (k.vec3()?, k.vec3()?);
                Some((self.b.line(scale3(p, u), d).ok()?, false))
            } else if r.name.ends_with("ellipse-curve") {
                let mut k = Cur::entity(&r.toks);
                let (cen, n, maj) = (k.vec3()?, k.vec3()?, k.vec3()?);
                let ratio = k.dbl()?;
                Some((
                    self.b
                        .ellipse(scale3(cen, u), n, scale3(maj, u), ratio)
                        .ok()?,
                    false,
                ))
            } else if r.name.ends_with("intcurve-curve") {
                let mut k = Cur::entity(&r.toks);
                let rev = k.boolean().unwrap_or(false);
                let open = r.toks.iter().position(|t| *t == Tok::Open)?;
                let body = subtype_body(self.sab, &r.toks, open)?;
                // Only the explicit exact-curve carrier has a verified BS3
                // interpretation. Intersection/deformation carriers include
                // approximation geometry and parent surfaces; interpreting that
                // BS3 as an exact edge also causes pathological native repairs.
                if !matches!(body.first(), Some(Tok::Id(n)) if n == "exact_int_cur") {
                    return None;
                }
                let bs = bs3_curve(body, u)?;
                // Enable the verified single-span family first. Multi-span
                // exact curves on the motor fixture trigger minutes of native
                // wire repair against surfaces whose control-net layout still
                // needs verification. Keep the existing, reported line fallback
                // until those curve/surface combinations can be imported safely.
                if bs.knots.len() != 2 || bs.periodic {
                    return None;
                }
                Some((self.b.bspline_curve(&bs).ok()?, rev))
            } else {
                None
            }
        })();
        self.curves.insert(c, res);
        res
    }

    /// Builder surface id and whether its normal is reversed w.r.t. OCCT's.
    fn surface(&mut self, s: i64) -> Option<(i32, bool)> {
        if let Some(v) = self.surfs.get(&s) {
            return *v;
        }
        let r = self.rec(s)?;
        let u = self.unit;
        let res = (|| -> Option<(i32, bool)> {
            let mut k = Cur::entity(&r.toks);
            if r.name.ends_with("plane-surface") {
                let (p, n, ud) = (k.vec3()?, k.vec3()?, k.vec3()?);
                Some((self.b.plane(scale3(p, u), n, ud).ok()?, false))
            } else if r.name.ends_with("cone-surface") {
                let (c, n, maj) = (k.vec3()?, k.vec3()?, k.vec3()?);
                let _ratio = k.dbl()?;
                k.interval();
                let (sina, cosa) = (k.dbl()?, k.dbl()?);
                Some((
                    self.b
                        .cone(scale3(c, u), n, scale3(maj, u), sina, cosa)
                        .ok()?,
                    cosa < 0.0,
                ))
            } else if r.name.ends_with("sphere-surface") {
                let c = k.vec3()?;
                let rad = k.dbl()?;
                let (ud, pole) = (k.vec3()?, k.vec3()?);
                Some((
                    self.b.sphere(scale3(c, u), rad * u, ud, pole).ok()?,
                    rad < 0.0,
                ))
            } else if r.name.ends_with("torus-surface") {
                let (c, n) = (k.vec3()?, k.vec3()?);
                let (big, small) = (k.dbl()?, k.dbl()?);
                let ud = k.vec3()?;
                Some((
                    self.b.torus(scale3(c, u), n, big * u, small * u, ud).ok()?,
                    small < 0.0,
                ))
            } else if r.name.ends_with("spline-surface") {
                let rev = k.boolean().unwrap_or(false);
                let open = r.toks.iter().position(|t| *t == Tok::Open)?;
                let body = subtype_body(self.sab, &r.toks, open)?;
                let is_exact = matches!(body.first(), Some(Tok::Id(n)) if n == "exact_spl_sur");
                if !is_exact {
                    return None; // procedural spline: fall back to a filled face
                }
                let bs = bs3_surface(body, u)?;
                Some((self.b.bspline_surface(&bs).ok()?, rev))
            } else {
                None
            }
        })();
        self.surfs.insert(s, res);
        res
    }

    fn edge(&mut self, e: i64) -> Option<i32> {
        if let Some(&id) = self.edges.get(&e) {
            return Some(id);
        }
        let r = self.rec(e)?;
        let mut k = Cur::entity(&r.toks);
        let vs = k.ptr()?;
        if matches!(k.peek(), Some(Tok::Dbl(_))) {
            k.i += 1;
        }
        let ve = k.ptr()?;
        if matches!(k.peek(), Some(Tok::Dbl(_))) {
            k.i += 1;
        }
        let _coedge = k.ptr()?;
        let curve = k.ptr()?;
        let sense = k.boolean().unwrap_or(false);
        let (v1, v2) = (self.vertex(vs)?, self.vertex(ve)?);
        let (cid, crev) = match self.curve(curve) {
            Some(x) => x,
            None => {
                self.stats.approx_edges += 1;
                (-1, false)
            }
        };
        let id = match self.b.edge(cid, v1, v2, sense ^ crev) {
            Ok(id) => id,
            Err(_) => {
                self.stats.approx_edges += 1;
                self.b.edge(-1, v1, v2, false).ok()?
            }
        };
        self.edges.insert(e, id);
        Some(id)
    }

    fn face(&mut self, f: i64) -> Option<i32> {
        let r = self.rec(f)?;
        let mut k = Cur::entity(&r.toks);
        let _next = k.ptr()?;
        let first_loop = k.ptr()?;
        let _shell = k.ptr()?;
        let _sub = k.ptr()?;
        let surf = k.ptr()?;
        let sense = k.boolean().unwrap_or(false);
        let mut loops: Vec<Vec<(i32, bool)>> = Vec::new();
        let mut lp = first_loop;
        let mut seen_loops = std::collections::HashSet::new();
        while lp >= 0 && seen_loops.insert(lp) {
            let lr = self.rec(lp)?;
            let mut lk = Cur::entity(&lr.toks);
            let next_loop = lk.ptr()?;
            let first_co = lk.ptr()?;
            let mut ring = Vec::new();
            let mut co = first_co;
            let mut g2 = 0;
            loop {
                let cr = self.rec(co)?;
                let mut ck = Cur::entity(&cr.toks);
                let next = ck.ptr()?;
                let _prev = ck.ptr()?;
                let _partner = ck.ptr()?;
                let edge = ck.ptr()?;
                let csense = ck.boolean().unwrap_or(false);
                ring.push((self.edge(edge)?, csense));
                co = next;
                g2 += 1;
                if co == first_co || co < 0 || g2 > 100_000 {
                    break;
                }
            }
            loops.push(ring);
            lp = next_loop;
        }
        let (sid, srev) = match self.surface(surf) {
            Some(x) => x,
            None => {
                self.stats.filled_faces += 1;
                (-1, false)
            }
        };
        self.stats.faces += 1;
        match self.b.face(sid, sense ^ srev, &loops) {
            Ok(id) => Some(id),
            Err(_) => {
                self.stats.failed_faces += 1;
                None
            }
        }
    }

    fn transform(&self, t: i64) -> Option<[f64; 12]> {
        let r = self.rec(t)?;
        let mut k = Cur::new(&r.toks);
        let (a, b, c, tr) = (k.vec3()?, k.vec3()?, k.vec3()?, k.vec3()?);
        let s = k.dbl().unwrap_or(1.0);
        // ACIS: p' = (p · M) · s + t (row vectors) → OCCT column form.
        Some([
            a[0] * s,
            b[0] * s,
            c[0] * s,
            tr[0] * self.unit,
            a[1] * s,
            b[1] * s,
            c[1] * s,
            tr[1] * self.unit,
            a[2] * s,
            b[2] * s,
            c[2] * s,
            tr[2] * self.unit,
        ])
    }

    /// Every body in the stream, one shape per lump.
    pub fn bodies(&mut self) -> Vec<Shape> {
        let mut out = Vec::new();
        let body_idx: Vec<usize> = self
            .sab
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.name == "body")
            .map(|(i, _)| i)
            .collect();
        for bi in body_idx {
            let r = &self.sab.records[bi];
            let mut k = Cur::entity(&r.toks);
            let Some(mut lump) = k.ptr() else { continue };
            let _wire = k.ptr();
            let trsf = k.ptr().and_then(|t| self.transform(t));
            let mut seen_lumps = std::collections::HashSet::new();
            while lump >= 0 && seen_lumps.insert(lump) {
                let Some(lr) = self.rec(lump) else { break };
                let mut lk = Cur::entity(&lr.toks);
                let next_lump = lk.ptr().unwrap_or(-1);
                let mut shell = lk.ptr().unwrap_or(-1);
                let mut faces = Vec::new();
                // Pointer chains are walked with visited sets: some streams
                // link the last element back to the first.
                let mut seen_shells = std::collections::HashSet::new();
                let mut seen_faces = std::collections::HashSet::new();
                while shell >= 0 && seen_shells.insert(shell) {
                    let Some(sr) = self.rec(shell) else { break };
                    let mut sk = Cur::entity(&sr.toks);
                    let next_shell = sk.ptr().unwrap_or(-1);
                    let _sub = sk.ptr();
                    let mut face = sk.ptr().unwrap_or(-1);
                    while face >= 0 && seen_faces.insert(face) {
                        let next_face = self
                            .rec(face)
                            .and_then(|fr| Cur::entity(&fr.toks).ptr())
                            .unwrap_or(-1);
                        if let Some(id) = self.face(face) {
                            faces.push(id);
                        }
                        face = next_face;
                    }
                    shell = next_shell;
                }
                if !faces.is_empty() {
                    if let Ok(s) = self.b.body(&faces, trsf) {
                        self.stats.bodies += 1;
                        out.push(s);
                    }
                }
                lump = next_lump;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rational quadratic quarter circle, encoded with the enum closure used
    /// by actual ASM exact_int_cur records. Its native construction must work.
    fn quarter_circle(closure: Tok) -> Vec<Tok> {
        let mut t = vec![
            Tok::Id("exact_int_cur".into()),
            Tok::Int(22601),
            Tok::Enum(0),
            Tok::Id("nurbs".into()),
            Tok::Int(2),
            closure,
            Tok::Int(2),
            Tok::Dbl(0.),
            Tok::Int(2),
            Tok::Dbl(1.),
            Tok::Int(2),
        ];
        for p in [
            [1., 0., 0., 1.],
            [1., 1., 0., std::f64::consts::FRAC_1_SQRT_2],
            [0., 1., 0., 1.],
        ] {
            t.extend(p.into_iter().map(Tok::Dbl));
        }
        t
    }

    #[test]
    fn actual_enum_closure_builds_a_rational_curve_without_line_fallback() {
        for closure in [Tok::Enum(0), Tok::Int(0)] {
            let bs = bs3_curve(&quarter_circle(closure), 10.).unwrap();
            assert_eq!(bs.degree, 2);
            assert_eq!(bs.mults, [3, 3]);
            assert_eq!(bs.poles, [[10., 0., 0.], [10., 10., 0.], [0., 10., 0.]]);
            assert_eq!(
                bs.weights.as_ref().unwrap()[1],
                std::f64::consts::FRAC_1_SQRT_2
            );
            assert!(!bs.periodic);
            assert!(Builder::new().bspline_curve(&bs).is_ok());
        }
    }

    #[test]
    fn exact_curve_import_does_not_promote_intersection_approximations_to_exact_edges() {
        for (kind, supported) in [
            ("exact_int_cur", true),
            ("int_int_cur", false),
            ("defm_int_cur", false),
        ] {
            let mut body = quarter_circle(Tok::Enum(0));
            body[0] = Tok::Id(kind.into());
            let mut toks = vec![
                Tok::Ptr(-1),
                Tok::Int(-1),
                Tok::Ptr(-1),
                Tok::Bool(false),
                Tok::Open,
            ];
            toks.extend(body);
            toks.push(Tok::Close);
            let sab = Sab {
                records: vec![crate::sab::Record {
                    name: "intcurve-curve".into(),
                    toks,
                }],
                subtypes: vec![(0, 4)],
                unit_mm: 10.,
            };
            assert_eq!(Decoder::new(&sab).curve(0).is_some(), supported, "{kind}");
        }
    }

    #[test]
    fn invalid_spline_counts_values_and_nested_geometry_are_not_guessed() {
        let valid = quarter_circle(Tok::Enum(0));
        let mut bad = valid.clone();
        bad[6] = Tok::Int(i64::MAX);
        assert!(bs3_curve(&bad, 10.).is_none());
        bad = valid.clone();
        bad[8] = Tok::Int(-1);
        assert!(bs3_curve(&bad, 10.).is_none());
        bad = valid.clone();
        bad[11] = Tok::Dbl(f64::NAN);
        assert!(bs3_curve(&bad, 10.).is_none());
        let mut nested = vec![Tok::Id("procedural".into()), Tok::Open];
        nested.extend(valid);
        nested.push(Tok::Close);
        assert!(
            bs3_curve(&nested, 10.).is_none(),
            "a child curve is not its parent geometry"
        );
    }

    #[test]
    fn multi_span_exact_curves_keep_the_reported_fallback_until_surface_layout_is_verified() {
        let mut body = quarter_circle(Tok::Enum(0));
        body[6] = Tok::Int(3);
        body.splice(9..9, [Tok::Dbl(0.5), Tok::Int(2)]);
        for p in [[-1., 1., 0., 1.], [-1., 0., 0., 1.]] {
            body.extend(p.into_iter().map(Tok::Dbl));
        }
        assert_eq!(bs3_curve(&body, 10.).unwrap().knots.len(), 3);
        let mut toks = vec![
            Tok::Ptr(-1),
            Tok::Int(-1),
            Tok::Ptr(-1),
            Tok::Bool(false),
            Tok::Open,
        ];
        toks.extend(body);
        toks.push(Tok::Close);
        let sab = Sab {
            records: vec![crate::sab::Record {
                name: "intcurve-curve".into(),
                toks,
            }],
            subtypes: vec![(0, 4)],
            unit_mm: 10.,
        };
        assert!(Decoder::new(&sab).curve(0).is_none());
    }

    #[test]
    fn invalid_subtype_references_do_not_fall_through_to_neighbor_data() {
        let sab = Sab {
            records: vec![],
            subtypes: vec![],
            unit_mm: 10.,
        };
        let toks = [Tok::Open, Tok::Id("ref".into()), Tok::Int(-1), Tok::Close];
        assert!(subtype_body(&sab, &toks, 0).is_none());
        let toks = [Tok::Open, Tok::Id("ref".into()), Tok::Int(0), Tok::Close];
        assert!(subtype_body(&sab, &toks, 0).is_none());
    }
}
