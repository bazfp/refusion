use crate::dual::{Dual, Real, MAXV};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

pub type P = usize;

/// A 2D point as a pair of parameter ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pt(pub P, pub P);

/// Where a radius comes from: a free parameter (circles) or the distance
/// between the centre and a defining point (arcs, which are centre/start/end).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rad {
    Param(P),
    Pts(Pt, Pt),
}

/// Primitive residual equations. Higher-level sketch constraints lower to one
/// or more of these (see `sketch.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Eq {
    RadiusDiff {
        r1: Rad,
        r2: Rad,
        d: f64,
    },
    OffsetPoint {
        a: Pt,
        b: Pt,
        p: Pt,
        end: bool,
        axis: u8,
        d: f64,
    },
    /// a - b - k = 0 (coincident / horizontal / vertical / concentric / h-v dimensions)
    Lin {
        a: P,
        b: P,
        k: f64,
    },
    /// |p - q| - d = 0
    Dist {
        p: Pt,
        q: Pt,
        d: f64,
    },
    /// signed distance of p from line (a,b) - d = 0
    PtLine {
        p: Pt,
        a: Pt,
        b: Pt,
        d: f64,
    },
    /// |p - c| - r = 0
    PtCircle {
        p: Pt,
        c: Pt,
        r: Rad,
    },
    /// normalised cross product of the two directions = 0
    Parallel {
        a1: Pt,
        a2: Pt,
        b1: Pt,
        b2: Pt,
    },
    /// normalised dot product = 0
    Perp {
        a1: Pt,
        a2: Pt,
        b1: Pt,
        b2: Pt,
    },
    /// signed angle from line a to line b - theta = 0 (wrapped)
    Angle {
        a1: Pt,
        a2: Pt,
        b1: Pt,
        b2: Pt,
        theta: f64,
    },
    /// |a| - |b| = 0
    EqualLen {
        a1: Pt,
        a2: Pt,
        b1: Pt,
        b2: Pt,
    },
    /// r1 - r2 = 0
    EqualRad {
        r1: Rad,
        r2: Rad,
    },
    /// r - d = 0
    RadiusIs {
        r: Rad,
        d: f64,
    },
    /// signed distance of circle centre from the line - side * r = 0
    TanLineCircle {
        a: Pt,
        b: Pt,
        c: Pt,
        r: Rad,
        side: f64,
    },
    /// |c1 - c2| - (r1 + r2) or |c1 - c2| - |r1 - r2|
    TanCircles {
        c1: Pt,
        r1: Rad,
        c2: Pt,
        r2: Rad,
        internal: bool,
    },
    /// p_axis - (a_axis + b_axis) / 2 = 0   (axis: 0 = x, 1 = y)
    Mid {
        p: Pt,
        a: Pt,
        b: Pt,
        axis: u8,
    },
    /// midpoint of (p,q) lies on line (a,b)
    SymMid {
        p: Pt,
        q: Pt,
        a: Pt,
        b: Pt,
    },
    /// (q - p) · (b - a) = 0, normalised
    SymPerp {
        p: Pt,
        q: Pt,
        a: Pt,
        b: Pt,
    },
}

/// Access to parameter values, generic over the scalar type.
pub trait Vars<T: Real> {
    fn get(&mut self, p: P) -> T;
}

#[inline]
fn pt<T: Real, V: Vars<T>>(v: &mut V, p: Pt) -> (T, T) {
    (v.get(p.0), v.get(p.1))
}

#[inline]
fn rad<T: Real, V: Vars<T>>(v: &mut V, r: Rad) -> T {
    match r {
        Rad::Param(p) => v.get(p),
        Rad::Pts(c, s) => {
            let (cx, cy) = pt(v, c);
            let (sx, sy) = pt(v, s);
            let (dx, dy) = (sx - cx, sy - cy);
            (dx * dx + dy * dy).sqrt()
        }
    }
}

#[inline]
fn dir<T: Real, V: Vars<T>>(v: &mut V, a: Pt, b: Pt) -> (T, T) {
    let (ax, ay) = pt(v, a);
    let (bx, by) = pt(v, b);
    (bx - ax, by - ay)
}

#[inline]
fn norm<T: Real>(x: T, y: T) -> T {
    (x * x + y * y).sqrt()
}

fn wrap_pi(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut r = (a + std::f64::consts::PI) % tau;
    if r < 0.0 {
        r += tau;
    }
    r - std::f64::consts::PI
}

impl Eq {
    /// Evaluate the residual. Written once; used with `f64` for values and
    /// with [`Dual`] for exact derivatives.
    pub fn residual<T: Real, V: Vars<T>>(&self, v: &mut V) -> T {
        match *self {
            Eq::RadiusDiff { r1, r2, d } => rad(v, r1) - rad(v, r2) - T::c(d),
            Eq::OffsetPoint {
                a,
                b,
                p,
                end,
                axis,
                d,
            } => {
                let (ax, ay, bx, by, px, py) = (
                    v.get(a.0),
                    v.get(a.1),
                    v.get(b.0),
                    v.get(b.1),
                    v.get(p.0),
                    v.get(p.1),
                );
                let l = norm(bx - ax, by - ay);
                if axis == 0 {
                    px - if end { bx } else { ax } + (by - ay) / l * T::c(d)
                } else {
                    py - if end { by } else { ay } - (bx - ax) / l * T::c(d)
                }
            }

            Eq::Lin { a, b, k } => v.get(a) - v.get(b) - T::c(k),
            Eq::Dist { p, q, d } => {
                let (dx, dy) = dir(v, p, q);
                norm(dx, dy) - T::c(d)
            }
            Eq::PtLine { p, a, b, d } => {
                let (px, py) = pt(v, p);
                let (ax, ay) = pt(v, a);
                let (bx, by) = pt(v, b);
                let (ux, uy) = (bx - ax, by - ay);
                let cross = ux * (py - ay) - uy * (px - ax);
                cross / norm(ux, uy) - T::c(d)
            }
            Eq::PtCircle { p, c, r } => {
                let (dx, dy) = dir(v, c, p);
                norm(dx, dy) - rad(v, r)
            }
            Eq::Parallel { a1, a2, b1, b2 } => {
                let (ux, uy) = dir(v, a1, a2);
                let (wx, wy) = dir(v, b1, b2);
                (ux * wy - uy * wx) / (norm(ux, uy) * norm(wx, wy))
            }
            Eq::Perp { a1, a2, b1, b2 } => {
                let (ux, uy) = dir(v, a1, a2);
                let (wx, wy) = dir(v, b1, b2);
                (ux * wx + uy * wy) / (norm(ux, uy) * norm(wx, wy))
            }
            Eq::Angle {
                a1,
                a2,
                b1,
                b2,
                theta,
            } => {
                let (ux, uy) = dir(v, a1, a2);
                let (wx, wy) = dir(v, b1, b2);
                let ang = (ux * wy - uy * wx).atan2(ux * wx + uy * wy);
                // Wrap the value but keep the derivative.
                let raw = ang.val() - theta;
                ang - T::c(theta) - T::c(raw - wrap_pi(raw))
            }
            Eq::EqualLen { a1, a2, b1, b2 } => {
                let (ux, uy) = dir(v, a1, a2);
                let (wx, wy) = dir(v, b1, b2);
                norm(ux, uy) - norm(wx, wy)
            }
            Eq::EqualRad { r1, r2 } => rad(v, r1) - rad(v, r2),
            Eq::RadiusIs { r, d } => rad(v, r) - T::c(d),
            Eq::TanLineCircle { a, b, c, r, side } => {
                let (cx, cy) = pt(v, c);
                let (ax, ay) = pt(v, a);
                let (bx, by) = pt(v, b);
                let (ux, uy) = (bx - ax, by - ay);
                let cross = ux * (cy - ay) - uy * (cx - ax);
                cross / norm(ux, uy) - rad(v, r).scale(side)
            }
            Eq::TanCircles {
                c1,
                r1,
                c2,
                r2,
                internal,
            } => {
                let (dx, dy) = dir(v, c1, c2);
                let d = norm(dx, dy);
                let (ra, rb) = (rad(v, r1), rad(v, r2));
                if internal {
                    // |r1 - r2| with the sign frozen at evaluation time
                    let s = if ra.val() >= rb.val() { 1.0 } else { -1.0 };
                    d - (ra - rb).scale(s)
                } else {
                    d - (ra + rb)
                }
            }
            Eq::Mid { p, a, b, axis } => {
                let (pp, aa, bb) = if axis == 0 {
                    (v.get(p.0), v.get(a.0), v.get(b.0))
                } else {
                    (v.get(p.1), v.get(a.1), v.get(b.1))
                };
                pp - (aa + bb).scale(0.5)
            }
            Eq::SymMid { p, q, a, b } => {
                let (px, py) = pt(v, p);
                let (qx, qy) = pt(v, q);
                let (ax, ay) = pt(v, a);
                let (bx, by) = pt(v, b);
                let (mx, my) = ((px + qx).scale(0.5), (py + qy).scale(0.5));
                let (ux, uy) = (bx - ax, by - ay);
                (ux * (my - ay) - uy * (mx - ax)) / norm(ux, uy)
            }
            Eq::SymPerp { p, q, a, b } => {
                let (dx, dy) = dir(v, p, q);
                let (ux, uy) = dir(v, a, b);
                (dx * ux + dy * uy) / norm(ux, uy)
            }
        }
    }

    /// Parameter ids touched by this equation (order of first use).
    pub fn params(&self, buf: &mut Vec<P>) {
        struct Rec<'a>(&'a mut Vec<P>);
        impl Vars<f64> for Rec<'_> {
            fn get(&mut self, p: P) -> f64 {
                if !self.0.contains(&p) {
                    self.0.push(p);
                }
                // Any non-degenerate value: only the access pattern matters.
                1.0 + p as f64 * 0.37
            }
        }
        buf.clear();
        let _ = self.residual::<f64, _>(&mut Rec(buf));
    }

    /// Remap parameter ids (used when merging coincident points).
    pub fn remap(&mut self, f: &dyn Fn(P) -> P) {
        let fp = |p: &mut Pt| {
            p.0 = f(p.0);
            p.1 = f(p.1);
        };
        let fr = |r: &mut Rad| match r {
            Rad::Param(p) => *p = f(*p),
            Rad::Pts(a, b) => {
                a.0 = f(a.0);
                a.1 = f(a.1);
                b.0 = f(b.0);
                b.1 = f(b.1);
            }
        };
        match self {
            Eq::RadiusDiff { r1, r2, .. } => {
                fr(r1);
                fr(r2);
            }
            Eq::OffsetPoint { a, b, p, .. } => {
                fp(a);
                fp(b);
                fp(p);
            }
            Eq::Lin { a, b, .. } => {
                *a = f(*a);
                *b = f(*b);
            }
            Eq::Dist { p, q, .. } => {
                fp(p);
                fp(q);
            }
            Eq::PtLine { p, a, b, .. }
            | Eq::SymMid { p, a, b, .. }
            | Eq::SymPerp { p, a, b, .. }
            | Eq::Mid { p, a, b, .. } => {
                fp(p);
                fp(a);
                fp(b);
                if let Eq::SymMid { q, .. } | Eq::SymPerp { q, .. } = self {
                    fp(q);
                }
            }
            Eq::PtCircle { p, c, r } => {
                fp(p);
                fp(c);
                fr(r);
            }
            Eq::Parallel { a1, a2, b1, b2 }
            | Eq::Perp { a1, a2, b1, b2 }
            | Eq::Angle { a1, a2, b1, b2, .. }
            | Eq::EqualLen { a1, a2, b1, b2 } => {
                fp(a1);
                fp(a2);
                fp(b1);
                fp(b2);
            }
            Eq::EqualRad { r1, r2 } => {
                fr(r1);
                fr(r2);
            }
            Eq::RadiusIs { r, .. } => fr(r),
            Eq::TanLineCircle { a, b, c, r, .. } => {
                fp(a);
                fp(b);
                fp(c);
                fr(r);
            }
            Eq::TanCircles { c1, r1, c2, r2, .. } => {
                fp(c1);
                fp(c2);
                fr(r1);
                fr(r2);
            }
        }
    }
}

struct Values<'a>(&'a [f64]);
impl Vars<f64> for Values<'_> {
    fn get(&mut self, p: P) -> f64 {
        self.0[p]
    }
}

/// Values for a Jacobian row: free parameters in `slots` become seeded duals.
struct Seeded<'a> {
    x: &'a [f64],
    slots: &'a [P],
}
impl Vars<Dual> for Seeded<'_> {
    fn get(&mut self, p: P) -> Dual {
        match self.slots.iter().position(|&s| s == p) {
            Some(i) => Dual::var(self.x[p], i),
            None => Dual::c(self.x[p]),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SolveOptions {
    pub max_iters: usize,
    pub tol: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport {
    pub converged: bool,
    pub iterations: usize,
    pub max_residual: f64,
    pub components: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    /// Remaining degrees of freedom of the whole sketch.
    pub dof: usize,
    /// Number of equations that are linearly dependent on the others.
    pub redundant: usize,
    /// Per parameter: is its value fully determined by the constraints?
    pub determined: Vec<bool>,
    pub error: Option<String>,
}

/// The numeric system: parameter values, which ones are locked, and the equations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct System {
    pub x: Vec<f64>,
    pub fixed: Vec<bool>,
    pub eqs: Vec<Eq>,
}

impl System {
    pub fn add_param(&mut self, v: f64) -> P {
        self.x.push(v);
        self.fixed.push(false);
        self.x.len() - 1
    }

    pub fn residuals(&self) -> Vec<f64> {
        self.eqs
            .iter()
            .map(|e| e.residual::<f64, _>(&mut Values(&self.x)))
            .collect()
    }

    pub fn max_residual(&self) -> f64 {
        self.eqs.iter().fold(0.0, |m, e| {
            let r = e.residual::<f64, _>(&mut Values(&self.x));
            if r.is_finite() {
                m.max(r.abs())
            } else {
                f64::INFINITY
            }
        })
    }

    fn invalid_input(&self) -> Option<String> {
        if self.x.len() != self.fixed.len() || self.x.iter().any(|v| !v.is_finite()) {
            return Some("Invalid parameter values or fixed-parameter array".into());
        }
        let mut touched = Vec::new();
        for (i, e) in self.eqs.iter().enumerate() {
            e.params(&mut touched);
            if touched.len() > MAXV || touched.iter().any(|&p| p >= self.x.len()) {
                return Some(format!("Equation {i} has invalid parameter references"));
            }
            if !e.residual::<f64, _>(&mut Values(&self.x)).is_finite() {
                return Some(format!(
                    "Equation {i} has a non-finite residual (degenerate geometry)"
                ));
            }
        }
        None
    }

    /// Split equations into independent groups linked by shared free
    /// parameters (union-find). Equations with no free parameter form no
    /// component; they can only be checked.
    pub fn components(&self) -> Vec<(Vec<usize>, Vec<P>)> {
        let n = self.x.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut buf = Vec::new();
        let mut eq_first: Vec<Option<P>> = vec![None; self.eqs.len()];
        for (ei, e) in self.eqs.iter().enumerate() {
            e.params(&mut buf);
            let mut first = None;
            for &p in buf.iter().filter(|&&p| !self.fixed[p]) {
                match first {
                    None => first = Some(p),
                    Some(f) => {
                        let (a, b) = (find(&mut parent, f), find(&mut parent, p));
                        parent[a] = b;
                    }
                }
            }
            eq_first[ei] = first;
        }
        let mut groups: std::collections::BTreeMap<usize, (Vec<usize>, Vec<P>)> =
            Default::default();
        for (ei, f) in eq_first.iter().enumerate() {
            if let Some(f) = f {
                let root = find(&mut parent, *f);
                groups.entry(root).or_default().0.push(ei);
            }
        }
        // Visit parameters once instead of repeated linear membership searches.
        for p in 0..n {
            if !self.fixed[p] {
                let root = find(&mut parent, p);
                if let Some((_, vars)) = groups.get_mut(&root) {
                    vars.push(p);
                }
            }
        }
        groups.into_values().collect()
    }

    /// Residual vector and dense Jacobian for a subset of equations w.r.t. `vars`.
    fn jacobian(&self, eqs: &[usize], vars: &[P]) -> (DVector<f64>, DMatrix<f64>) {
        let mut r = DVector::zeros(eqs.len());
        let mut j = DMatrix::zeros(eqs.len(), vars.len());
        let mut buf = Vec::new();
        for (row, &ei) in eqs.iter().enumerate() {
            let e = &self.eqs[ei];
            e.params(&mut buf);
            buf.retain(|&p| !self.fixed[p]);
            debug_assert!(buf.len() <= MAXV, "equation touches too many parameters");
            let d = e.residual::<Dual, _>(&mut Seeded {
                x: &self.x,
                slots: &buf,
            });
            r[row] = d.v;
            for (slot, &p) in buf.iter().enumerate() {
                if let Ok(col) = vars.binary_search(&p) {
                    j[(row, col)] = d.g[slot];
                }
            }
        }
        (r, j)
    }

    /// Solve all components. Damped least-norm Newton (Levenberg–Marquardt in
    /// the "minimum movement" form): dx = -Jᵀ (J Jᵀ + μI)⁻¹ r.
    pub fn solve(&mut self, opts: &SolveOptions) -> SolveReport {
        if opts.tol != 0. && (!opts.tol.is_finite() || opts.tol < 0.) {
            return SolveReport {
                converged: false,
                iterations: 0,
                max_residual: f64::INFINITY,
                components: 0,
                error: Some("Solver tolerance must be finite and positive".into()),
            };
        }
        if let Some(error) = self.invalid_input() {
            return SolveReport {
                converged: false,
                iterations: 0,
                max_residual: f64::INFINITY,
                components: 0,
                error: Some(error),
            };
        }
        let max_iters = if opts.max_iters == 0 {
            60
        } else {
            opts.max_iters
        };
        let tol = if opts.tol == 0.0 { 1e-9 } else { opts.tol };
        let comps = self.components();
        let mut total_iters = 0;
        let mut error = None;
        for (eqs, vars) in &comps {
            match self.solve_component(eqs, vars, max_iters, tol) {
                Ok(iters) => total_iters += iters,
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        let max_residual = self.max_residual();
        SolveReport {
            converged: error.is_none() && max_residual.is_finite() && max_residual <= tol * 10.0,
            iterations: total_iters,
            max_residual,
            components: comps.len(),
            error,
        }
    }

    fn solve_component(
        &mut self,
        eqs: &[usize],
        vars: &[P],
        max_iters: usize,
        tol: f64,
    ) -> Result<usize, String> {
        let m = eqs.len();
        let (mut r, mut j) = self.jacobian(eqs, vars);
        if !r.iter().chain(j.iter()).all(|v| v.is_finite()) {
            return Err(format!(
                "Non-finite Jacobian in component containing equation {}",
                eqs[0]
            ));
        }
        let mut err = r.norm_squared();
        let mut mu = 1e-10;
        let mut it = 0;
        while it < max_iters {
            if r.amax() <= tol {
                break;
            }
            it += 1;
            let jjt = &j * j.transpose();
            let scale = (0..m).map(|i| jjt[(i, i)]).fold(0.0, f64::max).max(1e-12);
            let mut accepted = false;
            for _ in 0..12 {
                let mut a = jjt.clone();
                for i in 0..m {
                    a[(i, i)] += mu * scale;
                }
                let y = match a.cholesky() {
                    Some(ch) => ch.solve(&r),
                    None => {
                        mu *= 10.0;
                        continue;
                    }
                };
                let dx = -(j.transpose() * y);
                let saved: Vec<f64> = vars.iter().map(|&p| self.x[p]).collect();
                // Cap the step so one iteration cannot fling geometry across the plane.
                let step_cap = 1e4;
                let k = (step_cap / dx.amax().max(1e-300)).min(1.0);
                for (c, &p) in vars.iter().enumerate() {
                    self.x[p] += k * dx[c];
                }
                let (r2, j2) = self.jacobian(eqs, vars);
                let e2 = r2.norm_squared();
                if e2.is_finite() && e2 < err && j2.iter().all(|v| v.is_finite()) {
                    r = r2;
                    j = j2;
                    err = e2;
                    mu = (mu * 0.2).max(1e-14);
                    accepted = true;
                    break;
                }
                for (c, &p) in vars.iter().enumerate() {
                    self.x[p] = saved[c];
                }
                mu *= 10.0;
            }
            if !accepted {
                break;
            }
        }
        Ok(it)
    }

    pub fn solve_drag(&mut self, targets: &[(P, f64)], opts: &SolveOptions) -> SolveReport {
        if targets
            .iter()
            .any(|&(p, v)| p >= self.x.len() || p >= self.fixed.len() || !v.is_finite())
        {
            return SolveReport {
                converged: false,
                iterations: 0,
                max_residual: f64::INFINITY,
                components: 0,
                error: Some("Invalid drag target".into()),
            };
        }
        let saved = self.x.clone();
        let movable: Vec<(P, f64)> = targets
            .iter()
            .copied()
            .filter(|(p, _)| !self.fixed[*p])
            .collect();
        for &(p, v) in &movable {
            self.x[p] = v;
            self.fixed[p] = true;
        }
        let rep = self.solve(opts);
        for &(p, _) in &movable {
            self.fixed[p] = false;
        }
        if rep.converged {
            return rep;
        }
        self.x = saved.clone();
        for &(p, v) in &movable {
            self.x[p] = v;
        }
        let rep = self.solve(opts);
        if !rep.converged {
            // Leave the previous valid state rather than a broken one.
            self.x = saved;
        }
        rep
    }

    /// Component-wise rank analysis: exact graph rank for linear differences,
    /// otherwise pivoted QR of the normalized Jacobian's row space. A parameter
    /// is determined when its coordinate axis lies entirely in that row space.
    pub fn analyze(&self) -> Analysis {
        if let Some(error) = self.invalid_input() {
            return Analysis {
                dof: self.x.len(),
                redundant: 0,
                determined: vec![false; self.x.len()],
                error: Some(error),
            };
        }
        let mut dof = self.fixed.iter().filter(|&&fixed| !fixed).count();
        let mut determined: Vec<bool> = self.fixed.clone();
        let mut total_rank = 0;
        for (eqs, vars) in self.components() {
            // Difference equations form a graph: a connected component has one
            // translation DOF unless any equation anchors it to a fixed value.
            let rank = if eqs.iter().all(|&i| matches!(self.eqs[i], Eq::Lin { .. })) {
                let anchored = eqs.iter().any(|&i| match self.eqs[i] {
                    Eq::Lin { a, b, .. } => self.fixed[a] || self.fixed[b],
                    _ => false,
                });
                if anchored {
                    for &p in &vars {
                        determined[p] = true;
                    }
                }
                vars.len() - usize::from(!anchored)
            } else {
                let (_, mut j) = self.jacobian(&eqs, &vars);
                if !j.iter().all(|v| v.is_finite()) {
                    return Analysis {
                        dof,
                        redundant: 0,
                        determined,
                        error: Some(format!(
                            "Non-finite Jacobian in component containing equation {}",
                            eqs[0]
                        )),
                    };
                }
                // Normalize equations before rank analysis: residuals mix
                // lengths, angles, and normalized dot/cross products.
                for row in 0..j.nrows() {
                    let norm = j.row(row).norm();
                    if norm > 0. {
                        j.row_mut(row).scale_mut(1. / norm);
                    }
                }
                // Pivoted QR on Jᵀ finds its row space directly, avoiding JᵀJ
                // (which squares the condition number) and a full dense SVD.
                let qr = j.transpose().col_piv_qr();
                let r = qr.r();
                let diagonal = r.nrows().min(r.ncols());
                let scale = (0..diagonal).map(|i| r[(i, i)].abs()).fold(1e-30, f64::max);
                let rank = (0..diagonal)
                    .take_while(|&i| r[(i, i)].abs() > scale * 1e-8)
                    .count();
                let q = qr.q();
                for (col, &p) in vars.iter().enumerate() {
                    let projected: f64 = (0..rank).map(|k| q[(col, k)].powi(2)).sum();
                    determined[p] = 1. - projected < 1e-8;
                }
                rank
            };
            total_rank += rank;
            dof -= rank;
        }
        Analysis {
            dof,
            redundant: self.eqs.len().saturating_sub(total_rank),
            determined,
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_and_invalid_systems_never_converge_or_enter_rank_analysis() {
        for fixed in [true, false] {
            let mut s = System {
                x: vec![0.; 6],
                fixed: vec![fixed; 6],
                eqs: vec![Eq::PtLine {
                    p: Pt(4, 5),
                    a: Pt(0, 1),
                    b: Pt(2, 3),
                    d: 5.,
                }],
            };
            let report = s.solve(&SolveOptions::default());
            assert!(!report.converged);
            assert!(report.error.as_ref().unwrap().contains("Equation 0"));
            assert!(s.analyze().error.is_some());
            assert_eq!(s.x, vec![0.; 6]);
        }
        let mut malformed = System {
            x: vec![0.],
            fixed: vec![false],
            eqs: vec![Eq::Lin { a: 0, b: 99, k: 0. }],
        };
        assert!(!malformed.solve(&SolveOptions::default()).converged);
        assert!(malformed.analyze().error.is_some());
    }

    #[test]
    fn analysis_handles_unreferenced_variables_and_large_linear_components() {
        let mut s = System::default();
        for i in 0..512 {
            let p = s.add_param(i as f64);
            if i > 0 {
                s.eqs.push(Eq::Lin {
                    a: p,
                    b: p - 1,
                    k: 1.,
                });
            }
        }
        s.add_param(99.); // an independent unconstrained parameter
        assert_eq!(s.analyze().dof, 2);
        s.fixed[0] = true;
        let analysis = s.analyze();
        assert_eq!(analysis.dof, 1);
        assert!(analysis.determined[..512].iter().all(|&d| d));
        assert!(!analysis.determined[512]);
    }

    #[test]
    fn direct_rank_analysis_preserves_triangle_dof_across_scales_and_components() {
        for scale in [1e-6, 1., 1e6] {
            let (mut s, p) = sys_with(&[(0., 0.), (3. * scale, 0.), (0., 4. * scale)]);
            for (a, b, d) in [(0, 1, 3.), (0, 2, 4.), (1, 2, 5.)] {
                s.eqs.push(Eq::Dist {
                    p: p[a],
                    q: p[b],
                    d: d * scale,
                });
            }
            let a = s.analyze();
            assert_eq!(a.dof, 3);
            assert_eq!(a.redundant, 0);
            assert!(a.error.is_none());
            s.fixed[0] = true;
            s.fixed[1] = true;
            assert_eq!(s.analyze().dof, 1);
            s.eqs.push(s.eqs[0]);
            assert_eq!(s.analyze().redundant, 1);
        }
    }

    fn sys_with(points: &[(f64, f64)]) -> (System, Vec<Pt>) {
        let mut s = System::default();
        let pts = points
            .iter()
            .map(|&(x, y)| Pt(s.add_param(x), s.add_param(y)))
            .collect();
        (s, pts)
    }

    #[test]
    fn rectangle_with_dimensions_is_fully_constrained() {
        let (mut s, p) = sys_with(&[(0.1, -0.2), (9.0, 0.5), (11.0, 4.0), (-0.3, 5.5)]);
        s.fixed[p[0].0] = true;
        s.fixed[p[0].1] = true;
        // horizontal bottom/top, vertical sides
        s.eqs.push(Eq::Lin {
            a: p[0].1,
            b: p[1].1,
            k: 0.0,
        });
        s.eqs.push(Eq::Lin {
            a: p[3].1,
            b: p[2].1,
            k: 0.0,
        });
        s.eqs.push(Eq::Lin {
            a: p[0].0,
            b: p[3].0,
            k: 0.0,
        });
        s.eqs.push(Eq::Lin {
            a: p[1].0,
            b: p[2].0,
            k: 0.0,
        });
        s.eqs.push(Eq::Dist {
            p: p[0],
            q: p[1],
            d: 10.0,
        });
        s.eqs.push(Eq::Dist {
            p: p[1],
            q: p[2],
            d: 5.0,
        });
        let rep = s.solve(&SolveOptions::default());
        assert!(rep.converged, "{rep:?}");
        let a = s.analyze();
        assert_eq!(a.dof, 0);
        assert_eq!(a.redundant, 0);
        assert!(a.determined.iter().all(|&d| d));
        assert!((s.x[p[2].0] - 10.1).abs() < 1e-7);
        assert!((s.x[p[2].1] - 4.8).abs() < 1e-7);
    }

    #[test]
    fn minimum_movement_keeps_untouched_geometry() {
        // Two independent segments; constraining one must not move the other.
        let (mut s, p) = sys_with(&[(0.0, 0.0), (3.0, 0.2), (10.0, 10.0), (12.0, 11.0)]);
        s.eqs.push(Eq::Lin {
            a: p[0].1,
            b: p[1].1,
            k: 0.0,
        });
        s.eqs.push(Eq::Lin {
            a: p[2].1,
            b: p[3].1,
            k: 0.0,
        });
        s.eqs.push(Eq::Dist {
            p: p[2],
            q: p[3],
            d: 4.0,
        });
        let before = s.x.clone();
        let rep = s.solve(&SolveOptions::default());
        assert!(rep.converged);
        assert_eq!(rep.components, 2);
        // First segment moved only by the minimal amount to become horizontal.
        assert!((s.x[p[0].1] - 0.1).abs() < 1e-6 && (s.x[p[1].1] - 0.1).abs() < 1e-6);
        assert!((s.x[p[0].0] - before[p[0].0]).abs() < 1e-9);
        let a = s.analyze();
        assert_eq!(a.dof, 8 - 3);
        assert!(!a.determined[p[0].0]);
    }

    #[test]
    fn redundancy_is_detected() {
        let (mut s, p) = sys_with(&[(0.0, 0.0), (5.0, 1.0)]);
        s.eqs.push(Eq::Lin {
            a: p[0].1,
            b: p[1].1,
            k: 0.0,
        });
        s.eqs.push(Eq::Lin {
            a: p[1].1,
            b: p[0].1,
            k: 0.0,
        });
        assert!(s.solve(&SolveOptions::default()).converged);
        assert_eq!(s.analyze().redundant, 1);
    }

    #[test]
    fn tangent_line_circle() {
        let mut s = System::default();
        let a = Pt(s.add_param(-5.0), s.add_param(0.0));
        let b = Pt(s.add_param(5.0), s.add_param(0.0));
        let c = Pt(s.add_param(0.0), s.add_param(2.0));
        let r = s.add_param(1.0);
        for p in [a.0, a.1, b.0, b.1] {
            s.fixed[p] = true;
        }
        s.eqs.push(Eq::TanLineCircle {
            a,
            b,
            c,
            r: Rad::Param(r),
            side: 1.0,
        });
        s.eqs.push(Eq::RadiusIs {
            r: Rad::Param(r),
            d: 3.0,
        });
        assert!(s.solve(&SolveOptions::default()).converged);
        assert!((s.x[c.1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn drag_follows_target_when_feasible() {
        let (mut s, p) = sys_with(&[(0.0, 0.0), (5.0, 0.0)]);
        s.fixed[p[0].0] = true;
        s.fixed[p[0].1] = true;
        s.eqs.push(Eq::Dist {
            p: p[0],
            q: p[1],
            d: 5.0,
        });
        // Drag the end towards (0, 9): it must stay on the circle of radius 5.
        let rep = s.solve_drag(&[(p[1].0, 0.0), (p[1].1, 9.0)], &SolveOptions::default());
        assert!(rep.converged);
        assert!((s.x[p[1].1] - 5.0).abs() < 1e-6, "{:?}", s.x);
    }
}
