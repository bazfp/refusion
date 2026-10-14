use std::ops::{Add, Div, Mul, Neg, Sub};

/// Maximum number of free variables a single equation may touch.
/// (Symmetry of two points about a line, or tangency between two arcs, uses 8.)
pub const MAXV: usize = 8;

pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn c(v: f64) -> Self;
    fn val(&self) -> f64;
    fn sqrt(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn scale(self, k: f64) -> Self {
        self * Self::c(k)
    }
}

impl Real for f64 {
    fn c(v: f64) -> Self {
        v
    }
    fn val(&self) -> f64 {
        *self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
}

/// A value together with its gradient w.r.t. up to [`MAXV`] local variables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub g: [f64; MAXV],
}

impl Dual {
    pub fn var(v: f64, slot: usize) -> Self {
        let mut g = [0.0; MAXV];
        g[slot] = 1.0;
        Dual { v, g }
    }
    #[inline]
    fn map(self, v: f64, dv: f64) -> Self {
        let mut g = self.g;
        for x in &mut g {
            *x *= dv;
        }
        Dual { v, g }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(mut self, o: Dual) -> Dual {
        self.v += o.v;
        for i in 0..MAXV {
            self.g[i] += o.g[i];
        }
        self
    }
}
impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(mut self, o: Dual) -> Dual {
        self.v -= o.v;
        for i in 0..MAXV {
            self.g[i] -= o.g[i];
        }
        self
    }
}
impl Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        let mut g = [0.0; MAXV];
        for i in 0..MAXV {
            g[i] = self.g[i] * o.v + o.g[i] * self.v;
        }
        Dual { v: self.v * o.v, g }
    }
}
impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let inv = 1.0 / o.v;
        let mut g = [0.0; MAXV];
        for i in 0..MAXV {
            g[i] = (self.g[i] * o.v - o.g[i] * self.v) * inv * inv;
        }
        Dual { v: self.v * inv, g }
    }
}
impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        self.map(-self.v, -1.0)
    }
}

impl Real for Dual {
    fn c(v: f64) -> Self {
        Dual { v, g: [0.0; MAXV] }
    }
    fn val(&self) -> f64 {
        self.v
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        // Guard the derivative at 0 (coincident points measured by distance).
        let d = if s > 1e-300 { 0.5 / s } else { 0.0 };
        self.map(s, d)
    }
    fn atan2(self, x: Self) -> Self {
        let y = self;
        let r2 = x.v * x.v + y.v * y.v;
        let mut g = [0.0; MAXV];
        if r2 > 1e-300 {
            for i in 0..MAXV {
                g[i] = (x.v * y.g[i] - y.v * x.g[i]) / r2;
            }
        }
        Dual {
            v: y.v.atan2(x.v),
            g,
        }
    }
    fn sin(self) -> Self {
        self.map(self.v.sin(), self.v.cos())
    }
    fn cos(self) -> Self {
        self.map(self.v.cos(), -self.v.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_of_distance() {
        let x = Dual::var(3.0, 0);
        let y = Dual::var(4.0, 1);
        let d = (x * x + y * y).sqrt();
        assert!((d.v - 5.0).abs() < 1e-12);
        assert!((d.g[0] - 0.6).abs() < 1e-12);
        assert!((d.g[1] - 0.8).abs() < 1e-12);
    }
    #[test]
    fn gradient_of_atan2() {
        let y = Dual::var(1.0, 0);
        let x = Dual::var(1.0, 1);
        let a = y.atan2(x);
        assert!((a.v - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((a.g[0] - 0.5).abs() < 1e-12);
        assert!((a.g[1] + 0.5).abs() < 1e-12);
    }
}
