use crate::{check, last_error, KernelError, Result, Shape};
use std::ffi::c_int;

#[repr(C)]
struct RawBuilder {
    _p: [u8; 0],
}

extern "C" {
    fn rf_bld_new() -> *mut RawBuilder;
    fn rf_bld_free(b: *mut RawBuilder);
    fn rf_bld_vertex(b: *mut RawBuilder, p: *const f64) -> c_int;
    fn rf_bld_line(b: *mut RawBuilder, p: *const f64, d: *const f64) -> c_int;
    fn rf_bld_ellipse(
        b: *mut RawBuilder,
        c: *const f64,
        n: *const f64,
        major: *const f64,
        ratio: f64,
    ) -> c_int;
    fn rf_bld_bspline_curve(
        b: *mut RawBuilder,
        degree: c_int,
        nk: c_int,
        knots: *const f64,
        mults: *const c_int,
        np: c_int,
        poles: *const f64,
        weights: *const f64,
        periodic: c_int,
    ) -> c_int;
    fn rf_bld_plane(b: *mut RawBuilder, p: *const f64, n: *const f64, u: *const f64) -> c_int;
    fn rf_bld_cone(
        b: *mut RawBuilder,
        c: *const f64,
        n: *const f64,
        major: *const f64,
        sina: f64,
        cosa: f64,
    ) -> c_int;
    fn rf_bld_sphere(
        b: *mut RawBuilder,
        c: *const f64,
        r: f64,
        u: *const f64,
        pole: *const f64,
    ) -> c_int;
    fn rf_bld_torus(
        b: *mut RawBuilder,
        c: *const f64,
        n: *const f64,
        big: f64,
        small: f64,
        u: *const f64,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn rf_bld_bspline_surface(
        b: *mut RawBuilder,
        du: c_int,
        dv: c_int,
        nku: c_int,
        ku: *const f64,
        mu: *const c_int,
        nkv: c_int,
        kv: *const f64,
        mv: *const c_int,
        nu: c_int,
        nv: c_int,
        poles: *const f64,
        weights: *const f64,
        uper: c_int,
        vper: c_int,
    ) -> c_int;
    fn rf_bld_extrusion(b: *mut RawBuilder, curve: c_int, d: *const f64) -> c_int;
    fn rf_bld_edge(
        b: *mut RawBuilder,
        curve: c_int,
        vs: c_int,
        ve: c_int,
        reversed: c_int,
    ) -> c_int;
    fn rf_bld_face(
        b: *mut RawBuilder,
        surf: c_int,
        reversed: c_int,
        nloops: c_int,
        loop_lens: *const c_int,
        coedges: *const c_int,
    ) -> c_int;
    fn rf_bld_body(
        b: *mut RawBuilder,
        n: c_int,
        face_ids: *const c_int,
        trsf: *const f64,
    ) -> *mut crate::RawShape;
    fn rf_is_valid(s: *const crate::RawShape) -> c_int;
    fn rf_is_solid(s: *const crate::RawShape) -> c_int;
}

/// A B-spline in OCCT form (end multiplicities already degree + 1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BSplineCurve {
    pub degree: i32,
    pub knots: Vec<f64>,
    pub mults: Vec<i32>,
    pub poles: Vec<[f64; 3]>,
    pub weights: Option<Vec<f64>>,
    pub periodic: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BSplineSurface {
    pub du: i32,
    pub dv: i32,
    pub ku: Vec<f64>,
    pub mu: Vec<i32>,
    pub kv: Vec<f64>,
    pub mv: Vec<i32>,
    /// poles[iu][iv], stored row-major with `nv` columns
    pub nu: usize,
    pub nv: usize,
    pub poles: Vec<[f64; 3]>,
    pub weights: Option<Vec<f64>>,
    pub uper: bool,
    pub vper: bool,
}

pub struct Builder(*mut RawBuilder);

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe { rf_bld_free(self.0) }
    }
}

fn id(v: c_int) -> Result<i32> {
    if v < 0 {
        Err(last_error())
    } else {
        Ok(v)
    }
}

fn native_count(n: usize) -> Result<c_int> {
    c_int::try_from(n).map_err(|_| KernelError("Spline array exceeds the native size limit".into()))
}

fn validate_knots(
    degree: i32,
    knots: &[f64],
    mults: &[i32],
    poles: usize,
    periodic: bool,
) -> Result<()> {
    if !(1..=25).contains(&degree) || knots.len() < 2 || knots.len() != mults.len() || poles < 2 {
        return Err(KernelError(
            "Invalid spline degree, knot/multiplicity lengths, or pole count".into(),
        ));
    }
    native_count(knots.len())?;
    native_count(poles)?;
    if !knots.iter().all(|k| k.is_finite()) || knots.windows(2).any(|k| k[0] >= k[1]) {
        return Err(KernelError(
            "Spline knots must be finite and strictly increasing".into(),
        ));
    }
    let mut sum = 0usize;
    for (i, &m) in mults.iter().enumerate() {
        let endpoint = i == 0 || i + 1 == mults.len();
        let max = degree + i32::from(endpoint && !periodic);
        if m < 1 || m > max {
            return Err(KernelError("Invalid spline knot multiplicity".into()));
        }
        sum = sum
            .checked_add(m as usize)
            .ok_or_else(|| KernelError("Spline multiplicity overflow".into()))?;
    }
    let expected = if periodic {
        if mults.first() != mults.last() {
            return Err(KernelError(
                "Periodic spline endpoint multiplicities must agree".into(),
            ));
        }
        sum.checked_sub(mults[0] as usize)
    } else {
        sum.checked_sub(degree as usize + 1)
    };
    if expected != Some(poles) {
        return Err(KernelError(
            "Spline knot multiplicities do not match its pole count".into(),
        ));
    }
    Ok(())
}

fn validate_poles(poles: &[[f64; 3]], weights: Option<&[f64]>) -> Result<()> {
    native_count(
        poles
            .len()
            .checked_mul(3)
            .ok_or_else(|| KernelError("Spline pole count overflow".into()))?,
    )?;
    if !poles.iter().flatten().all(|v| v.is_finite()) {
        return Err(KernelError("Spline poles must be finite".into()));
    }
    if weights
        .is_some_and(|w| w.len() != poles.len() || w.iter().any(|v| !v.is_finite() || *v <= 0.0))
    {
        return Err(KernelError(
            "Spline weights must be positive, finite, and match the poles".into(),
        ));
    }
    Ok(())
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        let raw = unsafe { rf_bld_new() };
        assert!(
            !raw.is_null(),
            "Could not allocate the native geometry builder"
        );
        Builder(raw)
    }
    pub fn vertex(&mut self, p: [f64; 3]) -> i32 {
        unsafe { rf_bld_vertex(self.0, p.as_ptr()) }
    }
    pub fn line(&mut self, p: [f64; 3], d: [f64; 3]) -> Result<i32> {
        id(unsafe { rf_bld_line(self.0, p.as_ptr(), d.as_ptr()) })
    }
    pub fn ellipse(
        &mut self,
        c: [f64; 3],
        n: [f64; 3],
        major: [f64; 3],
        ratio: f64,
    ) -> Result<i32> {
        id(unsafe { rf_bld_ellipse(self.0, c.as_ptr(), n.as_ptr(), major.as_ptr(), ratio) })
    }
    pub fn bspline_curve(&mut self, c: &BSplineCurve) -> Result<i32> {
        validate_knots(c.degree, &c.knots, &c.mults, c.poles.len(), c.periodic)?;
        validate_poles(&c.poles, c.weights.as_deref())?;
        let poles: Vec<f64> = c.poles.iter().flatten().copied().collect();
        let w = c
            .weights
            .as_ref()
            .map(|w| w.as_ptr())
            .unwrap_or(std::ptr::null());
        id(unsafe {
            rf_bld_bspline_curve(
                self.0,
                c.degree,
                c.knots.len() as c_int,
                c.knots.as_ptr(),
                c.mults.as_ptr(),
                c.poles.len() as c_int,
                poles.as_ptr(),
                w,
                c.periodic as c_int,
            )
        })
    }
    pub fn plane(&mut self, p: [f64; 3], n: [f64; 3], u: [f64; 3]) -> Result<i32> {
        id(unsafe { rf_bld_plane(self.0, p.as_ptr(), n.as_ptr(), u.as_ptr()) })
    }
    pub fn cone(
        &mut self,
        c: [f64; 3],
        n: [f64; 3],
        major: [f64; 3],
        sina: f64,
        cosa: f64,
    ) -> Result<i32> {
        id(unsafe { rf_bld_cone(self.0, c.as_ptr(), n.as_ptr(), major.as_ptr(), sina, cosa) })
    }
    pub fn sphere(&mut self, c: [f64; 3], r: f64, u: [f64; 3], pole: [f64; 3]) -> Result<i32> {
        id(unsafe { rf_bld_sphere(self.0, c.as_ptr(), r, u.as_ptr(), pole.as_ptr()) })
    }
    pub fn torus(
        &mut self,
        c: [f64; 3],
        n: [f64; 3],
        big: f64,
        small: f64,
        u: [f64; 3],
    ) -> Result<i32> {
        id(unsafe { rf_bld_torus(self.0, c.as_ptr(), n.as_ptr(), big, small, u.as_ptr()) })
    }
    pub fn bspline_surface(&mut self, s: &BSplineSurface) -> Result<i32> {
        let count =
            s.nu.checked_mul(s.nv)
                .ok_or_else(|| KernelError("Spline surface dimensions overflow".into()))?;
        // The C++ indexing expression also multiplies dimensions as a native int.
        native_count(
            count
                .checked_mul(3)
                .ok_or_else(|| KernelError("Spline surface dimensions overflow".into()))?,
        )?;
        if count != s.poles.len() {
            return Err(KernelError(
                "Spline surface dimensions do not match its poles".into(),
            ));
        }
        validate_knots(s.du, &s.ku, &s.mu, s.nu, s.uper)?;
        validate_knots(s.dv, &s.kv, &s.mv, s.nv, s.vper)?;
        validate_poles(&s.poles, s.weights.as_deref())?;
        let poles: Vec<f64> = s.poles.iter().flatten().copied().collect();
        let w = s
            .weights
            .as_ref()
            .map(|w| w.as_ptr())
            .unwrap_or(std::ptr::null());
        id(unsafe {
            rf_bld_bspline_surface(
                self.0,
                s.du,
                s.dv,
                s.ku.len() as c_int,
                s.ku.as_ptr(),
                s.mu.as_ptr(),
                s.kv.len() as c_int,
                s.kv.as_ptr(),
                s.mv.as_ptr(),
                s.nu as c_int,
                s.nv as c_int,
                poles.as_ptr(),
                w,
                s.uper as c_int,
                s.vper as c_int,
            )
        })
    }
    pub fn extrusion(&mut self, curve: i32, d: [f64; 3]) -> Result<i32> {
        id(unsafe { rf_bld_extrusion(self.0, curve, d.as_ptr()) })
    }
    /// `curve` = -1 for a straight segment between the vertices.
    pub fn edge(&mut self, curve: i32, vs: i32, ve: i32, reversed: bool) -> Result<i32> {
        id(unsafe { rf_bld_edge(self.0, curve, vs, ve, reversed as c_int) })
    }
    /// `surface` = -1 to fill a surface through the outer loop. Loops are
    /// lists of (edge id, coedge reversed).
    pub fn face(
        &mut self,
        surface: i32,
        reversed: bool,
        loops: &[Vec<(i32, bool)>],
    ) -> Result<i32> {
        let lens: Vec<c_int> = loops.iter().map(|l| l.len() as c_int).collect();
        let flat: Vec<c_int> = loops
            .iter()
            .flatten()
            .flat_map(|&(e, r)| [e, r as c_int])
            .collect();
        id(unsafe {
            rf_bld_face(
                self.0,
                surface,
                reversed as c_int,
                lens.len() as c_int,
                lens.as_ptr(),
                flat.as_ptr(),
            )
        })
    }
    /// Solid (or shell) from faces, then transformed by a 3×4 row-major matrix.
    pub fn body(&mut self, faces: &[i32], trsf: Option<[f64; 12]>) -> Result<Shape> {
        let t = trsf
            .as_ref()
            .map(|t| t.as_ptr())
            .unwrap_or(std::ptr::null());
        check(unsafe { rf_bld_body(self.0, faces.len() as c_int, faces.as_ptr(), t) })
    }
}

impl Shape {
    pub fn is_valid(&self) -> bool {
        unsafe { rf_is_valid(self.raw()) != 0 }
    }
    pub fn is_solid(&self) -> bool {
        unsafe { rf_is_solid(self.raw()) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_spline_buffers_are_rejected_before_ffi() {
        let mut b = Builder::new();
        let valid = BSplineCurve {
            degree: 1,
            knots: vec![0., 1.],
            mults: vec![2, 2],
            poles: vec![[0.; 3], [1.; 3]],
            ..Default::default()
        };
        assert!(b.bspline_curve(&valid).is_ok());
        let mut bad = valid.clone();
        bad.mults.pop();
        assert!(b.bspline_curve(&bad).is_err());
        bad = valid.clone();
        bad.weights = Some(vec![1.]);
        assert!(b.bspline_curve(&bad).is_err());
        bad = valid;
        bad.knots[1] = f64::NAN;
        assert!(b.bspline_curve(&bad).is_err());
        let mut surface = BSplineSurface {
            du: 1,
            dv: 1,
            ku: vec![0., 1.],
            kv: vec![0., 1.],
            mu: vec![2, 2],
            mv: vec![2, 2],
            nu: 2,
            nv: 2,
            poles: vec![[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 0.]],
            ..Default::default()
        };
        assert!(b.bspline_surface(&surface).is_ok());
        surface.poles.pop();
        assert!(b.bspline_surface(&surface).is_err());
        surface.nu = usize::MAX;
        assert!(b.bspline_surface(&surface).is_err());
        assert!(native_count(usize::MAX).is_err());
    }
    /// Build a unit cube from explicit topology, like the importer does.
    #[test]
    fn cube_from_topology() {
        let mut b = Builder::new();
        let c = |x: f64, y: f64, z: f64| [x, y, z];
        let p = [
            c(0., 0., 0.),
            c(1., 0., 0.),
            c(1., 1., 0.),
            c(0., 1., 0.),
            c(0., 0., 1.),
            c(1., 0., 1.),
            c(1., 1., 1.),
            c(0., 1., 1.),
        ];
        let v: Vec<i32> = p.iter().map(|&q| b.vertex(q)).collect();
        let mut e = std::collections::HashMap::new();
        let mut edge = |b: &mut Builder, i: usize, j: usize| -> (i32, bool) {
            let key = (i.min(j), i.max(j));
            let id = *e
                .entry(key)
                .or_insert_with(|| b.edge(-1, v[key.0], v[key.1], false).unwrap());
            (id, i > j)
        };
        // faces as vertex cycles, CCW seen from outside
        let quads = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let normals = [
            c(0., 0., -1.),
            c(0., 0., 1.),
            c(0., -1., 0.),
            c(1., 0., 0.),
            c(0., 1., 0.),
            c(-1., 0., 0.),
        ];
        let mut faces = vec![];
        for (q, n) in quads.iter().zip(normals) {
            let lp: Vec<(i32, bool)> = (0..4).map(|k| edge(&mut b, q[k], q[(k + 1) % 4])).collect();
            let u = crate::sub(p[q[1]], p[q[0]]);
            let s = b.plane(p[q[0]], n, u).unwrap();
            faces.push(b.face(s, false, &[lp]).unwrap());
        }
        let solid = b.body(&faces, None).unwrap();
        assert!(solid.is_solid());
        assert!((solid.volume() - 1.0).abs() < 1e-9, "{}", solid.volume());
        assert!(solid.is_valid());
        let before = solid.to_brep().unwrap();
        let another = b.body(&faces, None).unwrap();
        assert!(another.is_valid());
        assert_eq!(solid.to_brep().unwrap(), before);
    }
}
