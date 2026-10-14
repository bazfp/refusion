pub mod builder;

use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};
static MESH_KEY: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
pub(crate) struct RawShape {
    _p: [u8; 0],
}
#[repr(C)]
struct RawMesh {
    _p: [u8; 0],
}

extern "C" {
    fn rf_last_error() -> *const c_char;
    fn rf_shape_free(s: *mut RawShape);
    fn rf_count(s: *mut RawShape, kind: c_int) -> c_int;
    fn rf_make_face(
        frame: *const f64,
        loop_lens: *const c_int,
        nloops: c_int,
        segs: *const f64,
        tags: *const u64,
    ) -> *mut RawShape;
    fn rf_compound(shapes: *const *const RawShape, n: c_int) -> *mut RawShape;
    fn rf_extrude(
        face: *mut RawShape,
        dir: *const f64,
        d0: f64,
        d1: f64,
        feat: u64,
    ) -> *mut RawShape;
    fn rf_taper(s: *mut RawShape, dir: *const f64, origin: *const f64, angle: f64)
        -> *mut RawShape;
    fn rf_trim_plane(
        s: *mut RawShape,
        origin: *const f64,
        normal: *const f64,
        inside: *const f64,
        feat: u64,
    ) -> *mut RawShape;
    fn rf_revolve(
        face: *mut RawShape,
        origin: *const f64,
        axis: *const f64,
        angle: f64,
        feat: u64,
    ) -> *mut RawShape;
    fn rf_box(x: f64, y: f64, z: f64) -> *mut RawShape;
    fn rf_vertex_count(s: *mut RawShape) -> c_int;
    fn rf_vertex_pos(s: *mut RawShape, id: c_int, out: *mut f64) -> c_int;
    fn rf_measure(
        a: *mut RawShape,
        ka: c_int,
        ia: c_int,
        b: *mut RawShape,
        kb: c_int,
        ib: c_int,
        out: *mut f64,
    ) -> c_int;
    fn rf_make_vertex(p: *const f64) -> *mut RawShape;
    fn rf_make_segment(a: *const f64, b: *const f64) -> *mut RawShape;
    fn rf_face_copy(
        s: *mut RawShape,
        fid: c_int,
        body_edge_ids: *mut c_int,
        max: c_int,
    ) -> *mut RawShape;
    fn rf_set_edge_tags(s: *mut RawShape, tags: *const u64, n: c_int) -> c_int;
    fn rf_boolean(a: *mut RawShape, b: *mut RawShape, op: c_int) -> *mut RawShape;
    fn rf_touches(a: *const RawShape, b: *const RawShape) -> c_int;
    fn rf_solid_count(s: *const RawShape) -> c_int;
    fn rf_solid_get(s: *mut RawShape, i: c_int) -> *mut RawShape;
    fn rf_volume(s: *const RawShape) -> f64;
    fn rf_fillet(
        s: *mut RawShape,
        ids: *const c_int,
        n: c_int,
        r: f64,
        feat: u64,
        etags: *const u64,
    ) -> *mut RawShape;
    fn rf_chamfer(
        s: *mut RawShape,
        ids: *const c_int,
        n: c_int,
        d1: f64,
        d2: f64,
        feat: u64,
        etags: *const u64,
    ) -> *mut RawShape;
    fn rf_face_tags(s: *mut RawShape, out: *mut u64);
    fn rf_edge_faces(s: *mut RawShape, eid: c_int, out: *mut c_int) -> c_int;
    fn rf_edge_info(s: *mut RawShape, id: c_int, out: *mut f64) -> c_int;
    fn rf_edge_sample(s: *mut RawShape, id: c_int, defl: f64, out: *mut f64, max: c_int) -> c_int;
    fn rf_face_info(s: *mut RawShape, id: c_int, out: *mut f64) -> c_int;
    fn rf_mesh(s: *mut RawShape, lin: f64, ang: f64, clean: c_int) -> *mut RawMesh;
    fn rf_bbox(s: *const RawShape, out: *mut f64) -> c_int;
    fn rf_transform(
        s: *mut RawShape,
        kind: c_int,
        origin: *const f64,
        vector: *const f64,
        angle: f64,
        feat: u64,
    ) -> *mut RawShape;
    fn rf_shell(
        s: *mut RawShape,
        ids: *const c_int,
        n: c_int,
        thickness: f64,
        feat: u64,
    ) -> *mut RawShape;
    fn rf_mesh_counts(m: *const RawMesh, out: *mut u32) -> c_int;
    fn rf_mesh_copy(
        m: *const RawMesh,
        pos: *mut f32,
        nrm: *mut f32,
        idx: *mut u32,
        tri_face: *mut u32,
        edge_pts: *mut f32,
        edge_starts: *mut u32,
    );
    fn rf_mesh_free(m: *mut RawMesh);
    fn rf_read_step(path: *const c_char) -> *mut RawShape;
    fn rf_read_iges(path: *const c_char) -> *mut RawShape;
    fn rf_to_brep(s: *const RawShape) -> *mut c_char;
    fn rf_free_str(p: *mut c_char);
    fn rf_from_brep(text: *const c_char) -> *mut RawShape;
    fn rf_with_face_tags(s: *mut RawShape, tags: *const u64, n: c_int) -> *mut RawShape;
    fn rf_write_step(shapes: *const *const RawShape, n: c_int, path: *const c_char) -> c_int;
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
#[error("{0}")]
pub struct KernelError(pub String);

pub type Result<T> = std::result::Result<T, KernelError>;

pub(crate) fn last_error() -> KernelError {
    let s = unsafe { CStr::from_ptr(rf_last_error()) }
        .to_string_lossy()
        .into_owned();
    KernelError(if s.is_empty() {
        "kernel operation failed".into()
    } else {
        s
    })
}

#[derive(Default)]
struct EdgeNames {
    names: Vec<EdgeName>,
    candidates: std::collections::HashMap<EdgeName, Vec<u32>>,
}
struct Handle(*mut RawShape, u64, OnceLock<EdgeNames>);
struct MeshHandle(*mut RawMesh);
impl Drop for MeshHandle {
    fn drop(&mut self) {
        unsafe {
            rf_mesh_free(self.0);
        }
    }
}
// SAFETY: topology/index/naming data is immutable when returned to callers;
// check() eagerly builds the index. Edge tags can only be assigned privately
// before a fresh face_copy handle escapes. Face tag changes create a fresh
// wrapper; meshing deep-copies geometry in
// the shim before mutating triangulations/normals. Modelling operations either
// use non-destructive kernel modes or isolated inputs. Subshape wrappers may
// share topology, so mutation must never rely solely on Arc::get_mut.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { rf_shape_free(self.0) }
    }
}

/// An immutable B-rep shape (solid, face, or compound). Cheap to clone.
#[derive(Clone)]
pub struct Shape(Arc<Handle>);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shape(faces={}, edges={})",
            self.face_count(),
            self.edge_count()
        )
    }
}

pub(crate) fn check(p: *mut RawShape) -> Result<Shape> {
    if p.is_null() {
        Err(last_error())
    } else {
        if unsafe { rf_count(p, 0) } < 0 {
            let error = last_error();
            unsafe {
                rf_shape_free(p);
            }
            return Err(error);
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(Shape(Arc::new(Handle(
            p,
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            OnceLock::new(),
        ))))
    }
}

/// Plane frame: origin and two orthonormal in-plane axes (normal = x × y).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub origin: [f64; 3],
    pub x: [f64; 3],
    pub y: [f64; 3],
}

impl Frame {
    pub fn normal(&self) -> [f64; 3] {
        cross(self.x, self.y)
    }
    pub fn to_world(&self, p: [f64; 2]) -> [f64; 3] {
        [
            self.origin[0] + p[0] * self.x[0] + p[1] * self.y[0],
            self.origin[1] + p[0] * self.x[1] + p[1] * self.y[1],
            self.origin[2] + p[0] * self.x[2] + p[1] * self.y[2],
        ]
    }
    pub fn to_local(&self, p: [f64; 3]) -> [f64; 2] {
        let d = sub(p, self.origin);
        [dot(d, self.x), dot(d, self.y)]
    }
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// A 2D profile segment in plane coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Seg2 {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    Arc {
        c: [f64; 2],
        a: [f64; 2],
        b: [f64; 2],
        ccw: bool,
    },
}

/// Persistent edge name: names of the two adjacent faces, sorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct EdgeName {
    pub a: u64,
    pub b: u64,
}

impl EdgeName {
    pub fn hash(&self) -> u64 {
        mix(self.a, self.b)
    }
}

/// Deterministic 64-bit mixing (splitmix64), used to build names.
pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
pub fn mix(a: u64, b: u64) -> u64 {
    splitmix(a ^ splitmix(b))
}

/// Which part of a shape a measurement refers to (ids are 1-based).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sub {
    Whole,
    Face(u32),
    Edge(u32),
    Vertex(u32),
}

impl Sub {
    fn raw(self) -> (c_int, c_int) {
        match self {
            Sub::Whole => (0, 0),
            Sub::Face(i) => (1, i as c_int),
            Sub::Edge(i) => (2, i as c_int),
            Sub::Vertex(i) => (3, i as c_int),
        }
    }
}

pub fn min_distance(a: &Shape, sa: Sub, b: &Shape, sb: Sub) -> Result<(f64, [f64; 3], [f64; 3])> {
    let mut o = [0.0f64; 7];
    let (ka, ia) = sa.raw();
    let (kb, ib) = sb.raw();
    if unsafe { rf_measure(a.raw(), ka, ia, b.raw(), kb, ib, o.as_mut_ptr()) } == 0 {
        return Err(last_error());
    }
    Ok((o[0], [o[1], o[2], o[3]], [o[4], o[5], o[6]]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    Join,
    Cut,
    Intersect,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeKind {
    Line,
    Circle {
        center: [f64; 3],
        axis: [f64; 3],
        radius: f64,
    },
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeInfo {
    pub kind: EdgeKind,
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub mid: [f64; 3],
    pub tangent: [f64; 3],
    pub length: f64,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceInfo {
    /// Some(frame) for planar faces; the frame normal points out of the material.
    pub plane: Option<Frame>,
    pub centroid: [f64; 3],
    pub area: f64,
}

/// Triangle mesh of a shape with per-triangle face ids and per-edge polylines
/// (ids are 1-based, matching the kernel's topology index).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// Computed with geometry preparation; UI tooltips never integrate B-reps.
    pub volume: Option<f64>,
    pub bounds: Option<([f64; 3], [f64; 3])>,
    pub linear_deflection: f64,
    pub cache_key: u64,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub tri_face: Vec<u32>,
    pub edge_points: Vec<[f32; 3]>,
    /// edge i (1-based id i+1) spans edge_points[edge_starts[i]..edge_starts[i+1]]
    pub edge_starts: Vec<u32>,
}

/// Quantized half-pixel target with conservative size and absolute bounds.
/// Quantization and caller hysteresis avoid remeshing on small zoom changes.
pub fn display_deflection(bounds: Option<([f64; 3], [f64; 3])>, mm_per_pixel: Option<f64>) -> f64 {
    let diagonal = bounds
        .map(|(lo, hi)| {
            lo.into_iter()
                .zip(hi)
                .map(|(a, b)| (b - a).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .filter(|d| d.is_finite() && *d > 0.)
        .unwrap_or(50.);
    let max = (diagonal / 500.).clamp(0.0005, 0.5);
    let target = mm_per_pixel
        .filter(|v| v.is_finite() && *v > 0.)
        .map(|v| v * 0.5)
        .unwrap_or(diagonal / 1600.);
    let quantized = 0.0005 * 2f64.powf((target.max(0.0005) / 0.0005).log2().floor());
    quantized.clamp(0.0005, max)
}

impl Mesh {
    pub fn edge_polyline(&self, id: u32) -> &[[f32; 3]] {
        let i = (id - 1) as usize;
        &self.edge_points[self.edge_starts[i] as usize..self.edge_starts[i + 1] as usize]
    }
    pub fn edge_count(&self) -> usize {
        self.edge_starts.len().saturating_sub(1)
    }
}

impl Shape {
    pub fn translated(&self, delta: [f64; 3], feature: u64) -> Result<Shape> {
        if !delta.iter().all(|v| v.is_finite()) {
            return Err(KernelError("translation must be finite".into()));
        }
        check(unsafe {
            rf_transform(
                self.0 .0,
                0,
                [0.0; 3].as_ptr(),
                delta.as_ptr(),
                0.0,
                feature,
            )
        })
    }
    pub fn rotated(
        &self,
        origin: [f64; 3],
        axis: [f64; 3],
        angle: f64,
        feature: u64,
    ) -> Result<Shape> {
        if !origin.iter().chain(&axis).all(|v| v.is_finite())
            || !angle.is_finite()
            || dot(axis, axis) < 1e-20
        {
            return Err(KernelError("invalid rotation axis or angle".into()));
        }
        check(unsafe { rf_transform(self.0 .0, 1, origin.as_ptr(), axis.as_ptr(), angle, feature) })
    }
    pub fn mirrored(&self, frame: Frame, feature: u64) -> Result<Shape> {
        let n = frame.normal();
        if !frame.origin.iter().chain(&n).all(|v| v.is_finite()) || dot(n, n) < 1e-20 {
            return Err(KernelError("invalid mirror plane".into()));
        }
        check(unsafe {
            rf_transform(
                self.0 .0,
                2,
                frame.origin.as_ptr(),
                n.as_ptr(),
                0.0,
                feature,
            )
        })
    }
    pub fn shell(&self, faces: &[u32], signed_thickness: f64, feature: u64) -> Result<Shape> {
        if !signed_thickness.is_finite() || signed_thickness.abs() < 1e-9 {
            return Err(KernelError(
                "shell thickness must be nonzero and finite".into(),
            ));
        }
        let ids: Vec<c_int> = faces.iter().map(|&f| f as c_int).collect();
        let isolated = self.translated([0.0; 3], 0)?;
        check(unsafe {
            rf_shell(
                isolated.raw(),
                ids.as_ptr(),
                ids.len() as c_int,
                signed_thickness,
                feature,
            )
        })
    }
    /// Independent inward and outward offsets around the original skin.
    pub fn shell_both(
        &self,
        faces: &[u32],
        inside: f64,
        outside: f64,
        feature: u64,
    ) -> Result<Shape> {
        if !inside.is_finite() || !outside.is_finite() || inside <= 0. || outside <= 0. {
            return Err(KernelError(
                "Both shell thicknesses must be positive and finite".into(),
            ));
        }
        let inner = self.shell(faces, -inside, feature)?;
        let outer = self.shell(faces, outside, mix(feature, 1))?;
        let result = inner.boolean(&outer, BoolOp::Join)?;
        if !result.is_valid() || !result.is_solid() || result.solids().len() != 1 {
            return Err(KernelError(
                "Two-sided shell did not produce one valid solid".into(),
            ));
        }
        Ok(result)
    }
    pub(crate) fn raw(&self) -> *mut RawShape {
        self.0 .0
    }

    /// Unique id of this shape instance (for caching tessellations).
    pub fn uid(&self) -> u64 {
        self.0 .1
    }

    pub fn face_count(&self) -> usize {
        unsafe { rf_count(self.raw(), 0) as usize }
    }
    pub fn edge_count(&self) -> usize {
        unsafe { rf_count(self.raw(), 1) as usize }
    }

    /// Planar face bounded by `loops` (first = outer, CCW; others = holes, CW).
    /// `tags` (one per segment, same order) name the side faces later swept from each segment.
    pub fn face_from_loops(frame: &Frame, loops: &[Vec<Seg2>], tags: &[u64]) -> Result<Shape> {
        let f = [frame.origin, frame.x, frame.y].concat();
        let lens: Vec<c_int> = loops.iter().map(|l| l.len() as c_int).collect();
        let mut segs = Vec::new();
        for l in loops {
            for s in l {
                match *s {
                    Seg2::Line { a, b } => {
                        segs.extend_from_slice(&[0.0, a[0], a[1], b[0], b[1], 0.0, 0.0, 0.0])
                    }
                    Seg2::Arc { c, a, b, ccw } => segs.extend_from_slice(&[
                        1.0,
                        a[0],
                        a[1],
                        b[0],
                        b[1],
                        c[0],
                        c[1],
                        if ccw { 1.0 } else { 0.0 },
                    ]),
                }
            }
        }
        let n: usize = loops.iter().map(|l| l.len()).sum();
        let tags: Vec<u64> = (0..n)
            .map(|i| tags.get(i).copied().unwrap_or(i as u64 + 1))
            .collect();
        check(unsafe {
            rf_make_face(
                f.as_ptr(),
                lens.as_ptr(),
                lens.len() as c_int,
                segs.as_ptr(),
                tags.as_ptr(),
            )
        })
    }

    pub fn compound(shapes: &[Shape]) -> Result<Shape> {
        let raws: Vec<*const RawShape> = shapes.iter().map(|s| s.raw() as *const _).collect();
        check(unsafe { rf_compound(raws.as_ptr(), raws.len() as c_int) })
    }

    pub fn boxed(x: f64, y: f64, z: f64) -> Result<Shape> {
        check(unsafe { rf_box(x, y, z) })
    }

    /// A planar face of this body as a standalone face, its edges named after
    /// the body's edge names (so swept side faces get stable names).
    pub fn face_copy(&self, face: u32) -> Result<Shape> {
        let mut ids = vec![0 as c_int; self.edge_count()];
        let f = check(unsafe {
            rf_face_copy(
                self.raw(),
                face as c_int,
                ids.as_mut_ptr(),
                ids.len() as c_int,
            )
        })?;
        let tags: Vec<u64> = ids[..f.edge_count()]
            .iter()
            .map(|&e| {
                if e > 0 {
                    self.edge_name(e as u32).hash()
                } else {
                    0
                }
            })
            .collect();
        // f is newly created and has not been shared or published to callers.
        if unsafe { rf_set_edge_tags(f.raw(), tags.as_ptr(), tags.len() as c_int) } == 0 {
            return Err(last_error());
        }
        Ok(f)
    }

    pub fn vertex_count(&self) -> usize {
        unsafe { rf_vertex_count(self.raw()) as usize }
    }

    pub fn vertex_pos(&self, id: u32) -> Option<[f64; 3]> {
        let mut o = [0.0f64; 3];
        (unsafe { rf_vertex_pos(self.raw(), id as c_int, o.as_mut_ptr()) } != 0).then_some(o)
    }

    pub fn vertex(p: [f64; 3]) -> Result<Shape> {
        check(unsafe { rf_make_vertex(p.as_ptr()) })
    }

    pub fn segment(a: [f64; 3], b: [f64; 3]) -> Result<Shape> {
        check(unsafe { rf_make_segment(a.as_ptr(), b.as_ptr()) })
    }

    /// Sweep along `dir` from offset d0 to d1 (one side: 0..d, symmetric: -d/2..d/2).
    pub fn extrude(&self, dir: [f64; 3], d0: f64, d1: f64, feature: u64) -> Result<Shape> {
        if (d1 - d0).abs() < 1e-9 {
            return Err(KernelError("extrude distance is zero".into()));
        }
        check(unsafe { rf_extrude(self.raw(), dir.as_ptr(), d0, d1, feature) })
    }

    pub fn extrude_tapered(
        &self,
        dir: [f64; 3],
        d0: f64,
        d1: f64,
        angle: f64,
        feature: u64,
    ) -> Result<Shape> {
        if !angle.is_finite() || angle.abs() >= 85f64.to_radians() {
            return Err(KernelError(
                "Taper angle must be between -85 and 85 degrees".into(),
            ));
        }
        if angle.abs() < 1e-10 {
            return self.extrude(dir, d0, d1, feature);
        }
        if d0 > 0. || d1 < 0. {
            return Err(KernelError(
                "Tapered extents must start at the profile plane".into(),
            ));
        }
        let origin = self
            .face_info(1)
            .and_then(|i| i.plane)
            .ok_or_else(|| KernelError("Taper requires a planar profile".into()))?
            .origin;
        let part = |distance: f64| -> Result<Shape> {
            let axis = if distance < 0. { dir.map(|v| -v) } else { dir };
            let prism = self.extrude(axis, 0., distance.abs(), feature)?;
            check(unsafe { rf_taper(prism.raw(), axis.as_ptr(), origin.as_ptr(), angle) })
        };
        match (d0 < -1e-9, d1 > 1e-9) {
            (true, true) => part(d0)?.boolean(&part(d1)?, BoolOp::Join),
            (true, false) => part(d0),
            (false, true) => part(d1),
            _ => Err(KernelError(
                "Extrude extent must have positive length".into(),
            )),
        }
    }
    pub fn trim_at_plane(&self, plane: Frame, inside: [f64; 3], feature: u64) -> Result<Shape> {
        check(unsafe {
            rf_trim_plane(
                self.raw(),
                plane.origin.as_ptr(),
                plane.normal().as_ptr(),
                inside.as_ptr(),
                feature,
            )
        })
    }

    pub fn revolve(
        &self,
        origin: [f64; 3],
        axis: [f64; 3],
        angle: f64,
        feature: u64,
    ) -> Result<Shape> {
        if angle.abs() < 1e-9 {
            return Err(KernelError("revolve angle is zero".into()));
        }
        check(unsafe { rf_revolve(self.raw(), origin.as_ptr(), axis.as_ptr(), angle, feature) })
    }

    pub fn boolean(&self, tool: &Shape, op: BoolOp) -> Result<Shape> {
        let op = match op {
            BoolOp::Join => 0,
            BoolOp::Cut => 1,
            BoolOp::Intersect => 2,
        };
        check(unsafe { rf_boolean(self.raw(), tool.raw(), op) })
    }

    pub fn touches(&self, other: &Shape) -> bool {
        unsafe { rf_touches(self.raw(), other.raw()) != 0 }
    }

    /// Split a compound result into its solids.
    pub fn solids(&self) -> Vec<Shape> {
        let n = unsafe { rf_solid_count(self.raw()) };
        (0..n)
            .filter_map(|i| check(unsafe { rf_solid_get(self.raw(), i) }).ok())
            .collect()
    }

    pub fn volume(&self) -> f64 {
        unsafe { rf_volume(self.raw()) }
    }

    pub fn fillet(&self, edges: &[u32], radius: f64, feature: u64) -> Result<Shape> {
        let ids: Vec<c_int> = edges.iter().map(|&e| e as c_int).collect();
        let tags: Vec<u64> = edges.iter().map(|&e| self.edge_name(e).hash()).collect();
        let isolated = self.translated([0.0; 3], 0)?;
        check(unsafe {
            rf_fillet(
                isolated.raw(),
                ids.as_ptr(),
                ids.len() as c_int,
                radius,
                feature,
                tags.as_ptr(),
            )
        })
    }

    /// Equal-distance chamfer when `d2` is None.
    pub fn chamfer(&self, edges: &[u32], d1: f64, d2: Option<f64>, feature: u64) -> Result<Shape> {
        let ids: Vec<c_int> = edges.iter().map(|&e| e as c_int).collect();
        let tags: Vec<u64> = edges.iter().map(|&e| self.edge_name(e).hash()).collect();
        let isolated = self.translated([0.0; 3], 0)?;
        check(unsafe {
            rf_chamfer(
                isolated.raw(),
                ids.as_ptr(),
                ids.len() as c_int,
                d1,
                d2.unwrap_or(-1.0),
                feature,
                tags.as_ptr(),
            )
        })
    }

    /// Persistent face names, indexed by face id - 1.
    pub fn face_tags(&self) -> Vec<u64> {
        let mut out = vec![0u64; self.face_count()];
        unsafe { rf_face_tags(self.raw(), out.as_mut_ptr()) };
        out
    }

    /// 1-based ids of the faces adjacent to an edge.
    pub fn edge_faces(&self, id: u32) -> Vec<u32> {
        let mut out = [0 as c_int; 2];
        let n = unsafe { rf_edge_faces(self.raw(), id as c_int, out.as_mut_ptr()) } as usize;
        out[..n].iter().map(|&f| f as u32).collect()
    }

    /// Persistent name of an edge: the (unordered) names of its two faces.
    pub fn edge_name(&self, id: u32) -> EdgeName {
        id.checked_sub(1)
            .and_then(|i| self.edge_names().names.get(i as usize))
            .copied()
            .unwrap_or_default()
    }

    fn edge_names(&self) -> &EdgeNames {
        self.0 .2.get_or_init(|| {
            let tags = self.face_tags();
            let mut names = EdgeNames::default();
            for id in 1..=self.edge_count() as u32 {
                let mut faces: Vec<_> = self
                    .edge_faces(id)
                    .iter()
                    .filter_map(|&f| tags.get(f as usize - 1).copied())
                    .collect();
                faces.sort_unstable();
                let name = EdgeName {
                    a: faces.first().copied().unwrap_or(0),
                    b: faces.get(1).copied().unwrap_or(0),
                };
                names.names.push(name);
                names.candidates.entry(name).or_default().push(id);
            }
            names
        })
    }

    /// Find edges by persistent name; when several share a name (a face split
    /// in two), the one nearest `hint` wins. Falls back to pure geometry.
    pub fn find_edge(&self, name: EdgeName, hint: [f64; 3]) -> Option<u32> {
        if !hint.iter().all(|v| v.is_finite()) {
            return None;
        }
        let d2 = |p: [f64; 3]| {
            (p[0] - hint[0]).powi(2) + (p[1] - hint[1]).powi(2) + (p[2] - hint[2]).powi(2)
        };
        let nearest = |ids: &[u32]| {
            ids.iter()
                .filter_map(|&i| self.edge_info(i).map(|e| (i, d2(e.mid))))
                .filter(|(_, d)| d.is_finite())
                .min_by(|a, b| a.1.total_cmp(&b.1))
        };
        if name.a != 0 {
            if let Some(ids) = self.edge_names().candidates.get(&name) {
                if let Some((i, _)) = nearest(ids) {
                    return Some(i);
                }
            }
        }
        let ids: Vec<_> = (1..=self.edge_count() as u32).collect();
        let (i, d) = nearest(&ids)?;
        (d < 1e-6).then_some(i)
    }

    /// Find a face by persistent name (nearest centroid among equals).
    pub fn find_face(&self, tag: u64, hint: [f64; 3]) -> Option<u32> {
        if !hint.iter().all(|v| v.is_finite()) {
            return None;
        }
        let d2 = |p: [f64; 3]| {
            (p[0] - hint[0]).powi(2) + (p[1] - hint[1]).powi(2) + (p[2] - hint[2]).powi(2)
        };
        let tags = self.face_tags();
        let cands: Vec<u32> = (1..=tags.len() as u32)
            .filter(|&i| tags[i as usize - 1] == tag && tag != 0)
            .collect();
        let pool: Vec<u32> = if cands.is_empty() {
            (1..=tags.len() as u32).collect()
        } else {
            cands.clone()
        };
        let best = pool
            .into_iter()
            .filter_map(|i| self.face_info(i).map(|f| (i, d2(f.centroid))))
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        (!cands.is_empty() || best.1 < 1e-6).then_some(best.0)
    }

    pub fn edge_info(&self, id: u32) -> Option<EdgeInfo> {
        let mut o = [0.0f64; 22];
        if unsafe { rf_edge_info(self.raw(), id as c_int, o.as_mut_ptr()) } == 0 {
            return None;
        }
        let v3 = |i: usize| [o[i], o[i + 1], o[i + 2]];
        Some(EdgeInfo {
            kind: match o[0] as i32 {
                0 => EdgeKind::Line,
                1 => EdgeKind::Circle {
                    center: v3(14),
                    axis: v3(17),
                    radius: o[20],
                },
                _ => EdgeKind::Other,
            },
            start: v3(1),
            end: v3(4),
            mid: v3(7),
            tangent: v3(10),
            length: o[13],
            closed: o[21] != 0.0,
        })
    }

    pub fn edge_sample(&self, id: u32, deflection: f64) -> Vec<[f64; 3]> {
        let mut buf = vec![0.0f64; 3 * 4096];
        let n =
            unsafe { rf_edge_sample(self.raw(), id as c_int, deflection, buf.as_mut_ptr(), 4096) }
                as usize;
        (0..n)
            .map(|i| [buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]])
            .collect()
    }

    pub fn face_info(&self, id: u32) -> Option<FaceInfo> {
        let mut o = [0.0f64; 14];
        if unsafe { rf_face_info(self.raw(), id as c_int, o.as_mut_ptr()) } == 0 {
            return None;
        }
        let v3 = |i: usize| [o[i], o[i + 1], o[i + 2]];
        let plane = (o[0] == 0.0).then(|| {
            let n = v3(4);
            let x = v3(7);
            Frame {
                origin: v3(1),
                x,
                y: cross(n, x),
            }
        });
        Some(FaceInfo {
            plane,
            centroid: v3(10),
            area: o[13],
        })
    }

    pub fn mesh(&self, linear_deflection: f64, angular_deflection: f64) -> Result<Mesh> {
        self.mesh_opts(linear_deflection, angular_deflection, false)
    }

    /// Prepare render geometry and tooltip metadata together on a worker.
    pub fn mesh_for_display(
        &self,
        linear_deflection: f64,
        angular_deflection: f64,
    ) -> Result<Mesh> {
        let mut mesh = self.mesh(linear_deflection, angular_deflection)?;
        mesh.volume = Some(self.volume().abs());
        Ok(mesh)
    }

    /// Initial display quality scales with the body rather than a fixed tolerance.
    pub fn mesh_adaptive_display(&self) -> Result<Mesh> {
        let linear = display_deflection(self.bbox(), None);
        let mut mesh = self.mesh_opts(linear, 0.2, true)?;
        mesh.volume = Some(self.volume().abs());
        Ok(mesh)
    }

    /// Axis-aligned bounding box (min, max).
    pub fn bbox(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut o = [0.0f64; 6];
        (unsafe { rf_bbox(self.raw(), o.as_mut_ptr()) } != 0)
            .then(|| ([o[0], o[1], o[2]], [o[3], o[4], o[5]]))
    }

    /// Mesh a private native copy at the requested quality. `fresh` explicitly
    /// cleans that copy; shared source triangulations are never mutated.
    pub fn mesh_opts(
        &self,
        linear_deflection: f64,
        angular_deflection: f64,
        fresh: bool,
    ) -> Result<Mesh> {
        if !linear_deflection.is_finite()
            || linear_deflection <= 0.
            || !angular_deflection.is_finite()
            || angular_deflection <= 0.
        {
            return Err(KernelError(
                "Mesh deflections must be positive and finite".into(),
            ));
        }
        let m = unsafe {
            rf_mesh(
                self.raw(),
                linear_deflection,
                angular_deflection,
                fresh as c_int,
            )
        };
        if m.is_null() {
            return Err(last_error());
        }
        let m = MeshHandle(m);
        let mut c = [0u32; 4];
        if unsafe { rf_mesh_counts(m.0, c.as_mut_ptr()) } == 0 {
            return Err(last_error());
        }
        let mut out = Mesh {
            volume: None,
            bounds: self.bbox(),
            linear_deflection,
            cache_key: MESH_KEY.fetch_add(1, Ordering::Relaxed),
            positions: vec![[0.0; 3]; c[0] as usize],
            normals: vec![[0.0; 3]; c[0] as usize],
            indices: vec![0; 3 * c[1] as usize],
            tri_face: vec![0; c[1] as usize],
            edge_points: vec![[0.0; 3]; c[2] as usize],
            edge_starts: vec![0; c[3] as usize],
        };
        unsafe {
            rf_mesh_copy(
                m.0,
                out.positions.as_mut_ptr() as *mut f32,
                out.normals.as_mut_ptr() as *mut f32,
                out.indices.as_mut_ptr(),
                out.tri_face.as_mut_ptr(),
                out.edge_points.as_mut_ptr() as *mut f32,
                out.edge_starts.as_mut_ptr(),
            );
        }
        Ok(out)
    }
}

fn cpath(path: &std::path::Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|e| KernelError(e.to_string()))
}

/// Read a STEP file (millimetres); loose faces are sewn into solids.
pub fn read_step(path: &std::path::Path) -> Result<Shape> {
    let p = cpath(path)?;
    check(unsafe { rf_read_step(p.as_ptr()) })
}

/// Read an IGES file; surfaces are sewn into solids where they close.
pub fn read_iges(path: &std::path::Path) -> Result<Shape> {
    let p = cpath(path)?;
    check(unsafe { rf_read_iges(p.as_ptr()) })
}

impl Shape {
    /// OCCT BRep text of this shape (used to embed imported geometry).
    pub fn to_brep(&self) -> Result<String> {
        let p = unsafe { rf_to_brep(self.raw()) };
        if p.is_null() {
            return Err(last_error());
        }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        unsafe { rf_free_str(p) };
        Ok(s)
    }

    pub fn from_brep(text: &str) -> Result<Shape> {
        let c = CString::new(text).map_err(|e| KernelError(e.to_string()))?;
        check(unsafe { rf_from_brep(c.as_ptr()) })
    }

    /// Name the faces (index i → tags[i]); used for imported bodies.
    pub fn with_face_tags(&self, tags: &[u64]) -> Result<Shape> {
        let n =
            c_int::try_from(tags.len()).map_err(|_| KernelError("Too many face tags".into()))?;
        check(unsafe { rf_with_face_tags(self.raw(), tags.as_ptr(), n) })
    }
}

pub fn write_step(shapes: &[Shape], path: &std::path::Path) -> Result<()> {
    let raws: Vec<*const RawShape> = shapes.iter().map(|s| s.raw() as *const _).collect();
    let p =
        CString::new(path.to_string_lossy().as_bytes()).map_err(|e| KernelError(e.to_string()))?;
    if unsafe { rf_write_step(raws.as_ptr(), raws.len() as c_int, p.as_ptr()) } == 0 {
        return Err(last_error());
    }
    Ok(())
}

/// Binary STL from a set of meshes (millimetres).
pub fn write_stl(meshes: &[Mesh], path: &std::path::Path) -> std::io::Result<()> {
    write_stl_opts(meshes, path, false, 1.0, "refusion")
}

/// STL writer: binary or ASCII, coordinates multiplied by `scale`
/// (mm → export unit, e.g. 1/25.4 for inches).
pub fn write_stl_opts(
    meshes: &[Mesh],
    path: &std::path::Path,
    ascii: bool,
    scale: f32,
    name: &str,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind, Write};
    if !scale.is_finite()
        || scale <= 0.
        || meshes.iter().any(|m| {
            m.indices.len() % 3 != 0 || m.indices.iter().any(|&i| i as usize >= m.positions.len())
        })
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Invalid STL scale or triangle indices",
        ));
    }
    let tris = meshes
        .iter()
        .try_fold(0usize, |n, m| n.checked_add(m.indices.len() / 3))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "STL triangle count exceeds u32"))?;
    // Scale only the three coordinates being written, never clone mesh buffers.
    let facet = |m: &Mesh, t: &[u32]| {
        let [a, b, c] = [
            m.positions[t[0] as usize].map(|v| v * scale),
            m.positions[t[1] as usize].map(|v| v * scale),
            m.positions[t[2] as usize].map(|v| v * scale),
        ];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-30);
        [n.map(|x| x / norm), a, b, c]
    };
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    if ascii {
        writeln!(w, "solid {name}")?;
    } else {
        w.write_all(&[0u8; 80])?;
        w.write_all(&tris.to_le_bytes())?;
    }
    for m in meshes {
        for t in m.indices.chunks_exact(3) {
            let [n, a, b, c] = facet(m, t);
            if ascii {
                writeln!(
                    w,
                    "  facet normal {:e} {:e} {:e}\n    outer loop",
                    n[0], n[1], n[2]
                )?;
                for p in [a, b, c] {
                    writeln!(w, "      vertex {:e} {:e} {:e}", p[0], p[1], p[2])?;
                }
                writeln!(w, "    endloop\n  endfacet")?;
            } else {
                let mut bytes = [0u8; 50];
                for (i, x) in [n, a, b, c].into_iter().flatten().enumerate() {
                    bytes[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
                }
                w.write_all(&bytes)?;
            }
        }
    }
    if ascii {
        writeln!(w, "endsolid {name}")?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    const XY: Frame = Frame {
        origin: [0.0; 3],
        x: [1.0, 0.0, 0.0],
        y: [0.0, 1.0, 0.0],
    };

    fn rect(w: f64, h: f64) -> Vec<Seg2> {
        vec![
            Seg2::Line {
                a: [0.0, 0.0],
                b: [w, 0.0],
            },
            Seg2::Line {
                a: [w, 0.0],
                b: [w, h],
            },
            Seg2::Line {
                a: [w, h],
                b: [0.0, h],
            },
            Seg2::Line {
                a: [0.0, h],
                b: [0.0, 0.0],
            },
        ]
    }

    #[test]
    fn extrude_rectangle_volume() {
        let f = Shape::face_from_loops(&XY, &[rect(10.0, 5.0)], &[]).unwrap();
        let s = f.extrude([0.0, 0.0, 1.0], 0.0, 2.0, 7).unwrap();
        assert!((s.volume() - 100.0).abs() < 1e-6);
        assert_eq!(s.face_count(), 6);
        assert_eq!(s.edge_count(), 12);
    }

    #[test]
    fn face_with_hole_and_cut() {
        let circle = vec![
            Seg2::Arc {
                c: [5.0, 5.0],
                a: [7.0, 5.0],
                b: [3.0, 5.0],
                ccw: false,
            },
            Seg2::Arc {
                c: [5.0, 5.0],
                a: [3.0, 5.0],
                b: [7.0, 5.0],
                ccw: false,
            },
        ];
        let f = Shape::face_from_loops(&XY, &[rect(10.0, 10.0), circle], &[]).unwrap();
        let s = f.extrude([0.0, 0.0, 1.0], 0.0, 1.0, 7).unwrap();
        let expect = 100.0 - std::f64::consts::PI * 4.0;
        assert!((s.volume() - expect).abs() < 1e-4, "{}", s.volume());
        let cutter = Shape::boxed(5.0, 20.0, 20.0).unwrap();
        let cut = s.boolean(&cutter, BoolOp::Cut).unwrap();
        assert!((cut.volume() - expect / 2.0).abs() < 1e-4);
    }

    #[test]
    fn fillet_and_chamfer_box() {
        let b = Shape::boxed(10.0, 10.0, 10.0).unwrap();
        // pick a straight edge
        let e = (1..=b.edge_count() as u32)
            .find(|&i| b.edge_info(i).unwrap().kind == EdgeKind::Line)
            .unwrap();
        let f = b.fillet(&[e], 2.0, 9).unwrap();
        let removed = (4.0 - std::f64::consts::PI) * 10.0; // r²(1-π/4)·L
        assert!(
            (f.volume() - (1000.0 - removed)).abs() < 1e-3,
            "{}",
            f.volume()
        );
        let c = b.chamfer(&[e], 2.0, None, 9).unwrap();
        assert!((c.volume() - (1000.0 - 20.0)).abs() < 1e-6);
        let mesh = f.mesh(0.05, 0.3).unwrap();
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.edge_count(), f.edge_count());
    }

    #[test]
    fn extrude_a_body_face() {
        let b = Shape::boxed(10.0, 10.0, 10.0).unwrap();
        // top face (normal +Z)
        let top = (1..=b.face_count() as u32)
            .find(|&f| {
                b.face_info(f)
                    .unwrap()
                    .plane
                    .is_some_and(|p| p.normal()[2] > 0.99)
            })
            .unwrap();
        let n = b.face_info(top).unwrap().plane.unwrap().normal();
        let face = b.face_copy(top).unwrap();
        let grown = b
            .boolean(&face.extrude(n, 0.0, 5.0, 3).unwrap(), BoolOp::Join)
            .unwrap();
        assert!((grown.volume() - 1500.0).abs() < 1e-6, "{}", grown.volume());
        let cut = b
            .boolean(&face.extrude(n, -4.0, 0.0, 3).unwrap(), BoolOp::Cut)
            .unwrap();
        assert!((cut.volume() - 600.0).abs() < 1e-6, "{}", cut.volume());
        // side faces of the new prism carry names derived from the box's edges
        assert!(grown.face_tags().iter().all(|&t| t != 0));
    }

    #[test]
    fn measure_faces_edges_vertices() {
        let b = Shape::boxed(10.0, 20.0, 30.0).unwrap();
        let face_with = |n: [f64; 3]| {
            (1..=b.face_count() as u32)
                .find(|&f| dot(b.face_info(f).unwrap().plane.unwrap().normal(), n) > 0.99)
                .unwrap()
        };
        let (d, p1, p2) = min_distance(
            &b,
            Sub::Face(face_with([0.0, 0.0, -1.0])),
            &b,
            Sub::Face(face_with([0.0, 0.0, 1.0])),
        )
        .unwrap();
        assert!((d - 30.0).abs() < 1e-9 && (p2[2] - p1[2] - 30.0).abs() < 1e-9);
        assert_eq!(b.vertex_count(), 8);
        let far = Shape::vertex([10.0, 20.0, 40.0]).unwrap();
        let (d, _, _) = min_distance(&b, Sub::Whole, &far, Sub::Whole).unwrap();
        assert!((d - 10.0).abs() < 1e-9);
        let seg = Shape::segment([-5.0, 0.0, 0.0], [-5.0, 20.0, 0.0]).unwrap();
        let (d, _, _) =
            min_distance(&b, Sub::Face(face_with([-1.0, 0.0, 0.0])), &seg, Sub::Whole).unwrap();
        assert!((d - 5.0).abs() < 1e-9);
    }

    /// The point of persistent naming: a top edge found in a 2 mm tall
    /// extrusion is found again, by name, after the height changes to 7 mm.
    #[test]
    fn edge_names_survive_parameter_change() {
        let tags = [11, 12, 13, 14];
        let make = |h: f64| {
            Shape::face_from_loops(&XY, &[rect(10.0, 5.0)], &tags)
                .unwrap()
                .extrude([0.0, 0.0, 1.0], 0.0, h, 42)
                .unwrap()
        };
        let a = make(2.0);
        let top_front = (1..=a.edge_count() as u32)
            .find(|&i| {
                let e = a.edge_info(i).unwrap();
                (e.mid[2] - 2.0).abs() < 1e-9 && e.mid[1].abs() < 1e-9
            })
            .unwrap();
        let name = a.edge_name(top_front);
        assert_ne!(name.a, 0);
        let b = make(7.0);
        let found = b.find_edge(name, [0.0; 3]).unwrap();
        let e = b.edge_info(found).unwrap();
        assert!(
            (e.mid[2] - 7.0).abs() < 1e-9 && e.mid[1].abs() < 1e-9,
            "{e:?}"
        );
        // and it survives a cut through another region of the body
        let cut = b
            .boolean(&Shape::boxed(2.0, 2.0, 20.0).unwrap(), BoolOp::Cut)
            .unwrap();
        let found = cut.find_edge(name, [5.0, 0.0, 7.0]).unwrap();
        let e = cut.edge_info(found).unwrap();
        assert!(
            (e.mid[2] - 7.0).abs() < 1e-9 && e.mid[1].abs() < 1e-9,
            "{e:?}"
        );
        // fillet faces get names derived from the edge they replace
        let f = cut.fillet(&[found], 1.0, 77).unwrap();
        assert!(f.face_tags().iter().all(|&t| t != 0));
    }

    #[test]
    fn revolve_rectangle_makes_tube() {
        let r = vec![
            Seg2::Line {
                a: [2.0, 0.0],
                b: [3.0, 0.0],
            },
            Seg2::Line {
                a: [3.0, 0.0],
                b: [3.0, 4.0],
            },
            Seg2::Line {
                a: [3.0, 4.0],
                b: [2.0, 4.0],
            },
            Seg2::Line {
                a: [2.0, 4.0],
                b: [2.0, 0.0],
            },
        ];
        let f = Shape::face_from_loops(&XY, &[r], &[]).unwrap();
        let s = f
            .revolve([0.0; 3], [0.0, 1.0, 0.0], std::f64::consts::TAU, 7)
            .unwrap();
        let expect = std::f64::consts::PI * (9.0 - 4.0) * 4.0;
        assert!((s.volume().abs() - expect).abs() < 1e-3, "{}", s.volume());
    }
}

#[cfg(test)]
mod id_tests {
    use super::*;
    /// Mesh triangle face ids, face_info and topology indices must agree.
    #[test]
    fn mesh_face_ids_match_face_info() {
        let s = Shape::boxed(40.0, 10.0, 25.0).unwrap();
        let f = s.fillet(&[1, 2], 2.0, 5).unwrap();
        let m = f.mesh(0.05, 0.3).unwrap();
        for id in 1..=f.face_count() as u32 {
            let info = f.face_info(id).unwrap();
            let (mut c, mut n) = ([0.0f64; 3], 0.0);
            let mut area = 0.0;
            for (t, &fid) in m.tri_face.iter().enumerate() {
                if fid != id {
                    continue;
                }
                let p: Vec<[f64; 3]> = (0..3)
                    .map(|k| m.positions[m.indices[3 * t + k] as usize].map(|v| v as f64))
                    .collect();
                let a = 0.5 * {
                    let u = sub(p[1], p[0]);
                    let v = sub(p[2], p[0]);
                    let x = cross(u, v);
                    dot(x, x).sqrt()
                };
                for k in 0..3 {
                    c[k] += a * (p[0][k] + p[1][k] + p[2][k]) / 3.0;
                }
                area += a;
                n += 1.0;
            }
            assert!(n > 0.0, "face {id} has no triangles");
            let c = c.map(|v| v / area);
            let d = sub(c, info.centroid);
            assert!(
                dot(d, d).sqrt() < 0.5,
                "face {id}: mesh centroid {c:?} vs face_info {:?}",
                info.centroid
            );
        }
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    #[test]
    fn scaled_binary_stl_streams_coordinates_and_preserves_input_mesh() {
        let mesh = Mesh {
            positions: vec![[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        let before = mesh.positions.clone();
        let path =
            std::env::temp_dir().join(format!("refusion-scaled-stl-{}.stl", std::process::id()));
        write_stl_opts(std::slice::from_ref(&mesh), &path, false, 2., "triangle").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let coordinate = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(bytes.len(), 134);
        assert_eq!(
            [coordinate(96), coordinate(112), coordinate(128)],
            [2., 4., 6.]
        );
        assert_eq!(mesh.positions, before);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn step_and_stl_export() {
        let dir = std::env::temp_dir().join(format!("refusion-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let b = Shape::boxed(10.0, 20.0, 30.0).unwrap();
        let step = dir.join("box.step");
        write_step(&[b.clone()], &step).unwrap();
        let text = std::fs::read_to_string(&step).unwrap();
        assert!(text.starts_with("ISO-10303-21;") && text.contains("MANIFOLD_SOLID_BREP"));
        // STEP round trip and BRep text round trip keep the geometry.
        let back = read_step(&step).unwrap();
        assert!((back.volume() - 6000.0).abs() < 1e-6, "{}", back.volume());
        assert_eq!(back.solids().len(), 1);
        let again = Shape::from_brep(&back.to_brep().unwrap()).unwrap();
        assert!((again.volume() - 6000.0).abs() < 1e-6);
        assert_eq!(again.face_count(), 6);
        let stl = dir.join("box.stl");
        let m = b.mesh(0.1, 0.5).unwrap();
        write_stl(&[m.clone()], &stl).unwrap();
        assert_eq!(
            std::fs::metadata(&stl).unwrap().len(),
            84 + 50 * (m.indices.len() as u64 / 3)
        );
        let ascii = dir.join("box_ascii.stl");
        write_stl_opts(&[m.clone()], &ascii, true, 1.0 / 25.4, "box").unwrap();
        let t = std::fs::read_to_string(&ascii).unwrap();
        assert!(
            t.starts_with("solid box") && t.matches("facet normal").count() == m.indices.len() / 3
        );
        // coarser fresh mesh has no more triangles than the fine one
        let fine = b.mesh_opts(0.001, 0.05, true).unwrap();
        let coarse = b.mesh_opts(1.0, 0.5, true).unwrap();
        assert!(coarse.indices.len() <= fine.indices.len());
        let (lo, hi) = b.bbox().unwrap();
        assert!((hi[2] - lo[2] - 30.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }
}
#[cfg(test)]
mod readonly_tests {
    use super::*;
    #[test]
    fn shared_shapes_mesh_concurrently_and_retagging_does_not_mutate_aliases() {
        let source = Shape::boxed(10., 20., 30.).unwrap();
        let before = source.to_brep().unwrap();
        let tags = source.face_tags();
        let original_name = source.edge_name(1);
        let mut threads = vec![];
        for i in 0..4 {
            let clone = source.clone();
            threads.push(std::thread::spawn(move || {
                let renamed = clone.with_face_tags(&[100 + i; 6]).unwrap();
                assert_ne!(renamed.edge_name(1), clone.edge_name(1));
                for _ in 0..3 {
                    let mesh = clone.mesh_opts(0.01 + i as f64 * 0.01, 0.2, true).unwrap();
                    assert!(!mesh.indices.is_empty());
                    assert!(clone
                        .find_edge(clone.edge_name(1), clone.edge_info(1).unwrap().mid)
                        .is_some());
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(source.face_tags(), tags);
        assert_eq!(source.edge_name(1), original_name);
        assert_eq!(source.to_brep().unwrap(), before);
        assert!(source
            .find_edge(original_name, [f64::NAN, 0., 0.])
            .is_none());
    }
    #[test]
    fn prepared_mesh_volume_and_cached_edge_adjacency_match_geometry() {
        let shape = Shape::boxed(10.0, 20.0, 30.0).unwrap();
        let mesh = shape.mesh_for_display(0.03, 0.2).unwrap();
        assert!((mesh.volume.unwrap() - 6000.0).abs() < 1e-6);
        for _ in 0..3 {
            for edge in 1..=shape.edge_count() as u32 {
                let faces = shape.edge_faces(edge);
                assert_eq!(faces.len(), 2);
                assert_ne!(faces[0], faces[1]);
                assert!(faces.iter().all(|&f| (1..=6).contains(&f)));
            }
        }
    }
    #[test]
    fn background_operations_preserve_meshed_source_geometry() {
        let source = Shape::boxed(20.0, 20.0, 20.0).unwrap();
        source.mesh(0.1, 0.2).unwrap();
        let before = source.to_brep().unwrap();
        source.fillet(&[1], 1.0, 77).unwrap();
        source.chamfer(&[1], 1.0, None, 78).unwrap();
        source.shell(&[1], -1.0, 79).unwrap();
        let cutter = Shape::boxed(5.0, 5.0, 25.0).unwrap();
        source.boolean(&cutter, BoolOp::Cut).unwrap();
        assert_eq!(source.to_brep().unwrap(), before);
        assert!((source.volume() - 8000.0).abs() < 1e-6);
    }
}
#[cfg(test)]
mod adaptive_display_tests {
    use super::*;
    #[test]
    fn quality_scales_with_size_zoom_and_rejects_invalid_scales() {
        let small = Some(([0.; 3], [1.; 3]));
        let large = Some(([0.; 3], [1000.; 3]));
        assert!(display_deflection(small, None) < display_deflection(large, None));
        assert!(display_deflection(large, Some(0.001)) < display_deflection(large, Some(1.)));
        assert_eq!(
            display_deflection(large, Some(f64::NAN)),
            display_deflection(large, None)
        );
        assert_eq!(
            display_deflection(large, Some(1.001)),
            display_deflection(large, Some(1.002))
        );
    }
    #[test]
    fn display_refinement_reduces_error_and_preserves_source_and_topology_ids() {
        let frame = Frame {
            origin: [0.; 3],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        };
        let face = Shape::face_from_loops(
            &frame,
            &[vec![
                Seg2::Arc {
                    c: [0., 0.],
                    a: [100., 0.],
                    b: [-100., 0.],
                    ccw: true,
                },
                Seg2::Arc {
                    c: [0., 0.],
                    a: [-100., 0.],
                    b: [100., 0.],
                    ccw: true,
                },
            ]],
            &[1, 2],
        )
        .unwrap();
        let source = face.extrude([0., 0., 1.], 0., 10., 2).unwrap();
        let before = source.to_brep().unwrap();
        let coarse = source
            .translated([0.; 3], 0)
            .unwrap()
            .mesh_opts(0.5, 0.2, true)
            .unwrap();
        let fine = source
            .translated([0.; 3], 0)
            .unwrap()
            .mesh_opts(0.005, 0.2, true)
            .unwrap();
        eprintln!(
            "display quality: coarse {} triangles, fine {} triangles",
            coarse.indices.len() / 3,
            fine.indices.len() / 3
        );
        assert!(fine.indices.len() > coarse.indices.len());
        assert_ne!(coarse.cache_key, fine.cache_key);
        assert_eq!(coarse.edge_count(), fine.edge_count());
        let faces = |m: &Mesh| {
            m.tri_face
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(faces(&coarse), faces(&fine));
        assert_eq!(source.to_brep().unwrap(), before);
        assert!((source.volume() - std::f64::consts::PI * 100000.).abs() < 1e-5);
        let export = source.mesh_opts(0.0001, 0.03, true).unwrap();
        assert!(export.indices.len() > fine.indices.len());
    }
}
