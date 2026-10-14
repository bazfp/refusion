use crate::expr::{self, Unit};
use refusion_kernel::{self as k, BoolOp, EdgeKind, EdgeName, Frame, Seg2, Shape};
use refusion_sketch::{find_profiles, Constraint, EntityId, Geom, Profile, Seg, Sketch};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub type FeatureId = u32;
pub type BodyId = u32;

// ------------------------------------------------------------------ references

/// Persistent reference to a face (by name, with a positional hint).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceRef {
    pub body: BodyId,
    pub tag: u64,
    pub hint: [f64; 3],
}

/// Persistent reference to an edge: the names of its two faces + a hint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeRef {
    pub body: BodyId,
    pub a: u64,
    pub b: u64,
    pub hint: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PlaneRef {
    XY,
    XZ,
    YZ,
    Face(FaceRef),
    Construction(FeatureId),
    /// Independent plane of imported sketch geometry, persisted in the document.
    Fixed {
        origin: [f64; 3],
        x: [f64; 3],
        y: [f64; 3],
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OffsetPlaneFeature {
    pub base: PlaneRef,
    pub distance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LinearRef {
    X,
    Y,
    Z,
    Edge(EdgeRef),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AngledPlaneFeature {
    pub base: PlaneRef,
    pub axis: LinearRef,
    pub angle: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShellFeature {
    pub body: BodyId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_tag: Option<u64>,
    pub faces: Vec<FaceRef>,
    pub thickness: String,
    pub outward: bool,
    /// When set, thickness is the inside offset and this is the outside offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outside_thickness: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HoleKind {
    Simple,
    Counterbore { diameter: String, depth: String },
    Countersink { diameter: String, angle: String },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HoleExtent {
    Distance { depth: String },
    ThroughAll,
    ToFace { face: FaceRef, offset: String },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoleFeature {
    pub body: BodyId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_tag: Option<u64>,
    pub plane: PlaneRef,
    /// Expression coordinates in the placement plane; one drilled hole per center.
    pub centers: Vec<[String; 2]>,
    pub diameter: String,
    pub kind: HoleKind,
    pub extent: HoleExtent,
    /// None creates a flat bottom; Some is the included drill point angle.
    pub tip_angle: Option<String>,
    pub flip: bool,
}
impl HoleFeature {
    pub fn new(body: BodyId, plane: PlaneRef) -> Self {
        Self {
            body,
            body_tag: None,
            plane,
            centers: vec![["0 mm".into(), "0 mm".into()]],
            diameter: "5 mm".into(),
            kind: HoleKind::Simple,
            extent: HoleExtent::Distance {
                depth: "5 mm".into(),
            },
            tip_angle: Some("118 deg".into()),
            flip: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MirrorFeature {
    pub bodies: Vec<BodyId>,
    #[serde(default)]
    pub join: bool,
    pub plane: PlaneRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PatternKind {
    Rectangular {
        axis1: LinearRef,
        axis2: LinearRef,
        count1: String,
        count2: String,
        distance1: String,
        distance2: String,
        spacing: bool,
    },
    Circular {
        axis: LinearRef,
        count: String,
        angle: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternObjects {
    #[default]
    Bodies,
    Features,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatternFeature {
    #[serde(default)]
    pub objects: PatternObjects,
    #[serde(default)]
    pub features: Vec<FeatureId>,
    pub bodies: Vec<BodyId>,
    pub kind: PatternKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllDirection {
    Positive,
    Negative,
    Both,
}

/// Which sketch region(s) a feature uses: the bounding curves, plus an interior point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileRef {
    pub curves: Vec<EntityId>,
    pub hint: [f64; 2],
}

// ------------------------------------------------------------------ features

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub edge: EdgeRef,
    /// Sketch entities created for it (points first, curve last).
    pub entities: Vec<EntityId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SketchFeature {
    pub plane: PlaneRef,
    pub sketch: Sketch,
    pub projections: Vec<Projection>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Extent {
    ToFace {
        face: FaceRef,
        offset: String,
    },
    Tapered {
        extent: Box<Extent>,
        angle: String,
    },
    /// Signed distance along the sketch normal.
    OneSide {
        d: String,
    },
    /// Total length, centred on the sketch plane.
    Symmetric {
        d: String,
    },
    SymmetricHalf {
        d: String,
    },
    TwoSides {
        d1: String,
        d2: String,
    },
    ThroughAll {
        direction: AllDirection,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    NewBody,
    Join,
    Cut,
    Intersect,
}

impl Operation {
    pub const ALL: [Operation; 4] = [
        Operation::Join,
        Operation::Cut,
        Operation::Intersect,
        Operation::NewBody,
    ];
    pub fn label(&self) -> &'static str {
        match self {
            Operation::NewBody => "New Body",
            Operation::Join => "Join",
            Operation::Cut => "Cut",
            Operation::Intersect => "Intersect",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtrudeFeature {
    /// Sketch the profiles come from (0 when only body faces are extruded).
    pub sketch: FeatureId,
    pub profiles: Vec<ProfileRef>,
    #[serde(default)]
    pub faces: Vec<FaceRef>,
    pub extent: Extent,
    pub op: Operation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Axis {
    SketchLine(EntityId),
    Edge(EdgeRef),
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevolveDirection {
    #[default]
    OneSide,
    TwoSides,
    Symmetric,
}
fn right_angle() -> String {
    "90 deg".into()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RevolveFeature {
    pub sketch: FeatureId,
    #[serde(default)]
    pub faces: Vec<FaceRef>,
    #[serde(default)]
    pub direction: RevolveDirection,
    #[serde(default = "right_angle")]
    pub angle2: String,
    #[serde(default)]
    pub full: bool,
    #[serde(default)]
    pub project_axis: bool,
    pub profiles: Vec<ProfileRef>,
    pub axis: Axis,
    pub angle: String,
    pub op: Operation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilletFeature {
    pub edges: Vec<EdgeRef>,
    pub radius: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChamferKind {
    Equal { d: String },
    TwoDistances { d1: String, d2: String },
    DistanceAngle { d: String, angle: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChamferFeature {
    pub edges: Vec<EdgeRef>,
    pub kind: ChamferKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportFeature {
    /// Original file name, for display.
    pub source: String,
    /// Immutable payload shared by undo snapshots and background transactions.
    /// Serde still encodes this as an ordinary JSON string.
    pub brep: Arc<str>,
    #[serde(skip)]
    pub cache: Option<Shape>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FeatureKind {
    Sketch(SketchFeature),
    Extrude(ExtrudeFeature),
    Revolve(RevolveFeature),
    Fillet(FilletFeature),
    Chamfer(ChamferFeature),
    Import(ImportFeature),
    OffsetPlane(OffsetPlaneFeature),
    AngledPlane(AngledPlaneFeature),
    Shell(ShellFeature),
    Hole(HoleFeature),
    Mirror(MirrorFeature),
    Pattern(PatternFeature),
}

impl FeatureKind {
    pub fn type_name(&self) -> &'static str {
        match self {
            FeatureKind::Sketch(_) => "Sketch",
            FeatureKind::Extrude(_) => "Extrude",
            FeatureKind::Revolve(_) => "Revolve",
            FeatureKind::Fillet(_) => "Fillet",
            FeatureKind::Chamfer(_) => "Chamfer",
            FeatureKind::Import(_) => "Import",
            FeatureKind::OffsetPlane(_) => "Offset Plane",
            FeatureKind::AngledPlane(_) => "Plane at Angle",
            FeatureKind::Shell(_) => "Shell",
            FeatureKind::Hole(_) => "Hole",
            FeatureKind::Mirror(_) => "Mirror",
            FeatureKind::Pattern(p) => match p.kind {
                PatternKind::Rectangular { .. } => "Rectangular Pattern",
                PatternKind::Circular { .. } => "Circular Pattern",
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Feature {
    pub id: FeatureId,
    pub name: String,
    pub suppressed: bool,
    pub kind: FeatureKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub expr: String,
    pub comment: String,
}

// ------------------------------------------------------------------ state

#[derive(Clone, Debug)]
pub struct Body {
    pub id: BodyId,
    pub name: String,
    pub shape: Shape,
}

#[derive(Clone, Debug)]
pub struct SketchState {
    pub frame: Frame,
    pub profiles: Vec<Profile>,
    /// Solved line endpoints (sketch coordinates), e.g. for revolve axes.
    pub lines: BTreeMap<EntityId, ([f64; 2], [f64; 2])>,
}

impl SketchState {
    pub fn of(frame: Frame, sk: &Sketch) -> Self {
        if sk.validate_data().is_err() {
            return Self {
                frame,
                profiles: vec![],
                lines: BTreeMap::new(),
            };
        }
        let lines = sk
            .entities
            .iter()
            .filter_map(|(&id, e)| match e.geom {
                Geom::Line { a, b } => Some((id, (sk.pos(a), sk.pos(b)))),
                _ => None,
            })
            .collect();
        SketchState {
            frame,
            profiles: find_profiles(sk),
            lines,
        }
    }
}

/// The model after replaying the timeline up to some feature.
#[derive(Clone, Debug, Default)]
pub struct State {
    pub bodies: Vec<Body>,
    pub sketches: BTreeMap<FeatureId, Arc<SketchState>>,
    pub planes: BTreeMap<FeatureId, Frame>,
    pub next_body: BodyId,
}

impl State {
    pub fn linear_ref(&self, axis: &LinearRef) -> Result<([f64; 3], [f64; 3]), String> {
        Ok(match axis {
            LinearRef::X => ([0.0; 3], [1.0, 0.0, 0.0]),
            LinearRef::Y => ([0.0; 3], [0.0, 1.0, 0.0]),
            LinearRef::Z => ([0.0; 3], [0.0, 0.0, 1.0]),
            LinearRef::Edge(r) => {
                let (bi, id) = self.resolve_edge(r).ok_or("axis edge no longer exists")?;
                let e = self.bodies[bi]
                    .shape
                    .edge_info(id)
                    .ok_or("axis edge is unavailable")?;
                if e.kind != EdgeKind::Line {
                    return Err("select a straight edge as the axis".into());
                }
                let d = k::sub(e.end, e.start);
                let l = k::dot(d, d).sqrt();
                if l < 1e-9 {
                    return Err("axis edge has zero length".into());
                }
                (e.start, d.map(|v| v / l))
            }
        })
    }
    fn add_body(&mut self, shape: Shape) -> BodyId {
        self.next_body += 1;
        let id = self.next_body;
        self.bodies.push(Body {
            id,
            name: format!("Body{id}"),
            shape,
        });
        id
    }
    pub fn body(&self, id: BodyId) -> Option<&Body> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn edge_ref(&self, body: BodyId, edge: u32) -> Option<EdgeRef> {
        let b = self.body(body)?;
        let n = b.shape.edge_name(edge);
        Some(EdgeRef {
            body,
            a: n.a,
            b: n.b,
            hint: b.shape.edge_info(edge)?.mid,
        })
    }

    pub fn face_ref(&self, body: BodyId, face: u32) -> Option<FaceRef> {
        let b = self.body(body)?;
        let tag = *b.shape.face_tags().get(face as usize - 1)?;
        Some(FaceRef {
            body,
            tag,
            hint: b.shape.face_info(face)?.centroid,
        })
    }

    /// Resolve an edge reference in this state: (body index, edge id).
    pub fn resolve_edge(&self, r: &EdgeRef) -> Option<(usize, u32)> {
        let name = EdgeName { a: r.a, b: r.b };
        let order: Vec<usize> = {
            let mut v: Vec<usize> = (0..self.bodies.len()).collect();
            v.sort_by_key(|&i| self.bodies[i].id != r.body);
            v
        };
        for i in order {
            if let Some(e) = self.bodies[i].shape.find_edge(name, r.hint) {
                return Some((i, e));
            }
        }
        None
    }

    pub fn resolve_face(&self, r: &FaceRef) -> Option<(usize, u32)> {
        let mut order: Vec<usize> = (0..self.bodies.len()).collect();
        order.sort_by_key(|&i| self.bodies[i].id != r.body);
        order.into_iter().find_map(|i| {
            self.bodies[i]
                .shape
                .find_face(r.tag, r.hint)
                .map(|f| (i, f))
        })
    }

    pub fn plane_frame(&self, p: &PlaneRef) -> Result<Frame, String> {
        Ok(match p {
            PlaneRef::XY => Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            PlaneRef::XZ => Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 0.0, -1.0],
            },
            PlaneRef::YZ => Frame {
                origin: [0.0; 3],
                x: [0.0, 0.0, -1.0],
                y: [0.0, 1.0, 0.0],
            },
            PlaneRef::Face(f) => {
                let (bi, fi) = self
                    .resolve_face(f)
                    .ok_or("sketch plane face no longer exists")?;
                self.bodies[bi]
                    .shape
                    .face_info(fi)
                    .and_then(|i| i.plane)
                    .ok_or("sketch face is not planar")?
            }
            PlaneRef::Construction(id) => *self
                .planes
                .get(id)
                .ok_or("construction plane no longer exists")?,
            PlaneRef::Fixed { origin, x, y } => {
                if !origin.iter().chain(x).chain(y).all(|v| v.is_finite())
                    || (k::dot(*x, *x) - 1.).abs() > 1e-7
                    || (k::dot(*y, *y) - 1.).abs() > 1e-7
                    || k::dot(*x, *y).abs() > 1e-7
                {
                    return Err("Imported sketch plane is not a finite orthonormal frame".into());
                }
                Frame {
                    origin: *origin,
                    x: *x,
                    y: *y,
                }
            }
        })
    }

    fn apply(&mut self, tool: Shape, op: Operation) -> Result<(), String> {
        let e = |x: k::KernelError| x.0;
        let touching: Vec<usize> = if op == Operation::NewBody {
            Vec::new()
        } else {
            (0..self.bodies.len())
                .filter(|&i| self.bodies[i].shape.touches(&tool))
                .collect()
        };
        match op {
            Operation::NewBody => {
                for s in tool.solids() {
                    self.add_body(s);
                }
            }
            Operation::Join if touching.is_empty() => {
                for s in tool.solids() {
                    self.add_body(s);
                }
            }
            Operation::Join => {
                let mut acc = self.bodies[touching[0]]
                    .shape
                    .boolean(&tool, BoolOp::Join)
                    .map_err(e)?;
                for &i in &touching[1..] {
                    acc = acc
                        .boolean(&self.bodies[i].shape, BoolOp::Join)
                        .map_err(e)?;
                }
                let solids = acc.solids();
                let keep = self.bodies[touching[0]].id;
                let drop: Vec<BodyId> = touching[1..].iter().map(|&i| self.bodies[i].id).collect();
                self.bodies.retain(|b| !drop.contains(&b.id));
                let b = self.bodies.iter_mut().find(|b| b.id == keep).unwrap();
                b.shape = if solids.len() == 1 {
                    solids[0].clone()
                } else {
                    acc
                };
            }
            Operation::Cut | Operation::Intersect => {
                if touching.is_empty() {
                    return Err(format!(
                        "the {} does not reach any body",
                        if op == Operation::Cut {
                            "cut"
                        } else {
                            "intersection"
                        }
                    ));
                }
                let bop = if op == Operation::Cut {
                    BoolOp::Cut
                } else {
                    BoolOp::Intersect
                };
                let ids: Vec<BodyId> = touching.iter().map(|&i| self.bodies[i].id).collect();
                for id in ids {
                    let idx = self.bodies.iter().position(|b| b.id == id).unwrap();
                    let r = self.bodies[idx].shape.boolean(&tool, bop).map_err(e)?;
                    let mut solids = r.solids().into_iter();
                    match solids.next() {
                        None => {
                            self.bodies.remove(idx);
                        }
                        Some(first) => {
                            self.bodies[idx].shape = first;
                            for s in solids {
                                self.add_body(s);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ document

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Version 0 is the legacy unversioned JSON format; writers emit version 1.
    #[serde(default)]
    format_version: u32,
    pub params: Vec<Param>,
    pub features: Vec<Feature>,
    /// Number of features currently rolled forward (timeline marker).
    pub marker: usize,
    next_feature: FeatureId,
    #[serde(skip)]
    cache: Vec<Arc<State>>,
    #[serde(skip)]
    pub errors: HashMap<FeatureId, String>,
    #[serde(skip)]
    pub parameter_error: Option<String>,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            format_version: 1,
            params: vec![],
            features: vec![],
            marker: 0,
            next_feature: 0,
            cache: vec![],
            errors: HashMap::new(),
            parameter_error: None,
        }
    }
}

fn feature_tag(id: FeatureId) -> u64 {
    k::mix(0xFEA7_0000, id as u64)
}

impl Document {
    /// Explicitly detach geometry throughout the timeline when independent
    /// instances are required. Application meshing no longer needs this copy:
    /// the kernel mesher isolates its own inputs. Identical states share copies.
    pub fn detach_display_shapes(
        &mut self,
        cached: &std::collections::HashSet<u64>,
    ) -> Result<(), String> {
        let mut copies = HashMap::new();
        for state in &mut self.cache {
            if state.bodies.iter().all(|b| cached.contains(&b.shape.uid())) {
                continue;
            }
            for body in &mut Arc::make_mut(state).bodies {
                let uid = body.shape.uid();
                if !cached.contains(&uid) {
                    if let Some(copy) = copies.get(&uid) {
                        body.shape = Shape::clone(copy);
                    } else {
                        let copy = body.shape.translated([0.0; 3], 0).map_err(|e| e.0)?;
                        copies.insert(uid, copy.clone());
                        body.shape = copy;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn cached_shape_uids(&self) -> impl Iterator<Item = u64> + '_ {
        self.cache
            .iter()
            .flat_map(|s| s.bodies.iter())
            .map(|b| b.shape.uid())
    }
    pub fn cached_shapes(&self) -> Vec<Shape> {
        let mut seen = std::collections::HashSet::new();
        self.cache
            .iter()
            .flat_map(|s| s.bodies.iter())
            .filter(|b| seen.insert(b.shape.uid()))
            .map(|b| b.shape.clone())
            .collect()
    }
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------- params

    /// Evaluated values of all user parameters (in dependency order).
    pub fn param_values(&self) -> Result<HashMap<String, f64>, String> {
        Ok(self
            .param_quantities()?
            .0
            .into_iter()
            .map(|(name, quantity)| (name, quantity.value))
            .collect())
    }

    pub fn param_quantities(&self) -> Result<expr::Parameters, String> {
        let mut vals = expr::Parameters::default();
        let mut pending: Vec<&Param> = self.params.iter().collect();
        while !pending.is_empty() {
            let before = pending.len();
            let mut rest = Vec::new();
            for p in pending {
                let refs = expr::references(&p.expr);
                if refs.iter().all(|r| vals.0.contains_key(r)) {
                    // Unitless parameters remain scalars (e.g. pattern counts);
                    // the consuming field supplies mm/degrees when appropriate.
                    let v = expr::eval_quantity(&p.expr, Unit::Scalar, &vals)
                        .map_err(|e| format!("{}: {e}", p.name))?;
                    if vals.0.insert(p.name.clone(), v).is_some() {
                        return Err(format!("Duplicate parameter name {}", p.name));
                    }
                } else {
                    rest.push(p);
                }
            }
            if rest.len() == before {
                return Err(format!(
                    "parameter '{}' has a circular or unknown reference",
                    rest[0].name
                ));
            }
            pending = rest;
        }
        Ok(vals)
    }

    pub fn eval(&self, src: &str, unit: Unit) -> Result<f64, String> {
        let vals = self.param_quantities()?;
        expr::eval(src, unit, &vals).map_err(|e| e.to_string())
    }

    // -------------------------------------------------------------- timeline

    pub fn index_of(&self, id: FeatureId) -> Option<usize> {
        self.features.iter().position(|f| f.id == id)
    }

    pub fn feature(&self, id: FeatureId) -> Option<&Feature> {
        self.features.iter().find(|f| f.id == id)
    }

    pub fn feature_mut(&mut self, id: FeatureId) -> Option<&mut Feature> {
        self.features.iter_mut().find(|f| f.id == id)
    }

    /// Insert a feature at the marker and recompute from there.
    pub fn add_feature(&mut self, kind: FeatureKind) -> FeatureId {
        self.next_feature += 1;
        let id = self.next_feature;
        let n = self
            .features
            .iter()
            .filter(|f| f.kind.type_name() == kind.type_name())
            .count()
            + 1;
        let name = format!("{}{}", kind.type_name(), n);
        let at = self.marker.min(self.features.len());
        self.features.insert(
            at,
            Feature {
                id,
                name,
                suppressed: false,
                kind,
            },
        );
        self.marker = at + 1;
        self.recompute_from(at);
        id
    }

    pub fn remove_feature(&mut self, id: FeatureId) {
        if let Some(i) = self.index_of(id) {
            self.features.remove(i);
            if self.marker > i {
                self.marker -= 1;
            }
            self.errors.remove(&id);
            self.recompute_from(i);
        }
    }

    /// Call after changing a feature (or its sketch).
    pub fn changed(&mut self, id: FeatureId) {
        if let Some(i) = self.index_of(id) {
            self.recompute_from(i);
        }
    }

    pub fn set_marker(&mut self, m: usize) {
        let m = m.min(self.features.len());
        let old = self.marker;
        self.marker = m;
        if m > old {
            self.recompute_from(old);
        } else {
            self.cache.truncate(m);
        }
    }

    /// Recompute everything (after loading or editing parameters).
    pub fn recompute_all(&mut self) {
        self.recompute_from(0);
    }

    /// Model state after the last rolled-forward feature.
    pub fn state(&self) -> &State {
        static EMPTY: std::sync::OnceLock<State> = std::sync::OnceLock::new();
        self.cache
            .last()
            .map(Arc::as_ref)
            .unwrap_or_else(|| EMPTY.get_or_init(State::default))
    }

    /// Model state just before a feature (what that feature "sees").
    pub fn state_before(&self, id: FeatureId) -> &State {
        static EMPTY: std::sync::OnceLock<State> = std::sync::OnceLock::new();
        match self.index_of(id) {
            Some(0) | None => EMPTY.get_or_init(State::default),
            Some(i) => self
                .cache
                .get(i - 1)
                .map(Arc::as_ref)
                .unwrap_or_else(|| self.state()),
        }
    }

    pub fn state_after(&self, id: FeatureId) -> Option<&State> {
        self.cache.get(self.index_of(id)?).map(Arc::as_ref)
    }

    pub fn recompute_from(&mut self, i: usize) {
        let vals = match self.param_quantities() {
            Ok(vals) => {
                self.parameter_error = None;
                vals
            }
            Err(error) => {
                self.parameter_error = Some(error.clone());
                for f in self
                    .features
                    .iter()
                    .take(self.marker)
                    .filter(|f| !f.suppressed)
                {
                    self.errors.insert(f.id, error.clone());
                }
                return;
            }
        };
        self.cache.truncate(i);
        for idx in i..self.marker.min(self.features.len()) {
            let mut st = self
                .cache
                .last()
                .map(|s| s.as_ref().clone())
                .unwrap_or_default();
            let id = self.features[idx].id;
            self.errors.remove(&id);
            if !self.features[idx].suppressed {
                let sources = self.pattern_sources(idx);
                let feat = &mut self.features[idx];
                if let Err(e) = sources.and_then(|sources| compute(feat, &mut st, &vals, sources)) {
                    self.errors.insert(id, e);
                    // Keep whatever state the failed feature left; sketches register even on solve errors.
                    st = self
                        .cache
                        .last()
                        .map(|s| s.as_ref().clone())
                        .unwrap_or_default();
                    if let FeatureKind::Sketch(sf) = &self.features[idx].kind {
                        if let Ok(frame) = st.plane_frame(&sf.plane) {
                            st.sketches
                                .insert(id, Arc::new(SketchState::of(frame, &sf.sketch)));
                        }
                    }
                }
            }
            self.cache.push(Arc::new(st));
        }
    }

    /// Extract optimized feature geometry only when a feature pattern needs it.
    /// Timeline snapshots are shared; ordinary replay incurs no extra booleans.
    fn pattern_sources(&self, idx: usize) -> Result<Option<Vec<PatternSource>>, String> {
        let FeatureKind::Pattern(pattern) = &self.features[idx].kind else {
            return Ok(None);
        };
        if pattern.objects == PatternObjects::Bodies {
            return Ok(None);
        }
        if pattern.features.is_empty() {
            return Err("Select one or more earlier solid features".into());
        }
        let mut ids = pattern.features.clone();
        ids.sort_by_key(|id| self.index_of(*id));
        ids.dedup();
        let mut sources = Vec::new();
        for id in ids {
            let at = self
                .index_of(id)
                .filter(|at| *at < idx)
                .ok_or("Pattern must reference earlier features")?;
            let feature = &self.features[at];
            if !feature.kind.supports_feature_pattern() {
                return Err(format!(
                    "{} is not supported by feature patterns",
                    feature.name
                ));
            }
            if feature.suppressed || self.errors.contains_key(&id) {
                return Err(format!(
                    "Pattern source {} is suppressed or failed",
                    feature.name
                ));
            }
            let before = self.state_before(id);
            let after = self
                .state_after(id)
                .ok_or("Pattern source has no computed state")?;
            // Body removal/merging and splitting would make target identity ambiguous.
            if before.bodies.iter().any(|b| after.body(b.id).is_none()) {
                return Err(
                    "Feature patterns do not support sources that remove or merge bodies".into(),
                );
            }
            let start = sources.len();
            for body in &after.bodies {
                if let Some(old) = before.body(body.id) {
                    if old.shape.uid() == body.shape.uid() {
                        continue;
                    }
                    for (a, b, op) in [
                        (&old.shape, &body.shape, Operation::Cut),
                        (&body.shape, &old.shape, Operation::Join),
                    ] {
                        let delta = a.boolean(b, BoolOp::Cut).map_err(|e| e.0)?;
                        if delta.volume() > 1e-9 {
                            sources.push(PatternSource {
                                shape: delta,
                                target: Some(body.id),
                                op,
                            });
                        }
                    }
                } else {
                    if before.bodies.iter().any(|b| {
                        !after
                            .body(b.id)
                            .is_some_and(|a| a.shape.uid() == b.shape.uid())
                    }) {
                        return Err(
                            "Feature patterns do not support sources that split bodies".into()
                        );
                    }
                    sources.push(PatternSource {
                        shape: body.shape.clone(),
                        target: None,
                        op: Operation::NewBody,
                    });
                }
            }
            if sources.len() == start {
                return Err(format!(
                    "Pattern source {} has no solid effect",
                    feature.name
                ));
            }
        }
        Ok(Some(sources))
    }

    // -------------------------------------------------------------- sketch helpers

    pub fn project_edge(
        &mut self,
        sketch: FeatureId,
        edge: EdgeRef,
    ) -> Result<Vec<EntityId>, String> {
        let st = self.state_before(sketch).clone();
        let frame = st.plane_frame(match &self.feature(sketch).ok_or("no sketch")?.kind {
            FeatureKind::Sketch(s) => &s.plane,
            _ => return Err("not a sketch".into()),
        })?;
        let Some(FeatureKind::Sketch(sf)) = self.feature_mut(sketch).map(|f| &mut f.kind) else {
            unreachable!()
        };
        if let Some(existing) = sf
            .projections
            .iter()
            .find(|p| p.edge == edge && p.entities.iter().all(|id| sf.sketch.entity(*id).is_some()))
        {
            return Ok(existing.entities.clone());
        }
        let ents = make_projection(&st, &frame, &edge, &mut sf.sketch)?;
        sf.projections.push(Projection {
            edge,
            entities: ents.clone(),
        });
        Ok(ents)
    }

    /// Add an Import base feature holding `shape`.
    pub fn add_import(&mut self, source: &str, shape: &Shape) -> Result<FeatureId, String> {
        let brep = shape.to_brep().map_err(|e| e.0)?;
        let id = self.add_feature(FeatureKind::Import(ImportFeature {
            source: source.into(),
            brep: brep.into(),
            cache: Some(shape.clone()),
        }));
        match self.errors.get(&id) {
            Some(e) => Err(e.clone()),
            None => Ok(id),
        }
    }

    // -------------------------------------------------------------- io

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialisable")
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let mut d: Document = serde_json::from_str(s).map_err(|e| e.to_string())?;
        d.validate_loaded()?;
        d.format_version = 1;
        d.recompute_all();
        Ok(d)
    }
}

#[path = "validation.rs"]
mod validation;

// ------------------------------------------------------------------ compute

struct PatternSource {
    shape: Shape,
    target: Option<BodyId>,
    op: Operation,
}

impl FeatureKind {
    pub fn supports_feature_pattern(&self) -> bool {
        matches!(
            self,
            Self::Extrude(_)
                | Self::Revolve(_)
                | Self::Hole(_)
                | Self::Fillet(_)
                | Self::Chamfer(_)
        )
    }
}

fn compute(
    f: &mut Feature,
    st: &mut State,
    vals: &dyn expr::Scope,
    pattern_sources: Option<Vec<PatternSource>>,
) -> Result<(), String> {
    let ev = |s: &str, u: Unit| expr::eval(s, u, vals).map_err(|e| e.to_string());
    let tag = feature_tag(f.id);
    match &mut f.kind {
        FeatureKind::AngledPlane(plane) => {
            let base = st.plane_frame(&plane.base)?;
            let (origin, axis) = st.linear_ref(&plane.axis)?;
            let angle = ev(&plane.angle, Unit::Angle)?;
            if !angle.is_finite() {
                return Err("Plane angle must be finite".into());
            }
            let n = base.normal();
            if k::dot(axis, n).abs() > 1e-6 || k::dot(k::sub(origin, base.origin), n).abs() > 1e-5 {
                return Err("The axis must lie in the reference plane".into());
            }
            let y = rotate_vec(k::cross(n, axis), axis, angle);
            st.planes.insert(f.id, Frame { origin, x: axis, y });
            Ok(())
        }
        FeatureKind::Hole(hole) => compute_hole(st, hole, vals, tag),
        FeatureKind::Shell(shell) => {
            let bi = feature_body(st, shell.body, shell.body_tag)?;
            let t = ev(&shell.thickness, Unit::Length)?;
            if !t.is_finite() || t <= 0.0 {
                return Err("Shell thickness must be positive and finite".into());
            }
            let mut ids = Vec::new();
            for face in &shell.faces {
                let (b, id) = st.resolve_face(face).ok_or("shell face no longer exists")?;
                if b != bi {
                    return Err("Shell faces must belong to the selected body".into());
                }
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            let shape = if let Some(outside) = &shell.outside_thickness {
                let outside = ev(outside, Unit::Length)?;
                st.bodies[bi].shape.shell_both(&ids, t, outside, tag)
            } else {
                st.bodies[bi]
                    .shape
                    .shell(&ids, if shell.outward { t } else { -t }, tag)
            }
            .map_err(|e| e.0)?;
            if shape.solids().len() != 1 || shape.volume() <= 1e-9 {
                return Err("Shell did not produce one valid solid".into());
            }
            st.bodies[bi].shape = shape;
            Ok(())
        }
        FeatureKind::Mirror(mirror) => {
            let plane = st.plane_frame(&mirror.plane)?;
            let sources = selected_bodies(st, &mirror.bodies)?;
            let copies: Result<Vec<_>, _> = sources
                .iter()
                .enumerate()
                .map(|(i, s)| s.mirrored(plane, k::mix(tag, i as u64)).map_err(|e| e.0))
                .collect();
            for shape in copies? {
                if mirror.join {
                    st.apply(shape, Operation::Join)?;
                } else {
                    st.add_body(shape);
                }
            }
            Ok(())
        }
        FeatureKind::Pattern(pattern) => {
            let sources = match pattern_sources {
                Some(sources) => sources,
                None => selected_bodies(st, &pattern.bodies)?
                    .into_iter()
                    .map(|shape| PatternSource {
                        shape,
                        target: None,
                        op: Operation::NewBody,
                    })
                    .collect(),
            };
            let mut copies = Vec::new();
            match &pattern.kind {
                PatternKind::Rectangular {
                    axis1,
                    axis2,
                    count1,
                    count2,
                    distance1,
                    distance2,
                    spacing,
                } => {
                    let n1 = pattern_count(ev(count1, Unit::Scalar)?)?;
                    let n2 = pattern_count(ev(count2, Unit::Scalar)?)?;
                    if n1
                        .checked_mul(n2)
                        .and_then(|n| n.checked_mul(sources.len()))
                        .is_none_or(|n| n > 1000)
                    {
                        return Err("Pattern is limited to 1000 instances".into());
                    }
                    let (_, a1) = st.linear_ref(axis1)?;
                    let (_, a2) = st.linear_ref(axis2)?;
                    if n2 > 1 && k::dot(a1, a2).abs() > 1.0 - 1e-8 {
                        return Err("Pattern directions must not be parallel".into());
                    }
                    let d1 = ev(distance1, Unit::Length)?;
                    let d2 = ev(distance2, Unit::Length)?;
                    if !d1.is_finite()
                        || !d2.is_finite()
                        || n1 > 1 && d1.abs() < 1e-9
                        || n2 > 1 && d2.abs() < 1e-9
                    {
                        return Err("Active pattern distances must be finite and nonzero".into());
                    }
                    let step1 = if *spacing || n1 == 1 {
                        d1
                    } else {
                        d1 / (n1 - 1) as f64
                    };
                    let step2 = if *spacing || n2 == 1 {
                        d2
                    } else {
                        d2 / (n2 - 1) as f64
                    };
                    for i in 0..n1 {
                        for j in 0..n2 {
                            if i == 0 && j == 0 {
                                continue;
                            }
                            let delta = std::array::from_fn(|k| {
                                a1[k] * step1 * i as f64 + a2[k] * step2 * j as f64
                            });
                            for (b, s) in sources.iter().enumerate() {
                                copies.push((
                                    s.shape
                                        .translated(
                                            delta,
                                            k::mix(tag, ((i * n2 + j) * sources.len() + b) as u64),
                                        )
                                        .map_err(|e| e.0)?,
                                    s.target,
                                    s.op,
                                ));
                            }
                        }
                    }
                }
                PatternKind::Circular { axis, count, angle } => {
                    let count = pattern_count(ev(count, Unit::Scalar)?)?;
                    if count * sources.len() > 1000 {
                        return Err("Pattern is limited to 1000 instances".into());
                    }
                    let angle = ev(angle, Unit::Angle)?;
                    if !angle.is_finite()
                        || angle.abs() < 1e-9
                        || angle.abs() > std::f64::consts::TAU + 1e-9
                    {
                        return Err("Pattern angle must be nonzero and at most 360 degrees".into());
                    }
                    let (origin, axis) = st.linear_ref(axis)?;
                    let full = (angle.abs() - std::f64::consts::TAU).abs() < 1e-9;
                    let step = angle
                        / if full {
                            count as f64
                        } else {
                            (count - 1).max(1) as f64
                        };
                    for i in 1..count {
                        for (b, s) in sources.iter().enumerate() {
                            copies.push((
                                s.shape
                                    .rotated(
                                        origin,
                                        axis,
                                        step * i as f64,
                                        k::mix(tag, (i * sources.len() + b) as u64),
                                    )
                                    .map_err(|e| e.0)?,
                                s.target,
                                s.op,
                            ));
                        }
                    }
                }
            }
            for (shape, target, op) in copies {
                if let Some(target) = target {
                    let body = st
                        .bodies
                        .iter_mut()
                        .find(|b| b.id == target)
                        .ok_or("Pattern target body no longer exists")?;
                    let before = body.shape.volume();
                    let result = body
                        .shape
                        .boolean(
                            &shape,
                            if op == Operation::Cut {
                                BoolOp::Cut
                            } else {
                                BoolOp::Join
                            },
                        )
                        .map_err(|e| e.0)?;
                    if !result.is_valid() || result.solids().len() != 1 || result.volume() <= 1e-9 {
                        return Err(
                            "Feature pattern instance must leave one valid target solid".into()
                        );
                    }
                    if (result.volume() - before).abs() <= 1e-9 {
                        return Err(
                            "Feature pattern instance has no effect on its target body".into()
                        );
                    }
                    body.shape = result;
                } else {
                    st.add_body(shape);
                }
            }
            Ok(())
        }
        FeatureKind::OffsetPlane(plane) => {
            let mut frame = st.plane_frame(&plane.base)?;
            let d = ev(&plane.distance, Unit::Length)?;
            if !d.is_finite() {
                return Err("Plane offset must be finite".into());
            }
            let normal = frame.normal();
            for (i, n) in normal.iter().enumerate() {
                frame.origin[i] += n * d;
            }
            st.planes.insert(f.id, frame);
            Ok(())
        }
        FeatureKind::Sketch(sf) => {
            let frame = st.plane_frame(&sf.plane)?;
            update_projections(st, &frame, sf);
            // Dimension expressions → values.
            let ids: Vec<_> = sf
                .sketch
                .constraints
                .iter()
                .filter(|(_, r)| r.expr.is_some() && !r.driven)
                .map(|(&i, _)| i)
                .collect();
            for cid in ids {
                let rec = sf.sketch.constraints.get_mut(&cid).unwrap();
                let unit = if matches!(rec.c, Constraint::Angle { .. }) {
                    Unit::Angle
                } else {
                    Unit::Length
                };
                let v = ev(rec.expr.as_ref().unwrap(), unit)?;
                rec.c.set_value(v);
            }
            let rep = sf.sketch.solve();
            st.sketches
                .insert(f.id, Arc::new(SketchState::of(frame, &sf.sketch)));
            if !rep.converged {
                return Err(rep.error.unwrap_or_else(|| {
                    format!(
                        "sketch could not be solved (residual {:.2e})",
                        rep.max_residual
                    )
                }));
            }
            Ok(())
        }
        FeatureKind::Extrude(x) => {
            if x.profiles.is_empty() && x.faces.is_empty() {
                return Err("no profile or face selected".into());
            }
            // (face, direction) pairs: sketch profiles go along the sketch normal,
            // body faces along their own outward normal.
            let mut inputs: Vec<(Shape, [f64; 3])> = Vec::new();
            if !x.profiles.is_empty() {
                let ss = st
                    .sketches
                    .get(&x.sketch)
                    .ok_or("the sketch is not available")?
                    .clone();
                let n = ss.frame.normal();
                for f in profile_faces(&ss, &x.profiles, x.sketch)? {
                    inputs.push((f, n));
                }
            }
            for r in &x.faces {
                let (bi, fid) = st
                    .resolve_face(r)
                    .ok_or("a selected face no longer exists")?;
                let body = &st.bodies[bi].shape;
                let plane = body
                    .face_info(fid)
                    .and_then(|i| i.plane)
                    .ok_or("only planar faces can be extruded")?;
                inputs.push((body.face_copy(fid).map_err(|e| e.0)?, plane.normal()));
            }
            let (extent, taper) = match &x.extent {
                Extent::Tapered { extent, angle } => (extent.as_ref(), ev(angle, Unit::Angle)?),
                extent => (extent, 0.),
            };
            if !taper.is_finite() || taper.abs() >= 85f64.to_radians() {
                return Err("Taper angle must be between -85 and 85 degrees".into());
            }
            let target = if let Extent::ToFace { face, offset } = extent {
                let (bi, fid) = st
                    .resolve_face(face)
                    .ok_or("the target face no longer exists")?;
                let plane = st.bodies[bi]
                    .shape
                    .face_info(fid)
                    .and_then(|i| i.plane)
                    .ok_or("To face requires a planar target face")?;
                Some((plane, ev(offset, Unit::Length)?))
            } else {
                None
            };
            let distances = match extent {
                Extent::OneSide { d } => {
                    let d = ev(d, Unit::Length)?;
                    Some((d.min(0.0), d.max(0.0)))
                }
                Extent::Symmetric { d } => {
                    let d = ev(d, Unit::Length)?.abs();
                    Some((-d / 2.0, d / 2.0))
                }
                Extent::SymmetricHalf { d } => {
                    let d = ev(d, Unit::Length)?.abs();
                    Some((-d, d))
                }
                Extent::TwoSides { d1, d2 } => {
                    Some((-ev(d2, Unit::Length)?, ev(d1, Unit::Length)?))
                }
                Extent::ThroughAll { .. } | Extent::ToFace { .. } => None,
                Extent::Tapered { .. } => return Err("Nested taper extent is not supported".into()),
            };
            let mut tool: Option<Shape> = None;
            for (face, n) in inputs {
                let mut trim = None;
                let (d0, d1) = if let Some((mut plane, offset)) = target {
                    let pn = plane.normal();
                    let dot = |a: [f64; 3], b: [f64; 3]| {
                        a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f64>()
                    };
                    let denom = dot(n, pn);
                    if denom.abs() < 1e-8 {
                        return Err("Target face is parallel to the extrusion direction".into());
                    }
                    for i in 0..3 {
                        plane.origin[i] += n[i] * offset;
                    }
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    let (b0, b1) = face.bbox().ok_or("Profile bounds unavailable")?;
                    for mask in 0..8 {
                        let p: [f64; 3] =
                            std::array::from_fn(
                                |j| if mask & (1 << j) == 0 { b0[j] } else { b1[j] },
                            );
                        let d = dot(std::array::from_fn(|j| plane.origin[j] - p[j]), pn) / denom;
                        lo = lo.min(d);
                        hi = hi.max(d);
                    }
                    if !lo.is_finite()
                        || !hi.is_finite()
                        || lo * hi <= 0.
                        || lo.abs().min(hi.abs()) < 1e-7
                    {
                        return Err(
                            "Target face intersects the profile plane or has zero extent".into(),
                        );
                    }
                    let margin = (hi - lo).abs();
                    let distance = if lo > 0. {
                        hi + margin + 1e-5
                    } else {
                        lo - margin - 1e-5
                    };
                    let inside = face
                        .face_info(1)
                        .ok_or("Profile plane unavailable")?
                        .centroid;
                    trim = Some((plane, inside));
                    (distance.min(0.), distance.max(0.))
                } else if let Some(d) = distances {
                    d
                } else {
                    let Extent::ThroughAll { direction } = extent else {
                        unreachable!()
                    };
                    through_all(st, &face, n, *direction)?
                };
                if !d0.is_finite() || !d1.is_finite() || d1 - d0 <= 1e-9 {
                    return Err("Extrude extent must have positive finite length".into());
                }
                let mut s = face
                    .extrude_tapered(n, d0, d1, taper, tag)
                    .map_err(|e| e.0)?;
                if let Some((plane, inside)) = trim {
                    s = s.trim_at_plane(plane, inside, tag).map_err(|e| e.0)?;
                }
                if !s.is_solid() || !s.is_valid() || s.volume().abs() < 1e-9 {
                    return Err("Extrude produced no valid solid".into());
                }
                tool = Some(match tool {
                    None => s,
                    Some(t) => t.boolean(&s, BoolOp::Join).map_err(|e| e.0)?,
                });
            }
            st.apply(tool.unwrap(), x.op)
        }
        FeatureKind::Revolve(r) => {
            let ss = st.sketches.get(&r.sketch);
            let mut faces = if r.profiles.is_empty() {
                vec![]
            } else {
                profile_faces(
                    ss.ok_or("the sketch is not available")?,
                    &r.profiles,
                    r.sketch,
                )?
            };
            for reference in &r.faces {
                let (bi, id) = st
                    .resolve_face(reference)
                    .ok_or("revolve face no longer exists")?;
                let face = st.bodies[bi].shape.face_copy(id).map_err(|e| e.0)?;
                if face.face_info(1).and_then(|f| f.plane).is_none() {
                    return Err("Select planar faces to revolve".into());
                }
                faces.push(face);
            }
            let frame = faces
                .first()
                .and_then(|face| face.face_info(1))
                .and_then(|f| f.plane)
                .ok_or("Select a profile or planar face")?;
            for face in &faces {
                let other = face
                    .face_info(1)
                    .and_then(|f| f.plane)
                    .ok_or("Revolve requires planar profiles")?;
                if k::dot(frame.normal(), other.normal()).abs() < 1. - 1e-6
                    || k::dot(k::sub(other.origin, frame.origin), frame.normal()).abs() > 1e-5
                {
                    return Err("Revolve profiles must be coplanar".into());
                }
            }
            let angle = if r.full {
                std::f64::consts::TAU
            } else {
                ev(&r.angle, Unit::Angle)?
            };
            let (start, sweep) = if r.full {
                (0., angle)
            } else {
                match r.direction {
                    RevolveDirection::OneSide => (0., angle),
                    RevolveDirection::Symmetric => (-angle, 2. * angle),
                    RevolveDirection::TwoSides => {
                        let second = ev(&r.angle2, Unit::Angle)?;
                        if angle <= 0. || second < 0. {
                            return Err("Two-sided revolve angles must be positive".into());
                        }
                        (-second, angle + second)
                    }
                }
            };
            if !sweep.is_finite()
                || !start.is_finite()
                || sweep.abs() < 1e-9
                || sweep.abs() > std::f64::consts::TAU + 1e-9
            {
                return Err("Revolve sweep must be nonzero and at most 360 degrees".into());
            }
            let (mut origin, mut axis) = match &r.axis {
                Axis::Edge(reference) => {
                    let (bi, id) = st
                        .resolve_edge(reference)
                        .ok_or("Revolve axis no longer exists")?;
                    let edge = st.bodies[bi]
                        .shape
                        .edge_info(id)
                        .ok_or("Revolve axis is unavailable")?;
                    match edge.kind {
                        EdgeKind::Line => (edge.start, k::sub(edge.end, edge.start)),
                        EdgeKind::Circle { center, axis, .. } => (center, axis),
                        _ => return Err("Select a linear or circular axis object".into()),
                    }
                }
                _ => revolve_axis(&r.axis, ss.map(Arc::as_ref))?,
            };
            if r.project_axis {
                let n = frame.normal();
                origin = std::array::from_fn(|i| {
                    origin[i] - n[i] * k::dot(k::sub(origin, frame.origin), n)
                });
                axis = std::array::from_fn(|i| axis[i] - n[i] * k::dot(axis, n));
            }
            let length = k::dot(axis, axis).sqrt();
            if length < 1e-9 {
                return Err("Projected revolve axis has zero length; disable Project Axis or select another axis".into());
            }
            axis = axis.map(|v| v / length);
            let mut tool: Option<Shape> = None;
            for face in faces {
                let face = if start.abs() > 1e-12 {
                    face.rotated(origin, axis, start, tag).map_err(|e| e.0)?
                } else {
                    face
                };
                let solid = face.revolve(origin, axis, sweep, tag).map_err(|e| e.0)?;
                if !solid.is_solid() || !solid.is_valid() || solid.volume().abs() < 1e-9 {
                    return Err("Revolve produced no valid solid".into());
                }
                tool = Some(match tool {
                    None => solid,
                    Some(t) => t.boolean(&solid, BoolOp::Join).map_err(|e| e.0)?,
                });
            }
            st.apply(tool.unwrap(), r.op)
        }
        FeatureKind::Fillet(fl) => {
            let r = ev(&fl.radius, Unit::Length)?;
            if r <= 0.0 {
                return Err("radius must be positive".into());
            }
            blend(st, &fl.edges, |s, ids| s.fillet(ids, r, tag))
        }
        FeatureKind::Import(imp) => {
            if imp.cache.is_none() {
                imp.cache = Some(Shape::from_brep(&imp.brep).map_err(|e| e.0)?);
            }
            let shape = imp.cache.clone().unwrap();
            let mut parts = shape.solids();
            if parts.is_empty() {
                parts.push(shape); // surface / open shell bodies
            }
            let stem = std::path::Path::new(&imp.source)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or("Imported".into());
            let n = parts.len();
            for (i, part) in parts.into_iter().enumerate() {
                // Stable face names: (feature, part, face index).
                let tags: Vec<u64> = (0..part.face_count() as u64)
                    .map(|f| k::mix(k::mix(tag, 5), (i as u64) << 32 | f))
                    .collect();
                let part = part.with_face_tags(&tags).map_err(|e| e.0)?;
                let id = st.add_body(part);
                if let Some(b) = st.bodies.iter_mut().find(|b| b.id == id) {
                    b.name = if n == 1 {
                        stem.clone()
                    } else {
                        format!("{stem} {}", i + 1)
                    };
                }
            }
            Ok(())
        }
        FeatureKind::Chamfer(c) => {
            let (d1, d2) = match &c.kind {
                ChamferKind::Equal { d } => (ev(d, Unit::Length)?, None),
                ChamferKind::TwoDistances { d1, d2 } => {
                    (ev(d1, Unit::Length)?, Some(ev(d2, Unit::Length)?))
                }
                ChamferKind::DistanceAngle { d, angle } => {
                    let d = ev(d, Unit::Length)?;
                    (d, Some(d * ev(angle, Unit::Angle)?.tan()))
                }
            };
            if d1 <= 0.0 || d2.is_some_and(|v| v <= 0.0) {
                return Err("chamfer distances must be positive".into());
            }
            blend(st, &c.edges, |s, ids| s.chamfer(ids, d1, d2, tag))
        }
    }
}

fn selected_bodies(st: &State, ids: &[BodyId]) -> Result<Vec<Shape>, String> {
    if ids.is_empty() {
        return Err("Select at least one body".into());
    }
    let mut seen = std::collections::HashSet::new();
    ids.iter()
        .filter(|id| seen.insert(**id))
        .map(|id| {
            st.body(*id)
                .map(|b| b.shape.clone())
                .ok_or("Selected body no longer exists".into())
        })
        .collect()
}

fn pattern_count(value: f64) -> Result<usize, String> {
    if !value.is_finite() || value < 1.0 || value > 1000.0 || (value - value.round()).abs() > 1e-9 {
        return Err("Pattern quantity must be a whole number between 1 and 1000".into());
    }
    Ok(value as usize)
}

fn rotate_vec(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let cross = k::cross(axis, v);
    let dot = k::dot(axis, v);
    std::array::from_fn(|i| {
        v[i] * angle.cos() + cross[i] * angle.sin() + axis[i] * dot * (1.0 - angle.cos())
    })
}

fn bbox_corners(lo: [f64; 3], hi: [f64; 3]) -> impl Iterator<Item = [f64; 3]> {
    (0..8)
        .map(move |mask| std::array::from_fn(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] }))
}

fn through_all(
    st: &State,
    face: &Shape,
    normal: [f64; 3],
    direction: AllDirection,
) -> Result<(f64, f64), String> {
    let origin = face
        .face_info(1)
        .and_then(|f| f.plane)
        .ok_or("Through All requires a planar profile")?
        .origin;
    let plane = k::dot(origin, normal);
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for body in &st.bodies {
        if let Some((lo, hi)) = body.shape.bbox() {
            for p in bbox_corners(lo, hi) {
                let d = k::dot(p, normal) - plane;
                low = low.min(d);
                high = high.max(d);
            }
        }
    }
    if !low.is_finite() || !high.is_finite() {
        return Err("Through All requires a target body".into());
    }
    let margin = ((high - low) * 0.01).max(1.0);
    Ok(match direction {
        AllDirection::Positive => (0.0, (high + margin).max(margin)),
        AllDirection::Negative => ((low - margin).min(-margin), 0.0),
        AllDirection::Both => ((low - margin).min(-margin), (high + margin).max(margin)),
    })
}

/// Body ordinals can change when earlier patterns gain instances. New commands
/// persist an existing face tag as a stable anchor; old files retain ID lookup.
fn feature_body(st: &State, body: BodyId, anchor: Option<u64>) -> Result<usize, String> {
    st.bodies
        .iter()
        .position(|b| match anchor {
            Some(tag) => b.shape.face_tags().contains(&tag),
            None => b.id == body,
        })
        .ok_or_else(|| "Selected body no longer exists".into())
}

fn compute_hole(
    st: &mut State,
    hole: &HoleFeature,
    vals: &dyn expr::Scope,
    tag: u64,
) -> Result<(), String> {
    let ev = |s: &str, u: Unit| expr::eval(s, u, vals).map_err(|e| e.to_string());
    let positive = |s: &str| -> Result<f64, String> {
        let v = ev(s, Unit::Length)?;
        if !v.is_finite() || v <= 1e-8 {
            return Err("Hole lengths must be positive and finite".into());
        }
        Ok(v)
    };
    let angle = |s: &str| -> Result<f64, String> {
        let v = ev(s, Unit::Angle)?;
        if !v.is_finite() || v <= 1e-6 || v >= std::f64::consts::PI - 1e-6 {
            return Err("Hole angles must be between 0 and 180 degrees".into());
        }
        Ok(v)
    };
    let bi = feature_body(st, hole.body, hole.body_tag)?;
    let frame = st.plane_frame(&hole.plane)?;
    let axis = frame.normal().map(|v| v * if hole.flip { 1. } else { -1. });
    let radius = positive(&hole.diameter)? / 2.;
    if hole.centers.is_empty() {
        return Err("Select at least one hole center".into());
    }
    let tip = hole
        .tip_angle
        .as_ref()
        .map(|s| angle(s).map(|a| radius / (a / 2.).tan()))
        .transpose()?
        .unwrap_or(0.);
    let (entry_radius, shoulder) = match &hole.kind {
        HoleKind::Simple => (radius, 0.),
        HoleKind::Counterbore { diameter, depth } => (positive(diameter)? / 2., positive(depth)?),
        HoleKind::Countersink { diameter, angle: a } => {
            let r = positive(diameter)? / 2.;
            (r, (r - radius) / (angle(a)? / 2.).tan())
        }
    };
    if entry_radius < radius
        || (!matches!(hole.kind, HoleKind::Simple) && entry_radius <= radius + 1e-8)
    {
        return Err("Counterbore/countersink diameter must exceed the hole diameter".into());
    }
    // Publish only after all holes succeed, so an invalid later center cannot leave a partial cut.
    let original = &st.bodies[bi].shape;
    let mut result = original.clone();
    for (index, center) in hole.centers.iter().enumerate() {
        let xy = [ev(&center[0], Unit::Length)?, ev(&center[1], Unit::Length)?];
        if !xy.iter().all(|v| v.is_finite()) {
            return Err("Hole coordinates must be finite".into());
        }
        let origin = frame.to_world(xy);
        let depth = match &hole.extent {
            HoleExtent::Distance { depth } => positive(depth)?,
            HoleExtent::ThroughAll => {
                let (lo, hi) = original.bbox().ok_or("Hole body has no bounds")?;
                let mut reach: f64 = 0.;
                for x in [lo[0], hi[0]] {
                    for y in [lo[1], hi[1]] {
                        for z in [lo[2], hi[2]] {
                            reach = reach.max(k::dot(k::sub([x, y, z], origin), axis));
                        }
                    }
                }
                if reach <= 1e-8 {
                    return Err("Hole direction does not reach the selected body".into());
                }
                reach + (k::sub(hi, lo).iter().map(|v| v * v).sum::<f64>().sqrt() * 0.01).max(1.)
            }
            HoleExtent::ToFace { face, offset } => {
                let (b, id) = st
                    .resolve_face(face)
                    .ok_or("hole target face no longer exists")?;
                let target = st.bodies[b]
                    .shape
                    .face_info(id)
                    .and_then(|f| f.plane)
                    .ok_or("Hole target must be planar")?;
                let n = target.normal();
                let dot = k::dot(axis, n);
                if dot.abs() < 1. - 1e-6 {
                    return Err("Hole target must be perpendicular to the drill axis".into());
                }
                let d = k::dot(k::sub(target.origin, origin), n) / dot + ev(offset, Unit::Length)?;
                if !d.is_finite() || d <= 1e-8 {
                    return Err("Hole target must be ahead of its start plane".into());
                }
                d
            }
        };
        if shoulder >= depth {
            return Err("Counterbore/countersink depth must be less than hole depth".into());
        }
        // Depth locates the end of the cylindrical bore; an angled point extends beyond it.
        let mut points = vec![[0., 0.], [entry_radius, 0.]];
        match hole.kind {
            HoleKind::Counterbore { .. } => {
                points.push([entry_radius, shoulder]);
                points.push([radius, shoulder]);
            }
            HoleKind::Countersink { .. } => points.push([radius, shoulder]),
            HoleKind::Simple => {}
        }
        points.push([radius, depth]);
        points.push([0., depth + tip]);
        let segments: Vec<_> = (0..points.len())
            .map(|i| Seg2::Line {
                a: points[i],
                b: points[(i + 1) % points.len()],
            })
            .collect();
        let radial = Frame {
            origin,
            x: frame.x,
            y: axis,
        };
        let tool_tag = k::mix(tag, index as u64);
        let profile = Shape::face_from_loops(&radial, &[segments], &[]).map_err(|e| e.0)?;
        let tool = profile
            .revolve(origin, axis, std::f64::consts::TAU, tool_tag)
            .map_err(|e| e.0)?;
        let before = result.volume();
        let cut = result.boolean(&tool, BoolOp::Cut).map_err(|e| e.0)?;
        if !cut.is_valid() || !cut.is_solid() || cut.solids().len() != 1 || cut.volume() <= 1e-9 {
            return Err("Hole must leave one valid solid body".into());
        }
        if before - cut.volume() <= 1e-9 {
            return Err("Hole does not remove material from the selected body".into());
        }
        result = cut;
    }
    st.bodies[bi].shape = result;
    Ok(())
}

fn revolve_axis(axis: &Axis, ss: Option<&SketchState>) -> Result<([f64; 3], [f64; 3]), String> {
    Ok(match *axis {
        Axis::Edge(_) => return Err("Resolve model axis against its state".into()),
        Axis::X => ([0.0; 3], [1.0, 0.0, 0.0]),
        Axis::Y => ([0.0; 3], [0.0, 1.0, 0.0]),
        Axis::Z => ([0.0; 3], [0.0, 0.0, 1.0]),
        Axis::SketchLine(e) => {
            let ss = ss.ok_or("Select a sketch profile before a sketch axis")?;
            let (a, b) = *ss
                .lines
                .get(&e)
                .ok_or("the revolve axis line no longer exists")?;
            let (pa, pb) = (ss.frame.to_world(a), ss.frame.to_world(b));
            let d = k::sub(pb, pa);
            if k::dot(d, d) < 1e-18 {
                return Err("revolve axis has zero length".into());
            }
            (pa, d)
        }
    })
}

fn blend(
    st: &mut State,
    edges: &[EdgeRef],
    op: impl Fn(&Shape, &[u32]) -> k::Result<Shape>,
) -> Result<(), String> {
    if edges.is_empty() {
        return Err("no edges selected".into());
    }
    let mut by_body: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut missing = 0;
    for r in edges {
        match st.resolve_edge(r) {
            Some((bi, e)) => by_body.entry(bi).or_default().push(e),
            None => missing += 1,
        }
    }
    if by_body.is_empty() {
        return Err("the selected edges no longer exist".into());
    }
    for (bi, mut ids) in by_body {
        ids.sort_unstable();
        ids.dedup();
        let s = op(&st.bodies[bi].shape, &ids).map_err(|e| e.0)?;
        st.bodies[bi].shape = s;
    }
    if missing > 0 {
        return Err(format!(
            "{missing} edge(s) could not be found and were skipped"
        ));
    }
    Ok(())
}

/// Find the profile a reference means in the current sketch: identical
/// bounding curves first, then the one containing the hint point, then the
/// best curve overlap.
pub fn match_profile<'a>(profiles: &'a [Profile], r: &ProfileRef) -> Option<&'a Profile> {
    let mut want = r.curves.clone();
    want.sort();
    // Split regions can share the same source curves (for example, the inside
    // and outside of a circle crossing rectangle edges). Use the interior hint
    // to distinguish those regions rather than taking the first signature match.
    let exact: Vec<_> = profiles.iter().filter(|p| p.signature() == want).collect();
    if exact.len() == 1 {
        return exact.first().copied();
    }
    if !exact.is_empty() {
        return exact
            .iter()
            .find(|p| p.contains(r.hint))
            .copied()
            .or_else(|| {
                exact.into_iter().min_by(|a, b| {
                    let distance = |p: &Profile| {
                        let q = profile_ref(p).hint;
                        (q[0] - r.hint[0]).powi(2) + (q[1] - r.hint[1]).powi(2)
                    };
                    distance(a).total_cmp(&distance(b))
                })
            });
    }
    let containing: Vec<&Profile> = profiles.iter().filter(|p| p.contains(r.hint)).collect();
    if let Some(p) = containing
        .iter()
        .min_by(|a, b| a.area().abs().partial_cmp(&b.area().abs()).unwrap())
    {
        return Some(p);
    }
    profiles
        .iter()
        .map(|p| (p, p.signature().iter().filter(|c| want.contains(c)).count()))
        .filter(|(_, n)| *n * 2 >= want.len().max(1))
        .max_by_key(|(_, n)| *n)
        .map(|(p, _)| p)
}

pub fn profile_ref(p: &Profile) -> ProfileRef {
    // Interior point: centroid of the outer polygon nudged inside if needed.
    let poly = p.outer.polygon(1e-3);
    let mut hint = [0.0, 0.0];
    for q in &poly {
        hint[0] += q[0] / poly.len() as f64;
        hint[1] += q[1] / poly.len() as f64;
    }
    if !p.contains(hint) {
        // Sample along a scanline through the bounding box.
        let (lo, hi) = poly
            .iter()
            .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), q| {
                (
                    [lo[0].min(q[0]), lo[1].min(q[1])],
                    [hi[0].max(q[0]), hi[1].max(q[1])],
                )
            });
        'outer: for j in 1..20 {
            for i in 1..40 {
                let c = [
                    lo[0] + (hi[0] - lo[0]) * i as f64 / 40.0,
                    lo[1] + (hi[1] - lo[1]) * j as f64 / 20.0,
                ];
                if p.contains(c) {
                    hint = c;
                    break 'outer;
                }
            }
        }
    }
    ProfileRef {
        curves: p.signature(),
        hint,
    }
}

fn seg2(s: &Seg) -> Seg2 {
    match *s {
        Seg::Line { a, b } => Seg2::Line { a, b },
        Seg::Arc { c, a, b, ccw, .. } => Seg2::Arc { c, a, b, ccw },
    }
}

fn profile_faces(
    ss: &SketchState,
    refs: &[ProfileRef],
    sketch: FeatureId,
) -> Result<Vec<Shape>, String> {
    if refs.is_empty() {
        return Err("no profile selected".into());
    }
    let mut out = Vec::new();
    for r in refs {
        let p = match_profile(&ss.profiles, r).ok_or("a selected profile no longer exists")?;
        let loops: Vec<&refusion_sketch::Loop> =
            std::iter::once(&p.outer).chain(p.holes.iter()).collect();
        let mut segs = Vec::new();
        let mut tags = Vec::new();
        for lp in &loops {
            for (s, &ent) in lp.segs.iter().zip(&lp.curves) {
                // rank of this segment among the segments of the same curve
                let key = |s: &Seg| {
                    let (a, b) = (s.start(), s.end());
                    let m = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                    m[0].atan2(m[1]) * 1e3 + m[0] + m[1] * 1e-3
                };
                let rank = loops
                    .iter()
                    .flat_map(|l| l.segs.iter().zip(&l.curves))
                    .filter(|(o, &e)| e == ent && key(o) < key(s) - 1e-12)
                    .count();
                tags.push(k::mix(k::mix(sketch as u64, ent.0 as u64), rank as u64));
            }
            segs.push(lp.segs.iter().map(seg2).collect::<Vec<_>>());
        }
        out.push(Shape::face_from_loops(&ss.frame, &segs, &tags).map_err(|e| e.0)?);
    }
    Ok(out)
}

// ------------------------------------------------------------------ projection

fn make_projection(
    st: &State,
    frame: &Frame,
    r: &EdgeRef,
    sk: &mut Sketch,
) -> Result<Vec<EntityId>, String> {
    let (bi, e) = st.resolve_edge(r).ok_or("edge not found")?;
    let shape = &st.bodies[bi].shape;
    let info = shape.edge_info(e).ok_or("degenerate edge")?;
    let n = frame.normal();
    let mut ents = Vec::new();
    let fix = |sk: &mut Sketch, ids: &[EntityId]| {
        for id in ids {
            sk.entities.get_mut(id).unwrap().projected = true;
        }
    };
    match info.kind {
        EdgeKind::Circle {
            center,
            axis,
            radius,
        } if k::dot(axis, n).abs() > 1.0 - 1e-9 => {
            let c = sk.add_point(frame.to_local(center));
            if info.closed {
                let circle = sk.add_circle(c, radius);
                ents = vec![c, circle];
            } else {
                let (mut s, mut t) = (frame.to_local(info.start), frame.to_local(info.end));
                if k::dot(axis, n) < 0.0 {
                    std::mem::swap(&mut s, &mut t);
                }
                let (ps, pe) = (sk.add_point(s), sk.add_point(t));
                let arc = sk.add_arc(c, ps, pe);
                ents = vec![c, ps, pe, arc];
            }
        }
        EdgeKind::Line => {
            let (a, b) = (frame.to_local(info.start), frame.to_local(info.end));
            if (a[0] - b[0]).hypot(a[1] - b[1]) < 1e-9 {
                ents.push(sk.add_point(a)); // edge seen end-on projects to a point
            } else {
                let (pa, pb) = (sk.add_point(a), sk.add_point(b));
                let l = sk.add_line(pa, pb);
                ents = vec![pa, pb, l];
            }
        }
        _ => {
            // General curves: fixed polyline approximation.
            let pts = shape.edge_sample(e, 0.05);
            let ids: Vec<EntityId> = pts
                .iter()
                .map(|p| sk.add_point(frame.to_local(*p)))
                .collect();
            ents.extend(&ids);
            for w in ids.windows(2) {
                let l = sk.add_line(w[0], w[1]);
                ents.push(l);
            }
        }
    }
    fix(sk, &ents);
    Ok(ents)
}

fn update_projections(st: &State, frame: &Frame, sf: &mut SketchFeature) {
    for pj in &sf.projections {
        let Some((bi, e)) = st.resolve_edge(&pj.edge) else {
            continue;
        };
        let shape = &st.bodies[bi].shape;
        let Some(info) = shape.edge_info(e) else {
            continue;
        };
        let sk = &mut sf.sketch;
        let alive = pj.entities.iter().all(|id| sk.entities.contains_key(id));
        if !alive {
            continue;
        }
        match (info.kind, pj.entities.len()) {
            (EdgeKind::Line, 3) => {
                sk.set_pos(pj.entities[0], frame.to_local(info.start));
                sk.set_pos(pj.entities[1], frame.to_local(info.end));
            }
            (EdgeKind::Circle { center, radius, .. }, 2) => {
                sk.set_pos(pj.entities[0], frame.to_local(center));
                sk.set_radius(pj.entities[1], radius);
            }
            (EdgeKind::Circle { center, axis, .. }, 4) => {
                let (mut s, mut t) = (frame.to_local(info.start), frame.to_local(info.end));
                if k::dot(axis, frame.normal()) < 0.0 {
                    std::mem::swap(&mut s, &mut t);
                }
                sk.set_pos(pj.entities[0], frame.to_local(center));
                sk.set_pos(pj.entities[1], s);
                sk.set_pos(pj.entities[2], t);
            }
            _ => {
                let pts = shape.edge_sample(e, 0.05);
                let point_ids: Vec<EntityId> = pj
                    .entities
                    .iter()
                    .copied()
                    .filter(|&id| sk.is_point(id))
                    .collect();
                if pts.len() == point_ids.len() {
                    for (id, p) in point_ids.iter().zip(&pts) {
                        sk.set_pos(*id, frame.to_local(*p));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    #[test]
    fn timeline_states_share_prior_sketch_payloads() {
        let mut doc = Document::new();
        let mut sketch = Sketch::new();
        sketch.add_rectangle([0., 0.], [10., 10.]);
        let id = doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch,
            projections: vec![],
        }));
        doc.add_import("cube", &Shape::boxed(1., 1., 1.).unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(
            &doc.cache[0].sketches[&id],
            &doc.cache[1].sketches[&id]
        ));
    }
    #[test]
    fn snapshots_share_immutable_states_and_worker_detachment_isolated() {
        let mut doc = Document::new();
        doc.add_import("box", &Shape::boxed(10.0, 10.0, 10.0).unwrap())
            .unwrap();
        let mut copy = doc.clone();
        assert!(Arc::ptr_eq(&doc.cache[0], &copy.cache[0]));
        let original = doc.state().bodies[0].shape.uid();
        copy.detach_display_shapes(&std::collections::HashSet::new())
            .unwrap();
        assert!(!Arc::ptr_eq(&doc.cache[0], &copy.cache[0]));
        assert_eq!(doc.state().bodies[0].shape.uid(), original);
        assert_ne!(copy.state().bodies[0].shape.uid(), original);
        assert!((copy.state().bodies[0].shape.volume() - 1000.0).abs() < 1e-6);
    }
}
