//! Interactive sketching: tools, snapping, inference, dimensions, glyphs.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use refusion_model::kernel::Frame;
use refusion_model::sketch::{AddError, Constraint, ConstraintId, DimKind, EntityId, Geom, Sketch};
use refusion_model::{FeatureId, Unit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConKind {
    Coincident,
    Collinear,
    Concentric,
    Midpoint,
    Fix,
    Parallel,
    Perpendicular,
    HorizontalVertical,
    Tangent,
    Equal,
    Symmetric,
}

impl ConKind {
    pub const ALL: [ConKind; 11] = [
        ConKind::HorizontalVertical,
        ConKind::Coincident,
        ConKind::Tangent,
        ConKind::Equal,
        ConKind::Parallel,
        ConKind::Perpendicular,
        ConKind::Fix,
        ConKind::Midpoint,
        ConKind::Concentric,
        ConKind::Collinear,
        ConKind::Symmetric,
    ];
    pub fn label(&self) -> &'static str {
        match self {
            ConKind::Coincident => "Coincident",
            ConKind::Collinear => "Collinear",
            ConKind::Concentric => "Concentric",
            ConKind::Midpoint => "Midpoint",
            ConKind::Fix => "Fix/Unfix",
            ConKind::Parallel => "Parallel",
            ConKind::Perpendicular => "Perpendicular",
            ConKind::HorizontalVertical => "Horizontal/Vertical",
            ConKind::Tangent => "Tangent",
            ConKind::Equal => "Equal",
            ConKind::Symmetric => "Symmetry",
        }
    }
    fn needs(&self) -> usize {
        match self {
            ConKind::Fix => 1,
            ConKind::Symmetric => 3,
            _ => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Select,
    Line,
    Rectangle,
    CenterRectangle,
    Circle,
    Arc,
    ThreePointArc,
    TangentArc,
    Trim,
    Extend,
    Offset,
    Point,
    Dimension,
    Constrain(ConKind),
    Project,
}

impl Tool {
    pub fn hint(&self) -> &'static str {
        match self {
            Tool::Select => "Click to select, drag geometry to move it (the solver keeps constraints). Del deletes.",
            Tool::Line => "Click points to draw a line chain. Hold Shift to snap to line midpoints. Near-horizontal/vertical lines get H/V constraints. Esc ends the chain.",
            Tool::Rectangle => "Click two opposite corners.",
            Tool::CenterRectangle => "Click the centre, then a corner. Typed width and height are the full size.",
            Tool::Circle => "Click the centre, then a point on the circle. Hold Shift near a line midpoint to snap there.",
            Tool::Arc => "Click the centre, the start point, then sweep to the end point.",
            Tool::ThreePointArc => "Click the start, end, then a point on the arc. Endpoints snap to existing points.",
            Tool::TangentArc => "Click an existing line or arc endpoint, then the new endpoint. Type a radius to lock it.",
            Tool::Trim => "Click the segment to remove between intersections. Constraints on the trimmed curve are removed.",
            Tool::Extend => "Click near the endpoint to extend to the nearest intersecting curve. Constraints on the extended curve are removed.",
            Tool::Offset => "Select a curve, then click the offset side and distance. Type a distance to lock it.",
            Tool::Point => "Click to place a point. Hold Shift near a line midpoint to constrain it there.",
            Tool::Dimension => "Pick a line, circle/arc, two points, a point and a line, or two lines; click to place; type a value or expression. Hold Shift to pick line midpoints.",
            Tool::Constrain(_) => "Pick the geometry to constrain. Hold Shift near a line midpoint to pick it.",
            Tool::Project => "Select geometry to project, then OK or Enter. Escape cancels the selection.",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkSel {
    Ent(EntityId),
    Con(ConstraintId),
}

pub struct DimEdit {
    pub existing: Option<ConstraintId>,
    pub pending: Option<Constraint>,
    pub label: [f64; 2],
    pub text: String,
    pub at: Pos2,
    pub focus: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HudKind {
    Length,
    Angle,
    Width,
    Height,
    Diameter,
    Radius,
    Sweep,
}

impl HudKind {
    pub fn label(&self) -> &'static str {
        match self {
            HudKind::Length => "L",
            HudKind::Angle => "∠",
            HudKind::Width => "W",
            HudKind::Height => "H",
            HudKind::Diameter => "⌀",
            HudKind::Radius => "R",
            HudKind::Sweep => "∠",
        }
    }
    pub fn unit(&self) -> Unit {
        match self {
            HudKind::Angle | HudKind::Sweep => Unit::Angle,
            _ => Unit::Length,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HudField {
    pub kind: HudKind,
    pub text: String,
    /// Value in mm / radians once the user typed one.
    pub locked: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Hud {
    key: (u8, u8),
    pub fields: Vec<HudField>,
    pub active: usize,
    /// Ask the UI to focus the active field and select its text.
    pub refocus: bool,
}

impl Hud {
    fn new(key: (u8, u8), kinds: &[HudKind]) -> Self {
        Hud {
            key,
            fields: kinds
                .iter()
                .map(|&kind| HudField {
                    kind,
                    text: String::new(),
                    locked: None,
                })
                .collect(),
            active: 0,
            refocus: true,
        }
    }
    pub fn lock(&self, k: HudKind) -> Option<f64> {
        self.fields
            .iter()
            .find(|f| f.kind == k)
            .and_then(|f| f.locked)
    }
    fn live(&mut self, k: HudKind, v: f64) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.kind == k) {
            if f.locked.is_none() {
                f.text = if k.unit() == Unit::Angle {
                    fmt_len(v.to_degrees())
                } else {
                    fmt_len(v)
                };
            }
        }
    }
    /// Tab / Shift+Tab.
    pub fn cycle(&mut self, back: bool) {
        let n = self.fields.len();
        if n > 0 {
            self.active = if back {
                (self.active + n - 1) % n
            } else {
                (self.active + 1) % n
            };
            self.refocus = true;
        }
    }
    /// User edited field `i`: lock it to the evaluated value (empty = unlock).
    pub fn edit(&mut self, i: usize, eval: &dyn Fn(&str, Unit) -> Result<f64, String>) {
        let f = &mut self.fields[i];
        if f.text.trim().is_empty() {
            f.locked = None;
        } else if let Ok(v) = eval(&f.text, f.kind.unit()) {
            f.locked = Some(if f.kind.unit() == Unit::Angle {
                v
            } else {
                v.abs()
            });
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CurveSnap {
    On(EntityId),
    Midpoint(EntityId),
}

pub struct SketchSession {
    pub feature: FeatureId,
    pub frame: Frame,
    pub tool: Tool,
    pub restore_marker: usize,
    chain: Option<EntityId>,
    first: Option<([f64; 2], Option<EntityId>, Option<CurveSnap>)>, // pos, snapped point, snapped curve
    arc_start: Option<[f64; 2]>,
    arc_end_snap: Option<([f64; 2], Option<EntityId>, Option<CurveSnap>)>,
    arc_sweep: f64,
    arc_last: f64,
    pub picked: Vec<EntityId>,
    pending_dim: Option<Constraint>,
    pub selection: Vec<SkSel>,
    drag: Option<(EntityId, [f64; 2])>,
    pub dim_edit: Option<DimEdit>,
    pub hover_con: Option<ConstraintId>,
    pub inference: Option<&'static str>,
    /// Grid snapping spacing when "Snap to grid" is on (set by the app each frame).
    pub grid_snap: Option<f64>,
    /// Sketch Palette visibility of dimensions / constraint glyphs.
    pub show_dims: bool,
    pub show_constraints: bool,
    pub project_edges: Vec<refusion_model::EdgeRef>,
    pub project_bodies: bool,
    pub projection_link: bool,
    pub construction: bool,
    pub show_construction: bool,
    pub show_projected: bool,
    /// Dimension whose text is being dragged.
    drag_dim: Option<ConstraintId>,
    /// Set when a dimension drag starts (the app takes an undo snapshot).
    pub dim_moved: bool,
    pub hud: Option<Hud>,
    last_cursor: Option<[f64; 2]>,
    last_hover: Option<EntityId>,
    /// Radius typed while placing an arc's start point (dimensioned on commit).
    arc_radius_lock: Option<f64>,
    tangent_source: Option<(EntityId, EntityId)>,
    pub midpoint_target: Option<EntityId>,
    midpoint_refs: Vec<(EntityId, EntityId)>,
    reference_preview: Option<Sketch>,
}

/// Result of an interaction.
#[derive(Default)]
pub struct Outcome {
    pub changed: bool,
    pub message: Option<String>,
}

impl Outcome {
    fn changed() -> Self {
        Outcome {
            changed: true,
            message: None,
        }
    }
    fn msg(m: impl Into<String>) -> Self {
        Outcome {
            changed: false,
            message: Some(m.into()),
        }
    }
}

pub fn fmt_len(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn tangent_source(
    sk: &Sketch,
    hover: Option<EntityId>,
    snapped: Option<EntityId>,
) -> Option<(EntityId, EntityId)> {
    let point = snapped?;
    let endpoint = |id: EntityId| match sk.entity(id).map(|e| e.geom) {
        Some(Geom::Line { a, b }) => point == a || point == b,
        Some(Geom::Arc { s, e, .. }) => point == s || point == e,
        _ => false,
    };
    let curve = hover
        .filter(|h| endpoint(*h))
        .or_else(|| sk.users_of(point).into_iter().find(|id| endpoint(*id)))?;
    Some((point, curve))
}
fn tangent_geometry(
    sk: &Sketch,
    start: EntityId,
    source: EntityId,
    mut end: [f64; 2],
    radius: Option<f64>,
) -> Option<([f64; 2], [f64; 2], bool)> {
    let p = sk.pos(start);
    let t = match sk.entity(source)?.geom {
        Geom::Line { a, b } => {
            let other = sk.pos(if start == a {
                b
            } else if start == b {
                a
            } else {
                return None;
            });
            [p[0] - other[0], p[1] - other[1]]
        }
        Geom::Arc { c, s, e } => {
            let c = sk.pos(c);
            let sign = if start == e {
                1.
            } else if start == s {
                -1.
            } else {
                return None;
            };
            [-(p[1] - c[1]) * sign, (p[0] - c[0]) * sign]
        }
        _ => return None,
    };
    let len = t[0].hypot(t[1]);
    if len < 1e-9 {
        return None;
    }
    let n = [-t[1] / len, t[0] / len];
    let d = [end[0] - p[0], end[1] - p[1]];
    let side = d[0] * n[0] + d[1] * n[1];
    if side.abs() < 1e-9 || dist(p, end) < 1e-9 {
        return None;
    }
    let h = radius
        .map(|r| r.copysign(side))
        .unwrap_or((d[0] * d[0] + d[1] * d[1]) / (2. * side));
    if !h.is_finite() || h.abs() < 1e-9 || h.abs() > 1e8 {
        return None;
    }
    let c = [p[0] + n[0] * h, p[1] + n[1] * h];
    if radius.is_some() {
        let l = dist(c, end);
        if l < 1e-9 {
            return None;
        }
        end = [
            c[0] + (end[0] - c[0]) * h.abs() / l,
            c[1] + (end[1] - c[1]) * h.abs() / l,
        ];
    }
    if dist(p, end) < 1e-7 {
        return None;
    }
    Some((c, end, h > 0.))
}

/// Midpoint of a nondegenerate line in sketch coordinates.
fn line_midpoint(sk: &Sketch, e: EntityId) -> Option<[f64; 2]> {
    let (a, b) = sk.line_pts(e)?;
    let (a, b) = (sk.pos(a), sk.pos(b));
    (dist(a, b) > 1e-9).then_some([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5])
}

/// Closest point on an entity to `p` (sketch coordinates).
pub fn closest_on(sk: &Sketch, e: EntityId, p: [f64; 2]) -> [f64; 2] {
    match sk.entity(e).map(|x| x.geom) {
        Some(Geom::Point(_)) => sk.pos(e),
        Some(Geom::Line { a, b }) => {
            let (a, b) = (sk.pos(a), sk.pos(b));
            let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = ux * ux + uy * uy;
            let t = if l2 > 0.0 {
                (((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            [a[0] + t * ux, a[1] + t * uy]
        }
        Some(Geom::Circle { c, .. }) | Some(Geom::Arc { c, .. }) => {
            let c0 = sk.pos(c);
            let r = sk.radius(e).unwrap();
            let d = dist(p, c0).max(1e-12);
            [
                c0[0] + (p[0] - c0[0]) * r / d,
                c0[1] + (p[1] - c0[1]) * r / d,
            ]
        }
        None => p,
    }
}

impl SketchSession {
    pub fn new(feature: FeatureId, frame: Frame, restore_marker: usize) -> Self {
        SketchSession {
            feature,
            frame,
            tool: Tool::Line,
            restore_marker,
            chain: None,
            first: None,
            arc_start: None,
            arc_end_snap: None,
            arc_sweep: 0.0,
            arc_last: 0.0,
            picked: Vec::new(),
            pending_dim: None,
            selection: Vec::new(),
            drag: None,
            dim_edit: None,
            hover_con: None,
            inference: None,
            grid_snap: None,
            show_dims: true,
            show_constraints: true,
            project_edges: vec![],
            project_bodies: false,
            projection_link: true,
            construction: false,
            show_construction: true,
            show_projected: true,
            drag_dim: None,
            dim_moved: false,
            hud: None,
            last_cursor: None,
            last_hover: None,
            arc_radius_lock: None,
            tangent_source: None,
            midpoint_target: None,
            midpoint_refs: Vec::new(),
            reference_preview: None,
        }
    }

    pub fn set_tool(&mut self, t: Tool) {
        self.tool = t;
        self.cancel();
    }

    /// Esc: abandon the current tool operation.
    pub fn cancel(&mut self) -> bool {
        let busy = self.chain.is_some()
            || self.first.is_some()
            || self.arc_start.is_some()
            || !self.project_edges.is_empty()
            || !self.picked.is_empty()
            || self.pending_dim.is_some()
            || self.dim_edit.is_some();
        self.chain = None;
        self.first = None;
        self.arc_start = None;
        self.arc_end_snap = None;
        self.arc_sweep = 0.0;
        self.project_edges.clear();
        self.picked.clear();
        self.pending_dim = None;
        self.dim_edit = None;
        self.inference = None;
        self.hud = None;
        self.arc_radius_lock = None;
        self.tangent_source = None;
        self.midpoint_target = None;
        self.midpoint_refs.clear();
        self.reference_preview = None;
        busy
    }

    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Shift exposes midpoint references within a constant screen-space radius.
    /// Select retains Shift multi-selection; navigation and editing tools are unaffected.
    pub fn update_midpoint_snap(
        &mut self,
        sk: &Sketch,
        cursor: [f64; 2],
        shift: bool,
        to_screen: impl Fn([f64; 2]) -> Option<Pos2>,
    ) {
        self.midpoint_target = None;
        if !shift
            || matches!(
                self.tool,
                Tool::Select | Tool::Trim | Tool::Extend | Tool::Offset | Tool::Project
            )
            || (self.tool == Tool::TangentArc && self.tangent_source.is_none())
            || (self.tool == Tool::Arc && self.first.is_some())
        {
            return;
        }
        let Some(cursor) = to_screen(cursor) else {
            return;
        };
        self.midpoint_target = sk
            .entities
            .keys()
            .filter_map(|&e| {
                let entity = sk.entity(e)?;
                if entity.construction && !self.show_construction
                    || entity.projected && !self.show_projected
                {
                    return None;
                }
                let p = to_screen(line_midpoint(sk, e)?)?;
                let d = p.distance_sq(cursor);
                (d <= 7.5 * 7.5).then_some((e, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(e, _)| e);
    }

    /// References stay virtual until the constraint/dimension is committed.
    /// Escape and failed operations therefore leave no orphan sketch points.
    fn reference_sketch(&self, sk: &Sketch) -> Result<Sketch, String> {
        let mut work = sk.clone();
        for &(point, curve) in &self.midpoint_refs {
            let pos = line_midpoint(&work, curve)
                .ok_or("The midpoint reference is no longer available")?;
            if work.add_point(pos) != point {
                return Err(
                    "The sketch changed while picking midpoint references; restart the tool".into(),
                );
            }
            work.add_constraint(Constraint::Midpoint(point, curve))
                .map_err(|e| e.to_string())?;
        }
        Ok(work)
    }

    fn reference_point(&mut self, sk: &Sketch, curve: EntityId) -> Result<EntityId, String> {
        if let Some(&(point, _)) = self.midpoint_refs.iter().find(|(_, e)| *e == curve) {
            return Ok(point);
        }
        let mut work = self.reference_sketch(sk)?;
        let pos = line_midpoint(&work, curve).ok_or("Select a line midpoint")?;
        let point = work.add_point(pos);
        work.add_constraint(Constraint::Midpoint(point, curve))
            .map_err(|e| e.to_string())?;
        self.midpoint_refs.push((point, curve));
        self.reference_preview = Some(work);
        Ok(point)
    }

    /// Snap the cursor: to a hovered point, onto a hovered curve, else raw.
    fn snap(
        &self,
        sk: &Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
    ) -> ([f64; 2], Option<EntityId>, Option<CurveSnap>) {
        if let Some(e) = self.midpoint_target {
            if let Some(p) = line_midpoint(sk, e) {
                return (p, None, Some(CurveSnap::Midpoint(e)));
            }
        }
        match hover {
            Some(e) if sk.is_point(e) => (sk.pos(e), Some(e), None),
            Some(e) if sk.entity(e).is_some() => {
                (closest_on(sk, e, cursor), None, Some(CurveSnap::On(e)))
            }
            _ => (self.to_grid(cursor), None, None),
        }
    }

    /// Round a sketch position to the snap grid (identity when snapping is off).
    pub fn to_grid(&self, p: [f64; 2]) -> [f64; 2] {
        match self.grid_snap {
            Some(g) if g > 0.0 => [(p[0] / g).round() * g, (p[1] / g).round() * g],
            _ => p,
        }
    }

    /// Point entity for a snapped location (reusing an existing point).
    fn point_at(
        sk: &mut Sketch,
        snapped: ([f64; 2], Option<EntityId>, Option<CurveSnap>),
        after: &mut Vec<Constraint>,
    ) -> EntityId {
        match snapped {
            (_, Some(p), _) => p,
            (pos, None, Some(curve)) => {
                let p = sk.add_point(pos);
                after.push(match curve {
                    CurveSnap::On(curve) => Constraint::Coincident(p, curve),
                    CurveSnap::Midpoint(curve) => Constraint::Midpoint(p, curve),
                });
                p
            }
            (pos, None, None) => sk.add_point(pos),
        }
    }

    fn apply_all(sk: &mut Sketch, cs: Vec<Constraint>) {
        for c in cs {
            let _ = sk.add_constraint(c);
        }
    }

    /// Horizontal / vertical inference for a line from `a` to `b`.
    fn infer(a: [f64; 2], b: [f64; 2]) -> ([f64; 2], Option<&'static str>) {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let tol = 3.5f64.to_radians().tan();
        if dx.abs() > 1e-9 && (dy / dx).abs() < tol {
            ([b[0], a[1]], Some("H"))
        } else if dy.abs() > 1e-9 && (dx / dy).abs() < tol {
            ([a[0], b[1]], Some("V"))
        } else {
            (b, None)
        }
    }

    /// Which HUD fields apply to the tool's current stage.
    fn hud_key(&self) -> Option<((u8, u8), &'static [HudKind])> {
        match self.tool {
            Tool::Line if self.chain.is_some() => {
                Some(((1, 0), &[HudKind::Length, HudKind::Angle]))
            }
            Tool::Rectangle | Tool::CenterRectangle if self.first.is_some() => {
                Some(((2, 0), &[HudKind::Width, HudKind::Height]))
            }
            Tool::Circle if self.first.is_some() => Some(((3, 0), &[HudKind::Diameter])),
            Tool::TangentArc if self.first.is_some() => Some(((10, 0), &[HudKind::Radius])),
            Tool::Offset if !self.picked.is_empty() => Some(((9, 0), &[HudKind::Length])),
            Tool::Arc if self.first.is_some() && self.arc_start.is_none() => {
                Some(((4, 0), &[HudKind::Radius]))
            }
            Tool::Arc if self.arc_start.is_some() => Some(((4, 1), &[HudKind::Sweep])),
            _ => None,
        }
    }

    fn sync_hud(&mut self) {
        match self.hud_key() {
            None => self.hud = None,
            Some((key, kinds)) => {
                if self.hud.as_ref().is_none_or(|h| h.key != key) {
                    self.hud = Some(Hud::new(key, kinds));
                }
            }
        }
    }

    fn lock(&self, k: HudKind) -> Option<f64> {
        self.hud.as_ref().and_then(|h| h.lock(k))
    }

    /// Line end point honouring typed length / angle; returns (point, H/V inference).
    fn line_end(
        &self,
        a: [f64; 2],
        raw: [f64; 2],
        snapped_to_point: bool,
    ) -> ([f64; 2], Option<&'static str>) {
        let (len_l, ang_l) = (self.lock(HudKind::Length), self.lock(HudKind::Angle));
        if len_l.is_none() && ang_l.is_none() {
            return if snapped_to_point {
                (raw, None)
            } else {
                Self::infer(a, raw)
            };
        }
        let (dx, dy) = (raw[0] - a[0], raw[1] - a[1]);
        let mut len = dx.hypot(dy);
        let mut ang = if len > 1e-12 { dy.atan2(dx) } else { 0.0 };
        if let Some(t) = ang_l {
            ang = t;
        }
        if let Some(l) = len_l {
            len = l;
        }
        let p = [a[0] + len * ang.cos(), a[1] + len * ang.sin()];
        if ang_l.is_none() {
            // keep H/V inference for a typed length
            let (q, inf) = Self::infer(a, p);
            if inf.is_some() {
                let l = len;
                let d = [q[0] - a[0], q[1] - a[1]];
                let n = d[0].hypot(d[1]).max(1e-12);
                return ([a[0] + d[0] / n * l, a[1] + d[1] / n * l], inf);
            }
        }
        (p, None)
    }

    fn rect_corner(&self, p: [f64; 2], raw: [f64; 2]) -> [f64; 2] {
        let factor = if self.tool == Tool::CenterRectangle {
            0.5
        } else {
            1.0
        };
        let sgn = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };
        let mut w = raw[0] - p[0];
        let mut h = raw[1] - p[1];
        if let Some(l) = self.lock(HudKind::Width) {
            w = sgn(w) * l * factor;
        }
        if let Some(l) = self.lock(HudKind::Height) {
            h = sgn(h) * l * factor;
        }
        [p[0] + w, p[1] + h]
    }

    fn circle_radius(&self, c: [f64; 2], raw: [f64; 2]) -> f64 {
        self.lock(HudKind::Diameter)
            .map(|d| d * 0.5)
            .unwrap_or_else(|| dist(c, raw))
    }

    fn arc_start_point(&self, c: [f64; 2], raw: [f64; 2]) -> [f64; 2] {
        match self.lock(HudKind::Radius) {
            Some(r) => {
                let d = dist(c, raw).max(1e-12);
                [
                    c[0] + (raw[0] - c[0]) / d * r,
                    c[1] + (raw[1] - c[1]) / d * r,
                ]
            }
            None => raw,
        }
    }

    /// Current arc sweep (signed), honouring a typed sweep angle.
    fn arc_sweep_now(&self) -> f64 {
        match self.lock(HudKind::Sweep) {
            Some(s) => {
                let s = s.abs().min(std::f64::consts::TAU);
                if self.arc_sweep < 0.0 {
                    -s
                } else {
                    s
                }
            }
            None => self.arc_sweep,
        }
    }

    /// Add a dimension for a typed value; falls back to a driven dimension
    /// if it would over-constrain.
    fn add_dim(sk: &mut Sketch, c: Constraint) {
        let id = match sk.add_constraint(c.clone()) {
            Ok(Some(id)) => id,
            Ok(None) => return,
            Err(_) => sk.add_driven(c.clone()),
        };
        // Radius / diameter: text outside the circle, along the anchor ray.
        if let Constraint::Diameter { e, .. } | Constraint::Radius { e, .. } = c {
            if let (Some(anchor), Some(cen), Some(r)) =
                (dim_anchor(sk, &c), sk.center(e), sk.radius(e))
            {
                let cp = sk.pos(cen);
                let u = [
                    (anchor[0] - cp[0]) / r.max(1e-12),
                    (anchor[1] - cp[1]) / r.max(1e-12),
                ];
                if let Some(rec) = sk.constraints.get_mut(&id) {
                    rec.label = [u[0] * r * 0.6, u[1] * r * 0.6];
                }
            }
        }
        // Place length labels off the geometry, on the side away from the shape.
        if let Constraint::Length { line, .. } = c {
            let (a, b) = sk.line_pts(line).unwrap();
            let (pa, pb) = (sk.pos(a), sk.pos(b));
            let len = dist(pa, pb);
            let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
            let mut n = [
                -(pb[1] - pa[1]) / len.max(1e-12),
                (pb[0] - pa[0]) / len.max(1e-12),
            ];
            let chain = chain_from(sk, line);
            let pts: Vec<[f64; 2]> = chain
                .iter()
                .flat_map(|&e| sk.defining_points(e))
                .map(|p| sk.pos(p))
                .collect();
            if !pts.is_empty() {
                let c = pts.iter().fold([0.0, 0.0], |acc, p| {
                    [
                        acc[0] + p[0] / pts.len() as f64,
                        acc[1] + p[1] / pts.len() as f64,
                    ]
                });
                if (c[0] - mid[0]) * n[0] + (c[1] - mid[1]) * n[1] > 0.0 {
                    n = [-n[0], -n[1]];
                }
            }
            let off = (len * 0.18).max(1.0);
            if let Some(r) = sk.constraints.get_mut(&id) {
                r.label = [n[0] * off, n[1] * off];
            }
        }
    }

    /// Enter in the HUD: place the geometry at the last cursor position.
    pub fn commit_hud(
        &mut self,
        sk: &mut Sketch,
        eval: &dyn Fn(&str, Unit) -> Result<f64, String>,
    ) -> Outcome {
        let Some(cur) = self.last_cursor else {
            return Outcome::default();
        };
        let hover = self.last_hover;
        self.click(sk, hover, cur, Pos2::ZERO, eval, false)
    }

    /// Is a drawing operation waiting for its next point?
    pub fn in_progress(&self) -> bool {
        self.chain.is_some() || self.first.is_some() || self.arc_start.is_some()
    }

    pub fn is_drawing_tool(&self) -> bool {
        matches!(
            self.tool,
            Tool::Line
                | Tool::Rectangle
                | Tool::CenterRectangle
                | Tool::Circle
                | Tool::Arc
                | Tool::ThreePointArc
                | Tool::TangentArc
                | Tool::Offset
                | Tool::Point
        )
    }

    pub fn click(
        &mut self,
        sk: &mut Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
        at: Pos2,
        eval: &dyn Fn(&str, Unit) -> Result<f64, String>,
        shift: bool,
    ) -> Outcome {
        let last = sk.entities.keys().next_back().copied();
        let drawing = self.is_drawing_tool();
        let out = self.click_inner(sk, hover, cursor, at, eval, shift);
        if out.changed && drawing && self.construction {
            for (&id, entity) in &mut sk.entities {
                if last.is_none_or(|last| id > last) {
                    entity.construction = true;
                }
            }
        }
        out
    }

    fn click_inner(
        &mut self,
        sk: &mut Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
        at: Pos2,
        eval: &dyn Fn(&str, Unit) -> Result<f64, String>,
        shift: bool,
    ) -> Outcome {
        let _ = eval;
        let hover = if matches!(self.tool, Tool::Trim | Tool::Extend | Tool::Offset) {
            hover.and_then(|id| {
                if sk.is_point(id) {
                    let users = sk.users_of(id);
                    (users.len() == 1).then(|| users[0])
                } else {
                    Some(id)
                }
            })
        } else {
            hover
        };
        if !shift {
            self.midpoint_target = None;
        }
        let snapped = self.snap(sk, hover, cursor);
        let hover = if matches!(self.tool, Tool::Constrain(_))
            || (self.tool == Tool::Dimension
                && self.pending_dim.is_none()
                && self.dim_edit.is_none())
        {
            if let Some(curve) = self.midpoint_target {
                match self.reference_point(sk, curve) {
                    Ok(point) => Some(point),
                    Err(e) => return Outcome::msg(e),
                }
            } else {
                hover
            }
        } else {
            hover
        };
        match self.tool {
            Tool::Select => {
                match hover {
                    Some(e) => {
                        let s = SkSel::Ent(e);
                        if shift {
                            if let Some(i) = self.selection.iter().position(|x| *x == s) {
                                self.selection.remove(i);
                            } else {
                                self.selection.push(s);
                            }
                        } else {
                            self.selection = vec![s];
                        }
                    }
                    None if !shift => self.selection.clear(),
                    None => {}
                }
                Outcome::default()
            }
            Tool::Trim | Tool::Extend => {
                let Some(id) = hover else {
                    return Outcome {
                        changed: false,
                        message: Some("Select a curve".into()),
                    };
                };
                let result = if self.tool == Tool::Trim {
                    sk.trim(id, cursor)
                } else {
                    sk.extend(id, cursor)
                };
                match result {
                    Ok(()) => Outcome::changed(),
                    Err(e) => Outcome {
                        changed: false,
                        message: Some(e),
                    },
                }
            }
            Tool::Offset => {
                if self.picked.is_empty() {
                    if let Some(id) = hover {
                        if sk
                            .entity(id)
                            .is_some_and(|e| !matches!(e.geom, Geom::Point(_)))
                        {
                            self.picked.push(id);
                        }
                    }
                    return Outcome::default();
                }
                let id = self.picked[0];
                let mut d = offset_distance(sk, id, cursor);
                if let Some(v) = self.lock(HudKind::Length) {
                    d = if d < 0.0 { -v.abs() } else { v.abs() };
                }
                match sk.offset(id, d) {
                    Ok(_) => {
                        self.cancel();
                        Outcome::changed()
                    }
                    Err(e) => Outcome {
                        changed: false,
                        message: Some(e),
                    },
                }
            }
            Tool::TangentArc => {
                if self.tangent_source.is_none() {
                    let Some((point, curve)) = tangent_source(sk, hover, snapped.1) else {
                        return Outcome::msg("Pick an existing line or arc endpoint");
                    };
                    self.first = Some((sk.pos(point), Some(point), None));
                    self.tangent_source = Some((point, curve));
                    return Outcome::default();
                }
                let (start, source) = self.tangent_source.unwrap();
                let Some((centre, end, ccw)) =
                    tangent_geometry(sk, start, source, snapped.0, self.lock(HudKind::Radius))
                else {
                    return Outcome::msg("Move away from the endpoint tangent to define an arc");
                };
                let before = sk.clone();
                let mut after = Vec::new();
                let end_snap = if self.lock(HudKind::Radius).is_some() {
                    (end, None, None)
                } else {
                    snapped
                };
                let end_id = Self::point_at(sk, end_snap, &mut after);
                let c = sk.add_point(centre);
                let arc = if ccw {
                    sk.add_arc(c, start, end_id)
                } else {
                    sk.add_arc(c, end_id, start)
                };
                Self::apply_all(sk, after);
                if let Err(e) = sk.add_constraint(Constraint::Tangent(source, arc)) {
                    *sk = before;
                    return Outcome::msg(e.to_string());
                }
                if let Some(radius) = self.lock(HudKind::Radius) {
                    match sk.add_constraint(Constraint::Radius {
                        e: arc,
                        value: radius,
                    }) {
                        Ok(Some(id)) => {
                            if let Some(anchor) = dim_anchor(
                                sk,
                                &Constraint::Radius {
                                    e: arc,
                                    value: radius,
                                },
                            ) {
                                let cp = sk.pos(c);
                                if let Some(rec) = sk.constraints.get_mut(&id) {
                                    rec.label =
                                        [(anchor[0] - cp[0]) * 0.6, (anchor[1] - cp[1]) * 0.6];
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            *sk = before;
                            return Outcome::msg(e.to_string());
                        }
                    }
                }
                self.cancel();
                Outcome::changed()
            }
            Tool::ThreePointArc => {
                if self.first.is_none() {
                    self.first = Some(snapped);
                    return Outcome::default();
                }
                if self.arc_start.is_none() {
                    self.arc_start = Some(snapped.0);
                    self.arc_end_snap = Some(snapped);
                    return Outcome::default();
                }
                let first = self.first.unwrap();
                let end = self.arc_start.unwrap();
                let (center, ccw) =
                    match refusion_model::sketch::edit::three_point_arc(first.0, end, snapped.0) {
                        Ok(v) => v,
                        Err(e) => {
                            return Outcome {
                                changed: false,
                                message: Some(e),
                            }
                        }
                    };
                let mut after = Vec::new();
                let start_id = Self::point_at(sk, first, &mut after);
                let end_snap = self.arc_end_snap.unwrap_or((end, None, None));
                let end_id = Self::point_at(sk, end_snap, &mut after);
                let c = sk.add_point(center);
                if ccw {
                    sk.add_arc(c, start_id, end_id);
                } else {
                    sk.add_arc(c, end_id, start_id);
                }
                Self::apply_all(sk, after);
                self.cancel();
                Outcome::changed()
            }
            Tool::Point => {
                let mut after = Vec::new();
                Self::point_at(sk, snapped, &mut after);
                Self::apply_all(sk, after);
                Outcome::changed()
            }
            Tool::Line => {
                let mut after = Vec::new();
                match self.chain {
                    None => {
                        let p = Self::point_at(sk, snapped, &mut after);
                        Self::apply_all(sk, after);
                        self.chain = Some(p);
                        Outcome::changed()
                    }
                    Some(start) => {
                        let a = sk.pos(start);
                        let typed = self.lock(HudKind::Length).is_some()
                            || self.lock(HudKind::Angle).is_some();
                        let (pos, inf) = self.line_end(
                            a,
                            snapped.0,
                            snapped.1.is_some()
                                || matches!(snapped.2, Some(CurveSnap::Midpoint(_))),
                        );
                        if dist(a, pos) < 1e-9 {
                            return Outcome::default();
                        }
                        // A typed value wins over snapping to an existing point.
                        let snap = if typed {
                            (pos, None, None)
                        } else {
                            (pos, snapped.1, snapped.2)
                        };
                        let closes = snap.1.is_some();
                        let end = Self::point_at(sk, snap, &mut after);
                        let line = sk.add_line(start, end);
                        let ang = self
                            .lock(HudKind::Angle)
                            .map(|t| t.rem_euclid(std::f64::consts::PI));
                        let axis = |t: f64, target: f64| (t - target).abs() < 1e-9;
                        let inf = match ang {
                            Some(t) if axis(t, 0.0) || axis(t, std::f64::consts::PI) => Some("H"),
                            Some(t) if axis(t, std::f64::consts::FRAC_PI_2) => Some("V"),
                            Some(_) => None,
                            None => inf,
                        };
                        match inf {
                            Some("H") => after.push(Constraint::Horizontal(line)),
                            Some("V") => after.push(Constraint::Vertical(line)),
                            _ => {}
                        }
                        let len_lock = self.lock(HudKind::Length);
                        Self::apply_all(sk, after);
                        if let Some(l) = len_lock {
                            Self::add_dim(sk, Constraint::Length { line, value: l });
                        }
                        self.chain = if closes { None } else { Some(end) };
                        self.inference = None;
                        self.hud = None;
                        Outcome::changed()
                    }
                }
            }
            Tool::Rectangle | Tool::CenterRectangle => match self.first.take() {
                None => {
                    self.first = Some(snapped);
                    Outcome::default()
                }
                Some(first) => {
                    let typed =
                        self.lock(HudKind::Width).is_some() || self.lock(HudKind::Height).is_some();
                    let (p0, p1) = (first.0, self.rect_corner(first.0, snapped.0));
                    let snapped = if typed { (p1, None, None) } else { snapped };
                    if (p0[0] - p1[0]).abs() < 1e-9 || (p0[1] - p1[1]).abs() < 1e-9 {
                        self.first = Some(first);
                        return Outcome::default();
                    }
                    let (wl, hl) = (self.lock(HudKind::Width), self.lock(HudKind::Height));
                    self.hud = None;
                    let centered = self.tool == Tool::CenterRectangle;
                    let mut after = Vec::new();
                    let center = centered.then(|| Self::point_at(sk, first, &mut after));
                    let lines = if let Some(center) = center {
                        sk.add_center_rectangle(center, p1)
                    } else {
                        sk.add_rectangle(p0, p1)
                    };
                    if let Some(w) = wl {
                        Self::add_dim(
                            sk,
                            Constraint::Length {
                                line: lines[0],
                                value: w,
                            },
                        );
                    }
                    if let Some(h) = hl {
                        Self::add_dim(
                            sk,
                            Constraint::Length {
                                line: lines[1],
                                value: h,
                            },
                        );
                    }
                    let c0 = sk.line_pts(lines[0]).unwrap().0;
                    let c2 = sk.line_pts(lines[2]).unwrap().0;
                    for (corner, snap) in [(c0, first), (c2, snapped)]
                        .into_iter()
                        .skip(usize::from(centered))
                    {
                        match snap {
                            (_, Some(p), _) => after.push(Constraint::Coincident(p, corner)),
                            (_, None, Some(c)) => after.push(match c {
                                CurveSnap::On(c) => Constraint::Coincident(corner, c),
                                CurveSnap::Midpoint(c) => Constraint::Midpoint(corner, c),
                            }),
                            _ => {}
                        }
                    }
                    Self::apply_all(sk, after);
                    Outcome::changed()
                }
            },
            Tool::Circle => match self.first.take() {
                None => {
                    self.first = Some(snapped);
                    Outcome::default()
                }
                Some(first) => {
                    let dl = self.lock(HudKind::Diameter);
                    let r = self.circle_radius(first.0, snapped.0);
                    if r < 1e-9 {
                        self.first = Some(first);
                        return Outcome::default();
                    }
                    self.hud = None;
                    let mut after = Vec::new();
                    let c = Self::point_at(sk, first, &mut after);
                    let circle = sk.add_circle(c, r);
                    if let (None, (_, _, Some(CurveSnap::On(curve)))) = (dl, snapped) {
                        if curve != circle {
                            after.push(Constraint::Tangent(circle, curve));
                        }
                    }
                    if dl.is_none() && matches!(snapped.2, Some(CurveSnap::Midpoint(_))) {
                        let p = Self::point_at(sk, snapped, &mut after);
                        after.push(Constraint::Coincident(p, circle));
                    }
                    Self::apply_all(sk, after);
                    if let Some(d) = dl {
                        Self::add_dim(
                            sk,
                            Constraint::Diameter {
                                e: circle,
                                value: d,
                            },
                        );
                    }
                    Outcome::changed()
                }
            },
            Tool::Arc => {
                if self.first.is_none() {
                    self.first = Some(snapped);
                    return Outcome::default();
                }
                let first = self.first.unwrap();
                match self.arc_start {
                    None => {
                        let start = self.arc_start_point(first.0, snapped.0);
                        if dist(first.0, start) < 1e-9 {
                            return Outcome::default();
                        }
                        self.arc_radius_lock = self.lock(HudKind::Radius);
                        self.arc_start = Some(start);
                        self.arc_sweep = 0.0;
                        self.arc_last = (start[1] - first.0[1]).atan2(start[0] - first.0[0]);
                        self.hud = None;
                        Outcome::default()
                    }
                    Some(start) => {
                        let c = first.0;
                        let r = dist(c, start);
                        let a0 = (start[1] - c[1]).atan2(start[0] - c[0]);
                        let sweep = self.arc_sweep_now();
                        let ang = if self.lock(HudKind::Sweep).is_some() {
                            a0 + sweep
                        } else {
                            (snapped.0[1] - c[1]).atan2(snapped.0[0] - c[0])
                        };
                        let end = [c[0] + r * ang.cos(), c[1] + r * ang.sin()];
                        let mut after = Vec::new();
                        let ce = Self::point_at(sk, first, &mut after);
                        let start_id = sk
                            .entities
                            .keys()
                            .copied()
                            .find(|&id| sk.is_point(id) && dist(sk.pos(id), start) < 1e-7);
                        let end_id = sk
                            .entities
                            .keys()
                            .copied()
                            .find(|&id| sk.is_point(id) && dist(sk.pos(id), end) < 1e-7);
                        let ps = Self::point_at(sk, (start, start_id, None), &mut after);
                        let pe = Self::point_at(sk, (end, end_id, None), &mut after);
                        let arc = if sweep >= 0.0 {
                            sk.add_arc(ce, ps, pe)
                        } else {
                            sk.add_arc(ce, pe, ps)
                        };
                        Self::apply_all(sk, after);
                        if let Some(rl) = self.arc_radius_lock.take() {
                            Self::add_dim(sk, Constraint::Radius { e: arc, value: rl });
                        }
                        self.first = None;
                        self.arc_start = None;
                        self.hud = None;
                        Outcome::changed()
                    }
                }
            }
            Tool::Dimension => self.dim_click(sk, hover, cursor, at),
            Tool::Constrain(k) => match hover {
                Some(e) => {
                    if !self.picked.contains(&e) {
                        self.picked.push(e);
                    }
                    if self.picked.len() >= k.needs()
                        || (k == ConKind::HorizontalVertical && sk.line_pts(e).is_some())
                    {
                        let picked = std::mem::take(&mut self.picked);
                        let mut work = match self.reference_sketch(sk) {
                            Ok(work) => work,
                            Err(e) => return Outcome::msg(e),
                        };
                        let out = apply_constraint(&mut work, k, &picked);
                        if out.changed {
                            *sk = work;
                        }
                        self.midpoint_refs.clear();
                        self.reference_preview = None;
                        out
                    } else {
                        Outcome::default()
                    }
                }
                None => {
                    self.picked.clear();
                    self.midpoint_refs.clear();
                    self.reference_preview = None;
                    Outcome::default()
                }
            },
            Tool::Project => Outcome::default(), // handled by the app (needs bodies)
        }
    }

    fn dim_click(
        &mut self,
        sk: &mut Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
        at: Pos2,
    ) -> Outcome {
        let mut work = match self.reference_sketch(sk) {
            Ok(work) => work,
            Err(e) => return Outcome::msg(e),
        };
        self.dim_click_inner(&mut work, hover, cursor, at)
    }

    fn dim_click_inner(
        &mut self,
        sk: &mut Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
        at: Pos2,
    ) -> Outcome {
        if let Some(c) = self.pending_dim.take() {
            // This click places the label.
            let anchor = dim_anchor(sk, &c).unwrap_or(cursor);
            let c = refine_kind(sk, c, cursor);
            let measured = sk.measure(&c).unwrap_or(0.0);
            let text = if matches!(c, Constraint::Angle { .. }) {
                fmt_len(measured.to_degrees())
            } else {
                fmt_len(measured)
            };
            self.dim_edit = Some(DimEdit {
                existing: None,
                pending: Some(c),
                label: [cursor[0] - anchor[0], cursor[1] - anchor[1]],
                text,
                at,
                focus: true,
            });
            self.picked.clear();
            return Outcome::default();
        }
        match hover {
            Some(e) => {
                self.picked.push(e);
                match dimension_for(sk, &self.picked) {
                    Some(c)
                        if self.picked.len() == 2
                            || matches!(
                                c,
                                Constraint::Radius { .. } | Constraint::Diameter { .. }
                            ) =>
                    {
                        self.pending_dim = Some(c);
                    }
                    None if self.picked.len() >= 2 => {
                        self.picked.clear();
                        self.midpoint_refs.clear();
                        self.reference_preview = None;
                        return Outcome::msg("Cannot dimension that combination");
                    }
                    _ => {} // a single line or point: wait for a second pick (or a placement click)
                }
                Outcome::default()
            }
            None => {
                // Single line picked + click in space → length.
                if self.picked.len() == 1 && sk.line_pts(self.picked[0]).is_some() {
                    self.pending_dim = Some(Constraint::Length {
                        line: self.picked[0],
                        value: 0.0,
                    });
                    return self.dim_click_inner(sk, None, cursor, at);
                }
                self.picked.clear();
                self.midpoint_refs.clear();
                self.reference_preview = None;
                Outcome::default()
            }
        }
    }

    /// Commit the open dimension edit box.
    pub fn commit_dim(
        &mut self,
        sk: &mut Sketch,
        eval: &dyn Fn(&str, Unit) -> Result<f64, String>,
    ) -> Outcome {
        let mut work = match self.reference_sketch(sk) {
            Ok(work) => work,
            Err(e) => return Outcome::msg(e),
        };
        let out = self.commit_dim_inner(&mut work, eval);
        if out.changed {
            *sk = work;
            self.midpoint_refs.clear();
            self.reference_preview = None;
        }
        out
    }

    fn commit_dim_inner(
        &mut self,
        sk: &mut Sketch,
        eval: &dyn Fn(&str, Unit) -> Result<f64, String>,
    ) -> Outcome {
        let Some(ed) = self.dim_edit.take() else {
            return Outcome::default();
        };
        let c = ed.pending.clone().or_else(|| {
            ed.existing
                .and_then(|id| sk.constraints.get(&id))
                .map(|r| r.c.clone())
        });
        let Some(c) = c else {
            return Outcome::default();
        };
        let unit = if matches!(c, Constraint::Angle { .. }) {
            Unit::Angle
        } else {
            Unit::Length
        };
        let v = match eval(&ed.text, unit) {
            Ok(v) => v,
            Err(e) => {
                self.dim_edit = Some(ed);
                return Outcome::msg(e);
            }
        };
        // Remember the expression only when it refers to user parameters.
        let expr =
            (!refusion_model::expr::references(&ed.text).is_empty()).then(|| ed.text.clone());
        if !matches!(c, Constraint::Angle { .. }) && v <= 0.0 {
            self.dim_edit = Some(ed);
            return Outcome::msg("A dimension must be positive");
        }
        match ed.existing {
            Some(id) => {
                let driven = sk.constraints.get(&id).is_some_and(|r| r.driven);
                if driven {
                    return Outcome::msg("Driven dimensions cannot be edited");
                }
                match sk.set_dimension(id, v) {
                    Ok(()) => {
                        if let Some(r) = sk.constraints.get_mut(&id) {
                            r.expr = expr;
                        }
                        Outcome::changed()
                    }
                    Err(e) => Outcome::msg(e.to_string()),
                }
            }
            None => {
                let mut c = c;
                c.set_value(v);
                match sk.add_constraint(c.clone()) {
                    Ok(Some(id)) => {
                        let r = sk.constraints.get_mut(&id).unwrap();
                        r.label = ed.label;
                        r.expr = expr;
                        Outcome::changed()
                    }
                    Ok(None) => Outcome::changed(),
                    Err(AddError::Redundant) | Err(AddError::OverConstrained) => {
                        let id = sk.add_driven(c);
                        sk.constraints.get_mut(&id).unwrap().label = ed.label;
                        Outcome {
                            changed: true,
                            message: Some("That dimension would over-constrain the sketch — added as a driven (reference) dimension".into()),
                        }
                    }
                    Err(e) => Outcome::msg(e.to_string()),
                }
            }
        }
    }

    pub fn open_dim_edit(&mut self, sk: &Sketch, id: ConstraintId, at: Pos2) {
        self.midpoint_refs.clear();
        self.reference_preview = None;
        let Some(r) = sk.constraints.get(&id) else {
            return;
        };
        if r.driven {
            return;
        }
        let v = r.c.value().unwrap_or(0.0);
        let text = r.expr.clone().unwrap_or_else(|| {
            if matches!(r.c, Constraint::Angle { .. }) {
                fmt_len(v.to_degrees())
            } else {
                fmt_len(v)
            }
        });
        self.dim_edit = Some(DimEdit {
            existing: Some(id),
            pending: None,
            label: r.label,
            text,
            at,
            focus: true,
        });
    }

    // ------------------------------------------------------------- dragging

    pub fn drag_start(&mut self, sk: &Sketch, hover: Option<EntityId>, cursor: [f64; 2]) -> bool {
        if self.tool != Tool::Select {
            return false;
        }
        match hover {
            Some(e) if sk.entity(e).is_some_and(|x| !x.projected) => {
                self.drag = Some((e, cursor));
                true
            }
            _ => false,
        }
    }

    pub fn drag_to(&mut self, sk: &mut Sketch, cursor: [f64; 2]) {
        if let Some((e, grab)) = self.drag {
            // Dragged points land on grid intersections; curves move by whole grid steps.
            let target = if sk.is_point(e) {
                self.to_grid(cursor)
            } else {
                let d = self.to_grid([cursor[0] - grab[0], cursor[1] - grab[1]]);
                [grab[0] + d[0], grab[1] + d[1]]
            };
            if target != grab || !sk.is_point(e) {
                sk.drag(e, grab, target);
            }
            self.drag = Some((e, target));
        }
    }

    pub fn drag_end(&mut self) -> bool {
        self.drag.take().is_some()
    }

    pub fn delete_selection(&mut self, sk: &mut Sketch) -> Outcome {
        if self.selection.is_empty() {
            return Outcome::default();
        }
        for s in std::mem::take(&mut self.selection) {
            match s {
                SkSel::Ent(e) => {
                    if sk.entity(e).is_some() {
                        for u in sk.users_of(e) {
                            sk.remove_entity(u);
                        }
                        sk.remove_entity(e);
                    }
                }
                SkSel::Con(c) => sk.remove_constraint(c),
            }
        }
        sk.solve();
        Outcome::changed()
    }

    pub fn toggle_construction(&mut self, sk: &mut Sketch) -> Outcome {
        let mut any = false;
        for s in &self.selection {
            if let SkSel::Ent(e) = s {
                if let Some(ent) = sk.entities.get_mut(e) {
                    if !matches!(ent.geom, Geom::Point(_)) {
                        ent.construction = !ent.construction;
                        any = true;
                    }
                }
            }
        }
        if any {
            Outcome::changed()
        } else {
            Outcome::msg("Select lines, circles or arcs to toggle construction")
        }
    }

    // ------------------------------------------------------------- previews

    /// Rubber-band geometry for the current tool, in sketch coordinates.
    pub fn preview(
        &mut self,
        sk: &Sketch,
        hover: Option<EntityId>,
        cursor: [f64; 2],
    ) -> (Vec<([f64; 2], [f64; 2], bool)>, Vec<[f64; 2]>) {
        let snapped = self.snap(sk, hover, cursor);
        self.last_cursor = Some(cursor);
        self.last_hover = hover;
        self.sync_hud();
        let mut lines = Vec::new();
        let mut pts = vec![snapped.0];
        let circle = |c: [f64; 2],
                      r: f64,
                      a0: f64,
                      sweep: f64,
                      out: &mut Vec<([f64; 2], [f64; 2], bool)>,
                      dashed: bool| {
            let n = ((sweep.abs() / std::f64::consts::TAU) * 96.0)
                .ceil()
                .max(4.0) as usize;
            for i in 0..n {
                let t0 = a0 + sweep * i as f64 / n as f64;
                let t1 = a0 + sweep * (i + 1) as f64 / n as f64;
                out.push((
                    [c[0] + r * t0.cos(), c[1] + r * t0.sin()],
                    [c[0] + r * t1.cos(), c[1] + r * t1.sin()],
                    dashed,
                ));
            }
        };
        self.inference = None;
        match self.tool {
            Tool::Trim | Tool::Extend => {
                let hover = hover.and_then(|id| {
                    if sk.is_point(id) {
                        let users = sk.users_of(id);
                        (users.len() == 1).then(|| users[0])
                    } else {
                        Some(id)
                    }
                });
                if let Some(id) = hover {
                    if let Ok(seg) = refusion_model::sketch::edit::edit_preview(
                        sk,
                        id,
                        cursor,
                        self.tool == Tool::Extend,
                    ) {
                        match seg {
                            refusion_model::sketch::Seg::Line { a, b } => lines.push((a, b, false)),
                            refusion_model::sketch::Seg::Arc { c, r, a, .. } => {
                                let angle = (a[1] - c[1]).atan2(a[0] - c[0]);
                                preview_arc(c, r, angle, seg.sweep(), &mut lines);
                            }
                        }
                    }
                }
            }

            Tool::Line => {
                if let Some(start) = self.chain {
                    let a = sk.pos(start);
                    let (b, inf) = self.line_end(
                        a,
                        snapped.0,
                        snapped.1.is_some() || matches!(snapped.2, Some(CurveSnap::Midpoint(_))),
                    );
                    self.inference = inf;
                    lines.push((a, b, false));
                    pts = vec![b];
                    if let Some(h) = self.hud.as_mut() {
                        h.live(HudKind::Length, dist(a, b));
                        h.live(HudKind::Angle, (b[1] - a[1]).atan2(b[0] - a[0]));
                    }
                }
            }
            Tool::Rectangle | Tool::CenterRectangle => {
                if let Some(first) = self.first {
                    let q = self.rect_corner(first.0, snapped.0);
                    let centered = self.tool == Tool::CenterRectangle;
                    let p = if centered {
                        [2.0 * first.0[0] - q[0], 2.0 * first.0[1] - q[1]]
                    } else {
                        first.0
                    };
                    pts = vec![q];
                    if let Some(h) = self.hud.as_mut() {
                        h.live(HudKind::Width, (q[0] - p[0]).abs());
                        h.live(HudKind::Height, (q[1] - p[1]).abs());
                    }
                    let c = [p, [q[0], p[1]], q, [p[0], q[1]]];
                    for i in 0..4 {
                        lines.push((c[i], c[(i + 1) % 4], false));
                    }
                    if centered {
                        lines.push((c[0], c[2], true));
                        lines.push((c[1], c[3], true));
                    }
                }
            }
            Tool::Circle => {
                if let Some(first) = self.first {
                    let r = self.circle_radius(first.0, snapped.0);
                    if let Some(h) = self.hud.as_mut() {
                        h.live(HudKind::Diameter, 2.0 * r);
                    }
                    circle(first.0, r, 0.0, std::f64::consts::TAU, &mut lines, false);
                    let d = dist(first.0, snapped.0).max(1e-12);
                    let rim = [
                        first.0[0] + (snapped.0[0] - first.0[0]) / d * r,
                        first.0[1] + (snapped.0[1] - first.0[1]) / d * r,
                    ];
                    lines.push((first.0, rim, true));
                }
            }
            Tool::Offset => {
                if let Some(&id) = self.picked.first() {
                    let live = offset_distance(sk, id, cursor);
                    if let Some(h) = self.hud.as_mut() {
                        h.live(HudKind::Length, live.abs());
                    }
                    let d = self
                        .lock(HudKind::Length)
                        .map(|v| if live < 0.0 { -v } else { v })
                        .unwrap_or(live);
                    match sk.entities[&id].geom {
                        Geom::Line { a, b } => {
                            let a = sk.pos(a);
                            let b = sk.pos(b);
                            let l = dist(a, b).max(1e-12);
                            let n = [-(b[1] - a[1]) / l * d, (b[0] - a[0]) / l * d];
                            lines.push((
                                [a[0] + n[0], a[1] + n[1]],
                                [b[0] + n[0], b[1] + n[1]],
                                false,
                            ));
                        }
                        Geom::Circle { c, .. } => {
                            let r = sk.radius(id).unwrap() + d;
                            if r > 1e-7 {
                                preview_arc(sk.pos(c), r, 0.0, std::f64::consts::TAU, &mut lines);
                            }
                        }
                        Geom::Arc { c, s, e } => {
                            let c = sk.pos(c);
                            let s = sk.pos(s);
                            let e = sk.pos(e);
                            let a = (s[1] - c[1]).atan2(s[0] - c[0]);
                            let sweep = ((e[1] - c[1]).atan2(e[0] - c[0]) - a)
                                .rem_euclid(std::f64::consts::TAU);
                            let r = sk.radius(id).unwrap() + d;
                            if r > 1e-7 {
                                preview_arc(c, r, a, sweep, &mut lines);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Tool::TangentArc => {
                if let Some((start, source)) = self.tangent_source {
                    if let Some((c, end, ccw)) =
                        tangent_geometry(sk, start, source, snapped.0, self.lock(HudKind::Radius))
                    {
                        let a = sk.pos(start);
                        let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
                        let t1 = (end[1] - c[1]).atan2(end[0] - c[0]);
                        let sweep = if ccw {
                            (t1 - t0).rem_euclid(std::f64::consts::TAU)
                        } else {
                            -(t0 - t1).rem_euclid(std::f64::consts::TAU)
                        };
                        preview_arc(c, dist(c, a), t0, sweep, &mut lines);
                        pts = vec![end];
                        self.inference = Some("Tangent");
                        if let Some(h) = self.hud.as_mut() {
                            h.live(HudKind::Radius, dist(c, a));
                        }
                    }
                }
            }
            Tool::ThreePointArc => {
                if let Some(first) = self.first {
                    if let Some(end) = self.arc_start {
                        if let Ok((c, ccw)) =
                            refusion_model::sketch::edit::three_point_arc(first.0, end, snapped.0)
                        {
                            let a = (first.0[1] - c[1]).atan2(first.0[0] - c[0]);
                            let b = (end[1] - c[1]).atan2(end[0] - c[0]);
                            let sweep = if ccw {
                                (b - a).rem_euclid(std::f64::consts::TAU)
                            } else {
                                -(a - b).rem_euclid(std::f64::consts::TAU)
                            };
                            preview_arc(c, dist(c, first.0), a, sweep, &mut lines);
                        }
                    } else {
                        lines.push((first.0, snapped.0, true));
                    }
                }
            }
            Tool::Arc => {
                if let Some(first) = self.first {
                    let c = first.0;
                    match self.arc_start {
                        None => {
                            let s = self.arc_start_point(c, snapped.0);
                            if let Some(h) = self.hud.as_mut() {
                                h.live(HudKind::Radius, dist(c, s));
                            }
                            lines.push((c, s, true));
                        }
                        Some(s) => {
                            let r = dist(c, s);
                            let a0 = (s[1] - c[1]).atan2(s[0] - c[0]);
                            let ang = (snapped.0[1] - c[1]).atan2(snapped.0[0] - c[0]);
                            let mut d = ang - self.arc_last;
                            while d > std::f64::consts::PI {
                                d -= std::f64::consts::TAU;
                            }
                            while d < -std::f64::consts::PI {
                                d += std::f64::consts::TAU;
                            }
                            self.arc_sweep = (self.arc_sweep + d)
                                .clamp(-std::f64::consts::TAU, std::f64::consts::TAU);
                            self.arc_last = ang;
                            let sweep = self.arc_sweep_now();
                            if let Some(h) = self.hud.as_mut() {
                                h.live(HudKind::Sweep, sweep.abs());
                            }
                            circle(c, r, a0, sweep, &mut lines, false);
                            lines.push((c, s, true));
                        }
                    }
                }
            }
            _ => {}
        }
        (lines, pts)
    }

    // ------------------------------------------------------------- overlay

    /// Draw dimensions and constraint glyphs; handle their clicks. Returns
    /// true if the pointer is over a glyph or label (so the viewport should
    /// not treat the click as a geometry pick).
    pub fn overlay(
        &mut self,
        ui: &mut egui::Ui,
        sk: &mut Sketch,
        to_screen: &dyn Fn([f64; 2]) -> Option<Pos2>,
        cursor: Option<[f64; 2]>,
    ) -> bool {
        let painter = ui.painter().clone();
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let clicked = ui.input(|i| i.pointer.primary_clicked());
        let double = ui.input(|i| {
            i.pointer
                .button_double_clicked(egui::PointerButton::Primary)
        });
        let mut over = false;
        self.hover_con = None;
        let dim_color = Color32::from_rgb(20, 20, 26);
        let font = FontId::proportional(13.0);

        let (pressed, down) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
        if !down {
            self.drag_dim = None;
        }
        // Text dragging: move the label with the cursor; the graphics follow.
        if let (Some(id), Some(cur)) = (self.drag_dim, cursor) {
            let c = sk.constraints.get(&id).map(|r| r.c.clone());
            if let Some(anchor) = c.as_ref().and_then(|c| dim_anchor(sk, c)) {
                if let Some(r) = sk.constraints.get_mut(&id) {
                    r.label = [cur[0] - anchor[0], cur[1] - anchor[1]];
                }
            }
            over = true;
        }
        let ids: Vec<ConstraintId> = if self.show_dims {
            sk.constraints
                .iter()
                .filter(|(_, r)| r.c.is_dimension())
                .map(|(&i, _)| i)
                .collect()
        } else {
            vec![]
        };
        for id in ids {
            let rec = sk.constraints[&id].clone();
            let Some(anchor) = dim_anchor(sk, &rec.c) else {
                continue;
            };
            let t = [anchor[0] + rec.label[0], anchor[1] + rec.label[1]];
            let v = rec.c.value().unwrap_or(0.0);
            let mut text = dim_text(&rec.c, v);
            if rec.driven {
                text = format!("({text})");
            } else if rec.expr.is_some() {
                text = format!("fx: {text}");
            }
            let selected = self.selection.contains(&SkSel::Con(id)) || self.drag_dim == Some(id);
            let col = if selected {
                Color32::from_rgb(20, 110, 250)
            } else {
                dim_color
            };
            let Some(spec) = crate::dim_draw::spec(sk, &rec.c, t) else {
                continue;
            };
            let Some(rect) = draw_dimension(
                &painter,
                &spec,
                text,
                font.clone(),
                col,
                to_screen,
                pointer,
                selected,
            ) else {
                continue;
            };
            let hovered = pointer.is_some_and(|p| rect.contains(p));
            if hovered {
                over = true;
                self.hover_con = Some(id);
                if double {
                    self.open_dim_edit(sk, id, rect.right_top());
                } else if pressed && self.tool == Tool::Select {
                    // start dragging the text (and select the dimension)
                    self.drag_dim = Some(id);
                    self.dim_moved = true;
                    self.selection = vec![SkSel::Con(id)];
                } else if clicked {
                    self.selection = vec![SkSel::Con(id)];
                }
            }
        }

        // Constraint glyphs, stacked per entity
        let mut stack: std::collections::HashMap<EntityId, i32> = Default::default();
        let ids: Vec<ConstraintId> = if self.show_constraints {
            sk.constraints
                .iter()
                .filter(|(_, r)| !r.c.is_dimension())
                .map(|(&i, _)| i)
                .collect()
        } else {
            vec![]
        };
        for id in ids {
            let c = sk.constraints[&id].c.clone();
            let (g, host) = glyph_for(sk, &c);
            let Some(host) = host else { continue };
            let Some(base) = glyph_anchor(sk, host) else {
                continue;
            };
            let Some(p) = to_screen(base) else { continue };
            let k = stack.entry(host).or_insert(0);
            let pos = p + Vec2::new(14.0 + 18.0 * *k as f32, -14.0);
            *k += 1;
            let rect = Rect::from_center_size(pos, Vec2::splat(16.0));
            let hovered = pointer.is_some_and(|q| rect.contains(q));
            let selected = self.selection.contains(&SkSel::Con(id));
            let bg = if selected {
                Color32::from_rgb(40, 120, 250)
            } else if hovered {
                Color32::from_rgb(200, 225, 255)
            } else {
                Color32::from_rgba_unmultiplied(250, 250, 250, 220)
            };
            painter.rect(
                rect,
                2.0,
                bg,
                Stroke::new(1.0_f32, Color32::from_gray(120)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                g,
                FontId::proportional(11.0),
                if selected {
                    Color32::WHITE
                } else {
                    Color32::from_rgb(30, 30, 40)
                },
            );
            if hovered {
                over = true;
                self.hover_con = Some(id);
                if clicked {
                    self.selection = vec![SkSel::Con(id)];
                }
            }
        }

        // Pending dimension preview follows the cursor
        let preview_sk = self.reference_preview.as_ref().unwrap_or(sk);
        if let (Some(c), Some(cur)) = (&self.pending_dim, cursor) {
            let sk = preview_sk;
            let mut c2 = refine_kind(sk, c.clone(), cur);
            if let Some(v) = sk.measure(&c2) {
                c2.set_value(v);
                if let Some(spec) = crate::dim_draw::spec(sk, &c2, cur) {
                    draw_dimension(
                        &painter,
                        &spec,
                        dim_text(&c2, v),
                        font.clone(),
                        Color32::from_rgb(20, 110, 250),
                        to_screen,
                        None,
                        true,
                    );
                }
            }
        }
        if let Some(p) = self
            .midpoint_target
            .and_then(|e| line_midpoint(sk, e))
            .and_then(to_screen)
        {
            let color = Color32::from_rgb(20, 110, 250);
            let corners = [
                p + Vec2::new(0.0, -7.0),
                p + Vec2::new(6.0, 4.0),
                p + Vec2::new(-6.0, 4.0),
            ];
            painter.add(egui::Shape::closed_line(
                corners.to_vec(),
                egui::Stroke::new(1.5_f32, color),
            ));
            painter.line_segment(
                [p - Vec2::splat(2.0), p + Vec2::splat(2.0)],
                egui::Stroke::new(1.0_f32, color),
            );
            painter.line_segment(
                [p + Vec2::new(-2.0, 2.0), p + Vec2::new(2.0, -2.0)],
                egui::Stroke::new(1.0_f32, color),
            );
        }
        // Inference badge
        if let (Some(inf), Some(p)) = (self.inference, pointer) {
            painter.text(
                p + Vec2::new(14.0, 14.0),
                Align2::LEFT_TOP,
                inf,
                FontId::proportional(12.0),
                Color32::from_rgb(20, 110, 250),
            );
        }
        over
    }
}

fn dim_text(c: &Constraint, v: f64) -> String {
    match c {
        Constraint::Angle { .. } => format!("{}°", fmt_len(v.to_degrees())),
        Constraint::Radius { .. } => format!("R{}", fmt_len(v)),
        Constraint::Diameter { .. } => format!("⌀{}", fmt_len(v)),
        _ => fmt_len(v),
    }
}

/// Draw one dimension: extension/dimension lines, filled arrowheads, arcs,
/// and right-reading text on a background box that breaks the line.
/// Returns the text's screen rectangle (for hit testing).
#[allow(clippy::too_many_arguments)]
fn draw_dimension(
    painter: &egui::Painter,
    spec: &crate::dim_draw::DimSpec,
    text: String,
    font: FontId,
    col: Color32,
    to_screen: &dyn Fn([f64; 2]) -> Option<Pos2>,
    pointer: Option<Pos2>,
    emphasised: bool,
) -> Option<Rect> {
    let galley = painter.layout_no_wrap(text, font, col);
    let size = galley.size();
    let pr = crate::dim_draw::layout(spec, to_screen, size.x)?;
    let st = Stroke::new(if emphasised { 1.4_f32 } else { 1.0_f32 }, col);
    for (a, b) in &pr.lines {
        painter.line_segment([*a, *b], st);
    }
    for poly in &pr.polys {
        painter.add(egui::Shape::line(poly.clone(), st));
    }
    for (tip, d) in &pr.arrows {
        painter.add(egui::Shape::convex_polygon(
            crate::dim_draw::arrow_triangle(*tip, *d).to_vec(),
            col,
            Stroke::NONE,
        ));
    }
    // rotated text with a background box
    let (sin, cos) = pr.angle.sin_cos();
    let rot = |v: Vec2| Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
    let half = size * 0.5 + Vec2::new(3.0, 1.0);
    let corners: Vec<Pos2> = [
        Vec2::new(-half.x, -half.y),
        Vec2::new(half.x, -half.y),
        Vec2::new(half.x, half.y),
        Vec2::new(-half.x, half.y),
    ]
    .iter()
    .map(|c| pr.text + rot(*c))
    .collect();
    let bbox = Rect::from_points(&corners);
    let hovered = pointer.is_some_and(|p| bbox.contains(p));
    let bg = if hovered {
        Color32::from_rgb(222, 236, 255)
    } else {
        Color32::from_rgb(244, 245, 248)
    };
    painter.add(egui::Shape::convex_polygon(corners, bg, Stroke::NONE));
    let top_left = pr.text - rot(size * 0.5);
    painter.add(egui::epaint::TextShape::new(top_left, galley, col).with_angle(pr.angle));
    Some(bbox)
}

/// Horizontal / vertical / aligned from where the label is placed (point–point only).
fn refine_kind(sk: &Sketch, c: Constraint, place: [f64; 2]) -> Constraint {
    if let Constraint::Distance { a, b, value, .. } = c {
        if sk.is_point(a) && sk.is_point(b) {
            let (p, q) = (sk.pos(a), sk.pos(b));
            let (x0, x1) = (p[0].min(q[0]), p[0].max(q[0]));
            let (y0, y1) = (p[1].min(q[1]), p[1].max(q[1]));
            let in_x = place[0] >= x0 && place[0] <= x1;
            let in_y = place[1] >= y0 && place[1] <= y1;
            let kind = if in_x && !in_y {
                DimKind::Horizontal
            } else if in_y && !in_x {
                DimKind::Vertical
            } else {
                DimKind::Aligned
            };
            return Constraint::Distance { a, b, kind, value };
        }
    }
    c
}

pub fn dimension_for(sk: &Sketch, picked: &[EntityId]) -> Option<Constraint> {
    let kind = |e: EntityId| sk.entity(e).map(|x| x.geom);
    let round = |g: Geom| matches!(g, Geom::Circle { .. } | Geom::Arc { .. });
    let dist = |a: EntityId, b: EntityId| Constraint::Distance {
        a,
        b,
        kind: DimKind::Aligned,
        value: 0.0,
    };
    match *picked {
        [e] => match kind(e)? {
            Geom::Circle { .. } => Some(Constraint::Diameter { e, value: 0.0 }),
            Geom::Arc { .. } => Some(Constraint::Radius { e, value: 0.0 }),
            Geom::Line { .. } => Some(Constraint::Length {
                line: e,
                value: 0.0,
            }),
            Geom::Point(_) => None,
        },
        [a, b] => {
            let (ka, kb) = (kind(a)?, kind(b)?);
            match (ka, kb) {
                (Geom::Point(_), Geom::Point(_) | Geom::Line { .. })
                | (Geom::Line { .. }, Geom::Point(_)) => Some(dist(a, b)),
                (Geom::Line { a: a1, b: a2 }, Geom::Line { a: b1, b: b2 }) => {
                    let (p, q, r, s) = (sk.pos(a1), sk.pos(a2), sk.pos(b1), sk.pos(b2));
                    let (ux, uy, wx, wy) = (q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]);
                    let cross = (ux * wy - uy * wx) / (ux.hypot(uy) * wx.hypot(wy)).max(1e-300);
                    Some(if cross.abs() < 1e-6 {
                        dist(a, b)
                    } else {
                        Constraint::Angle { a, b, value: 0.0 }
                    })
                }
                // point ↔ centre, centre ↔ centre
                (Geom::Point(_), g) if round(g) => Some(dist(a, sk.center(b)?)),
                (g, Geom::Point(_)) if round(g) => Some(dist(sk.center(a)?, b)),
                (g, h) if round(g) && round(h) => Some(dist(sk.center(a)?, sk.center(b)?)),
                _ => None,
            }
        }
        _ => None,
    }
}

pub fn dim_anchor(sk: &Sketch, c: &Constraint) -> Option<[f64; 2]> {
    let mid = |p: [f64; 2], q: [f64; 2]| [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
    Some(match *c {
        Constraint::Length { line, .. } => {
            let (a, b) = sk.line_pts(line)?;
            mid(sk.pos(a), sk.pos(b))
        }
        Constraint::Offset { a, b, .. } => mid(glyph_anchor(sk, a)?, glyph_anchor(sk, b)?),
        Constraint::Distance { a, b, .. } => {
            let pa = if sk.is_point(a) {
                sk.pos(a)
            } else {
                glyph_anchor(sk, a)?
            };
            let pb = if sk.is_point(b) {
                sk.pos(b)
            } else {
                glyph_anchor(sk, b)?
            };
            mid(pa, pb)
        }
        Constraint::Radius { e, .. } | Constraint::Diameter { e, .. } => {
            let c = sk.pos(sk.center(e)?);
            let r = sk.radius(e)?;
            let t = match sk.entity(e)?.geom {
                Geom::Arc { c: cc, s, e: end } => {
                    let (cc, s, end) = (sk.pos(cc), sk.pos(s), sk.pos(end));
                    let a0 = (s[1] - cc[1]).atan2(s[0] - cc[0]);
                    let sw = ((end[1] - cc[1]).atan2(end[0] - cc[0]) - a0)
                        .rem_euclid(std::f64::consts::TAU);
                    a0 + sw * 0.5
                }
                _ => std::f64::consts::FRAC_PI_4,
            };
            [c[0] + r * t.cos(), c[1] + r * t.sin()]
        }
        Constraint::Angle { a, b, .. } => mid(glyph_anchor(sk, a)?, glyph_anchor(sk, b)?),
        _ => return None,
    })
}

fn glyph_anchor(sk: &Sketch, e: EntityId) -> Option<[f64; 2]> {
    Some(match sk.entity(e)?.geom {
        Geom::Point(_) => sk.pos(e),
        Geom::Line { a, b } => {
            let (p, q) = (sk.pos(a), sk.pos(b));
            [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5]
        }
        Geom::Circle { c, .. } | Geom::Arc { c, .. } => {
            let cc = sk.pos(c);
            let r = sk.radius(e)?;
            [cc[0] + r * 0.707, cc[1] - r * 0.707]
        }
    })
}

fn glyph_for(sk: &Sketch, c: &Constraint) -> (&'static str, Option<EntityId>) {
    use Constraint::*;
    match *c {
        Coincident(a, b) => ("∘", Some(if sk.is_point(a) { a } else { b })),
        Horizontal(a) => ("H", Some(a)),
        Vertical(a) => ("V", Some(a)),
        HorizontalPts(a, _) => ("H", Some(a)),
        VerticalPts(a, _) => ("V", Some(a)),
        Parallel(a, _) => ("∥", Some(a)),
        Perpendicular(a, _) => ("⊥", Some(a)),
        Tangent(a, _) => ("◠", Some(a)),
        Equal(a, _) => ("=", Some(a)),
        Concentric(a, _) => ("◎", Some(a)),
        Collinear(a, _) => ("⋯", Some(a)),
        Midpoint(p, _) => ("△", Some(p)),
        Symmetric(_, _, l) => ("⋈", Some(l)),
        Fix(e) => ("🔒", Some(e)),
        _ => ("", None),
    }
}

/// Apply a constraint tool to picked entities.
pub fn apply_constraint(sk: &mut Sketch, k: ConKind, p: &[EntityId]) -> Outcome {
    let is_line = |e: EntityId| sk.line_pts(e).is_some();
    let c = match (k, p) {
        (ConKind::Fix, [e]) => {
            if let Some((&id, _)) = sk
                .constraints
                .iter()
                .find(|(_, r)| r.c == Constraint::Fix(*e))
            {
                sk.remove_constraint(id);
                sk.solve();
                return Outcome::changed();
            }
            Constraint::Fix(*e)
        }
        (ConKind::HorizontalVertical, [e]) if is_line(*e) => {
            let (a, b) = sk.line_pts(*e).unwrap();
            let (p, q) = (sk.pos(a), sk.pos(b));
            if (q[0] - p[0]).abs() >= (q[1] - p[1]).abs() {
                Constraint::Horizontal(*e)
            } else {
                Constraint::Vertical(*e)
            }
        }
        (ConKind::HorizontalVertical, [a, b]) if sk.is_point(*a) && sk.is_point(*b) => {
            let (p, q) = (sk.pos(*a), sk.pos(*b));
            if (q[0] - p[0]).abs() >= (q[1] - p[1]).abs() {
                Constraint::HorizontalPts(*a, *b)
            } else {
                Constraint::VerticalPts(*a, *b)
            }
        }
        (ConKind::Coincident, [a, b]) => Constraint::Coincident(*a, *b),
        (ConKind::Collinear, [a, b]) => Constraint::Collinear(*a, *b),
        (ConKind::Concentric, [a, b]) => Constraint::Concentric(*a, *b),
        (ConKind::Midpoint, [a, b]) => {
            if sk.is_point(*a) {
                Constraint::Midpoint(*a, *b)
            } else {
                Constraint::Midpoint(*b, *a)
            }
        }
        (ConKind::Parallel, [a, b]) => Constraint::Parallel(*a, *b),
        (ConKind::Perpendicular, [a, b]) => Constraint::Perpendicular(*a, *b),
        (ConKind::Tangent, [a, b]) => Constraint::Tangent(*a, *b),
        (ConKind::Equal, [a, b]) => Constraint::Equal(*a, *b),
        (ConKind::Symmetric, [a, b, l]) => Constraint::Symmetric(*a, *b, *l),
        _ => return Outcome::msg("That constraint does not apply to this selection"),
    };
    match sk.add_constraint(c) {
        Ok(_) => Outcome::changed(),
        Err(e) => Outcome::msg(e.to_string()),
    }
}

// ------------------------------------------------------------------ selection

pub fn chain_from(sk: &Sketch, start: EntityId) -> Vec<EntityId> {
    let Some(e0) = sk.entity(start) else {
        return vec![];
    };
    if matches!(e0.geom, Geom::Point(_)) {
        return vec![start];
    }
    let construction = e0.construction;
    let ends = |e: EntityId| -> Vec<EntityId> {
        match sk.entity(e).map(|x| x.geom) {
            Some(Geom::Line { a, b }) => vec![a, b],
            Some(Geom::Arc { s, e, .. }) => vec![s, e],
            _ => vec![],
        }
    };
    let mut out = vec![start];
    let mut queue = vec![start];
    while let Some(cur) = queue.pop() {
        for p in ends(cur) {
            for u in sk.users_of(p) {
                let same_kind = sk.entity(u).is_some_and(|x| {
                    x.construction == construction && !matches!(x.geom, Geom::Circle { .. })
                });
                // A curve "uses" a point only as an endpoint for chaining (not an arc centre).
                if same_kind && !out.contains(&u) && ends(u).contains(&p) {
                    out.push(u);
                    queue.push(u);
                }
            }
        }
    }
    out
}

/// Screen-space polyline of an entity (points: one vertex).
fn screen_poly(
    sk: &Sketch,
    e: EntityId,
    to_screen: &dyn Fn([f64; 2]) -> Option<Pos2>,
) -> Option<Vec<Pos2>> {
    let pts: Vec<[f64; 2]> = match sk.entity(e)?.geom {
        Geom::Point(_) => vec![sk.pos(e)],
        Geom::Line { a, b } => vec![sk.pos(a), sk.pos(b)],
        Geom::Circle { c, r } => {
            let (c, r) = (sk.pos(c), sk.x[r]);
            (0..=64)
                .map(|i| {
                    let t = std::f64::consts::TAU * i as f64 / 64.0;
                    [c[0] + r * t.cos(), c[1] + r * t.sin()]
                })
                .collect()
        }
        Geom::Arc { c, s, e: end } => {
            let (c, s, end) = (sk.pos(c), sk.pos(s), sk.pos(end));
            let r = dist(c, s);
            let a0 = (s[1] - c[1]).atan2(s[0] - c[0]);
            let mut sw =
                ((end[1] - c[1]).atan2(end[0] - c[0]) - a0).rem_euclid(std::f64::consts::TAU);
            if sw < 1e-9 {
                sw = std::f64::consts::TAU;
            }
            (0..=64)
                .map(|i| {
                    let t = a0 + sw * i as f64 / 64.0;
                    [c[0] + r * t.cos(), c[1] + r * t.sin()]
                })
                .collect()
        }
    };
    pts.into_iter().map(to_screen).collect()
}

fn seg_hits_rect(a: Pos2, b: Pos2, r: Rect) -> bool {
    if r.contains(a) || r.contains(b) {
        return true;
    }
    let edges = [
        (r.left_top(), r.right_top()),
        (r.right_top(), r.right_bottom()),
        (r.right_bottom(), r.left_bottom()),
        (r.left_bottom(), r.left_top()),
    ];
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    edges.iter().any(|&(c, d)| {
        let (d1, d2, d3, d4) = (
            cross(c, d, a),
            cross(c, d, b),
            cross(a, b, c),
            cross(a, b, d),
        );
        (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
    })
}

/// Window (fully inside) or crossing (touching) selection of sketch entities.
pub fn box_select(
    sk: &Sketch,
    r: Rect,
    crossing: bool,
    to_screen: &dyn Fn([f64; 2]) -> Option<Pos2>,
) -> Vec<EntityId> {
    sk.entities
        .keys()
        .copied()
        .filter(|&e| {
            let Some(poly) = screen_poly(sk, e, to_screen) else {
                return false;
            };
            if crossing {
                poly.iter().any(|p| r.contains(*p))
                    || poly.windows(2).any(|w| seg_hits_rect(w[0], w[1], r))
            } else {
                poly.iter().all(|p| r.contains(*p))
            }
        })
        .collect()
}

impl SketchSession {
    /// Select-tool press on an entity: select (Shift toggles), returns true.
    pub fn select_at(&mut self, hover: Option<EntityId>, shift: bool) {
        match hover {
            Some(e) => {
                let s = SkSel::Ent(e);
                if shift {
                    if let Some(i) = self.selection.iter().position(|x| *x == s) {
                        self.selection.remove(i);
                    } else {
                        self.selection.push(s);
                    }
                } else if !self.selection.contains(&s) {
                    self.selection = vec![s];
                }
            }
            None if !shift => self.selection.clear(),
            None => {}
        }
    }

    /// Replace (or with `add`, extend) the selection.
    pub fn set_selection(&mut self, ents: Vec<EntityId>, add: bool, remove: bool) {
        if !add && !remove {
            self.selection.clear();
        }
        for e in ents {
            let s = SkSel::Ent(e);
            if remove {
                self.selection.retain(|x| *x != s);
            } else if !self.selection.contains(&s) {
                self.selection.push(s);
            }
        }
    }

    pub fn selected_entities(&self) -> Vec<EntityId> {
        self.selection
            .iter()
            .filter_map(|s| {
                if let SkSel::Ent(e) = s {
                    Some(*e)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_rectangle_has_full_size_hud_and_stays_centered_after_resize() {
        let mut sk = Sketch::new();
        let center = sk.add_point([10.0, 20.0]);
        sk.entities.get_mut(&center).unwrap().projected = true;
        let mut s = SketchSession::new(
            1,
            Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        s.set_tool(Tool::CenterRectangle);
        let eval = |t: &str, u: Unit| {
            refusion_model::expr::eval(t, u, &std::collections::HashMap::new())
                .map_err(|e| e.to_string())
        };
        s.click(
            &mut sk,
            Some(center),
            [10.0, 20.0],
            Pos2::ZERO,
            &eval,
            false,
        );
        let (preview, _) = s.preview(&sk, None, [13.0, 18.0]);
        assert_eq!(preview.len(), 6); // perimeter and two construction diagonals
        assert_eq!(s.hud.as_ref().unwrap().fields[0].text, "6");
        let hud = s.hud.as_mut().unwrap();
        hud.fields[0].text = "40".into();
        hud.edit(0, &eval);
        hud.fields[1].text = "10".into();
        hud.edit(1, &eval);
        s.click(&mut sk, None, [13.0, 18.0], Pos2::ZERO, &eval, false);
        assert_eq!(sk.analyze_now().dof, 0);
        assert_eq!(sk.entities.values().filter(|e| e.construction).count(), 2);
        let profiles = refusion_model::sketch::find_profiles(&sk);
        assert_eq!(profiles.len(), 1);
        let mut corners: Vec<_> = sk
            .entities
            .keys()
            .filter(|&&e| sk.is_point(e) && e != center)
            .map(|&e| sk.pos(e))
            .collect();
        corners.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        for (got, expected) in
            corners
                .iter()
                .zip([[-10.0, 15.0], [-10.0, 25.0], [30.0, 15.0], [30.0, 25.0]])
        {
            assert!(dist(*got, expected) < 1e-6, "{corners:?}");
        }
        let dimension = sk
            .constraints
            .iter()
            .find_map(|(&id, rec)| {
                matches!(rec.c, Constraint::Length { value, .. } if value == 40.0).then_some(id)
            })
            .unwrap();
        if let Constraint::Length { value, .. } = &mut sk.constraints.get_mut(&dimension).unwrap().c
        {
            *value = 60.0;
        }
        assert!(sk.solve().converged);
        for e in sk.entities.values().filter(|e| e.construction) {
            if let Geom::Line { a, b } = e.geom {
                let (a, b) = (sk.pos(a), sk.pos(b));
                assert!(dist([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], [10.0, 20.0]) < 1e-6);
            }
        }
    }

    #[test]
    fn center_rectangle_rejects_zero_area_and_cancel_resets_it() {
        let mut sk = Sketch::new();
        let mut s = SketchSession::new(
            1,
            Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        s.set_tool(Tool::CenterRectangle);
        let eval = |_: &str, _: Unit| Err("unused".to_string());
        s.click(&mut sk, None, [0.0, 0.0], Pos2::ZERO, &eval, false);
        s.click(&mut sk, None, [0.0, 10.0], Pos2::ZERO, &eval, false);
        assert!(sk.entities.is_empty());
        assert!(s.in_progress());
        s.cancel();
        assert!(!s.in_progress());
    }

    #[test]
    fn chain_selects_rectangle_but_not_separate_line() {
        let mut sk = Sketch::new();
        let rect = sk.add_rectangle([0.0, 0.0], [10.0, 5.0]);
        let other = sk.add_line_pts([20.0, 0.0], [30.0, 0.0]);
        let mut c = chain_from(&sk, rect[2]);
        c.sort();
        let mut want = rect.to_vec();
        want.sort();
        assert_eq!(c, want);
        assert_eq!(chain_from(&sk, other), vec![other]);
    }

    #[test]
    fn window_vs_crossing() {
        let mut sk = Sketch::new();
        let inside = sk.add_line_pts([1.0, 1.0], [2.0, 2.0]);
        let crossing = sk.add_line_pts([5.0, 5.0], [15.0, 5.0]);
        let id = |p: [f64; 2]| Some(Pos2::new(p[0] as f32, p[1] as f32));
        let r = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(10.0, 10.0));
        let w = box_select(&sk, r, false, &id);
        assert!(w.contains(&inside) && !w.contains(&crossing));
        let c = box_select(&sk, r, true, &id);
        assert!(c.contains(&inside) && c.contains(&crossing));
    }

    #[test]
    fn grid_snap_rounds_free_points_but_not_geometry_snaps() {
        let mut sk = Sketch::new();
        let p = sk.add_point([1.23, 4.56]);
        let mut s = SketchSession::new(
            1,
            refusion_model::kernel::Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        assert_eq!(s.snap(&sk, None, [12.3, -7.6]).0, [12.3, -7.6]);
        s.grid_snap = Some(5.0);
        assert_eq!(s.snap(&sk, None, [12.3, -7.6]).0, [10.0, -10.0]);
        // hovering an existing point still snaps to that point, off-grid
        assert_eq!(s.snap(&sk, Some(p), [12.3, -7.6]).0, [1.23, 4.56]);
        // a rectangle drawn with grid snap has grid corners
        s.set_tool(Tool::Rectangle);
        let eval = |t: &str, u: Unit| {
            refusion_model::expr::eval(t, u, &std::collections::HashMap::new())
                .map_err(|e| e.to_string())
        };
        s.click(&mut sk, None, [1.4, 2.6], Pos2::ZERO, &eval, false);
        s.click(&mut sk, None, [21.0, 13.2], Pos2::ZERO, &eval, false);
        let mut xs: Vec<f64> = sk
            .entities
            .keys()
            .filter(|&&e| e != p && sk.is_point(e))
            .map(|&e| sk.pos(e)[0])
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup();
        assert_eq!(xs, vec![0.0, 20.0]);
    }

    #[test]
    fn hud_locks_drive_geometry_and_dimensions() {
        let mut sk = Sketch::new();
        let mut s = SketchSession::new(
            1,
            refusion_model::kernel::Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        s.set_tool(Tool::Rectangle);
        let eval = |t: &str, u: Unit| {
            refusion_model::expr::eval(t, u, &std::collections::HashMap::new())
                .map_err(|e| e.to_string())
        };
        s.click(&mut sk, None, [0.0, 0.0], Pos2::ZERO, &eval, false);
        s.preview(&sk, None, [3.0, -2.0]);
        let h = s.hud.as_mut().unwrap();
        assert_eq!(h.fields[0].text, "3"); // live width
        h.fields[0].text = "40".into();
        h.edit(0, &eval);
        s.preview(&sk, None, [3.0, -2.0]);
        s.commit_hud(&mut sk, &eval);
        // width locked to 40 (in the cursor's direction), height follows cursor
        let xs: Vec<f64> = sk
            .entities
            .keys()
            .filter(|&&e| sk.is_point(e))
            .map(|&e| sk.pos(e)[0])
            .collect();
        assert!(xs.iter().any(|&x| (x - 40.0).abs() < 1e-6), "{xs:?}");
        assert!(sk.constraints.values().any(
            |r| matches!(r.c, Constraint::Length { value, .. } if (value - 40.0).abs() < 1e-9)
                && !r.driven
        ));
    }
}

fn offset_distance(sk: &Sketch, id: EntityId, p: [f64; 2]) -> f64 {
    match sk.entities[&id].geom {
        Geom::Line { a, b } => {
            let a = sk.pos(a);
            let b = sk.pos(b);
            ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / dist(a, b).max(1e-12)
        }
        Geom::Circle { c, .. } | Geom::Arc { c, .. } => dist(sk.pos(c), p) - sk.radius(id).unwrap(),
        _ => 0.0,
    }
}
fn preview_arc(
    c: [f64; 2],
    r: f64,
    a: f64,
    sweep: f64,
    lines: &mut Vec<([f64; 2], [f64; 2], bool)>,
) {
    let n = ((sweep.abs() * 24.0).ceil() as usize).clamp(2, 160);
    for i in 0..n {
        let p = |j: usize| {
            let t = a + sweep * j as f64 / n as f64;
            [c[0] + r * t.cos(), c[1] + r * t.sin()]
        };
        lines.push((p(i), p(i + 1), false));
    }
}
#[cfg(test)]
mod new_tool_tests {
    use super::*;
    fn eval(s: &str, u: Unit) -> Result<f64, String> {
        refusion_model::expr::eval(s, u, &std::collections::HashMap::new())
            .map_err(|e| e.to_string())
    }
    #[test]
    fn three_point_arc_reuses_both_snapped_endpoints_and_cancel_resets() {
        let mut sk = Sketch::new();
        let a = sk.add_point([1.0, 0.0]);
        let b = sk.add_point([-1.0, 0.0]);
        let mut session = SketchSession::new(
            1,
            Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        session.set_tool(Tool::ThreePointArc);
        assert!(
            !session
                .click(&mut sk, Some(a), [1.0, 0.0], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(
            !session
                .click(&mut sk, Some(b), [-1.0, 0.0], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(
            session
                .click(&mut sk, None, [0.0, 1.0], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(sk
            .entities
            .values()
            .any(|e| matches!(e.geom,Geom::Arc{s,e,..}if s==a&&e==b)));
        assert!(!session.in_progress());
        assert!(sk.solve().converged);
        session.click(&mut sk, Some(a), [1.0, 0.0], Pos2::ZERO, &eval, false);
        assert!(session.cancel());
        assert!(session.arc_end_snap.is_none());
    }
    #[test]
    fn offset_clicks_create_editable_associative_dimension() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let mut session = SketchSession::new(
            1,
            Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        session.set_tool(Tool::Offset);
        assert!(
            !session
                .click(&mut sk, Some(line), [5.0, 0.0], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(
            session
                .click(&mut sk, None, [5.0, 3.0], Pos2::ZERO, &eval, false)
                .changed
        );
        let rec = sk
            .constraints
            .values()
            .find(|r| matches!(r.c, Constraint::Offset { .. }))
            .unwrap();
        assert!(rec.c.is_dimension());
        assert!((rec.c.value().unwrap() - 3.0).abs() < 1e-6);
        assert!(dim_anchor(&sk, &rec.c).is_some());
        assert!(crate::dim_draw::spec(&sk, &rec.c, [5.0, 1.5]).is_some());
    }
}
#[cfg(test)]
mod endpoint_tool_tests {
    use super::*;
    #[test]
    fn extend_accepts_a_unique_curve_endpoint() {
        let mut sk = Sketch::new();
        let a = sk.add_point([0.0, 0.0]);
        let b = sk.add_point([2.0, 0.0]);
        sk.add_line(a, b);
        sk.add_line_pts([5.0, -1.0], [5.0, 1.0]);
        let mut s = SketchSession::new(
            1,
            Frame {
                origin: [0.0; 3],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
            },
            1,
        );
        s.set_tool(Tool::Extend);
        let eval = |_: &str, _: Unit| Ok(0.0);
        assert!(
            s.click(&mut sk, Some(b), [2.0, 0.0], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(sk
            .entities
            .iter()
            .any(|(&id, e)| matches!(e.geom, Geom::Point(_)) && sk.pos(id) == [5.0, 0.0]));
    }
}
#[cfg(test)]
mod tangent_arc_tests {
    use super::*;
    fn session() -> SketchSession {
        let mut s = SketchSession::new(
            1,
            Frame {
                origin: [0.; 3],
                x: [1., 0., 0.],
                y: [0., 1., 0.],
            },
            1,
        );
        s.set_tool(Tool::TangentArc);
        s
    }
    fn eval(s: &str, u: Unit) -> Result<f64, String> {
        refusion_model::expr::eval(s, u, &std::collections::HashMap::<String, f64>::new())
            .map_err(|e| e.to_string())
    }
    #[test]
    fn both_sweep_directions_reuse_endpoints_and_keep_tangent_on_edit() {
        for side in [-1., 1.] {
            let mut sk = Sketch::new();
            let a = sk.add_point([0., 0.]);
            let b = sk.add_point([10., 0.]);
            let line = sk.add_line(a, b);
            let end = sk.add_point([15., 5. * side]);
            let mut s = session();
            assert!(
                !s.click(&mut sk, Some(b), [10., 0.], Pos2::ZERO, &eval, false)
                    .changed
            );
            assert!(!s.preview(&sk, Some(end), [15., 5. * side]).0.is_empty());
            let out = s.click(
                &mut sk,
                Some(end),
                [15., 5. * side],
                Pos2::ZERO,
                &eval,
                false,
            );
            assert!(out.changed, "{:?}", out.message);
            let arc = sk
                .entities
                .iter()
                .find_map(|(&id, e)| {
                    matches!(e.geom,Geom::Arc {s,e,..} if (s==b&&e==end)||(s==end&&e==b))
                        .then_some(id)
                })
                .unwrap();
            assert!(sk
                .constraints
                .values()
                .any(|r| matches!(r.c,Constraint::Tangent(l,e) if l==line&&e==arc)));
            assert!(sk.solve().converged);
            assert!(!s.in_progress());
            let p = sk.pos(end);
            sk.set_pos(end, [p[0], p[1] + side]);
            assert!(sk.solve().converged);
        }
    }
    #[test]
    fn follows_arc_endpoint_and_radius_lock_and_cancel_is_nonmutating() {
        let mut sk = Sketch::new();
        let c = sk.add_point([0., 0.]);
        let a = sk.add_point([5., 0.]);
        let b = sk.add_point([0., 5.]);
        let arc = sk.add_arc(c, a, b);
        assert!(tangent_source(&sk, Some(c), Some(c)).is_none());
        let mut s = session();
        s.click(&mut sk, Some(b), [0., 5.], Pos2::ZERO, &eval, false);
        let (centre, _, ccw) = tangent_geometry(&sk, b, arc, [-5., 0.], None).unwrap();
        assert!(dist(centre, [0., 0.]) < 1e-7 && ccw);
        s.preview(&sk, None, [-5., 0.]);
        let h = s.hud.as_mut().unwrap();
        h.fields[0].text = "3 mm".into();
        h.fields[0].locked = Some(3.);
        let out = s.click(&mut sk, None, [-5., 0.], Pos2::ZERO, &eval, false);
        assert!(out.changed, "{:?}", out.message);
        assert!(sk
            .constraints
            .values()
            .any(|r| matches!(r.c,Constraint::Radius {value,..} if (value-3.).abs()<1e-6)));
        assert!(sk.solve().converged);
        let count = sk.entities.len();
        s.click(&mut sk, Some(a), [5., 0.], Pos2::ZERO, &eval, false);
        assert!(s.cancel());
        assert_eq!(sk.entities.len(), count);
    }
    #[test]
    fn rejects_free_points_and_collinear_endpoints_without_mutation() {
        let mut sk = Sketch::new();
        let p = sk.add_point([0., 0.]);
        let mut s = session();
        let n = sk.entities.len();
        assert!(s
            .click(&mut sk, Some(p), [0., 0.], Pos2::ZERO, &eval, false)
            .message
            .is_some());
        assert_eq!(sk.entities.len(), n);
        let a = sk.add_point([-5., 0.]);
        sk.add_line(a, p);
        s.click(&mut sk, Some(p), [0., 0.], Pos2::ZERO, &eval, false);
        let n = sk.entities.len();
        assert!(
            !s.click(&mut sk, None, [5., 0.], Pos2::ZERO, &eval, false)
                .changed
        );
        assert_eq!(sk.entities.len(), n);
    }
}

#[cfg(test)]
mod midpoint_shortcut_tests {
    use super::*;
    fn session(tool: Tool) -> SketchSession {
        let mut s = SketchSession::new(
            1,
            Frame {
                origin: [0.; 3],
                x: [1., 0., 0.],
                y: [0., 1., 0.],
            },
            1,
        );
        s.set_tool(tool);
        s
    }
    fn eval(t: &str, u: Unit) -> Result<f64, String> {
        refusion_model::expr::eval(t, u, &std::collections::HashMap::new())
            .map_err(|e| e.to_string())
    }
    fn snap(s: &mut SketchSession, sk: &Sketch, p: [f64; 2], shift: bool) {
        s.update_midpoint_snap(sk, p, shift, |q| {
            Some(Pos2::new(q[0] as f32 * 10., q[1] as f32 * 10.))
        });
    }
    fn midpoint_pair(sk: &Sketch, line: EntityId) -> EntityId {
        sk.constraints
            .values()
            .find_map(|r| match r.c {
                Constraint::Midpoint(p, l) if l == line => Some(p),
                _ => None,
            })
            .unwrap()
    }
    #[test]
    fn shift_snap_is_screen_sized_and_preserves_select_multi_selection() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0., 0.], [20., 0.]);
        let mut s = session(Tool::Point);
        s.grid_snap = Some(3.0);
        snap(&mut s, &sk, [10.7, 0.], true);
        assert_eq!(s.midpoint_target, Some(line));
        assert_eq!(s.snap(&sk, Some(line), [10.7, 0.]).0, [10., 0.]);
        snap(&mut s, &sk, [10.8, 0.], true);
        assert_eq!(s.midpoint_target, None);
        snap(&mut s, &sk, [10., 0.], false);
        assert_eq!(s.midpoint_target, None);
        s.set_tool(Tool::Select);
        snap(&mut s, &sk, [10., 0.], true);
        assert_eq!(s.midpoint_target, None);
        s.select_at(Some(line), true);
        assert_eq!(s.selection, vec![SkSel::Ent(line)]);
        s.select_at(Some(line), true);
        assert!(s.selection.is_empty());
    }
    #[test]
    fn line_endpoint_has_a_persistent_midpoint_and_does_not_infer_it_off_target() {
        let mut sk = Sketch::new();
        let source = sk.add_line_pts([0., 0.], [20., 0.]);
        let mut s = session(Tool::Line);
        s.click(&mut sk, None, [3., 9.], Pos2::ZERO, &eval, false);
        snap(&mut s, &sk, [10.2, 0.1], true);
        let (_, pts) = s.preview(&sk, Some(source), [10.2, 0.1]);
        assert_eq!(pts, vec![[10., 0.]]);
        assert!(
            s.click(&mut sk, Some(source), [10.2, 0.1], Pos2::ZERO, &eval, true)
                .changed
        );
        let p = midpoint_pair(&sk, source);
        assert!(dist(sk.pos(p), [10., 0.]) < 1e-7);
        let (a, b) = sk.line_pts(source).unwrap();
        sk.set_pos(b, [30., 4.]);
        sk.solve();
        let (a, b) = (sk.pos(a), sk.pos(b));
        assert!(dist(sk.pos(p), [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]) < 1e-6);
    }
    #[test]
    fn circle_center_remembers_midpoint_after_shift_is_released() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0., 0.], [20., 0.]);
        let mut s = session(Tool::Circle);
        snap(&mut s, &sk, [10., 0.], true);
        assert!(
            !s.click(&mut sk, Some(line), [10., 0.], Pos2::ZERO, &eval, true)
                .changed
        );
        assert!(
            s.click(&mut sk, None, [10., 3.], Pos2::ZERO, &eval, false)
                .changed
        );
        let p = midpoint_pair(&sk, line);
        assert!(sk
            .entities
            .values()
            .any(|e| matches!(e.geom, Geom::Circle {c, ..} if c==p)));
    }
    #[test]
    fn coincident_shift_pick_commits_atomically_and_failed_constraint_leaves_no_points() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0., 0.], [20., 0.]);
        let point = sk.add_point([10., 4.]);
        let count = sk.entities.len();
        let mut s = session(Tool::Constrain(ConKind::Coincident));
        snap(&mut s, &sk, [10., 0.], true);
        assert!(
            !s.click(&mut sk, Some(line), [10., 0.], Pos2::ZERO, &eval, true)
                .changed
        );
        assert_eq!(sk.entities.len(), count);
        assert!(
            s.click(&mut sk, Some(point), [10., 4.], Pos2::ZERO, &eval, false)
                .changed
        );
        let p = midpoint_pair(&sk, line);
        assert!(dist(sk.pos(p), line_midpoint(&sk, line).unwrap()) < 1e-6);
        let before = sk.entities.len();
        s.set_tool(Tool::Constrain(ConKind::Parallel));
        snap(&mut s, &sk, [10., 0.], true);
        s.click(&mut sk, Some(line), [10., 0.], Pos2::ZERO, &eval, true);
        let out = s.click(&mut sk, Some(line), [5., 0.], Pos2::ZERO, &eval, false);
        assert!(!out.changed && out.message.is_some());
        assert_eq!(sk.entities.len(), before);
    }
    #[test]
    fn midpoint_dimension_cancel_is_clean_and_commit_serializes_persistent_reference() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0., 0.], [20., 0.]);
        let p = sk.add_point([10., 6.]);
        let count = sk.entities.len();
        let mut s = session(Tool::Dimension);
        for cancel in [true, false] {
            snap(&mut s, &sk, [10., 0.], true);
            s.click(&mut sk, Some(line), [10., 0.], Pos2::ZERO, &eval, true);
            s.click(&mut sk, Some(p), [10., 6.], Pos2::ZERO, &eval, false);
            s.click(&mut sk, None, [15., 3.], Pos2::ZERO, &eval, false);
            assert!(s.dim_edit.is_some());
            assert_eq!(sk.entities.len(), count);
            if cancel {
                s.cancel();
                assert_eq!(sk.entities.len(), count);
                assert!(sk.constraints.is_empty());
            } else {
                s.dim_edit.as_mut().unwrap().text = "8 mm".into();
                assert!(s.commit_dim(&mut sk, &eval).changed);
                assert_eq!(sk.entities.len(), count + 1);
                let encoded = serde_json::to_string(&sk).unwrap();
                let restored: Sketch = serde_json::from_str(&encoded).unwrap();
                assert_eq!(midpoint_pair(&restored, line), midpoint_pair(&sk, line));
                let mid = midpoint_pair(&sk, line);
                assert!(sk
                    .constraints
                    .values()
                    .any(|r| r.c.entities().contains(&mid) && r.c.is_dimension()));
                assert!(sk.constraints.values().any(
                    |r| matches!(r.c,Constraint::Distance {value,..} if (value-8.).abs()<1e-9)
                ));
            }
        }
    }
    #[test]
    fn typed_line_length_takes_priority_over_midpoint_snap() {
        let mut sk = Sketch::new();
        let line = sk.add_line_pts([0., 10.], [20., 10.]);
        let mut s = session(Tool::Line);
        s.click(&mut sk, None, [0., 0.], Pos2::ZERO, &eval, false);
        s.preview(&sk, None, [10., 10.]);
        let hud = s.hud.as_mut().unwrap();
        hud.fields[0].text = "5".into();
        hud.edit(0, &eval);
        snap(&mut s, &sk, [10., 10.], true);
        assert!(
            s.click(&mut sk, Some(line), [10., 10.], Pos2::ZERO, &eval, true)
                .changed
        );
        assert!(!sk
            .constraints
            .values()
            .any(|r| matches!(r.c, Constraint::Midpoint(..))));
        assert!(sk
            .constraints
            .values()
            .any(|r| matches!(r.c,Constraint::Length {value,..} if (value-5.).abs()<1e-9)));
    }
}

#[cfg(test)]
mod construction_input_tests {
    use super::*;
    #[test]
    fn palette_construction_applies_only_to_new_drawing_geometry() {
        let mut sketch = Sketch::new();
        let original = sketch.add_rectangle([20., 20.], [30., 30.]);
        let mut session = SketchSession::new(
            1,
            Frame {
                origin: [0.; 3],
                x: [1., 0., 0.],
                y: [0., 1., 0.],
            },
            1,
        );
        session.set_tool(Tool::Rectangle);
        session.construction = true;
        let eval = |s: &str, unit| {
            refusion_model::expr::eval(s, unit, &std::collections::HashMap::new())
                .map_err(|e| e.to_string())
        };
        session.click(&mut sketch, None, [0., 0.], Pos2::ZERO, &eval, false);
        assert!(
            session
                .click(&mut sketch, None, [10., 10.], Pos2::ZERO, &eval, false)
                .changed
        );
        assert!(original.iter().all(|id| !sketch.entities[id].construction));
        assert_eq!(refusion_model::sketch::find_profiles(&sketch).len(), 1);
        session.construction = false;
        session.set_tool(Tool::Circle);
        session.click(&mut sketch, None, [40., 40.], Pos2::ZERO, &eval, false);
        session.click(&mut sketch, None, [45., 40.], Pos2::ZERO, &eval, false);
        assert_eq!(refusion_model::sketch::find_profiles(&sketch).len(), 2);
    }
}
