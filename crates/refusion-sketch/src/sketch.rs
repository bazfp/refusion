use crate::solver::{Analysis, Eq, Pt, Rad, SolveOptions, SolveReport, System, P};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConstraintId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Geom {
    Point(Pt),
    Line {
        a: EntityId,
        b: EntityId,
    },
    Circle {
        c: EntityId,
        r: P,
    },
    /// Counter-clockwise from `s` to `e` around `c`.
    Arc {
        c: EntityId,
        s: EntityId,
        e: EntityId,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub geom: Geom,
    pub construction: bool,
    pub projected: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimKind {
    Aligned,
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    Coincident(EntityId, EntityId),
    Horizontal(EntityId),
    Vertical(EntityId),
    HorizontalPts(EntityId, EntityId),
    VerticalPts(EntityId, EntityId),
    Parallel(EntityId, EntityId),
    Perpendicular(EntityId, EntityId),
    Tangent(EntityId, EntityId),
    Equal(EntityId, EntityId),
    Concentric(EntityId, EntityId),
    Collinear(EntityId, EntityId),
    Midpoint(EntityId, EntityId),
    Symmetric(EntityId, EntityId, EntityId),
    Fix(EntityId),
    Offset {
        a: EntityId,
        b: EntityId,
        value: f64,
    },
    Distance {
        a: EntityId,
        b: EntityId,
        kind: DimKind,
        value: f64,
    },
    Length {
        line: EntityId,
        value: f64,
    },
    Radius {
        e: EntityId,
        value: f64,
    },
    Diameter {
        e: EntityId,
        value: f64,
    },
    /// Unsigned angle between two lines, radians.
    Angle {
        a: EntityId,
        b: EntityId,
        value: f64,
    },
}

impl Constraint {
    pub fn is_dimension(&self) -> bool {
        matches!(
            self,
            Constraint::Distance { .. }
                | Constraint::Offset { .. }
                | Constraint::Length { .. }
                | Constraint::Radius { .. }
                | Constraint::Diameter { .. }
                | Constraint::Angle { .. }
        )
    }

    pub fn entities(&self) -> Vec<EntityId> {
        use Constraint::*;
        match *self {
            Horizontal(a) | Vertical(a) | Fix(a) => vec![a],
            Length { line, .. } => vec![line],
            Radius { e, .. } | Diameter { e, .. } => vec![e],
            Coincident(a, b)
            | HorizontalPts(a, b)
            | VerticalPts(a, b)
            | Parallel(a, b)
            | Perpendicular(a, b)
            | Tangent(a, b)
            | Equal(a, b)
            | Concentric(a, b)
            | Collinear(a, b)
            | Midpoint(a, b)
            | Distance { a, b, .. }
            | Offset { a, b, .. }
            | Angle { a, b, .. } => vec![a, b],
            Symmetric(a, b, l) => vec![a, b, l],
        }
    }

    fn remap(&mut self, from: EntityId, to: EntityId) {
        use Constraint::*;
        let f = |e: &mut EntityId| {
            if *e == from {
                *e = to
            }
        };
        match self {
            Horizontal(a) | Vertical(a) | Fix(a) => f(a),
            Length { line, .. } => f(line),
            Radius { e, .. } | Diameter { e, .. } => f(e),
            Coincident(a, b)
            | HorizontalPts(a, b)
            | VerticalPts(a, b)
            | Parallel(a, b)
            | Perpendicular(a, b)
            | Tangent(a, b)
            | Equal(a, b)
            | Concentric(a, b)
            | Collinear(a, b)
            | Midpoint(a, b)
            | Distance { a, b, .. }
            | Offset { a, b, .. }
            | Angle { a, b, .. } => {
                f(a);
                f(b)
            }
            Symmetric(a, b, l) => {
                f(a);
                f(b);
                f(l)
            }
        }
    }

    /// Value of a dimension, if this is one.
    pub fn value(&self) -> Option<f64> {
        use Constraint::*;
        match *self {
            Offset { value, .. }
            | Distance { value, .. }
            | Length { value, .. }
            | Radius { value, .. }
            | Diameter { value, .. }
            | Angle { value, .. } => Some(value),
            _ => None,
        }
    }

    pub fn set_value(&mut self, v: f64) {
        use Constraint::*;
        match self {
            Offset { value, .. }
            | Distance { value, .. }
            | Length { value, .. }
            | Radius { value, .. }
            | Diameter { value, .. }
            | Angle { value, .. } => *value = v,
            _ => {}
        }
    }

    pub fn short_name(&self) -> &'static str {
        use Constraint::*;
        match self {
            Coincident(..) => "Coincident",
            Horizontal(..) | HorizontalPts(..) => "Horizontal",
            Vertical(..) | VerticalPts(..) => "Vertical",
            Parallel(..) => "Parallel",
            Perpendicular(..) => "Perpendicular",
            Tangent(..) => "Tangent",
            Equal(..) => "Equal",
            Concentric(..) => "Concentric",
            Collinear(..) => "Collinear",
            Midpoint(..) => "Midpoint",
            Symmetric(..) => "Symmetric",
            Fix(..) => "Fix",
            Offset { .. } => "Offset",
            Distance { .. } => "Distance",
            Length { .. } => "Length",
            Radius { .. } => "Radius",
            Diameter { .. } => "Diameter",
            Angle { .. } => "Angle",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintRec {
    pub c: Constraint,
    /// Driven (reference) dimension: displays a measured value, adds no equation.
    pub driven: bool,
    /// Optional parameter expression the model evaluates into the value.
    pub expr: Option<String>,
    /// Label placement offset for dimensions (sketch units, relative to the anchor).
    pub label: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub enum AddError {
    /// No solution exists with the new constraint.
    OverConstrained,
    /// The constraint adds no information (would make the system redundant).
    Redundant,
    Invalid(&'static str),
}

impl std::fmt::Display for AddError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddError::OverConstrained => write!(f, "This would over-constrain the sketch"),
            AddError::Redundant => write!(f, "The sketch is already constrained this way"),
            AddError::Invalid(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sketch {
    pub x: Vec<f64>,
    pub entities: BTreeMap<EntityId, Entity>,
    pub constraints: BTreeMap<ConstraintId, ConstraintRec>,
    next_entity: u32,
    next_constraint: u32,
    #[serde(skip)]
    pub analysis: Option<Analysis>,
}

impl Sketch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn geometry_snapshot(&self) -> Self {
        Self {
            x: self.x.clone(),
            entities: self.entities.clone(),
            next_entity: self.next_entity,
            ..Self::default()
        }
    }

    /// Validate persisted/public data before lowering it into indexed equations.
    pub fn validate_data(&self) -> Result<(), String> {
        if self.x.iter().any(|v| !v.is_finite()) {
            return Err("Sketch parameters must be finite".into());
        }
        if self
            .entities
            .keys()
            .next_back()
            .is_some_and(|id| id.0 >= self.next_entity)
            || self
                .constraints
                .keys()
                .next_back()
                .is_some_and(|id| id.0 >= self.next_constraint)
            || self.next_entity == u32::MAX
            || self.next_constraint == u32::MAX
        {
            return Err("Invalid sketch ID allocation counters".into());
        }
        for (&id, entity) in &self.entities {
            match entity.geom {
                Geom::Point(Pt(x, y)) if x >= self.x.len() || y >= self.x.len() => {
                    return Err(format!("Point {} has invalid parameter indices", id.0))
                }
                Geom::Circle { r, .. } if r >= self.x.len() || self.x[r] <= 1e-9 => {
                    return Err(format!("Circle {} has an invalid radius", id.0))
                }
                _ => {}
            }
            if self
                .defining_points_geom(&entity.geom)
                .iter()
                .any(|&p| !self.is_point(p))
            {
                return Err(format!(
                    "Entity {} references a missing/non-point entity",
                    id.0
                ));
            }
        }
        for (&id, rec) in &self.constraints {
            self.validate(&rec.c)
                .map_err(|e| format!("Constraint {}: {e}", id.0))?;
            if rec.c.value().is_some_and(|v| !v.is_finite())
                || rec.label.iter().any(|v| !v.is_finite())
            {
                return Err(format!("Constraint {} contains non-finite values", id.0));
            }
        }
        for (&id, entity) in &self.entities {
            if matches!(entity.geom, Geom::Arc { .. })
                && self.radius(id).is_none_or(|r| !r.is_finite() || r <= 1e-9)
            {
                return Err(format!("Arc {} has an invalid radius", id.0));
            }
        }
        Ok(())
    }

    fn param(&mut self, v: f64) -> P {
        self.x.push(v);
        self.x.len() - 1
    }

    fn insert(&mut self, geom: Geom) -> EntityId {
        let id = EntityId(self.next_entity);
        self.next_entity += 1;
        self.entities.insert(
            id,
            Entity {
                geom,
                construction: false,
                projected: false,
            },
        );
        self.analysis = None;
        id
    }

    // ---------------------------------------------------------------- creation

    pub fn add_point(&mut self, p: [f64; 2]) -> EntityId {
        let pt = Pt(self.param(p[0]), self.param(p[1]));
        self.insert(Geom::Point(pt))
    }

    pub fn add_line(&mut self, a: EntityId, b: EntityId) -> EntityId {
        self.insert(Geom::Line { a, b })
    }

    pub fn add_line_pts(&mut self, a: [f64; 2], b: [f64; 2]) -> EntityId {
        let (a, b) = (self.add_point(a), self.add_point(b));
        self.add_line(a, b)
    }

    pub fn add_circle(&mut self, c: EntityId, r: f64) -> EntityId {
        let r = self.param(r);
        self.insert(Geom::Circle { c, r })
    }

    pub fn add_arc(&mut self, c: EntityId, s: EntityId, e: EntityId) -> EntityId {
        self.insert(Geom::Arc { c, s, e })
    }

    /// Two-corner rectangle: four lines sharing corners, two horizontal, two vertical.
    pub fn add_rectangle(&mut self, p0: [f64; 2], p1: [f64; 2]) -> [EntityId; 4] {
        let c = [
            self.add_point(p0),
            self.add_point([p1[0], p0[1]]),
            self.add_point(p1),
            self.add_point([p0[0], p1[1]]),
        ];
        let l = [
            self.add_line(c[0], c[1]),
            self.add_line(c[1], c[2]),
            self.add_line(c[2], c[3]),
            self.add_line(c[3], c[0]),
        ];
        for (i, &line) in l.iter().enumerate() {
            let k = if i % 2 == 0 {
                Constraint::Horizontal(line)
            } else {
                Constraint::Vertical(line)
            };
            self.push_constraint(k, false);
        }
        l
    }

    /// Axis-aligned rectangle centred on an existing point. Construction
    /// diagonals keep the centre associative when dimensions or corners move.
    pub fn add_center_rectangle(&mut self, center: EntityId, corner: [f64; 2]) -> [EntityId; 4] {
        let c = self.pos(center);
        let opposite = [2.0 * c[0] - corner[0], 2.0 * c[1] - corner[1]];
        let lines = self.add_rectangle(opposite, corner);
        let a = self.line_pts(lines[0]).unwrap();
        let b = self.line_pts(lines[2]).unwrap();
        for (p, q) in [(a.0, b.0), (a.1, b.1)] {
            let diagonal = self.add_line(p, q);
            self.entities.get_mut(&diagonal).unwrap().construction = true;
            self.push_constraint(Constraint::Midpoint(center, diagonal), false);
        }
        lines
    }

    // ---------------------------------------------------------------- queries

    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    pub fn is_point(&self, id: EntityId) -> bool {
        matches!(self.entities.get(&id).map(|e| e.geom), Some(Geom::Point(_)))
    }

    pub fn point_pt(&self, id: EntityId) -> Option<Pt> {
        match self.entities.get(&id)?.geom {
            Geom::Point(p) => Some(p),
            _ => None,
        }
    }

    pub fn pos(&self, id: EntityId) -> [f64; 2] {
        let p = self.point_pt(id).expect("not a point");
        [self.x[p.0], self.x[p.1]]
    }

    pub fn set_pos(&mut self, id: EntityId, v: [f64; 2]) {
        let p = self.point_pt(id).expect("not a point");
        self.x[p.0] = v[0];
        self.x[p.1] = v[1];
    }

    /// Radius of a circle or arc.
    pub fn radius(&self, id: EntityId) -> Option<f64> {
        match self.entities.get(&id)?.geom {
            Geom::Circle { r, .. } => Some(self.x[r]),
            Geom::Arc { c, s, .. } => {
                let (c, s) = (self.pos(c), self.pos(s));
                Some(((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2)).sqrt())
            }
            _ => None,
        }
    }

    pub fn set_radius(&mut self, id: EntityId, v: f64) {
        if let Some(Geom::Circle { r, .. }) = self.entities.get(&id).map(|e| e.geom) {
            self.x[r] = v;
        }
    }

    pub fn center(&self, id: EntityId) -> Option<EntityId> {
        match self.entities.get(&id)?.geom {
            Geom::Circle { c, .. } | Geom::Arc { c, .. } => Some(c),
            _ => None,
        }
    }

    pub fn line_pts(&self, id: EntityId) -> Option<(EntityId, EntityId)> {
        match self.entities.get(&id)?.geom {
            Geom::Line { a, b } => Some((a, b)),
            _ => None,
        }
    }

    /// The point entities that define an entity (for points: itself).
    pub fn defining_points(&self, id: EntityId) -> Vec<EntityId> {
        match self.entities.get(&id).map(|e| e.geom) {
            Some(Geom::Point(_)) => vec![id],
            Some(Geom::Line { a, b }) => vec![a, b],
            Some(Geom::Circle { c, .. }) => vec![c],
            Some(Geom::Arc { c, s, e }) => vec![c, s, e],
            None => vec![],
        }
    }

    /// Parameters owned by an entity (including its defining points).
    pub fn entity_params(&self, id: EntityId) -> Vec<P> {
        let mut out = Vec::new();
        for p in self.defining_points(id) {
            if let Some(Pt(x, y)) = self.point_pt(p) {
                out.push(x);
                out.push(y);
            }
        }
        if let Some(Geom::Circle { r, .. }) = self.entities.get(&id).map(|e| e.geom) {
            out.push(r);
        }
        out
    }

    /// Entities that reference a point (lines, circles, arcs).
    pub fn users_of(&self, pt: EntityId) -> Vec<EntityId> {
        self.entities
            .iter()
            .filter(|(&id, e)| id != pt && self.defining_points_geom(&e.geom).contains(&pt))
            .map(|(&id, _)| id)
            .collect()
    }

    fn defining_points_geom(&self, g: &Geom) -> Vec<EntityId> {
        match *g {
            Geom::Point(_) => vec![],
            Geom::Line { a, b } => vec![a, b],
            Geom::Circle { c, .. } => vec![c],
            Geom::Arc { c, s, e } => vec![c, s, e],
        }
    }

    // ---------------------------------------------------------------- editing

    fn push_constraint(&mut self, c: Constraint, driven: bool) -> ConstraintId {
        let id = ConstraintId(self.next_constraint);
        self.next_constraint += 1;
        self.constraints.insert(
            id,
            ConstraintRec {
                c,
                driven,
                expr: None,
                label: [0.0, 0.0],
            },
        );
        self.analysis = None;
        id
    }

    pub fn add_constraint(&mut self, c: Constraint) -> Result<Option<ConstraintId>, AddError> {
        self.validate(&c)?;
        // Point–point coincidence: merge the two points into one shared point.
        if let Constraint::Coincident(a, b) = c {
            if self.is_point(a) && self.is_point(b) {
                if a == b {
                    return Ok(None);
                }
                let mut trial = self.clone();
                trial.merge_points(a, b);
                let base = self.analyze_now();
                let rep = trial.solve();
                if !rep.converged {
                    return Err(AddError::OverConstrained);
                }
                if trial.analyze_now().redundant > base.redundant {
                    return Err(AddError::Redundant);
                }
                *self = trial;
                return Ok(None);
            }
        }
        let base = self.analyze_now();
        let fixing = matches!(c, Constraint::Fix(_));
        let mut trial = self.clone();
        let id = trial.push_constraint(c, false);
        let rep = trial.solve();
        if !rep.converged {
            return Err(AddError::OverConstrained);
        }
        let after = trial.analyze_now();
        // Fix can make an arc's implicit equal-radius equation constant.
        // It is useful whenever it removes freedom, despite that zero row.
        if if fixing {
            after.dof >= base.dof
        } else {
            after.redundant > base.redundant
        } {
            return Err(AddError::Redundant);
        }
        *self = trial;
        Ok(Some(id))
    }

    /// Add a driven (reference) dimension. Never fails.
    pub fn add_driven(&mut self, mut c: Constraint) -> ConstraintId {
        if let Some(v) = self.measure(&c) {
            c.set_value(v);
        }
        self.push_constraint(c, true)
    }

    /// Change a driving dimension's value and re-solve. Reverts if infeasible.
    pub fn set_dimension(&mut self, id: ConstraintId, value: f64) -> Result<(), AddError> {
        let saved = self.clone();
        match self.constraints.get_mut(&id) {
            Some(rec) => rec.c.set_value(value),
            None => return Err(AddError::Invalid("no such dimension")),
        }
        if !self.solve().converged {
            *self = saved;
            return Err(AddError::OverConstrained);
        }
        Ok(())
    }

    pub fn remove_constraint(&mut self, id: ConstraintId) {
        self.constraints.remove(&id);
        self.analysis = None;
    }

    /// Delete an entity, constraints on it, and points nothing else uses.
    pub fn remove_entity(&mut self, id: EntityId) {
        let Some(e) = self.entities.remove(&id) else {
            return;
        };
        self.constraints
            .retain(|_, r| !r.c.entities().contains(&id));
        for p in self.defining_points_geom(&e.geom) {
            if self.users_of(p).is_empty() {
                self.entities.remove(&p);
                self.constraints.retain(|_, r| !r.c.entities().contains(&p));
            }
        }
        self.analysis = None;
    }

    /// Replace point `b` by point `a` everywhere. If `b` is projected (fixed)
    /// it survives instead so the fixed position wins.
    pub fn merge_points(&mut self, a: EntityId, b: EntityId) -> EntityId {
        let (keep, gone) = if self.entities[&b].projected && !self.entities[&a].projected {
            (b, a)
        } else {
            (a, b)
        };
        for e in self.entities.values_mut() {
            let f = |x: &mut EntityId| {
                if *x == gone {
                    *x = keep
                }
            };
            match &mut e.geom {
                Geom::Point(_) => {}
                Geom::Line { a, b } => {
                    f(a);
                    f(b)
                }
                Geom::Circle { c, .. } => f(c),
                Geom::Arc { c, s, e } => {
                    f(c);
                    f(s);
                    f(e)
                }
            }
        }
        for r in self.constraints.values_mut() {
            r.c.remap(gone, keep);
        }
        self.entities.remove(&gone);
        // Drop constraints that became trivial (e.g. Coincident(p, p)).
        self.constraints.retain(|_, r| {
            let es = r.c.entities();
            !(es.len() == 2 && es[0] == es[1])
        });
        self.analysis = None;
        keep
    }

    fn validate(&self, c: &Constraint) -> Result<(), AddError> {
        use Constraint::*;
        for e in c.entities() {
            if !self.entities.contains_key(&e) {
                return Err(AddError::Invalid("unknown entity"));
            }
        }
        let kind = |e: EntityId| self.entities[&e].geom;
        let is_line = |e| matches!(kind(e), Geom::Line { .. });
        let is_round = |e| matches!(kind(e), Geom::Circle { .. } | Geom::Arc { .. });
        let is_pt = |e| matches!(kind(e), Geom::Point(_));
        let ok = match *c {
            Horizontal(a) | Vertical(a) => is_line(a),
            HorizontalPts(a, b) | VerticalPts(a, b) => is_pt(a) && is_pt(b),
            Parallel(a, b) | Perpendicular(a, b) | Collinear(a, b) => is_line(a) && is_line(b),
            Angle { a, b, .. } => is_line(a) && is_line(b),
            Tangent(a, b) => {
                (is_line(a) && is_round(b))
                    || (is_round(a) && is_line(b))
                    || (is_round(a) && is_round(b))
            }
            Equal(a, b) => (is_line(a) && is_line(b)) || (is_round(a) && is_round(b)),
            Concentric(a, b) => is_round(a) && is_round(b),
            Midpoint(p, l) => is_pt(p) && is_line(l),
            Coincident(a, b) => is_pt(a) || is_pt(b),
            Symmetric(a, b, l) => {
                is_line(l)
                    && ((is_pt(a) && is_pt(b))
                        || (is_line(a) && is_line(b))
                        || (is_round(a) && is_round(b)))
            }
            Fix(_) => true,
            Offset { a, b, value } => {
                value.is_finite() && ((is_line(a) && is_line(b)) || (is_round(a) && is_round(b)))
            }
            Distance { a, b, .. } => {
                (is_pt(a) && is_pt(b))
                    || (is_pt(a) && is_line(b))
                    || (is_line(a) && is_pt(b))
                    || (is_line(a) && is_line(b))
            }
            Length { line, .. } => is_line(line),
            Radius { e, .. } | Diameter { e, .. } => is_round(e),
        };
        if ok {
            Ok(())
        } else {
            Err(AddError::Invalid(
                "constraint does not apply to this selection",
            ))
        }
    }

    // ---------------------------------------------------------------- lowering

    fn rad_of(&self, id: EntityId) -> Rad {
        match self.entities[&id].geom {
            Geom::Circle { r, .. } => Rad::Param(r),
            Geom::Arc { c, s, .. } => {
                Rad::Pts(self.point_pt(c).unwrap(), self.point_pt(s).unwrap())
            }
            _ => unreachable!("not round"),
        }
    }

    fn line_of(&self, id: EntityId) -> (Pt, Pt) {
        let (a, b) = self.line_pts(id).unwrap();
        (self.point_pt(a).unwrap(), self.point_pt(b).unwrap())
    }

    fn val(&self, p: Pt) -> [f64; 2] {
        [self.x[p.0], self.x[p.1]]
    }

    fn signed_line_dist(&self, p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
        (ux * (p[1] - a[1]) - uy * (p[0] - a[0])) / (ux * ux + uy * uy).sqrt().max(1e-300)
    }

    fn sign(v: f64) -> f64 {
        if v < 0.0 {
            -1.0
        } else {
            1.0
        }
    }

    /// Lower one constraint to primitive equations. Orientation choices
    /// (which side of a line, internal vs external tangency, which of the
    /// four line-angle branches) are taken from the current geometry, the way
    /// an interactive solver must to avoid flipping.
    fn lower(&self, c: &Constraint, out: &mut Vec<Eq>, fixed: &mut Vec<P>) {
        use Constraint::*;
        let pt = |e| self.point_pt(e).unwrap();
        match *c {
            Coincident(a, b) => {
                let (p, curve) = if self.is_point(a) { (a, b) } else { (b, a) };
                let p = pt(p);
                match self.entities[&curve].geom {
                    Geom::Point(q) => {
                        out.push(Eq::Lin {
                            a: p.0,
                            b: q.0,
                            k: 0.0,
                        });
                        out.push(Eq::Lin {
                            a: p.1,
                            b: q.1,
                            k: 0.0,
                        });
                    }
                    Geom::Line { .. } => {
                        let (a, b) = self.line_of(curve);
                        out.push(Eq::PtLine { p, a, b, d: 0.0 });
                    }
                    Geom::Circle { c, .. } | Geom::Arc { c, .. } => {
                        out.push(Eq::PtCircle {
                            p,
                            c: pt(c),
                            r: self.rad_of(curve),
                        });
                    }
                }
            }
            Horizontal(l) => {
                let (a, b) = self.line_of(l);
                out.push(Eq::Lin {
                    a: a.1,
                    b: b.1,
                    k: 0.0,
                });
            }
            Vertical(l) => {
                let (a, b) = self.line_of(l);
                out.push(Eq::Lin {
                    a: a.0,
                    b: b.0,
                    k: 0.0,
                });
            }
            HorizontalPts(a, b) => out.push(Eq::Lin {
                a: pt(a).1,
                b: pt(b).1,
                k: 0.0,
            }),
            VerticalPts(a, b) => out.push(Eq::Lin {
                a: pt(a).0,
                b: pt(b).0,
                k: 0.0,
            }),
            Parallel(l1, l2) => {
                let ((a1, a2), (b1, b2)) = (self.line_of(l1), self.line_of(l2));
                out.push(Eq::Parallel { a1, a2, b1, b2 });
            }
            Perpendicular(l1, l2) => {
                let ((a1, a2), (b1, b2)) = (self.line_of(l1), self.line_of(l2));
                out.push(Eq::Perp { a1, a2, b1, b2 });
            }
            Collinear(l1, l2) => {
                let ((a1, a2), (b1, b2)) = (self.line_of(l1), self.line_of(l2));
                out.push(Eq::PtLine {
                    p: b1,
                    a: a1,
                    b: a2,
                    d: 0.0,
                });
                out.push(Eq::PtLine {
                    p: b2,
                    a: a1,
                    b: a2,
                    d: 0.0,
                });
            }
            Tangent(e1, e2) => {
                let is_line = |e| matches!(self.entities[&e].geom, Geom::Line { .. });
                if is_line(e1) || is_line(e2) {
                    let (l, k) = if is_line(e1) { (e1, e2) } else { (e2, e1) };
                    let (a, b) = self.line_of(l);
                    let c = pt(self.center(k).unwrap());
                    if let Geom::Arc { s, e, .. } = self.entities[&k].geom {
                        let shared = [s, e].into_iter().find(|p| pt(*p) == a || pt(*p) == b);
                        if let Some(shared) = shared {
                            out.push(Eq::Perp {
                                a1: a,
                                a2: b,
                                b1: pt(shared),
                                b2: c,
                            });
                            return;
                        }
                    }
                    let side =
                        Self::sign(self.signed_line_dist(self.val(c), self.val(a), self.val(b)));
                    out.push(Eq::TanLineCircle {
                        a,
                        b,
                        c,
                        r: self.rad_of(k),
                        side,
                    });
                } else {
                    let (c1, c2) = (pt(self.center(e1).unwrap()), pt(self.center(e2).unwrap()));
                    if let (Geom::Arc { s: a, e: b, .. }, Geom::Arc { s: c, e: d, .. }) =
                        (self.entities[&e1].geom, self.entities[&e2].geom)
                    {
                        if let Some(shared) = [a, b].into_iter().find(|p| *p == c || *p == d) {
                            out.push(Eq::Parallel {
                                a1: pt(shared),
                                a2: c1,
                                b1: pt(shared),
                                b2: c2,
                            });
                            return;
                        }
                    }
                    let (r1, r2) = (self.radius(e1).unwrap(), self.radius(e2).unwrap());
                    let (p1, p2) = (self.val(c1), self.val(c2));
                    let d = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
                    let internal = d < r1.max(r2) - 1e-12 * (1.0 + r1.max(r2));
                    out.push(Eq::TanCircles {
                        c1,
                        r1: self.rad_of(e1),
                        c2,
                        r2: self.rad_of(e2),
                        internal,
                    });
                }
            }
            Equal(a, b) => {
                if let (Geom::Line { .. }, Geom::Line { .. }) =
                    (self.entities[&a].geom, self.entities[&b].geom)
                {
                    let ((a1, a2), (b1, b2)) = (self.line_of(a), self.line_of(b));
                    out.push(Eq::EqualLen { a1, a2, b1, b2 });
                } else {
                    out.push(Eq::EqualRad {
                        r1: self.rad_of(a),
                        r2: self.rad_of(b),
                    });
                }
            }
            Concentric(a, b) => {
                let (c1, c2) = (pt(self.center(a).unwrap()), pt(self.center(b).unwrap()));
                out.push(Eq::Lin {
                    a: c1.0,
                    b: c2.0,
                    k: 0.0,
                });
                out.push(Eq::Lin {
                    a: c1.1,
                    b: c2.1,
                    k: 0.0,
                });
            }
            Midpoint(p, l) => {
                let (a, b) = self.line_of(l);
                out.push(Eq::Mid {
                    p: pt(p),
                    a,
                    b,
                    axis: 0,
                });
                out.push(Eq::Mid {
                    p: pt(p),
                    a,
                    b,
                    axis: 1,
                });
            }
            Symmetric(e1, e2, l) => {
                let (a, b) = self.line_of(l);
                let pairs: Vec<(Pt, Pt)> = match (self.entities[&e1].geom, self.entities[&e2].geom)
                {
                    (Geom::Point(p), Geom::Point(q)) => vec![(p, q)],
                    (Geom::Line { .. }, Geom::Line { .. }) => {
                        let ((p1, p2), (q1, q2)) = (self.line_of(e1), self.line_of(e2));
                        // Pair endpoints the way they currently mirror best.
                        let mirror = |p: [f64; 2]| {
                            let (a, b) = (self.val(a), self.val(b));
                            let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
                            let t = ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / (ux * ux + uy * uy);
                            let f = [a[0] + t * ux, a[1] + t * uy];
                            [2.0 * f[0] - p[0], 2.0 * f[1] - p[1]]
                        };
                        let d = |u: [f64; 2], v: [f64; 2]| (u[0] - v[0]).hypot(u[1] - v[1]);
                        let m1 = mirror(self.val(p1));
                        if d(m1, self.val(q1)) <= d(m1, self.val(q2)) {
                            vec![(p1, q1), (p2, q2)]
                        } else {
                            vec![(p1, q2), (p2, q1)]
                        }
                    }
                    _ => {
                        out.push(Eq::EqualRad {
                            r1: self.rad_of(e1),
                            r2: self.rad_of(e2),
                        });
                        vec![(pt(self.center(e1).unwrap()), pt(self.center(e2).unwrap()))]
                    }
                };
                for (p, q) in pairs {
                    out.push(Eq::SymMid { p, q, a, b });
                    out.push(Eq::SymPerp { p, q, a, b });
                }
            }
            Offset { a, b, value } => {
                if matches!(self.entities[&a].geom, Geom::Line { .. }) {
                    let (p, q) = self.line_of(a);
                    let (r, s) = self.line_of(b);
                    for (end, v) in [(false, r), (true, s)] {
                        for axis in 0..2 {
                            out.push(Eq::OffsetPoint {
                                a: p,
                                b: q,
                                p: v,
                                end,
                                axis,
                                d: value,
                            });
                        }
                    }
                } else {
                    out.push(Eq::RadiusDiff {
                        r1: self.rad_of(b),
                        r2: self.rad_of(a),
                        d: value,
                    });
                    let (ca, cb) = (pt(self.center(a).unwrap()), pt(self.center(b).unwrap()));
                    if ca != cb {
                        out.push(Eq::Lin {
                            a: ca.0,
                            b: cb.0,
                            k: 0.0,
                        });
                        out.push(Eq::Lin {
                            a: ca.1,
                            b: cb.1,
                            k: 0.0,
                        });
                    }
                    if let (Geom::Arc { s: sa, e: ea, .. }, Geom::Arc { s: sb, e: eb, .. }) =
                        (self.entities[&a].geom, self.entities[&b].geom)
                    {
                        for (v, w) in [(sa, sb), (ea, eb)] {
                            out.push(Eq::Parallel {
                                a1: ca,
                                a2: pt(v),
                                b1: cb,
                                b2: pt(w),
                            });
                        }
                    }
                }
            }
            Fix(e) => fixed.extend(self.entity_params(e)),
            Distance { a, b, kind, value } => {
                let (pa, pb) = (self.is_point(a), self.is_point(b));
                if pa && pb {
                    let (p, q) = (pt(a), pt(b));
                    match kind {
                        DimKind::Aligned => out.push(Eq::Dist { p, q, d: value }),
                        DimKind::Horizontal => {
                            let s = Self::sign(self.x[q.0] - self.x[p.0]);
                            out.push(Eq::Lin {
                                a: q.0,
                                b: p.0,
                                k: s * value,
                            });
                        }
                        DimKind::Vertical => {
                            let s = Self::sign(self.x[q.1] - self.x[p.1]);
                            out.push(Eq::Lin {
                                a: q.1,
                                b: p.1,
                                k: s * value,
                            });
                        }
                    }
                } else {
                    // point–line, or line–line (parallel lines: measured from b's first point)
                    let (p, l) = if pa {
                        (pt(a), b)
                    } else if pb {
                        (pt(b), a)
                    } else {
                        (self.line_of(b).0, a)
                    };
                    let (la, lb) = self.line_of(l);
                    let s =
                        Self::sign(self.signed_line_dist(self.val(p), self.val(la), self.val(lb)));
                    out.push(Eq::PtLine {
                        p,
                        a: la,
                        b: lb,
                        d: s * value,
                    });
                }
            }
            Length { line, value } => {
                let (p, q) = self.line_of(line);
                out.push(Eq::Dist { p, q, d: value });
            }
            Radius { e, value } => out.push(Eq::RadiusIs {
                r: self.rad_of(e),
                d: value,
            }),
            Diameter { e, value } => out.push(Eq::RadiusIs {
                r: self.rad_of(e),
                d: value * 0.5,
            }),
            Angle { a, b, value } => {
                let ((a1, a2), (b1, b2)) = (self.line_of(a), self.line_of(b));
                let cur = self.signed_angle(a, b);
                let pi = std::f64::consts::PI;
                let candidates = [value, -value, pi - value, -(pi - value)];
                let theta = candidates
                    .into_iter()
                    .min_by(|x, y| (x - cur).abs().partial_cmp(&(y - cur).abs()).unwrap())
                    .unwrap();
                out.push(Eq::Angle {
                    a1,
                    a2,
                    b1,
                    b2,
                    theta,
                });
            }
        }
    }

    fn signed_angle(&self, a: EntityId, b: EntityId) -> f64 {
        let ((a1, a2), (b1, b2)) = (self.line_of(a), self.line_of(b));
        let (a1, a2, b1, b2) = (self.val(a1), self.val(a2), self.val(b1), self.val(b2));
        let (ux, uy) = (a2[0] - a1[0], a2[1] - a1[1]);
        let (wx, wy) = (b2[0] - b1[0], b2[1] - b1[1]);
        (ux * wy - uy * wx).atan2(ux * wx + uy * wy)
    }

    /// Current measured value of a dimension (used for driven dimensions and
    /// for pre-filling the dimension edit box).
    pub fn measure(&self, c: &Constraint) -> Option<f64> {
        use Constraint::*;
        let d = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
        Some(match *c {
            Offset { a, b, .. } => {
                if let (Some((p, q)), Some((v, _))) = (self.line_pts(a), self.line_pts(b)) {
                    self.signed_line_dist(self.pos(v), self.pos(p), self.pos(q))
                } else {
                    self.radius(b)? - self.radius(a)?
                }
            }
            Distance { a, b, kind, .. } => {
                let (pa, pb) = (self.is_point(a), self.is_point(b));
                if pa && pb {
                    let (p, q) = (self.pos(a), self.pos(b));
                    match kind {
                        DimKind::Aligned => d(p, q),
                        DimKind::Horizontal => (q[0] - p[0]).abs(),
                        DimKind::Vertical => (q[1] - p[1]).abs(),
                    }
                } else {
                    let (p, l) = if pa {
                        (self.pos(a), b)
                    } else if pb {
                        (self.pos(b), a)
                    } else {
                        (self.pos(self.line_pts(b)?.0), a)
                    };
                    let (la, lb) = self.line_pts(l)?;
                    self.signed_line_dist(p, self.pos(la), self.pos(lb)).abs()
                }
            }
            Length { line, .. } => {
                let (a, b) = self.line_pts(line)?;
                d(self.pos(a), self.pos(b))
            }
            Radius { e, .. } => self.radius(e)?,
            Diameter { e, .. } => 2.0 * self.radius(e)?,
            Angle { a, b, .. } => {
                let s = self.signed_angle(a, b).abs();
                s.min(std::f64::consts::PI - s)
            }
            _ => return None,
        })
    }

    /// Build the numeric system from the current entities and constraints.
    pub fn build_system(&self) -> System {
        let mut eqs = Vec::new();
        let mut fixed_list = Vec::new();
        let mut used = vec![false; self.x.len()];
        for (&id, e) in &self.entities {
            for p in self.entity_params(id) {
                used[p] = true;
            }
            if e.projected {
                fixed_list.extend(self.entity_params(id));
            }
            if let Geom::Arc { c, s, e: end } = e.geom {
                let (c, s, end) = (
                    self.point_pt(c).unwrap(),
                    self.point_pt(s).unwrap(),
                    self.point_pt(end).unwrap(),
                );
                eqs.push(Eq::EqualRad {
                    r1: Rad::Pts(c, s),
                    r2: Rad::Pts(c, end),
                });
            }
        }
        for rec in self.constraints.values().filter(|r| !r.driven) {
            self.lower(&rec.c, &mut eqs, &mut fixed_list);
        }
        let mut fixed: Vec<bool> = used.iter().map(|u| !u).collect();
        for p in fixed_list {
            fixed[p] = true;
        }
        System {
            x: self.x.clone(),
            fixed,
            eqs,
        }
    }

    pub fn solve(&mut self) -> SolveReport {
        if let Err(error) = self.validate_data() {
            return SolveReport {
                converged: false,
                iterations: 0,
                max_residual: f64::INFINITY,
                components: 0,
                error: Some(error),
            };
        }
        let mut sys = self.build_system();
        let mut rep = sys.solve(&SolveOptions::default());
        self.x = sys.x;
        if self.constraints.values().any(|r| {
            !r.driven
                && matches!(r.c,Constraint::Offset{value,..}if !value.is_finite()||value.abs()<1e-7)
        }) {
            rep.converged = false;
        }
        if self
            .entities
            .keys()
            .any(|&id| self.radius(id).is_some_and(|r| !r.is_finite() || r <= 1e-9))
        {
            rep.converged = false;
        }
        self.refresh_driven();
        self.analysis = None;
        rep
    }

    /// Drag an entity towards a sketch-space target (point: move it; line:
    /// translate it; circle: change radius when grabbing the rim).
    pub fn drag(&mut self, id: EntityId, grab: [f64; 2], target: [f64; 2]) -> SolveReport {
        if let Err(error) = self.validate_data() {
            return SolveReport {
                converged: false,
                iterations: 0,
                max_residual: f64::INFINITY,
                components: 0,
                error: Some(error),
            };
        }
        let mut targets = Vec::new();
        let delta = [target[0] - grab[0], target[1] - grab[1]];
        match self.entities.get(&id).map(|e| e.geom) {
            Some(Geom::Point(p)) => {
                targets.push((p.0, target[0]));
                targets.push((p.1, target[1]));
            }
            Some(Geom::Circle { c, r }) => {
                let cp = self.pos(c);
                targets.push((r, (target[0] - cp[0]).hypot(target[1] - cp[1])));
            }
            Some(_) => {
                for p in self.defining_points(id) {
                    let pt = self.point_pt(p).unwrap();
                    targets.push((pt.0, self.x[pt.0] + delta[0]));
                    targets.push((pt.1, self.x[pt.1] + delta[1]));
                }
            }
            None => {}
        }
        let mut sys = self.build_system();
        let mut rep = sys.solve_drag(
            &targets,
            &SolveOptions {
                max_iters: 25,
                tol: 1e-9,
            },
        );
        let previous = std::mem::replace(&mut self.x, sys.x);
        if self
            .entities
            .keys()
            .any(|&id| self.radius(id).is_some_and(|r| !r.is_finite() || r <= 1e-9))
        {
            self.x = previous;
            rep.converged = false;
            rep.error = Some("Dragging would create a zero or invalid radius".into());
        }
        self.refresh_driven();
        self.analysis = None;
        rep
    }

    fn refresh_driven(&mut self) {
        let ids: Vec<ConstraintId> = self
            .constraints
            .iter()
            .filter(|(_, r)| r.driven)
            .map(|(&i, _)| i)
            .collect();
        for id in ids {
            let c = self.constraints[&id].c.clone();
            if let Some(v) = self.measure(&c) {
                self.constraints.get_mut(&id).unwrap().c.set_value(v);
            }
        }
    }

    pub fn analyze_now(&self) -> Analysis {
        if let Err(error) = self.validate_data() {
            return Analysis {
                dof: self.x.len(),
                redundant: 0,
                determined: vec![false; self.x.len()],
                error: Some(error),
            };
        }
        self.build_system().analyze()
    }

    /// Cached analysis (recomputed after edits).
    pub fn analysis(&mut self) -> &Analysis {
        if self.analysis.is_none() {
            self.analysis = Some(self.analyze_now());
        }
        self.analysis.as_ref().unwrap()
    }

    pub fn is_fully_constrained(&mut self, id: EntityId) -> bool {
        let params = self.entity_params(id);
        let a = self.analysis();
        params
            .iter()
            .all(|&p| a.determined.get(p).copied().unwrap_or(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_tool_and_dimensions() {
        let mut s = Sketch::new();
        let l = s.add_rectangle([0.0, 0.0], [10.0, 5.0]);
        let (a, _) = s.line_pts(l[0]).unwrap();
        s.add_constraint(Constraint::Fix(a)).unwrap();
        s.add_constraint(Constraint::Length {
            line: l[0],
            value: 20.0,
        })
        .unwrap();
        assert_eq!(s.analyze_now().dof, 1);
        s.add_constraint(Constraint::Length {
            line: l[1],
            value: 8.0,
        })
        .unwrap();
        assert_eq!(s.analyze_now().dof, 0);
        // A third dimension on the opposite side is redundant → refused.
        assert_eq!(
            s.add_constraint(Constraint::Length {
                line: l[2],
                value: 20.0
            }),
            Err(AddError::Redundant)
        );
        // ...unless driven.
        s.add_driven(Constraint::Length {
            line: l[2],
            value: 0.0,
        });
        let (_, c) = s.line_pts(l[1]).unwrap();
        assert!((s.pos(c)[0] - 20.0).abs() < 1e-7 && (s.pos(c)[1] - 8.0).abs() < 1e-7);
        for id in l {
            assert!(s.is_fully_constrained(id));
        }
    }

    #[test]
    fn conflicting_dimension_is_rejected() {
        let mut s = Sketch::new();
        let l = s.add_line_pts([0.0, 0.0], [5.0, 0.0]);
        s.add_constraint(Constraint::Horizontal(l)).unwrap();
        let (a, b) = s.line_pts(l).unwrap();
        s.add_constraint(Constraint::Distance {
            a,
            b,
            kind: DimKind::Horizontal,
            value: 5.0,
        })
        .unwrap();
        let before = s.x.clone();
        assert!(s
            .add_constraint(Constraint::Distance {
                a,
                b,
                kind: DimKind::Vertical,
                value: 3.0
            })
            .is_err());
        assert_eq!(s.x, before);
    }

    #[test]
    fn shared_point_coincidence_merges() {
        let mut s = Sketch::new();
        let l1 = s.add_line_pts([0.0, 0.0], [5.0, 0.0]);
        let l2 = s.add_line_pts([5.2, 0.1], [5.0, 5.0]);
        let (_, b1) = s.line_pts(l1).unwrap();
        let (a2, _) = s.line_pts(l2).unwrap();
        s.add_constraint(Constraint::Coincident(b1, a2)).unwrap();
        assert_eq!(s.line_pts(l1).unwrap().1, s.line_pts(l2).unwrap().0);
        assert_eq!(s.entities.len(), 5);
    }

    #[test]
    fn tangent_arc_and_line() {
        let mut s = Sketch::new();
        let l = s.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        let (_, end) = s.line_pts(l).unwrap();
        let c = s.add_point([10.0, 3.5]);
        let e = s.add_point([13.0, 3.0]);
        let arc = s.add_arc(c, end, e);
        s.add_constraint(Constraint::Tangent(l, arc)).unwrap();
        s.add_constraint(Constraint::Radius { e: arc, value: 3.0 })
            .unwrap();
        let cp = s.pos(c);
        let (a, b) = s.line_pts(l).unwrap();
        let dist = s.signed_line_dist(cp, s.pos(a), s.pos(b)).abs();
        assert!((dist - 3.0).abs() < 1e-7, "{dist}");
    }

    #[test]
    fn symmetric_points_about_line() {
        let mut s = Sketch::new();
        let axis = s.add_line_pts([0.0, -10.0], [0.0, 10.0]);
        s.add_constraint(Constraint::Fix(axis)).unwrap();
        let p = s.add_point([-3.0, 1.0]);
        let q = s.add_point([4.0, 2.0]);
        s.add_constraint(Constraint::Symmetric(p, q, axis)).unwrap();
        let (pp, qq) = (s.pos(p), s.pos(q));
        assert!((pp[0] + qq[0]).abs() < 1e-7 && (pp[1] - qq[1]).abs() < 1e-7);
    }

    #[test]
    fn angle_dimension() {
        let mut s = Sketch::new();
        let l1 = s.add_line_pts([0.0, 0.0], [10.0, 0.0]);
        s.add_constraint(Constraint::Fix(l1)).unwrap();
        let (a, _) = s.line_pts(l1).unwrap();
        let p = s.add_point([7.0, 6.0]);
        let l2 = s.add_line(a, p);
        s.add_constraint(Constraint::Angle {
            a: l1,
            b: l2,
            value: 30f64.to_radians(),
        })
        .unwrap();
        let pp = s.pos(p);
        assert!((pp[1].atan2(pp[0]).to_degrees() - 30.0).abs() < 1e-6);
    }
}
