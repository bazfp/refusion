//! The refusion application: panels, timeline, commands and viewport interaction.

use crate::camera::{named_view, Camera};
use crate::gpu::{Gpu, PICK_WIN};
use crate::scene::{self, Pick, PickFilter, SceneInput};
use crate::sketch_mode::{ConKind, Outcome, SketchSession, Tool};
use eframe::egui::{self, Color32, Key, PointerButton, Pos2, RichText, Sense, Vec2};
use glam::{DVec2, DVec3};
use refusion_model::kernel::{self, Frame, Mesh};
use refusion_model::sketch::{EntityId, Geom, Profile, Sketch};
use refusion_model::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pending GPU work is not a background hit. Keep a resolved hover only while
/// its geometry/camera context is still current; a fresh empty result clears it.
fn resolve_hover_readback(
    previous: Option<Pick>,
    previous_scene: Option<u64>,
    scene: u64,
    result: Option<Option<Pick>>,
) -> Option<Pick> {
    result.unwrap_or_else(|| {
        if previous_scene == Some(scene) {
            previous
        } else {
            None
        }
    })
}

/// Only an intentional drag that begins a new drawing operation may place a
/// second point on release. Ordinary moving clicks and ongoing chains place once.
struct DrawingPress {
    at: Pos2,
    time: f64,
    feature: FeatureId,
    tool: Tool,
    started_operation: bool,
}

impl DrawingPress {
    fn finishes_drag(&self, s: &SketchSession, at: Pos2, time: f64, drag_stopped: bool) -> bool {
        self.started_operation
            && s.feature == self.feature
            && s.tool == self.tool
            && s.in_progress()
            && drag_stopped
            && time - self.time >= 0.2
            && at.distance(self.at) > 6.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NavigationMode {
    Orbit,
    Pan,
    Zoom,
}

enum Mode {
    Model,
    Sketch(SketchSession),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtentKind {
    ToFace,
    OneSide,
    Symmetric,
    TwoSides,
    ThroughAll,
}

#[derive(Clone)]
struct ExtrudeCmd {
    edit: Option<FeatureId>,
    sketch: Option<FeatureId>,
    profiles: Vec<ProfileRef>,
    faces: Vec<FaceRef>,
    op_user_set: bool,
    kind: ExtentKind,
    all_direction: AllDirection,
    half_length: bool,
    target: Option<FaceRef>,
    pick_target: bool,
    offset: String,
    taper: String,
    d1: String,
    d2: String,
    op: Operation,
}

#[derive(Clone)]
struct RevolveCmd {
    edit: Option<FeatureId>,
    sketch: Option<FeatureId>,
    profiles: Vec<ProfileRef>,
    faces: Vec<FaceRef>,
    direction: RevolveDirection,
    angle2: String,
    full: bool,
    project_axis: bool,
    axis: Option<Axis>,
    pick_axis: bool,
    angle: String,
    op: Operation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChamferType {
    Equal,
    TwoDistances,
    DistanceAngle,
}

#[derive(Clone)]
struct BlendCmd {
    edit: Option<FeatureId>,
    edges: Vec<EdgeRef>,
    radius: String,
    chamfer: Option<ChamferType>,
    d2: String,
    angle: String,
    show_preview: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SolidPick {
    Bodies,
    Plane,
    Axis1,
    Axis2,
    HoleCenter,
    HoleTarget,
}

#[derive(Clone)]
enum Command {
    None,
    NewSketch,
    Extrude(ExtrudeCmd),
    Revolve(RevolveCmd),
    Blend(BlendCmd),
    Measure(crate::measure::MeasureCmd),
    Solid {
        edit: Option<FeatureId>,
        kind: FeatureKind,
        pick: SolidPick,
    },
    OffsetPlane {
        edit: Option<FeatureId>,
        base: Option<PlaneRef>,
        distance: String,
    },
}

#[derive(PartialEq, Eq)]
struct ClickContext {
    epoch: u64,
    command: std::mem::Discriminant<Command>,
    sketch: Option<(FeatureId, Tool)>,
    marker: usize,
    selection_mode: u8,
    bodies: Vec<u64>,
}
struct QueuedClick {
    readback: crate::gpu::ClickPick,
    context: ClickContext,
    picks: Vec<Pick>,
    prio: Vec<u8>,
    cursor: Option<[f64; 2]>,
    at: Pos2,
    shift: bool,
}

/// An undo step: the document plus the timeline marker the *model* should be
/// at (snapshots taken while editing a sketch hold a rolled-back marker).
struct Snap {
    doc: Document,
    marker: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Refine {
    High,
    Medium,
    Low,
    Custom,
}

struct StlExport {
    refine: Refine,
    surface_dev: f64,
    normal_dev_deg: f64,
    binary: bool,
    unit: usize,
    per_body: bool,
    bodies: Vec<(BodyId, String, bool)>,
}

const STL_UNITS: [(&str, f32); 5] = [
    ("mm", 1.0),
    ("cm", 0.1),
    ("m", 0.001),
    ("in", 1.0 / 25.4),
    ("ft", 1.0 / 304.8),
];

impl Refine {
    /// (surface deviation as a fraction of the body diagonal, normal deviation in degrees)
    fn preset(self) -> (f64, f64) {
        match self {
            Refine::High => (0.0005, 5.0),
            Refine::Medium => (0.002, 10.0),
            Refine::Low | Refine::Custom => (0.01, 22.5),
        }
    }
}

struct Toast {
    text: String,
    at: Instant,
    error: bool,
}

pub struct RefusionApp {
    doc: Document,
    undo: Vec<Snap>,
    redo: Vec<Snap>,
    path: Option<PathBuf>,
    dirty: bool,
    cam: Camera,
    gpu: Option<Gpu>,
    meshes: HashMap<u64, Arc<Mesh>>,
    sketch_derived: crate::sketch_derived::Worker,
    sketch_analysis: crate::sketch_derived::AnalysisWorker,
    analysis_ticket: Option<(FeatureId, u64)>,
    profile_cache: scene::ProfileCache,
    mode: Mode,
    cmd: Command,
    compute: crate::compute::Worker,
    compute_ticket: u64,
    replay_pending: bool,
    preview_pending: bool,
    deferred_file: Option<(PathBuf, bool)>,
    pending_sketch: Option<(FeatureId, usize)>,
    replay_restore_marker: Option<usize>,
    preview: Option<Document>,
    preview_error: Option<String>,
    hover: Option<Pick>,
    hover_scene: Option<u64>,
    selection: Vec<Pick>,
    hidden_bodies: HashSet<BodyId>,
    hidden_sketches: HashSet<FeatureId>,
    hidden_planes: HashSet<FeatureId>,
    toasts: Vec<Toast>,
    show_params: bool,
    param_draft: Vec<Param>,
    picks: Vec<Pick>,
    prio: Vec<u8>,
    timeline_sel: Option<FeatureId>,
    press: Option<DrawingPress>,
    navigation: Option<NavigationMode>,
    cube_drag: bool,
    profiler: Option<crate::perf::ViewportProfiler>,
    mesh_lod: crate::mesh_lod::Worker,
    mesh_pending: HashMap<u64, f64>,
    mesh_failed: HashMap<u64, f64>,
    mesh_view_scale: f64,
    mesh_view_since: Instant,
    click_epoch: u64,
    pending_clicks: VecDeque<QueuedClick>,
    sketch_constrained: HashMap<EntityId, bool>,
    sketch_dof: usize,
    analysis_dirty: bool,
    last_frame: Instant,
    last_recompute_ms: f64,
    gpu_name: String,
    box_start: Option<Pos2>,
    hud_pos: Pos2,
    viewport_rect: egui::Rect,
    stl_export: Option<StlExport>,
    /// Sketch "Snap to grid" and fixed grid spacing (None = adaptive).
    grid_snap: bool,
    grid_fixed: Option<f64>,
    view: ViewOptions,
    file_job: Option<crate::file_load::Job>,
    ctx: egui::Context,
}

/// Sketch Palette display options.
struct ViewOptions {
    sketch_grid: bool,
    profiles: bool,
    points: bool,
    dimensions: bool,
    constraints: bool,
    construction: bool,
    projected: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        ViewOptions {
            sketch_grid: true,
            profiles: true,
            points: true,
            dimensions: true,
            constraints: true,
            construction: true,
            projected: true,
        }
    }
}

/// Tool panel actions, run after the panel is drawn.
#[derive(Clone, Copy)]
enum Action {
    NewSketch,
    Extrude,
    Revolve,
    Fillet,
    Chamfer,
    Measure,
    Params,
    OffsetPlane,
    Shell,
    Hole,
    Mirror,
    RectPattern,
    CircularPattern,
    AngledPlane,
    Insert,
    SelectModel,
    Tool(Tool),
    Constraint(ConKind),
    Construction,
    LookAt,
    FinishSketch,
}

impl Action {
    fn icon(self) -> crate::tool_icons::Icon {
        use crate::tool_icons::Icon as I;
        match self {
            Self::NewSketch => I::Sketch,
            Self::Extrude => I::Extrude,
            Self::Revolve => I::Revolve,
            Self::Hole => I::Hole,
            Self::Fillet => I::Fillet,
            Self::Chamfer => I::Chamfer,
            Self::Shell => I::Shell,
            Self::Mirror => I::Mirror,
            Self::RectPattern => I::RectPattern,
            Self::CircularPattern => I::CircularPattern,
            Self::OffsetPlane => I::OffsetPlane,
            Self::AngledPlane => I::AngledPlane,
            Self::Measure => I::Measure,
            Self::Insert => I::Insert,
            Self::Params => I::Params,
            Self::SelectModel => I::Select,
            Self::FinishSketch => I::Finish,
            Self::Construction => I::Line,
            Self::LookAt => I::Sketch,
            Self::Tool(t) => match t {
                Tool::Select => I::Select,
                Tool::Line => I::Line,
                Tool::Rectangle => I::Rectangle,
                Tool::CenterRectangle => I::CenterRectangle,
                Tool::Circle => I::Circle,
                Tool::Arc => I::Arc,
                Tool::ThreePointArc => I::ThreePointArc,
                Tool::TangentArc => I::TangentArc,
                Tool::Trim => I::Trim,
                Tool::Extend => I::Extend,
                Tool::Offset => I::Offset,
                Tool::Point => I::Point,
                Tool::Dimension => I::Dimension,
                Tool::Project => I::Project,
                Tool::Constrain(k) => Self::Constraint(k).icon(),
            },
            Self::Constraint(k) => match k {
                ConKind::HorizontalVertical => I::HorizontalVertical,
                ConKind::Coincident => I::Coincident,
                ConKind::Tangent => I::Tangent,
                ConKind::Equal => I::Equal,
                ConKind::Parallel => I::Parallel,
                ConKind::Perpendicular => I::Perpendicular,
                ConKind::Fix => I::Fix,
                ConKind::Midpoint => I::Midpoint,
                ConKind::Concentric => I::Concentric,
                ConKind::Collinear => I::Collinear,
                ConKind::Symmetric => I::Symmetry,
            },
        }
    }
}

/// A ribbon panel: promoted icon buttons over a "TITLE ▾" dropdown menu.
fn ribbon_panel(
    ui: &mut egui::Ui,
    title: &str,
    promoted: &[(&str, &str, bool, Action)],
    menu: impl FnOnce(&mut egui::Ui, &mut Option<Action>),
    act: &mut Option<Action>,
) {
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for &(label, key, active, a) in promoted {
                let r = ui.add_sized([34.0, 34.0], egui::Button::new("").selected(active));
                let icon = if label == "Press Pull" {
                    crate::tool_icons::Icon::PressPull
                } else {
                    a.icon()
                };
                crate::tool_icons::paint(
                    ui.painter(),
                    egui::Rect::from_center_size(r.rect.center(), Vec2::splat(25.)),
                    icon,
                );
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, label));
                let r = if key.is_empty() {
                    r.on_hover_text(label)
                } else {
                    r.on_hover_text(format!("{label}  ({key})"))
                };
                if r.clicked() {
                    *act = Some(a);
                }
            }
            if promoted.is_empty() {
                ui.allocate_exact_size(Vec2::new(34.0, 34.0), egui::Sense::hover());
            }
        });
        let response = ui
            .menu_button(
                RichText::new(format!("{title}    ")).small().strong(),
                |ui| menu(ui, act),
            )
            .response;
        let p = Pos2::new(response.rect.right() - 7., response.rect.center().y);
        ui.painter().add(egui::Shape::line(
            vec![
                p + Vec2::new(-2., -1.),
                p + Vec2::new(0., 1.),
                p + Vec2::new(2., -1.),
            ],
            egui::Stroke::new(1_f32, ui.visuals().text_color()),
        ));
    });
    ui.separator();
}

fn icon_submenu(
    ui: &mut egui::Ui,
    label: &str,
    icon: crate::tool_icons::Icon,
    menu: impl FnOnce(&mut egui::Ui),
) {
    let response = ui.menu_button(format!("      {label}"), menu).response;
    crate::tool_icons::paint(
        ui.painter(),
        egui::Rect::from_center_size(
            Pos2::new(response.rect.left() + 15., response.rect.center().y),
            Vec2::splat(20.),
        ),
        icon,
    );
}

fn menu_item(ui: &mut egui::Ui, label: &str, a: Action, act: &mut Option<Action>) {
    let key = match a {
        Action::NewSketch => "S",
        Action::Extrude if label == "Press Pull" => "Q",
        Action::Extrude => "E",
        Action::Hole => "H",
        Action::Fillet => "F",
        Action::Measure => "I",
        Action::Tool(Tool::Line) => "L",
        Action::Tool(Tool::Rectangle) => "R",
        Action::Tool(Tool::Circle) => "C",
        Action::Tool(Tool::Arc) => "A",
        Action::Tool(Tool::Dimension) => "D",
        Action::Tool(Tool::Project) => "P",
        Action::Tool(Tool::Trim) => "T",
        Action::Tool(Tool::Offset) => "O",
        _ => "",
    };
    let response = ui.add(
        egui::Button::new(format!("      {label}"))
            .shortcut_text(key)
            .min_size(Vec2::new(220., 28.)),
    );
    let icon = if label == "Press Pull" {
        crate::tool_icons::Icon::PressPull
    } else if label == "Window Selection" {
        crate::tool_icons::Icon::WindowSelect
    } else {
        a.icon()
    };
    crate::tool_icons::paint(
        ui.painter(),
        egui::Rect::from_center_size(
            Pos2::new(response.rect.left() + 15., response.rect.center().y),
            Vec2::splat(20.),
        ),
        icon,
    );
    if response.clicked() {
        *act = Some(a);
        ui.close_menu();
    }
}

impl RefusionApp {
    pub fn new(cc: &eframe::CreationContext<'_>, open: Option<PathBuf>) -> Self {
        let gpu = cc.wgpu_render_state.as_ref().map(|rs| Gpu::new(&rs.device));
        let gpu_name = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| {
                let i = rs.adapter.get_info();
                format!("{:?} · {}", i.backend, i.name)
            })
            .unwrap_or_default();
        eprintln!("refusion: rendering with {gpu_name}");
        // egui's bundled fonts lack the geometric symbols used for constraint
        // glyphs (∥ ⊥ ⌀ △ …); fall back to a system font that has them.
        let mut fonts = egui::FontDefinitions::default();
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
        ] {
            if let Ok(bytes) = std::fs::read(path) {
                fonts.font_data.insert(
                    "symbols".into(),
                    Arc::new(egui::FontData::from_owned(bytes)),
                );
                for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts
                        .families
                        .entry(fam)
                        .or_default()
                        .push("symbols".into());
                }
                break;
            }
        }
        cc.egui_ctx.set_fonts(fonts);
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(6.0, 5.0);
        cc.egui_ctx.set_style(style);
        let mut app = Self::empty(cc.egui_ctx.clone(), gpu, gpu_name);
        if let Some(p) = open {
            app.open_path(p);
        }
        app
    }

    fn empty(ctx: egui::Context, gpu: Option<Gpu>, gpu_name: String) -> Self {
        RefusionApp {
            doc: Document::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            path: None,
            dirty: false,
            cam: Camera::default(),
            gpu,
            meshes: HashMap::new(),
            sketch_derived: crate::sketch_derived::Worker::new(ctx.clone()),
            sketch_analysis: crate::sketch_derived::AnalysisWorker::new(ctx.clone()),
            analysis_ticket: None,
            profile_cache: scene::ProfileCache::default(),
            mode: Mode::Model,
            cmd: Command::None,
            compute: crate::compute::Worker::new(ctx.clone()),
            compute_ticket: 0,
            replay_pending: false,
            preview_pending: false,
            deferred_file: None,
            pending_sketch: None,
            replay_restore_marker: None,
            preview: None,
            preview_error: None,
            hover: None,
            hover_scene: None,
            selection: Vec::new(),
            hidden_bodies: HashSet::new(),
            hidden_sketches: HashSet::new(),
            hidden_planes: HashSet::new(),
            toasts: Vec::new(),
            show_params: false,
            param_draft: Vec::new(),
            picks: Vec::new(),
            prio: Vec::new(),
            timeline_sel: None,
            press: None,
            navigation: None,
            cube_drag: false,
            profiler: crate::perf::ViewportProfiler::from_env(),
            mesh_lod: crate::mesh_lod::Worker::new(ctx.clone()),
            mesh_pending: HashMap::new(),
            mesh_failed: HashMap::new(),
            mesh_view_scale: 0.,
            mesh_view_since: Instant::now(),
            click_epoch: 0,
            pending_clicks: VecDeque::new(),
            sketch_constrained: HashMap::new(),
            sketch_dof: 0,
            analysis_dirty: true,
            last_frame: Instant::now(),
            last_recompute_ms: 0.0,
            gpu_name,
            box_start: None,
            hud_pos: Pos2::ZERO,
            viewport_rect: egui::Rect::NOTHING,
            stl_export: None,
            grid_snap: false,
            grid_fixed: None,
            view: ViewOptions::default(),
            file_job: None,
            ctx: ctx.clone(),
        }
    }

    // ------------------------------------------------------------------ helpers

    fn toast(&mut self, text: impl Into<String>, error: bool) {
        self.toasts.push(Toast {
            text: text.into(),
            at: Instant::now(),
            error,
        });
    }

    fn snapshot(&mut self) {
        let snap = self.snap_of(self.doc.clone());
        self.push_undo(snap);
        self.dirty = true;
    }

    fn reset_home_view(&mut self) {
        // Replace the complete camera, including its pending animation, so a
        // later tick cannot restore an old pan/zoom or named-view transition.
        self.cam = Camera::default();
        self.navigation = None;
        self.cube_drag = false;
        self.press = None;
        self.box_start = None;
        self.hover = None;
        self.hover_scene = None;
        self.ctx.request_repaint();
    }

    fn new_document(&mut self) {
        self.snapshot();
        self.doc = Document::new();
        self.hidden_bodies.clear();
        self.hidden_sketches.clear();
        self.hidden_planes.clear();
        self.selection.clear();
        self.profile_cache = scene::ProfileCache::default();
        self.path = None;
        self.mode = Mode::Model;
        self.cancel_command();
        self.reset_home_view();
    }

    fn snap_of(&self, doc: Document) -> Snap {
        let marker = match &self.mode {
            Mode::Sketch(s) => s.restore_marker,
            Mode::Model => doc.marker,
        };
        Snap { doc, marker }
    }

    fn push_undo(&mut self, s: Snap) {
        self.click_epoch = self.click_epoch.wrapping_add(1);
        self.pending_clicks.clear();
        self.undo.push(s);
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.dirty = true;
    }

    fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            let old = std::mem::replace(&mut self.doc, s.doc);
            let cur = self.snap_of(old);
            self.redo.push(cur);
            self.after_history_jump(s.marker);
        }
    }

    fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            let old = std::mem::replace(&mut self.doc, s.doc);
            let cur = self.snap_of(old);
            self.undo.push(cur);
            self.after_history_jump(s.marker);
        }
    }

    fn after_history_jump(&mut self, model_marker: usize) {
        // Stay in sketch mode if that sketch still exists, else return to the model.
        let keep_sketch = match &self.mode {
            Mode::Sketch(s) => self
                .doc
                .index_of(s.feature)
                .is_some_and(|i| self.doc.marker == i + 1),
            Mode::Model => false,
        };
        if !keep_sketch {
            self.mode = Mode::Model;
            if self.doc.marker != model_marker {
                self.timed_recompute(move |d| d.set_marker(model_marker));
            }
        }
        if let Mode::Sketch(s) = &mut self.mode {
            s.cancel();
            s.selection.clear();
            s.drag_end();
        }
        self.analysis_dirty = true;
        self.cancel_command();
        self.selection.clear();
    }

    fn eval_fn(&self) -> impl Fn(&str, Unit) -> Result<f64, String> {
        let vals = self.doc.param_quantities();
        move |s: &str, u: Unit| {
            expr::eval(s, u, vals.as_ref().map_err(Clone::clone)?).map_err(|e| e.to_string())
        }
    }

    fn sketch_mut(doc: &mut Document, id: FeatureId) -> Option<&mut Sketch> {
        match &mut doc.feature_mut(id)?.kind {
            FeatureKind::Sketch(sf) => Some(&mut sf.sketch),
            _ => None,
        }
    }

    fn sketch_ref(doc: &Document, id: FeatureId) -> Option<&Sketch> {
        match &doc.feature(id)?.kind {
            FeatureKind::Sketch(sf) => Some(&sf.sketch),
            _ => None,
        }
    }

    fn timed_recompute(&mut self, f: impl FnOnce(&mut Document) + Send + 'static) {
        self.replay_pending = true;
        self.compute.submit(crate::compute::Request {
            ticket: self.compute_ticket,
            replay: true,
            doc: self.doc.clone(),
            cached: self.meshes.clone(),
            edit: Box::new(move |d| {
                f(d);
                None
            }),
        });
    }
    fn prune_meshes(&mut self) {
        let mut used: HashSet<_> = self.doc.cached_shape_uids().collect();
        for snap in self.undo.iter().chain(self.redo.iter()) {
            used.extend(snap.doc.cached_shape_uids());
        }
        if let Some(preview) = &self.preview {
            used.extend(preview.cached_shape_uids());
        }
        self.profile_cache
            .retain(&self.doc.features.iter().map(|f| f.id).collect());
        self.meshes.retain(|uid, _| used.contains(uid));
        self.mesh_failed.retain(|uid, _| used.contains(uid));
        self.mesh_pending.retain(|uid, _| used.contains(uid));
    }
    fn poll_compute(&mut self) {
        let mut changed = false;
        while let Some(result) = self.compute.poll() {
            changed = true;
            match result {
                Ok(result) if result.replay => {
                    self.replay_pending = false;
                    self.replay_restore_marker = None;
                    self.last_recompute_ms = result.ms;
                    self.doc = result.doc;
                    self.meshes.extend(result.meshes);
                    if let Some(error) = result.error {
                        self.toast(error, true);
                    }
                    if let Some((id, restore)) = self.pending_sketch.take() {
                        self.enter_sketch(id, restore);
                    }
                    if !matches!(self.cmd, Command::None) {
                        self.rebuild_preview();
                    }
                }
                Ok(result) if result.ticket == self.compute_ticket => {
                    self.preview_pending = false;
                    self.last_recompute_ms = result.ms;
                    self.preview_error = result.error;
                    self.preview = Some(result.doc);
                    self.meshes.extend(result.meshes);
                }
                Ok(_) => {}
                Err(e) if e.replay || e.ticket == self.compute_ticket => {
                    self.replay_pending = false;
                    self.pending_sketch = None;
                    self.replay_restore_marker = None;
                    self.preview_pending = false;
                    self.preview_error = Some(e.error.clone());
                    self.toast(e.error, true);
                }
                Err(_) => {}
            }
        }
        if changed {
            self.prune_meshes();
        }
    }

    // ------------------------------------------------------------------ file

    fn file_busy(&self) -> bool {
        self.replay_pending || self.file_job.as_ref().is_some_and(|j| !j.is_cancelled())
    }

    fn insert_path(&mut self, p: PathBuf) {
        self.start_file_load(p, true);
    }
    fn open_path(&mut self, p: PathBuf) {
        self.start_file_load(p, false);
    }

    fn start_file_load(&mut self, p: PathBuf, insert: bool) {
        if self.file_job.is_some() {
            self.toast(
                "A file worker is still running; please wait before starting another load",
                false,
            );
            return;
        }
        self.mode_to_model();
        if self.replay_pending {
            self.deferred_file = Some((p, insert));
            return;
        }
        self.selection.clear();
        self.show_params = false;
        self.stl_export = None;
        let base = insert.then(|| self.doc.clone());
        self.file_job = Some(crate::file_load::Job::start(
            p,
            base,
            self.meshes.clone(),
            self.ctx.clone(),
        ));
        self.ctx.request_repaint();
    }

    fn poll_file_load(&mut self) {
        use std::sync::mpsc::TryRecvError;
        let Some(job) = &self.file_job else { return };
        let result = match job.result.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                Err("File loading worker stopped unexpectedly".into())
            }
        };
        let job = self.file_job.take().unwrap();
        if job.is_cancelled() {
            return;
        }
        match result {
            Ok(ready) => {
                if job.insert {
                    self.snapshot();
                } else {
                    self.undo.clear();
                    self.redo.clear();
                    self.hidden_bodies.clear();
                    self.hidden_planes.clear();
                    self.hidden_sketches = ready
                        .doc
                        .features
                        .iter()
                        .filter_map(|f| match &f.kind {
                            FeatureKind::Extrude(x) => Some(x.sketch),
                            FeatureKind::Revolve(r) => Some(r.sketch),
                            _ => None,
                        })
                        .collect();
                    self.path = ready.native.then(|| job.path.clone());
                    self.dirty = !ready.native;
                }
                self.doc = ready.doc;
                self.meshes = ready.meshes;
                self.analysis_dirty = true;
                self.hover = None;
                self.timeline_sel = None;
                self.fit_all();
                for note in ready.notes {
                    self.toast(note, false);
                }
                self.toast(
                    format!(
                        "{} {}",
                        if job.insert { "Inserted" } else { "Opened" },
                        job.path.display()
                    ),
                    false,
                );
            }
            Err(e) => self.toast(format!("Could not load {}: {e}", job.path.display()), true),
        }
    }

    fn file_progress(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.file_job else { return };
        if job.is_cancelled() {
            return;
        }
        egui::Window::new("Loading model")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 150.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(job.path.file_name().unwrap_or_default().to_string_lossy());
                });
                ui.label("Preparing geometry… You can still navigate the current model.");
                if ui.button("Cancel").clicked() {
                    job.cancel();
                }
            });
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    fn save(&mut self, as_new: bool) {
        let path = if as_new || self.path.is_none() {
            rfd::FileDialog::new()
                .add_filter("refusion design", &["rfz"])
                .set_file_name("design.rfz")
                .save_file()
        } else {
            self.path.clone()
        };
        if let Some(p) = path {
            match std::fs::write(&p, self.doc.to_json()) {
                Ok(()) => {
                    self.toast(format!("Saved {}", p.display()), false);
                    self.path = Some(p);
                    self.dirty = false;
                }
                Err(e) => self.toast(format!("Save failed: {e}"), true),
            }
        }
    }

    fn export(&mut self, step: bool) {
        let shapes: Vec<kernel::Shape> = self
            .doc
            .state()
            .bodies
            .iter()
            .filter(|b| !self.hidden_bodies.contains(&b.id))
            .map(|b| b.shape.clone())
            .collect();
        if shapes.is_empty() {
            self.toast("Nothing to export", true);
            return;
        }
        let (ext, name) = if step {
            ("step", "design.step")
        } else {
            ("stl", "design.stl")
        };
        let Some(p) = rfd::FileDialog::new()
            .add_filter(ext, &[ext])
            .set_file_name(name)
            .save_file()
        else {
            return;
        };
        let r = if step {
            kernel::write_step(&shapes, &p).map_err(|e| e.0)
        } else {
            let meshes: Vec<Mesh> = shapes
                .iter()
                .filter_map(|s| s.mesh(0.01, 0.1).ok())
                .collect();
            kernel::write_stl(&meshes, &p).map_err(|e| e.to_string())
        };
        match r {
            Ok(()) => self.toast(format!("Exported {}", p.display()), false),
            Err(e) => self.toast(format!("Export failed: {e}"), true),
        }
    }

    fn open_stl_export(&mut self) {
        let bodies: Vec<(BodyId, String, bool)> = self
            .doc
            .state()
            .bodies
            .iter()
            .map(|b| (b.id, b.name.clone(), !self.hidden_bodies.contains(&b.id)))
            .collect();
        if bodies.is_empty() {
            self.toast("Nothing to export", true);
            return;
        }
        let (f, deg) = Refine::Medium.preset();
        let diag = self
            .doc
            .state()
            .bodies
            .iter()
            .filter_map(|b| b.shape.bbox())
            .map(|(lo, hi)| DVec3::from(hi).distance(DVec3::from(lo)))
            .fold(0.0, f64::max);
        self.stl_export = Some(StlExport {
            refine: Refine::Medium,
            surface_dev: f * diag,
            normal_dev_deg: deg,
            binary: true,
            unit: 0,
            per_body: false,
            bodies,
        });
    }

    fn stl_export_window(&mut self, ctx: &egui::Context) {
        let Some(ex) = self.stl_export.as_mut() else {
            return;
        };
        let mut open = true;
        let mut go = false;
        egui::Window::new("Export STL")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.strong("Bodies");
                for (_, name, on) in ex.bodies.iter_mut() {
                    ui.checkbox(on, name.as_str());
                }
                ui.separator();
                egui::Grid::new("stl").num_columns(2).show(ui, |ui| {
                    ui.label("Refinement");
                    ui.horizontal(|ui| {
                        for (r, n) in [
                            (Refine::High, "High"),
                            (Refine::Medium, "Medium"),
                            (Refine::Low, "Low"),
                            (Refine::Custom, "Custom"),
                        ] {
                            ui.selectable_value(&mut ex.refine, r, n);
                        }
                    });
                    ui.end_row();
                    if ex.refine == Refine::Custom {
                        ui.label("Surface deviation");
                        ui.add(
                            egui::DragValue::new(&mut ex.surface_dev)
                                .range(0.0001..=100.0)
                                .speed(0.001)
                                .suffix(" mm"),
                        );
                        ui.end_row();
                        ui.label("Normal deviation");
                        ui.add(
                            egui::DragValue::new(&mut ex.normal_dev_deg)
                                .range(0.5..=60.0)
                                .speed(0.1)
                                .suffix("°"),
                        );
                        ui.end_row();
                    }
                    ui.label("Format");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut ex.binary, true, "Binary");
                        ui.selectable_value(&mut ex.binary, false, "ASCII");
                    });
                    ui.end_row();
                    ui.label("Units");
                    ui.horizontal(|ui| {
                        for (i, (n, _)) in STL_UNITS.iter().enumerate() {
                            ui.selectable_value(&mut ex.unit, i, *n);
                        }
                    });
                    ui.end_row();
                    ui.label("Structure");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut ex.per_body, false, "One file");
                        ui.selectable_value(&mut ex.per_body, true, "One file per body");
                    });
                    ui.end_row();
                });
                ui.separator();
                if ui.button("Export…").clicked() {
                    go = true;
                }
            });
        if !open {
            self.stl_export = None;
        }
        if go {
            self.do_stl_export();
        }
    }

    fn do_stl_export(&mut self) {
        let Some(ex) = self.stl_export.as_ref() else {
            return;
        };
        let chosen: Vec<(String, kernel::Shape)> = ex
            .bodies
            .iter()
            .filter(|(_, _, on)| *on)
            .filter_map(|(id, name, _)| {
                self.doc
                    .state()
                    .body(*id)
                    .map(|b| (name.clone(), b.shape.clone()))
            })
            .collect();
        if chosen.is_empty() {
            self.toast("Select at least one body", true);
            return;
        }
        let mut meshed = Vec::new();
        for (name, shape) in &chosen {
            let (lin, deg) = match ex.refine {
                Refine::Custom => (ex.surface_dev, ex.normal_dev_deg),
                r => {
                    let (f, deg) = r.preset();
                    let diag = shape
                        .bbox()
                        .map(|(lo, hi)| DVec3::from(hi).distance(DVec3::from(lo)))
                        .unwrap_or(100.0);
                    (f * diag, deg)
                }
            };
            match shape.mesh_opts(lin.max(1e-4), deg.to_radians(), true) {
                Ok(m) => meshed.push((name.clone(), m)),
                Err(e) => {
                    self.toast(format!("Meshing {name} failed: {}", e.0), true);
                    return;
                }
            }
        }
        // Export meshing uses isolated shapes and preserves the display cache.
        let (ascii, scale) = (!ex.binary, STL_UNITS[ex.unit].1);
        let tris: usize = meshed.iter().map(|(_, m)| m.indices.len() / 3).sum();
        let result = if ex.per_body {
            match rfd::FileDialog::new()
                .set_title("Folder for STL files")
                .pick_folder()
            {
                None => return,
                Some(dir) => meshed
                    .iter()
                    .try_for_each(|(name, m)| {
                        kernel::write_stl_opts(
                            std::slice::from_ref(m),
                            &dir.join(format!("{name}.stl")),
                            ascii,
                            scale,
                            name,
                        )
                    })
                    .map(|_| dir),
            }
        } else {
            match rfd::FileDialog::new()
                .add_filter("STL", &["stl"])
                .set_file_name("design.stl")
                .save_file()
            {
                None => return,
                Some(p) => {
                    let ms: Vec<Mesh> = meshed.into_iter().map(|(_, m)| m).collect();
                    kernel::write_stl_opts(&ms, &p, ascii, scale, "refusion").map(|_| p)
                }
            }
        };
        match result {
            Ok(p) => {
                self.toast(
                    format!("Exported {tris} triangles to {}", p.display()),
                    false,
                );
                self.stl_export = None;
            }
            Err(e) => self.toast(format!("STL export failed: {e}"), true),
        }
    }

    fn fit_all(&mut self) {
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        for b in &self.doc.state().bodies {
            if self.hidden_bodies.contains(&b.id) {
                continue;
            }
            if let Some(m) = self.meshes.get(&b.shape.uid()) {
                for p in &m.positions {
                    let p = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        for (&id, state) in &self.doc.state().sketches {
            if self.hidden_sketches.contains(&id) {
                continue;
            }
            let Some(FeatureKind::Sketch(sf)) = self.doc.feature(id).map(|f| &f.kind) else {
                continue;
            };
            for (&entity, geom) in &sf.sketch.entities {
                let (centre, radius) = match geom.geom {
                    Geom::Point(_) => (sf.sketch.pos(entity), 0.0),
                    Geom::Circle { c, .. } | Geom::Arc { c, .. } => {
                        (sf.sketch.pos(c), sf.sketch.radius(entity).unwrap_or(0.0))
                    }
                    _ => continue,
                };
                for dx in [-radius, radius] {
                    for dy in [-radius, radius] {
                        let p = DVec3::from(state.frame.to_world([centre[0] + dx, centre[1] + dy]));
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                }
            }
        }
        if lo.x > hi.x {
            lo = DVec3::splat(-50.0);
            hi = DVec3::splat(50.0);
        }
        self.cam.fit(lo, hi);
    }

    fn plane_from_pick(&self, pick: Option<Pick>) -> Option<PlaneRef> {
        match pick? {
            Pick::Plane(k) => [PlaneRef::XY, PlaneRef::XZ, PlaneRef::YZ]
                .get(k as usize)
                .cloned(),
            Pick::ConstructionPlane(id) => Some(PlaneRef::Construction(id)),
            Pick::Face { body, face } => {
                self.doc.state().body(body)?.shape.face_info(face)?.plane?;
                self.doc.state().face_ref(body, face).map(PlaneRef::Face)
            }
            _ => None,
        }
    }

    fn start_offset_plane(&mut self, edit: Option<FeatureId>) {
        self.mode_to_model();
        let (base, distance) = match edit.and_then(|id| self.doc.feature(id)).map(|f| &f.kind) {
            Some(FeatureKind::OffsetPlane(p)) => (Some(p.base.clone()), p.distance.clone()),
            _ => (
                self.plane_from_pick(self.selection.first().copied()),
                "10 mm".into(),
            ),
        };
        self.cmd = Command::OffsetPlane {
            edit,
            base,
            distance,
        };
        self.rebuild_preview();
    }

    // ------------------------------------------------------------------ sketch mode

    fn start_new_sketch(&mut self, plane: PlaneRef, frame: Frame) {
        self.snapshot();
        let mut sk = Sketch::new();
        let o = sk.add_point(frame.to_local([0.0, 0.0, 0.0]));
        sk.entities.get_mut(&o).unwrap().projected = true;
        let restore = self.doc.marker + 1;
        let id = self.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane,
            sketch: sk,
            projections: vec![],
        }));
        self.enter_sketch(id, restore);
    }

    fn enter_sketch(&mut self, id: FeatureId, restore_marker: usize) {
        if self.replay_pending {
            self.pending_sketch = Some((
                id,
                restore_marker.max(self.replay_restore_marker.unwrap_or(restore_marker)),
            ));
            return;
        }
        let Some(idx) = self.doc.index_of(id) else {
            return;
        };
        if self.doc.marker < idx + 1 {
            self.pending_sketch = Some((id, restore_marker));
            self.timed_recompute(move |d| d.set_marker(idx + 1));
            return;
        }
        if self.doc.marker != idx + 1 {
            self.doc.set_marker(idx + 1);
        }
        let Some(ss) = self.doc.state().sketches.get(&id) else {
            self.toast("Sketch plane could not be resolved", true);
            return;
        };
        let fr = ss.frame;
        self.cam
            .look_at_plane(DVec3::from(fr.origin), DVec3::from(fr.x), DVec3::from(fr.y));
        self.hidden_sketches.remove(&id);
        self.mode = Mode::Sketch(SketchSession::new(id, fr, restore_marker.max(idx + 1)));
        self.cancel_command();
        self.selection.clear();
        self.analysis_dirty = true;
    }

    fn finish_sketch(&mut self) {
        let Mode::Sketch(s) = std::mem::replace(&mut self.mode, Mode::Model) else {
            return;
        };
        let restore = s.restore_marker.min(self.doc.features.len());
        let id = s.feature;
        self.replay_restore_marker = Some(restore);
        self.timed_recompute(move |d| {
            d.changed(id);
            d.set_marker(restore);
        });
        if let Some(e) = self.doc.errors.values().next().cloned() {
            self.toast(format!("Recompute: {e}"), true);
        }
    }

    fn sketch_outcome(&mut self, before: Option<Document>, o: Outcome) {
        if o.changed {
            if let Some(b) = before {
                let snap = self.snap_of(b);
                self.push_undo(snap);
            }
            self.analysis_dirty = true;
        }
        if let Some(m) = o.message {
            self.toast(m, true);
        }
    }

    fn refresh_analysis(&mut self) {
        if let Mode::Sketch(s) = &self.mode {
            if let Some(sk) = Self::sketch_ref(&self.doc, s.feature).filter(|sk| sk.x.len() > 128) {
                if let Some(data) = self.sketch_analysis.get(sk) {
                    let ticket = (s.feature, data.ticket);
                    if self.analysis_dirty || self.analysis_ticket != Some(ticket) {
                        self.sketch_dof = data.analysis.dof;
                        self.sketch_constrained = data.constrained.clone();
                        self.analysis_ticket = Some(ticket);
                        self.analysis_dirty = false;
                    }
                } else {
                    self.analysis_dirty = true;
                    if self
                        .analysis_ticket
                        .is_none_or(|(feature, _)| feature != s.feature)
                    {
                        self.sketch_constrained.clear();
                    }
                }
                return;
            }
        }
        if !self.analysis_dirty {
            return;
        }
        self.analysis_dirty = false;
        let Mode::Sketch(s) = &self.mode else { return };
        let Some(sk) = Self::sketch_mut(&mut self.doc, s.feature) else {
            return;
        };
        let ids: Vec<EntityId> = sk.entities.keys().copied().collect();
        self.sketch_constrained = ids
            .iter()
            .map(|&e| (e, sk.is_fully_constrained(e)))
            .collect();
        self.sketch_dof = sk.analysis().dof;
    }

    // ------------------------------------------------------------------ commands

    fn cancel_command(&mut self) {
        self.click_epoch = self.click_epoch.wrapping_add(1);
        self.pending_clicks.clear();
        self.compute_ticket = self.compute_ticket.wrapping_add(1);
        self.preview_pending = false;
        self.cmd = Command::None;
        self.preview = None;
        self.preview_error = None;
        self.prune_meshes();
    }

    fn start_extrude(&mut self, edit: Option<FeatureId>) {
        self.cancel_command();
        self.mode_to_model();
        let mut c = ExtrudeCmd {
            edit,
            sketch: None,
            profiles: vec![],
            faces: vec![],
            op_user_set: false,
            kind: ExtentKind::OneSide,
            all_direction: AllDirection::Positive,
            half_length: false,
            target: None,
            pick_target: false,
            offset: "0 mm".into(),
            taper: "0 deg".into(),
            d1: "10 mm".into(),
            d2: "10 mm".into(),
            op: if self.doc.state().bodies.is_empty() {
                Operation::NewBody
            } else {
                Operation::Join
            },
        };
        if let Some(Feature {
            kind: FeatureKind::Extrude(x),
            ..
        }) = edit.and_then(|id| self.doc.feature(id))
        {
            c.sketch = (x.sketch != 0).then_some(x.sketch);
            c.profiles = x.profiles.clone();
            c.faces = x.faces.clone();
            c.op = x.op;
            c.op_user_set = true;
            let extent = if let Extent::Tapered { extent, angle } = &x.extent {
                c.taper = angle.clone();
                extent.as_ref()
            } else {
                &x.extent
            };
            match extent {
                Extent::ToFace { face, offset } => {
                    c.kind = ExtentKind::ToFace;
                    c.target = Some(face.clone());
                    c.offset = offset.clone();
                }
                Extent::Tapered { .. } => {}
                Extent::OneSide { d } => c.d1 = d.clone(),
                Extent::Symmetric { d } => {
                    c.kind = ExtentKind::Symmetric;
                    c.d1 = d.clone()
                }
                Extent::SymmetricHalf { d } => {
                    c.kind = ExtentKind::Symmetric;
                    c.half_length = true;
                    c.d1 = d.clone();
                }
                Extent::ThroughAll { direction } => {
                    c.kind = ExtentKind::ThroughAll;
                    c.all_direction = *direction;
                }
                Extent::TwoSides { d1, d2 } => {
                    c.kind = ExtentKind::TwoSides;
                    c.d1 = d1.clone();
                    c.d2 = d2.clone();
                }
            }
            self.hidden_sketches.remove(&x.sketch);
        }
        self.cmd = Command::Extrude(c);
        self.rebuild_preview();
    }

    fn start_revolve(&mut self, edit: Option<FeatureId>) {
        self.mode_to_model();
        let mut c = RevolveCmd {
            edit,
            sketch: None,
            profiles: vec![],
            faces: vec![],
            direction: RevolveDirection::OneSide,
            angle2: "90 deg".into(),
            full: true,
            project_axis: true,
            axis: None,
            pick_axis: false,
            angle: "360 deg".into(),
            op: if self.doc.state().bodies.is_empty() {
                Operation::NewBody
            } else {
                Operation::Join
            },
        };
        if let Some(Feature {
            kind: FeatureKind::Revolve(r),
            ..
        }) = edit.and_then(|id| self.doc.feature(id))
        {
            c.sketch = Some(r.sketch);
            c.profiles = r.profiles.clone();
            c.faces = r.faces.clone();
            c.direction = r.direction;
            c.angle2 = r.angle2.clone();
            c.full = r.full
                || expr::eval(
                    &r.angle,
                    Unit::Angle,
                    &self.doc.param_quantities().unwrap_or_default(),
                )
                .is_ok_and(|a| (a.abs() - std::f64::consts::TAU).abs() < 1e-9);
            c.project_axis = r.project_axis;
            c.axis = Some(r.axis.clone());
            c.angle = r.angle.clone();
            c.op = r.op;
            self.hidden_sketches.remove(&r.sketch);
        }
        self.cmd = Command::Revolve(c);
        self.rebuild_preview();
    }

    fn start_blend(&mut self, chamfer: bool, edit: Option<FeatureId>) {
        self.mode_to_model();
        let mut c = BlendCmd {
            edit,
            edges: vec![],
            radius: if chamfer {
                "1 mm".into()
            } else {
                "2 mm".into()
            },
            chamfer: chamfer.then_some(ChamferType::Equal),
            d2: "1 mm".into(),
            angle: "45 deg".into(),
            show_preview: true,
        };
        match edit
            .and_then(|id| self.doc.feature(id))
            .map(|f| f.kind.clone())
        {
            Some(FeatureKind::Fillet(f)) => {
                c.edges = f.edges;
                c.radius = f.radius;
            }
            Some(FeatureKind::Chamfer(f)) => {
                c.edges = f.edges;
                match f.kind {
                    ChamferKind::Equal { d } => c.radius = d,
                    ChamferKind::TwoDistances { d1, d2 } => {
                        c.chamfer = Some(ChamferType::TwoDistances);
                        c.radius = d1;
                        c.d2 = d2;
                    }
                    ChamferKind::DistanceAngle { d, angle } => {
                        c.chamfer = Some(ChamferType::DistanceAngle);
                        c.radius = d;
                        c.angle = angle;
                    }
                }
            }
            _ => {
                // Use pre-selected edges.
                let st = self.doc.state();
                c.edges = self
                    .selection
                    .iter()
                    .filter_map(|p| {
                        if let Pick::Edge { body, edge } = *p {
                            st.edge_ref(body, edge)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }
        self.cmd = Command::Blend(c);
        self.rebuild_preview();
    }

    fn start_measure(&mut self) {
        if let Mode::Sketch(s) = &mut self.mode {
            s.set_tool(Tool::Select);
        } else {
            self.mode_to_model();
        }
        self.cmd = Command::Measure(crate::measure::MeasureCmd {
            precision: 2,
            ..Default::default()
        });
    }

    fn refresh_measure(&mut self) {
        let mut state = self.doc.state().clone();
        if let Mode::Sketch(s) = &self.mode {
            if let Some(sketch) = Self::sketch_ref(&self.doc, s.feature) {
                state
                    .sketches
                    .insert(s.feature, Arc::new(SketchState::of(s.frame, sketch)));
            }
        }
        if let Command::Measure(m) = &mut self.cmd {
            m.result = Some(crate::measure::compute(
                &self.doc,
                &state,
                m.a,
                m.b,
                m.precision,
            ));
        }
    }

    fn mode_to_model(&mut self) {
        if matches!(self.mode, Mode::Sketch(_)) {
            self.finish_sketch();
        }
        self.cancel_command();
    }

    /// The feature the open command would create.
    fn command_feature(&self) -> Option<(Option<FeatureId>, FeatureKind)> {
        Some(match &self.cmd {
            Command::Solid { edit, kind, .. } => (*edit, kind.clone()),
            Command::OffsetPlane {
                edit,
                base,
                distance,
            } => (
                *edit,
                FeatureKind::OffsetPlane(OffsetPlaneFeature {
                    base: base.clone()?,
                    distance: distance.clone(),
                }),
            ),
            Command::Extrude(c) => {
                let extent = match c.kind {
                    ExtentKind::ToFace => Extent::ToFace {
                        face: c.target.clone()?,
                        offset: c.offset.clone(),
                    },
                    ExtentKind::ThroughAll => Extent::ThroughAll {
                        direction: c.all_direction,
                    },
                    ExtentKind::OneSide => Extent::OneSide { d: c.d1.clone() },
                    ExtentKind::Symmetric => {
                        if c.half_length {
                            Extent::SymmetricHalf { d: c.d1.clone() }
                        } else {
                            Extent::Symmetric { d: c.d1.clone() }
                        }
                    }
                    ExtentKind::TwoSides => Extent::TwoSides {
                        d1: c.d1.clone(),
                        d2: c.d2.clone(),
                    },
                };
                if c.profiles.is_empty() && c.faces.is_empty() {
                    return None;
                }
                (
                    c.edit,
                    FeatureKind::Extrude(ExtrudeFeature {
                        sketch: if c.profiles.is_empty() { 0 } else { c.sketch? },
                        profiles: c.profiles.clone(),
                        faces: c.faces.clone(),
                        extent: Extent::Tapered {
                            extent: Box::new(extent),
                            angle: c.taper.clone(),
                        },
                        op: c.op,
                    }),
                )
            }
            Command::Revolve(c) => (
                c.edit,
                FeatureKind::Revolve(RevolveFeature {
                    faces: c.faces.clone(),
                    direction: c.direction,
                    angle2: c.angle2.clone(),
                    full: c.full,
                    project_axis: c.project_axis,
                    sketch: if c.profiles.is_empty() { 0 } else { c.sketch? },
                    profiles: c.profiles.clone(),
                    axis: c.axis.clone()?,
                    angle: c.angle.clone(),
                    op: c.op,
                }),
            ),
            Command::Blend(c) => {
                if c.edges.is_empty() {
                    return None;
                }
                let kind = match c.chamfer {
                    None => FeatureKind::Fillet(FilletFeature {
                        edges: c.edges.clone(),
                        radius: c.radius.clone(),
                    }),
                    Some(t) => FeatureKind::Chamfer(ChamferFeature {
                        edges: c.edges.clone(),
                        kind: match t {
                            ChamferType::Equal => ChamferKind::Equal {
                                d: c.radius.clone(),
                            },
                            ChamferType::TwoDistances => ChamferKind::TwoDistances {
                                d1: c.radius.clone(),
                                d2: c.d2.clone(),
                            },
                            ChamferType::DistanceAngle => ChamferKind::DistanceAngle {
                                d: c.radius.clone(),
                                angle: c.angle.clone(),
                            },
                        },
                    }),
                };
                (c.edit, kind)
            }
            _ => return None,
        })
    }

    /// Replay the timeline with the command's feature to show a live preview.
    fn rebuild_preview(&mut self) {
        self.compute_ticket = self.compute_ticket.wrapping_add(1);
        self.preview_pending = false;
        self.preview = None;
        self.preview_error = None;
        let has_bodies = !self.doc.state().bodies.is_empty();
        let vals = self.doc.param_quantities().unwrap_or_default();
        if let Command::Extrude(c) = &mut self.cmd {
            if !c.op_user_set {
                let on_body = !c.faces.is_empty()
                    || c.sketch
                        .and_then(|id| self.doc.feature(id))
                        .is_some_and(|f| matches!(&f.kind, FeatureKind::Sketch(sf) if matches!(sf.plane, PlaneRef::Face(_))));
                let negative = if c.kind == ExtentKind::ToFace {
                    let state = self.doc.state();
                    let target = c
                        .target
                        .as_ref()
                        .and_then(|r| state.resolve_face(r))
                        .and_then(|(bi, fid)| state.bodies[bi].shape.face_info(fid))
                        .and_then(|i| i.plane);
                    let source = c
                        .sketch
                        .and_then(|id| state.sketches.get(&id).map(|s| s.frame))
                        .or_else(|| {
                            c.faces
                                .first()
                                .and_then(|r| state.resolve_face(r))
                                .and_then(|(bi, fid)| state.bodies[bi].shape.face_info(fid))
                                .and_then(|i| i.plane)
                        });
                    match (source, target) {
                        (Some(source), Some(target)) => {
                            let n = source.normal();
                            let tn = target.normal();
                            let denom = kernel::dot(n, tn);
                            denom.abs() > 1e-8
                                && kernel::dot(kernel::sub(target.origin, source.origin), tn)
                                    / denom
                                    + expr::eval(&c.offset, Unit::Length, &vals).unwrap_or(0.)
                                    < 0.
                        }
                        _ => false,
                    }
                } else {
                    c.kind == ExtentKind::OneSide
                        && expr::eval(&c.d1, Unit::Length, &vals).is_ok_and(|v| v < 0.0)
                };
                c.op = if !has_bodies {
                    Operation::NewBody
                } else if on_body && negative {
                    Operation::Cut
                } else {
                    Operation::Join
                };
            }
        }
        let Some((edit, kind)) = self.command_feature() else {
            return;
        };
        if self.replay_pending {
            return;
        }
        self.preview_pending = true;
        self.compute.submit(crate::compute::Request {
            ticket: self.compute_ticket,
            replay: false,
            doc: self.doc.clone(),
            cached: self.meshes.clone(),
            edit: Box::new(move |d| {
                Some(match edit {
                    Some(id) => {
                        if let Some(f) = d.feature_mut(id) {
                            f.kind = kind;
                        }
                        d.changed(id);
                        id
                    }
                    None => d.add_feature(kind),
                })
            }),
        });
    }

    fn commit_command(&mut self) {
        if self.preview_pending || self.replay_pending {
            self.toast(
                "Computing preview; wait for the result before confirming",
                false,
            );
            return;
        }
        if self.preview.is_none() {
            self.rebuild_preview();
        }
        let Some(p) = self.preview.take() else {
            self.toast("Nothing to create yet — make a selection first", true);
            return;
        };
        if let Some(e) = &self.preview_error {
            self.toast(format!("Feature failed: {e}"), true);
            self.preview = Some(p);
            return;
        }
        match &self.cmd {
            Command::Extrude(ExtrudeCmd {
                sketch: Some(s), ..
            })
            | Command::Revolve(RevolveCmd {
                sketch: Some(s), ..
            }) => {
                self.hidden_sketches.insert(*s);
            }
            _ => {}
        }
        self.snapshot();
        self.doc = p;
        self.cancel_command();
        self.selection.clear();
    }

    fn update_mesh_quality(&mut self, mm_per_pixel: f64, ctx: &egui::Context) {
        while let Some(result) = self.mesh_lod.poll() {
            if self.mesh_pending.get(&result.uid) != Some(&result.linear) {
                continue;
            }
            self.mesh_pending.remove(&result.uid);
            if !self.meshes.contains_key(&result.uid) {
                continue;
            }
            match result.mesh {
                Ok(mesh) => {
                    self.meshes.insert(result.uid, mesh);
                    self.mesh_failed.remove(&result.uid);
                }
                Err(error) => {
                    self.mesh_failed.insert(result.uid, result.linear);
                    self.toast(format!("Display refinement: {error}"), true);
                }
            }
        }
        if (mm_per_pixel - self.mesh_view_scale).abs() > mm_per_pixel.abs().max(1e-8) * 0.005 {
            self.mesh_view_scale = mm_per_pixel;
            self.mesh_view_since = Instant::now();
        }
        if self.mesh_view_since.elapsed() < Duration::from_millis(200) {
            ctx.request_repaint_after(Duration::from_millis(200));
            return;
        }
        if self.file_busy() || self.preview_pending || self.replay_pending {
            return;
        }
        let mut requests = Vec::new();
        let states =
            std::iter::once(self.doc.state()).chain(self.preview.iter().map(|p| p.state()));
        for state in states {
            for body in &state.bodies {
                let uid = body.shape.uid();
                if self.hidden_bodies.contains(&body.id) || self.mesh_pending.contains_key(&uid) {
                    continue;
                }
                let Some(mesh) = self.meshes.get(&uid) else {
                    continue;
                };
                let desired = kernel::display_deflection(mesh.bounds, Some(mm_per_pixel));
                if self.mesh_failed.get(&uid) == Some(&desired) || mesh.linear_deflection <= 0. {
                    continue;
                }
                if mesh.linear_deflection > desired * 1.8 || mesh.linear_deflection < desired / 4. {
                    requests.push(crate::mesh_lod::Request {
                        uid,
                        shape: body.shape.clone(),
                        linear: desired,
                        volume: mesh.volume,
                    });
                    self.mesh_pending.insert(uid, desired);
                }
            }
        }
        if !requests.is_empty() {
            self.mesh_lod.submit(requests);
        }
    }

    fn click_context(&self) -> ClickContext {
        ClickContext {
            epoch: self.click_epoch,
            command: std::mem::discriminant(&self.cmd),
            sketch: match &self.mode {
                Mode::Sketch(s) => Some((s.feature, s.tool)),
                _ => None,
            },
            marker: self.doc.marker,
            selection_mode: match &self.cmd {
                Command::Extrude(c) if c.pick_target => 1,
                Command::Revolve(c) if c.pick_axis => 5,
                _ if matches!(&self.mode, Mode::Sketch(s) if s.tool == Tool::Project && s.project_bodies) => {
                    6
                }
                Command::Solid {
                    pick: SolidPick::Plane,
                    ..
                } => 2,
                Command::Solid {
                    pick: SolidPick::Axis1,
                    ..
                } => 3,
                Command::Solid {
                    pick: SolidPick::Axis2,
                    ..
                } => 4,
                Command::Solid {
                    pick: SolidPick::HoleCenter,
                    ..
                } => 5,
                Command::Solid {
                    pick: SolidPick::HoleTarget,
                    ..
                } => 6,
                _ => 0,
            },
            bodies: self
                .doc
                .state()
                .bodies
                .iter()
                .map(|b| b.shape.uid())
                .collect(),
        }
    }
    fn poll_clicks(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };
        while let Some(front) = self.pending_clicks.front() {
            if front.context != self.click_context() || self.file_busy() {
                self.pending_clicks.pop_front();
                continue;
            }
            let Some(result) = self
                .pending_clicks
                .front_mut()
                .unwrap()
                .readback
                .poll(&rs.device)
            else {
                ctx.request_repaint();
                break;
            };
            let click = self.pending_clicks.pop_front().unwrap();
            match result {
                Ok(ids) => {
                    let pick = scene::resolve_pick(&ids, PICK_WIN, &click.picks, &click.prio);
                    self.on_click(pick, click.cursor, click.at, click.shift);
                }
                Err(error) => self.toast(error, true),
            }
        }
    }

    // ------------------------------------------------------------------ picking → actions

    fn on_click(
        &mut self,
        pick: Option<Pick>,
        cursor_local: Option<[f64; 2]>,
        at: Pos2,
        shift: bool,
    ) {
        if let Command::Measure(m) = &mut self.cmd {
            if let Some(p) = pick {
                if m.a.is_none() || m.b.is_some() {
                    m.a = Some(p);
                    m.b = None;
                } else {
                    m.b = Some(p);
                }
                self.refresh_measure();
            }
            return;
        }
        match &mut self.mode {
            Mode::Sketch(sess) => {
                let fid = sess.feature;
                if sess.tool == Tool::Project {
                    let st = self.doc.state_before(fid);
                    let edges: Vec<EdgeRef> = match pick {
                        Some(Pick::Edge { body, edge }) if !sess.project_bodies => {
                            st.edge_ref(body, edge).into_iter().collect()
                        }
                        Some(Pick::Face { body, face }) => st
                            .body(body)
                            .map(|b| {
                                (1..=b.shape.edge_count() as u32)
                                    .filter(|&e| {
                                        sess.project_bodies || b.shape.edge_faces(e).contains(&face)
                                    })
                                    .filter_map(|e| st.edge_ref(body, e))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        _ => vec![],
                    };
                    let remove =
                        !edges.is_empty() && edges.iter().all(|e| sess.project_edges.contains(e));
                    for edge in edges {
                        if remove {
                            sess.project_edges.retain(|e| e != &edge);
                        } else if !sess.project_edges.contains(&edge) {
                            sess.project_edges.push(edge);
                        }
                    }
                    return;
                }
                let hover_ent = match pick {
                    Some(Pick::Ent { sketch, ent }) if sketch == fid => Some(ent),
                    _ => None,
                };
                let Some(cur) = cursor_local else { return };
                let before = self.doc.clone();
                let eval = self.eval_fn();
                let Mode::Sketch(sess) = &mut self.mode else {
                    unreachable!()
                };
                let Some(sk) = Self::sketch_mut(&mut self.doc, fid) else {
                    return;
                };
                let o = sess.click(sk, hover_ent, cur, at, &eval, shift);
                self.sketch_outcome(Some(before), o);
            }
            Mode::Model => match &mut self.cmd {
                Command::Solid {
                    edit,
                    kind,
                    pick: target,
                } => {
                    let st = edit
                        .map(|id| self.doc.state_before(id))
                        .unwrap_or_else(|| self.doc.state());
                    let mut changed = false;
                    if let FeatureKind::Hole(f) = kind {
                        if let Some(Pick::Face { body, face }) = pick {
                            if let Some(fr) = st
                                .body(body)
                                .and_then(|b| b.shape.face_info(face))
                                .and_then(|f| f.plane)
                            {
                                if let Some(reference) = st.face_ref(body, face) {
                                    if *target == SolidPick::HoleTarget {
                                        let offset = match &f.extent {
                                            HoleExtent::ToFace { offset, .. } => offset.clone(),
                                            _ => "0 mm".into(),
                                        };
                                        f.extent = HoleExtent::ToFace {
                                            face: reference,
                                            offset,
                                        };
                                        changed = true;
                                    } else {
                                        let xy = if self.viewport_rect.is_positive() {
                                            let local = at - self.viewport_rect.min;
                                            self.cam
                                                .ray(
                                                    DVec2::new(local.x as f64, local.y as f64),
                                                    DVec2::new(
                                                        self.viewport_rect.width() as f64,
                                                        self.viewport_rect.height() as f64,
                                                    ),
                                                )
                                                .hit_plane(
                                                    DVec3::from(fr.origin),
                                                    DVec3::from(fr.normal()),
                                                )
                                                .map(|p| fr.to_local(p.into()))
                                        } else {
                                            Some([0., 0.])
                                        };
                                        if let Some(xy) = xy {
                                            let plane = PlaneRef::Face(reference);
                                            let same_plane = f.plane == plane && f.body == body;
                                            if !shift || !same_plane {
                                                f.centers.clear();
                                            }
                                            f.body = body;
                                            f.body_tag = st
                                                .body(body)
                                                .and_then(|b| b.shape.face_tags().first().copied());
                                            f.plane = plane;
                                            f.centers.push([
                                                format!("{} mm", xy[0]),
                                                format!("{} mm", xy[1]),
                                            ]);
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    } else if *target == SolidPick::Plane {
                        let plane = match pick {
                            Some(Pick::Plane(k)) => [PlaneRef::XY, PlaneRef::XZ, PlaneRef::YZ]
                                .get(k as usize)
                                .cloned(),
                            Some(Pick::ConstructionPlane(id)) => Some(PlaneRef::Construction(id)),
                            Some(Pick::Face { body, face }) => st
                                .body(body)
                                .and_then(|b| b.shape.face_info(face))
                                .and_then(|f| f.plane)
                                .and_then(|_| st.face_ref(body, face))
                                .map(PlaneRef::Face),
                            _ => None,
                        };
                        if let Some(plane) = plane {
                            match kind {
                                FeatureKind::Mirror(f) => {
                                    f.plane = plane;
                                    changed = true;
                                }
                                FeatureKind::AngledPlane(f) => {
                                    f.base = plane;
                                    changed = true;
                                }
                                _ => {}
                            }
                        }
                    } else if matches!(target, SolidPick::Axis1 | SolidPick::Axis2) {
                        if let Some(Pick::Edge { body, edge }) = pick {
                            if let Some(reference) = st.edge_ref(body, edge) {
                                let axis = LinearRef::Edge(reference);
                                match kind {
                                    FeatureKind::AngledPlane(f) => {
                                        f.axis = axis;
                                        changed = true;
                                    }
                                    FeatureKind::Pattern(f) => match &mut f.kind {
                                        PatternKind::Circular { axis: slot, .. } => {
                                            *slot = axis;
                                            changed = true;
                                        }
                                        PatternKind::Rectangular { axis1, axis2, .. } => {
                                            if *target == SolidPick::Axis1 {
                                                *axis1 = axis;
                                            } else {
                                                *axis2 = axis;
                                            }
                                            changed = true;
                                        }
                                    },
                                    _ => {}
                                }
                            }
                        }
                    } else if let Some(Pick::Face { body, face }) = pick {
                        match kind {
                            FeatureKind::Shell(f) => {
                                if f.body != body {
                                    f.body = body;
                                    f.body_tag = st
                                        .body(body)
                                        .and_then(|b| b.shape.face_tags().first().copied());
                                    f.faces.clear();
                                }
                                if let Some(r) = st.face_ref(body, face) {
                                    if let Some(i) = f.faces.iter().position(|v| v == &r) {
                                        f.faces.remove(i);
                                    } else {
                                        f.faces.push(r);
                                    }
                                    changed = true;
                                }
                            }
                            FeatureKind::Mirror(f) => {
                                if let Some(i) = f.bodies.iter().position(|id| *id == body) {
                                    f.bodies.remove(i);
                                } else {
                                    f.bodies.push(body);
                                }
                                changed = true;
                            }
                            FeatureKind::Pattern(f) if f.objects == PatternObjects::Bodies => {
                                if let Some(i) = f.bodies.iter().position(|id| *id == body) {
                                    f.bodies.remove(i);
                                } else {
                                    f.bodies.push(body);
                                }
                                changed = true;
                            }
                            _ => {}
                        }
                    }
                    if changed {
                        self.rebuild_preview();
                    } else if pick.is_some()
                        && matches!(
                            target,
                            SolidPick::Plane | SolidPick::Axis1 | SolidPick::Axis2
                        )
                    {
                        let message = if *target == SolidPick::Plane {
                            "Select an origin plane, construction plane or planar face"
                        } else {
                            "Select a straight body edge for the axis or direction"
                        };
                        self.toast(message, true);
                    }
                }
                Command::OffsetPlane { .. } => {
                    let base = self.plane_from_pick(pick);
                    if let Some(base) = base {
                        if let Command::OffsetPlane { base: slot, .. } = &mut self.cmd {
                            *slot = Some(base);
                        }
                        self.rebuild_preview();
                    }
                }
                Command::NewSketch => match pick {
                    Some(Pick::ConstructionPlane(id)) => {
                        if let Some(&frame) = self.doc.state().planes.get(&id) {
                            self.cmd = Command::None;
                            self.start_new_sketch(PlaneRef::Construction(id), frame);
                        }
                    }
                    Some(Pick::Plane(k)) => {
                        let plane = [PlaneRef::XY, PlaneRef::XZ, PlaneRef::YZ][k as usize].clone();
                        let frame = match k {
                            0 => Frame {
                                origin: [0.0; 3],
                                x: [1.0, 0.0, 0.0],
                                y: [0.0, 1.0, 0.0],
                            },
                            1 => Frame {
                                origin: [0.0; 3],
                                x: [1.0, 0.0, 0.0],
                                y: [0.0, 0.0, -1.0],
                            },
                            _ => Frame {
                                origin: [0.0; 3],
                                x: [0.0, 0.0, -1.0],
                                y: [0.0, 1.0, 0.0],
                            },
                        };
                        self.cmd = Command::None;
                        self.start_new_sketch(plane, frame);
                    }
                    Some(Pick::Face { body, face }) => {
                        let st = self.doc.state();
                        let info = st.body(body).and_then(|b| b.shape.face_info(face));
                        match (info.and_then(|i| i.plane), st.face_ref(body, face)) {
                            (Some(frame), Some(r)) => {
                                self.cmd = Command::None;
                                self.start_new_sketch(PlaneRef::Face(r), frame);
                            }
                            _ => self.toast("Pick a planar face or an origin plane", true),
                        }
                    }
                    _ => {}
                },
                Command::Extrude(c) if c.kind == ExtentKind::ToFace && c.pick_target => {
                    if let Some(Pick::Face { body, face }) = pick {
                        let st = self.doc.state();
                        if st
                            .body(body)
                            .and_then(|b| b.shape.face_info(face))
                            .is_some_and(|i| i.plane.is_some())
                        {
                            c.target = st.face_ref(body, face);
                            c.pick_target = false;
                            self.rebuild_preview();
                        } else {
                            self.toast("Select a planar target face", true);
                        }
                    }
                }
                Command::Extrude(_) | Command::Revolve(_) => {
                    let st = self.doc.state();
                    let mut changed = false;
                    match pick {
                        Some(Pick::Profile { sketch, index }) if !matches!(&self.cmd, Command::Revolve(c) if c.pick_axis) => {
                            if let Some(p) =
                                st.sketches.get(&sketch).and_then(|s| s.profiles.get(index))
                            {
                                let r = profile_ref(p);
                                let (sk_slot, profiles) = match &mut self.cmd {
                                    Command::Extrude(c) => (&mut c.sketch, &mut c.profiles),
                                    Command::Revolve(c) => (&mut c.sketch, &mut c.profiles),
                                    _ => unreachable!(),
                                };
                                if *sk_slot != Some(sketch) {
                                    *sk_slot = Some(sketch);
                                    profiles.clear();
                                }
                                if let Some(i) = profiles.iter().position(|q| {
                                    st.sketches
                                        .get(&sketch)
                                        .and_then(|s| match_profile(&s.profiles, q))
                                        .is_some_and(|found| std::ptr::eq(found, p))
                                }) {
                                    profiles.remove(i);
                                } else {
                                    profiles.push(r);
                                }
                                changed = true;
                            }
                        }
                        Some(Pick::Edge { body, edge }) if matches!(&self.cmd, Command::Revolve(c) if c.pick_axis) => {
                            if let Some(reference) = st.edge_ref(body, edge) {
                                let kind = st
                                    .body(body)
                                    .and_then(|b| b.shape.edge_info(edge))
                                    .map(|e| e.kind);
                                if matches!(
                                    kind,
                                    Some(kernel::EdgeKind::Line | kernel::EdgeKind::Circle { .. })
                                ) {
                                    if let Command::Revolve(c) = &mut self.cmd {
                                        c.axis = Some(Axis::Edge(reference));
                                        c.pick_axis = false;
                                        changed = true;
                                    }
                                } else {
                                    self.toasts.push(Toast {
                                        text: "Select a linear or circular axis object".into(),
                                        at: Instant::now(),
                                        error: true,
                                    });
                                }
                            }
                        }
                        Some(Pick::Face { body, face }) if !matches!(&self.cmd, Command::Revolve(c) if c.pick_axis) =>
                        {
                            if let Command::Revolve(c) = &mut self.cmd {
                                if st
                                    .body(body)
                                    .and_then(|b| b.shape.face_info(face))
                                    .is_some_and(|f| f.plane.is_some())
                                {
                                    if let Some(reference) = st.face_ref(body, face) {
                                        if let Some(i) =
                                            c.faces.iter().position(|f| f == &reference)
                                        {
                                            c.faces.remove(i);
                                        } else {
                                            c.faces.push(reference);
                                        }
                                        changed = true;
                                    }
                                } else {
                                    self.toasts.push(Toast {
                                        text: "Select a planar profile face".into(),
                                        at: Instant::now(),
                                        error: true,
                                    });
                                }
                            }
                            if let Command::Extrude(c) = &mut self.cmd {
                                let planar = st
                                    .body(body)
                                    .and_then(|b| b.shape.face_info(face))
                                    .is_some_and(|i| i.plane.is_some());
                                if !planar {
                                    self.toasts.push(Toast {
                                        text: "Only planar faces can be extruded".into(),
                                        at: Instant::now(),
                                        error: true,
                                    });
                                } else if let Some(r) = st.face_ref(body, face) {
                                    if let Some(i) = c.faces.iter().position(|f| {
                                        f.tag == r.tag && dist3(f.hint, r.hint) < 1e-6
                                    }) {
                                        c.faces.remove(i);
                                    } else {
                                        c.faces.push(r);
                                    }
                                    changed = true;
                                }
                            }
                        }
                        Some(Pick::Ent { sketch, ent }) => {
                            if let Command::Revolve(c) = &mut self.cmd {
                                if !c.pick_axis {
                                    return;
                                }
                                let is_line = Self::sketch_ref(&self.doc, sketch)
                                    .and_then(|s| s.line_pts(ent))
                                    .is_some();
                                if is_line && (c.sketch.is_none() || c.sketch == Some(sketch)) {
                                    c.sketch = Some(sketch);
                                    c.axis = Some(Axis::SketchLine(ent));
                                    c.pick_axis = false;
                                    changed = true;
                                } else {
                                    self.toasts.push(Toast {
                                        text: "Select a straight line in the profile sketch for the axis".into(),
                                        at: Instant::now(),
                                        error: true,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                    if changed {
                        self.rebuild_preview();
                    }
                }
                Command::Blend(c) => {
                    let st = match c.edit {
                        Some(id) => self.doc.state_before(id),
                        None => self.doc.state(),
                    };
                    let toggle =
                        |r: EdgeRef, c: &mut BlendCmd| {
                            if let Some(i) = c.edges.iter().position(|e| {
                                e.a == r.a && e.b == r.b && dist3(e.hint, r.hint) < 1e-6
                            }) {
                                c.edges.remove(i);
                            } else {
                                c.edges.push(r);
                            }
                        };
                    match pick {
                        Some(Pick::Edge { body, edge }) => {
                            if let Some(r) = st.edge_ref(body, edge) {
                                toggle(r, c);
                            }
                        }
                        Some(Pick::Face { body, face }) => {
                            if let Some(b) = st.body(body) {
                                for e in (1..=b.shape.edge_count() as u32)
                                    .filter(|&e| b.shape.edge_faces(e).contains(&face))
                                {
                                    if let Some(r) = st.edge_ref(body, e) {
                                        if !c.edges.iter().any(|x| {
                                            x.a == r.a && x.b == r.b && dist3(x.hint, r.hint) < 1e-6
                                        }) {
                                            c.edges.push(r);
                                        }
                                    }
                                }
                            }
                        }
                        _ => return,
                    }
                    self.rebuild_preview();
                }
                Command::Measure(_) => {
                    unreachable!("Measure is handled before sketch/model dispatch")
                }
                Command::None => match pick {
                    Some(p @ (Pick::Face { .. } | Pick::Edge { .. })) => {
                        if shift {
                            if let Some(i) = self.selection.iter().position(|x| *x == p) {
                                self.selection.remove(i);
                            } else {
                                self.selection.push(p);
                            }
                        } else {
                            self.selection = vec![p];
                        }
                    }
                    _ => {
                        if !shift {
                            self.selection.clear();
                        }
                    }
                },
            },
        }
    }

    // ------------------------------------------------------------------ keyboard

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if self.navigation.is_some() && ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.navigation = None;
            self.press = None;
            self.box_start = None;
            return;
        }
        // HUD keys must work while its text field has focus.
        if let Mode::Sketch(s) = &mut self.mode {
            if s.hud.is_some() {
                let back = ctx.input_mut(|i| i.consume_key(egui::Modifiers::SHIFT, Key::Tab));
                let fwd = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Tab));
                if back || fwd {
                    s.hud.as_mut().unwrap().cycle(back);
                }
                if ctx.input(|i| i.key_pressed(Key::Escape)) {
                    s.cancel();
                    ctx.memory_mut(|m| m.request_focus(egui::Id::NULL));
                    return;
                }
                if ctx.input(|i| i.key_pressed(Key::Enter)) {
                    let fid = s.feature;
                    let before = self.doc.clone();
                    let eval = self.eval_fn();
                    let Mode::Sketch(s) = &mut self.mode else {
                        return;
                    };
                    if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                        let o = s.commit_hud(sk, &eval);
                        self.sketch_outcome(Some(before), o);
                    }
                    return;
                }
            }
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let (ctrl, shift) = ctx.input(|i| (i.modifiers.command, i.modifiers.shift));
        let pressed = |k: Key| ctx.input(|i| i.key_pressed(k));
        if ctrl {
            if pressed(Key::Z) {
                if shift {
                    self.redo()
                } else {
                    self.undo()
                }
            }
            if pressed(Key::Y) {
                self.redo();
            }
            if pressed(Key::S) {
                self.save(shift);
            }
            if pressed(Key::O) {
                if let Some(p) = open_dialog().pick_file() {
                    self.open_path(p);
                }
            }
            return;
        }
        if pressed(Key::Escape) && matches!(self.cmd, Command::Measure(_)) {
            self.cancel_command();
            return;
        }
        if pressed(Key::I) && matches!(self.mode, Mode::Sketch(_)) {
            self.start_measure();
            return;
        }
        if pressed(Key::Escape) {
            self.click_epoch = self.click_epoch.wrapping_add(1);
            self.pending_clicks.clear();
            match &mut self.mode {
                Mode::Sketch(s) => {
                    if !s.cancel() {
                        if s.tool != Tool::Select {
                            s.set_tool(Tool::Select);
                        } else {
                            s.selection.clear();
                        }
                    }
                }
                Mode::Model => {
                    if !matches!(self.cmd, Command::None) {
                        self.cancel_command();
                    } else {
                        self.selection.clear();
                    }
                }
            }
        }
        if pressed(Key::Enter) {
            self.commit_projection();
            if matches!(self.mode, Mode::Model)
                && !matches!(self.cmd, Command::None | Command::NewSketch)
            {
                self.commit_command();
            }
        }
        if pressed(Key::F6) || (pressed(Key::F) && matches!(self.mode, Mode::Sketch(_)) && false) {
            self.fit_all();
        }
        match &mut self.mode {
            Mode::Sketch(s) => {
                let tool = if pressed(Key::L) {
                    Some(Tool::Line)
                } else if pressed(Key::R) {
                    Some(Tool::Rectangle)
                } else if pressed(Key::C) {
                    Some(Tool::Circle)
                } else if pressed(Key::T) {
                    Some(Tool::Trim)
                } else if pressed(Key::O) {
                    Some(Tool::Offset)
                } else if pressed(Key::A) {
                    Some(Tool::Arc)
                } else if pressed(Key::D) {
                    Some(Tool::Dimension)
                } else if pressed(Key::P) {
                    Some(Tool::Project)
                } else if pressed(Key::Space) {
                    Some(Tool::Select)
                } else if pressed(Key::G) {
                    self.grid_snap = !self.grid_snap;
                    None
                } else {
                    None
                };
                if pressed(Key::Enter)
                    && !matches!(
                        s.tool,
                        Tool::Select | Tool::Dimension | Tool::Constrain(_) | Tool::Project
                    )
                {
                    s.set_tool(Tool::Select);
                }
                if let Some(t) = tool {
                    if matches!(self.cmd, Command::Measure(_)) {
                        self.cmd = Command::None;
                    }
                    self.click_epoch = self.click_epoch.wrapping_add(1);
                    self.pending_clicks.clear();
                    s.set_tool(t);
                }
                let fid = s.feature;
                if pressed(Key::Delete) || pressed(Key::Backspace) {
                    let before = self.doc.clone();
                    let Mode::Sketch(s) = &mut self.mode else {
                        unreachable!()
                    };
                    if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                        let o = s.delete_selection(sk);
                        self.sketch_outcome(Some(before), o);
                    }
                } else if pressed(Key::X) {
                    let before = self.doc.clone();
                    let Mode::Sketch(s) = &mut self.mode else {
                        unreachable!()
                    };
                    if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                        let o = s.toggle_construction(sk);
                        self.sketch_outcome(Some(before), o);
                    }
                }
            }
            Mode::Model => {
                if pressed(Key::E) || pressed(Key::Q) {
                    self.start_extrude(None);
                } else if pressed(Key::H) {
                    self.run_action(Action::Hole);
                } else if pressed(Key::F) {
                    self.start_blend(false, None);
                } else if pressed(Key::I) {
                    self.start_measure();
                } else if pressed(Key::S) {
                    self.cmd = Command::NewSketch;
                } else if pressed(Key::Delete) {
                    if let Some(id) = self.timeline_sel.take() {
                        self.snapshot();
                        self.timed_recompute(move |d| d.remove_feature(id));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ panels

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New").clicked() {
                    self.new_document();
                    ui.close_menu();
                }
                if ui.button("Open…  (Ctrl+O)").clicked() {
                    ui.close_menu();
                    if let Some(p) = open_dialog().pick_file() {
                        self.open_path(p);
                    }
                }
                if ui
                    .button("Insert into design…")
                    .on_hover_text("Add a STEP, IGES or F3D .f3d model as an Import feature")
                    .clicked()
                {
                    ui.close_menu();
                    self.insert_dialog();
                }
                if ui.button("Save  (Ctrl+S)").clicked() {
                    ui.close_menu();
                    self.save(false);
                }
                if ui.button("Save As…").clicked() {
                    ui.close_menu();
                    self.save(true);
                }
                ui.separator();
                if ui.button("Export STEP…").clicked() {
                    ui.close_menu();
                    self.export(true);
                }
                if ui.button("Export STL…").clicked() {
                    ui.close_menu();
                    self.open_stl_export();
                }
            });
            if ui
                .add_enabled(!self.undo.is_empty(), egui::Button::new("⟲"))
                .on_hover_text("Undo (Ctrl+Z)")
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(!self.redo.is_empty(), egui::Button::new("⟳"))
                .on_hover_text("Redo (Ctrl+Y)")
                .clicked()
            {
                self.redo();
            }
        });
        self.ribbon(ui);
    }

    fn ribbon(&mut self, ui: &mut egui::Ui) {
        let mut act: Option<Action> = None;
        let sketching = matches!(self.mode, Mode::Sketch(_));
        ui.horizontal(|ui| {
            ui.menu_button(RichText::new("DESIGN").strong(), |ui| {
                ui.label("Workspaces");
                ui.add_enabled(false, egui::Button::new("✔ Design"));
            });
            ui.separator();
            let tab = |ui: &mut egui::Ui, name: &str, on: bool| {
                let t = RichText::new(name).strong();
                ui.add(
                    egui::Button::new(if on {
                        t.color(Color32::from_rgb(90, 160, 255))
                    } else {
                        t.weak()
                    })
                    .frame(on),
                )
                .clicked()
            };
            if tab(ui, "SOLID", !sketching) && sketching {
                act = Some(Action::FinishSketch);
            }
            if sketching {
                tab(ui, "SKETCH", true);
            }
        });
        ui.add_space(2.0);
        egui::ScrollArea::horizontal()
            .id_salt("ribbon_tools")
            .show(ui, |ui| {
                ui.horizontal(|ui| match &self.mode {
                    Mode::Model => {
                        let cmd = &self.cmd;
                        ribbon_panel(
                            ui,
                            "CREATE",
                            &[
                                (
                                    "Create Sketch",
                                    "S",
                                    matches!(cmd, Command::NewSketch),
                                    Action::NewSketch,
                                ),
                                (
                                    "Extrude",
                                    "E",
                                    matches!(cmd, Command::Extrude(_)),
                                    Action::Extrude,
                                ),
                                (
                                    "Revolve",
                                    "",
                                    matches!(cmd, Command::Revolve(_)),
                                    Action::Revolve,
                                ),
                                (
                                    "Hole",
                                    "H",
                                    matches!(
                                        cmd,
                                        Command::Solid {
                                            kind: FeatureKind::Hole(_),
                                            ..
                                        }
                                    ),
                                    Action::Hole,
                                ),
                                (
                                    "Rectangular Pattern",
                                    "",
                                    matches!(
                                        cmd,
                                        Command::Solid {
                                            kind: FeatureKind::Pattern(PatternFeature {
                                                kind: PatternKind::Rectangular { .. },
                                                ..
                                            }),
                                            ..
                                        }
                                    ),
                                    Action::RectPattern,
                                ),
                            ],
                            |ui, act| {
                                menu_item(ui, "Create Sketch", Action::NewSketch, act);
                                ui.separator();
                                menu_item(ui, "Extrude", Action::Extrude, act);
                                menu_item(ui, "Revolve", Action::Revolve, act);
                                ui.separator();
                                menu_item(ui, "Hole", Action::Hole, act);
                                ui.separator();
                                icon_submenu(
                                    ui,
                                    "Pattern",
                                    crate::tool_icons::Icon::RectPattern,
                                    |ui| {
                                        menu_item(
                                            ui,
                                            "Rectangular Pattern",
                                            Action::RectPattern,
                                            act,
                                        );
                                        menu_item(
                                            ui,
                                            "Circular Pattern",
                                            Action::CircularPattern,
                                            act,
                                        );
                                    },
                                );
                                menu_item(ui, "Mirror", Action::Mirror, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "MODIFY",
                            &[
                                ("Press Pull", "Q", false, Action::Extrude),
                                (
                                    "Fillet",
                                    "F",
                                    matches!(cmd, Command::Blend(BlendCmd { chamfer: None, .. })),
                                    Action::Fillet,
                                ),
                                (
                                    "Shell",
                                    "",
                                    matches!(
                                        cmd,
                                        Command::Solid {
                                            kind: FeatureKind::Shell(_),
                                            ..
                                        }
                                    ),
                                    Action::Shell,
                                ),
                            ],
                            |ui, act| {
                                menu_item(ui, "Press Pull", Action::Extrude, act);
                                ui.separator();
                                menu_item(ui, "Fillet", Action::Fillet, act);
                                menu_item(ui, "Chamfer", Action::Chamfer, act);
                                ui.separator();
                                menu_item(ui, "Shell", Action::Shell, act);
                                ui.separator();
                                menu_item(ui, "Change Parameters", Action::Params, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "CONSTRUCT",
                            &[(
                                "Offset Plane",
                                "",
                                matches!(cmd, Command::OffsetPlane { .. }),
                                Action::OffsetPlane,
                            )],
                            |ui, act| {
                                menu_item(ui, "Offset Plane", Action::OffsetPlane, act);
                                menu_item(ui, "Plane at Angle", Action::AngledPlane, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "INSPECT",
                            &[(
                                "Measure",
                                "I",
                                matches!(cmd, Command::Measure(_)),
                                Action::Measure,
                            )],
                            |ui, act| {
                                menu_item(ui, "Measure", Action::Measure, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "INSERT",
                            &[("Insert (STEP / IGES / f3d)", "", false, Action::Insert)],
                            |ui, act| {
                                menu_item(ui, "Insert STEP / IGES / f3d…", Action::Insert, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "SELECT",
                            &[(
                                "Select",
                                "Esc",
                                matches!(cmd, Command::None),
                                Action::SelectModel,
                            )],
                            |ui, act| {
                                menu_item(ui, "Select", Action::SelectModel, act);
                            },
                            &mut act,
                        );
                    }
                    Mode::Sketch(s) => {
                        let tool = s.tool;
                        let t = |x: Tool| tool == x;
                        ribbon_panel(
                            ui,
                            "CREATE",
                            &[
                                ("Line", "L", t(Tool::Line), Action::Tool(Tool::Line)),
                                (
                                    "2-Point Rectangle",
                                    "R",
                                    t(Tool::Rectangle),
                                    Action::Tool(Tool::Rectangle),
                                ),
                                (
                                    "Center Diameter Circle",
                                    "C",
                                    t(Tool::Circle),
                                    Action::Tool(Tool::Circle),
                                ),
                                (
                                    "Sketch Dimension",
                                    "D",
                                    t(Tool::Dimension),
                                    Action::Tool(Tool::Dimension),
                                ),
                            ],
                            |ui, act| {
                                menu_item(ui, "Line", Action::Tool(Tool::Line), act);
                                icon_submenu(
                                    ui,
                                    "Rectangle",
                                    crate::tool_icons::Icon::Rectangle,
                                    |ui| {
                                        menu_item(
                                            ui,
                                            "2-Point Rectangle",
                                            Action::Tool(Tool::Rectangle),
                                            act,
                                        );
                                        menu_item(
                                            ui,
                                            "Center Rectangle",
                                            Action::Tool(Tool::CenterRectangle),
                                            act,
                                        );
                                    },
                                );
                                icon_submenu(ui, "Circle", crate::tool_icons::Icon::Circle, |ui| {
                                    menu_item(
                                        ui,
                                        "Center Diameter Circle",
                                        Action::Tool(Tool::Circle),
                                        act,
                                    )
                                });
                                icon_submenu(
                                    ui,
                                    "Arc",
                                    crate::tool_icons::Icon::ThreePointArc,
                                    |ui| {
                                        menu_item(
                                            ui,
                                            "3-Point Arc",
                                            Action::Tool(Tool::ThreePointArc),
                                            act,
                                        );
                                        menu_item(
                                            ui,
                                            "Center Point Arc",
                                            Action::Tool(Tool::Arc),
                                            act,
                                        );
                                        menu_item(
                                            ui,
                                            "Tangent Arc",
                                            Action::Tool(Tool::TangentArc),
                                            act,
                                        );
                                    },
                                );
                                menu_item(ui, "Point", Action::Tool(Tool::Point), act);
                                icon_submenu(
                                    ui,
                                    "Project / Include",
                                    crate::tool_icons::Icon::Project,
                                    |ui| menu_item(ui, "Project", Action::Tool(Tool::Project), act),
                                );
                                ui.separator();
                                menu_item(
                                    ui,
                                    "Sketch Dimension",
                                    Action::Tool(Tool::Dimension),
                                    act,
                                );
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "MODIFY",
                            &[
                                ("Trim", "T", t(Tool::Trim), Action::Tool(Tool::Trim)),
                                ("Offset", "O", t(Tool::Offset), Action::Tool(Tool::Offset)),
                            ],
                            |ui, act| {
                                menu_item(ui, "Offset", Action::Tool(Tool::Offset), act);
                                ui.separator();
                                menu_item(ui, "Trim", Action::Tool(Tool::Trim), act);
                                menu_item(ui, "Extend", Action::Tool(Tool::Extend), act);
                                ui.separator();
                                menu_item(ui, "Change Parameters", Action::Params, act);
                            },
                            &mut act,
                        );
                        let cons: Vec<(&str, &str, bool, Action)> = ConKind::ALL
                            .iter()
                            .map(|&k| (k.label(), "", t(Tool::Constrain(k)), Action::Constraint(k)))
                            .collect();
                        ribbon_panel(
                            ui,
                            "CONSTRAINTS",
                            &cons,
                            |ui, act| {
                                for k in ConKind::ALL {
                                    menu_item(ui, k.label(), Action::Constraint(k), act);
                                }
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "INSPECT",
                            &[("Measure", "I", false, Action::Measure)],
                            |ui, act| {
                                menu_item(ui, "Measure", Action::Measure, act);
                            },
                            &mut act,
                        );
                        ribbon_panel(
                            ui,
                            "SELECT",
                            &[(
                                "Select",
                                "Space",
                                t(Tool::Select),
                                Action::Tool(Tool::Select),
                            )],
                            |ui, act| {
                                menu_item(ui, "Select", Action::Tool(Tool::Select), act);
                                menu_item(ui, "Window Selection", Action::Tool(Tool::Select), act);
                            },
                            &mut act,
                        );
                        // StopSketchPanel
                        ui.vertical(|ui| {
                            let response = ui.add_sized([96., 34.], egui::Button::new(""));
                            crate::tool_icons::paint(
                                ui.painter(),
                                egui::Rect::from_center_size(
                                    response.rect.center(),
                                    Vec2::splat(25.),
                                ),
                                crate::tool_icons::Icon::Finish,
                            );
                            if response.on_hover_text("Finish Sketch").clicked() {
                                act = Some(Action::FinishSketch);
                            }
                            ui.label(RichText::new("FINISH SKETCH").small().strong());
                        });
                    }
                });
            });
        if let Some(a) = act {
            self.run_action(a);
        }
    }

    fn commit_projection(&mut self) {
        let Mode::Sketch(s) = &self.mode else { return };
        if s.tool != Tool::Project || s.project_edges.is_empty() {
            return;
        }
        let (fid, edges, linked) = (s.feature, s.project_edges.clone(), s.projection_link);
        let before = self.doc.clone();
        let mut work = before.clone();
        let (start, initial_entities) = match work.feature(fid).map(|f| &f.kind) {
            Some(FeatureKind::Sketch(sf)) => (sf.projections.len(), sf.sketch.entities.len()),
            _ => return,
        };
        for edge in edges {
            if let Err(e) = work.project_edge(fid, edge) {
                self.toast(e, true);
                return;
            }
        }
        if !linked {
            let Some(FeatureKind::Sketch(sf)) = work.feature_mut(fid).map(|f| &mut f.kind) else {
                return;
            };
            sf.projections.truncate(start);
        }
        if let Mode::Sketch(s) = &mut self.mode {
            s.set_tool(Tool::Select);
        }
        let unchanged = matches!(work.feature(fid).map(|f| &f.kind), Some(FeatureKind::Sketch(sf)) if sf.sketch.entities.len() == initial_entities && sf.projections.len() == start);
        if unchanged {
            self.toast("Selected geometry is already projected", false);
            return;
        }
        self.doc = work;
        self.sketch_outcome(
            Some(before),
            Outcome {
                changed: true,
                message: None,
            },
        );
    }

    fn projection_window(&mut self, ctx: &egui::Context) {
        let Mode::Sketch(s) = &mut self.mode else {
            return;
        };
        if s.tool != Tool::Project {
            return;
        }
        let mut ok = false;
        let mut cancel = false;
        let previous = (s.project_bodies, s.project_edges.len());
        egui::Window::new("PROJECT")
            .anchor(egui::Align2::RIGHT_TOP, [-16., 250.])
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                let mut active = true;
                if selection_field(ui, "Geometry", s.project_edges.len(), &mut active, true) {
                    s.project_edges.clear();
                }
                egui::ComboBox::from_label("Selection Filter")
                    .selected_text(if s.project_bodies {
                        "Bodies"
                    } else {
                        "Specified Entities"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut s.project_bodies, false, "Specified Entities");
                        ui.selectable_value(&mut s.project_bodies, true, "Bodies");
                    });
                ui.checkbox(&mut s.projection_link, "Projection Link");
                ui.label("Select body edges or faces. Bodies selects all body edges.");
                ui.horizontal(|ui| {
                    ok = ui
                        .add_enabled(
                            !s.project_edges.is_empty(),
                            egui::Button::new("OK  (Enter)"),
                        )
                        .clicked();
                    cancel = ui.button("Cancel  (Esc)").clicked();
                });
            });
        if previous != (s.project_bodies, s.project_edges.len()) {
            self.click_epoch = self.click_epoch.wrapping_add(1);
            self.pending_clicks.clear();
        }
        if cancel {
            if let Mode::Sketch(s) = &mut self.mode {
                s.set_tool(Tool::Select);
            }
        }
        if ok {
            self.commit_projection();
        }
    }

    fn sketch_palette(&mut self, ctx: &egui::Context) {
        if !matches!(self.mode, Mode::Sketch(_)) {
            return;
        }
        self.projection_window(ctx);
        let mut act = None;
        egui::Window::new("SKETCH PALETTE")
            .anchor(
                egui::Align2::RIGHT_TOP,
                [
                    -16.0,
                    if matches!(&self.mode, Mode::Sketch(s) if s.tool == Tool::Project) {
                        470.0
                    } else {
                        250.0
                    },
                ],
            )
            .resizable(false)
            .default_open(true)
            .default_width(190.0)
            .max_width(210.0)
            .show(ctx, |ui| {
                if let Mode::Sketch(s) = &self.mode {
                    let choices: &[(Tool, &str)] = match s.tool {
                        Tool::Rectangle | Tool::CenterRectangle => &[
                            (Tool::Rectangle, "2-Point Rectangle"),
                            (Tool::CenterRectangle, "Center Rectangle"),
                        ],
                        Tool::Arc | Tool::ThreePointArc | Tool::TangentArc => &[
                            (Tool::Arc, "Center Point Arc"),
                            (Tool::ThreePointArc, "3-Point Arc"),
                            (Tool::TangentArc, "Tangent Arc"),
                        ],
                        _ => &[],
                    };
                    if !choices.is_empty() {
                        ui.strong("Feature Options");
                        for &(tool, label) in choices {
                            if ui.selectable_label(s.tool == tool, label).clicked() {
                                act = Some(Action::Tool(tool));
                            }
                        }
                        ui.separator();
                    }
                }
                ui.strong("Options");
                egui::Grid::new("palette").num_columns(2).show(ui, |ui| {
                    ui.label("Linetype");
                    if let Mode::Sketch(s) = &mut self.mode {
                        if ui
                            .selectable_label(s.construction, "Construction")
                            .on_hover_text(
                                "Draw construction geometry; X toggles selected geometry",
                            )
                            .clicked()
                        {
                            s.construction = !s.construction;
                            if !s.selection.is_empty() {
                                act = Some(Action::Construction);
                            }
                        }
                    }
                    ui.end_row();
                    ui.label("Look At");
                    if ui
                        .small_button("👁")
                        .on_hover_text("View the sketch plane straight on")
                        .clicked()
                    {
                        act = Some(Action::LookAt);
                    }
                    ui.end_row();
                    ui.label("Sketch Grid");
                    ui.checkbox(&mut self.view.sketch_grid, "");
                    ui.end_row();
                    ui.label("Snap");
                    ui.checkbox(&mut self.grid_snap, "")
                        .on_hover_text("G — snap to the grid (existing geometry snaps first)");
                    ui.end_row();
                    ui.label("Show Profile");
                    ui.checkbox(&mut self.view.profiles, "");
                    ui.end_row();
                    ui.label("Show Points");
                    ui.checkbox(&mut self.view.points, "");
                    ui.end_row();
                    ui.label("Show Dimensions");
                    ui.checkbox(&mut self.view.dimensions, "");
                    ui.end_row();
                    ui.label("Show Constraints");
                    ui.checkbox(&mut self.view.constraints, "");
                    ui.end_row();
                    ui.label("Show Construction Geometries");
                    ui.checkbox(&mut self.view.construction, "");
                    ui.end_row();
                    ui.label("Show Projected Geometries");
                    ui.checkbox(&mut self.view.projected, "");
                    ui.end_row();
                });
                ui.add_space(4.0);
                if ui
                    .add_sized(
                        [190.0, 24.0],
                        egui::Button::new(
                            RichText::new("Finish Sketch").color(Color32::from_rgb(80, 200, 120)),
                        ),
                    )
                    .clicked()
                {
                    act = Some(Action::FinishSketch);
                }
            });
        if let Some(a) = act {
            self.run_action(a);
        }
    }

    /// Grid and Snaps settings (shared by the navigation bar).
    fn grid_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.view.sketch_grid, "Layout Grid");
        ui.checkbox(&mut self.grid_snap, "Snap to Grid");
        ui.separator();
        ui.label("Grid Settings");
        let mut adaptive = self.grid_fixed.is_none();
        if ui.radio_value(&mut adaptive, true, "Adaptive").clicked() {
            self.grid_fixed = None;
        }
        ui.horizontal(|ui| {
            if ui.radio_value(&mut adaptive, false, "Fixed").clicked() && self.grid_fixed.is_none()
            {
                self.grid_fixed = Some(5.0);
            }
            if let Some(g) = self.grid_fixed.as_mut() {
                ui.add(
                    egui::DragValue::new(g)
                        .range(0.01..=1000.0)
                        .speed(0.1)
                        .suffix(" mm"),
                );
            }
        });
    }

    fn run_action(&mut self, a: Action) {
        if matches!(self.cmd, Command::Measure(_)) && !matches!(a, Action::Measure) {
            self.cancel_command();
        }
        match a {
            Action::Hole
            | Action::Shell
            | Action::Mirror
            | Action::RectPattern
            | Action::CircularPattern
            | Action::AngledPlane => {
                self.mode_to_model();
                let mut bodies: Vec<_> = self
                    .selection
                    .iter()
                    .filter_map(|p| match p {
                        Pick::Face { body, .. } | Pick::Edge { body, .. } => Some(*body),
                        _ => None,
                    })
                    .collect();
                bodies.sort_unstable();
                bodies.dedup();
                let kind = match a {
                    Action::Hole => {
                        let Some(body) = bodies
                            .first()
                            .copied()
                            .or_else(|| self.doc.state().bodies.first().map(|b| b.id))
                        else {
                            self.toast("Create a solid body first", true);
                            return;
                        };
                        let face = self.selection.iter().find_map(|p| match *p {
                            Pick::Face { body: b, face } if b == body => {
                                self.doc.state().face_ref(b, face)
                            }
                            _ => None,
                        });
                        let mut hole = HoleFeature::new(
                            body,
                            face.clone().map(PlaneRef::Face).unwrap_or(PlaneRef::XY),
                        );
                        hole.body_tag = self
                            .doc
                            .state()
                            .body(body)
                            .and_then(|b| b.shape.face_tags().first().copied());
                        if face.is_none() {
                            hole.centers.clear();
                        }
                        FeatureKind::Hole(hole)
                    }
                    Action::Shell => {
                        let Some(body) = bodies
                            .first()
                            .or_else(|| self.doc.state().bodies.first().map(|b| &b.id))
                        else {
                            self.toast("Create a solid body first", true);
                            return;
                        };
                        FeatureKind::Shell(ShellFeature {
                            body: *body,
                            body_tag: self
                                .doc
                                .state()
                                .body(*body)
                                .and_then(|b| b.shape.face_tags().first().copied()),
                            faces: vec![],
                            thickness: "1 mm".into(),
                            outward: false,
                            outside_thickness: None,
                        })
                    }
                    Action::Mirror => FeatureKind::Mirror(MirrorFeature {
                        join: false,
                        bodies,
                        plane: PlaneRef::YZ,
                    }),
                    Action::RectPattern => FeatureKind::Pattern(PatternFeature {
                        objects: PatternObjects::Bodies,
                        features: vec![],
                        bodies,
                        kind: PatternKind::Rectangular {
                            axis1: LinearRef::X,
                            axis2: LinearRef::Y,
                            count1: "3".into(),
                            count2: "1".into(),
                            distance1: "20 mm".into(),
                            distance2: "20 mm".into(),
                            spacing: true,
                        },
                    }),
                    Action::CircularPattern => FeatureKind::Pattern(PatternFeature {
                        objects: PatternObjects::Bodies,
                        features: vec![],
                        bodies,
                        kind: PatternKind::Circular {
                            axis: LinearRef::Z,
                            count: "4".into(),
                            angle: "360 deg".into(),
                        },
                    }),
                    _ => FeatureKind::AngledPlane(AngledPlaneFeature {
                        base: PlaneRef::XY,
                        axis: LinearRef::X,
                        angle: "45 deg".into(),
                    }),
                };
                let pick = if matches!(kind, FeatureKind::Hole(_)) {
                    SolidPick::HoleCenter
                } else {
                    SolidPick::Bodies
                };
                self.cmd = Command::Solid {
                    edit: None,
                    kind,
                    pick,
                };
                self.rebuild_preview();
            }
            Action::OffsetPlane => self.start_offset_plane(None),
            Action::NewSketch => {
                self.cancel_command();
                self.cmd = Command::NewSketch;
            }
            Action::Extrude => self.start_extrude(None),
            Action::Revolve => self.start_revolve(None),
            Action::Fillet => self.start_blend(false, None),
            Action::Chamfer => self.start_blend(true, None),
            Action::Measure => self.start_measure(),
            Action::Params => {
                self.show_params = true;
                self.param_draft = self.doc.params.clone();
            }
            Action::Insert => self.insert_dialog(),
            Action::SelectModel => self.cancel_command(),
            Action::Tool(t) => {
                self.click_epoch = self.click_epoch.wrapping_add(1);
                self.pending_clicks.clear();
                if let Mode::Sketch(s) = &mut self.mode {
                    s.set_tool(t);
                }
            }
            Action::Constraint(k) => self.apply_constraint_tool(k),
            Action::Construction => {
                let Mode::Sketch(s) = &mut self.mode else {
                    return;
                };
                let fid = s.feature;
                let before = self.doc.clone();
                let Mode::Sketch(s) = &mut self.mode else {
                    return;
                };
                if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                    let o = s.toggle_construction(sk);
                    self.sketch_outcome(Some(before), o);
                }
            }
            Action::LookAt => {
                if let Mode::Sketch(s) = &self.mode {
                    let fr = s.frame;
                    self.cam.look_at_plane(
                        DVec3::from(fr.origin),
                        DVec3::from(fr.x),
                        DVec3::from(fr.y),
                    );
                }
            }
            Action::FinishSketch => self.finish_sketch(),
        }
    }

    fn insert_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("CAD models", &["step", "stp", "iges", "igs", "f3d"])
            .pick_file()
        {
            self.insert_path(p);
        }
    }

    fn apply_constraint_tool(&mut self, k: ConKind) {
        let before = self.doc.clone();
        let Mode::Sketch(s) = &mut self.mode else {
            return;
        };
        let sel = s.selected_entities();
        s.set_tool(Tool::Constrain(k));
        if sel.is_empty() {
            return;
        }
        let fid = s.feature;
        let Some(sk) = Self::sketch_mut(&mut self.doc, fid) else {
            return;
        };
        let per_entity = matches!(k, ConKind::HorizontalVertical | ConKind::Fix)
            && sel
                .iter()
                .all(|&e| sk.line_pts(e).is_some() || k == ConKind::Fix);
        let mut o = Outcome::default();
        if per_entity {
            for e in &sel {
                let r = crate::sketch_mode::apply_constraint(sk, k, &[*e]);
                o.changed |= r.changed;
                o.message = o.message.or(r.message);
            }
        } else {
            o = crate::sketch_mode::apply_constraint(sk, k, &sel);
        }
        s.selection.clear();
        self.sketch_outcome(Some(before), o);
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        ui.heading("Browser");
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::CollapsingHeader::new("Bodies")
                .default_open(true)
                .show(ui, |ui| {
                    let bodies: Vec<(BodyId, String, Option<f64>)> = self
                        .doc
                        .state()
                        .bodies
                        .iter()
                        .map(|b| {
                            (
                                b.id,
                                b.name.clone(),
                                self.meshes.get(&b.shape.uid()).and_then(|m| m.volume),
                            )
                        })
                        .collect();
                    if bodies.is_empty() {
                        ui.weak("(none)");
                    }
                    for (id, name, vol) in bodies {
                        let mut vis = !self.hidden_bodies.contains(&id);
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut vis, "").changed() {
                                if vis {
                                    self.hidden_bodies.remove(&id);
                                } else {
                                    self.hidden_bodies.insert(id);
                                }
                            }
                            ui.label(name).on_hover_text(
                                vol.map(|v| format!("Volume {v:.2} mm³"))
                                    .unwrap_or_else(|| "Preparing body geometry…".into()),
                            );
                        });
                    }
                });
            egui::CollapsingHeader::new("Sketches")
                .default_open(true)
                .show(ui, |ui| {
                    let sketches: Vec<(FeatureId, String)> = self
                        .doc
                        .features
                        .iter()
                        .take(self.doc.marker)
                        .filter(|f| matches!(f.kind, FeatureKind::Sketch(_)))
                        .map(|f| (f.id, f.name.clone()))
                        .collect();
                    if sketches.is_empty() {
                        ui.weak("(none)");
                    }
                    for (id, name) in sketches {
                        let mut vis = !self.hidden_sketches.contains(&id);
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut vis, "").changed() {
                                if vis {
                                    self.hidden_sketches.remove(&id);
                                } else {
                                    self.hidden_sketches.insert(id);
                                }
                            }
                            if ui.link(name).on_hover_text("Edit sketch").clicked()
                                && !matches!(&self.mode,Mode::Sketch(s)if s.feature==id)
                            {
                                let m = self.doc.marker;
                                self.mode_to_model();
                                self.enter_sketch(id, m);
                            }
                        });
                    }
                });
            egui::CollapsingHeader::new("Construction")
                .default_open(true)
                .show(ui, |ui| {
                    let planes: Vec<_> = self
                        .doc
                        .state()
                        .planes
                        .iter()
                        .map(|(&id, &frame)| {
                            (
                                id,
                                frame,
                                self.doc
                                    .feature(id)
                                    .map(|f| f.name.clone())
                                    .unwrap_or_default(),
                            )
                        })
                        .collect();
                    if planes.is_empty() {
                        ui.weak("(none)");
                    }
                    for (id, frame, name) in planes {
                        ui.horizontal(|ui| {
                            let mut visible = !self.hidden_planes.contains(&id);
                            if ui.checkbox(&mut visible, "").changed() {
                                if visible {
                                    self.hidden_planes.remove(&id);
                                } else {
                                    self.hidden_planes.insert(id);
                                }
                            }
                            if ui.link(name).on_hover_text("Edit offset plane").clicked() {
                                self.start_offset_plane(Some(id));
                            }
                            if ui.small_button("Sketch").clicked() {
                                self.mode_to_model();
                                self.start_new_sketch(PlaneRef::Construction(id), frame);
                            }
                        });
                    }
                });
            if !self.doc.params.is_empty() {
                egui::CollapsingHeader::new("Parameters")
                    .default_open(false)
                    .show(ui, |ui| {
                        let vals = self.doc.param_quantities().unwrap_or_default();
                        for p in &self.doc.params {
                            ui.label(format!(
                                "{} = {} ({})",
                                p.name,
                                p.expr,
                                vals.0
                                    .get(&p.name)
                                    .map(|v| sketch_fmt(v.value))
                                    .unwrap_or("?".into())
                            ));
                        }
                    });
            }
        });
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        let mut action: Option<(&'static str, FeatureId)> = None;
        let mut set_marker: Option<usize> = None;
        ui.horizontal(|ui| {
            let n = self.doc.features.len();
            let m = self.doc.marker;
            if ui.small_button("⏮").on_hover_text("Roll to start").clicked() {
                set_marker = Some(0);
            }
            if ui.small_button("◀").on_hover_text("Step back").clicked() && m > 0 {
                set_marker = Some(m - 1);
            }
            if ui.small_button("▶").on_hover_text("Step forward").clicked() && m < n {
                set_marker = Some(m + 1);
            }
            if ui.small_button("⏭").on_hover_text("Roll to end").clicked() {
                set_marker = Some(n);
            }
            ui.separator();
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, f) in self.doc.features.iter().enumerate() {
                        if i == m {
                            ui.label(RichText::new("┃").color(Color32::from_rgb(255, 170, 40)).strong());
                        }
                        use crate::tool_icons::Icon as I;
                        let icon = match f.kind {
                            FeatureKind::Sketch(_)=>I::Sketch,FeatureKind::Extrude(_)=>I::Extrude,FeatureKind::Revolve(_)=>I::Revolve,
                            FeatureKind::Fillet(_)=>I::Fillet,FeatureKind::Chamfer(_)=>I::Chamfer,FeatureKind::Import(_)=>I::Insert,
                            FeatureKind::OffsetPlane(_)=>I::OffsetPlane,FeatureKind::AngledPlane(_)=>I::AngledPlane,
                            FeatureKind::Shell(_)=>I::Shell,FeatureKind::Hole(_)=>I::Hole,FeatureKind::Mirror(_)=>I::Mirror,
                            FeatureKind::Pattern(PatternFeature {kind:PatternKind::Rectangular {..},..})=>I::RectPattern,
                            FeatureKind::Pattern(_)=>I::CircularPattern,
                        };
                        let mut text = RichText::new(format!("      {}", f.name));
                        if let Some(_e) = self.doc.errors.get(&f.id) {
                            text = text.color(Color32::from_rgb(240, 80, 70));
                        }
                        if f.suppressed {
                            text = text.strikethrough().weak();
                        }
                        if i >= m {
                            text = text.weak();
                        }
                        let selected = match &self.cmd {
                            Command::Solid { kind: FeatureKind::Pattern(p), .. } if p.objects == PatternObjects::Features => p.features.contains(&f.id),
                            _ => self.timeline_sel == Some(f.id),
                        };
                        let r = ui.selectable_label(selected, text);
                        let mut painter=ui.painter().clone();
                        painter.set_opacity(if f.suppressed || i >= m {0.4}else{1.});
                        crate::tool_icons::paint(&painter,egui::Rect::from_center_size(Pos2::new(r.rect.left()+11.,r.rect.center().y),Vec2::splat(18.)),icon);
                        let r = if let Some(e) = self.doc.errors.get(&f.id) { r.on_hover_text(e) } else { r.on_hover_text(f.kind.type_name()) };
                        if matches!(&self.cmd, Command::Solid { kind: FeatureKind::Pattern(p), .. } if p.objects == PatternObjects::Features) {
                            if r.clicked() && f.kind.supports_feature_pattern() && !f.suppressed && !self.doc.errors.contains_key(&f.id) {
                                let end = match &self.cmd { Command::Solid { edit: Some(id), .. } => self.doc.index_of(*id).unwrap_or(m), _ => m };
                                if i < end { action = Some(("pattern_select", f.id)); }
                            }
                        } else if r.double_clicked() {
                            action = Some(("edit", f.id));
                        } else if r.clicked() {
                            self.timeline_sel = Some(f.id);
                        }
                        r.context_menu(|ui| {
                            if ui.button("Edit").clicked() {
                                action = Some(("edit", f.id));
                                ui.close_menu();
                            }
                            if ui.button(if f.suppressed { "Unsuppress" } else { "Suppress" }).clicked() {
                                action = Some(("suppress", f.id));
                                ui.close_menu();
                            }
                            if ui.button("Roll history marker here").clicked() {
                                action = Some(("roll", f.id));
                                ui.close_menu();
                            }
                            if ui.button("Delete").clicked() {
                                action = Some(("delete", f.id));
                                ui.close_menu();
                            }
                        });
                    }
                    if m == n {
                        ui.label(RichText::new("┃").color(Color32::from_rgb(255, 170, 40)).strong());
                    }
                });
            });
        });
        if let Some(m) = set_marker {
            self.mode_to_model();
            self.timed_recompute(move |d| d.set_marker(m));
        }
        if let Some((what, id)) = action {
            match what {
                "pattern_select" => {
                    if let Command::Solid {
                        kind: FeatureKind::Pattern(p),
                        pick,
                        ..
                    } = &mut self.cmd
                    {
                        *pick = SolidPick::Bodies;
                        if p.features.contains(&id) {
                            p.features.retain(|v| *v != id);
                        } else {
                            p.features.push(id);
                        }
                    }
                    self.rebuild_preview();
                }
                "edit" => self.edit_feature(id),
                "suppress" => {
                    self.snapshot();
                    if let Some(f) = self.doc.feature_mut(id) {
                        f.suppressed = !f.suppressed;
                    }
                    self.timed_recompute(move |d| d.changed(id));
                }
                "roll" => {
                    let i = self.doc.index_of(id).unwrap_or(0);
                    self.mode_to_model();
                    self.timed_recompute(move |d| d.set_marker(i + 1));
                }
                "delete" => {
                    self.mode_to_model();
                    self.snapshot();
                    self.timed_recompute(move |d| d.remove_feature(id));
                }
                _ => {}
            }
        }
    }

    fn edit_feature(&mut self, id: FeatureId) {
        let Some(kind) = self.doc.feature(id).map(|f| f.kind.type_name()) else {
            return;
        };
        match kind {
            "Sketch" => {
                let m = self.doc.marker;
                self.mode_to_model();
                self.enter_sketch(id, m);
            }
            "Extrude" => self.start_extrude(Some(id)),
            "Revolve" => self.start_revolve(Some(id)),
            "Fillet" => self.start_blend(false, Some(id)),
            "Chamfer" => self.start_blend(true, Some(id)),
            "Hole"
            | "Shell"
            | "Mirror"
            | "Pattern"
            | "Rectangular Pattern"
            | "Circular Pattern"
            | "Plane at Angle" => {
                self.mode_to_model();
                if let Some(f) = self.doc.feature(id) {
                    let pick = if matches!(f.kind, FeatureKind::Hole(_)) {
                        SolidPick::HoleCenter
                    } else {
                        SolidPick::Bodies
                    };
                    self.cmd = Command::Solid {
                        edit: Some(id),
                        kind: f.kind.clone(),
                        pick,
                    };
                    self.rebuild_preview();
                }
            }
            "Offset Plane" => self.start_offset_plane(Some(id)),
            _ => {}
        }
    }

    fn command_window(&mut self, ctx: &egui::Context) {
        if matches!(self.cmd, Command::None)
            || (matches!(self.mode, Mode::Sketch(_)) && !matches!(self.cmd, Command::Measure(_)))
        {
            return;
        }
        let title = match &self.cmd {
            Command::Solid { kind, .. } => kind.type_name(),
            Command::NewSketch => "Create Sketch",
            Command::OffsetPlane { .. } => "Offset Plane",
            Command::Extrude(_) => "Extrude",
            Command::Revolve(_) => "Revolve",
            Command::Blend(BlendCmd { chamfer: None, .. }) => "Fillet",
            Command::Blend(_) => "Chamfer",
            Command::Measure(_) => "Measure",
            Command::None => "",
        };
        let mut changed = false;
        let mut ok = false;
        let mut cancel = false;
        let bodies = match &self.cmd {
            Command::Solid { edit: Some(id), .. } => self.doc.state_before(*id).bodies.clone(),
            _ => self.doc.state().bodies.clone(),
        };
        let pattern_end = match &self.cmd {
            Command::Solid { edit: Some(id), .. } => {
                self.doc.index_of(*id).unwrap_or(self.doc.marker)
            }
            _ => self.doc.marker,
        };
        let pattern_features: Vec<_> = self
            .doc
            .features
            .iter()
            .take(pattern_end)
            .filter(|f| {
                f.kind.supports_feature_pattern()
                    && !f.suppressed
                    && !self.doc.errors.contains_key(&f.id)
            })
            .map(|f| (f.id, f.name.clone()))
            .collect();
        let edges_count_hint = bodies.len();
        let parameter_values = self.doc.param_quantities().unwrap_or_default();
        egui::Window::new(title.to_uppercase()).anchor(egui::Align2::RIGHT_TOP, [-16.0, 250.0]).resizable(false).collapsible(false).show(ctx, |ui| {
            ui.set_min_width(250.0);
            match &mut self.cmd {
                Command::Solid { kind, pick, .. } => match kind {
                    FeatureKind::Hole(f) => {
                        ui.label("Placement: At Point on a planar face");
                        ui.selectable_value(pick, SolidPick::HoleCenter, "Face / Point: Select or reposition");
                        ui.label("Shift-click adds centers on the same face. Coordinates are local to the face.");
                        let mut remove = None;
                        egui::ScrollArea::vertical().max_height(150.).id_salt("hole_centers").show(ui, |ui| {
                        for (index, center) in f.centers.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!("Center {}", index+1));
                                if ui.small_button("×").clicked() { remove = Some(index); }
                            });
                            changed |= text_field(ui, "X", &mut center[0]);
                            changed |= text_field(ui, "Y", &mut center[1]);
                        }
                        });
                        if let Some(index) = remove { f.centers.remove(index); changed = true; }
                        changed |= text_field(ui, "Diameter", &mut f.diameter);
                        let mut ty = match f.kind { HoleKind::Simple => 0, HoleKind::Counterbore { .. } => 1, HoleKind::Countersink { .. } => 2 };
                        let before = ty;
                        egui::ComboBox::from_label("Hole Type").selected_text(["Simple", "Counterbore", "Countersink"][ty]).show_ui(ui, |ui| {
                            ui.selectable_value(&mut ty, 0, "Simple"); ui.selectable_value(&mut ty, 1, "Counterbore"); ui.selectable_value(&mut ty, 2, "Countersink");
                        });
                        if ty != before { f.kind = match ty { 1 => HoleKind::Counterbore { diameter: "8 mm".into(), depth: "2 mm".into() }, 2 => HoleKind::Countersink { diameter: "8 mm".into(), angle: "90 deg".into() }, _ => HoleKind::Simple }; changed = true; }
                        match &mut f.kind {
                            HoleKind::Counterbore { diameter, depth } => { changed |= text_field(ui, "Counterbore Diameter", diameter); changed |= text_field(ui, "Counterbore Depth", depth); }
                            HoleKind::Countersink { diameter, angle } => { changed |= text_field(ui, "Countersink Diameter", diameter); changed |= text_field(ui, "Countersink Angle", angle); }
                            _ => {}
                        }
                        let mut extent = match f.extent { HoleExtent::Distance { .. } => 0, HoleExtent::ThroughAll => 1, HoleExtent::ToFace { .. } => 2 };
                        let old_extent = extent;
                        egui::ComboBox::from_label("Extents").selected_text(["Distance", "All", "To"][extent]).show_ui(ui, |ui| {
                            ui.selectable_value(&mut extent, 0, "Distance"); ui.selectable_value(&mut extent, 1, "All");
                            ui.add_enabled_ui(matches!(f.plane, PlaneRef::Face(_)), |ui| { ui.selectable_value(&mut extent, 2, "To"); });
                        });
                        if extent != old_extent {
                            f.extent = match extent { 1 => HoleExtent::ThroughAll, 2 => {
                                let PlaneRef::Face(face) = &f.plane else { unreachable!() };
                                *pick = SolidPick::HoleTarget;
                                HoleExtent::ToFace { face: face.clone(), offset: "0 mm".into() }
                            }, _ => HoleExtent::Distance { depth: "5 mm".into() } }; changed = true;
                        }
                        match &mut f.extent {
                            HoleExtent::Distance { depth } => { changed |= text_field(ui, "Depth", depth); ui.label("Depth measures the cylindrical bore; angled tip extends below."); }
                            HoleExtent::ToFace { offset, .. } => { ui.selectable_value(pick, SolidPick::HoleTarget, "To Face: Select / Reselect"); changed |= text_field(ui, "Offset", offset); }
                            _ => {}
                        }
                        let mut angled = f.tip_angle.is_some(); let before = angled;
                        egui::ComboBox::from_label("Drill Point").selected_text(if angled { "Angle" } else { "Flat" }).show_ui(ui, |ui| {
                            ui.selectable_value(&mut angled, false, "Flat"); ui.selectable_value(&mut angled, true, "Angle");
                        });
                        if angled != before { f.tip_angle = if angled { Some("118 deg".into()) } else { None }; changed = true; }
                        if let Some(angle) = &mut f.tip_angle { changed |= text_field(ui, "Tip Angle", angle); }
                        changed |= ui.checkbox(&mut f.flip, "Flip Direction").changed();
                        let target = bodies.iter().find(|b| f.body_tag.map_or(b.id == f.body, |tag| b.shape.face_tags().contains(&tag)));
                        ui.label(format!("Object To Cut: {}", target.map(|b| b.name.clone()).unwrap_or_else(|| format!("Body {}", f.body))));
                    }
                    FeatureKind::Shell(f) => {
                        let selected = bodies.iter().find(|b| f.body_tag.map_or(b.id == f.body, |tag| b.shape.face_tags().contains(&tag)));
                        egui::ComboBox::from_label("Body").selected_text(selected.map(|b| b.name.clone()).unwrap_or_else(|| format!("Body {}", f.body))).show_ui(ui, |ui| {
                            for b in &bodies {
                                if ui.selectable_value(&mut f.body, b.id, &b.name).changed() {
                                    f.faces.clear();
                                    f.body_tag = b.shape.face_tags().first().copied();
                                    changed = true;
                                }
                            }
                        });
                        let mut direction = if f.outside_thickness.is_some() { 2 } else if f.outward { 1 } else { 0 };
                        let old_direction = direction;
                        egui::ComboBox::from_label("Direction").selected_text(["Inside", "Outside", "Both"][direction]).show_ui(ui, |ui| {
                            ui.selectable_value(&mut direction, 0, "Inside");
                            ui.selectable_value(&mut direction, 1, "Outside");
                            ui.selectable_value(&mut direction, 2, "Both");
                        });
                        if direction != old_direction {
                            f.outward = direction == 1;
                            f.outside_thickness = if direction == 2 { Some("1 mm".into()) } else { None };
                            changed = true;
                        }
                        changed |= text_field(ui, if direction == 1 { "Outside Thickness" } else { "Inside Thickness" }, &mut f.thickness);
                        if let Some(outside) = &mut f.outside_thickness { changed |= text_field(ui, "Outside Thickness", outside); }
                            let mut active = true;
                            ui.label("Select faces to remove; no faces creates a closed shell.");
                            if selection_field(ui, "Faces", f.faces.len(), &mut active, true) {
                            f.faces.clear();
                            changed = true;
                        }
                    }
                    FeatureKind::Mirror(f) => {
                        ui.label("Object Type: Bodies");
                        egui::ComboBox::from_label("Operation").selected_text(if f.join { "Join" } else { "New Body" }).show_ui(ui, |ui| {
                            changed |= ui.selectable_value(&mut f.join, false, "New Body").changed();
                            changed |= ui.selectable_value(&mut f.join, true, "Join").changed();
                        });
                            let mut plane_mode = *pick == SolidPick::Plane;
                            if selection_field(ui, "Objects", f.bodies.len(), &mut plane_mode, false) {
                                f.bodies.clear();
                                changed = true;
                            }
                            *pick = if plane_mode { SolidPick::Plane } else { SolidPick::Bodies };
                            changed |= body_choices(ui, &bodies, &mut f.bodies);
                        changed |= plane_choices(ui, &mut f.plane);
                        ui.selectable_value(pick, SolidPick::Plane, "Mirror Plane: Select / Reselect");
                    }
                    FeatureKind::AngledPlane(f) => {
                        changed |= plane_choices(ui, &mut f.base);
                        changed |= axis_choices(ui, "Axis", &mut f.axis);
                        changed |= text_field(ui, "Angle", &mut f.angle);
                        ui.selectable_value(pick, SolidPick::Plane, "Reference Plane: Select / Reselect");
                        ui.selectable_value(pick, SolidPick::Axis1, "Axis: Select / Reselect straight edge");
                    }
                    FeatureKind::Pattern(f) => {
                        let old_type = f.objects;
                        egui::ComboBox::from_label("Object Type").selected_text(if f.objects == PatternObjects::Bodies { "Bodies" } else { "Features" }).show_ui(ui, |ui| {
                            changed |= ui.selectable_value(&mut f.objects, PatternObjects::Bodies, "Bodies").changed();
                            changed |= ui.selectable_value(&mut f.objects, PatternObjects::Features, "Features").changed();
                        });
                        if old_type != f.objects { f.bodies.clear(); f.features.clear(); *pick = SolidPick::Bodies; }
                        if f.objects == PatternObjects::Bodies {
                            changed |= body_choices(ui, &bodies, &mut f.bodies);
                        } else {
                            ui.label("Select solid features below or in the timeline.");
                            egui::ScrollArea::vertical().id_salt("pattern_features").max_height(150.).show(ui, |ui| {
                                for (id, name) in &pattern_features {
                                    let mut selected = f.features.contains(id);
                                    if ui.checkbox(&mut selected, name).changed() {
                                        *pick = SolidPick::Bodies;
                                        if selected { f.features.push(*id); } else { f.features.retain(|v| v != id); }
                                        changed = true;
                                    }
                                }
                            });
                            ui.label("Compute Type: Identical");
                            ui.label("Copies retain the source depth and termination.");
                        }
                        let count = if f.objects == PatternObjects::Bodies { f.bodies.len() } else { f.features.len() };
                        ui.selectable_value(pick, SolidPick::Bodies, format!("Objects: {count} selected — Select"));
                        if ui.add_enabled(count > 0, egui::Button::new("Clear objects")).clicked() {
                            f.bodies.clear(); f.features.clear(); changed = true;
                        }
                            match &mut f.kind {
                            PatternKind::Rectangular { axis1, axis2, count1, count2, distance1, distance2, spacing } => {
                                changed |= axis_choices(ui, "Direction 1", axis1);
                                changed |= text_field(ui, "Quantity 1", count1);
                                changed |= text_field(ui, "Distance 1", distance1);
                                changed |= axis_choices(ui, "Direction 2", axis2);
                                changed |= text_field(ui, "Quantity 2", count2);
                                changed |= text_field(ui, "Distance 2", distance2);
                                ui.selectable_value(pick, SolidPick::Axis1, "Pick direction 1 edge");
                                ui.selectable_value(pick, SolidPick::Axis2, "Pick direction 2 edge");
                                    egui::ComboBox::from_label("Distribution")
                                        .selected_text(if *spacing { "Spacing" } else { "Extent" })
                                        .show_ui(ui, |ui| {
                                            changed |= ui.selectable_value(spacing, false, "Extent").changed();
                                            changed |= ui.selectable_value(spacing, true, "Spacing").changed();
                                        });
                            }
                            PatternKind::Circular { axis, count, angle } => {
                                changed |= axis_choices(ui, "Axis", axis);
                                changed |= text_field(ui, "Quantity", count);
                                let mut full = expr::eval(angle, Unit::Angle, &parameter_values).is_ok_and(|a| (a.abs() - std::f64::consts::TAU).abs() < 1e-9);
                                egui::ComboBox::from_label("Distribution").selected_text(if full { "Full" } else { "Partial" }).show_ui(ui, |ui| {
                                    if ui.selectable_value(&mut full, true, "Full").changed() { *angle = "360 deg".into(); changed = true; }
                                    if ui.selectable_value(&mut full, false, "Partial").changed() { *angle = "90 deg".into(); changed = true; }
                                });
                                if !full { changed |= text_field(ui, "Angle", angle); }
                                ui.selectable_value(pick, SolidPick::Axis1, "Axis: Select / Reselect straight edge");
                            }
                        }
                        ui.label("Quantity includes the original; maximum 1000 instances.");
                    }
                    _ => {}
                },
                Command::NewSketch => {
                    ui.label("Pick an origin plane, construction plane or a planar face.");
                }
                Command::OffsetPlane { base, distance, .. } => {
                        let mut active = true;
                        if selection_field(ui, "Plane", usize::from(base.is_some()), &mut active, true) {
                            *base = None;
                            changed = true;
                        }
                        ui.label("Select an origin plane, construction plane or planar face.");
                    ui.horizontal(|ui| {
                        ui.label("Distance");
                        changed |= ui.add(egui::TextEdit::singleline(distance).desired_width(120.0)).changed();
                    });
                    ui.label("Positive offsets follow the base plane normal.");
                }
                Command::Extrude(c) => {
                        if selection_field(ui, "Profiles", c.profiles.len() + c.faces.len(), &mut c.pick_target, false) {
                            c.profiles.clear();
                            c.faces.clear();
                            changed = true;
                        }
                        ui.label("Start: Profile Plane");
                        let mut direction = match c.kind {
                            ExtentKind::Symmetric => ExtentKind::Symmetric,
                            ExtentKind::TwoSides => ExtentKind::TwoSides,
                            _ => ExtentKind::OneSide,
                        };
                    egui::ComboBox::from_label("Direction")
                            .selected_text(match direction {
                                ExtentKind::Symmetric => "Symmetric",
                                ExtentKind::TwoSides => "Two Sides",
                                _ => "One Side",
                            })
                            .show_ui(ui, |ui| {
                                for (v, label) in [
                                    (ExtentKind::OneSide, "One Side"),
                                    (ExtentKind::TwoSides, "Two Sides"),
                                    (ExtentKind::Symmetric, "Symmetric"),
                                ] {
                                    if ui.selectable_value(&mut direction, v, label).changed() {
                                        c.kind = v;
                                        c.pick_target = false;
                                        changed = true;
                                    }
                                }
                            });
                        if direction == ExtentKind::OneSide {
                            egui::ComboBox::from_label("Extent Type")
                        .selected_text(match c.kind {
                            ExtentKind::ToFace => "To Object",
                            ExtentKind::ThroughAll => "All",
                                    _ => "Distance",
                        })
                        .show_ui(ui, |ui| {
                                    for (v, label) in [
                                        (ExtentKind::OneSide, "Distance"),
                                        (ExtentKind::ToFace, "To Object"),
                                        (ExtentKind::ThroughAll, "All"),
                                    ] {
                                        if ui.selectable_value(&mut c.kind, v, label).changed() {
                                c.pick_target = v == ExtentKind::ToFace && c.target.is_none();
                                changed = true;
                            }
                                    }
                                });
                        } else {
                            ui.label("Extent Type: Distance");
                        }
                        if c.kind == ExtentKind::ThroughAll {
                        egui::ComboBox::new("extrude-all-direction", "All Direction").selected_text(match c.all_direction { AllDirection::Positive => "Positive", AllDirection::Negative => "Negative", AllDirection::Both => "Both Sides" }).show_ui(ui, |ui| {
                            for (v, n) in
                                [(AllDirection::Positive, "Positive"), (AllDirection::Negative, "Negative"), (AllDirection::Both, "Both sides"),
                                ]
                            {
                                changed |= ui.selectable_value(&mut c.all_direction, v, n).changed();
                            }
                        });
                    }
                    if c.kind == ExtentKind::ToFace {
                            if selection_field(ui, "Object", usize::from(c.target.is_some()), &mut c.pick_target, true) {
                                c.target = None;
                                c.pick_target = true;
                                changed = true;
                        }
                            ui.label("Planar faces only; extends to the underlying plane.");
                        ui.horizontal(|ui| {
                            ui.label("Offset");
                            changed |= ui
                                .add(egui::TextEdit::singleline(&mut c.offset).desired_width(120.))
                                .on_hover_text("Signed offset along the profile normal")
                                .changed();
                        });
                    }
                    if !matches!(c.kind, ExtentKind::ThroughAll | ExtentKind::ToFace) {
                        ui.horizontal(|ui| {
                            ui.label(if c.kind == ExtentKind::TwoSides { "Side 1" } else { "Distance" });
                            changed |= ui.add(egui::TextEdit::singleline(&mut c.d1).desired_width(120.0)).changed();
                            if c.kind == ExtentKind::OneSide && ui.button("⇅").on_hover_text("Flip direction").clicked() {
                                c.d1 = if let Some(rest) = c.d1.trim().strip_prefix('-') {
                                    rest.trim().to_string()
                                } else {
                                    format!("-({})", c.d1.trim())
                                };
                                changed = true;
                            }
                        });
                    }
                    if c.kind == ExtentKind::Symmetric {
                        egui::ComboBox::from_label("Measurement").selected_text(if c.half_length { "Half Length" } else { "Whole Length" }).show_ui(ui, |ui| {
                            changed |= ui.selectable_value(&mut c.half_length, true, "Half Length").changed();
                            changed |= ui.selectable_value(&mut c.half_length, false, "Whole Length").changed();
                        });
                    }
                    if c.kind == ExtentKind::TwoSides {
                        ui.horizontal(|ui| {
                            ui.label("Side 2");
                            changed |= ui.add(egui::TextEdit::singleline(&mut c.d2).desired_width(120.0)).changed();
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Taper Angle");
                        changed |= ui
                            .add(egui::TextEdit::singleline(&mut c.taper).desired_width(120.))
                            .on_hover_text("Positive angles narrow away from the profile; negative angles widen. Angle expressions are supported.",
                                )
                            .changed();
                    });
                    if op_combo(ui, &mut c.op) {
                        c.op_user_set = true;
                        changed = true;
                    }
                }
                Command::Revolve(c) => {
                        if selection_field(ui, "Profile", c.profiles.len() + c.faces.len(), &mut c.pick_axis, false) {
                            c.profiles.clear();
                            c.faces.clear();
                            changed = true;
                        }
                        if selection_field(ui, "Axis", usize::from(c.axis.is_some()), &mut c.pick_axis, true) {
                            c.axis = None;
                            c.pick_axis = true;
                            changed = true;
                        }
                        ui.label("Select a sketch line, linear/circular edge or origin axis.");
                    ui.horizontal(|ui| {
                        ui.label("Axis");
                        let txt = match &c.axis {
                            None => "pick a sketch line…".to_string(),
                            Some(Axis::SketchLine(e)) => format!("sketch line #{}", e.0),
                            Some(Axis::Edge(_)) => "Selected model edge".into(),
                            Some(a) => format!("{a:?} axis"),
                        };
                        egui::ComboBox::from_id_salt("axis").selected_text(txt).show_ui(ui, |ui| {
                            for (a, n) in [(Axis::X, "X axis"), (Axis::Y, "Y axis"), (Axis::Z, "Z axis")] {
                                if ui.selectable_label(c.axis.as_ref() == Some(&a), n).clicked() {
                                    c.axis = Some(a);
                                        c.pick_axis = false;
                                        changed = true;
                                }
                            }
                        });
                    });
                    changed |= ui.checkbox(&mut c.project_axis, "Project Axis").changed();
                    egui::ComboBox::from_label("Extent Type").selected_text(if c.full { "Full" } else { "Partial" }).show_ui(ui, |ui| {
                        changed |= ui.selectable_value(&mut c.full, true, "Full").changed();
                        if ui.selectable_value(&mut c.full, false, "Partial").changed() { if c.angle == "360 deg" { c.angle = "90 deg".into(); } changed = true; }
                    });
                    if !c.full {
                        egui::ComboBox::from_label("Direction").selected_text(match c.direction { RevolveDirection::OneSide => "One Side", RevolveDirection::TwoSides => "Two Sides", RevolveDirection::Symmetric => "Symmetric" }).show_ui(ui, |ui| {
                            for (value, label) in [(RevolveDirection::OneSide, "One Side"), (RevolveDirection::TwoSides, "Two Sides"), (RevolveDirection::Symmetric, "Symmetric")] { changed |= ui.selectable_value(&mut c.direction, value, label).changed(); }
                        });
                        changed |= text_field(ui, if c.direction == RevolveDirection::TwoSides { "Angle 1" } else { "Angle" }, &mut c.angle);
                        if c.direction == RevolveDirection::TwoSides { changed |= text_field(ui, "Angle 2", &mut c.angle2); }
                    }
                    changed |= op_combo(ui, &mut c.op);
                }
                Command::Blend(c) => {
                        let mut active = true;
                        if selection_field(ui, "Edges / Faces", c.edges.len(), &mut active, true) {
                            c.edges.clear();
                            changed = true;
                        }
                        ui.label("Select edges or a face for its boundary edges.");
                    if let Some(t) = &mut c.chamfer {
                        egui::ComboBox::from_label("Type")
                            .selected_text(match t {
                                ChamferType::Equal => "Equal Distance",
                                ChamferType::TwoDistances => "Two Distance",
                                ChamferType::DistanceAngle => "Distance And Angle",
                            })
                            .show_ui(ui, |ui| {
                                changed |= ui.selectable_value(t, ChamferType::Equal, "Equal Distance").changed();
                                changed |= ui.selectable_value(t, ChamferType::TwoDistances, "Two Distance").changed();
                                changed |= ui.selectable_value(t, ChamferType::DistanceAngle, "Distance And Angle").changed();
                            });
                    }
                    ui.horizontal(|ui| {
                        ui.label(if c.chamfer.is_some() { "Distance" } else { "Radius" });
                        changed |= ui.add(egui::TextEdit::singleline(&mut c.radius).desired_width(120.0)).changed();
                    });
                    match c.chamfer {
                        Some(ChamferType::TwoDistances) => {
                            ui.horizontal(|ui| {
                                ui.label("Distance 2");
                                changed |= ui.add(egui::TextEdit::singleline(&mut c.d2).desired_width(120.0)).changed();
                            });
                        }
                        Some(ChamferType::DistanceAngle) => {
                            ui.horizontal(|ui| {
                                ui.label("Angle");
                                changed |= ui.add(egui::TextEdit::singleline(&mut c.angle).desired_width(120.0)).changed();
                            });
                        }
                        _ => {}
                    }
                    ui.checkbox(&mut c.show_preview, "Preview result").on_hover_text("Untick to pick more edges on the unmodified body");
                    let _ = edges_count_hint;
                }
                Command::Measure(m) => {
                    let r = m.result.clone().unwrap_or_default();
                    egui::Grid::new("measure-sel").num_columns(2).show(ui, |ui| {
                        ui.label("Selection 1");
                        ui.label(if m.a.is_some() { r.name_a.as_str() } else { "pick a face, edge, vertex or sketch entity" });
                        ui.end_row();
                        ui.label("Selection 2");
                        ui.label(if m.b.is_some() { r.name_b.as_str() } else { "—" });
                        ui.end_row();
                    });
                    ui.separator();
                    egui::Grid::new("measure-res").num_columns(2).striped(true).show(ui, |ui| {
                        for (k, v) in &r.info {
                            ui.strong(k);
                            ui.monospace(v);
                            ui.end_row();
                        }
                    });
                    if let Some(e) = &r.error {
                        ui.colored_label(Color32::from_rgb(240, 90, 80), e);
                    }
                    ui.horizontal(|ui| {
                        ui.label("Precision");
                        for p in 0..=4 {
                            if ui.selectable_label(m.precision == p, format!("{p}")).clicked() {
                                m.precision = p;
                                changed = true;
                            }
                        }
                    });
                    if ui.button("Clear Selection").clicked() {
                        m.a = None;
                        m.b = None;
                        changed = true;
                    }
                }
                Command::None => {}
            }
            if let Some(e) = &self.preview_error {
                ui.colored_label(Color32::from_rgb(240, 90, 80), e);
            }
            ui.separator();
            ui.horizontal(|ui| {
                if self.preview_pending {
                    ui.spinner();
                    ui.label("Computing preview…");
                }
                if !matches!(self.cmd, Command::NewSketch | Command::Measure(_))
                    && ui
                        .add_enabled(
                            !self.preview_pending && !self.replay_pending && self.preview.is_some() && self.preview_error.is_none(),
                            egui::Button::new("OK  (Enter)"),
                        )
                        .clicked()
                {
                    ok = true;
                }
                if ui.button(if matches!(self.cmd, Command::Measure(_)) { "Close  (Esc)" } else { "Cancel  (Esc)" }).clicked() {
                    cancel = true;
                }
            });
        },
        );
        if changed {
            self.click_epoch = self.click_epoch.wrapping_add(1);
            self.rebuild_preview();
            self.refresh_measure();
        }
        if ok {
            self.commit_command();
        }
        if cancel {
            self.cancel_command();
        }
    }

    fn params_window(&mut self, ctx: &egui::Context) {
        if !self.show_params {
            return;
        }
        let mut open = true;
        let mut apply = false;
        egui::Window::new("Parameters")
            .open(&mut open)
            .resizable(true)
            .default_width(420.0)
            .show(ctx, |ui| {
                let vals = {
                    let mut d = Document::new();
                    d.params = self.param_draft.clone();
                    d.param_values()
                };
                egui::Grid::new("params")
                    .striped(true)
                    .num_columns(4)
                    .show(ui, |ui| {
                        ui.strong("Name");
                        ui.strong("Expression");
                        ui.strong("Value");
                        ui.label("");
                        ui.end_row();
                        let mut remove = None;
                        for (i, p) in self.param_draft.iter_mut().enumerate() {
                            ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(90.0));
                            ui.add(egui::TextEdit::singleline(&mut p.expr).desired_width(140.0));
                            match &vals {
                                Ok(v) => ui.label(
                                    v.get(&p.name).map(|x| sketch_fmt(*x)).unwrap_or_default(),
                                ),
                                Err(_) => ui.label("—"),
                            };
                            if ui.small_button("✖").clicked() {
                                remove = Some(i);
                            }
                            ui.end_row();
                        }
                        if let Some(i) = remove {
                            self.param_draft.remove(i);
                        }
                    });
                if let Err(e) = &vals {
                    ui.colored_label(Color32::from_rgb(240, 90, 80), e);
                }
                ui.horizontal(|ui| {
                    if ui.button("+ Add parameter").clicked() {
                        let n = self.param_draft.len() + 1;
                        self.param_draft.push(Param {
                            name: format!("p{n}"),
                            expr: "10 mm".into(),
                            comment: String::new(),
                        });
                    }
                    if ui
                        .add_enabled(vals.is_ok(), egui::Button::new("Apply"))
                        .clicked()
                    {
                        apply = true;
                    }
                });
                ui.weak(
                    "Use parameter names in any dimension or feature distance, e.g. width / 2.",
                );
            });
        if apply {
            self.snapshot();
            self.doc.params = self.param_draft.clone();
            self.timed_recompute(move |d| d.recompute_all());
            self.rebuild_preview();
        }
        if !open {
            self.show_params = false;
        }
    }

    // ------------------------------------------------------------------ viewport

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let profile_start = Instant::now();
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.viewport_rect = rect;
        let ppp = ui.ctx().pixels_per_point();
        let size = DVec2::new(rect.width() as f64, rect.height() as f64);
        let pointer = ui
            .input(|i| i.pointer.hover_pos())
            .filter(|p| rect.contains(*p));
        self.update_mesh_quality(
            2. * self.cam.distance * (self.cam.fov_y * 0.5).tan() / (size.y * ppp as f64).max(1.),
            ui.ctx(),
        );
        let local_px = |p: Pos2| DVec2::new((p.x - rect.min.x) as f64, (p.y - rect.min.y) as f64);

        let cube_zone = egui::Rect::from_center_size(
            Pos2::new(rect.max.x - 76.0, rect.min.y + 66.0),
            Vec2::splat(130.0),
        );
        let nav_zone = egui::Rect::from_center_size(
            Pos2::new(rect.center().x, rect.max.y - 22.0),
            Vec2::new(460.0, 34.0),
        );
        // ---- camera input
        let (scroll, delta, mmb, rmb, lmb, shift, alt) = ui.input(|i| {
            (
                i.smooth_scroll_delta.y,
                i.pointer.delta(),
                i.pointer.button_down(PointerButton::Middle),
                i.pointer.button_down(PointerButton::Secondary),
                i.pointer.button_down(PointerButton::Primary),
                i.modifiers.shift,
                i.modifiers.alt,
            )
        });
        if let Some(p) = pointer.filter(|p| {
            resp.contains_pointer() && !cube_zone.contains(*p) && !nav_zone.contains(*p)
        }) {
            if scroll != 0.0 {
                self.cam.zoom(scroll as f64, local_px(p), size);
            }
        }
        if resp.hovered() || resp.dragged() {
            if mmb {
                if shift {
                    self.cam.orbit(delta.x as f64, delta.y as f64);
                } else {
                    self.cam.pan(delta.x as f64, delta.y as f64, size.y);
                }
            }
            if rmb || (lmb && alt) {
                self.cam.orbit(delta.x as f64, delta.y as f64);
            }
        }
        if let Some(mode) = self.navigation {
            if resp.dragged_by(PointerButton::Primary) {
                match mode {
                    NavigationMode::Orbit => self.cam.orbit(delta.x as f64, delta.y as f64),
                    NavigationMode::Pan => self.cam.pan(delta.x as f64, delta.y as f64, size.y),
                    NavigationMode::Zoom => self.cam.zoom(-delta.y as f64 * 4.0, size * 0.5, size),
                }
                ui.ctx().request_repaint();
            }
        }
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().min(0.1);
        self.last_frame = now;
        if self.cam.tick(dt) {
            ui.ctx().request_repaint();
        }

        // ---- sketch-plane cursor
        let sketch_frame = if let Mode::Sketch(s) = &self.mode {
            Some(s.frame)
        } else {
            None
        };
        let cursor_local = |p: Pos2, cam: &Camera| -> Option<[f64; 2]> {
            let fr = sketch_frame?;
            let ray = cam.ray(local_px(p), size);
            let hit = ray.hit_plane(DVec3::from(fr.origin), DVec3::from(fr.normal()))?;
            Some(fr.to_local(hit.into()))
        };
        let cursor = pointer.and_then(|p| cursor_local(p, &self.cam));

        // ---- what to show
        self.refresh_analysis();
        let base_state: &State = match &self.cmd {
            Command::Blend(BlendCmd { edit: Some(id), .. })
            | Command::Solid { edit: Some(id), .. } => self.doc.state_before(*id),
            _ => self.doc.state(),
        };
        let show_preview = match &self.cmd {
            Command::Blend(c) => c.show_preview,
            _ => true,
        };
        let state: &State = match (&self.preview, show_preview) {
            (Some(p), true) => p.state(),
            _ => base_state,
        };
        let split_pick = matches!(
            self.cmd,
            Command::Blend(_) | Command::Extrude(_) | Command::Solid { .. }
        ) && !std::ptr::eq(state, base_state);
        let mut used = HashSet::new();
        for b in state.bodies.iter().chain(base_state.bodies.iter()) {
            let uid = b.shape.uid();
            used.insert(uid);
            // Worker results include all timeline meshes; rendering never meshes.
            self.meshes
                .entry(uid)
                .or_insert_with(|| Arc::new(Mesh::default()));
        }
        // Retain meshes for undo/timeline snapshots; new meshes are built by the worker.

        let active = if let Mode::Sketch(s) = &self.mode {
            Some(s.feature)
        } else {
            None
        };
        let mut sketches: Vec<(FeatureId, &Sketch, Frame, bool)> = Vec::new();
        for f in self.doc.features.iter().take(self.doc.marker) {
            if let FeatureKind::Sketch(sf) = &f.kind {
                let is_active = active == Some(f.id);
                if (self.hidden_sketches.contains(&f.id) && !is_active) || f.suppressed {
                    continue;
                }
                if let Some(ss) = self.doc.state().sketches.get(&f.id) {
                    sketches.push((f.id, &sf.sketch, ss.frame, is_active));
                }
            }
        }
        let live_profiles: Arc<Vec<Profile>>;
        let mut profile_sketches: Vec<(FeatureId, &[Profile], Frame)> = Vec::new();
        if let (Some(id), Some(fr)) = (active, sketch_frame) {
            live_profiles = if self.view.profiles {
                Self::sketch_ref(&self.doc, id)
                    .and_then(|sk| self.sketch_derived.get(sk))
                    .map(|data| {
                        self.profile_cache
                            .install(id, data.profiles.clone(), data.meshes.clone());
                        data.profiles.clone()
                    })
                    .unwrap_or_default()
            } else {
                Arc::default()
            };
            profile_sketches.push((id, &live_profiles, fr));
        } else if matches!(self.cmd, Command::Extrude(_) | Command::Revolve(_)) {
            for (id, _, _, _) in &sketches {
                if let Some(ss) = self.doc.state().sketches.get(id) {
                    profile_sketches.push((*id, &ss.profiles, ss.frame));
                }
            }
        }

        let filter = match (&self.mode, &self.cmd) {
            (_, Command::Measure(_)) => PickFilter {
                faces: true,
                edges: true,
                vertices: true,
                sketch_ents: true,
                ..Default::default()
            },
            (Mode::Sketch(s), _) if s.tool == Tool::Project => PickFilter {
                faces: true,
                edges: !s.project_bodies,
                ..Default::default()
            },
            (Mode::Sketch(_), _) => PickFilter {
                sketch_ents: true,
                ..Default::default()
            },
            (_, Command::NewSketch | Command::OffsetPlane { .. }) => PickFilter {
                faces: true,
                planes: true,
                ..Default::default()
            },
            (_, Command::Extrude(c)) if c.kind == ExtentKind::ToFace && c.pick_target => {
                PickFilter {
                    faces: true,
                    ..Default::default()
                }
            }
            (_, Command::Extrude(_)) => PickFilter {
                profiles: true,
                faces: true,
                ..Default::default()
            },
            (_, Command::Revolve(c)) if c.pick_axis => PickFilter {
                sketch_ents: true,
                edges: true,
                ..Default::default()
            },
            (_, Command::Revolve(_)) => PickFilter {
                profiles: true,
                faces: true,
                ..Default::default()
            },
            (_, Command::Blend(_)) => PickFilter {
                edges: true,
                faces: true,
                ..Default::default()
            },
            (
                _,
                Command::Solid {
                    pick: SolidPick::Plane,
                    ..
                },
            ) => PickFilter {
                faces: true,
                planes: true,
                ..Default::default()
            },
            (
                _,
                Command::Solid {
                    pick: SolidPick::Axis1 | SolidPick::Axis2,
                    ..
                },
            ) => PickFilter {
                edges: true,
                ..Default::default()
            },
            (
                _,
                Command::Solid {
                    kind: FeatureKind::Pattern(p),
                    ..
                },
            ) if p.objects == PatternObjects::Features => PickFilter::default(),
            (_, Command::Solid { .. }) => PickFilter {
                faces: true,
                ..Default::default()
            },
            (_, Command::None) => PickFilter {
                faces: true,
                edges: true,
                ..Default::default()
            },
        };

        // selection highlights for commands
        let mut selection: Vec<Pick> = self.selection.clone();
        if let Mode::Sketch(s) = &self.mode {
            selection = s
                .selection
                .iter()
                .filter_map(|x| {
                    if let crate::sketch_mode::SkSel::Ent(e) = x {
                        Some(Pick::Ent {
                            sketch: s.feature,
                            ent: *e,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            selection.extend(s.picked.iter().map(|e| Pick::Ent {
                sketch: s.feature,
                ent: *e,
            }));
            for edge in &s.project_edges {
                if let Some((bi, id)) = base_state.resolve_edge(edge) {
                    selection.push(Pick::Edge {
                        body: base_state.bodies[bi].id,
                        edge: id,
                    });
                }
            }
        }
        if let Command::Solid { kind, .. } = &self.cmd {
            let bodies: &[BodyId] = match kind {
                FeatureKind::Mirror(f) => &f.bodies,
                FeatureKind::Pattern(f) if f.objects == PatternObjects::Bodies => &f.bodies,
                _ => &[],
            };
            for body in bodies {
                if let Some(b) = base_state.body(*body) {
                    for face in 1..=b.shape.face_count() as u32 {
                        selection.push(Pick::Face { body: *body, face });
                    }
                }
            }
            let axes = match kind {
                FeatureKind::AngledPlane(f) => vec![&f.axis],
                FeatureKind::Pattern(PatternFeature {
                    kind: PatternKind::Rectangular { axis1, axis2, .. },
                    ..
                }) => vec![axis1, axis2],
                FeatureKind::Pattern(PatternFeature {
                    kind: PatternKind::Circular { axis, .. },
                    ..
                }) => vec![axis],
                _ => vec![],
            };
            for axis in axes {
                if let LinearRef::Edge(r) = axis {
                    if let Some((b, edge)) = base_state.resolve_edge(r) {
                        selection.push(Pick::Edge {
                            body: base_state.bodies[b].id,
                            edge,
                        });
                    }
                }
            }
        }
        match &self.cmd {
            Command::Solid {
                kind: FeatureKind::Hole(f),
                ..
            } => {
                let mut refs = Vec::new();
                if let PlaneRef::Face(face) = &f.plane {
                    refs.push(face);
                }
                if let HoleExtent::ToFace { face, .. } = &f.extent {
                    refs.push(face);
                }
                for reference in refs {
                    if let Some((b, face)) = base_state.resolve_face(reference) {
                        selection.push(Pick::Face {
                            body: base_state.bodies[b].id,
                            face,
                        });
                    }
                }
            }
            Command::Solid {
                kind: FeatureKind::Shell(f),
                ..
            } => {
                for face in &f.faces {
                    if let Some((b, face)) = base_state.resolve_face(face) {
                        selection.push(Pick::Face {
                            body: base_state.bodies[b].id,
                            face,
                        });
                    }
                }
            }
            Command::Extrude(ExtrudeCmd {
                sketch: Some(sk),
                profiles,
                ..
            })
            | Command::Revolve(RevolveCmd {
                sketch: Some(sk),
                profiles,
                ..
            }) => {
                if let Some(ss) = self.doc.state().sketches.get(sk) {
                    for r in profiles {
                        if let Some(i) = ss.profiles.iter().position(|p| {
                            match_profile(&ss.profiles, r).is_some_and(|q| std::ptr::eq(q, p))
                        }) {
                            selection.push(Pick::Profile {
                                sketch: *sk,
                                index: i,
                            });
                        }
                    }
                }
                if let Command::Revolve(RevolveCmd {
                    axis: Some(Axis::SketchLine(e)),
                    ..
                }) = &self.cmd
                {
                    selection.push(Pick::Ent {
                        sketch: *sk,
                        ent: *e,
                    });
                }
            }
            _ => {}
        }
        if let Command::Revolve(c) = &self.cmd {
            for r in &c.faces {
                if let Some((bi, id)) = base_state.resolve_face(r) {
                    selection.push(Pick::Face {
                        body: base_state.bodies[bi].id,
                        face: id,
                    });
                }
            }
            if let Some(Axis::Edge(r)) = &c.axis {
                if let Some((bi, id)) = base_state.resolve_edge(r) {
                    selection.push(Pick::Edge {
                        body: base_state.bodies[bi].id,
                        edge: id,
                    });
                }
            }
        }
        if let Command::Extrude(c) = &self.cmd {
            for r in &c.faces {
                if let Some((bi, f)) = base_state.resolve_face(r) {
                    selection.push(Pick::Face {
                        body: base_state.bodies[bi].id,
                        face: f,
                    });
                }
            }
        }
        match &self.cmd {
            Command::Blend(c) => {
                for r in &c.edges {
                    if let Some((bi, e)) = base_state.resolve_edge(r) {
                        selection.push(Pick::Edge {
                            body: base_state.bodies[bi].id,
                            edge: e,
                        });
                    }
                }
            }
            _ => {}
        }
        let mut measure_line: Option<([f64; 3], [f64; 3], f64)> = None;
        if let Command::Measure(m) = &self.cmd {
            selection.extend(m.a.iter().chain(m.b.iter()).copied());
            if let Some((d, p1, p2)) = m.result.as_ref().and_then(|r| r.distance) {
                measure_line = Some((p1, p2, d));
            }
        }
        let mut emphasis = Vec::new();
        if let Mode::Sketch(s) = &self.mode {
            if let Some(cid) = s.hover_con {
                if let Some(r) =
                    Self::sketch_ref(&self.doc, s.feature).and_then(|sk| sk.constraints.get(&cid))
                {
                    emphasis.extend(r.c.entities().into_iter().map(|e| Pick::Ent {
                        sketch: s.feature,
                        ent: e,
                    }));
                }
            }
        }

        // tool previews
        let mut preview_lines = Vec::new();
        let mut preview_points = Vec::new();
        if let (Mode::Sketch(s), Some(cur), Some(fr)) = (&mut self.mode, cursor, sketch_frame) {
            let hover_ent = match self.hover {
                Some(Pick::Ent { sketch, ent }) if sketch == s.feature => Some(ent),
                _ => None,
            };
            if let Some(sk) = Self::sketch_ref(&self.doc, s.feature) {
                let cam = &self.cam;
                s.update_midpoint_snap(
                    sk,
                    cur,
                    ui.input(|i| i.modifiers.shift && !i.modifiers.ctrl && !i.modifiers.alt),
                    |q| {
                        let p = cam.project(DVec3::from(fr.to_world(q)), size)?;
                        Some(Pos2::new(rect.min.x + p.x as f32, rect.min.y + p.y as f32))
                    },
                );
                if !matches!(
                    s.tool,
                    Tool::Select | Tool::Constrain(_) | Tool::Dimension | Tool::Project
                ) {
                    let (ls, ps) = s.preview(sk, hover_ent, cur);
                    for (a, b, dashed) in ls {
                        let color = match s.tool {
                            Tool::Trim => crate::gpu::rgba(224, 65, 50, 255),
                            Tool::Extend => crate::gpu::rgba(40, 160, 75, 255),
                            _ => scene::colors::preview(),
                        };
                        preview_lines.push((fr.to_world(a), fr.to_world(b), color, dashed));
                    }
                    for p in ps {
                        preview_points.push(fr.to_world(p));
                    }
                }
            }
        }
        if cursor.is_none() {
            if let Mode::Sketch(s) = &mut self.mode {
                s.midpoint_target = None;
            }
        }

        if let Some((p1, p2, _)) = measure_line {
            preview_lines.push((p1, p2, crate::gpu::rgba(230, 120, 20, 255), true));
            preview_points.push(p1);
            preview_points.push(p2);
        }
        let grid_extent = self.cam.distance * 1.2;
        if let Mode::Sketch(s) = &mut self.mode {
            s.grid_snap = self.grid_snap.then(|| {
                self.grid_fixed
                    .unwrap_or_else(|| scene::grid_step(grid_extent))
            });
            s.show_dims = self.view.dimensions;
            s.show_constraints = self.view.constraints;
            s.show_construction = self.view.construction;
            s.show_projected = self.view.projected;
        }
        let grid = match sketch_frame {
            Some(_) if !self.view.sketch_grid => None,
            Some(fr) => Some((fr, self.cam.distance * 1.2)),
            None => Some((
                Frame {
                    origin: [0.0; 3],
                    x: [1.0, 0.0, 0.0],
                    y: [0.0, 0.0, -1.0],
                },
                self.cam.distance * 1.2,
            )),
        };
        // Selected / hovered base-state edges drawn on top of the preview.
        let mut overlay_polylines = Vec::new();
        if split_pick {
            let hovered = match self.hover {
                Some(p @ (Pick::Edge { .. } | Pick::Face { .. })) => Some(p),
                _ => None,
            };
            for (p, color) in selection
                .iter()
                .map(|p| (*p, crate::gpu::rgba(26, 115, 250, 255)))
                .chain(hovered.map(|p| (p, crate::gpu::rgba(80, 160, 255, 255))))
            {
                let (body, edges): (BodyId, Vec<u32>) = match p {
                    Pick::Edge { body, edge } => (body, vec![edge]),
                    // A face is shown by its boundary edges.
                    Pick::Face { body, face } => match base_state.body(body) {
                        Some(b) => (
                            body,
                            (1..=b.shape.edge_count() as u32)
                                .filter(|&e| b.shape.edge_faces(e).contains(&face))
                                .collect(),
                        ),
                        None => continue,
                    },
                    _ => continue,
                };
                if let Some(m) = base_state
                    .body(body)
                    .and_then(|b| self.meshes.get(&b.shape.uid()))
                {
                    for e in edges {
                        overlay_polylines.push((m.edge_polyline(e).to_vec(), color));
                    }
                }
            }
        }
        let pick_inp = split_pick.then(|| SceneInput {
            state: base_state,
            sketches: vec![],
            profile_sketches: profile_sketches.clone(),
            hidden_bodies: &self.hidden_bodies,
            hidden_planes: &self.hidden_planes,
            filter,
            hover: None,
            selection: &[],
            emphasis: &[],
            show_planes: false,
            grid: None,
            grid_step: None,
            preview_lines: vec![],
            preview_points: vec![],
            overlay_polylines: vec![],
            constrained: &self.sketch_constrained,
            view_scale: self.cam.distance,
            show_points: true,
            show_construction: true,
            show_projected: true,
        });
        let inp = SceneInput {
            state,
            sketches,
            profile_sketches,
            hidden_bodies: &self.hidden_bodies,
            hidden_planes: &self.hidden_planes,
            filter: if split_pick {
                PickFilter::default()
            } else {
                filter
            },
            hover: self.hover,
            selection: &selection,
            emphasis: &emphasis,
            show_planes: matches!(
                self.cmd,
                Command::NewSketch
                    | Command::OffsetPlane { .. }
                    | Command::Solid {
                        pick: SolidPick::Plane,
                        ..
                    }
            ),
            grid,
            grid_step: if sketch_frame.is_some() {
                self.grid_fixed
            } else {
                None
            },
            preview_lines,
            preview_points,
            overlay_polylines,
            constrained: &self.sketch_constrained,
            view_scale: self.cam.distance,
            show_points: self.view.points,
            show_construction: self.view.construction,
            show_projected: self.view.projected,
        };
        let aspect = size.x / size.y.max(1.0);
        let finish = |sc: &mut scene::Scene, cam: &Camera| {
            sc.frame.view_proj = cam.view_proj(aspect);
            sc.frame.ortho_bias = cam.ortho_bias();
            sc.frame.eye = cam.eye().as_vec3();
            sc.frame.light = scene::light_dir(cam.right(), cam.up(), cam.forward());
            sc.frame.clear = if active.is_some() {
                [0.93, 0.94, 0.96]
            } else {
                [0.83, 0.86, 0.90]
            };
        };
        let mut sc = scene::build_cached(&inp, &self.meshes, &mut self.profile_cache);
        finish(&mut sc, &self.cam);
        let mut pick_sc = pick_inp
            .as_ref()
            .map(|pi| scene::build_cached(pi, &self.meshes, &mut self.profile_cache));
        if let Some(ps) = pick_sc.as_mut() {
            finish(ps, &self.cam);
        }

        // ---- render + pick
        let phys = [
            (rect.width() * ppp).round() as u32,
            (rect.height() * ppp).round() as u32,
        ];
        let mut new_hover = None;
        let mut hover_scene = None;
        let cpu_pick = matches!(&self.mode,Mode::Sketch(s) if s.tool!=Tool::Project);
        let defer_click = resp.clicked()
            && !cpu_pick
            && !self.file_busy()
            && !alt
            && self.pending_clicks.len() < 16;
        let mut click_readback = None;
        if let (Some(gpu), Some(rs)) = (self.gpu.as_mut(), frame.wgpu_render_state()) {
            gpu.begin_frame();
            let busy = self.navigation.is_some()
                || self.cube_drag
                || ui.input(|i| {
                    i.pointer.button_down(PointerButton::Middle)
                        || i.pointer.button_down(PointerButton::Secondary)
                        || (i.modifiers.alt && i.pointer.primary_down())
                });
            let px = pointer
                .filter(|p| {
                    !busy
                        && resp.contains_pointer()
                        && !cube_zone.contains(*p)
                        && !nav_zone.contains(*p)
                })
                .map(|p| {
                    [
                        ((p.x - rect.min.x) * ppp) as u32,
                        ((p.y - rect.min.y) * ppp) as u32,
                    ]
                });
            // Separate pick scene: upload it and pick before drawing the display scene.
            if let (Some(ps), Some(px)) = (&pick_sc, px) {
                {
                    let mut renderer = rs.renderer.write();
                    gpu.ensure_size(&rs.device, &mut renderer, phys);
                }
                gpu.upload(&rs.device, &rs.queue, phys, &ps.frame);
                if defer_click {
                    click_readback = gpu.request_click(&rs.device, &rs.queue, px);
                }
                let stamp = gpu.pick_scene_stamp();
                let result = gpu
                    .pick(&rs.device, &rs.queue, px)
                    .map(|ids| scene::resolve_pick(&ids, PICK_WIN, &ps.picks, &ps.prio));
                new_hover = resolve_hover_readback(self.hover, self.hover_scene, stamp, result);
                hover_scene = Some(stamp);
                if result.is_none() {
                    ui.ctx().request_repaint();
                }
            }
            let tex = {
                let mut renderer = rs.renderer.write();
                gpu.render(&rs.device, &rs.queue, &mut renderer, phys, &sc.frame)
            };
            ui.painter().image(
                tex,
                rect,
                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            if let (None, Some(px)) = (&pick_sc, px) {
                if defer_click {
                    click_readback = gpu.request_click(&rs.device, &rs.queue, px);
                }
                let stamp = gpu.pick_scene_stamp();
                let result = gpu
                    .pick(&rs.device, &rs.queue, px)
                    .map(|ids| scene::resolve_pick(&ids, PICK_WIN, &sc.picks, &sc.prio));
                new_hover = resolve_hover_readback(self.hover, self.hover_scene, stamp, result);
                hover_scene = Some(stamp);
                if result.is_none() {
                    ui.ctx().request_repaint();
                }
            }
        }
        if cpu_pick {
            hover_scene = None;
            new_hover = pointer.and_then(|p| {
                scene::pick_sketch_screen(&sc, p, |q| {
                    let v = self.cam.project(DVec3::from(q.map(|x| x as f64)), size)?;
                    Some(Pos2::new(rect.min.x + v.x as f32, rect.min.y + v.y as f32))
                })
            });
        }
        let used_scene = pick_sc.unwrap_or(sc);
        self.picks = used_scene.picks;
        self.prio = used_scene.prio;
        drop(inp);
        drop(pick_inp);

        // ---- overlays (sketch dimensions & glyphs) — they take precedence over picks
        let cam = self.cam.clone();
        let mut over_overlay = false;
        if let (Mode::Sketch(s), Some(fr)) = (&mut self.mode, sketch_frame) {
            let fid = s.feature;
            let to_screen = |q: [f64; 2]| -> Option<Pos2> {
                let p = cam.project(DVec3::from(fr.to_world(q)), size)?;
                Some(Pos2::new(rect.min.x + p.x as f32, rect.min.y + p.y as f32))
            };
            let before = ui
                .input(|i| i.pointer.primary_pressed())
                .then(|| self.doc.clone());
            if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
                child.set_clip_rect(rect);
                over_overlay = s.overlay(&mut child, sk, &to_screen, cursor);
            }
            if std::mem::take(&mut s.dim_moved) {
                let marker = s.restore_marker;
                if let Some(doc) = before {
                    self.push_undo(Snap { doc, marker });
                }
            }
        }
        // The ViewCube and navigation bar sit on the canvas: they swallow input there.
        if self.navigation.is_some()
            || self.cube_drag
            || pointer.is_some_and(|p| cube_zone.contains(p) || nav_zone.contains(p))
        {
            over_overlay = true;
            new_hover = None;
        }
        if self.hover != new_hover {
            ui.ctx().request_repaint();
        }
        self.hover = if over_overlay { None } else { new_hover };
        self.hover_scene = if over_overlay { None } else { hover_scene };

        // ---- clicks and drags
        // A click places once on press. Release places only for deliberate drags
        // that started a new operation, never to advance an existing line chain.
        let (primary_pressed, primary_released, shift, ctrl, latest) = ui.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.primary_released(),
                i.modifiers.shift,
                i.modifiers.ctrl,
                i.pointer.latest_pos(),
            )
        });
        let sketch_tool = if let Mode::Sketch(s) = &self.mode {
            Some((s.tool, s.is_drawing_tool()))
        } else {
            None
        };
        let sketch_handles_press = matches!(sketch_tool, Some((Tool::Select, _)) | Some((_, true)));
        let to_screen_sk = {
            let cam = self.cam.clone();
            move |q: [f64; 2]| -> Option<Pos2> {
                let fr = sketch_frame?;
                let p = cam.project(DVec3::from(fr.to_world(q)), size)?;
                Some(Pos2::new(rect.min.x + p.x as f32, rect.min.y + p.y as f32))
            }
        };
        if primary_pressed
            && !self.file_busy()
            && resp.contains_pointer()
            && pointer.is_some()
            && !alt
            && !over_overlay
        {
            self.press = None;
            if let (Mode::Sketch(s), Some(cur)) = (&mut self.mode, cursor) {
                let hover_ent = match self.hover {
                    Some(Pick::Ent { sketch, ent }) if sketch == s.feature => Some(ent),
                    _ => None,
                };
                if s.tool == Tool::Select {
                    if hover_ent.is_some() {
                        s.select_at(hover_ent, shift);
                        if let Some(sk) = Self::sketch_ref(&self.doc, s.feature) {
                            if s.drag_start(sk, hover_ent, cur) {
                                let snap = Snap {
                                    doc: self.doc.clone(),
                                    marker: s.restore_marker,
                                };
                                self.push_undo(snap);
                            }
                        }
                    } else {
                        self.box_start = pointer;
                    }
                } else if s.is_drawing_tool() {
                    let p = pointer.unwrap();
                    self.press = Some(DrawingPress {
                        at: p,
                        time: ui.input(|i| i.time),
                        feature: s.feature,
                        tool: s.tool,
                        started_operation: !s.in_progress(),
                    });
                    let pick = self.hover;
                    self.on_click(pick, Some(cur), p, shift);
                }
            }
        }
        if let Mode::Sketch(s) = &mut self.mode {
            if s.is_dragging() {
                if lmb {
                    if let Some(cur) = cursor {
                        let fid = s.feature;
                        if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                            s.drag_to(sk, cur);
                        }
                    }
                } else if s.drag_end() {
                    self.analysis_dirty = true;
                    self.dirty = true;
                }
            }
        }
        if primary_released {
            let press = self.press.take();
            let p1 = latest.or(pointer);
            let release_allowed = !self.file_busy() && !alt && !over_overlay;
            if let (Mode::Sketch(s), Some(p1)) = (&mut self.mode, p1) {
                if let Some(b0) = self.box_start.take() {
                    let r = egui::Rect::from_two_pos(b0, p1);
                    if r.width() > 4.0 || r.height() > 4.0 {
                        // left→right: window (fully inside); right→left: crossing
                        let crossing = p1.x < b0.x;
                        if let Some(sk) = Self::sketch_ref(&self.doc, s.feature) {
                            let ents =
                                crate::sketch_mode::box_select(sk, r, crossing, &to_screen_sk);
                            s.set_selection(ents, shift, ctrl);
                        }
                    } else if !shift {
                        s.selection.clear();
                    }
                } else if let Some(press) = press {
                    if release_allowed
                        && rect.contains(p1)
                        && press.finishes_drag(
                            s,
                            p1,
                            ui.input(|i| i.time),
                            resp.drag_stopped_by(PointerButton::Primary),
                        )
                    {
                        let cur = cursor_local(p1, &self.cam);
                        let pick = self.hover;
                        self.on_click(pick, cur, p1, shift);
                    }
                }
            }
        }
        // Rubber band for window / crossing selection
        if let (Some(b0), Some(p)) = (self.box_start, pointer) {
            let r = egui::Rect::from_two_pos(b0, p);
            let painter = ui.painter_at(rect);
            if p.x < b0.x {
                painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(60, 200, 90, 40));
                let st = egui::Stroke::new(1.2_f32, Color32::from_rgb(40, 160, 70));
                for (a, b) in [
                    (r.left_top(), r.right_top()),
                    (r.right_top(), r.right_bottom()),
                    (r.right_bottom(), r.left_bottom()),
                    (r.left_bottom(), r.left_top()),
                ] {
                    painter.add(egui::Shape::dashed_line(&[a, b], st, 6.0, 4.0));
                }
            } else {
                painter.rect(
                    r,
                    0.0,
                    Color32::from_rgba_unmultiplied(60, 120, 255, 40),
                    egui::Stroke::new(1.2_f32, Color32::from_rgb(40, 100, 240)),
                    egui::StrokeKind::Inside,
                );
            }
        }
        // Double-click a sketch curve: select the connected chain (Shift adds).
        if resp.double_clicked() && !over_overlay {
            if let Mode::Sketch(s) = &mut self.mode {
                if let (Tool::Select, Some(Pick::Ent { sketch, ent })) = (s.tool, self.hover) {
                    if sketch == s.feature {
                        if let Some(sk) = Self::sketch_ref(&self.doc, sketch) {
                            let chain = crate::sketch_mode::chain_from(sk, ent);
                            s.set_selection(chain, shift, false);
                        }
                    }
                }
            }
        }
        if resp.clicked() && !self.file_busy() && !over_overlay && !alt && !sketch_handles_press {
            if let Some(p) = resp.interact_pointer_pos() {
                if cpu_pick {
                    self.on_click(self.hover, cursor_local(p, &self.cam), p, shift);
                } else if let Some(readback) = click_readback {
                    self.pending_clicks.push_back(QueuedClick {
                        readback,
                        context: self.click_context(),
                        picks: self.picks.clone(),
                        prio: self.prio.clone(),
                        cursor: cursor_local(p, &self.cam),
                        at: p,
                        shift,
                    });
                    ui.ctx().request_repaint();
                }
            }
        }
        if resp.secondary_clicked() {
            if let Mode::Sketch(s) = &mut self.mode {
                s.cancel();
            }
        }

        // ---- heads-up dimension input while drawing
        self.hud_ui(ui.ctx(), pointer);

        // ---- dimension edit box
        self.dim_edit_box(ui.ctx());

        // ---- measurement label
        if let Some((p1, p2, d)) = measure_line {
            let mid = DVec3::from(p1).lerp(DVec3::from(p2), 0.5);
            if let Some(sp) = self.cam.project(mid, size) {
                let pos = Pos2::new(rect.min.x + sp.x as f32, rect.min.y + sp.y as f32);
                let prec = if let Command::Measure(m) = &self.cmd {
                    m.precision
                } else {
                    2
                };
                let painter = ui.painter_at(rect);
                let g = painter.layout_no_wrap(
                    format!("{d:.prec$} mm"),
                    egui::FontId::proportional(14.0),
                    Color32::WHITE,
                );
                let r = egui::Rect::from_center_size(
                    pos + Vec2::new(0.0, -14.0),
                    g.size() + Vec2::new(10.0, 6.0),
                );
                painter.rect_filled(r, 4.0, Color32::from_rgb(200, 100, 15));
                painter.galley(r.min + Vec2::new(5.0, 3.0), g, Color32::WHITE);
            }
        }

        // ---- view controls
        self.view_cube(ui, rect);
        self.nav_bar(ui, rect);

        if let Some(profiler) = &mut self.profiler {
            profiler.record(
                profile_start,
                self.doc.state().bodies.len(),
                self.picks.len(),
            );
            ui.ctx().request_repaint();
        }
        // ---- toasts and status
        let painter = ui.painter_at(rect);
        self.toasts
            .retain(|t| t.at.elapsed() < Duration::from_secs(5));
        let mut y = rect.max.y - 12.0;
        for t in self.toasts.iter().rev() {
            let col = if t.error {
                Color32::from_rgb(170, 40, 30)
            } else {
                Color32::from_rgb(30, 90, 50)
            };
            let galley = painter.layout(
                t.text.clone(),
                egui::FontId::proportional(14.0),
                Color32::WHITE,
                (rect.width() - 40.0).max(100.0),
            );
            let r = egui::Rect::from_min_size(
                Pos2::new(rect.min.x + 12.0, y - galley.size().y - 8.0),
                galley.size() + Vec2::new(16.0, 8.0),
            );
            painter.rect_filled(r, 4.0, col.gamma_multiply(0.92));
            painter.galley(r.min + Vec2::new(8.0, 4.0), galley, Color32::WHITE);
            y = r.min.y - 6.0;
        }
        if !self.toasts.is_empty() {
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }
        if let Some(Pick::Face { .. } | Pick::Edge { .. }) = self.hover {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }

    fn hud_ui(&mut self, ctx: &egui::Context, pointer: Option<Pos2>) {
        let Mode::Sketch(s) = &mut self.mode else {
            return;
        };
        if s.hud.is_none() {
            return;
        }
        if let Some(p) = pointer {
            self.hud_pos = p;
        }
        let vals = self.doc.param_quantities().unwrap_or_default();
        let eval = move |t: &str, u: Unit| expr::eval(t, u, &vals).map_err(|e| e.to_string());
        let hud = s.hud.as_mut().unwrap();
        egui::Area::new(egui::Id::new("sketch-hud"))
            .fixed_pos(self.hud_pos + Vec2::new(18.0, 14.0))
            .order(egui::Order::Foreground)
            .interactable(true)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(4.0)
                    .show(ui, |ui| {
                        for i in 0..hud.fields.len() {
                            ui.horizontal(|ui| {
                                let f = &hud.fields[i];
                                let label = RichText::new(f.kind.label()).strong().color(
                                    if f.locked.is_some() {
                                        Color32::from_rgb(255, 190, 80)
                                    } else {
                                        Color32::from_gray(200)
                                    },
                                );
                                ui.label(label);
                                let unit = if f.kind.unit() == Unit::Angle {
                                    "°"
                                } else {
                                    "mm"
                                };
                                let id = egui::Id::new(("hud-field", i));
                                let out = egui::TextEdit::singleline(&mut hud.fields[i].text)
                                    .id(id)
                                    .desired_width(72.0)
                                    .show(ui);
                                ui.weak(unit);
                                if out.response.changed() {
                                    hud.active = i;
                                    hud.edit(i, &eval);
                                }
                                let is_active = hud.active == i;
                                if is_active && (hud.refocus || !out.response.has_focus()) {
                                    out.response.request_focus();
                                }
                                // While unlocked, keep the live value selected so typing replaces it.
                                if is_active && hud.fields[i].locked.is_none() {
                                    let mut st = out.state.clone();
                                    let n = hud.fields[i].text.chars().count();
                                    st.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                        egui::text::CCursor::new(0),
                                        egui::text::CCursor::new(n),
                                    )));
                                    st.store(ui.ctx(), id);
                                }
                            });
                        }
                        hud.refocus = false;
                        ui.weak("Tab: next · Enter: place");
                    });
            });
    }

    fn dim_edit_box(&mut self, ctx: &egui::Context) {
        let Mode::Sketch(s) = &mut self.mode else {
            return;
        };
        let Some(ed) = &mut s.dim_edit else { return };
        let mut commit = false;
        let mut cancel = false;
        egui::Area::new(egui::Id::new("dim-edit"))
            .fixed_pos(ed.at + Vec2::new(8.0, -10.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    let out = egui::TextEdit::singleline(&mut ed.text)
                        .desired_width(110.0)
                        .hint_text("value or expression")
                        .show(ui);
                    if ed.focus {
                        out.response.request_focus();
                        let mut st = out.state;
                        let n = ed.text.chars().count();
                        st.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                            egui::text::CCursor::new(0),
                            egui::text::CCursor::new(n),
                        )));
                        st.store(ui.ctx(), out.response.id);
                        ed.focus = false;
                    }
                    if ui.input(|i| i.key_pressed(Key::Enter)) {
                        commit = true;
                    }
                    if ui.input(|i| i.key_pressed(Key::Escape)) {
                        cancel = true;
                    }
                });
            });
        let fid = s.feature;
        if cancel {
            s.cancel();
            ctx.memory_mut(|m| m.request_focus(egui::Id::NULL));
        } else if commit {
            let before = self.doc.clone();
            let eval = self.eval_fn();
            let Mode::Sketch(s) = &mut self.mode else {
                return;
            };
            if let Some(sk) = Self::sketch_mut(&mut self.doc, fid) {
                let o = s.commit_dim(sk, &eval);
                self.sketch_outcome(Some(before), o);
            }
        }
    }

    /// Sharp faces with invisible surface patches for edge and corner clicks.
    fn view_cube(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let c = Pos2::new(rect.max.x - 70.0, rect.min.y + 72.0);
        let regions = crate::view_cube::regions(self.cam.rot, c, 28.0);
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let hovered = pointer.and_then(|p| regions.iter().position(|r| r.contains(p)));
        let painter = ui.painter_at(rect);
        let faces = crate::view_cube::faces(self.cam.rot, c, 28.0);
        for face in &faces {
            painter.add(egui::Shape::convex_polygon(
                face.polygon.clone(),
                Color32::from_rgb(232, 234, 238),
                egui::Stroke::new(1.0_f32, Color32::from_gray(120)),
            ));
        }
        if let Some(i) = hovered {
            for region in regions
                .iter()
                .filter(|r| r.direction == regions[i].direction)
            {
                painter.add(egui::Shape::convex_polygon(
                    region.polygon.clone(),
                    Color32::from_rgb(170, 205, 255),
                    egui::Stroke::NONE,
                ));
            }
        }
        for face in &faces {
            if (self.cam.rot.inverse() * face.direction).z > 0.35 {
                painter.text(
                    face.center(),
                    egui::Align2::CENTER_CENTER,
                    face.label.to_uppercase(),
                    egui::FontId::proportional(9.5),
                    Color32::from_gray(60),
                );
            }
        }
        // Stable ownership across changing projected faces and leaving the cube.
        let bounds = egui::Rect::from_center_size(c, Vec2::splat(86.0));
        let response = ui.interact(
            bounds,
            ui.id().with("cube_gesture"),
            Sense::click_and_drag(),
        );
        if response.drag_started_by(PointerButton::Primary) {
            let origin = ui.input(|i| i.pointer.press_origin());
            self.cube_drag = origin.is_some_and(|p| regions.iter().any(|r| r.contains(p)));
        }
        if self.cube_drag && response.dragged_by(PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.cam.orbit(delta.x as f64, delta.y as f64);
            ui.ctx().request_repaint();
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            self.cube_drag = false;
        }
        if let Some(i) = hovered {
            response
                .clone()
                .on_hover_text(format!("{} · drag to orbit", regions[i].label));
            if response.clicked() {
                self.cam
                    .set_view(crate::view_cube::rotation(regions[i].direction));
            }
        }
        let home =
            egui::Rect::from_center_size(Pos2::new(c.x - 48.0, c.y - 48.0), Vec2::splat(18.0));
        let response = ui
            .interact(home, ui.id().with("cube_home"), Sense::click())
            .on_hover_text("Home view");
        painter.text(
            home.center(),
            egui::Align2::CENTER_CENTER,
            "⌂",
            egui::FontId::proportional(16.0),
            if response.hovered() {
                Color32::from_rgb(40, 110, 240)
            } else {
                Color32::from_gray(90)
            },
        );
        if response.clicked() {
            self.reset_home_view();
        }
    }

    fn nav_bar(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let area = egui::Rect::from_center_size(
            Pos2::new(rect.center().x, rect.max.y - 22.0),
            Vec2::new(450.0, 28.0),
        );
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(area)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        egui::Frame::popup(child.style())
            .inner_margin(egui::Margin::symmetric(6, 2))
            .show(&mut child, |ui| {
                ui.horizontal(|ui| {
                    if let Mode::Sketch(s) = &self.mode {
                        let fr = s.frame;
                        if ui.small_button("👁").on_hover_text("Look At").clicked() {
                            self.cam.look_at_plane(
                                DVec3::from(fr.origin),
                                DVec3::from(fr.x),
                                DVec3::from(fr.y),
                            );
                        }
                    }
                    if ui.small_button("⛶").on_hover_text("Fit (F6)").clicked() {
                        self.fit_all();
                    }
                    if ui.small_button("⌂").on_hover_text("Home view").clicked() {
                        self.cam.set_view(named_view("Home"));
                    }
                    ui.separator();
                    for (mode, label, hint) in [
                        (
                            NavigationMode::Orbit,
                            "Orbit",
                            "Drag to orbit around the view centre",
                        ),
                        (NavigationMode::Pan, "Pan", "Drag to move the view"),
                        (
                            NavigationMode::Zoom,
                            "Zoom",
                            "Drag up to zoom in, down to zoom out",
                        ),
                    ] {
                        if ui
                            .selectable_label(self.navigation == Some(mode), label)
                            .on_hover_text(format!("{hint} · Esc to exit"))
                            .clicked()
                        {
                            self.navigation = if self.navigation == Some(mode) {
                                None
                            } else {
                                Some(mode)
                            };
                            self.press = None;
                            self.box_start = None;
                        }
                    }
                    ui.separator();
                    ui.menu_button("🖵 Display", |ui| {
                        ui.label("Camera");
                        if ui.radio(self.cam.ortho, "Orthographic").clicked() {
                            self.cam.ortho = true;
                        }
                        if ui.radio(!self.cam.ortho, "Perspective").clicked() {
                            self.cam.ortho = false;
                        }
                    });
                    ui.menu_button("⊞ Grid and Snaps", |ui| self.grid_settings(ui));
                });
            });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let hint = if let Some(mode) = self.navigation {
                format!(
                    "{}: drag on the canvas · Esc to exit",
                    match mode {
                        NavigationMode::Orbit => "Orbit",
                        NavigationMode::Pan => "Pan",
                        NavigationMode::Zoom => "Zoom",
                    }
                )
            } else {
                match (&self.mode, &self.cmd) {
                    (Mode::Sketch(s), _) => s.tool.hint().to_string(),
                    (_, Command::NewSketch) => "Pick an origin plane, construction plane or a planar face to sketch on".into(),
                    (_, Command::Solid { pick: SolidPick::Plane, .. }) => {
                        "Select an origin plane, construction plane, or planar face".into()
                    }
                    (_, Command::Solid { pick: SolidPick::Axis1 | SolidPick::Axis2, .. }) => {
                        "Select a straight body edge for the axis or direction".into()
                    }
                    (_, Command::Solid { kind: FeatureKind::Hole(_), .. }) => "Select a planar face to place a hole; Shift-click adds centers; Enter to finish".into(),
                    (_, Command::Solid { .. }) => "Select bodies or shell removal faces; set values; Enter to finish".into(),
                    (_, Command::OffsetPlane { .. }) => "Pick a plane or planar face; enter a signed offset distance".into(),
                    (_, Command::Extrude(_)) => "Click profiles to extrude; set distance; Enter to finish".into(),
                    (_, Command::Revolve(_)) => "Click profiles, then a sketch line (or pick X/Y/Z) as axis".into(),
                    (_, Command::Blend(_)) => "Click edges or faces; set size; Enter to finish".into(),
                    (_, Command::Measure(_)) => "Click two faces, edges, vertices or sketch points; a third click starts over".into(),
                    _ => "Middle-drag pan · Shift+middle / right-drag orbit · wheel zoom · S sketch · E extrude · F fillet".into(),
                }
            };
            ui.label(hint);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(&self.gpu_name);
                ui.weak(format!("recompute {:.1} ms", self.last_recompute_ms));
                if let Mode::Sketch(_) = &self.mode {
                    let txt = if self.analysis_dirty {
                        RichText::new("Analyzing constraints…").color(Color32::from_rgb(120, 170, 255))
                    } else if self.sketch_dof == 0 {
                        RichText::new("Fully constrained").color(Color32::from_rgb(90, 200, 110))
                    } else {
                        RichText::new(format!("{} DOF remaining", self.sketch_dof)).color(Color32::from_rgb(120, 170, 255))
                    };
                    ui.label(txt);
                }
                if let Some(h) = self.hover {
                    ui.weak(match h {
                        Pick::Face { body, face } => format!("Body{body} face {face}"),
                        Pick::Edge { body, edge } => format!("Body{body} edge {edge}"),
                        Pick::Vertex { body, vertex } => format!("Body{body} vertex {vertex}"),
                        Pick::Ent { ent, .. } => format!("sketch entity #{}", ent.0),
                        Pick::Profile { index, .. } => format!("profile {index}"),
                        Pick::ConstructionPlane(id) => format!("Construction plane #{id}"),
                        Pick::Plane(k) => ["XY plane", "XZ plane", "YZ plane"][k as usize].into(),
                    });
                }
                if !self.doc.errors.is_empty() {
                    ui.colored_label(Color32::from_rgb(240, 90, 80), format!("{} feature error(s)", self.doc.errors.len()));
                }
            });
        });
    }
}

fn open_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new()
        .add_filter(
            "All supported",
            &["rfz", "step", "stp", "iges", "igs", "f3d"],
        )
        .add_filter("refusion design", &["rfz"])
        .add_filter("STEP", &["step", "stp"])
        .add_filter("IGES", &["iges", "igs"])
        .add_filter("F3D archive", &["f3d"])
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn sketch_fmt(v: f64) -> String {
    crate::sketch_mode::fmt_len(v)
}

fn op_combo(ui: &mut egui::Ui, op: &mut Operation) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label("Operation")
        .selected_text(op.label())
        .show_ui(ui, |ui| {
            for o in Operation::ALL {
                changed |= ui.selectable_value(op, o, o.label()).changed();
            }
        });
    changed
}

impl eframe::App for RefusionApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let title = format!(
            "refusion — {}{}",
            self.path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or("untitled".into()),
            if self.dirty { " •" } else { "" }
        );
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        self.poll_file_load();
        self.poll_compute();
        if !self.replay_pending {
            if let Some((p, insert)) = self.deferred_file.take() {
                self.start_file_load(p, insert);
            }
        }
        let busy = self.file_busy();
        if !busy {
            self.shortcuts(ctx);
        }
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_enabled_ui(!busy, |ui| self.toolbar(ui));
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.status_bar(ui));
        egui::TopBottomPanel::bottom("timeline")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_enabled_ui(!busy, |ui| self.timeline(ui));
            });
        egui::SidePanel::left("browser")
            .default_width(200.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.add_enabled_ui(!busy, |ui| self.browser(ui));
            });
        if !busy {
            self.command_window(ctx);
        }
        self.sketch_palette(ctx);
        if !busy {
            self.params_window(ctx);
            self.stl_export_window(ctx);
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.viewport(ui, frame));
        self.poll_clicks(ctx, frame);
        self.file_progress(ctx);
        if self.replay_pending {
            egui::Window::new("Updating model")
                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -80.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Recomputing timeline… Camera navigation remains available.");
                    });
                });
        }
        let _ = EntityId(0);
        let _ = Geom::Point;
    }
}

/// Selection fields own the canvas pick mode; clearing geometry rebuilds the preview.
fn selection_field(
    ui: &mut egui::Ui,
    label: &str,
    count: usize,
    mode: &mut bool,
    value: bool,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let text = if count == 0 {
            "Select".into()
        } else {
            format!("{count} selected")
        };
        if ui.selectable_label(*mode == value, text).clicked() {
            *mode = value;
        }
        ui.add_enabled(count > 0, egui::Button::new("×"))
            .on_hover_text("Clear selection")
            .clicked()
    })
    .inner
}

fn text_field(ui: &mut egui::Ui, label: &str, value: &mut String) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(value).desired_width(120.0))
            .changed()
    })
    .inner
}
fn axis_choices(ui: &mut egui::Ui, label: &str, axis: &mut LinearRef) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label(label)
        .selected_text(match axis {
            LinearRef::X => "X axis",
            LinearRef::Y => "Y axis",
            LinearRef::Z => "Z axis",
            LinearRef::Edge(_) => "Selected edge",
        })
        .show_ui(ui, |ui| {
            for (v, n) in [
                (LinearRef::X, "X"),
                (LinearRef::Y, "Y"),
                (LinearRef::Z, "Z"),
            ] {
                changed |= ui.selectable_value(axis, v, n).changed();
            }
        });
    changed
}
fn plane_choices(ui: &mut egui::Ui, plane: &mut PlaneRef) -> bool {
    let mut changed = false;
    egui::ComboBox::from_label("Plane")
        .selected_text(match plane {
            PlaneRef::XY => "XY plane",
            PlaneRef::XZ => "XZ plane",
            PlaneRef::YZ => "YZ plane",
            PlaneRef::Face(_) => "Selected planar face",
            PlaneRef::Construction(_) => "Selected construction plane",
            PlaneRef::Fixed { .. } => "Fixed plane",
        })
        .show_ui(ui, |ui| {
            for (v, n) in [
                (PlaneRef::XY, "XY"),
                (PlaneRef::XZ, "XZ"),
                (PlaneRef::YZ, "YZ"),
            ] {
                changed |= ui.selectable_value(plane, v, n).changed();
            }
        });
    changed
}
fn body_choices(ui: &mut egui::Ui, bodies: &[Body], selected: &mut Vec<BodyId>) -> bool {
    let mut changed = false;
    ui.label("Bodies");
    for b in bodies {
        let mut on = selected.contains(&b.id);
        if ui.checkbox(&mut on, &b.name).changed() {
            selected.retain(|id| *id != b.id);
            if on {
                selected.push(b.id);
            }
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    fn design() -> (Document, FeatureId, FeatureId) {
        let mut doc = Document::new();
        let mut sketch = Sketch::new();
        sketch.add_rectangle([0.0, 0.0], [10.0, 10.0]);
        let first = doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch,
            projections: vec![],
        }));
        let profile = profile_ref(&doc.state().sketches[&first].profiles[0]);
        doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch: first,
            profiles: vec![profile],
            faces: vec![],
            extent: Extent::OneSide { d: "10 mm".into() },
            op: Operation::NewBody,
        }));
        let second = doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XZ,
            sketch: Sketch::new(),
            projections: vec![],
        }));
        (doc, first, second)
    }
    fn wait(app: &mut RefusionApp) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.replay_pending {
            assert!(Instant::now() < deadline, "replay timeout");
            app.poll_compute();
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn repeated_sketch_entry_during_replay_preserves_rollback_and_restore_marker() {
        let (doc, first, _) = design();
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc = doc;
        app.enter_sketch(first, 3);
        assert_eq!(app.doc.marker, 1);
        app.mode_to_model();
        app.enter_sketch(first, 1);
        assert!(app.replay_pending);
        assert!(matches!(app.mode, Mode::Model));
        wait(&mut app);
        assert_eq!(app.doc.marker, 1);
        assert!(app.doc.state().bodies.is_empty());
        assert!(matches!(&app.mode,Mode::Sketch(s)if s.feature==first&&s.restore_marker==3));
        app.finish_sketch();
        wait(&mut app);
        assert_eq!(app.doc.marker, 3);
        assert_eq!(app.doc.state().bodies.len(), 1);
    }
    #[test]
    fn switching_to_later_sketch_waits_for_replay() {
        let (doc, first, second) = design();
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc = doc;
        app.enter_sketch(first, 3);
        app.mode_to_model();
        app.enter_sketch(second, 1);
        assert!(matches!(app.mode, Mode::Model));
        wait(&mut app);
        assert_eq!(app.doc.marker, 3);
        assert!(matches!(&app.mode,Mode::Sketch(s)if s.feature==second&&s.restore_marker==3));
        assert_eq!(app.doc.state().bodies.len(), 1);
    }
    #[test]
    fn later_sketch_edit_from_rolled_back_timeline_uses_worker() {
        let (doc, _, second) = design();
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc = doc;
        app.doc.set_marker(0);
        app.enter_sketch(second, 0);
        assert!(app.replay_pending);
        assert!(matches!(app.mode, Mode::Model));
        wait(&mut app);
        assert!(matches!(&app.mode,Mode::Sketch(s)if s.feature==second&&s.restore_marker==3));
        assert_eq!(app.doc.state().bodies.len(), 1);
    }
}

#[cfg(test)]
mod navigation_tests {
    use super::*;
    fn assert_home(app: &mut RefusionApp) {
        let home = Camera::default();
        assert_eq!(app.cam.target, home.target);
        assert_eq!(app.cam.distance, home.distance);
        assert_eq!(app.cam.rot, home.rot);
        assert_eq!(app.cam.fov_y, home.fov_y);
        assert_eq!(app.cam.ortho, home.ortho);
        assert_eq!(app.cam.view_proj(1.5), home.view_proj(1.5));
        assert!(!app.cam.tick(1.0));
        assert_eq!(app.cam.view_proj(1.5), home.view_proj(1.5));
        assert!(app.navigation.is_none() && !app.cube_drag && app.press.is_none());
        assert!(app.hover.is_none() && app.hover_scene.is_none());
    }
    fn move_away_from_home(app: &mut RefusionApp) {
        app.cam.target = DVec3::new(150., -300., 70.);
        app.cam.distance = 17.;
        app.cam.ortho = false;
        app.cam.fov_y = 0.9;
        app.cam.set_view(named_view("Top"));
        app.cam.tick(0.1);
    }
    #[test]
    fn cube_home_button_restores_every_camera_setting_and_cancels_old_animation() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        move_away_from_home(&mut app);
        let home = Pos2::new(682., 24.);
        frame(&mut app, &ctx, vec![]);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(home), button(home, true)],
        );
        frame(&mut app, &ctx, vec![button(home, false)]);
        assert_home(&mut app);
    }
    #[test]
    fn new_document_restores_home_after_sketch_camera_and_navigation_changes() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        let id = app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch: Sketch::new(),
            projections: vec![],
        }));
        app.enter_sketch(id, 1);
        app.path = Some(PathBuf::from("previous.rfz"));
        move_away_from_home(&mut app);
        app.navigation = Some(NavigationMode::Zoom);
        app.cube_drag = true;
        app.hover = Some(Pick::Plane(0));
        app.hover_scene = Some(123);
        app.new_document();
        assert!(app.doc.features.is_empty() && app.path.is_none());
        assert!(matches!(app.mode, Mode::Model));
        assert_home(&mut app);
    }
    #[test]
    fn fit_finds_a_sketch_only_design_far_from_the_origin() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        let mut sketch = Sketch::new();
        let centre = sketch.add_point([0.0, 0.0]);
        sketch.add_circle(centre, 80.0);
        app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::Fixed {
                origin: [10000.0, 2000.0, -3000.0],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 0.0, -1.0],
            },
            sketch,
            projections: vec![],
        }));
        app.fit_all();
        app.cam.tick(1.0);
        assert!(app
            .cam
            .target
            .abs_diff_eq(DVec3::new(10000.0, 2000.0, -3000.0), 1e-7));
        assert!(app.cam.distance > 80.0);
    }
    #[test]
    fn cube_face_click_still_snaps_after_drag_support() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        let region = crate::view_cube::regions(app.cam.rot, Pos2::new(730.0, 72.0), 28.0)
            .into_iter()
            .find(|r| r.direction == DVec3::Y)
            .unwrap();
        let p = region.center();
        frame(&mut app, &ctx, vec![]);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(p), button(p, true)],
        );
        frame(&mut app, &ctx, vec![button(p, false)]);
        app.cam.tick(1.0);
        assert!(app.cam.forward().abs_diff_eq(-DVec3::Y, 1e-9));
        assert!(app.cam.up().abs_diff_eq(-DVec3::Z, 1e-9));
    }
    fn frame(app: &mut RefusionApp, ctx: &egui::Context, events: Vec<egui::Event>) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new(800.0, 600.0),
            )),
            events,
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            app.shortcuts(ctx);
            egui::CentralPanel::default().show(ctx, |ui| {
                app.view_cube(
                    ui,
                    egui::Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0)),
                );
            });
        });
    }
    fn button(p: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: p,
            button: PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }
    #[test]
    fn cube_drag_survives_leaving_cube_without_snapping_on_release() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        let p = Pos2::new(730.0, 72.0);
        frame(&mut app, &ctx, vec![]);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(p), button(p, true)],
        );
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(p + Vec2::new(-20.0, 10.0))],
        );
        assert!(app.cube_drag);
        let before = app.cam.rot;
        let outside = p + Vec2::new(-150.0, 90.0);
        frame(&mut app, &ctx, vec![egui::Event::PointerMoved(outside)]);
        assert!(!app.cam.rot.abs_diff_eq(before, 1e-6));
        frame(&mut app, &ctx, vec![button(outside, false)]);
        assert!(!app.cube_drag);
        let released = app.cam.rot;
        app.cam.tick(1.0);
        assert!(app.cam.rot.abs_diff_eq(released, 1e-9));
    }
    #[test]
    fn viewport_drag_released_over_cube_does_not_snap() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        let before = app.cam.rot;
        let outside = Pos2::new(500.0, 250.0);
        let cube = Pos2::new(730.0, 72.0);
        frame(&mut app, &ctx, vec![]);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(outside), button(outside, true)],
        );
        frame(&mut app, &ctx, vec![egui::Event::PointerMoved(cube)]);
        frame(&mut app, &ctx, vec![button(cube, false)]);
        app.cam.tick(1.0);
        assert!(app.cam.rot.abs_diff_eq(before, 1e-9));
    }
    #[test]
    fn escape_exits_navigation_without_cancelling_sketch_tool() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        app.navigation = Some(NavigationMode::Pan);
        let sketch = Sketch::new();
        let id = app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch: sketch.clone(),
            projections: vec![],
        }));
        app.enter_sketch(id, 1);
        if let Mode::Sketch(s) = &mut app.mode {
            s.set_tool(Tool::Line);
        }
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(app.navigation.is_none());
        assert!(matches!(&app.mode, Mode::Sketch(s) if s.tool == Tool::Line));
        assert!(sketch.entities.is_empty());
    }
}
#[cfg(test)]
mod deferred_selection_tests {
    #[test]
    fn extrude_panel_separates_direction_extent_and_target_selection() {
        let ctx = egui::Context::default();
        let mut app = RefusionApp::empty(ctx.clone(), None, String::new());
        app.start_extrude(None);
        for (kind, expected) in [
            (ExtentKind::OneSide, "Distance"),
            (ExtentKind::ToFace, "To Object"),
            (ExtentKind::ThroughAll, "All"),
        ] {
            let Command::Extrude(c) = &mut app.cmd else {
                panic!()
            };
            c.kind = kind;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(1500., 920.),
                    )),
                    ..Default::default()
                },
                |ctx| app.command_window(ctx),
            );
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(1500., 920.),
                    )),
                    ..Default::default()
                },
                |ctx| app.command_window(ctx),
            );
            let texts: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| {
                    if let egui::Shape::Text(t) = &shape.shape {
                        Some(t.galley.text().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            for label in ["Profiles", "Direction", "Extent Type", expected] {
                assert!(
                    texts.iter().any(|text| text == label),
                    "missing {label}: {texts:?}"
                );
            }
            if kind == ExtentKind::ToFace {
                assert!(texts.iter().any(|text| text == "Object"));
            }
            assert!(app.preview.is_none());
        }
    }

    #[test]
    fn selection_field_selects_and_clears_with_real_pointer_events() {
        let ctx = egui::Context::default();
        let mut mode = false;
        let mut count = 2;
        let mut draw = |events: Vec<egui::Event>| {
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, Vec2::new(600., 400.))),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        if selection_field(ui, "Object", count, &mut mode, true) {
                            count = 0;
                        }
                    });
                },
            )
        };
        let output = draw(vec![]);
        let text_position = |text: &str| {
            output
                .shapes
                .iter()
                .find_map(|shape| {
                    if let egui::Shape::Text(t) = &shape.shape {
                        if t.galley.text() == text {
                            return Some(t.pos + t.galley.size() / 2.);
                        }
                    }
                    None
                })
                .unwrap()
        };
        for at in [text_position("2 selected"), text_position("×")] {
            draw(vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ]);
            draw(vec![egui::Event::PointerButton {
                pos: at,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }]);
        }
        drop(draw);
        assert!(mode);
        assert_eq!(count, 0);
    }

    #[test]
    fn revolve_fields_route_profiles_and_axis_independently() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        let mut sketch = Sketch::new();
        let lines = sketch.add_rectangle([0., 0.], [10., 10.]);
        let id = app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch,
            projections: vec![],
        }));
        app.start_revolve(None);
        let profile = Pick::Profile {
            sketch: id,
            index: 0,
        };
        app.on_click(Some(profile), None, Pos2::ZERO, false);
        let line = Pick::Ent {
            sketch: id,
            ent: lines[0],
        };
        app.on_click(Some(line), None, Pos2::ZERO, false);
        let Command::Revolve(c) = &mut app.cmd else {
            panic!()
        };
        assert_eq!(c.profiles.len(), 1);
        assert!(c.axis.is_none());
        let old = app.click_context();
        let Command::Revolve(c) = &mut app.cmd else {
            panic!()
        };
        c.pick_axis = true;
        assert!(old != app.click_context());
        app.on_click(Some(profile), None, Pos2::ZERO, false);
        let Command::Revolve(c) = &app.cmd else {
            panic!()
        };
        assert_eq!(c.profiles.len(), 1);
        app.on_click(Some(line), None, Pos2::ZERO, false);
        let Command::Revolve(c) = &app.cmd else {
            panic!()
        };
        assert_eq!(c.axis, Some(Axis::SketchLine(lines[0])));
        assert!(!c.pick_axis);
        assert_eq!(c.profiles.len(), 1);
        app.cancel_command();
        assert!(matches!(app.cmd, Command::None));
        assert_eq!(app.doc.features.len(), 1);
        // Drain native preview work before this isolated test process exits.
        app.compute.submit(crate::compute::Request {
            ticket: u64::MAX,
            replay: true,
            doc: Document::new(),
            cached: HashMap::new(),
            edit: Box::new(|_| None),
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(Ok(result)) = app.compute.poll() {
                if result.ticket == u64::MAX {
                    break;
                }
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    use super::*;
    #[test]
    fn hover_survives_pending_frames_and_clears_on_a_real_background_hit() {
        let edge = Some(Pick::Edge { body: 1, edge: 2 });
        let other = Some(Pick::Edge { body: 1, edge: 3 });
        let mut hover = resolve_hover_readback(None, None, 42, Some(edge));
        // Mouse motion starts new readbacks. Highlight repaint frames can arrive
        // before those complete and must not alternate the edge with background.
        for _ in 0..20 {
            hover = resolve_hover_readback(hover, Some(42), 42, None);
            assert_eq!(hover, edge);
        }
        hover = resolve_hover_readback(hover, Some(42), 42, Some(other));
        assert_eq!(hover, other);
        hover = resolve_hover_readback(hover, Some(42), 42, Some(None));
        assert_eq!(hover, None);
        assert_eq!(resolve_hover_readback(hover, Some(42), 42, None), None);
    }
    #[test]
    fn pending_hover_does_not_survive_camera_geometry_or_viewport_exit() {
        let edge = Some(Pick::Edge { body: 1, edge: 2 });
        assert_eq!(resolve_hover_readback(edge, Some(42), 43, None), None);
        // Leaving the pickable viewport clears the stored context, including
        // navigation and overlay ownership. Re-entry cannot revive an old hit.
        assert_eq!(resolve_hover_readback(edge, None, 42, None), None);
        assert_eq!(resolve_hover_readback(edge, Some(42), 43, Some(edge)), edge);
    }
    #[test]
    fn camera_navigation_keeps_click_context_but_cancel_and_history_changes_invalidate_it() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        let context = app.click_context();
        app.cam.orbit(20., 10.);
        assert!(context == app.click_context());
        app.cancel_command();
        assert!(context != app.click_context());
        let context = app.click_context();
        app.snapshot();
        assert!(context != app.click_context());
        let context = app.click_context();
        app.start_extrude(None);
        assert!(context != app.click_context());
    }
    #[test]
    fn extrude_edit_restores_taper_and_persistent_target_face() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        let shape = kernel::Shape::boxed(20., 20., 10.).unwrap();
        app.doc.add_import("target", &shape).unwrap();
        let body = &app.doc.state().bodies[0];
        let face = app.doc.state().face_ref(body.id, 1).unwrap();
        let id = app.doc.add_feature(FeatureKind::Extrude(ExtrudeFeature {
            sketch: 0,
            profiles: vec![],
            faces: vec![face.clone()],
            extent: Extent::Tapered {
                extent: Box::new(Extent::ToFace {
                    face: face.clone(),
                    offset: "2 mm".into(),
                }),
                angle: "-4 deg".into(),
            },
            op: Operation::NewBody,
        }));
        app.start_extrude(Some(id));
        let Command::Extrude(c) = &app.cmd else {
            panic!()
        };
        assert!(c.kind == ExtentKind::ToFace);
        assert_eq!(c.target, Some(face));
        assert_eq!(c.offset, "2 mm");
        assert_eq!(c.taper, "-4 deg");
        let (_, FeatureKind::Extrude(f)) = app.command_feature().unwrap() else {
            panic!()
        };
        assert!(matches!(f.extent, Extent::Tapered { .. }));
    }
    #[test]
    fn to_face_into_a_body_defaults_to_cut_and_target_mode_changes_invalidate_clicks() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc
            .add_import("Box", &kernel::Shape::boxed(20., 20., 10.).unwrap())
            .unwrap();
        let body = &app.doc.state().bodies[0];
        let top = (1..=body.shape.face_count() as u32)
            .find(|&f| {
                body.shape
                    .face_info(f)
                    .unwrap()
                    .plane
                    .is_some_and(|p| p.normal()[2] > 0.99)
            })
            .unwrap();
        let bottom = (1..=body.shape.face_count() as u32)
            .find(|&f| {
                body.shape
                    .face_info(f)
                    .unwrap()
                    .plane
                    .is_some_and(|p| p.normal()[2] < -0.99)
            })
            .unwrap();
        let source = app.doc.state().face_ref(body.id, top).unwrap();
        let target = app.doc.state().face_ref(body.id, bottom).unwrap();
        app.start_extrude(None);
        let context = app.click_context();
        let Command::Extrude(c) = &mut app.cmd else {
            panic!()
        };
        c.faces = vec![source];
        c.target = Some(target);
        c.kind = ExtentKind::ToFace;
        c.pick_target = true;
        assert!(context != app.click_context());
        let Command::Extrude(c) = &mut app.cmd else {
            panic!()
        };
        c.pick_target = false;
        app.rebuild_preview();
        let Command::Extrude(c) = &app.cmd else {
            panic!()
        };
        assert_eq!(c.op, Operation::Cut);
    }
}

#[cfg(test)]
mod drawing_click_tests {
    use super::*;
    fn eval(t: &str, u: Unit) -> Result<f64, String> {
        refusion_model::expr::eval(t, u, &HashMap::new()).map_err(|e| e.to_string())
    }
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
    fn press(s: &SketchSession) -> DrawingPress {
        DrawingPress {
            at: Pos2::new(100., 100.),
            time: 1.,
            feature: s.feature,
            tool: s.tool,
            started_operation: !s.in_progress(),
        }
    }
    #[test]
    fn moving_click_places_once_in_every_drawing_tool() {
        for tool in [
            Tool::Line,
            Tool::Rectangle,
            Tool::CenterRectangle,
            Tool::Circle,
            Tool::Arc,
            Tool::ThreePointArc,
            Tool::TangentArc,
            Tool::Point,
            Tool::Offset,
        ] {
            let mut sk = Sketch::new();
            let line = sk.add_line_pts([0., 0.], [20., 0.]);
            let endpoint = sk.line_pts(line).unwrap().0;
            let mut s = session(tool);
            let press = press(&s);
            let hover = match tool {
                Tool::TangentArc => Some(endpoint),
                Tool::Offset => Some(line),
                _ => None,
            };
            s.click(&mut sk, hover, [0., 0.], Pos2::ZERO, &eval, false);
            let entities = sk.entities.len();
            let constraints = sk.constraints.len();
            // A quick click moves thirty pixels before release. That motion
            // must not turn the same button gesture into another tool step.
            let release = Pos2::new(130., 115.);
            assert!(!press.finishes_drag(&s, release, 1.08, true));
            if press.finishes_drag(&s, release, 1.08, true) {
                s.click(&mut sk, None, [3., 1.5], release, &eval, false);
            }
            assert_eq!(sk.entities.len(), entities);
            assert_eq!(sk.constraints.len(), constraints);
        }
    }
    #[test]
    fn ongoing_line_chain_never_places_an_extra_segment_on_release() {
        let mut sk = Sketch::new();
        let mut s = session(Tool::Line);
        s.click(&mut sk, None, [0., 0.], Pos2::ZERO, &eval, false);
        for (i, p) in [[10., 3.], [18., 8.], [22., 12.]].into_iter().enumerate() {
            let press = press(&s);
            s.click(&mut sk, None, p, Pos2::ZERO, &eval, false);
            // Even a slow moving click is not a new press-drag operation.
            assert!(!press.finishes_drag(&s, Pos2::new(160., 130.), 2., true));
            assert_eq!(
                sk.entities
                    .values()
                    .filter(|e| matches!(e.geom, Geom::Line { .. }))
                    .count(),
                i + 1
            );
            assert_eq!(
                sk.entities
                    .values()
                    .filter(|e| matches!(e.geom, Geom::Point(_)))
                    .count(),
                i + 2
            );
        }
    }
    #[test]
    fn deliberate_initial_drag_still_finishes_lines_rectangles_and_circles() {
        for tool in [
            Tool::Line,
            Tool::Rectangle,
            Tool::CenterRectangle,
            Tool::Circle,
        ] {
            let mut sk = Sketch::new();
            let mut s = session(tool);
            let press = press(&s);
            s.click(&mut sk, None, [0., 0.], Pos2::ZERO, &eval, false);
            let release = Pos2::new(160., 130.);
            assert!(press.finishes_drag(&s, release, 1.6, true));
            assert!(
                s.click(&mut sk, None, [6., 3.], release, &eval, false)
                    .changed
            );
            let curves = sk
                .entities
                .values()
                .filter(|e| !matches!(e.geom, Geom::Point(_)) && !e.construction)
                .count();
            assert_eq!(
                curves,
                if matches!(tool, Tool::Rectangle | Tool::CenterRectangle) {
                    4
                } else {
                    1
                }
            );
        }
    }
    #[test]
    fn release_requires_a_real_drag_and_the_original_tool_context() {
        let mut sk = Sketch::new();
        let mut s = session(Tool::Line);
        let press = press(&s);
        s.click(&mut sk, None, [0., 0.], Pos2::ZERO, &eval, false);
        assert!(!press.finishes_drag(&s, Pos2::new(101., 100.), 2., true));
        assert!(!press.finishes_drag(&s, Pos2::new(160., 130.), 2., false));
        s.feature = 2;
        assert!(!press.finishes_drag(&s, Pos2::new(160., 130.), 2., true));
        s.feature = 1;
        s.set_tool(Tool::Circle);
        s.click(&mut sk, None, [0., 0.], Pos2::ZERO, &eval, false);
        assert!(!press.finishes_drag(&s, Pos2::new(160., 130.), 2., true));
        s.cancel();
        assert!(!press.finishes_drag(&s, Pos2::new(160., 130.), 2., true));
    }
}

#[cfg(test)]
mod cad_input_parity_tests {
    use super::*;
    fn drain(app: &mut RefusionApp) {
        app.compute.submit(crate::compute::Request {
            ticket: u64::MAX,
            replay: true,
            doc: Document::new(),
            cached: HashMap::new(),
            edit: Box::new(|_| None),
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(Ok(result)) = app.compute.poll() {
                if result.ticket == u64::MAX {
                    break;
                }
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn projection_app() -> (RefusionApp, FeatureId, Pick) {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc
            .add_import("box", &kernel::Shape::boxed(10., 10., 10.).unwrap())
            .unwrap();
        let id = app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch: Sketch::new(),
            projections: vec![],
        }));
        app.mode = Mode::Sketch(SketchSession::new(
            id,
            app.doc.state().sketches[&id].frame,
            app.doc.marker,
        ));
        let Mode::Sketch(s) = &mut app.mode else {
            panic!()
        };
        s.set_tool(Tool::Project);
        let body = &app.doc.state().bodies[0];
        let face = (1..=body.shape.face_count() as u32)
            .find(|&f| {
                body.shape
                    .face_info(f)
                    .unwrap()
                    .plane
                    .is_some_and(|p| p.normal()[2] > 0.99)
            })
            .unwrap();
        let pick = Pick::Face {
            body: body.id,
            face,
        };
        (app, id, pick)
    }
    #[test]
    fn project_is_staged_cancelable_and_commits_one_undo_step() {
        let (mut app, id, pick) = projection_app();
        let before = app.doc.to_json();
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        assert_eq!(app.doc.to_json(), before);
        assert!(app.undo.is_empty());
        let Mode::Sketch(s) = &mut app.mode else {
            panic!()
        };
        assert_eq!(s.project_edges.len(), 4);
        s.cancel();
        assert!(s.project_edges.is_empty());
        assert_eq!(app.doc.to_json(), before);
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        app.commit_projection();
        assert_eq!(app.undo.len(), 1);
        let Some(FeatureKind::Sketch(sf)) = app.doc.feature(id).map(|f| &f.kind) else {
            panic!()
        };
        assert_eq!(sf.projections.len(), 4);
        let committed = app.doc.to_json();
        app.undo();
        assert_eq!(app.doc.to_json(), before);
        app.redo();
        assert_eq!(app.doc.to_json(), committed);
        drain(&mut app);
    }
    #[test]
    fn projection_commit_is_atomic_on_bad_references_and_deduplicates_existing_sources() {
        let (mut app, id, pick) = projection_app();
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        let before = app.doc.to_json();
        let Mode::Sketch(session) = &mut app.mode else {
            panic!()
        };
        let mut invalid = session.project_edges[0].clone();
        invalid.body = BodyId::MAX;
        invalid.a = u64::MAX;
        invalid.b = u64::MAX;
        invalid.hint = [1e12; 3];
        session.project_edges.push(invalid);
        app.commit_projection();
        assert_eq!(app.doc.to_json(), before);
        assert!(app.undo.is_empty());
        let Mode::Sketch(session) = &mut app.mode else {
            panic!()
        };
        session.project_edges.pop();
        app.commit_projection();
        let count = RefusionApp::sketch_ref(&app.doc, id)
            .unwrap()
            .entities
            .len();
        let Mode::Sketch(session) = &mut app.mode else {
            panic!()
        };
        session.set_tool(Tool::Project);
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        app.commit_projection();
        assert_eq!(
            RefusionApp::sketch_ref(&app.doc, id)
                .unwrap()
                .entities
                .len(),
            count
        );
        assert_eq!(app.undo.len(), 1);
        drain(&mut app);
    }
    #[test]
    fn project_filter_changes_context_and_unlinked_geometry_replays_without_sources() {
        let (mut app, id, pick) = projection_app();
        let context = app.click_context();
        let Mode::Sketch(s) = &mut app.mode else {
            panic!()
        };
        s.project_bodies = true;
        s.projection_link = false;
        assert!(context != app.click_context());
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        let Mode::Sketch(s) = &app.mode else { panic!() };
        assert_eq!(s.project_edges.len(), 12);
        // A second pick toggles the complete body off, without changing the sketch.
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        let Mode::Sketch(s) = &mut app.mode else {
            panic!()
        };
        assert!(s.project_edges.is_empty());
        s.project_bodies = false;
        app.on_click(Some(pick), None, Pos2::ZERO, false);
        app.commit_projection();
        let Some(FeatureKind::Sketch(sf)) = app.doc.feature(id).map(|f| &f.kind) else {
            panic!()
        };
        assert!(sf.projections.is_empty());
        assert!(!sf.sketch.entities.is_empty());
        let count = sf.sketch.entities.len();
        app.doc.features[0].suppressed = true;
        let replay = Document::from_json(&app.doc.to_json()).unwrap();
        assert!(replay.errors.is_empty());
        let FeatureKind::Sketch(sf) = &replay.feature(id).unwrap().kind else {
            panic!()
        };
        assert_eq!(sf.sketch.entities.len(), count);
        drain(&mut app);
    }
    #[test]
    fn overlapping_regions_toggle_independently_in_the_extrude_command() {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc = Document::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../refusion-model/tests/fixtures/intersect.rfz"
        )))
        .unwrap();
        app.start_extrude(None);
        let indices: Vec<_> = app.doc.state().sketches[&1]
            .profiles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.signature().len() == 3)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(indices.len(), 2);
        for &index in &indices {
            app.on_click(
                Some(Pick::Profile { sketch: 1, index }),
                None,
                Pos2::ZERO,
                false,
            );
        }
        let Command::Extrude(c) = &app.cmd else {
            panic!()
        };
        assert_eq!(c.profiles.len(), 2);
        app.on_click(
            Some(Pick::Profile {
                sketch: 1,
                index: indices[0],
            }),
            None,
            Pos2::ZERO,
            false,
        );
        let Command::Extrude(c) = &app.cmd else {
            panic!()
        };
        assert_eq!(c.profiles.len(), 1);
        let wanted = &app.doc.state().sketches[&1].profiles[indices[1]];
        let actual = match_profile(&app.doc.state().sketches[&1].profiles, &c.profiles[0]).unwrap();
        assert!(std::ptr::eq(wanted, actual));
        app.cancel_command();
        drain(&mut app);
    }
}

#[cfg(test)]
mod hole_shell_command_tests {
    use super::*;
    fn app() -> RefusionApp {
        let mut app = RefusionApp::empty(egui::Context::default(), None, String::new());
        app.doc
            .add_import("stock", &kernel::Shape::boxed(20., 20., 20.).unwrap())
            .unwrap();
        app.viewport_rect = egui::Rect::from_min_size(Pos2::ZERO, Vec2::splat(800.));
        app.cam.fit(DVec3::ZERO, DVec3::splat(20.));
        app.cam.tick(1.);
        app
    }
    fn face_pick(app: &RefusionApp, n: [f64; 3]) -> Pick {
        let b = &app.doc.state().bodies[0];
        let face = (1..=b.shape.face_count() as u32)
            .find(|&i| {
                b.shape
                    .face_info(i)
                    .and_then(|f| f.plane)
                    .is_some_and(|f| kernel::dot(f.normal(), n) > 0.99)
            })
            .unwrap();
        Pick::Face { body: b.id, face }
    }
    fn wait(app: &mut RefusionApp) {
        let start = Instant::now();
        while app.preview_pending || app.replay_pending {
            assert!(start.elapsed() < Duration::from_secs(10));
            app.poll_compute();
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn draw(app: &mut RefusionApp, events: Vec<egui::Event>) -> egui::FullOutput {
        let ctx = app.ctx.clone();
        ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(1500., 1400.),
                )),
                events,
                ..Default::default()
            },
            |ctx| app.command_window(ctx),
        )
    }
    fn position(output: &egui::FullOutput, text: &str) -> Pos2 {
        output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::Shape::Text(t) if t.galley.text() == text => {
                    Some(t.pos + t.galley.size() / 2.)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing {text}"))
    }
    fn click(app: &mut RefusionApp, at: Pos2) -> egui::FullOutput {
        draw(
            app,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
        );
        draw(
            app,
            vec![egui::Event::PointerButton {
                pos: at,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }],
        )
    }
    fn choose(app: &mut RefusionApp, current: &str, next: &str) {
        draw(app, vec![]);
        let output = draw(app, vec![]);
        let at = position(&output, current);
        click(app, at);
        let output = draw(app, vec![]);
        let at = position(&output, next);
        click(app, at);
    }
    #[test]
    fn feature_pattern_timeline_click_toggles_source_without_opening_an_editor() {
        let mut app = app();
        let mut hole = HoleFeature::new(1, PlaneRef::XY);
        hole.centers = vec![["3 mm".into(), "3 mm".into()]];
        hole.flip = true;
        hole.tip_angle = None;
        let source = app.doc.add_feature(FeatureKind::Hole(hole));
        let name = format!("      {}", app.doc.feature(source).unwrap().name);
        app.run_action(Action::RectPattern);
        wait(&mut app);
        choose(&mut app, "Bodies", "Features");
        wait(&mut app);
        let draw_timeline = |app: &mut RefusionApp, events| {
            let ctx = app.ctx.clone();
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(1500., 1400.),
                    )),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| app.timeline(ui));
                },
            )
        };
        draw_timeline(&mut app, vec![]);
        for selected in [true, false] {
            let output = draw_timeline(&mut app, vec![]);
            let at = position(&output, &name);
            draw_timeline(
                &mut app,
                vec![
                    egui::Event::PointerMoved(at),
                    egui::Event::PointerButton {
                        pos: at,
                        button: PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                ],
            );
            draw_timeline(
                &mut app,
                vec![egui::Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }],
            );
            let Command::Solid {
                kind: FeatureKind::Pattern(p),
                ..
            } = &app.cmd
            else {
                panic!()
            };
            assert_eq!(p.features.contains(&source), selected);
            wait(&mut app);
        }
        app.cancel_command();
        wait(&mut app);
    }
    #[test]
    fn feature_pattern_panel_selection_preview_commit_edit_cancel_and_undo() {
        let mut app = app();
        let mut hole = HoleFeature::new(1, PlaneRef::XY);
        hole.centers = vec![["3 mm".into(), "3 mm".into()]];
        hole.flip = true;
        hole.tip_angle = None;
        hole.diameter = "4 mm".into();
        hole.extent = HoleExtent::ThroughAll;
        let source = app.doc.add_feature(FeatureKind::Hole(hole));
        let name = app.doc.feature(source).unwrap().name.clone();
        app.run_action(Action::RectPattern);
        wait(&mut app);
        choose(&mut app, "Bodies", "Features");
        wait(&mut app);
        assert!(app.preview_error.is_some());
        let output = draw(&mut app, vec![]);
        click(&mut app, position(&output, &name));
        let Command::Solid {
            kind: FeatureKind::Pattern(p),
            ..
        } = &mut app.cmd
        else {
            panic!()
        };
        assert_eq!(p.features, vec![source]);
        assert!(p.bodies.is_empty());
        if let PatternKind::Rectangular { distance1, .. } = &mut p.kind {
            *distance1 = "5 mm".into();
        }
        app.rebuild_preview();
        wait(&mut app);
        assert!(app.preview_error.is_none(), "{:?}", app.preview_error);
        let before = app.doc.state().bodies[0].shape.volume();
        app.commit_command();
        wait(&mut app);
        let after = app.doc.state().bodies[0].shape.volume();
        assert!((after - (8000. - 3. * std::f64::consts::PI * 4. * 20.)).abs() < 1e-4);
        app.undo();
        wait(&mut app);
        assert!((app.doc.state().bodies[0].shape.volume() - before).abs() < 1e-4);
        app.redo();
        wait(&mut app);
        assert!((app.doc.state().bodies[0].shape.volume() - after).abs() < 1e-4);
        let pattern = app.doc.features.last().unwrap().id;
        app.edit_feature(pattern);
        wait(&mut app);
        let Command::Solid {
            kind: FeatureKind::Pattern(p),
            ..
        } = &app.cmd
        else {
            panic!()
        };
        assert_eq!(p.objects, PatternObjects::Features);
        assert_eq!(p.features, vec![source]);
        let output = draw(&mut app, vec![]);
        click(&mut app, position(&output, "Clear objects"));
        wait(&mut app);
        assert!(app.preview_error.is_some());
        app.cancel_command();
        wait(&mut app);
        assert!((app.doc.state().bodies[0].shape.volume() - after).abs() < 1e-4);
    }
    #[test]
    fn hole_face_placement_preview_commit_edit_cancel_and_undo_redo() {
        let mut app = app();
        let top = face_pick(&app, [0., 0., 1.]);
        app.run_action(Action::Hole);
        wait(&mut app);
        assert_eq!(app.doc.features.len(), 1);
        let xy = app
            .cam
            .project(DVec3::new(10., 10., 20.), DVec2::splat(800.))
            .unwrap();
        app.on_click(Some(top), None, Pos2::new(xy.x as f32, xy.y as f32), false);
        wait(&mut app);
        assert!(app.preview_error.is_none(), "{:?}", app.preview_error);
        let Command::Solid {
            kind: FeatureKind::Hole(h),
            ..
        } = &app.cmd
        else {
            panic!()
        };
        let center = [
            app.doc.eval(&h.centers[0][0], Unit::Length).unwrap(),
            app.doc.eval(&h.centers[0][1], Unit::Length).unwrap(),
        ];
        let world = app
            .doc
            .state()
            .plane_frame(&h.plane)
            .unwrap()
            .to_world(center);
        assert!((world[0] - 10.).abs() < 1e-5 && (world[1] - 10.).abs() < 1e-5);
        app.commit_command();
        assert_eq!(app.doc.features.len(), 2);
        assert_eq!(app.undo.len(), 1);
        let volume = app.doc.state().bodies[0].shape.volume();
        assert!(volume < 8000.);
        app.undo();
        wait(&mut app);
        assert_eq!(app.doc.features.len(), 1);
        app.redo();
        wait(&mut app);
        assert_eq!(app.doc.features.len(), 2);
        let id = app.doc.features[1].id;
        app.edit_feature(id);
        wait(&mut app);
        let Command::Solid {
            kind: FeatureKind::Hole(h),
            ..
        } = &mut app.cmd
        else {
            panic!()
        };
        h.diameter = "6 mm".into();
        app.rebuild_preview();
        wait(&mut app);
        app.cancel_command();
        assert!((app.doc.state().bodies[0].shape.volume() - volume).abs() < 1e-6);
        app.edit_feature(id);
        wait(&mut app);
        app.commit_command();
        assert_eq!(app.doc.features.len(), 2);
    }
    #[test]
    fn hole_panel_type_extent_tip_and_target_selection() {
        let mut app = app();
        app.selection = vec![face_pick(&app, [0., 0., 1.])];
        app.run_action(Action::Hole);
        wait(&mut app);
        choose(&mut app, "Simple", "Counterbore");
        wait(&mut app);
        let output = draw(&mut app, vec![]);
        position(&output, "Counterbore Diameter");
        position(&output, "Counterbore Depth");
        choose(&mut app, "Distance", "To");
        assert!(matches!(
            app.cmd,
            Command::Solid {
                pick: SolidPick::HoleTarget,
                ..
            }
        ));
        let Command::Solid {
            kind: FeatureKind::Hole(h),
            ..
        } = &mut app.cmd
        else {
            panic!()
        };
        let HoleExtent::ToFace { offset, .. } = &mut h.extent else {
            panic!()
        };
        *offset = "-1 mm".into();
        let bottom = face_pick(&app, [0., 0., -1.]);
        app.on_click(Some(bottom), None, Pos2::ZERO, false);
        wait(&mut app);
        assert!(app.preview_error.is_none(), "{:?}", app.preview_error);
        choose(&mut app, "Angle", "Flat");
        wait(&mut app);
        let Command::Solid {
            kind: FeatureKind::Hole(h),
            ..
        } = &app.cmd
        else {
            panic!()
        };
        assert!(h.tip_angle.is_none());
        assert!(matches!(h.kind, HoleKind::Counterbore { .. }));
        assert!(matches!(&h.extent, HoleExtent::ToFace { offset, .. } if offset == "-1 mm"));
        app.cancel_command();
        assert_eq!(app.doc.features.len(), 1);
    }
    #[test]
    fn shell_both_panel_independent_fields_preview_and_edit_restore() {
        let mut app = app();
        app.run_action(Action::Shell);
        wait(&mut app);
        choose(&mut app, "Inside", "Both");
        wait(&mut app);
        let output = draw(&mut app, vec![]);
        position(&output, "Inside Thickness");
        position(&output, "Outside Thickness");
        let Command::Solid {
            kind: FeatureKind::Shell(f),
            ..
        } = &mut app.cmd
        else {
            panic!()
        };
        f.outside_thickness = Some("2 mm".into());
        app.rebuild_preview();
        wait(&mut app);
        assert!(app.preview_error.is_none(), "{:?}", app.preview_error);
        app.commit_command();
        let id = app.doc.features[1].id;
        assert!(
            (app.doc.state().bodies[0].shape.volume() - (24f64.powi(3) - 18f64.powi(3))).abs()
                < 1e-5
        );
        app.edit_feature(id);
        wait(&mut app);
        let Command::Solid {
            kind: FeatureKind::Shell(f),
            ..
        } = &app.cmd
        else {
            panic!()
        };
        assert_eq!(f.outside_thickness.as_deref(), Some("2 mm"));
        choose(&mut app, "Both", "Outside");
        wait(&mut app);
        let Command::Solid {
            kind: FeatureKind::Shell(f),
            ..
        } = &app.cmd
        else {
            panic!()
        };
        assert!(f.outward && f.outside_thickness.is_none());
        app.cancel_command();
    }
}

#[cfg(test)]
mod cad_ribbon_tests {
    use super::*;
    fn new_app() -> RefusionApp {
        RefusionApp::empty(egui::Context::default(), None, String::new())
    }
    fn draw(app: &mut RefusionApp, events: Vec<egui::Event>) -> egui::FullOutput {
        let ctx = app.ctx.clone();
        ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(1500., 900.),
                )),
                events,
                ..Default::default()
            },
            |ctx| {
                egui::TopBottomPanel::top("ribbon_test").show(ctx, |ui| app.ribbon(ui));
            },
        )
    }
    fn text_positions(out: &egui::FullOutput) -> Vec<(String, Pos2)> {
        out.shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some((
                    t.galley.text().trim().to_owned(),
                    t.pos + t.galley.size() / 2.,
                )),
                _ => None,
            })
            .collect()
    }
    fn at(out: &egui::FullOutput, text: &str) -> Pos2 {
        text_positions(out)
            .iter()
            .find(|(t, _)| t == text)
            .unwrap_or_else(|| panic!("missing {text}: {:?}", text_positions(out)))
            .1
    }
    fn click(app: &mut RefusionApp, at: Pos2) -> egui::FullOutput {
        draw(
            app,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
        );
        draw(
            app,
            vec![egui::Event::PointerButton {
                pos: at,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }],
        )
    }
    fn open(app: &mut RefusionApp, label: &str) -> egui::FullOutput {
        draw(app, vec![]);
        let out = draw(app, vec![]);
        click(app, at(&out, label));
        draw(app, vec![])
    }
    fn ordered(out: &egui::FullOutput, labels: &[&str]) {
        let ys: Vec<_> = labels.iter().map(|t| at(out, t).y).collect();
        assert!(
            ys.windows(2).all(|p| p[0] < p[1]),
            "wrong ordering {labels:?}: {ys:?}"
        );
    }
    fn sketch_app() -> (RefusionApp, FeatureId, EntityId, EntityId) {
        let mut app = new_app();
        let mut sk = Sketch::new();
        let a = sk.add_point([0., 0.]);
        let b = sk.add_point([3., 4.]);
        let id = app.doc.add_feature(FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::XY,
            sketch: sk,
            projections: vec![],
        }));
        app.enter_sketch(id, 1);
        (app, id, a, b)
    }
    #[test]
    fn solid_create_groups_patterns_before_mirror_and_promotes_supported_cad_tools() {
        let mut app = new_app();
        let out = open(&mut app, "CREATE");
        ordered(
            &out,
            &[
                "Create Sketch",
                "Extrude",
                "Revolve",
                "Hole",
                "Pattern",
                "Mirror",
            ],
        );
        click(&mut app, at(&out, "Pattern"));
        let out = draw(&mut app, vec![]);
        ordered(&out, &["Rectangular Pattern", "Circular Pattern"]);
        click(&mut app, at(&out, "Circular Pattern"));
        assert!(matches!(
            app.cmd,
            Command::Solid {
                kind: FeatureKind::Pattern(PatternFeature {
                    kind: PatternKind::Circular { .. },
                    ..
                }),
                ..
            }
        ));
        app.cancel_command();
        let out = open(&mut app, "MODIFY");
        ordered(
            &out,
            &[
                "Press Pull",
                "Fillet",
                "Chamfer",
                "Shell",
                "Change Parameters",
            ],
        );
    }
    #[test]
    fn sketch_arc_modify_and_constraint_order_match_supported_cad_commands() {
        let (mut app, _, _, _) = sketch_app();
        let out = open(&mut app, "CREATE");
        click(&mut app, at(&out, "Arc"));
        let out = draw(&mut app, vec![]);
        ordered(&out, &["3-Point Arc", "Center Point Arc", "Tangent Arc"]);
        click(&mut app, at(&out, "3-Point Arc"));
        assert!(matches!(&app.mode,Mode::Sketch(s) if s.tool==Tool::ThreePointArc));
        let out = open(&mut app, "MODIFY");
        ordered(&out, &["Offset", "Trim", "Extend", "Change Parameters"]);
        // Close the open popup before checking another panel.
        click(&mut app, at(&out, "Extend"));
        let out = open(&mut app, "CONSTRAINTS");
        ordered(
            &out,
            &ConKind::ALL.iter().map(|c| c.label()).collect::<Vec<_>>(),
        );
    }
    #[test]
    fn inspect_measure_keeps_active_sketch_and_measures_uncommitted_points() {
        let (mut app, id, a, b) = sketch_app();
        let marker = app.doc.marker;
        let undo = app.undo.len();
        app.run_action(Action::Measure);
        assert!(matches!(&app.mode,Mode::Sketch(s) if s.feature==id));
        assert_eq!(app.doc.marker, marker);
        app.on_click(
            Some(Pick::Ent { sketch: id, ent: a }),
            None,
            Pos2::ZERO,
            false,
        );
        app.on_click(
            Some(Pick::Ent { sketch: id, ent: b }),
            None,
            Pos2::ZERO,
            false,
        );
        let Command::Measure(m) = &app.cmd else {
            panic!()
        };
        assert!((m.result.as_ref().unwrap().distance.unwrap().0 - 5.).abs() < 1e-6);
        let ctx = app.ctx.clone();
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(1500., 900.),
                )),
                ..Default::default()
            },
            |ctx| app.command_window(ctx),
        );
        app.run_action(Action::Tool(Tool::Line));
        assert!(matches!(app.cmd, Command::None));
        assert!(matches!(&app.mode,Mode::Sketch(s) if s.tool==Tool::Line));
        assert_eq!(app.undo.len(), undo);
    }
    #[test]
    #[ignore = "requires a Vulkan adapter; renders toolbar and icon artwork for visual inspection"]
    fn cad_ribbon_and_tool_icons_render_on_gpu() {
        use crate::gpu::async_gpu_tests::{block_on, GPU_TEST_LOCK};
        use eframe::{egui_wgpu, wgpu};
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let mut renderer =
            egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, None, 1, false);
        let (mut app, _, _, _) = sketch_app();
        let ctx = app.ctx.clone();
        let mut output = None;
        let mut sprite_rects = Vec::new();
        for _ in 0..2 {
            sprite_rects.clear();
            output=Some(ctx.run(egui::RawInput {screen_rect:Some(egui::Rect::from_min_size(Pos2::ZERO,Vec2::new(1536.,1280.))),..Default::default()},|ctx| {
            egui::CentralPanel::default().show(ctx,|ui| {
                ui.heading("REFUSION / CAD TOOL ICONS");ui.label("Native samples: 16 / 20 / 24 / 32 px · no enlargement");ui.add_space(16.);
                egui::Grid::new("icon_gallery").spacing(Vec2::new(16.,12.)).show(ui,|ui| {
                    for (i,icon) in crate::tool_icons::Icon::ALL.into_iter().enumerate() {ui.vertical(|ui| {
                        let (rect,_)=ui.allocate_exact_size(Vec2::new(140.,96.),Sense::hover());
                        for (i,size) in [16.,20.,24.,32.].into_iter().enumerate() {
                            let x=rect.left()+[12.,40.,74.,118.][i];
                            for (theme,y) in [("dark",rect.center().y-22.),("light",rect.center().y+22.)] {
                                if theme=="light" {ui.painter().rect_filled(egui::Rect::from_center_size(Pos2::new(x,y),Vec2::new(40.,40.)),0.,Color32::from_rgb(242,243,246));}
                                let sample=egui::Rect::from_center_size(Pos2::new(x,y),Vec2::splat(size));
                                crate::tool_icons::paint_with_theme(ui.painter(),sample,icon,theme=="dark");
                                sprite_rects.push(serde_json::json!({"icon":icon.name(),"theme":theme,"size":size,"x":sample.left(),"y":sample.top(),"w":sample.width(),"h":sample.height()}));
                            }
                        }
                        ui.label(icon.name());
                    });if i%8==7 {ui.end_row();}}
                });ui.separator();ui.heading("SOLID TOOLBAR");let mode=std::mem::replace(&mut app.mode,Mode::Model);ui.push_id("solid_preview",|ui|app.ribbon(ui));app.mode=mode;
                ui.separator();ui.heading("SKETCH TOOLBAR");ui.push_id("sketch_preview",|ui|app.ribbon(ui));
            });
        }));
            for (id, delta) in &output.as_ref().unwrap().textures_delta.set {
                renderer.update_texture(&device, &queue, *id, delta);
            }
        }
        let output = output.unwrap();
        let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        for (id, delta) in output.textures_delta.set {
            renderer.update_texture(&device, &queue, id, &delta);
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ribbon preview"),
            size: wgpu::Extent3d {
                width: 1536,
                height: 1280,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [1536, 1280],
            pixels_per_point: output.pixels_per_point,
        };
        let commands = renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &desc);
        {
            let view = texture.create_view(&Default::default());
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ribbon UI"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            renderer.render(&mut pass, &jobs, &desc);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1536 * 1280 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1536 * 4),
                    rows_per_image: Some(1280),
                },
            },
            wgpu::Extent3d {
                width: 1536,
                height: 1280,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(commands.into_iter().chain([encoder.finish()]));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
        let pixels = buffer.slice(..).get_mapped_range();
        assert!(
            pixels
                .chunks_exact(4)
                .filter(|p| p[0] != p[1] || p[1] != p[2])
                .count()
                > 15000
        );
        for sample in &sprite_rects {
            let (x, y, w, h) = (
                sample["x"].as_f64().unwrap() as u32,
                sample["y"].as_f64().unwrap() as u32,
                sample["w"].as_f64().unwrap() as u32,
                sample["h"].as_f64().unwrap() as u32,
            );
            let mut visible = 0;
            for row in y..y + h {
                for col in x..x + w {
                    let offset = ((row * 1536 + col) * 4) as usize;
                    let p = &pixels[offset..offset + 4];
                    if p[0].abs_diff(p[1]) > 10 || p[1].abs_diff(p[2]) > 10 {
                        visible += 1;
                    }
                }
            }
            assert!(
                visible >= 3,
                "missing small/medium artwork: {sample} ({visible} pixels)"
            );
        }
        if let Some(dir) = std::env::var_os("REFUSION_RENDER_ARTIFACTS") {
            let mut ppm = b"P6\n1536 1280\n255\n".to_vec();
            for p in pixels.chunks_exact(4) {
                ppm.extend_from_slice(&p[..3]);
            }
            std::fs::write(std::path::Path::new(&dir).join("cad-tool-icons.ppm"), ppm).unwrap();
            std::fs::write(
                std::path::Path::new(&dir).join("cad-tool-icons-rects.json"),
                serde_json::to_string_pretty(&sprite_rects).unwrap(),
            )
            .unwrap();
        }
    }
}
