//! Turns the model into GPU draw data and a pick table.

use crate::gpu::{rgba, BodyDraw, FlatVertex, FrameData, LineInst, PointInst};
use glam::{DVec3, Vec3};
use refusion_model::kernel::{Frame, Mesh};
use refusion_model::sketch::{EntityId, Geom, Profile, Sketch};
use refusion_model::{BodyId, FeatureId, State};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Something under the cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pick {
    Face {
        body: BodyId,
        face: u32,
    },
    Edge {
        body: BodyId,
        edge: u32,
    },
    Vertex {
        body: BodyId,
        vertex: u32,
    },
    Ent {
        sketch: FeatureId,
        ent: EntityId,
    },
    Profile {
        sketch: FeatureId,
        index: usize,
    },
    /// Origin plane: 0 = XY, 1 = XZ, 2 = YZ.
    Plane(u8),
    ConstructionPlane(FeatureId),
}

/// What may be picked right now.
#[derive(Clone, Copy, Debug, Default)]
pub struct PickFilter {
    pub faces: bool,
    pub edges: bool,
    pub sketch_ents: bool,
    pub profiles: bool,
    pub planes: bool,
    pub vertices: bool,
}

pub mod colors {
    use crate::gpu::rgba;
    pub fn body() -> u32 {
        rgba(196, 201, 208, 255)
    }
    pub fn edge() -> u32 {
        rgba(35, 38, 44, 255)
    }
    pub fn free() -> u32 {
        rgba(24, 92, 224, 255)
    }
    pub fn fixed() -> u32 {
        rgba(18, 18, 22, 255)
    }
    pub fn construction() -> u32 {
        rgba(232, 140, 30, 255)
    }
    pub fn projected() -> u32 {
        rgba(150, 56, 200, 255)
    }
    pub fn inactive() -> u32 {
        rgba(70, 74, 86, 255)
    }
    pub fn profile() -> u32 {
        rgba(255, 196, 120, 70)
    }
    pub fn preview() -> u32 {
        rgba(24, 92, 224, 200)
    }
}

pub struct SceneInput<'a> {
    pub state: &'a State,
    /// Sketch features to draw: (id, sketch, frame, active?)
    pub sketches: Vec<(FeatureId, &'a Sketch, Frame, bool)>,
    /// Sketch whose profiles should be shown and pickable.
    pub profile_sketches: Vec<(FeatureId, &'a [Profile], Frame)>,
    pub hidden_bodies: &'a HashSet<BodyId>,
    pub hidden_planes: &'a HashSet<FeatureId>,
    pub filter: PickFilter,
    pub hover: Option<Pick>,
    pub selection: &'a [Pick],
    /// Entities to emphasise (e.g. those of a hovered constraint glyph).
    pub emphasis: &'a [Pick],
    pub show_planes: bool,
    pub grid: Option<(Frame, f64)>,
    /// Fixed grid spacing (None = adaptive).
    pub grid_step: Option<f64>,
    /// Extra transient geometry in world space (tool previews).
    pub preview_lines: Vec<([f64; 3], [f64; 3], u32, bool)>,
    pub preview_points: Vec<[f64; 3]>,
    /// World polylines drawn on top of everything (e.g. selected edges over a preview).
    pub overlay_polylines: Vec<(Vec<[f32; 3]>, u32)>,
    /// Per active-sketch entity: fully constrained?
    pub constrained: &'a HashMap<EntityId, bool>,
    pub view_scale: f64,
    /// Draw the active sketch's points (Sketch Palette "Show Points").
    pub show_points: bool,
    pub show_construction: bool,
    pub show_projected: bool,
}

pub struct Scene<'a> {
    pub frame: FrameData<'a>,
    pub picks: Vec<Pick>,
    /// Pick priority per pick (lower wins): points 0, curves 1, edges 2, profiles 3, faces 4, planes 5.
    pub prio: Vec<u8>,
}

fn v3(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

impl<'a> Scene<'a> {
    /// Allocate a pick id. Disabled picks still get an id (so selection
    /// highlighting works) but can never win a pick (priority 255).
    fn id(&mut self, p: Pick, prio: u8, enabled: bool) -> u32 {
        self.picks.push(p);
        self.prio.push(if enabled { prio } else { 255 });
        self.picks.len() as u32
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfileMesh {
    pub points: Vec<[f64; 2]>,
    pub indices: Vec<u32>,
}
#[derive(Default)]
pub struct ProfileCache {
    entries: HashMap<FeatureId, (Arc<Vec<Profile>>, Arc<Vec<ProfileMesh>>)>,
}
impl ProfileCache {
    pub fn install(
        &mut self,
        id: FeatureId,
        profiles: Arc<Vec<Profile>>,
        meshes: Arc<Vec<ProfileMesh>>,
    ) {
        self.entries.insert(id, (profiles, meshes));
    }
    fn meshes(&mut self, id: FeatureId, profiles: &[Profile]) -> &[ProfileMesh] {
        let entry = self
            .entries
            .entry(id)
            .or_insert_with(|| (Arc::new(vec![]), Arc::new(vec![])));
        if entry.0.as_slice() != profiles {
            *entry = (
                Arc::new(profiles.to_vec()),
                Arc::new(profiles.iter().map(triangulate_profile).collect()),
            );
        }
        &entry.1
    }
    pub fn retain(&mut self, live: &HashSet<FeatureId>) {
        self.entries.retain(|id, _| live.contains(id));
    }
}

#[cfg(test)]
pub fn build<'a>(inp: &SceneInput, meshes: &'a HashMap<u64, Arc<Mesh>>) -> Scene<'a> {
    build_cached(inp, meshes, &mut ProfileCache::default())
}

pub fn build_cached<'a>(
    inp: &SceneInput,
    meshes: &'a HashMap<u64, Arc<Mesh>>,
    profile_cache: &mut ProfileCache,
) -> Scene<'a> {
    let mut sc = Scene {
        frame: FrameData::default(),
        picks: Vec::new(),
        prio: Vec::new(),
    };
    let f = inp.filter;

    // ---- bodies
    for b in &inp.state.bodies {
        if inp.hidden_bodies.contains(&b.id) {
            continue;
        }
        let Some(mesh) = meshes.get(&b.shape.uid()) else {
            continue;
        };
        let base = sc.picks.len() as u32 + 1;
        let nf = b.shape.face_count() as u32;
        for face in 1..=nf {
            sc.picks.push(Pick::Face { body: b.id, face });
            sc.prio.push(if f.faces { 4 } else { 255 });
        }
        if f.vertices {
            for v in 1..=b.shape.vertex_count() as u32 {
                if let Some(p) = b.shape.vertex_pos(v) {
                    let id = sc.id(
                        Pick::Vertex {
                            body: b.id,
                            vertex: v,
                        },
                        0,
                        true,
                    );
                    sc.frame.points.push(PointInst {
                        pos: v3(p),
                        color: colors::edge(),
                        size: 6.0,
                        id,
                        shape: 0,
                    });
                }
            }
        }
        let edge_base = sc.picks.len() as u32 + 1;
        sc.frame.bodies.push(BodyDraw {
            key: if mesh.cache_key == 0 {
                b.shape.uid()
            } else {
                mesh.cache_key
            },
            mesh,
            base,
            edge_base,
            color: colors::body(),
        });
        for e in 1..=mesh.edge_count() as u32 {
            sc.id(
                Pick::Edge {
                    body: b.id,
                    edge: e,
                },
                2,
                f.edges,
            );
        }
    }

    // ---- origin planes
    if inp.show_planes {
        let s = (inp.view_scale * 0.25) as f32;
        let quads: [([f32; 3], [f32; 3], u32); 3] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], rgba(70, 110, 230, 60)),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], rgba(70, 190, 90, 60)),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], rgba(230, 80, 70, 60)),
        ];
        for (k, (u, v, c)) in quads.iter().enumerate() {
            let id = sc.id(Pick::Plane(k as u8), 5, f.planes);
            let base = sc.frame.flat_v.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = [
                    s * (a * u[0] + b * v[0]),
                    s * (a * u[1] + b * v[1]),
                    s * (a * u[2] + b * v[2]),
                ];
                sc.frame.flat_v.push(FlatVertex {
                    pos: p,
                    color: *c,
                    id,
                });
            }
            sc.frame
                .flat_i
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            let corners: Vec<[f32; 3]> = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .iter()
                .map(|(a, b)| {
                    [
                        s * (a * u[0] + b * v[0]),
                        s * (a * u[1] + b * v[1]),
                        s * (a * u[2] + b * v[2]),
                    ]
                })
                .collect();
            for i in 0..4 {
                sc.frame.lines.push(LineInst {
                    p0: corners[i],
                    p1: corners[(i + 1) % 4],
                    color: (c & 0x00FF_FFFF) | 0xC000_0000,
                    width: 1.2,
                    id,
                    style: 0,
                });
            }
        }
    }

    for (&feature, frame) in &inp.state.planes {
        if inp.hidden_planes.contains(&feature) {
            continue;
        }
        let id = sc.id(Pick::ConstructionPlane(feature), 5, f.planes);
        let s = inp.view_scale * 0.25;
        let corners: Vec<_> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .into_iter()
            .map(|(a, b)| v3(frame.to_world([a * s, b * s])))
            .collect();
        let base = sc.frame.flat_v.len() as u32;
        for &pos in &corners {
            sc.frame.flat_v.push(FlatVertex {
                pos,
                color: rgba(232, 140, 30, 50),
                id,
            });
        }
        sc.frame
            .flat_i
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        for i in 0..4 {
            sc.frame.lines.push(LineInst {
                p0: corners[i],
                p1: corners[(i + 1) % 4],
                color: colors::construction(),
                width: 1.2,
                id,
                style: 0,
            });
        }
    }
    sc.frame.under_count = sc.frame.flat_i.len() as u32;

    // ---- grid
    if let Some((fr, extent)) = inp.grid {
        let step = inp.grid_step.unwrap_or_else(|| grid_step(extent));
        let n_max = 400.0; // keep a fixed fine grid from exploding
        let step = step.max(extent / n_max);
        let n = (extent / step).ceil() as i32;
        let half = n as f64 * step;
        for i in -n..=n {
            let t = i as f64 * step;
            let every = (grid_major(step) / step).round().max(1.0) as i32;
            let major = i % every == 0;
            let c = if i == 0 {
                rgba(120, 125, 135, 0)
            } else if major {
                rgba(120, 125, 140, 110)
            } else {
                rgba(140, 145, 155, 55)
            };
            if i == 0 {
                continue;
            }
            for (a, b) in [([t, -half], [t, half]), ([-half, t], [half, t])] {
                sc.frame.lines.push(LineInst {
                    p0: v3(fr.to_world(a)),
                    p1: v3(fr.to_world(b)),
                    color: c,
                    width: if major { 1.2 } else { 1.0 },
                    id: 0,
                    style: 0,
                });
            }
        }
        // Axes through the origin
        let ax = |a: [f64; 3], b: [f64; 3], c: u32| LineInst {
            p0: v3(a),
            p1: v3(b),
            color: c,
            width: 1.6,
            id: 0,
            style: 0,
        };
        let e = half;
        sc.frame
            .lines
            .push(ax([-e, 0.0, 0.0], [e, 0.0, 0.0], rgba(214, 64, 60, 200)));
        sc.frame
            .lines
            .push(ax([0.0, -e, 0.0], [0.0, e, 0.0], rgba(60, 168, 70, 200)));
        sc.frame
            .lines
            .push(ax([0.0, 0.0, -e], [0.0, 0.0, e], rgba(60, 96, 220, 200)));
    }

    // ---- profiles: local-space triangulation changes only with geometry.
    for (sk, profiles, fr) in &inp.profile_sketches {
        let meshes = profile_cache.meshes(*sk, profiles);
        for (index, mesh) in meshes.iter().enumerate() {
            let id = sc.id(Pick::Profile { sketch: *sk, index }, 3, f.profiles);
            let base = sc.frame.flat_v.len() as u32;
            sc.frame
                .flat_v
                .extend(mesh.points.iter().map(|&q| FlatVertex {
                    pos: v3(fr.to_world(q)),
                    color: colors::profile(),
                    id,
                }));
            sc.frame
                .flat_i
                .extend(mesh.indices.iter().map(|i| base + i));
        }
    }

    // ---- sketches
    for (sk_id, sk, fr, active) in &inp.sketches {
        let w = |p: [f64; 2]| v3(fr.to_world(p));
        for (&eid, e) in &sk.entities {
            if *active
                && (e.construction && !inp.show_construction || e.projected && !inp.show_projected)
            {
                continue;
            }
            let pick = Pick::Ent {
                sketch: *sk_id,
                ent: eid,
            };
            let is_point = matches!(e.geom, Geom::Point(_));
            if is_point {
                if !*active || !inp.show_points {
                    continue;
                }
                let id = sc.id(pick, 0, f.sketch_ents);
                let c = if e.projected {
                    colors::projected()
                } else if inp.constrained.get(&eid).copied().unwrap_or(false) {
                    colors::fixed()
                } else {
                    colors::free()
                };
                sc.frame.points.push(PointInst {
                    pos: w(sk.pos(eid)),
                    color: c,
                    size: 7.0,
                    id,
                    shape: 0,
                });
                continue;
            }
            let id = sc.id(pick, 1, f.sketch_ents);
            let color = if !*active {
                if e.construction {
                    rgba(200, 140, 70, 200)
                } else if e.projected {
                    rgba(150, 90, 190, 200)
                } else {
                    colors::inactive()
                }
            } else if e.construction {
                colors::construction()
            } else if e.projected {
                colors::projected()
            } else if inp.constrained.get(&eid).copied().unwrap_or(false) {
                colors::fixed()
            } else {
                colors::free()
            };
            let style = if e.construction { 1 } else { 0 };
            let width = if *active { 2.2 } else { 1.4 };
            let mut seg = |a: [f32; 3], b: [f32; 3]| {
                sc.frame.lines.push(LineInst {
                    p0: a,
                    p1: b,
                    color,
                    width,
                    id,
                    style,
                })
            };
            match e.geom {
                Geom::Line { a, b } => seg(w(sk.pos(a)), w(sk.pos(b))),
                Geom::Circle { c, r } => {
                    let (c, r) = (sk.pos(c), sk.x[r]);
                    let n = 96;
                    for i in 0..n {
                        let t0 = std::f64::consts::TAU * i as f64 / n as f64;
                        let t1 = std::f64::consts::TAU * (i + 1) as f64 / n as f64;
                        seg(
                            w([c[0] + r * t0.cos(), c[1] + r * t0.sin()]),
                            w([c[0] + r * t1.cos(), c[1] + r * t1.sin()]),
                        );
                    }
                }
                Geom::Arc { c, s, e: end } => {
                    let (c, s, end) = (sk.pos(c), sk.pos(s), sk.pos(end));
                    let r = (s[0] - c[0]).hypot(s[1] - c[1]);
                    let t0 = (s[1] - c[1]).atan2(s[0] - c[0]);
                    let mut sweep = ((end[1] - c[1]).atan2(end[0] - c[0]) - t0)
                        .rem_euclid(std::f64::consts::TAU);
                    if sweep < 1e-9 {
                        sweep = std::f64::consts::TAU;
                    }
                    let n = ((sweep / std::f64::consts::TAU * 96.0).ceil() as usize).max(4);
                    for i in 0..n {
                        let a0 = t0 + sweep * i as f64 / n as f64;
                        let a1 = t0 + sweep * (i + 1) as f64 / n as f64;
                        seg(
                            w([c[0] + r * a0.cos(), c[1] + r * a0.sin()]),
                            w([c[0] + r * a1.cos(), c[1] + r * a1.sin()]),
                        );
                    }
                }
                Geom::Point(_) => {}
            }
        }
    }

    // ---- tool previews (always on top)
    for (a, b, c, dashed) in &inp.preview_lines {
        sc.frame.overlay_lines.push(LineInst {
            p0: v3(*a),
            p1: v3(*b),
            color: *c,
            width: 2.0,
            id: 0,
            style: if *dashed { 3 } else { 2 },
        });
    }
    for (pl, c) in &inp.overlay_polylines {
        for w in pl.windows(2) {
            sc.frame.overlay_lines.push(LineInst {
                p0: w[0],
                p1: w[1],
                color: *c,
                width: 3.5,
                id: 0,
                style: 2,
            });
        }
    }
    for p in &inp.preview_points {
        sc.frame.points.push(PointInst {
            pos: v3(*p),
            color: colors::preview(),
            size: 8.0,
            id: 0,
            shape: 2 | 4,
        });
    }

    // ---- highlight flags
    let mut flags = vec![0u32; sc.picks.len() + 1];
    if inp.hover.is_some() || !inp.selection.is_empty() || !inp.emphasis.is_empty() {
        let selected: HashSet<_> = inp.selection.iter().chain(inp.emphasis).copied().collect();
        for (i, p) in sc.picks.iter().enumerate() {
            if inp.hover == Some(*p) {
                flags[i + 1] |= 1;
            }
            if selected.contains(p) {
                flags[i + 1] |= 2;
            }
        }
    }
    sc.frame.flags = flags;
    sc
}

/// Choose the best pick from a window of ids around the cursor: lowest
/// priority class first (points beat curves beat edges …), then nearest.
pub fn resolve_pick(ids: &[u32], win: u32, picks: &[Pick], prio: &[u8]) -> Option<Pick> {
    let c = (win / 2) as i32;
    let mut best: Option<(u8, i32, usize)> = None;
    for (k, &id) in ids.iter().enumerate() {
        if id == 0 || id as usize > picks.len() {
            continue;
        }
        let i = id as usize - 1;
        let pr = prio[i];
        if pr == 255 {
            continue;
        }
        let (x, y) = ((k as u32 % win) as i32 - c, (k as u32 / win) as i32 - c);
        let d = x * x + y * y;
        // Faces/planes/profiles only count directly under the cursor.
        if pr >= 3 && d > 2 {
            continue;
        }
        if best.is_none_or(|(bp, bd, _)| (pr, d) < (bp, bd)) {
            best = Some((pr, d, i));
        }
    }
    best.map(|(_, _, i)| picks[i])
}

pub fn triangulate_profile(p: &Profile) -> ProfileMesh {
    let mut data: Vec<f64> = Vec::new();
    let mut holes: Vec<usize> = Vec::new();
    let mut pts2: Vec<[f64; 2]> = Vec::new();
    let push_loop = |poly: Vec<[f64; 2]>, data: &mut Vec<f64>, pts2: &mut Vec<[f64; 2]>| {
        for q in poly {
            data.push(q[0]);
            data.push(q[1]);
            pts2.push(q);
        }
    };
    push_loop(p.outer.polygon(0.05), &mut data, &mut pts2);
    for h in &p.holes {
        holes.push(pts2.len());
        push_loop(h.polygon(0.05), &mut data, &mut pts2);
    }
    let indices = earcutr::earcut(&data, &holes, 2)
        .unwrap_or_default()
        .into_iter()
        .map(|i| i as u32)
        .collect();
    ProfileMesh {
        points: pts2,
        indices,
    }
}

#[cfg(test)]
mod profile_cache_tests {
    use super::*;
    #[test]
    fn triangulation_is_reused_until_exact_profile_geometry_changes() {
        let mut sk = Sketch::new();
        sk.add_rectangle([0., 0.], [10., 10.]);
        let profiles = refusion_model::sketch::find_profiles(&sk);
        let mut cache = ProfileCache::default();
        assert_eq!(cache.meshes(1, &profiles)[0].indices.len(), 6);
        let first = cache.entries[&1].1.clone();
        cache.meshes(1, &profiles.clone());
        assert!(Arc::ptr_eq(&first, &cache.entries[&1].1));
        sk.x[0] = -2.;
        cache.meshes(1, &refusion_model::sketch::find_profiles(&sk));
        assert!(!Arc::ptr_eq(&first, &cache.entries[&1].1));
    }
}

pub fn grid_step(extent: f64) -> f64 {
    // Largest 1-2-5 step giving at least ~25 lines across the extent.
    let target = (extent / 25.0).max(1e-3);
    let p = 10f64.powf(target.log10().floor());
    let step = [5.0, 2.0, 1.0]
        .into_iter()
        .map(|m| m * p)
        .find(|&s| s <= target)
        .unwrap_or(p);
    step.max(1e-3)
}

/// Major grid lines every power of ten above the step.
pub fn grid_major(step: f64) -> f64 {
    10f64.powf((step * 1.0001).log10().floor() + 1.0)
}

pub fn light_dir(right: DVec3, up: DVec3, forward: DVec3) -> Vec3 {
    (-forward * 0.55 + up * 0.65 - right * 0.35)
        .normalize()
        .as_vec3()
}

#[cfg(test)]
mod body_cache_tests {
    use super::*;
    #[test]
    fn cached_edge_bases_match_pick_table_with_vertices_and_hidden_bodies() {
        let mut doc = refusion_model::Document::new();
        doc.add_import(
            "box",
            &refusion_model::kernel::Shape::boxed(4., 5., 6.).unwrap(),
        )
        .unwrap();
        let b = &doc.state().bodies[0];
        let meshes = HashMap::from([(b.shape.uid(), Arc::new(b.shape.mesh(0.03, 0.2).unwrap()))]);
        let hidden = HashSet::new();
        let constrained = HashMap::new();
        let selected = [Pick::Edge {
            body: b.id,
            edge: 1,
        }];
        let mut inp = SceneInput {
            state: doc.state(),
            sketches: vec![],
            profile_sketches: vec![],
            hidden_bodies: &hidden,
            hidden_planes: &hidden,
            filter: PickFilter {
                faces: true,
                edges: true,
                vertices: true,
                ..Default::default()
            },
            hover: None,
            selection: &selected,
            emphasis: &[],
            show_planes: false,
            grid: None,
            grid_step: None,
            preview_lines: vec![],
            preview_points: vec![],
            overlay_polylines: vec![],
            constrained: &constrained,
            view_scale: 20.,
            show_points: true,
            show_construction: true,
            show_projected: true,
        };
        let scene = build(&inp, &meshes);
        assert!(scene.frame.lines.is_empty());
        let base = scene.frame.bodies[0].edge_base;
        assert_eq!(scene.picks[base as usize - 1], selected[0]);
        assert_eq!(scene.frame.flags[base as usize], 2);
        let mut body_hidden = HashSet::new();
        body_hidden.insert(b.id);
        inp.hidden_bodies = &body_hidden;
        assert!(build(&inp, &meshes).frame.bodies.is_empty());
    }
}

/// Sketch gestures resolve immediately from their already prepared screen geometry.
/// Only enabled sketch entities participate; body topology still uses GPU depth.
pub fn pick_sketch_screen(
    sc: &Scene,
    cursor: eframe::egui::Pos2,
    project: impl Fn([f32; 3]) -> Option<eframe::egui::Pos2>,
) -> Option<Pick> {
    let mut best: Option<(u8, f32, u32)> = None;
    let mut consider = |id: u32, d: f32| {
        if id == 0 || d > 7.5 {
            return;
        }
        let i = id as usize - 1;
        if !matches!(sc.picks.get(i), Some(Pick::Ent { .. })) || sc.prio[i] == 255 {
            return;
        }
        let key = (sc.prio[i], d, id);
        if best.is_none_or(|b| key < b) {
            best = Some(key);
        }
    };
    for point in &sc.frame.points {
        if let Some(p) = project(point.pos) {
            consider(point.id, p.distance(cursor));
        }
    }
    for line in &sc.frame.lines {
        if line.id == 0 {
            continue;
        }
        if let (Some(a), Some(b)) = (project(line.p0), project(line.p1)) {
            let ab = b - a;
            let t = if ab.length_sq() > 1e-12 {
                ((cursor - a).dot(ab) / ab.length_sq()).clamp(0., 1.)
            } else {
                0.
            };
            consider(line.id, (a + ab * t).distance(cursor));
        }
    }
    best.map(|(_, _, id)| sc.picks[id as usize - 1])
}
#[cfg(test)]
mod immediate_sketch_pick_tests {
    use super::*;
    use eframe::egui::Pos2;
    #[test]
    fn click_without_previous_hover_prioritizes_endpoints_and_ignores_disabled_and_body_picks() {
        let a = Pick::Ent {
            sketch: 3,
            ent: EntityId(1),
        };
        let b = Pick::Ent {
            sketch: 3,
            ent: EntityId(2),
        };
        let mut sc = Scene {
            frame: FrameData::default(),
            picks: vec![a, b, Pick::Vertex { body: 2, vertex: 1 }],
            prio: vec![0, 1, 0],
        };
        sc.frame.points.push(PointInst {
            pos: [10., 10., 0.],
            color: 0,
            size: 6.,
            id: 1,
            shape: 0,
        });
        sc.frame.points.push(PointInst {
            pos: [13., 10., 0.],
            color: 0,
            size: 6.,
            id: 3,
            shape: 0,
        });
        sc.frame.lines.push(LineInst {
            p0: [0., 10., 0.],
            p1: [30., 10., 0.],
            color: 0,
            width: 1.,
            id: 2,
            style: 0,
        });
        let project = |p: [f32; 3]| Some(Pos2::new(p[0], p[1]));
        assert_eq!(
            pick_sketch_screen(&sc, Pos2::new(13., 10.), project),
            Some(a)
        );
        sc.prio[0] = 255;
        assert_eq!(
            pick_sketch_screen(&sc, Pos2::new(13., 10.), project),
            Some(b)
        );
        assert_eq!(pick_sketch_screen(&sc, Pos2::new(70., 70.), project), None);
    }
}

#[cfg(test)]
mod palette_visibility_tests {
    use super::*;
    #[test]
    fn hidden_construction_and_projected_entities_are_neither_drawn_nor_pickable() {
        let mut sketch = Sketch::new();
        let a = sketch.add_point([0., 0.]);
        let b = sketch.add_point([10., 0.]);
        let construction = sketch.add_line(a, b);
        sketch.entities.get_mut(&construction).unwrap().construction = true;
        let center = sketch.add_point([20., 0.]);
        let projected = sketch.add_circle(center, 5.);
        sketch.entities.get_mut(&projected).unwrap().projected = true;
        let frame = Frame {
            origin: [0.; 3],
            x: [1., 0., 0.],
            y: [0., 1., 0.],
        };
        let state = State::default();
        let empty = HashSet::new();
        let constrained = HashMap::new();
        let meshes = HashMap::new();
        let mut input = SceneInput {
            state: &state,
            sketches: vec![(1, &sketch, frame, true)],
            profile_sketches: vec![],
            hidden_bodies: &empty,
            hidden_planes: &empty,
            filter: PickFilter {
                sketch_ents: true,
                ..Default::default()
            },
            hover: None,
            selection: &[],
            emphasis: &[],
            show_planes: false,
            grid: None,
            grid_step: None,
            preview_lines: vec![],
            preview_points: vec![],
            overlay_polylines: vec![],
            constrained: &constrained,
            view_scale: 20.,
            show_points: true,
            show_construction: false,
            show_projected: false,
        };
        let hidden = build(&input, &meshes);
        assert!(!hidden.picks.contains(&Pick::Ent {
            sketch: 1,
            ent: construction
        }));
        assert!(!hidden.picks.contains(&Pick::Ent {
            sketch: 1,
            ent: projected
        }));
        input.show_construction = true;
        input.show_projected = true;
        let shown = build(&input, &meshes);
        assert!(shown.picks.contains(&Pick::Ent {
            sketch: 1,
            ent: construction
        }));
        assert!(shown.picks.contains(&Pick::Ent {
            sketch: 1,
            ent: projected
        }));
        assert!(shown.frame.lines.len() > hidden.frame.lines.len());
    }
}

#[cfg(test)]
pub(crate) fn fixture_scene<'a>(
    doc: &refusion_model::Document,
    meshes: &'a HashMap<u64, Arc<Mesh>>,
) -> Scene<'a> {
    let hidden = HashSet::new();
    let constrained = HashMap::new();
    build(
        &SceneInput {
            state: doc.state(),
            sketches: doc
                .features
                .iter()
                .filter_map(|f| {
                    let refusion_model::FeatureKind::Sketch(s) = &f.kind else {
                        return None;
                    };
                    Some((
                        f.id,
                        &s.sketch,
                        doc.state().sketches.get(&f.id)?.frame,
                        false,
                    ))
                })
                .collect(),
            profile_sketches: vec![],
            hidden_bodies: &hidden,
            hidden_planes: &hidden,
            filter: PickFilter {
                faces: true,
                edges: true,
                vertices: true,
                planes: true,
                ..Default::default()
            },
            hover: None,
            selection: &[],
            emphasis: &[],
            show_planes: true,
            grid: None,
            grid_step: None,
            preview_lines: vec![],
            preview_points: vec![],
            overlay_polylines: vec![],
            constrained: &constrained,
            view_scale: 100.,
            show_points: true,
            show_construction: true,
            show_projected: true,
        },
        meshes,
    )
}

#[cfg(test)]
mod all_features_render_tests {
    use super::*;
    #[test]
    fn both_feature_fixtures_build_complete_render_and_pick_scenes() {
        for fixture in [
            include_str!("../../../examples/all-features.rfz"),
            include_str!("../../../examples/all-features-combined.rfz"),
        ] {
            let doc = refusion_model::Document::from_json(fixture).unwrap();
            assert!(doc.errors.is_empty());
            let meshes = doc
                .state()
                .bodies
                .iter()
                .map(|b| (b.shape.uid(), Arc::new(b.shape.mesh(0.15, 0.25).unwrap())))
                .collect();
            let scene = fixture_scene(&doc, &meshes);
            assert_eq!(scene.frame.bodies.len(), doc.state().bodies.len());
            assert_eq!(scene.prio.len(), scene.picks.len());
            for (body, draw) in doc.state().bodies.iter().zip(&scene.frame.bodies) {
                assert!(!draw.mesh.indices.is_empty());
                for face in 1..=body.shape.face_count() as u32 {
                    assert_eq!(
                        scene.picks[(draw.base + face - 2) as usize],
                        Pick::Face {
                            body: body.id,
                            face
                        }
                    );
                }
                for edge in 1..=body.shape.edge_count() as u32 {
                    assert_eq!(
                        scene.picks[(draw.edge_base + edge - 2) as usize],
                        Pick::Edge {
                            body: body.id,
                            edge
                        }
                    );
                }
            }
            assert!(!scene.frame.lines.is_empty());
            assert!(!scene.frame.flat_i.is_empty());
            assert!(scene.frame.lines.iter().all(|l| l
                .p0
                .iter()
                .chain(&l.p1)
                .all(|v| v.is_finite())));
            assert!(scene
                .frame
                .points
                .iter()
                .all(|p| p.pos.iter().all(|v| v.is_finite())));
        }
    }
}
