//! Geometry-derived sketch data is prepared once per geometry change on a
//! worker. A single latest-request slot bounds pending work during dragging.
use crate::latest_worker::LatestWorker;
use crate::scene::{triangulate_profile, ProfileMesh};
use refusion_model::sketch::{find_profiles, Profile, Sketch};
use std::sync::Arc;

struct Request {
    ticket: u64,
    sketch: Sketch,
}
pub struct Derived {
    pub ticket: u64,
    pub profiles: Arc<Vec<Profile>>,
    pub meshes: Arc<Vec<ProfileMesh>>,
}
pub struct Worker {
    worker: LatestWorker<Request, Derived>,
    requested: Option<Sketch>,
    ready: Option<Derived>,
    ticket: u64,
}
impl Worker {
    pub fn new(ctx: eframe::egui::Context) -> Self {
        let worker = LatestWorker::new("refusion-sketch-profiles", ctx, |request: Request| {
            let profiles: Vec<Profile> = find_profiles(&request.sketch);
            let meshes = profiles.iter().map(triangulate_profile).collect();
            Derived {
                ticket: request.ticket,
                profiles: Arc::new(profiles),
                meshes: Arc::new(meshes),
            }
        });
        Self {
            worker,
            requested: None,
            ready: None,
            ticket: 0,
        }
    }
    pub fn get(&mut self, sketch: &Sketch) -> Option<&Derived> {
        // Compare exact geometry, rather than a lossy hash or a revision that
        // public field edits could forget to increment. Labels/constraints do
        // not invalidate profiles when solved coordinates stay unchanged.
        if self
            .requested
            .as_ref()
            .is_none_or(|s| s.x != sketch.x || s.entities != sketch.entities)
        {
            self.ticket = self.ticket.wrapping_add(1);
            let snapshot = sketch.geometry_snapshot();
            self.requested = Some(snapshot.clone());
            self.ready = None;
            self.worker.submit(Request {
                ticket: self.ticket,
                sketch: snapshot,
            });
        }
        while let Some(result) = self.worker.poll() {
            if result.ticket == self.ticket {
                self.ready = Some(result);
            }
        }
        self.ready.as_ref()
    }
}
pub struct AnalysisData {
    pub ticket: u64,
    pub analysis: refusion_model::sketch::solver::Analysis,
    pub constrained: std::collections::HashMap<refusion_model::sketch::EntityId, bool>,
}
pub struct AnalysisWorker {
    worker: LatestWorker<Request, AnalysisData>,
    requested: Option<Sketch>,
    ready: Option<AnalysisData>,
    ticket: u64,
}
impl AnalysisWorker {
    pub fn new(ctx: eframe::egui::Context) -> Self {
        let worker = LatestWorker::new("refusion-sketch-analysis", ctx, |mut r: Request| {
            let analysis = r.sketch.analysis().clone();
            let ids: Vec<_> = r.sketch.entities.keys().copied().collect();
            let constrained = ids
                .into_iter()
                .map(|id| (id, r.sketch.is_fully_constrained(id)))
                .collect();
            AnalysisData {
                ticket: r.ticket,
                analysis,
                constrained,
            }
        });
        Self {
            worker,
            requested: None,
            ready: None,
            ticket: 0,
        }
    }
    pub fn get(&mut self, sketch: &Sketch) -> Option<&AnalysisData> {
        let changed = self.requested.as_ref().is_none_or(|s| {
            s.x != sketch.x
                || s.entities != sketch.entities
                || s.constraints.len() != sketch.constraints.len()
                || s.constraints
                    .iter()
                    .zip(&sketch.constraints)
                    .any(|((id, a), (other, b))| id != other || a.c != b.c || a.driven != b.driven)
        });
        if changed {
            self.ticket = self.ticket.wrapping_add(1);
            self.ready = None;
            let mut snapshot = sketch.clone();
            snapshot.analysis = None;
            self.requested = Some(snapshot.clone());
            self.worker.submit(Request {
                ticket: self.ticket,
                sketch: snapshot,
            });
        }
        while let Some(data) = self.worker.poll() {
            if data.ticket == self.ticket {
                self.ready = Some(data);
            }
        }
        self.ready.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wait(worker: &mut Worker, sketch: &Sketch) -> Arc<Vec<Profile>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(data) = worker.get(sketch) {
                return data.profiles.clone();
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
    #[test]
    fn cached_profiles_follow_exact_geometry_and_ignore_stale_work() {
        let mut sk = Sketch::new();
        sk.add_rectangle([0., 0.], [10., 10.]);
        let mut worker = Worker::new(eframe::egui::Context::default());
        let first = wait(&mut worker, &sk);
        assert_eq!(first.len(), 1);
        assert!(Arc::ptr_eq(&first, &wait(&mut worker, &sk)));
        sk.x[0] = -1.;
        assert!(worker.get(&sk).is_none());
        sk.x[0] = -2.;
        let latest = wait(&mut worker, &sk);
        assert!(!Arc::ptr_eq(&first, &latest));
        assert_eq!(*latest, find_profiles(&sk));
    }

    #[test]
    fn background_analysis_invalidates_for_constraints_without_geometry_changes() {
        let mut sk = Sketch::new();
        let point = sk.add_point([0., 0.]);
        for i in 1..65 {
            sk.add_point([i as f64, 0.]);
        }
        let mut worker = AnalysisWorker::new(eframe::egui::Context::default());
        let wait = |worker: &mut AnalysisWorker, sk: &Sketch| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if let Some(data) = worker.get(sk) {
                    return (data.ticket, data.analysis.dof);
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
        };
        let (first, dof) = wait(&mut worker, &sk);
        assert_eq!(dof, 130);
        sk.add_constraint(refusion_model::sketch::Constraint::Fix(point))
            .unwrap();
        let (next, dof) = wait(&mut worker, &sk);
        assert_ne!(first, next);
        assert_eq!(dof, 128);
    }
}
