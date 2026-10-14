//! View-aware display-only remeshing. The displayed B-rep is never mutated.
use refusion_model::kernel::{Mesh, Shape};
use std::{
    collections::HashMap,
    sync::{mpsc, Arc},
};
pub struct Request {
    pub uid: u64,
    pub shape: Shape,
    pub linear: f64,
    pub volume: Option<f64>,
}
pub struct Result {
    pub uid: u64,
    pub linear: f64,
    pub mesh: std::result::Result<Arc<Mesh>, String>,
}
pub struct Worker {
    tx: mpsc::Sender<Vec<Request>>,
    rx: mpsc::Receiver<Result>,
}
impl Worker {
    pub fn new(ctx: eframe::egui::Context) -> Self {
        let (tx, input) = mpsc::channel::<Vec<Request>>();
        let (output, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("refusion-display-mesh".into())
            .spawn(move || {
                while let Ok(mut requests) = input.recv() {
                    // Superseded requests are display-only: coalesce by source UID.
                    let mut latest: HashMap<_, _> =
                        requests.drain(..).map(|r| (r.uid, r)).collect();
                    while let Ok(next) = input.try_recv() {
                        for r in next {
                            latest.insert(r.uid, r);
                        }
                    }
                    for (_, r) in latest {
                        let mesh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut mesh =
                                r.shape.mesh_opts(r.linear, 0.2, true).map_err(|e| e.0)?;
                            mesh.volume = r.volume;
                            if std::env::var_os("REFUSION_PERF").is_some() {
                                eprintln!(
                                    "refusion-mesh-lod: uid={} deflection={} mm triangles={}",
                                    r.uid,
                                    r.linear,
                                    mesh.indices.len() / 3
                                );
                            }
                            Ok::<_, String>(Arc::new(mesh))
                        }))
                        .unwrap_or_else(|_| Err("Display meshing worker failed".into()));
                        if output
                            .send(Result {
                                uid: r.uid,
                                linear: r.linear,
                                mesh,
                            })
                            .is_err()
                        {
                            return;
                        }
                        ctx.request_repaint();
                    }
                }
            })
            .expect("display mesh worker");
        Self { tx, rx }
    }
    pub fn submit(&self, requests: Vec<Request>) {
        let _ = self.tx.send(requests);
    }
    pub fn poll(&self) -> Option<Result> {
        self.rx.try_recv().ok()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_uses_isolated_geometry_and_keeps_source_identity() {
        let shape = Shape::boxed(10., 20., 30.).unwrap();
        let before = shape.to_brep().unwrap();
        let uid = shape.uid();
        let w = Worker::new(eframe::egui::Context::default());
        w.submit(vec![Request {
            uid,
            shape: shape.clone(),
            linear: 0.005,
            volume: Some(6000.),
        }]);
        let result =
            w.rx.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        assert_eq!(result.uid, uid);
        let mesh = result.mesh.unwrap();
        assert_eq!(mesh.volume, Some(6000.));
        assert_eq!(mesh.edge_count(), 12);
        assert_eq!(mesh.linear_deflection, 0.005);
        assert_eq!(shape.to_brep().unwrap(), before);
    }
}
