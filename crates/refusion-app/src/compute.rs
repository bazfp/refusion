//! A single native modelling worker; pending previews coalesce and stale results
//! never become visible. Meshing happens on isolated geometry in the worker.
use refusion_model::kernel::Mesh;
use refusion_model::{Document, FeatureId};
use std::{
    collections::HashMap,
    sync::{mpsc, Arc},
    time::Instant,
};
pub type Edit = Box<dyn FnOnce(&mut Document) -> Option<FeatureId> + Send>;
pub struct Request {
    pub ticket: u64,
    pub replay: bool,
    pub doc: Document,
    pub cached: HashMap<u64, Arc<Mesh>>,
    pub edit: Edit,
}
pub struct Result {
    pub ticket: u64,
    pub replay: bool,
    pub doc: Document,
    pub meshes: HashMap<u64, Arc<Mesh>>,
    pub error: Option<String>,
    pub ms: f64,
}
pub struct Failure {
    pub ticket: u64,
    pub replay: bool,
    pub error: String,
}
pub struct Worker {
    tx: mpsc::Sender<Request>,
    rx: mpsc::Receiver<std::result::Result<Result, Failure>>,
}
impl Worker {
    pub fn new(ctx: eframe::egui::Context) -> Self {
        let (tx, input) = mpsc::channel::<Request>();
        let (output, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("refusion-model".into())
            .spawn(move || {
                while let Ok(mut request) = input.recv() {
                    // Only previews can be superseded. Timeline edits must be applied.
                    if !request.replay {
                        let deadline = Instant::now() + std::time::Duration::from_millis(80);
                        while let Some(remaining) = deadline.checked_duration_since(Instant::now())
                        {
                            let Ok(next) = input.recv_timeout(remaining) else {
                                break;
                            };
                            request = next;
                            if request.replay {
                                break;
                            }
                        }
                    }
                    let ticket = request.ticket;
                    let replay = request.replay;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let start = Instant::now();
                        let id = (request.edit)(&mut request.doc);
                        let mut error = request
                            .doc
                            .parameter_error
                            .clone()
                            .or_else(|| id.and_then(|id| request.doc.errors.get(&id).cloned()));
                        if request.replay {
                            error = request.doc.parameter_error.clone().or_else(|| {
                                request
                                    .doc
                                    .features
                                    .iter()
                                    .find_map(|f| request.doc.errors.get(&f.id).cloned())
                            });
                        }
                        let mut meshes = request.cached;
                        for shape in request.doc.cached_shapes() {
                            if !meshes.contains_key(&shape.uid()) {
                                match shape.mesh_adaptive_display() {
                                    Ok(mesh) => {
                                        meshes.insert(shape.uid(), Arc::new(mesh));
                                    }
                                    Err(e) => error = Some(format!("Meshing: {e}")),
                                }
                            }
                        }
                        Result {
                            ticket: request.ticket,
                            replay: request.replay,
                            doc: request.doc,
                            meshes,
                            error,
                            ms: start.elapsed().as_secs_f64() * 1000.0,
                        }
                    }))
                    .map_err(|_| Failure {
                        ticket,
                        replay,
                        error: "Modelling worker panicked".into(),
                    });
                    if output.send(result).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("modelling worker");
        Self { tx, rx }
    }
    pub fn submit(&self, request: Request) {
        let _ = self.tx.send(request);
    }
    pub fn poll(&self) -> Option<std::result::Result<Result, Failure>> {
        self.rx.try_recv().ok()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn request(ticket: u64, replay: bool, edit: Edit) -> Request {
        Request {
            ticket,
            replay,
            doc: Document::new(),
            cached: HashMap::new(),
            edit,
        }
    }
    fn result(worker: &Worker) -> std::result::Result<Result, Failure> {
        worker
            .rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker response")
    }
    #[test]
    fn submitting_does_not_wait_for_native_work_and_pending_previews_coalesce() {
        let worker = Worker::new(eframe::egui::Context::default());
        let (entered, wait) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        worker.submit(request(
            1,
            true,
            Box::new(move |_| {
                entered.send(()).unwrap();
                gate.recv().unwrap();
                None
            }),
        ));
        wait.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.submit(request(
            2,
            false,
            Box::new(|_| panic!("superseded preview must never run")),
        ));
        worker.submit(request(
            3,
            false,
            Box::new(|d| {
                d.add_import(
                    "cube",
                    &refusion_model::kernel::Shape::boxed(10.0, 10.0, 10.0).unwrap(),
                )
                .unwrap();
                None
            }),
        ));
        release.send(()).unwrap();
        let first = result(&worker).ok().unwrap();
        assert!(first.replay);
        let newest = result(&worker).ok().unwrap();
        assert_eq!(newest.ticket, 3);
        assert!(!newest.replay);
        assert_eq!(newest.doc.state().bodies.len(), 1);
        for shape in newest.doc.cached_shapes() {
            assert!(!newest.meshes[&shape.uid()].indices.is_empty());
        }
    }
    #[test]
    fn panic_preserves_its_ticket_and_worker_recovers() {
        let worker = Worker::new(eframe::egui::Context::default());
        worker.submit(request(42, true, Box::new(|_| panic!("test panic"))));
        let failure = result(&worker).err().unwrap();
        assert_eq!(failure.ticket, 42);
        assert!(failure.replay);
        worker.submit(request(43, true, Box::new(|_| None)));
        assert_eq!(result(&worker).ok().unwrap().ticket, 43);
    }
    #[test]
    fn meshing_preserves_shared_shapes_and_prepares_earlier_timeline_states() {
        let mut doc = Document::new();
        doc.add_import(
            "first",
            &refusion_model::kernel::Shape::boxed(2.0, 3.0, 4.0).unwrap(),
        )
        .unwrap();
        doc.add_import(
            "second",
            &refusion_model::kernel::Shape::boxed(5.0, 6.0, 7.0).unwrap(),
        )
        .unwrap();
        let original = doc.cached_shapes();
        let worker = Worker::new(eframe::egui::Context::default());
        worker.submit(Request {
            ticket: 1,
            replay: true,
            doc: doc.clone(),
            cached: HashMap::new(),
            edit: Box::new(|_| None),
        });
        let prepared = result(&worker).ok().unwrap();
        assert!(prepared.error.is_none());
        for shape in prepared.doc.cached_shapes() {
            assert!(original.iter().any(|s| s.uid() == shape.uid()));
            assert!(!prepared.meshes[&shape.uid()].positions.is_empty());
        }
        assert_eq!(
            doc.cached_shapes()
                .iter()
                .map(|s| s.uid())
                .collect::<Vec<_>>(),
            original.iter().map(|s| s.uid()).collect::<Vec<_>>()
        );
    }
}
