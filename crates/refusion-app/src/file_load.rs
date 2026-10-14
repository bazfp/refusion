//! Prepare a complete file transaction off the UI thread. A cancelled result
//! is discarded; OpenCASCADE calls already in progress cannot be interrupted.
use refusion_model::{kernel, Document};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
};

pub struct Prepared {
    pub doc: Document,
    pub meshes: HashMap<u64, Arc<kernel::Mesh>>,
    pub native: bool,
    pub notes: Vec<String>,
}

pub struct Job {
    pub path: PathBuf,
    pub insert: bool,
    pub cancelled: Arc<AtomicBool>,
    pub result: mpsc::Receiver<Result<Prepared, String>>,
}

impl Job {
    pub fn start(
        path: PathBuf,
        base: Option<Document>,
        meshes: HashMap<u64, Arc<kernel::Mesh>>,
        ctx: eframe::egui::Context,
    ) -> Self {
        let (sender, result) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel = cancelled.clone();
        let source = path.clone();
        let insert = base.is_some();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepare(&source, base, meshes, &cancel)
            }))
            .unwrap_or_else(|_| Err("File loading worker failed".into()));
            let _ = sender.send(result);
            ctx.request_repaint();
        });
        Self {
            path,
            insert,
            cancelled,
            result,
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

fn prepare(
    path: &Path,
    base: Option<Document>,
    mut meshes: HashMap<u64, Arc<kernel::Mesh>>,
    cancel: &AtomicBool,
) -> Result<Prepared, String> {
    let check = || {
        if cancel.load(Ordering::Relaxed) {
            Err("Cancelled".to_string())
        } else {
            Ok(())
        }
    };
    check()?;
    let mut notes = Vec::new();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let native = ext == "rfz";
    // Existing bodies are shared read-only with the displayed document. Never
    // mesh those on the worker: OCCT meshing mutates the shape's triangulation.
    let existing: HashSet<_> = base.iter().flat_map(|d| d.cached_shape_uids()).collect();
    let mut doc = if native {
        Document::from_json(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)?
    } else {
        base.unwrap_or_else(Document::new)
    };
    if !native {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut sketches = Vec::new();
        let parts = match ext.as_str() {
            "step" | "stp" => vec![(name, kernel::read_step(path).map_err(|e| e.0)?)],
            "iges" | "igs" => vec![(name, kernel::read_iges(path).map_err(|e| e.0)?)],
            "f3d" => {
                let report = refusion_model::f3d_design_import(path)?;
                notes.extend(report.warnings.iter().cloned());
                if report.stats.approx_edges > 0
                    || report.stats.filled_faces > 0
                    || report.stats.failed_faces > 0
                {
                    notes.push(format!(
                        "F3D body import used {} approximated edge(s) and {} filled face(s); {} face(s) could not be rebuilt. Imported geometry may differ from the source.",
                        report.stats.approx_edges, report.stats.filled_faces, report.stats.failed_faces
                    ));
                }
                if !report.sketches.is_empty() {
                    let omitted: usize = report.sketches.iter().map(|s| s.unsupported).sum();
                    notes.push(format!("Imported {} sketch(es) as editable geometry. F3D constraints, dimensions and construction flags are not restored; {} other entities omitted.", report.sketches.len(), omitted));
                }
                if report.sketches_skipped > 0 {
                    notes.push(format!(
                        "{} source sketch(es) use unsupported geometry or stream layouts and were not imported.",
                        report.sketches_skipped
                    ));
                }
                sketches = report.sketches;
                report.parts
            }
            _ => return Err(format!("Unsupported file type .{ext}")),
        };
        if parts.is_empty() && sketches.is_empty() {
            return Err("File contains no supported bodies or sketches".into());
        }
        for (name, shape) in parts {
            check()?;
            doc.add_import(&name, &shape)?;
        }
        check()?;
        refusion_model::add_f3d_sketches(&mut doc, sketches);
    }
    for shape in doc.cached_shapes() {
        check()?;
        let uid = shape.uid();
        if !existing.contains(&uid) {
            meshes.insert(
                uid,
                Arc::new(shape.mesh_adaptive_display().map_err(|e| e.0)?),
            );
        }
    }
    check()?;
    Ok(Prepared {
        doc,
        meshes,
        native,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_insert_keeps_the_original_document() {
        let original = Document::new();
        let before = original.to_json();
        let result = prepare(
            Path::new("/missing.step"),
            Some(original.clone()),
            HashMap::new(),
            &AtomicBool::new(false),
        );
        assert!(result.is_err());
        assert_eq!(original.to_json(), before);
    }

    #[test]
    fn cancelled_job_does_not_read_or_change_a_document() {
        let result = prepare(
            Path::new("/missing.step"),
            Some(Document::new()),
            HashMap::new(),
            &AtomicBool::new(true),
        );
        assert!(matches!(result, Err(e) if e == "Cancelled"));
    }

    #[test]
    fn native_file_is_replayed_and_meshed_before_delivery() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/bracket.rfz");
        let prepared = prepare(&path, None, HashMap::new(), &AtomicBool::new(false)).unwrap();
        assert!(prepared.native);
        assert!(prepared.doc.errors.is_empty(), "{:?}", prepared.doc.errors);
        assert!(!prepared.doc.state().bodies.is_empty());
        for body in &prepared.doc.state().bodies {
            assert!(!prepared.meshes[&body.shape.uid()].positions.is_empty());
        }
        // The worker result is a replayable document, not a mesh-only import.
        let loaded = Document::from_json(&prepared.doc.to_json()).unwrap();
        assert!(loaded.errors.is_empty());
        assert_eq!(
            loaded.state().bodies.len(),
            prepared.doc.state().bodies.len()
        );
    }

    #[test]
    fn insert_prepares_one_transaction_without_mutating_the_displayed_design() {
        let original = Document::from_json(include_str!("../../../examples/bracket.rfz")).unwrap();
        let before = original.to_json();
        let path = std::env::temp_dir().join(format!(
            "refusion-worker-insert-{}.step",
            std::process::id()
        ));
        kernel::write_step(&[original.state().bodies[0].shape.clone()], &path).unwrap();
        let prepared = prepare(
            &path,
            Some(original.clone()),
            HashMap::new(),
            &AtomicBool::new(false),
        )
        .unwrap();
        let _ = std::fs::remove_file(path);
        assert!(!prepared.native);
        assert_eq!(original.to_json(), before);
        assert_eq!(prepared.doc.features.len(), original.features.len() + 1);
        assert_eq!(
            prepared.doc.state().bodies.len(),
            original.state().bodies.len() + 1
        );
        assert!(prepared.doc.errors.is_empty());
        // Existing shapes were not remeshed on the worker.
        assert!(!prepared
            .meshes
            .contains_key(&original.state().bodies[0].shape.uid()));
        let imported = prepared.doc.state().bodies.last().unwrap();
        assert!(!prepared.meshes[&imported.shape.uid()].positions.is_empty());
    }

    #[test]
    fn worker_reports_failure_without_waiting_on_the_ui_thread() {
        let job = Job::start(
            PathBuf::from("/missing.step"),
            None,
            HashMap::new(),
            eframe::egui::Context::default(),
        );
        assert!(job
            .result
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .is_err());
    }
}
