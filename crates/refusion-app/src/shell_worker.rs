//! Shell offsets run in a separate process: a native fault must not terminate
//! the desktop or corrupt the document and shapes shared by other workers.
use refusion_model::kernel::{KernelError, Result, Shape};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub const ARG: &str = "--internal-shell-worker";
const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Serialize, Deserialize)]
struct Request {
    brep: String,
    tags: Vec<u64>,
    faces: Vec<u32>,
    thickness: f64,
    feature: u64,
    #[serde(default)]
    round_join: bool,
}

#[derive(Serialize, Deserialize)]
struct Response {
    brep: String,
    tags: Vec<u64>,
}

pub fn shell(
    shape: &Shape,
    faces: &[u32],
    thickness: f64,
    feature: u64,
    round_join: bool,
) -> Result<Shape> {
    let executable = std::env::current_exe().map_err(error)?;
    shell_at(&executable, shape, faces, thickness, feature, round_join)
}

pub(crate) fn shell_at(
    executable: &Path,
    shape: &Shape,
    faces: &[u32],
    thickness: f64,
    feature: u64,
    round_join: bool,
) -> Result<Shape> {
    let mut request = Request {
        brep: shape.to_brep()?,
        tags: shape.face_tags(),
        faces: faces.to_vec(),
        thickness,
        feature,
        round_join,
    };
    let response = match run(executable, &request, TIMEOUT) {
        Ok(response) => response,
        Err(original) if !round_join => {
            // Retry in a fresh process, never in the faulted native address
            // space. Some chamfer intersections cannot use the sharp join path.
            request.round_join = true;
            run(executable, &request, TIMEOUT).map_err(|fallback| {
                error(format!(
                    "{original}; alternate offset also failed: {fallback}"
                ))
            })?
        }
        Err(e) => return Err(e),
    };
    let result = Shape::from_brep(&response.brep)?;
    if response.tags.len() != result.face_count() {
        return Err(error("Shell worker returned inconsistent face names"));
    }
    result.with_face_tags(&response.tags)
}

fn error(message: impl std::fmt::Display) -> KernelError {
    KernelError(format!("Shell: {message}"))
}

fn run(executable: &Path, request: &Request, timeout: Duration) -> Result<Response> {
    // Files avoid pipe deadlocks and bound the wait even if the native worker
    // fails before reading its input or hangs while producing its output.
    let temp = tempfile::tempdir().map_err(error)?;
    let input = temp.path().join("request.json");
    let output = temp.path().join("response.json");
    std::fs::write(&input, serde_json::to_vec(request).map_err(error)?).map_err(error)?;
    let mut child = Command::new(executable)
        .arg(ARG)
        .arg(&input)
        .arg(&output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(error)?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(error(format!(
                        "geometry worker stopped ({status}). Try a smaller thickness or different faces; your design is unchanged."
                    )));
                }
                break;
            }
            Ok(None) if start.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(10));
            }
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error(match result {
                    Err(e) => format!("could not monitor geometry worker: {e}"),
                    _ => "geometry operation timed out; your design is unchanged".into(),
                }));
            }
        }
    }
    let bytes = std::fs::read(output).map_err(error)?;
    let response: std::result::Result<Response, String> =
        serde_json::from_slice(&bytes).map_err(error)?;
    response.map_err(error)
}

fn compute(request: Request) -> Result<Response> {
    let shape = Shape::from_brep(&request.brep)?.with_face_tags(&request.tags)?;
    let result = shape.shell_with_join(
        &request.faces,
        request.thickness,
        request.feature,
        request.round_join,
    )?;
    if !result.is_valid() || !result.is_solid() || result.solids().len() != 1 {
        return Err(error("offset did not produce one valid solid"));
    }
    Ok(Response {
        brep: result.to_brep()?,
        tags: result.face_tags(),
    })
}

/// Dispatched before graphics or the parent-process executor is initialized.
pub fn entry() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        std::process::exit(2);
    }
    let result = (|| -> std::result::Result<Response, String> {
        let bytes = std::fs::read(&args[2]).map_err(|e| e.to_string())?;
        let request = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        compute(request).map_err(|e| e.to_string())
    })();
    let written = serde_json::to_vec(&result)
        .ok()
        .is_some_and(|bytes| std::fs::write(&args[3], bytes).is_ok());
    if !written {
        std::process::exit(2);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn request() -> Request {
        Request {
            brep: String::new(),
            tags: vec![],
            faces: vec![],
            thickness: -1.,
            feature: 7,
            round_join: false,
        }
    }

    fn script(text: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("worker");
        std::fs::write(&path, format!("#!/bin/sh\n{text}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        (temp, path)
    }

    #[test]
    fn native_segfault_returns_error_and_parent_can_run_another_worker() {
        let (_temp, crashed) = script("ulimit -c 0\nkill -SEGV $$");
        let failure = run(&crashed, &request(), Duration::from_secs(2))
            .err()
            .unwrap();
        assert!(failure.to_string().contains("signal: 11"), "{failure}");
        let (_temp, healthy) = script("printf '%s' '{\"Err\":\"invalid thickness\"}' > \"$3\"");
        let failure = run(&healthy, &request(), Duration::from_secs(2))
            .err()
            .unwrap();
        assert!(
            failure.to_string().contains("invalid thickness"),
            "{failure}"
        );
    }

    #[test]
    fn hung_native_worker_is_killed_and_reaped() {
        let (_temp, worker) = script("exec sleep 30");
        let start = Instant::now();
        let failure = run(&worker, &request(), Duration::from_millis(50))
            .err()
            .unwrap();
        assert!(failure.to_string().contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
