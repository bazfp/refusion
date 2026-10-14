pub mod acis;
pub mod sab;
pub mod sketches;
mod stream;

use refusion_kernel::Shape;
use std::io::Read;

#[derive(Debug, Default)]
pub struct Report {
    pub parts: Vec<(String, Shape)>,
    pub sketches: Vec<sketches::ImportedSketch>,
    pub sketches_skipped: usize,
    pub stats: acis::Stats,
    pub blobs: usize,
    pub sheets_skipped: usize,
    pub duplicates: usize,
    /// Recoverable omissions or unsupported sketch streams; bodies can still load.
    pub warnings: Vec<String>,
}

/// Read all BREP blobs of the active design.
pub fn read_blobs(path: &std::path::Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("not a F3D archive: {e}"))?;
    let mut out = Vec::new();
    for i in 0..z.len() {
        let mut e = z.by_index(i).map_err(|e| e.to_string())?;
        let name = e.name().to_string();
        let active = name
            .split_once('/')
            .is_some_and(|(asset, _)| asset.ends_with("[Active]"))
            && name.contains("Breps.BlobParts/");
        if active && (name.ends_with(".smb") || name.ends_with(".smbh")) {
            let mut v = Vec::new();
            e.read_to_end(&mut v).map_err(|e| format!("{name}: {e}"))?;
            out.push((name, v));
        }
    }
    // .smb before .smbh so the history-free copy wins de-duplication
    out.sort_by_key(|(n, _)| (n.ends_with(".smbh"), n.clone()));
    Ok(out)
}

fn signature(s: &Shape) -> (usize, i64, [i64; 6]) {
    let q = |v: f64| (v * 1000.0).round() as i64;
    let (lo, hi) = s.bbox().unwrap_or(([0.0; 3], [0.0; 3]));
    (
        s.face_count(),
        q(s.volume().abs()),
        [q(lo[0]), q(lo[1]), q(lo[2]), q(hi[0]), q(hi[1]), q(hi[2])],
    )
}

/// Bounding/volume signatures only select candidates. Exact BRep equality
/// proves duplication; ambiguous reconstructions are retained conservatively.
#[derive(Default)]
struct Duplicates(std::collections::HashMap<(usize, i64, [i64; 6]), Vec<Shape>>);
impl Duplicates {
    fn insert(&mut self, shape: &Shape) -> bool {
        let candidates = self.0.entry(signature(shape)).or_default();
        if !candidates.is_empty() {
            if let Ok(brep) = shape.to_brep() {
                if candidates
                    .iter()
                    .any(|s| s.to_brep().is_ok_and(|other| other == brep))
                {
                    return false;
                }
            }
        }
        candidates.push(shape.clone());
        true
    }
}

pub fn decode_smb(data: &[u8], stats: &mut acis::Stats) -> Result<Vec<Shape>, String> {
    let sab = sab::parse(data).map_err(|e| e.0)?;
    let mut d = acis::Decoder::new(&sab);
    let bodies = d.bodies();
    let s = &d.stats;
    stats.bodies += s.bodies;
    stats.faces += s.faces;
    stats.failed_faces += s.failed_faces;
    stats.filled_faces += s.filled_faces;
    stats.approx_edges += s.approx_edges;
    Ok(bodies)
}

pub fn import(path: &std::path::Path) -> Result<Report, String> {
    let blobs = read_blobs(path)?;
    let (sketches, sketches_skipped, warnings) = read_sketches_report(path)?;
    if blobs.is_empty() && sketches.is_empty() {
        return Err("the archive contains no supported bodies or sketch geometry".into());
    }
    let stem = path
        .file_stem()
        .map(|s| {
            s.to_string_lossy()
                .trim_start_matches('_')
                .split('.')
                .next()
                .unwrap_or("F3D")
                .to_string()
        })
        .unwrap_or("F3D".into());
    let mut rep = Report {
        sketches,
        sketches_skipped,
        warnings,
        blobs: blobs.len(),
        ..Default::default()
    };
    let mut seen = Duplicates::default();
    let mut sheets = Vec::new();
    for (blob_name, data) in &blobs {
        let shapes = decode_smb(data, &mut rep.stats)?;
        for s in shapes {
            if !s.is_solid() {
                rep.sheets_skipped += 1;
                sheets.push(s);
                continue;
            }
            if !s.is_valid() {
                rep.warnings.push(format!(
                    "Skipped invalid solid reconstructed from {blob_name}"
                ));
                continue;
            }
            if !seen.insert(&s) {
                rep.duplicates += 1;
                continue;
            }
            let n = rep.parts.len() + 1;
            rep.parts.push((format!("{stem} {n}"), s));
        }
    }
    if rep.parts.is_empty() {
        // A surface-only design: bring the sheets in rather than nothing.
        for s in sheets {
            if seen.insert(&s) {
                let n = rep.parts.len() + 1;
                rep.parts.push((format!("{stem} surface {n}"), s));
            }
        }
    }
    if rep.parts.len() == 1 {
        rep.parts[0].0 = stem;
    }
    Ok(rep)
}

#[cfg(test)]
mod duplicate_tests {
    use super::*;
    #[test]
    fn equal_bounds_and_volume_do_not_prove_geometry_identity() {
        let cube = Shape::boxed(10., 10., 10.).unwrap();
        let recess = |offset| {
            cube.boolean(
                &Shape::boxed(2., 2., 2.)
                    .unwrap()
                    .translated(offset, 0)
                    .unwrap(),
                refusion_kernel::BoolOp::Cut,
            )
            .unwrap()
        };
        let first = recess([2., 2., 9.]);
        let second = recess([6., 6., 9.]);
        assert_eq!(signature(&first), signature(&second));
        let mut seen = Duplicates::default();
        assert!(seen.insert(&first));
        assert!(seen.insert(&second));
        assert!(!seen.insert(&Shape::from_brep(&first.to_brep().unwrap()).unwrap()));
    }
}

/// Read original sketch geometry from each active design stream. The paired
/// metadata identifies classes independently of version-specific numeric IDs.
pub fn read_sketches(path: &std::path::Path) -> Result<Vec<sketches::ImportedSketch>, String> {
    read_sketches_report(path).map(|r| r.0)
}
fn read_sketches_report(
    path: &std::path::Path,
) -> Result<(Vec<sketches::ImportedSketch>, usize, Vec<String>), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let names: Vec<_> = archive
        .file_names()
        .filter(|n| {
            n.split_once('/')
                .is_some_and(|(asset, _)| asset.ends_with("[Active]"))
                && n.ends_with("/MetaStream.dat")
        })
        .map(str::to_owned)
        .collect();
    let mut result = Vec::new();
    let mut skipped = 0;
    let mut warnings = Vec::new();
    for name in names {
        let bulk_name = name.replace("MetaStream.dat", "BulkStream.dat");
        let mut meta = Vec::new();
        archive
            .by_name(&name)
            .map_err(|e| e.to_string())?
            .read_to_end(&mut meta)
            .map_err(|e| e.to_string())?;
        let mut bulk = Vec::new();
        if let Ok(mut entry) = archive.by_name(&bulk_name) {
            entry.read_to_end(&mut bulk).map_err(|e| e.to_string())?;
        }
        match sketches::decode_indexed(&meta, &bulk) {
            Ok((decoded, omitted)) => {
                skipped += omitted;
                result.extend(decoded);
            }
            Err(e) => {
                skipped += sketches::declared_count(&meta);
                warnings.push(format!("Skipped sketch stream {name}: {e}"));
            }
        }
    }
    for (i, sketch) in result.iter_mut().enumerate() {
        sketch.name = format!("F3D Sketch {}", i + 1);
    }
    Ok((result, skipped, warnings))
}
