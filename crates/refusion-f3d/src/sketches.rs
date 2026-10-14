use crate::stream::Metadata;
use refusion_kernel::Frame;
use std::collections::HashMap;
const SKETCH: &str = "44A64366-4BD3-4B24-881A-F94C206E8F2D";
const TRANSFORM: &str = "F47A46FB-DA27-4AFA-8A72-1B38FA596E23";
const POINT: &str = "C2CEDAE7-1716-47C1-B7B1-07B70081D0FB";
const CIRCLE: &str = "F0130424-8B7E-4092-93C9-1CA807482534";
const LINE: &str = "DCA267ED-D615-4934-B64F-AD805E8003E2";
#[derive(Debug)]
pub struct ImportedSketch {
    pub name: String,
    pub frame: Frame,
    pub curves: Vec<Curve>,
    pub points: Vec<[f64; 2]>,
    pub unsupported: usize,
}
#[derive(Debug, Clone)]
pub enum Curve {
    Line([f64; 2], [f64; 2]),
    Circle {
        centre: [f64; 2],
        radius: f64,
    },
    Arc {
        centre: [f64; 2],
        radius: f64,
        start: f64,
        sweep: f64,
    },
}
fn u32_at(b: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(b.get(at..at.checked_add(4)?)?.try_into().ok()?) as usize)
}
fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        b.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}
fn dbl(b: &[u8], at: usize) -> Option<f64> {
    let v = f64::from_le_bytes(b.get(at..at.checked_add(8)?)?.try_into().ok()?);
    v.is_finite().then_some(v)
}
fn millimetres(v: f64) -> Option<f64> {
    let value = v * 10.0;
    (value.is_finite() && value.abs() < 1e10).then_some(value)
}
fn ptr(b: &[u8], at: usize) -> Option<u64> {
    if b.get(at) != Some(&1) || b.get(at + 9..at + 11)? != [0, 0] {
        return None;
    }
    u64_at(b, at + 1)
}
fn string<'a>(b: &'a [u8], at: &mut usize) -> Option<&'a str> {
    let n = u32_at(b, *at)?;
    *at = at.checked_add(4)?;
    let value = std::str::from_utf8(b.get(*at..at.checked_add(n)?)?).ok()?;
    *at += n;
    Some(value)
}
fn attributes(b: &[u8]) -> Option<usize> {
    if b.get(..5)? != [0, 0, 0, 0, 0] {
        return None;
    }
    if b.get(5) == Some(&0) {
        return Some(10);
    }
    if b.get(5) != Some(&1) {
        return None;
    }
    let n = u32_at(b, 6)?;
    if n > 128 {
        return None;
    }
    let mut at = 10;
    for _ in 0..n {
        string(b, &mut at)?;
        if string(b, &mut at)? != "IntrinsicMetaTypeuint64" {
            return None;
        }
        u64_at(b, at)?;
        at += 8;
    }
    Some(at)
}
fn frame(b: &[u8]) -> Option<Frame> {
    if b.len() != 147 || b.get(..7)? != [0; 7] {
        return None;
    }
    let m: Vec<_> = (0..16).map(|i| dbl(b, 7 + i * 8)).collect::<Option<_>>()?;
    let fr = Frame {
        origin: [millimetres(m[3])?, millimetres(m[7])?, millimetres(m[11])?],
        x: [m[0], m[4], m[8]],
        y: [m[1], m[5], m[9]],
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    if (dot(fr.x, fr.x) - 1.).abs() > 1e-7
        || (dot(fr.y, fr.y) - 1.).abs() > 1e-7
        || dot(fr.x, fr.y).abs() > 1e-7
    {
        return None;
    }
    Some(fr)
}
fn point(b: &[u8]) -> Option<[f64; 2]> {
    let at = attributes(b)? + 17;
    let (x, y, z) = (dbl(b, at)?, dbl(b, at + 8)?, dbl(b, at + 16)?);
    if z.abs() > 1e-7 {
        return None;
    }
    Some([millimetres(x)?, millimetres(y)?])
}
fn round(b: &[u8]) -> Option<Curve> {
    let at = attributes(b)?;
    let (x, y, z) = (dbl(b, at)?, dbl(b, at + 8)?, dbl(b, at + 16)?);
    let (nx, ny, nz) = (dbl(b, at + 24)?, dbl(b, at + 32)?, dbl(b, at + 40)?);
    let (ux, uy, uz) = (dbl(b, at + 48)?, dbl(b, at + 56)?, dbl(b, at + 64)?);
    let (r, a, end) = (
        millimetres(dbl(b, at + 72)?)?,
        dbl(b, at + 80)?,
        dbl(b, at + 88)?,
    );
    if z.abs() > 1e-7
        || nx.abs() > 1e-7
        || ny.abs() > 1e-7
        || (nz.abs() - 1.).abs() > 1e-7
        || uz.abs() > 1e-7
        || (ux * ux + uy * uy - 1.).abs() > 1e-7
        || r <= 1e-9
        || end <= a
    {
        return None;
    }
    let sweep = end - a;
    if sweep > std::f64::consts::TAU + 1e-7 {
        return None;
    }
    if (sweep - std::f64::consts::TAU).abs() < 1e-7 {
        return Some(Curve::Circle {
            centre: [millimetres(x)?, millimetres(y)?],
            radius: r,
        });
    }
    let angle = uy.atan2(ux);
    let start = if nz > 0. { angle + a } else { angle - end };
    Some(Curve::Arc {
        centre: [millimetres(x)?, millimetres(y)?],
        radius: r,
        start,
        sweep,
    })
}
pub fn declared_count(meta: &[u8]) -> usize {
    Metadata::parse(meta).map_or(0, |m| m.classes().get(SKETCH).map_or(0, |ids| ids.len()))
}
pub fn decode(meta: &[u8], bulk: &[u8]) -> Vec<ImportedSketch> {
    decode_indexed(meta, bulk)
        .map(|(sketches, _)| sketches)
        .unwrap_or_default()
}

/// Parse indexed records, returning a diagnostic instead of guessing boundaries.
pub(crate) fn decode_indexed(
    meta: &[u8],
    bulk: &[u8],
) -> Result<(Vec<ImportedSketch>, usize), String> {
    let metadata = Metadata::parse(meta)?;
    let declared = metadata.classes().get(SKETCH).map_or(0, |ids| ids.len());
    let cls = metadata.classes();
    let rec = metadata.records(bulk)?;
    let is = |guid: &str, id: u64| cls.get(guid).is_some_and(|ids| ids.contains(&id));
    let mut ids: Vec<_> = cls.get(SKETCH).into_iter().flatten().copied().collect();
    ids.sort_unstable();
    let mut out = Vec::new();
    for id in ids {
        let Some(payloads) = rec.get(&id) else {
            continue;
        };
        let Some((fr, members)) = payloads.iter().find_map(|b| {
            let transform = ptr(b, 4)?;
            if !is(TRANSFORM, transform) {
                return None;
            }
            let fr = rec.get(&transform)?.iter().find_map(|v| frame(v))?;
            // Geometry list follows a short versioned prefix. Validate its full
            // pointer array and count instead of relying on the numeric type.
            for at in 20..40.min(b.len()) {
                let Some(n) = u32_at(b, at) else { continue };
                if n == 0 || n > 100_000 {
                    continue;
                }
                let Some(end) = (at + 4).checked_add(n * 11) else {
                    continue;
                };
                if end > b.len() {
                    continue;
                }
                let Some(members) = (0..n)
                    .map(|i| ptr(b, at + 4 + i * 11))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                if members.iter().all(|m| rec.contains_key(m)) {
                    return Some((fr, members));
                }
            }
            None
        }) else {
            continue;
        };
        let mut sketch = ImportedSketch {
            name: format!("F3D Sketch {}", out.len() + 1),
            frame: fr,
            curves: vec![],
            points: vec![],
            unsupported: 0,
        };
        let points: HashMap<_, _> = members
            .iter()
            .filter(|m| is(POINT, **m))
            .filter_map(|m| Some((*m, rec.get(m)?.iter().find_map(|v| point(v))?)))
            .collect();
        for member in members {
            let Some(values) = rec.get(&member) else {
                continue;
            };
            if is(POINT, member) {
                if let Some(p) = points.get(&member) {
                    sketch.points.push(*p);
                } else {
                    sketch.unsupported += 1;
                }
            } else if is(CIRCLE, member) {
                if let Some(c) = values.iter().find_map(|v| round(v)) {
                    sketch.curves.push(c);
                } else {
                    sketch.unsupported += 1;
                }
            } else if is(LINE, member) {
                let line = values.iter().find_map(|v| {
                    let at = attributes(v)? + 96;
                    Some((*points.get(&ptr(v, at)?)?, *points.get(&ptr(v, at + 11)?)?))
                });
                if let Some((a, b)) = line {
                    if (a[0] - b[0]).hypot(a[1] - b[1]) > 1e-9 {
                        sketch.curves.push(Curve::Line(a, b));
                    }
                } else {
                    sketch.unsupported += 1;
                }
            } else {
                sketch.unsupported += 1;
            }
        }
        if !sketch.curves.is_empty() || !sketch.points.is_empty() {
            out.push(sketch);
        }
    }
    let skipped = declared.saturating_sub(out.len());
    Ok((out, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn class(meta: &mut Vec<u8>, guid: &str, ids: &[u64]) {
        meta.extend(36u32.to_le_bytes());
        meta.extend(guid.as_bytes());
        meta.extend(36u32.to_le_bytes());
        meta.extend(b"00000000-0000-0000-0000-000000000000");
        meta.extend(0u32.to_le_bytes());
        meta.extend(8u32.to_le_bytes());
        meta.extend(b"Geometry");
        meta.extend((ids.len() as u32).to_le_bytes());
        for id in ids {
            meta.extend(id.to_le_bytes());
        }
    }
    fn record(b: &mut Vec<u8>, indexes: &mut Vec<(u64, u64)>, id: u64, kind: u32, data: &[u8]) {
        indexes.push((id, b.len() as u64));
        let kind = kind.to_string();
        b.extend((kind.len() as u32).to_le_bytes());
        b.extend(kind.as_bytes());
        b.extend(id.to_le_bytes());
        b.extend(data);
    }
    fn reference(id: u64) -> Vec<u8> {
        let mut b = vec![1];
        b.extend(id.to_le_bytes());
        b.extend([0, 0]);
        b
    }
    fn fixture() -> (Vec<u8>, Vec<u8>) {
        let mut meta = vec![];
        for (guid, ids) in [
            (SKETCH, vec![10]),
            (TRANSFORM, vec![1]),
            (POINT, vec![2, 3]),
            (LINE, vec![4]),
            (CIRCLE, vec![5, 6]),
        ] {
            class(&mut meta, guid, &ids);
        }
        let mut bulk = vec![];
        let mut indexes = vec![];
        let mut tr = vec![0; 7];
        for i in 0..16 {
            tr.extend((if i % 5 == 0 { 1f64 } else { 0f64 }).to_le_bytes());
        }
        tr.resize(147, 0);
        record(&mut bulk, &mut indexes, 1, 257, &tr);
        for (id, p) in [(2, [0f64, 0., 0.]), (3, [1., 2., 0.])] {
            let mut b = vec![0; 27];
            for d in p {
                b.extend(d.to_le_bytes());
            }
            record(&mut bulk, &mut indexes, id, 258, &b);
        }
        let mut line = vec![0; 106];
        line.extend(reference(2));
        line.extend(reference(3));
        record(&mut bulk, &mut indexes, 4, 259, &line);
        for (id, nz, end) in [
            (5, 1f64, std::f64::consts::TAU),
            (6, -1., std::f64::consts::PI),
        ] {
            let mut b = vec![0; 10];
            for d in [0f64, 0., 0., 0., 0., nz, 1., 0., 0., 2., 0., end] {
                b.extend(d.to_le_bytes());
            }
            record(&mut bulk, &mut indexes, id, 260, &b);
        }
        let mut sk = vec![0; 26];
        sk[4..15].copy_from_slice(&reference(1));
        sk.extend(5u32.to_le_bytes());
        for id in [2, 3, 4, 5, 6] {
            sk.extend(reference(id));
        }
        record(&mut bulk, &mut indexes, 10, 256, &sk);
        // Complete v8 metadata, including class catalog and actual bulk offsets.
        let mut header = vec![];
        header.extend(6u32.to_le_bytes());
        header.extend(b"Design");
        header.extend(0u32.to_le_bytes()); // segment ID
        header.extend(0u32.to_le_bytes()); // empty wide UUID
        header.extend(1234u32.to_le_bytes());
        header.extend(8u32.to_le_bytes());
        header.extend(0u64.to_le_bytes()); // software version
        header.extend(0u32.to_le_bytes()); // type
        header.extend(0u32.to_le_bytes()); // add-in
        header.extend(1u32.to_le_bytes()); // Neutron type
        header.extend(0u32.to_le_bytes()); // UUID list
        header.extend(5u32.to_le_bytes()); // class count
        header.extend(meta);
        header.extend(0u32.to_le_bytes()); // special entity set
        header.extend((indexes.len() as u32).to_le_bytes());
        for (id, offset) in indexes {
            header.extend(id.to_le_bytes());
            header.extend(offset.to_le_bytes());
        }
        header.extend(0u32.to_le_bytes()); // extensions
        header.extend(11u64.to_le_bytes()); // ID counter
        header.extend(0u32.to_le_bytes()); // reference contexts
        header.extend(0u32.to_le_bytes()); // migrator map
        (header, bulk)
    }
    #[test]
    fn metadata_ids_recover_lines_circles_arcs_and_mm_units() {
        let (m, b) = fixture();
        let s = decode(&m, &b);
        assert_eq!(s.len(), 1);
        let s = &s[0];
        assert_eq!(s.curves.len(), 3);
        assert_eq!(s.unsupported, 0);
        assert!(matches!(s.curves[0], Curve::Line([0., 0.], [10., 20.])));
        assert!(matches!(s.curves[1],Curve::Circle{radius,..}if radius==20.));
        assert!(
            matches!(s.curves[2],Curve::Arc{start,sweep,..}if (start+std::f64::consts::PI).abs()<1e-9 && (sweep-std::f64::consts::PI).abs()<1e-9)
        );
        assert_eq!(s.frame.x, [1., 0., 0.]);
        assert_eq!(s.frame.y, [0., 1., 0.]);
    }
    #[test]
    fn versioned_metadata_and_embedded_header_bytes_use_only_indexed_boundaries() {
        let (meta, mut bulk) = fixture();
        // Header-shaped bytes in an unused point prefix used to split a point
        // record and silently discard its coordinates in the marker scanner.
        let point_start = 15 + 147;
        let fake = [
            3u32.to_le_bytes().as_slice(),
            b"999",
            &9999u64.to_le_bytes(),
        ]
        .concat();
        bulk[point_start + 15 + 10..point_start + 15 + 25].copy_from_slice(&fake);
        for version in [4u32, 5, 6, 7, 8] {
            let mut m = meta.clone();
            m[22..26].copy_from_slice(&version.to_le_bytes());
            if version < 6 {
                m[18..22].copy_from_slice(&134228292u32.to_le_bytes());
            }
            if version < 6 {
                m.drain(26..34);
            }
            let remove = if version == 4 {
                8
            } else if version < 8 {
                4
            } else {
                0
            };
            m.truncate(m.len() - remove);
            let (sketches, skipped) = decode_indexed(&m, &bulk).unwrap();
            assert_eq!(skipped, 0, "version {version}");
            assert_eq!(sketches[0].curves.len(), 3, "version {version}");
            assert_eq!(sketches[0].points, [[0., 0.], [10., 20.]]);
        }
    }

    #[test]
    fn malformed_metadata_or_catalog_ids_are_rejected() {
        let (meta, mut bulk) = fixture();
        for len in 0..meta.len() {
            assert!(decode_indexed(&meta[..len], &bulk).is_err());
        }
        let mut bad = meta.clone();
        bad[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_indexed(&bad, &bulk).is_err());
        bulk[7..15].copy_from_slice(&9876u64.to_le_bytes());
        assert!(decode_indexed(&meta, &bulk)
            .unwrap_err()
            .contains("mismatch"));
    }

    #[test]
    fn malformed_sketch_stream_produces_a_recoverable_archive_notice() {
        use std::io::Write;
        let (mut meta, bulk) = fixture();
        meta[22..26].copy_from_slice(&9u32.to_le_bytes());
        let path =
            std::env::temp_dir().join(format!("refusion-sketch-notice-{}.f3d", std::process::id()));
        let mut archive = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        for (name, bytes) in [("MetaStream.dat", &meta), ("BulkStream.dat", &bulk)] {
            archive
                .start_file(
                    format!("AssetName[Active]/Design1/{name}"),
                    zip::write::SimpleFileOptions::default(),
                )
                .unwrap();
            archive.write_all(bytes).unwrap();
        }
        archive.finish().unwrap();
        let (sketches, _, warnings) = crate::read_sketches_report(&path).unwrap();
        assert!(sketches.is_empty());
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("version 9") && warnings[0].contains("Design1/MetaStream.dat")
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn truncated_or_unknown_streams_do_not_panic_or_invent_geometry() {
        let (m, b) = fixture();
        for len in 0..b.len() {
            let _ = decode(&m, &b[..len]);
        }
        assert!(decode(&[], &b).is_empty());
        assert!(decode(&m, &[]).is_empty());
        let mut bad = b.clone();
        let at = 15 + 7;
        bad[at..at + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(decode(&m, &bad).is_empty());
    }
    #[test]
    fn archive_without_bodies_can_import_editable_sketch_geometry() {
        use std::io::Write;
        let (m, b) = fixture();
        let path =
            std::env::temp_dir().join(format!("refusion-sketch-only-{}.f3d", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("AssetName[Active]/Design1/MetaStream.dat", options)
            .unwrap();
        zip.write_all(&m).unwrap();
        zip.start_file("AssetName[Active]/Design1/BulkStream.dat", options)
            .unwrap();
        zip.write_all(&b).unwrap();
        zip.finish().unwrap();
        let report = crate::import(&path).unwrap();
        assert!(report.parts.is_empty());
        assert_eq!(report.sketches.len(), 1);
        std::fs::remove_file(path).unwrap();
    }
}
