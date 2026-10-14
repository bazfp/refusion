#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Dbl(f64),
    Str(String),
    Bool(bool),
    Ptr(i64),
    Id(String),
    Open,
    Close,
    Pos([f64; 3]),
    Vec3([f64; 3]),
    Enum(i64),
    V2([f64; 2]),
}

#[derive(Clone, Debug)]
pub struct Record {
    pub name: String,
    /// Tokens after the type identifiers.
    pub toks: Vec<Tok>,
}

#[derive(Debug)]
pub struct Sab {
    pub records: Vec<Record>,
    pub unit_mm: f64,
    /// (record, token index of the `Open`) of each sub-type definition.
    pub subtypes: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct SabError(pub String);

fn rd<const N: usize>(d: &[u8], p: &mut usize) -> Result<[u8; N], SabError> {
    let end = p
        .checked_add(N)
        .ok_or_else(|| SabError("SAB offset overflow".into()))?;
    let b: [u8; N] = d
        .get(*p..end)
        .ok_or_else(|| SabError(format!("unexpected end of data at byte {p}")))?
        .try_into()
        .unwrap();
    *p = end;
    Ok(b)
}

fn rd_str(d: &[u8], p: &mut usize, n: usize) -> Result<String, SabError> {
    let end = p
        .checked_add(n)
        .ok_or_else(|| SabError("SAB string length overflow".into()))?;
    let b = d
        .get(*p..end)
        .ok_or_else(|| SabError(format!("unexpected end of string at byte {p}")))?;
    *p = end;
    Ok(String::from_utf8_lossy(b).into_owned())
}

pub fn parse(d: &[u8]) -> Result<Sab, SabError> {
    let mut p = if d.starts_with(b"ASM BinaryFile") {
        14
    } else if d.starts_with(b"ACIS BinaryFile") {
        15
    } else {
        return Err(SabError("not a SAB stream".into()));
    };
    let width = match d.get(p) {
        Some(b'4') => {
            p += 1;
            4
        }
        Some(b'8') => {
            p += 1;
            8
        }
        Some(b) if b.is_ascii_digit() => {
            return Err(SabError(format!(
                "unsupported SAB integer width {}",
                *b as char
            )))
        }
        _ if d.starts_with(b"ACIS BinaryFile") => 4,
        _ => return Err(SabError("missing SAB integer width".into())),
    };
    if width == 8 {
        rd::<32>(d, &mut p)?;
    } else {
        rd::<16>(d, &mut p)?;
    }
    let word = |p: &mut usize| -> Result<i64, SabError> {
        if width == 8 {
            Ok(i64::from_le_bytes(rd(d, p)?))
        } else {
            Ok(i32::from_le_bytes(rd(d, p)?) as i64)
        }
    };
    let mut all: Vec<Vec<Tok>> = Vec::new();
    let mut cur: Vec<Tok> = Vec::new();
    while p < d.len() {
        let t = d[p];
        p += 1;
        let tok = match t {
            0x04 => Tok::Int(word(&mut p)?),
            0x06 => Tok::Dbl(f64::from_le_bytes(rd(d, &mut p)?)),
            0x07 => {
                let n = rd::<1>(d, &mut p)?[0] as usize;
                Tok::Str(rd_str(d, &mut p, n)?)
            }
            0x08 => {
                let n = u16::from_le_bytes(rd(d, &mut p)?) as usize;
                Tok::Str(rd_str(d, &mut p, n)?)
            }
            0x12 => {
                let n = u32::from_le_bytes(rd(d, &mut p)?) as usize;
                Tok::Str(rd_str(d, &mut p, n)?)
            }
            0x0A => Tok::Bool(true),
            0x0B => Tok::Bool(false),
            0x0C => Tok::Ptr(word(&mut p)?),
            0x0D | 0x0E => {
                let n = rd::<1>(d, &mut p)?[0] as usize;
                Tok::Id(rd_str(d, &mut p, n)?)
            }
            0x0F => Tok::Open,
            0x10 => Tok::Close,
            0x11 => {
                all.push(std::mem::take(&mut cur));
                continue;
            }
            0x13 | 0x14 => {
                let mut v = [0.0; 3];
                for x in &mut v {
                    *x = f64::from_le_bytes(rd(d, &mut p)?);
                }
                if t == 0x13 {
                    Tok::Pos(v)
                } else {
                    Tok::Vec3(v)
                }
            }
            0x15 => Tok::Enum(word(&mut p)?),
            0x16 => {
                let a = f64::from_le_bytes(rd(d, &mut p)?);
                let b = f64::from_le_bytes(rd(d, &mut p)?);
                Tok::V2([a, b])
            }
            0x17 => Tok::Dbl(f64::from_le_bytes(rd(d, &mut p)?)),
            _ => {
                return Err(SabError(format!(
                    "unsupported SAB tag 0x{t:02x} at byte {}",
                    p - 1
                )))
            }
        };
        cur.push(tok);
        // Section sentinels end with a final identifier, without an 0x11.
        // They do not consume entity indices. In particular, history_stream
        // and a following Timestamp attribute must retain their own names.
        if t == 0x0D && cur.iter().all(|t| matches!(t, Tok::Id(_))) {
            let name = cur
                .iter()
                .map(|t| if let Tok::Id(s) = t { s.as_str() } else { "" })
                .collect::<Vec<_>>()
                .join("-");
            if name == "End-of-ASM-data" {
                if p != d.len() {
                    return Err(SabError(format!("data after SAB end marker at byte {p}")));
                }
                cur.clear();
            } else if name == "Begin-of-ASM-History-Data" || name == "End-of-ASM-History-Section" {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        return Err(SabError(format!("unterminated SAB record at byte {p}")));
    }
    if all.is_empty() {
        return Err(SabError("missing SAB header record".into()));
    }
    let unit_mm = 10.0;
    let mut records = Vec::with_capacity(all.len());
    let mut subtypes = Vec::new();
    for (ri, toks) in all.into_iter().enumerate() {
        let skip = if ri == 0 {
            toks.iter()
                .position(|t| matches!(t, Tok::Id(_)))
                .unwrap_or(0)
        } else {
            0
        };
        let toks = &toks[skip..];
        let n_ids = toks.iter().take_while(|t| matches!(t, Tok::Id(_))).count();
        let name = toks[..n_ids]
            .iter()
            .map(|t| if let Tok::Id(s) = t { s.as_str() } else { "" })
            .collect::<Vec<_>>()
            .join("-");
        let body: Vec<Tok> = toks[n_ids..].to_vec();
        for (i, t) in body.iter().enumerate() {
            if *t == Tok::Open && body.get(i + 1) != Some(&Tok::Id("ref".into())) {
                subtypes.push((ri, i));
            }
        }
        records.push(Record { name, toks: body });
    }
    Ok(Sab {
        records,
        unit_mm,
        subtypes,
    })
}

impl Sab {
    /// Tokens of sub-type definition `n` (starting after its `Open`).
    pub fn subtype(&self, n: usize) -> Option<&[Tok]> {
        let &(r, i) = self.subtypes.get(n)?;
        block_body(&self.records.get(r)?.toks, i)
    }
}

/// A subtype ends at its matching brace, not at the end of the containing record.
pub(crate) fn block_body(toks: &[Tok], open: usize) -> Option<&[Tok]> {
    if toks.get(open) != Some(&Tok::Open) {
        return None;
    }
    let mut depth = 0usize;
    for (i, tok) in toks.iter().enumerate().skip(open) {
        match tok {
            Tok::Open => depth += 1,
            Tok::Close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&toks[open + 1..i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identifier(d: &mut Vec<u8>, tag: u8, value: &str) {
        d.extend([tag, value.len() as u8]);
        d.extend(value.as_bytes());
    }

    fn sentinel(d: &mut Vec<u8>, parts: &[&str]) {
        for (i, part) in parts.iter().enumerate() {
            identifier(d, if i + 1 == parts.len() { 0x0d } else { 0x0e }, part);
        }
    }

    #[test]
    fn binary_file8_keeps_wide_integer_pointer_and_enum_values() {
        let mut data = b"ASM BinaryFile8".to_vec();
        data.extend([0; 32]);
        identifier(&mut data, 0x0d, "asmheader");
        data.push(0x11);
        identifier(&mut data, 0x0d, "point");
        for tag in [0x04, 0x0c, 0x15] {
            data.push(tag);
            data.extend((1i64 << 40).to_le_bytes());
        }
        data.push(0x11);
        let sab = parse(&data).unwrap();
        assert_eq!(
            sab.records[1].toks,
            [Tok::Int(1 << 40), Tok::Ptr(1 << 40), Tok::Enum(1 << 40)]
        );
        assert!(parse(&data[..46]).is_err());
    }

    #[test]
    fn history_sentinels_do_not_merge_names_or_shift_entity_indices() {
        let mut data = b"ASM BinaryFile4".to_vec();
        data.extend([0; 16]);
        identifier(&mut data, 0x0d, "asmheader");
        data.push(0x11);
        identifier(&mut data, 0x0d, "body");
        data.push(0x11);
        sentinel(&mut data, &["Begin", "of", "ASM", "History", "Data"]);
        identifier(&mut data, 0x0d, "history_stream");
        data.push(0x11);
        sentinel(&mut data, &["End", "of", "ASM", "History", "Section"]);
        sentinel(&mut data, &["ATTRIB_CUSTOM", "attrib"]);
        data.push(0x0c);
        data.extend(1i32.to_le_bytes()); // body still index 1
        data.push(0x11);
        sentinel(&mut data, &["End", "of", "ASM", "data"]);
        let sab = parse(&data).unwrap();
        assert_eq!(
            sab.records
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            [
                "asmheader",
                "body",
                "history_stream",
                "ATTRIB_CUSTOM-attrib"
            ]
        );
        assert_eq!(sab.records[3].toks, [Tok::Ptr(1)]);
    }

    #[test]
    fn unknown_tags_and_unterminated_records_are_errors_with_offsets() {
        let mut data = b"ASM BinaryFile4".to_vec();
        data.extend([0; 16]);
        identifier(&mut data, 0x0d, "asmheader");
        assert!(parse(&data).unwrap_err().0.contains("unterminated"));
        data.push(0x11);
        data.push(0x99);
        let error = parse(&data).unwrap_err().0;
        assert!(error.contains("0x99") && error.contains(&(data.len() - 1).to_string()));
    }

    #[test]
    fn subtypes_stop_at_the_matching_brace_including_nested_blocks() {
        let sab = Sab {
            records: vec![Record {
                name: "spline-surface".into(),
                toks: vec![
                    Tok::Open,
                    Tok::Id("exact_spl_sur".into()),
                    Tok::Open,
                    Tok::Id("nested".into()),
                    Tok::Close,
                    Tok::Close,
                    Tok::Dbl(99.),
                ],
            }],
            subtypes: vec![(0, 0), (0, 2)],
            unit_mm: 10.,
        };
        assert_eq!(sab.subtype(0).unwrap().len(), 4);
        assert_eq!(sab.subtype(1).unwrap(), [Tok::Id("nested".into())]);
        assert!(block_body(&[Tok::Open], 0).is_none());
    }

    /// Hand-built stream: header record + `point` + `straight-curve` + a
    /// sub-type and a `{ ref 0 }` back-reference.
    #[test]
    fn tokenises_records_and_subtypes() {
        let mut d = b"ASM BinaryFile4".to_vec();
        d.extend([0u8; 16]);
        let id = |d: &mut Vec<u8>, tag: u8, s: &str| {
            d.push(tag);
            d.push(s.len() as u8);
            d.extend(s.as_bytes());
        };
        // header record
        id(&mut d, 0x07, "Autodesk Neutron");
        d.push(0x06);
        d.extend(10.0f64.to_le_bytes());
        id(&mut d, 0x0D, "asmheader");
        d.push(0x11);
        // point
        id(&mut d, 0x0D, "point");
        d.push(0x0C);
        d.extend((-1i32).to_le_bytes());
        d.push(0x13);
        for v in [1.0f64, 2.0, 3.0] {
            d.extend(v.to_le_bytes());
        }
        d.push(0x11);
        // intcurve with a sub-type, then a record referring back to it
        id(&mut d, 0x0E, "intcurve");
        id(&mut d, 0x0D, "curve");
        d.push(0x0F);
        id(&mut d, 0x0D, "exact_int_cur");
        d.push(0x10);
        d.push(0x11);
        id(&mut d, 0x0E, "intcurve");
        id(&mut d, 0x0D, "curve");
        d.push(0x0F);
        id(&mut d, 0x0D, "ref");
        d.push(0x04);
        d.extend(0i32.to_le_bytes());
        d.push(0x10);
        d.push(0x11);

        let s = parse(&d).unwrap();
        assert_eq!(s.records.len(), 4);
        assert_eq!(s.records[0].name, "asmheader");
        assert_eq!(s.records[1].name, "point");
        assert_eq!(s.records[1].toks[1], Tok::Pos([1.0, 2.0, 3.0]));
        assert_eq!(s.records[2].name, "intcurve-curve");
        assert_eq!(s.subtypes.len(), 1, "ref blocks are not definitions");
        assert_eq!(s.subtype(0).unwrap()[0], Tok::Id("exact_int_cur".into()));
        assert_eq!(s.unit_mm, 10.0);
    }
}
