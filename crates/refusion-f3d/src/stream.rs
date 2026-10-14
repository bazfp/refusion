use std::collections::{HashMap, HashSet};

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("stream offset overflow")?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| format!("truncated stream at byte {}", self.at))?;
        self.at = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<&'a str, String> {
        let n = self.u32()? as usize;
        std::str::from_utf8(self.take(n)?).map_err(|_| "invalid UTF-8 stream string".into())
    }
    fn wide(&mut self) -> Result<(), String> {
        let n = (self.u32()? as usize)
            .checked_mul(2)
            .ok_or("wide string length overflow")?;
        self.take(n)?;
        Ok(())
    }
    /// Bounds counts before allocating or iterating, including on malformed files.
    fn count(&mut self, minimum_bytes: usize) -> Result<usize, String> {
        let n = self.u32()? as usize;
        if n > (self.bytes.len() - self.at) / minimum_bytes {
            return Err(format!("invalid stream count {n} at byte {}", self.at - 4));
        }
        Ok(n)
    }
}

struct Class {
    identifier: String,
    ids: HashSet<u64>,
}

pub(crate) struct Metadata {
    classes: Vec<Class>,
    /// Both main and extension records; IDs alone are not unique record keys.
    indexes: Vec<(u64, u64, bool)>,
}

impl Metadata {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        r.string()?; // segment user name
        r.u32()?; // segment ID
        r.wide()?; // segment UUID
                   // Current writers emit 1234; older valid files store other values here.
                   // The native loader reads this word without using it as a magic check.
        r.u32()?;
        let version = r.u32()?;
        if !(4..=8).contains(&version) {
            return Err(format!("unsupported F3D segment version {version}"));
        }
        if version >= 6 {
            r.u64()?; // software version
        }
        r.string()?; // segment type
        r.string()?; // add-in
        r.u32()?; // Neutron type
        for _ in 0..r.count(4)? {
            r.wide()?;
        }
        let mut classes = Vec::new();
        for _ in 0..r.count(20)? {
            let identifier = r.string()?.to_owned();
            r.string()?; // base identifier (possibly empty)
            r.u32()?; // class version
            r.string()?; // add-in
            let mut ids = HashSet::new();
            for _ in 0..r.count(8)? {
                ids.insert(r.u64()?);
            }
            classes.push(Class { identifier, ids });
        }
        for _ in 0..r.count(8)? {
            r.u64()?; // special entity set, not bulk offsets
        }
        let mut indexes = Vec::new();
        for main in [true, false] {
            for _ in 0..r.count(16)? {
                indexes.push((r.u64()?, r.u64()?, main));
            }
        }
        r.u64()?; // ID counter (all supported versions have it)
        if version > 4 {
            for _ in 0..r.count(28)? {
                r.u64()?; // XSegRefTable context key, not ordinal
                for _ in 0..4 {
                    r.wide()?; // link, document GUID/URN, asset GUID
                }
                r.u32()?; // target segment ID
            }
        }
        if version > 7 {
            for _ in 0..r.count(8)? {
                r.string()?;
                r.u32()?; // module migrator maximum
            }
        }
        if r.at != bytes.len() {
            return Err(format!("unexpected metadata tail at byte {}", r.at));
        }
        Ok(Self { classes, indexes })
    }

    pub(crate) fn classes(&self) -> HashMap<String, HashSet<u64>> {
        let mut out: HashMap<String, HashSet<u64>> = HashMap::new();
        for c in &self.classes {
            out.entry(c.identifier.to_ascii_uppercase())
                .or_default()
                .extend(&c.ids);
        }
        out
    }

    pub(crate) fn records<'a>(
        &self,
        bulk: &'a [u8],
    ) -> Result<HashMap<u64, Vec<&'a [u8]>>, String> {
        let mut starts = self.indexes.clone();
        starts.sort_unstable_by_key(|e| e.1);
        if starts.first().is_some_and(|e| e.1 != 0) || (starts.is_empty() && !bulk.is_empty()) {
            return Err("bulk stream has no indexed start at byte 0".into());
        }
        let mut out: HashMap<u64, Vec<&[u8]>> = HashMap::new();
        for (i, &(expected_id, offset, main)) in starts.iter().enumerate() {
            let start = usize::try_from(offset).map_err(|_| "bulk offset overflow")?;
            let end = if let Some(next) = starts.get(i + 1) {
                usize::try_from(next.1).map_err(|_| "bulk offset overflow")?
            } else {
                bulk.len()
            };
            if start >= end {
                return Err(format!("duplicate or unordered bulk offset {offset}"));
            }
            let data = bulk
                .get(start..end)
                .ok_or_else(|| format!("bulk offset {offset} outside stream"))?;
            let mut r = Reader { bytes: data, at: 0 };
            let name = r.string()?;
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("invalid bulk metatype at byte {offset}"));
            }
            let ordinal = name.parse::<usize>().ok().and_then(|n| n.checked_sub(256));
            let class = ordinal
                .and_then(|n| self.classes.get(n))
                .ok_or_else(|| format!("unknown bulk metatype {name} at byte {offset}"))?;
            let id = r.u64()?;
            if id != expected_id || (main && !class.ids.contains(&id)) {
                return Err(format!("bulk entity/catalog mismatch at byte {offset}"));
            }
            out.entry(id).or_default().push(&data[r.at..]);
        }
        Ok(out)
    }
}
