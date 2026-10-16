//! Disk-backed GL program binary cache for the GLES backend.
//!
//! Compiling shaders is expensive on many GLES drivers (PowerVR can take
//! seconds per pipeline, hundreds of pipelines per app). GL program binaries
//! (`glGetProgramBinary` / `glProgramBinary`, core in GL ES 3.0 and GL 4.1)
//! persist the driver's compiled program between runs: on a cache hit the
//! driver skips its own compile and link entirely.
//!
//! Binaries are driver-specific by spec — `glProgramBinary` fails for a
//! binary in an unknown format (e.g. after a driver update), so a stale
//! entry never breaks anything: the pipeline falls back to a normal compile
//! and the entry is replaced.
//!
//! Entries are keyed by a content hash of the final GLSL sources (post-naga,
//! including pipeline constants and bounds-check policy output) plus the GLSL
//! version and multiview configuration — everything that changes what the
//! driver compiles. The pipeline layout is deliberately *not* part of the
//! key: uniform block bindings and sampler slots are program state applied
//! after creation, so binaries are safely shared across layouts.
//!
//! Cache location: `WGPU_GLES_PROGRAM_CACHE` env var if set; otherwise, on
//! Android, the app's internal files dir (derived from `/proc/self/cmdline`,
//! no Android API access needed). Disabled (no-op) everywhere else.

use std::{
    fs,
    path::PathBuf,
    str,
    string::String,
    vec::Vec,
};

use naga::FastHashMap;
use parking_lot::Mutex;

/// File format: `format: u32 LE | len: u32 LE | payload`. The file name is
/// the hex-encoded content key.
pub(crate) struct ProgramBinaryDiskCache {
    dir: PathBuf,
    entries: Mutex<FastHashMap<u64, StoredBinary>>,
}

#[derive(Clone)]
pub(crate) struct StoredBinary {
    pub format: u32,
    pub data: Vec<u8>,
}

impl ProgramBinaryDiskCache {
    /// Open the cache, loading any existing entries. Returns `None` when no
    /// cache location is available.
    pub(crate) fn open() -> Option<Self> {
        let dir = cache_dir()?;
        let _ = fs::create_dir_all(&dir);

        let mut entries = FastHashMap::default();
        if let Ok(files) = fs::read_dir(&dir) {
            for file in files.flatten() {
                let name = file.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(key) = key_from_file_name(name) else { continue };
                let Ok(raw) = fs::read(file.path()) else { continue };
                let Some(binary) = decode(&raw) else { continue };
                entries.insert(key, binary);
            }
        }

        Some(Self {
            dir,
            entries: Mutex::new(entries),
        })
    }

    pub(crate) fn get(&self, key: u64) -> Option<StoredBinary> {
        self.entries.lock().get(&key).cloned()
    }

    /// Persist a binary and keep it in memory. Failures are non-fatal: an
    /// unwritable cache just means every run recompiles.
    pub(crate) fn store(&self, key: u64, format: u32, data: &[u8]) {
        self.entries.lock().insert(
            key,
            StoredBinary {
                format,
                data: data.to_vec(),
            },
        );
        let mut raw = Vec::with_capacity(data.len() + 8);
        raw.extend_from_slice(&format.to_le_bytes());
        raw.extend_from_slice(&(data.len() as u32).to_le_bytes());
        raw.extend_from_slice(data);
        let path = self.dir.join(file_name(key));
        let tmp = self.dir.join(format!("{}.tmp", file_name(key)));
        if fs::write(&tmp, &raw).is_ok() {
            let _ = fs::rename(&tmp, &path);
        }
    }

    /// Drop a stale entry: after a load failure the driver won't accept this
    /// binary again until the cache is refreshed.
    pub(crate) fn invalidate(&self, key: u64) {
        self.entries.lock().remove(&key);
        let _ = fs::remove_file(self.dir.join(file_name(key)));
    }
}

fn decode(raw: &[u8]) -> Option<StoredBinary> {
    if raw.len() < 8 {
        return None;
    }
    let format = u32::from_le_bytes(raw[0..4].try_into().ok()?);
    let len = u32::from_le_bytes(raw[4..8].try_into().ok()?) as usize;
    if raw.len() != 8 + len {
        return None;
    }
    Some(StoredBinary {
        format,
        data: raw[8..].to_vec(),
    })
}

fn file_name(key: u64) -> String {
    format!("{key:016x}.wpubin")
}

fn key_from_file_name(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".wpubin")?;
    if stem.len() != 16 {
        return None;
    }
    u64::from_str_radix(stem, 16).ok()
}

/// FNV-1a over length-framed parts; stable across runs and compilers.
pub(crate) fn hash_parts(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        h ^= part.len() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("WGPU_GLES_PROGRAM_CACHE") {
        return if dir.is_empty() { None } else { Some(dir.into()) };
    }
    if cfg!(target_os = "android") {
        // The app's internal files dir, derived without any Android API:
        // the process name is the package name, and apps can always write
        // their own data dir.
        let cmdline = fs::read_to_string("/proc/self/cmdline").ok()?;
        let package = cmdline.split('\0').next()?;
        if package.is_empty() {
            return None;
        }
        return Some(PathBuf::from(format!(
            "/data/user/0/{package}/files/wgpu-gles-programs"
        )));
    }
    None
}
