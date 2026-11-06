//! Minimal GGUF (v3) reader for LFM3 weights.
//!
//! Parses the header, metadata KV pairs, and tensor info. Does not yet
//! perform inference; provides the foundation for loading real LFM3 weights.

use std::fs::File;
use std::io::{BufReader, Read};

#[derive(Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_str(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn read_value(r: &mut impl Read, ty: u32) -> std::io::Result<GgufValue> {
    match ty {
        0 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufValue::U8(b[0]))
        }
        1 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufValue::I8(b[0] as i8))
        }
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufValue::U16(u16::from_le_bytes(b)))
        }
        3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufValue::I16(i16::from_le_bytes(b)))
        }
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] != 0
        })),
        8 => Ok(GgufValue::Str(read_str(r)?)),
        9 => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(read_value(r, elem_ty)?);
            }
            Ok(GgufValue::Array(v))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => {
            let v = read_u64(r)?;
            Ok(GgufValue::I64(v as i64))
        }
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown gguf type {}", ty),
        )),
    }
}

pub fn parse(path: &str) -> std::io::Result<GgufFile> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    // magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }
    let version = read_u32(&mut r)?;
    if version < 2 || version > 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported gguf version {}", version),
        ));
    }
    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;

    // metadata
    let mut metadata = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = read_str(&mut r)?;
        let ty = read_u32(&mut r)?;
        let val = read_value(&mut r, ty)?;
        metadata.push((key, val));
    }

    // tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_str(&mut r)?;
        let ndim = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(&mut r)?);
        }
        let dtype = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    Ok(GgufFile {
        version,
        tensor_count,
        kv_count,
        metadata,
        tensors,
    })
}

pub fn print_summary(path: &str) {
    match parse(path) {
        Ok(g) => {
            println!(
                "GGUF v{}: {} tensors, {} kv",
                g.version, g.tensor_count, g.kv_count
            );
            for (k, v) in &g.metadata {
                println!("  meta: {} = {:?}", k, v);
            }
            println!("  tensors (first 20):");
            for t in g.tensors.iter().take(20) {
                println!(
                    "    {} {:?} dtype={} off={}",
                    t.name, t.shape, t.dtype, t.offset
                );
            }
            println!("  ... total {}", g.tensors.len());
        }
        Err(e) => println!("GGUF parse error: {}", e),
    }
}
