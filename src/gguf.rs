//! Minimal GGUF (v2/v3) reader for LFM3 weights.

use std::fs;

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

#[derive(Debug, Clone)]
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
    pub data_offset: u64,
    pub bytes: Vec<u8>,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("eof".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f64(&mut self) -> Result<f64, String> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        let b = self.read_bytes(len)?;
        String::from_utf8(b.to_vec()).map_err(|e| e.to_string())
    }
}

fn read_value(c: &mut Cursor) -> Result<GgufValue, String> {
    let ty = c.u32()?;
    match ty {
        0 => Ok(GgufValue::U8(c.u8()?)),
        1 => Ok(GgufValue::I8(c.u8()? as i8)),
        2 => Ok(GgufValue::U16(c.u16()?)),
        3 => Ok(GgufValue::I16(c.u16()? as i16)),
        4 => Ok(GgufValue::U32(c.u32()?)),
        5 => Ok(GgufValue::I32(c.i32()?)),
        6 => Ok(GgufValue::F32(c.f32()?)),
        7 => Ok(GgufValue::Bool(c.u8()? != 0)),
        8 => Ok(GgufValue::Str(c.str()?)),
        9 => {
            let et = c.u32()?;
            let n = c.u64()? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(read_value_typed(c, et)?);
            }
            Ok(GgufValue::Array(v))
        }
        10 => Ok(GgufValue::U64(c.u64()?)),
        11 => Ok(GgufValue::I64(c.u64()? as i64)),
        12 => Ok(GgufValue::F64(c.f64()?)),
        _ => Err(format!("unknown type {}", ty)),
    }
}

fn read_value_typed(c: &mut Cursor, ty: u32) -> Result<GgufValue, String> {
    // reuse by pushing ty back? Instead duplicate minimal: read based on ty
    match ty {
        0 => Ok(GgufValue::U8(c.u8()?)),
        1 => Ok(GgufValue::I8(c.u8()? as i8)),
        2 => Ok(GgufValue::U16(c.u16()?)),
        3 => Ok(GgufValue::I16(c.u16()? as i16)),
        4 => Ok(GgufValue::U32(c.u32()?)),
        5 => Ok(GgufValue::I32(c.i32()?)),
        6 => Ok(GgufValue::F32(c.f32()?)),
        7 => Ok(GgufValue::Bool(c.u8()? != 0)),
        8 => Ok(GgufValue::Str(c.str()?)),
        10 => Ok(GgufValue::U64(c.u64()?)),
        11 => Ok(GgufValue::I64(c.u64()? as i64)),
        12 => Ok(GgufValue::F64(c.f64()?)),
        _ => Err(format!("unknown array type {}", ty)),
    }
}

pub fn parse(path: &str) -> Result<GgufFile, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let mut c = Cursor::new(&bytes);
    let magic = c.read_bytes(4)?;
    if magic != b"GGUF" {
        return Err("not GGUF".into());
    }
    let version = c.u32()?;
    if version < 2 || version > 3 {
        return Err(format!("bad version {}", version));
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;

    let mut metadata = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = c.str()?;
        let val = read_value(&mut c)?;
        metadata.push((key, val));
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = c.str()?;
        let ndim = c.u32()? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(c.u64()?);
        }
        let dtype = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    // alignment
    let align = metadata
        .iter()
        .find_map(|(k, v)| {
            if k == "general.alignment" {
                match v {
                    GgufValue::U32(x) => Some(*x as u64),
                    GgufValue::U64(x) => Some(*x),
                    _ => None,
                }
            } else {
                None
            }
        })
        .unwrap_or(32);

    let mut data_offset = c.pos as u64;
    if data_offset % align != 0 {
        data_offset += align - (data_offset % align);
    }

    Ok(GgufFile {
        version,
        tensor_count,
        kv_count,
        metadata,
        tensors,
        data_offset,
        bytes,
    })
}

pub fn print_summary(path: &str) {
    match parse(path) {
        Ok(g) => {
            println!(
                "GGUF v{}: {} tensors, {} kv, data_offset={}",
                g.version, g.tensor_count, g.kv_count, g.data_offset
            );
            for (k, v) in &g.metadata {
                println!("  meta: {} = {:?}", k, v);
            }
            println!("  tensors ({}):", g.tensors.len());
            for t in g.tensors.iter() {
                println!(
                    "    {} {:?} dtype={} off={}",
                    t.name, t.shape, t.dtype, t.offset
                );
            }
        }
        Err(e) => println!("GGUF parse error: {}", e),
    }
}

/// Get raw bytes for a tensor's data.
pub fn tensor_bytes<'a>(g: &'a GgufFile, t: &TensorInfo) -> &'a [u8] {
    let start = (g.data_offset + t.offset) as usize;
    // compute numel
    let numel: u64 = t.shape.iter().product();
    // bytes per element by dtype
    let bpe: usize = match t.dtype {
        0 => 4,  // F32
        1 => 2,  // F16
        24 => 2, // BF16 (some use 30)
        30 => 2, // BF16
        8 => 1,  // Q8_0 (approx; block-based, handle separately)
        12 => 1, // Q4_K (block-based)
        _ => 1,
    };
    // For block-based quants, size is not numel*bpe; compute via blocks.
    let size = match t.dtype {
        12 => {
            // Q4_K: 144 bytes per 256 values
            let blocks = (numel as usize + 255) / 256;
            blocks * 144
        }
        8 => {
            // Q8_0: 34 bytes per 32 values
            let blocks = (numel as usize + 31) / 32;
            blocks * 34
        }
        _ => numel as usize * bpe,
    };
    &g.bytes[start..start + size]
}

/// Dequantize a tensor to f32.
pub fn dequant_tensor(g: &GgufFile, t: &TensorInfo) -> Vec<f32> {
    let numel: usize = t.shape.iter().product::<u64>() as usize;
    let data = tensor_bytes(g, t);
    let mut out = vec![0.0f32; numel];
    match t.dtype {
        0 => {
            // F32
            for i in 0..numel {
                out[i] = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        1 => {
            // F16
            for i in 0..numel {
                let h = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
                out[i] = crate::q4k::f16_to_f32(h);
            }
        }
        30 => {
            // BF16
            for i in 0..numel {
                let h = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
                out[i] = f32::from_bits((h as u32) << 16);
            }
        }
        12 => {
            // Q4_K
            let nblocks = (numel + 255) / 256;
            for b in 0..nblocks {
                let off = b * 144;
                let ooff = b * 256;
                if off + 144 <= data.len() && ooff + 256 <= out.len() {
                    crate::q4k::dequant_super_block(
                        &data[off..off + 144],
                        &mut out[ooff..ooff + 256],
                    );
                }
            }
        }
        8 => {
            // Q8_0: 32 values per block, 34 bytes (2 scale f16 + 32 int8)
            let nblocks = (numel + 31) / 32;
            for b in 0..nblocks {
                let off = b * 34;
                let ooff = b * 32;
                if off + 34 <= data.len() {
                    let scale =
                        crate::q4k::f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
                    for i in 0..32usize {
                        if ooff + i < out.len() {
                            out[ooff + i] = scale * (data[off + 2 + i] as i8 as f32);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}
