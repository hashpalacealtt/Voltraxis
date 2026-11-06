//! Q4_K (llama.cpp) dequantization.

/// Dequantize a Q4_K super-block (144 bytes) into 256 f32 values.
pub fn dequant_super_block(sb: &[u8], out: &mut [f32]) {
    assert!(sb.len() >= 144);
    assert!(out.len() >= 256);

    // d, dmin as f16
    let d = f16_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([sb[2], sb[3]]));

    // scales: 12 bytes at offset 4
    let scales = &sb[4..16];
    // quants: 128 bytes at offset 16
    let qs = &sb[16..144];

    // per-block scale/min (8 blocks of 32)
    let mut sc = [0.0f32; 8];
    let mut m = [0.0f32; 8];
    for i in 0..4usize {
        sc[i] = d * (scales[i] & 0x3F) as f32;
        m[i] = dmin * (scales[i + 4] & 0x3F) as f32;
    }
    for i in 4..8usize {
        sc[i] = d * (((scales[i + 4] & 0x0F) as usize) | (((scales[i] >> 6) as usize) << 4)) as f32;
        m[i] = dmin
            * (((scales[i + 4] >> 4) as usize) | (((scales[i + 8 - 4] >> 6) as usize) << 4)) as f32;
    }
    // fix min indexing for i>=4: standard formula uses scales[i-4] high bits
    // Recompute mins for i>=4 properly:
    for i in 4..8usize {
        m[i] = dmin
            * (((scales[i + 4] >> 4) as usize) | (((scales[i - 4] >> 6) as usize) << 4)) as f32;
    }

    // dequantize
    for (blk, (s, mn)) in sc.iter().zip(m.iter()).enumerate() {
        let qoff = blk * 16; // 16 bytes = 32 nibbles per block
        let ooff = blk * 32;
        for l in 0..32usize {
            let q = if l % 2 == 0 {
                qs[qoff + l / 2] & 0x0F
            } else {
                (qs[qoff + l / 2] >> 4) & 0x0F
            };
            out[ooff + l] = *s * (q as f32) - *mn;
        }
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    // IEEE 754 half -> single
    let sign = (h >> 15) as u32;
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // subnormal
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let exp32 = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (exp32 << 23) | (m << 13))
        }
    } else if exp == 0x1F {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (exp32 << 23) | (mant << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_runs() {
        let sb = vec![0u8; 144];
        let mut out = vec![0.0f32; 256];
        dequant_super_block(&sb, &mut out);
        // all zeros input -> d=0, dmin=0 -> all zeros
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f16_roundtrip() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }
}
