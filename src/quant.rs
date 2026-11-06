/// Default group size for 4-bit affine quantization (matches LFM3 MLX-4bit).
pub const DEFAULT_GROUP_SIZE: usize = 64;

/// A 4-bit affinely quantized tensor.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub n: usize,
    pub group_size: usize,
    /// Packed nibbles, two per byte (low nibble first).
    pub packed: Vec<u8>,
    /// Per-group scale.
    pub scales: Vec<f32>,
    /// Per-group zero point.
    pub zeros: Vec<f32>,
}

/// Quantize f32 values to 4-bit affine, grouped.
pub fn quantize_affine(values: &[f32], group_size: usize) -> QuantizedTensor {
    assert!(group_size > 0);
    let n = values.len();
    let num_groups = if n == 0 {
        0
    } else {
        (n + group_size - 1) / group_size
    };
    let mut scales = Vec::with_capacity(num_groups);
    let mut zeros = Vec::with_capacity(num_groups);
    let mut packed = vec![0u8; (n + 1) / 2];

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(n);
        let slice = &values[start..end];

        let mut vmin = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for &v in slice {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
        let scale = if vmax > vmin {
            (vmax - vmin) / 15.0
        } else {
            1.0
        };
        let zero = vmin;
        scales.push(scale);
        zeros.push(zero);

        for (i, &v) in slice.iter().enumerate() {
            let q = ((v - zero) / scale).round().clamp(0.0, 15.0) as u8;
            let idx = start + i;
            if idx % 2 == 0 {
                packed[idx / 2] = q;
            } else {
                packed[idx / 2] |= q << 4;
            }
        }
    }

    QuantizedTensor {
        n,
        group_size,
        packed,
        scales,
        zeros,
    }
}

/// Dequantize back to f32.
pub fn dequantize(q: &QuantizedTensor) -> Vec<f32> {
    let mut out = vec![0.0f32; q.n];
    for i in 0..q.n {
        let g = i / q.group_size;
        let nibble = if i % 2 == 0 {
            q.packed[i / 2] & 0x0F
        } else {
            (q.packed[i / 2] >> 4) & 0x0F
        };
        out[i] = q.zeros[g] + (nibble as f32) * q.scales[g];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_shape() {
        let vals: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let q = quantize_affine(&vals, DEFAULT_GROUP_SIZE);
        let dq = dequantize(&q);
        assert_eq!(dq.len(), vals.len());
    }

    #[test]
    fn packed_size() {
        let vals: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let q = quantize_affine(&vals, 64);
        assert_eq!(q.packed.len(), 64);
        assert_eq!(q.scales.len(), 2);
    }

    #[test]
    fn error_bounded() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let q = quantize_affine(&vals, 64);
        let dq = dequantize(&q);
        let max_err = vals
            .iter()
            .zip(&dq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 4-bit => 16 levels over range 0..63 => step ~4.2, err <= step
        assert!(max_err <= 4.5, "max_err={}", max_err);
    }

    #[test]
    fn empty_input() {
        let q = quantize_affine(&[], 64);
        assert_eq!(q.n, 0);
        assert!(q.packed.is_empty());
        assert!(dequantize(&q).is_empty());
    }
}
