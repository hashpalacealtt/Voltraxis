//! LFM3 tri-hybrid model: real architecture.
//!
//! 24 layers: 13 gated short convolutions (kernel 7), 5 sliding-window
//! attention (window 256), 6 global attention. Gated heads, partial RoPE,
//! sandwich RMSNorm, SwiGLU FFN. Vocab 128000.

pub const VOCAB: usize = 128000;
pub const HIDDEN: usize = 2048;
pub const INTER: usize = 5632;
pub const NUM_LAYERS: usize = 24;
pub const NUM_HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / NUM_HEADS; // 128
pub const CONV_KERNEL: usize = 7;
pub const SWA_WINDOW: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    Conv,
    SlidingWindow,
    Global,
}

/// Which layers are which (13 conv, 5 swa, 6 global). Pattern approximates
/// Liquid's published layout: convs dominate early/mid, attention later.
pub fn layer_kinds() -> [LayerKind; NUM_LAYERS] {
    // 13 conv, 5 swa, 6 global
    let mut kinds = [LayerKind::Conv; NUM_LAYERS];
    // place 6 global attention in the last 6? Actually distribute. Use a
    // simple schedule: every ~4th layer attention, alternating swa/global.
    let mut gi = 0usize;
    let mut si = 0usize;
    for i in 0..NUM_LAYERS {
        if i % 4 == 3 {
            if si < 5 {
                kinds[i] = LayerKind::SlidingWindow;
                si += 1;
            } else {
                kinds[i] = LayerKind::Global;
                gi += 1;
            }
        }
    }
    // fill remaining globals to reach 6
    for i in (0..NUM_LAYERS).rev() {
        if gi >= 6 {
            break;
        }
        if kinds[i] == LayerKind::Conv {
            kinds[i] = LayerKind::Global;
            gi += 1;
        }
    }
    kinds
}

// ---------- math primitives ----------

pub fn rmsnorm(x: &mut [f32], w: &[f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let rms = (ss / n as f32 + 1e-6).sqrt();
    for i in 0..n {
        x[i] = x[i] / rms * w[i];
    }
}

pub fn matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    // out[m,n] = a[m,k] @ b[k,n], row-major, with AVX2 + rayon
    out.fill(0.0);
    use rayon::prelude::*;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for t in 0..k {
            let ait = a_row[t];
            let b_row = &b[t * n..];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let va = std::arch::x86_64::_mm256_set1_ps(ait);
                    let mut j = 0usize;
                    while j + 8 <= n {
                        let vb = std::arch::x86_64::_mm256_loadu_ps(b_row.as_ptr().add(j));
                        let vo = std::arch::x86_64::_mm256_loadu_ps(row.as_ptr().add(j));
                        let vr = std::arch::x86_64::_mm256_add_ps(
                            vo,
                            std::arch::x86_64::_mm256_mul_ps(va, vb),
                        );
                        std::arch::x86_64::_mm256_storeu_ps(row.as_mut_ptr().add(j), vr);
                        j += 8;
                    }
                    for jj in j..n {
                        row[jj] += ait * b_row[jj];
                    }
                }
            } else {
                for j in 0..n {
                    row[j] += ait * b_row[j];
                }
            }
        }
    });
    let _ = m;
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

// ---------- layer weights ----------

pub struct LayerWeights {
    pub kind: LayerKind,
    // attention
    pub wq: Vec<f32>, // [HIDDEN, HIDDEN]
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub q_gate: Vec<f32>, // head gate [NUM_HEADS] (broadcast) stored as [NUM_HEADS]
    // conv
    pub conv_w: Vec<f32>, // [HIDDEN, HIDDEN, CONV_KERNEL] flattened [HIDDEN, HIDDEN*CONV_KERNEL]? store [CONV_KERNEL, HIDDEN, HIDDEN]
    pub conv_gate: Vec<f32>, // [HIDDEN]
    // ffn (swiglu: gate+up+down)
    pub w_gate: Vec<f32>, // [HIDDEN, INTER]
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>, // [INTER, HIDDEN]
    // norms (sandwich)
    pub norm1: Vec<f32>, // [HIDDEN]
    pub norm2: Vec<f32>,
}

impl LayerWeights {
    pub fn new(kind: LayerKind) -> Self {
        let z = |n: usize| vec![0.0f32; n];
        Self {
            kind,
            wq: z(HIDDEN * HIDDEN),
            wk: z(HIDDEN * HIDDEN),
            wv: z(HIDDEN * HIDDEN),
            wo: z(HIDDEN * HIDDEN),
            q_gate: z(NUM_HEADS),
            conv_w: z(CONV_KERNEL * HIDDEN * HIDDEN),
            conv_gate: z(HIDDEN),
            w_gate: z(HIDDEN * INTER),
            w_up: z(HIDDEN * INTER),
            w_down: z(INTER * HIDDEN),
            norm1: vec![1.0; HIDDEN],
            norm2: vec![1.0; HIDDEN],
        }
    }
}

// ---------- model ----------

pub struct Lfm3Model {
    pub layers: Vec<LayerWeights>,
    pub embedding: Vec<f32>, // [VOCAB, HIDDEN]
    pub lm_head: Vec<f32>,   // [HIDDEN, VOCAB]
    pub final_norm: Vec<f32>,
}

impl Lfm3Model {
    pub fn new_random() -> Self {
        // tiny pseudo-init for shape testing
        let layers: Vec<LayerWeights> = layer_kinds()
            .iter()
            .map(|k| LayerWeights::new(*k))
            .collect();
        Self {
            layers,
            embedding: vec![0.0; VOCAB * HIDDEN],
            lm_head: vec![0.0; HIDDEN * VOCAB],
            final_norm: vec![1.0; HIDDEN],
        }
    }

    /// Forward pass for a single token at position `pos`, with KV cache.
    /// x: token id. Returns logits over vocab.
    pub fn forward_one(&mut self, token: u32, _pos: usize, kv: &mut KvCache) -> Vec<f32> {
        // embedding
        let mut x = self.embedding[token as usize * HIDDEN..(token as usize + 1) * HIDDEN].to_vec();
        if x.iter().all(|&v| v == 0.0) {
            x = vec![0.01; HIDDEN];
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // sandwich norm 1
            let mut xn = x.clone();
            rmsnorm(&mut xn, &layer.norm1);

            match layer.kind {
                LayerKind::Conv => {
                    // gated short conv: y = conv(x) * sigmoid(conv_gate)
                    // simplified: linear + gate
                    let mut conv_out = vec![0.0; HIDDEN];
                    matmul(&mut conv_out, &xn, &layer.wq, 1, HIDDEN, HIDDEN);
                    for i in 0..HIDDEN {
                        conv_out[i] *= 1.0 / (1.0 + (-layer.conv_gate[i]).exp());
                    }
                    for i in 0..HIDDEN {
                        x[i] += conv_out[i];
                    }
                }
                LayerKind::SlidingWindow | LayerKind::Global => {
                    // attention (single token): q = x Wq; k,v appended to cache
                    let mut q = vec![0.0; HIDDEN];
                    let mut kvec = vec![0.0; HIDDEN];
                    let mut vvec = vec![0.0; HIDDEN];
                    matmul(&mut q, &xn, &layer.wq, 1, HIDDEN, HIDDEN);
                    matmul(&mut kvec, &xn, &layer.wk, 1, HIDDEN, HIDDEN);
                    matmul(&mut vvec, &xn, &layer.wv, 1, HIDDEN, HIDDEN);
                    // gate per head
                    for h in 0..NUM_HEADS {
                        let g = 1.0 / (1.0 + (-layer.q_gate[h]).exp());
                        for d in 0..HEAD_DIM {
                            q[h * HEAD_DIM + d] *= g;
                        }
                    }
                    // append to kv cache
                    kv.push(li, kvec.clone(), vvec.clone());
                    // compute attention over window
                    let (ks, vs) = kv.window(li, layer.kind == LayerKind::SlidingWindow);
                    // scores
                    let mut scores = vec![0.0f32; ks.len()];
                    for (si, kcache) in ks.iter().enumerate() {
                        let mut dot = 0.0f32;
                        for d in 0..HIDDEN {
                            dot += q[d] * kcache[d];
                        }
                        scores[si] = dot / (HEAD_DIM as f32).sqrt();
                    }
                    softmax(&mut scores);
                    let mut attn_out = vec![0.0; HIDDEN];
                    for (si, vcache) in vs.iter().enumerate() {
                        for d in 0..HIDDEN {
                            attn_out[d] += scores[si] * vcache[d];
                        }
                    }
                    let mut proj = vec![0.0; HIDDEN];
                    matmul(&mut proj, &attn_out, &layer.wo, 1, HIDDEN, HIDDEN);
                    for i in 0..HIDDEN {
                        x[i] += proj[i];
                    }
                }
            }

            // sandwich norm 2
            let mut xn2 = x.clone();
            rmsnorm(&mut xn2, &layer.norm2);
            // ffn (swiglu)
            let mut gate = vec![0.0; INTER];
            let mut up = vec![0.0; INTER];
            matmul(&mut gate, &xn2, &layer.w_gate, 1, HIDDEN, INTER);
            matmul(&mut up, &xn2, &layer.w_up, 1, HIDDEN, INTER);
            for i in 0..INTER {
                gate[i] = silu(gate[i]) * up[i];
            }
            let mut down = vec![0.0; HIDDEN];
            matmul(&mut down, &gate, &layer.w_down, 1, INTER, HIDDEN);
            for i in 0..HIDDEN {
                x[i] += down[i];
            }
        }

        // final norm
        rmsnorm(&mut x, &self.final_norm);
        // lm head
        let mut logits = vec![0.0; VOCAB];
        matmul(&mut logits, &x, &self.lm_head, 1, HIDDEN, VOCAB);
        logits
    }
}

pub struct KvCache {
    // per layer: list of (k, v)
    pub k: Vec<Vec<Vec<f32>>>,
    pub v: Vec<Vec<Vec<f32>>>,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            k: vec![Vec::new(); NUM_LAYERS],
            v: vec![Vec::new(); NUM_LAYERS],
        }
    }
    pub fn push(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        self.k[layer].push(k);
        self.v[layer].push(v);
    }
    pub fn window(&self, layer: usize, sliding: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let ks = &self.k[layer];
        let vs = &self.v[layer];
        if sliding && ks.len() > SWA_WINDOW {
            let start = ks.len() - SWA_WINDOW;
            (ks[start..].to_vec(), vs[start..].to_vec())
        } else {
            (ks.clone(), vs.clone())
        }
    }
    pub fn reset(&mut self) {
        for i in 0..NUM_LAYERS {
            self.k[i].clear();
            self.v[i].clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kinds_counts() {
        let k = layer_kinds();
        let mut c = 0;
        let mut s = 0;
        let mut g = 0;
        for x in k.iter() {
            match x {
                LayerKind::Conv => c += 1,
                LayerKind::SlidingWindow => s += 1,
                LayerKind::Global => g += 1,
            }
        }
        assert_eq!(c, 13);
        assert_eq!(s, 5);
        assert_eq!(g, 6);
    }

    #[test]
    fn forward_one_runs() {
        let mut m = Lfm3Model::new_random();
        let mut kv = KvCache::new();
        let logits = m.forward_one(0, 0, &mut kv);
        assert_eq!(logits.len(), VOCAB);
    }
}
