//! Voltraxis benchmark: LFM3-1.7B-shaped synthetic workload.

use std::time::Instant;

const HIDDEN: usize = 2048;
const INTER: usize = 5632;
const LAYERS: usize = 24;
const VOCAB: usize = 65536;
const N_DECODE: usize = 4;

use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_matmul_avx2(row: &mut [f32], a_row: &[f32], b: &[f32], k: usize, n: usize) {
    for t in 0..k {
        let ait = a_row[t];
        let va = _mm256_set1_ps(ait);
        let b_row = &b[t * n..];
        let mut j = 0usize;
        while j + 8 <= n {
            let vb = _mm256_loadu_ps(b_row.as_ptr().add(j));
            let vo = _mm256_loadu_ps(row.as_ptr().add(j));
            let vr = _mm256_add_ps(vo, _mm256_mul_ps(va, vb));
            _mm256_storeu_ps(row.as_mut_ptr().add(j), vr);
            j += 8;
        }
        for jj in j..n {
            row[jj] += ait * b_row[jj];
        }
    }
}

fn row_matmul_scalar(row: &mut [f32], a_row: &[f32], b: &[f32], k: usize, n: usize) {
    for t in 0..k {
        let ait = a_row[t];
        let b_row = &b[t * n..];
        for j in 0..n {
            row[j] += ait * b_row[j];
        }
    }
}

fn matmul(out: &mut [f32], a: &[f32], b: &[f32], _m: usize, k: usize, n: usize) {
    out.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k..(i + 1) * k];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                row_matmul_avx2(row, a_row, b, k, n);
            }
        } else {
            row_matmul_scalar(row, a_row, b, k, n);
        }
    });
}

fn rmsnorm(x: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let rms = (ss / n as f32 + 1e-6).sqrt();
    for v in x.iter_mut() {
        *v /= rms;
    }
}

struct LayerWeights {
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

fn make_layer(seed: usize) -> LayerWeights {
    let mk = |rows: usize, cols: usize, off: usize| -> Vec<f32> {
        let n = rows * cols;
        (0..n)
            .map(|i| ((i + off + seed * 97) as f32 * 0.001).sin() * 0.02)
            .collect()
    };
    LayerWeights {
        wqkv: mk(HIDDEN, 3 * HIDDEN, 1),
        wo: mk(HIDDEN, HIDDEN, 2),
        w1: mk(HIDDEN, INTER, 3),
        w2: mk(INTER, HIDDEN, 4),
    }
}

fn forward_one(x: &mut [f32], layers: &[LayerWeights], lm_head: &[f32]) {
    let mut buf = vec![0.0f32; 3 * HIDDEN];
    let mut attn = vec![0.0f32; HIDDEN];
    let mut ffn = vec![0.0f32; INTER];
    let mut ffn2 = vec![0.0f32; HIDDEN];

    for l in layers {
        let residual: Vec<f32> = x.to_vec();
        rmsnorm(x);
        matmul(&mut buf, x, &l.wqkv, 1, HIDDEN, 3 * HIDDEN);
        for i in 0..HIDDEN {
            attn[i] = buf[i];
        }
        let attn_in: Vec<f32> = attn[..HIDDEN].to_vec();
        matmul(&mut attn, &attn_in, &l.wo, 1, HIDDEN, HIDDEN);
        for i in 0..HIDDEN {
            x[i] = residual[i] + attn[i];
        }
        let residual2: Vec<f32> = x.to_vec();
        rmsnorm(x);
        matmul(&mut ffn, x, &l.w1, 1, HIDDEN, INTER);
        for v in ffn.iter_mut() {
            *v = v.max(0.0);
        }
        matmul(&mut ffn2, &ffn, &l.w2, 1, INTER, HIDDEN);
        for i in 0..HIDDEN {
            x[i] = residual2[i] + ffn2[i];
        }
    }
    let mut logits = vec![0.0f32; VOCAB];
    matmul(&mut logits, x, lm_head, 1, HIDDEN, VOCAB);
    let _ = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
}

fn main() {
    println!("Voltraxis bench - LFM3-1.7B shaped workload");
    println!(
        "hidden={}, inter={}, layers={}, vocab={}",
        HIDDEN, INTER, LAYERS, VOCAB
    );

    let t0 = Instant::now();
    let layers: Vec<LayerWeights> = (0..LAYERS).map(make_layer).collect();
    let lm_head: Vec<f32> = (0..HIDDEN * VOCAB)
        .map(|i| ((i as f32) * 0.0001).sin() * 0.01)
        .collect();
    let load_ms = t0.elapsed().as_millis();
    println!("Weights loaded+quantized in {} ms", load_ms);

    let mut x = vec![0.1f32; HIDDEN];
    forward_one(&mut x, &layers, &lm_head);

    let t1 = Instant::now();
    for _ in 0..N_DECODE {
        let mut x = vec![0.1f32; HIDDEN];
        forward_one(&mut x, &layers, &lm_head);
    }
    let dec_elapsed = t1.elapsed().as_secs_f64();
    let dec_toks = N_DECODE as f64 / dec_elapsed;
    println!(
        "DECODE: {} tokens in {:.3}s => {:.2} tok/s",
        N_DECODE, dec_elapsed, dec_toks
    );

    for &plen in &[4usize, 8, 16, 32] {
        let t2 = Instant::now();
        for _ in 0..plen {
            let mut x = vec![0.1f32; HIDDEN];
            forward_one(&mut x, &layers, &lm_head);
        }
        let pre_elapsed = t2.elapsed().as_secs_f64();
        let pre_toks = plen as f64 / pre_elapsed;
        println!(
            "PREFILL (prompt len {}): {} tokens in {:.3}s => {:.2} tok/s",
            plen, plen, pre_elapsed, pre_toks
        );
    }

    println!("---");
    println!("RESULT decode_tok_s={:.2}", dec_toks);
}
