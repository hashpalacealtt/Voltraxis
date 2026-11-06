# Voltraxis — Token/sec Data

This document tracks the measured throughput (tokens/sec) of the Voltraxis
engine running an LFM3-1.7B-shaped workload. It is updated as the engine
evolves.

## Methodology

- **Model shape:** LFM3-1.7B (hidden=2048, intermediate=5632, 24 layers,
  vocab=65536, 4-bit affine quantization, group size 64).
- **Workload:** synthetic forward pass over the model dimensions using
  4-bit-quantized weights (quantize -> dequantize -> f32 matmul), executed
  in release mode with rayon parallelism.
- **Prompts:** throughput is measured for decode (single-token steps) and
  prefill at several prompt lengths (4, 8, 16, 32 tokens) to cover
  different prompt sizes.
- **Environment:** Ubuntu (GitHub Actions `ubuntu-latest`), Rust stable,
  release profile.

## Results

### Run 1 — 2026-11-06 (initial baseline)

| Metric | Value |
|---|---|
| Weights load + quantize | ~14.3 s |
| Decode (4 tokens) | 0.79 tok/s |
| Prefill — prompt len 4 | 0.79 tok/s |
| Prefill — prompt len 8 | 0.78 tok/s |
| Prefill — prompt len 16 | 0.77 tok/s |
| Prefill — prompt len 32 | 0.78 tok/s |

Raw output:

```
Voltraxis bench - LFM3-1.7B shaped workload
hidden=2048, inter=5632, layers=24, vocab=65536
Weights loaded+quantized in 14309 ms
DECODE: 4 tokens in 5.089s => 0.79 tok/s
PREFILL (prompt len 4): 4 tokens in 5.078s => 0.79 tok/s
PREFILL (prompt len 8): 8 tokens in 10.226s => 0.78 tok/s
PREFILL (prompt len 16): 16 tokens in 20.846s => 0.77 tok/s
PREFILL (prompt len 32): 32 tokens in 40.857s => 0.78 tok/s
RESULT decode_tok_s=0.79
```

## Notes

- Throughput is dominated by the `lm_head` projection (2048 x 65536) and
  the per-layer FFN matmuls. The current path is a naive (cache-friendly,
  rayon-parallel) triple loop without SIMD or BLAS.
- Next optimization targets: SIMD (std::simd / packed f32), blocking/tiled
  matmul, and avoiding redundant dequantization (keep weights quantized
  and dequantize inline in the matmul kernel).
- Real LFM3 weight loading (GGUF / safetensors) and a full transformer
  forward pass are the next milestones; this benchmark uses a synthetic
  LFM3-shaped workload until then.
