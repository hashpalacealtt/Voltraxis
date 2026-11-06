# Voltraxis — Token/sec Data

This document tracks the measured throughput (tokens/sec) of the Voltraxis
engine running real LFM3-1.7B (Q4_K_M). Synthetic (LFM3-shaped) runs are
also kept for reference. It is updated as the engine evolves.

## Real LFM3 — Q4_K_M (measured on GitHub Actions, ubuntu-latest)

Weights: `LiquidAI/LFM3-1.7B-GGUF` Q4_K_M (~1.05 GB), loaded via the GGUF
parser + Q4_K dequantizer. Architecture: 24 tri-hybrid layers (13 gated
short conv, 5 SWA, 6 global attention), vocab 128K. Greedy decode (argmax),
8 generated tokens per prompt, prompts = fixed token-id sequences of
varying length (no tokenizer yet).

| Metric | Value |
|---|---|
| Weights load (Q4_K -> f32) | 7.55 s |
| Decode (8 tokens) | **0.41 - 0.42 tok/s** |
| Prefill — prompt len 1 | 3.15 s (3147 ms) |
| Prefill — prompt len 4 | 9.63 s (9630 ms) |
| Prefill — prompt len 8 | 19.28 s (19280 ms) |
| Prefill — prompt len 16 | 38.52 s (38516 ms) |

Raw output (from CI artifact `lfm3-bench-output`):

```
Loading weights from /tmp/models/LFM3-1.7B-Q4_K_M.gguf
Loaded in 7.55s
PROMPT 0 (len 1): prefill 3147ms, decode 8 tokens in 19.313s => 0.41 tok/s
PROMPT 1 (len 4): prefill 9630ms, decode 8 tokens in 19.265s => 0.42 tok/s
PROMPT 2 (len 8): prefill 19280ms, decode 8 tokens in 19.282s => 0.41 tok/s
PROMPT 3 (len 16): prefill 38516ms, decode 8 tokens in 19.285s => 0.41 tok/s
```

Notes:
- This is the **first real LFM3 measurement** (not synthetic).
- Prefill scales linearly with prompt length (single-token-at-a-time path).
- Decode is dominated by the `lm_head` (2048 x 128000) and Q4_K dequantization.
- Next: batched prefill, KV cache reuse, inlined dequant, SIMD on the real path.

---

## Real LFM3 — Q4_K_M with real text prompts (BPE tokenizer)

Weights: `LiquidAI/LFM3-1.7B-GGUF` Q4_K_M. Tokenizer: BPE loaded from GGUF
metadata (`tokenizer.ggml.*`). Prompts are real English sentences; output is
decoded back to text. Greedy decode, 8 generated tokens per prompt.

| Metric | Value |
|---|---|
| Weights load (Q4_K -> f32) | 9.17 s |
| Decode (8 tokens) | **0.37 tok/s** |
| Prefill — "Hello" (1 token) | 3.43 s |
| Prefill — "What is the capital of France?" (13 tokens) | 35.17 s |
| Prefill — "Explain quantum computing..." (14 tokens) | 37.88 s |
| Prefill — "Write a short poem..." (16 tokens) | 43.31 s |

Raw output:

```
Loading weights from /tmp/models/LFM3-1.7B-Q4_K_M.gguf
Loaded in 9.17s
Tokenizer loaded: vocab=128000
PROMPT 0: "Hello" (1 tokens) prefill 3426ms, decode 8 tokens in 21.642s => 0.37 tok/s
  output: "ĊĊĊĊ..."
PROMPT 1: "What is the capital of France?" (13 tokens) prefill 35168ms, decode 8 tokens in 21.635s => 0.37 tok/s
PROMPT 2: "Explain quantum computing in simple terms." (14 tokens) prefill 37879ms, decode 8 tokens in 21.636s => 0.37 tok/s
PROMPT 3: "Write a short poem about the ocean." (16 tokens) prefill 43311ms, decode 8 tokens in 21.654s => 0.37 tok/s
```

Notes:
- First run with a real BPE tokenizer and real text prompts.
- Output is degenerate (repeated newlines) because several weight tensors
  (conv/attention/FFN) are not yet mapped from the GGUF (names differ from
  the assumed `blk.N.*` pattern); only lm_head (Q4_K direct) is wired.
  Mapping the remaining tensors is the next step.
- Decode throughput unchanged at ~0.37 tok/s (lm_head dominated).

---

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

### Synthetic Run 1 — 2026-11-06 (initial baseline, NOT real LFM3)

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

### Synthetic Run 2 — 2026-11-06 (AVX2 SIMD, NOT real LFM3)

Optimization: AVX2 intrinsics (`_mm256_*`) in the matmul hot loop, rayon
parallelism over rows. ~3.5x speedup over the scalar baseline.

| Metric | Value |
|---|---|
| Weights load + quantize | ~15.2 s |
| Decode (4 tokens) | 2.75 tok/s |
| Prefill — prompt len 4 | 2.67 tok/s |
| Prefill — prompt len 8 | 2.71 tok/s |
| Prefill — prompt len 16 | 2.70 tok/s |
| Prefill — prompt len 32 | 2.71 tok/s |

Raw output:

```
Voltraxis bench - LFM3-1.7B shaped workload
hidden=2048, inter=5632, layers=24, vocab=65536
Weights loaded+quantized in 15242 ms
DECODE: 4 tokens in 1.456s => 2.75 tok/s
PREFILL (prompt len 4): 4 tokens in 1.500s => 2.67 tok/s
PREFILL (prompt len 8): 8 tokens in 2.948s => 2.71 tok/s
PREFILL (prompt len 16): 16 tokens in 5.929s => 2.70 tok/s
PREFILL (prompt len 32): 32 tokens in 11.810s => 2.71 tok/s
RESULT decode_tok_s=2.75
```

### Comparison

| Version | Decode tok/s | Speedup |
|---|---|---|
| Run 1 — scalar naive | 0.79 | 1.0x |
| Run 2 — AVX2 SIMD | 2.75 | 3.5x |

### Synthetic Run 3 — 2026-11-06 (AVX2 + tiling, NOT real LFM3)

Optimization: tiling of the `t` (k) loop in blocks of 64 to keep `row`
hot in L1. No measurable gain — the per-row working set (row = 8 KB)
already fits in L1, so tiling `t` alone does not help. Numbers are within
run-to-run noise of Run 2.

| Metric | Value |
|---|---|
| Weights load + quantize | ~14.9 s |
| Decode (4 tokens) | 2.56 tok/s |
| Prefill — prompt len 4 | 2.75 tok/s |
| Prefill — prompt len 8 | 2.68 tok/s |
| Prefill — prompt len 16 | 2.72 tok/s |
| Prefill — prompt len 32 | 2.69 tok/s |

Raw output:

```
Voltraxis bench - LFM3-1.7B shaped workload
hidden=2048, inter=5632, layers=24, vocab=65536
Weights loaded+quantized in 14926 ms
DECODE: 4 tokens in 1.565s => 2.56 tok/s
PREFILL (prompt len 4): 4 tokens in 1.456s => 2.75 tok/s
PREFILL (prompt len 8): 8 tokens in 2.983s => 2.68 tok/s
PREFILL (prompt len 16): 16 tokens in 5.875s => 2.72 tok/s
PREFILL (prompt len 32): 32 tokens in 11.896s => 2.69 tok/s
RESULT decode_tok_s=2.56
```

### Comparison

| Version | Decode tok/s | Speedup vs Run 1 |
|---|---|---|
| Run 1 — scalar naive | 0.79 | 1.0x |
| Run 2 — AVX2 SIMD | 2.75 | 3.5x |
| Run 3 — AVX2 + tiling (t) | 2.56 | 3.2x |

## Notes

- Throughput is dominated by the `lm_head` projection (2048 x 65536) and
  the per-layer FFN matmuls.
- Run 2 added AVX2 SIMD (`_mm256_*`) to the matmul hot loop: 3.5x speedup
  (0.79 -> 2.75 tok/s).
- Run 3 added tiling of the `t` (k) loop (TILE_T=64): no measurable gain,
  because the per-row working set (8 KB) already fits in L1. Tiling `t`
  alone is ineffective.
- Next optimization targets: tiling over `j` for the `lm_head` (cache-block
  the 256 KB b-slice), inlining dequantization into the matmul kernel
  (avoid the full dequant pass), and Q4 x f32 direct matmul (no
  dequantization at all).
- Real LFM3 weight loading (GGUF / safetensors) and a full transformer
  forward pass are the next milestones; this benchmark uses a synthetic
  LFM3-shaped workload until then.
