# Voltraxis

![Language: Rust](https://img.shields.io/badge/language-Rust-orange?logo=rust&logoColor=white)
![Stars](https://img.shields.io/github/stars/hashpalacealtt/Voltraxis?style=social)
![Contributors](https://img.shields.io/github/contributors/hashpalacealtt/Voltraxis)
![License](https://img.shields.io/badge/license-Voltraxis%20(restrictive)-red)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

> **Voltraxis** is an AI engine built for raw power: running the **LFM3** model at
> shocking speed, without altering the model, in its **4-bit** quantized form.

## Contribution graph

![Contribution graph](https://github-readme-activity-graph.vercel.app/graph?username=hashpalacealtt&theme=github-compact&hide_border=true)

---

## 🎯 Goal

Run **LFM3** (Liquid Foundation Models 3, by Liquid AI) at maximum inference
speed, **without modifying or degrading the model**, using the official
**4-bit** quantized weights.

- **No alteration** of the model weights or architecture.
- **4-bit** precision (affine quantization, groups of 64) for minimal memory
  footprint (~1 GB for LFM3-1.7B).
- **Shocking speed**: target throughput far beyond reference runtimes, while
  preserving output fidelity bit-for-bit against the quantized reference.
- Built in **Rust** for memory safety, zero-cost abstractions, and bare-metal
  performance.

## 🧠 About LFM3

LFM3 is Liquid AI's third generation of Liquid Foundation Models, released
October 27, 2026. It is a family of six open-weight, ~2B-parameter models
sharing a single tri-hybrid backbone trained on 22T tokens:

- **Text** (LFM3-1.7B) — 256K context window, agentic post-training
- **Vision-Language** (LFM3-VL-2B) — image & video understanding, 128K context
- **Audio** — speech and audio understanding
- **Omni** (LFM3-Omni-2B) — vision + audio + language in one model
- **Decision** models — for agentic / tool-use scenarios

At 4-bit precision the text model fits in ~1 GB, making it runnable on a
phone or laptop. Voltraxis aims to make that run **fast**.

References:
- Liquid AI blog: https://www.liquid.ai/blog/lfm3
- Hugging Face: https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit
- Docs: https://docs.liquid.ai/lfm/models/lfm3-1.7b

## 📊 Performance

See [DATA.md](./DATA.md) for measured tokens/sec benchmarks.

## 🚧 Status

**In development.** This README is evolving — sections will be filled in as
the engine matures.

Planned milestones:

- [ ] Core runtime scaffold (Rust)
- [ ] 4-bit weight loader (affine, group size 64)
- [ ] LFM3-1.7B text inference path
- [ ] Benchmark vs. reference runtime
- [ ] Vision / Omni / Audio modalities
- [ ] On-device (phone) target

## 🛠️ Tech stack

- **Language:** Rust
- **Target:** LFM3 4-bit (MLX-4bit compatible weights)
- **Goal:** zero-alteration, max-throughput inference

## 🤝 Contributing

Contributions are welcome via pull request. Please note the project is
governed by the restrictive **Voltraxis License**: modifications are only
accepted through the official contribution workflow, and commercial use is
not permitted. See [LICENSE](./LICENSE).

## 📄 License

This project is licensed under the **Voltraxis License** (restrictive).
See [LICENSE](./LICENSE) for full terms.

---

*This document is a living document and will evolve as Voltraxis grows.*
