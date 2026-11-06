use voltraxis::{Lfm3Config, quant};

fn main() {
    let cfg = Lfm3Config::lfm3_1_7b();
    println!("Voltraxis core");
    println!("Model: LFM3-1.7B");
    println!("Hidden size: {}", cfg.hidden_size);
    println!("Layers: {}", cfg.num_layers);
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Context length: {}", cfg.context_length);
    println!("Quantization: {}", cfg.quantization);

    let vals: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    let q = quant::quantize_affine(&vals, quant::DEFAULT_GROUP_SIZE);
    let dq = quant::dequantize(&q);
    let max_err = vals
        .iter()
        .zip(&dq)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "Quantized {} values into {} bytes (max_err={})",
        vals.len(),
        q.packed.len(),
        max_err
    );
}
