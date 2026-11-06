/// LFM3 model configuration (scaffold).
#[derive(Debug, Clone)]
pub struct Lfm3Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub quantization: &'static str,
}

impl Lfm3Config {
    /// LFM3-1.7B text model config.
    pub fn lfm3_1_7b() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 24,
            vocab_size: 65536,
            context_length: 256 * 1024,
            quantization: "4-bit affine (group 64)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_sane() {
        let c = Lfm3Config::lfm3_1_7b();
        assert!(c.hidden_size > 0);
        assert!(c.num_layers > 0);
        assert!(c.context_length >= 1024);
    }
}
