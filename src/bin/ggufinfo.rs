use std::env;
use voltraxis::gguf;

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/opencode/models/LFM3-1.7B-Q4_K_M.gguf".to_string());
    gguf::print_summary(&path);
}
