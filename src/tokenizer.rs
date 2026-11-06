//! Minimal BPE tokenizer loaded from GGUF metadata.

use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: HashMap<(String, String), usize>, // rank
    pub bos: u32,
    pub eos: u32,
}

impl Tokenizer {
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> Option<Self> {
        // find tokens array
        let tokens_val = &g
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.tokens")?
            .1;
        let scores_val = g
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.scores");
        let merges_val = g
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.merges");
        let bos = g
            .metadata
            .iter()
            .find_map(|(k, v)| {
                if k == "tokenizer.ggml.bos_token_id" {
                    match v {
                        crate::gguf::GgufValue::U32(x) => Some(*x),
                        crate::gguf::GgufValue::I32(x) => Some(*x as u32),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or(1);
        let eos = g
            .metadata
            .iter()
            .find_map(|(k, v)| {
                if k == "tokenizer.ggml.eos_token_id" {
                    match v {
                        crate::gguf::GgufValue::U32(x) => Some(*x),
                        crate::gguf::GgufValue::I32(x) => Some(*x as u32),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or(2);

        let vocab: Vec<String> = match tokens_val {
            crate::gguf::GgufValue::Array(arr) => arr
                .iter()
                .filter_map(|v| match v {
                    crate::gguf::GgufValue::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => return None,
        };
        let mut token_to_id = HashMap::new();
        for (i, t) in vocab.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merges = HashMap::new();
        if let Some(mv) = merges_val {
            if let crate::gguf::GgufValue::Array(arr) = &mv.1 {
                for (rank, v) in arr.iter().enumerate() {
                    if let crate::gguf::GgufValue::Str(s) = v {
                        // format "a b"
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
                        }
                    }
                }
            }
        }
        let _ = scores_val;
        Some(Tokenizer {
            vocab,
            token_to_id,
            merges,
            bos,
            eos,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // BPE: start with chars, apply best merge repeatedly
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        // apply merges
        loop {
            // find best merge
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&r) = self.merges.get(&(tokens[i].clone(), tokens[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            match best_idx {
                Some(i) => {
                    let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }
        // map to ids (fallback: try whole token, else 0)
        tokens
            .iter()
            .map(|t| *self.token_to_id.get(t).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.vocab.get(id as usize).cloned().unwrap_or_default())
            .collect()
    }
}
