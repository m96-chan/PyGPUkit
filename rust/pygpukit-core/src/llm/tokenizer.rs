//! Simple BPE tokenizer for GPT-2 style models
//!
//! Loads tokenizer.json format and provides basic encode/decode functionality.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Error type for tokenizer operations
#[derive(Debug)]
pub enum TokenizerError {
    IoError(std::io::Error),
    ParseError(String),
    InvalidToken(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::IoError(e) => write!(f, "IO error: {}", e),
            TokenizerError::ParseError(e) => write!(f, "Parse error: {}", e),
            TokenizerError::InvalidToken(t) => write!(f, "Invalid token: {}", t),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::IoError(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::ParseError(e.to_string())
    }
}

/// GPT-2 style tokenizer.json model section
#[derive(Debug, Deserialize)]
struct TokenizerModel {
    #[serde(rename = "type")]
    model_type: Option<String>,
    vocab: HashMap<String, u32>,
    merges: Option<Vec<String>>,
}

/// GPT-2 style tokenizer.json added_tokens section
#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

/// GPT-2 style tokenizer.json format
#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

/// BPE merge rule
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BpePair(String, String);

/// Simple BPE tokenizer
pub struct Tokenizer {
    /// Token string to ID mapping
    encoder: HashMap<String, u32>,
    /// ID to token string mapping
    decoder: HashMap<u32, String>,
    /// BPE merge rules (pair -> rank, lower is earlier)
    bpe_ranks: HashMap<BpePair, usize>,
    /// Special tokens
    special_tokens: HashMap<String, u32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl Tokenizer {
    /// Load tokenizer from tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tokenizer_json: TokenizerJson = serde_json::from_reader(reader)?;

        Self::from_json(tokenizer_json)
    }

    /// Load tokenizer from JSON string
    pub fn from_json_str(json: &str) -> Result<Self, TokenizerError> {
        let tokenizer_json: TokenizerJson = serde_json::from_str(json)?;
        Self::from_json(tokenizer_json)
    }

    fn from_json(json: TokenizerJson) -> Result<Self, TokenizerError> {
        let encoder = json.model.vocab;
        let mut decoder: HashMap<u32, String> = HashMap::new();

        for (token, id) in &encoder {
            decoder.insert(*id, token.clone());
        }

        // Parse BPE merges
        let mut bpe_ranks = HashMap::new();
        if let Some(merges) = json.model.merges {
            for (rank, merge) in merges.iter().enumerate() {
                let parts: Vec<&str> = merge.split(' ').collect();
                if parts.len() == 2 {
                    let pair = BpePair(parts[0].to_string(), parts[1].to_string());
                    bpe_ranks.insert(pair, rank);
                }
            }
        }

        // Process added tokens (special tokens)
        let mut special_tokens = HashMap::new();
        for added in json.added_tokens {
            if added.special {
                special_tokens.insert(added.content.clone(), added.id);
            }
            // Also add to encoder/decoder
            if !encoder.contains_key(&added.content) {
                decoder.insert(added.id, added.content);
            }
        }

        let vocab_size = encoder.len();

        Ok(Tokenizer {
            encoder,
            decoder,
            bpe_ranks,
            special_tokens,
            vocab_size,
        })
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get BOS token ID if available
    pub fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<|endoftext|>").copied()
            .or_else(|| self.special_tokens.get("<s>").copied())
    }

    /// Get EOS token ID if available
    pub fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<|endoftext|>").copied()
            .or_else(|| self.special_tokens.get("</s>").copied())
    }

    /// Get PAD token ID if available
    pub fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<|padding|>").copied()
            .or_else(|| self.special_tokens.get("<pad>").copied())
    }

    /// Convert bytes to unicode representation (GPT-2 style)
    fn byte_to_unicode() -> HashMap<u8, char> {
        let mut byte_encoder: HashMap<u8, char> = HashMap::new();
        let mut n = 0u32;

        // Printable ASCII range and some extended chars
        for b in 33u8..=126 {
            byte_encoder.insert(b, char::from_u32(b as u32).unwrap());
        }
        for b in 161u8..=172 {
            byte_encoder.insert(b, char::from_u32(b as u32).unwrap());
        }
        for b in 174u8..=255 {
            byte_encoder.insert(b, char::from_u32(b as u32).unwrap());
        }

        // Map remaining bytes to unicode codepoints starting at 256
        for b in 0u8..=255 {
            if !byte_encoder.contains_key(&b) {
                byte_encoder.insert(b, char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }

        byte_encoder
    }

    /// Convert unicode back to bytes
    fn unicode_to_byte() -> HashMap<char, u8> {
        let byte_encoder = Self::byte_to_unicode();
        byte_encoder.into_iter().map(|(k, v)| (v, k)).collect()
    }

    /// Get consecutive pairs from a list of symbols
    fn get_pairs(word: &[String]) -> Vec<BpePair> {
        let mut pairs = Vec::new();
        for i in 0..word.len().saturating_sub(1) {
            pairs.push(BpePair(word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    /// Apply BPE to a word
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return vec![];
        }

        // Convert token to unicode chars (GPT-2 byte encoding)
        let byte_encoder = Self::byte_to_unicode();
        let word: Vec<String> = token
            .bytes()
            .map(|b| byte_encoder.get(&b).unwrap_or(&'?').to_string())
            .collect();

        if word.len() == 1 {
            return word;
        }

        let mut word = word;

        loop {
            let pairs = Self::get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            // Find the pair with lowest rank (highest priority)
            let best_pair = pairs
                .iter()
                .filter_map(|p| self.bpe_ranks.get(p).map(|r| (p, r)))
                .min_by_key(|(_, r)| *r);

            let Some((bigram, _)) = best_pair else {
                break;
            };

            // Merge the best pair
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                // Find next occurrence of first element of bigram
                let j = word[i..].iter().position(|s| *s == bigram.0);

                if let Some(j) = j {
                    new_word.extend(word[i..i + j].iter().cloned());
                    i += j;

                    if i < word.len() - 1 && word[i] == bigram.0 && word[i + 1] == bigram.1 {
                        // Merge
                        new_word.push(format!("{}{}", bigram.0, bigram.1));
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                } else {
                    new_word.extend(word[i..].iter().cloned());
                    break;
                }
            }

            word = new_word;

            if word.len() == 1 {
                break;
            }
        }

        word
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Simple word-level tokenization (split on whitespace and punctuation)
        // GPT-2 uses a more complex regex, but this is MVP
        let words = Self::simple_tokenize(text);

        for word in words {
            let bpe_tokens = self.bpe(&word);
            for bpe_token in bpe_tokens {
                if let Some(&id) = self.encoder.get(&bpe_token) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// Simple tokenization (split on whitespace, keep leading space)
    fn simple_tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current);
                    current = String::new();
                }
                // GPT-2 encodes space as part of next token
                current.push(ch);
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let unicode_to_byte = Self::unicode_to_byte();

        let mut bytes = Vec::new();

        for &id in token_ids {
            if let Some(token) = self.decoder.get(&id) {
                for ch in token.chars() {
                    if let Some(&b) = unicode_to_byte.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Get token string for an ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.decoder.get(&id).map(|s| s.as_str())
    }

    /// Get ID for a token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_to_unicode_coverage() {
        let byte_encoder = Tokenizer::byte_to_unicode();
        // Should have mapping for all 256 bytes
        assert_eq!(byte_encoder.len(), 256);
    }

    #[test]
    fn test_unicode_to_byte_inverse() {
        let byte_encoder = Tokenizer::byte_to_unicode();
        let unicode_decoder = Tokenizer::unicode_to_byte();

        for (b, c) in &byte_encoder {
            assert_eq!(unicode_decoder.get(c), Some(b));
        }
    }
}
