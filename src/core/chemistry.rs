// ALPS-SLN Protocol — Chemistry and QuerySignature
//
// Chemistry represents accumulated signal signatures on a hypha using
// element-wise minimum MinHash. This preserves the Jaccard estimation
// property (averaging would break it).
//
// QuerySignature is a MinHash signature used for query routing via
// chemical similarity matching.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// MinHash-based query signature for similarity-based routing.
///
/// Each byte position represents a MinHash band. Similarity between
/// a QuerySignature and a Chemistry is computed as the fraction of
/// matching positions (Jaccard estimate).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuerySignature {
    /// 64-byte MinHash signature.
    #[serde(with = "BigArray")]
    pub minhash: [u8; 64],
}

impl QuerySignature {
    /// Creates a new QuerySignature with the given MinHash values.
    pub fn new(minhash: [u8; 64]) -> Self {
        Self { minhash }
    }
}

impl Default for QuerySignature {
    fn default() -> Self {
        // Default query starts at all-0xFF (no data seen / wildcard match).
        Self {
            minhash: [0xFF; 64],
        }
    }
}

/// Accumulated chemical signature on a hypha.
///
/// Chemistry uses element-wise minimum MinHash accumulation:
/// when a new digest signature is deposited, each byte position
/// takes the minimum of the existing value and the new value.
/// This preserves the Jaccard estimation property of MinHash.
///
/// A fresh Chemistry starts with all-0xFF (maximum values),
/// meaning "no data has been seen yet."
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chemistry {
    /// 64-byte MinHash signature accumulated via element-wise minimum.
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],

    /// Number of signals that have contributed to this chemistry.
    pub signal_count: u64,
}

impl Chemistry {
    /// Creates a new Chemistry with uninitialized MinHash (all 0xFF)
    /// and zero signal count.
    pub fn new() -> Self {
        Self {
            signature: [0xFF; 64],
            signal_count: 0,
        }
    }

    /// Computes similarity between this Chemistry and a QuerySignature.
    ///
    /// Returns the fraction of byte positions where the values match,
    /// which is a Jaccard similarity estimate via MinHash.
    ///
    /// Returns a value in [0.0, 1.0].
    pub fn similarity(&self, query: &QuerySignature) -> f64 {
        let matching = self
            .signature
            .iter()
            .zip(query.minhash.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / 64.0
    }

    /// Deposits a digest signature into this Chemistry using
    /// element-wise minimum accumulation.
    ///
    /// For each byte position, the resulting value is
    /// `min(self.signature[i], digest_signature[i])`.
    /// This preserves the MinHash Jaccard estimation property.
    pub fn deposit(&mut self, digest_signature: &[u8; 64]) {
        for (slot, &incoming) in self.signature.iter_mut().zip(digest_signature.iter()) {
            *slot = (*slot).min(incoming);
        }
        self.signal_count += 1;
    }
}

impl Default for Chemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chemistry_new_starts_all_0xff() {
        let chem = Chemistry::new();
        assert_eq!(chem.signature, [0xFF; 64]);
        assert_eq!(chem.signal_count, 0);
    }

    #[test]
    fn query_signature_default_is_all_0xff() {
        let qs = QuerySignature::default();
        assert_eq!(qs.minhash, [0xFF; 64]);
    }

    #[test]
    fn deposit_takes_element_wise_minimum() {
        let mut chem = Chemistry::new();
        // Fresh chemistry is all 0xFF. Deposit a signature where the first
        // 32 bytes are 0x10 and the rest are 0xFF.
        let mut sig = [0xFF; 64];
        sig[..32].fill(0x10);
        chem.deposit(&sig);

        // First 32 bytes should now be min(0xFF, 0x10) = 0x10
        for i in 0..32 {
            assert_eq!(chem.signature[i], 0x10, "byte {} should be 0x10", i);
        }
        // Remaining 32 bytes should still be min(0xFF, 0xFF) = 0xFF
        for i in 32..64 {
            assert_eq!(chem.signature[i], 0xFF, "byte {} should be 0xFF", i);
        }
    }

    #[test]
    fn deposit_increments_signal_count() {
        let mut chem = Chemistry::new();
        let sig = [0x42; 64];
        chem.deposit(&sig);
        chem.deposit(&sig);
        assert_eq!(chem.signal_count, 2);
    }

    #[test]
    fn similarity_identical_signatures_returns_1() {
        let mut chem = Chemistry::new();
        let sig = [0x42; 64];
        chem.deposit(&sig);

        let query = QuerySignature::new([0x42; 64]);
        let sim = chem.similarity(&query);
        assert!(
            (sim - 1.0).abs() < f64::EPSILON,
            "expected 1.0, got {}",
            sim
        );
    }

    #[test]
    fn similarity_completely_different_returns_0() {
        // Build a chemistry where every byte is 0x00 (minimum possible).
        let mut chem = Chemistry::new();
        chem.deposit(&[0x00; 64]);
        // Query where every byte is 0xFF — no position matches 0x00.
        let query = QuerySignature::new([0xFF; 64]);
        let sim = chem.similarity(&query);
        assert!(sim.abs() < f64::EPSILON, "expected 0.0, got {}", sim);
    }

    #[test]
    fn similarity_partial_match() {
        // Build a chemistry: first 32 bytes = 0xAA, last 32 bytes = 0xBB.
        let mut chem = Chemistry::new();
        let mut sig = [0xBB; 64];
        sig[..32].fill(0xAA);
        chem.deposit(&sig);

        // Query: first 32 bytes = 0xAA (match), last 32 bytes = 0xCC (no match).
        let mut q = [0xCC; 64];
        q[..32].fill(0xAA);
        let query = QuerySignature::new(q);
        let sim = chem.similarity(&query);
        assert!(
            (sim - 0.5).abs() < f64::EPSILON,
            "expected 0.5, got {}",
            sim
        );
    }

    #[test]
    fn similarity_fresh_chemistry_vs_fresh_query() {
        let chem = Chemistry::new(); // all 0xFF
        let query = QuerySignature::default(); // all 0xFF
        let sim = chem.similarity(&query);
        assert!(
            (sim - 1.0).abs() < f64::EPSILON,
            "expected 1.0, got {}",
            sim
        );
    }
}
