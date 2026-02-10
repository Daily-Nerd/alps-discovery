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
