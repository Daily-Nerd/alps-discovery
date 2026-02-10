// ALPS Discovery SDK — Spore Stub
//
// Minimal stub satisfying the Enzyme trait signature.
// Local discovery does not use the spore storage system.

use crate::core::config::SporeConfig;

/// Stub spore — no data storage, satisfies the enzyme trait signature.
pub struct Spore;

impl Spore {
    /// Creates a new stub Spore (ignores config).
    pub fn new(_config: SporeConfig) -> Self {
        Self
    }
}
