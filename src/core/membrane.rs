// ALPS Discovery SDK — Membrane State
//
// Minimal membrane state type (passthrough for local discovery).

use std::time::Duration;

/// Current membrane state snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct MembraneState {
    /// Current permeability value.
    pub permeability: f64,
    /// Whether deep processing mode is active.
    pub deep_processing_active: bool,
    /// Number of signals currently buffered.
    pub buffered_count: usize,
    /// Duration spent at the permeability floor.
    pub floor_duration: Duration,
    /// Duration spent below the sporulation threshold.
    pub below_sporulation_duration: Duration,
    /// Total signals admitted since creation.
    pub total_admitted: u64,
    /// Total signals dissolved since creation.
    pub total_dissolved: u64,
    /// Total signals processed since creation.
    pub total_processed: u64,
    /// Admitted signal rate.
    pub admitted_rate: f64,
    /// Dissolved signal rate.
    pub dissolved_rate: f64,
}
