// ALPS Discovery SDK
//
// Local agent discovery powered by bio-inspired multi-kernel routing.
// Self-contained package with vendored core types.

use pyo3::prelude::*;

pub mod core;
pub mod error;
pub mod network;
pub mod pybridge;
pub mod query;
pub mod scorer;

/// Feature-gated re-export of internal types for benchmarking.
/// Only compiled when the `bench` feature is enabled.
#[cfg(feature = "bench")]
pub mod bench_internals {
    // Re-export pipeline internals
    pub use crate::network::pipeline::{
        compute_feedback_factor, run_pipeline, run_pipeline_with_scores, similarity_to_ci,
        ScoredCandidate, FEEDBACK_STRENGTH,
    };

    // Re-export registry internals
    pub use crate::network::registry::{
        name_to_hypha_id, record_failure, record_success, register_agent, tick, AgentRecord,
        FeedbackIndex, FeedbackRecord,
    };

    // Re-export adapter types
    pub use crate::network::enzyme_adapter::EnzymeAdapter;
    pub use crate::network::scorer_adapter::ScorerAdapter;
}

/// ALPS Discovery — local agent discovery via bio-inspired routing.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pybridge::PyLocalNetwork>()?;
    m.add_class::<pybridge::PyDiscoveryResult>()?;
    m.add_class::<pybridge::PyExplainedResult>()?;
    m.add_class::<pybridge::PyDiscoveryResponse>()?;
    m.add_class::<pybridge::PyQuery>()?;
    m.add_class::<pybridge::PyTfIdfScorer>()?;
    m.add_class::<pybridge::PyMycorrhizalPropagator>()?;
    m.add_class::<pybridge::PyCircuitBreakerConfig>()?;
    m.add_function(wrap_pyfunction!(pybridge::capabilities_from_a2a_rust, m)?)?;
    Ok(())
}
