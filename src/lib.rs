// ALPS Discovery SDK
//
// Local agent discovery powered by bio-inspired multi-kernel routing.
// Self-contained package with vendored core types.

use pyo3::prelude::*;

pub mod core;
pub mod network;
pub mod pybridge;
pub mod query;
pub mod scorer;

/// ALPS Discovery — local agent discovery via bio-inspired routing.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pybridge::PyLocalNetwork>()?;
    m.add_class::<pybridge::PyDiscoveryResult>()?;
    m.add_class::<pybridge::PyExplainedResult>()?;
    m.add_class::<pybridge::PyDiscoveryResponse>()?;
    m.add_class::<pybridge::PyQuery>()?;
    m.add_class::<pybridge::PyTfIdfScorer>()?;
    Ok(())
}
