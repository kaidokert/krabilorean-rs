//! Bounded online streaming statistics.

mod crossings;
mod extrema;
mod runs;

pub use crossings::{ThresholdCrossings, ThresholdCrossingsSnapshot};
pub use extrema::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
pub use runs::{ThresholdRuns, ThresholdRunsSnapshot};
