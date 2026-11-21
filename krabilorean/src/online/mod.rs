//! Bounded online streaming statistics.

mod crossings;
mod extrema;
mod runs;
mod successive_difference;

pub use crossings::{ThresholdCrossings, ThresholdCrossingsSnapshot};
pub use extrema::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
pub use runs::{ThresholdRuns, ThresholdRunsSnapshot};
pub use successive_difference::{SuccessiveDifference, SuccessiveDifferenceSnapshot};
