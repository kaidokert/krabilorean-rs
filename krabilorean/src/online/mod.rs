//! Bounded online streaming statistics.

mod crossings;
mod extrema;
mod histogram;
mod moments;
mod power_of_two;
mod runs;
mod successive_difference;

pub use crossings::{ThresholdCrossings, ThresholdCrossingsSnapshot};
pub use extrema::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
pub use histogram::{HistogramSnapshot, LinearHistogram};
pub use moments::{RawMomentsSnapshot, RunningMoments};
pub use power_of_two::{
    PowerOfTwoHistogram, PowerOfTwoHistogramSnapshot, power_of_two_buckets_are_valid,
};
pub use runs::{ThresholdRuns, ThresholdRunsSnapshot};
pub use successive_difference::{SuccessiveDifference, SuccessiveDifferenceSnapshot};
