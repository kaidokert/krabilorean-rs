//! Bounded online streaming statistics.

mod crossings;
mod ew_variance;
mod ewma;
mod extrema;
mod histogram;
mod moments;
mod power_of_two;
mod runs;
mod successive_difference;

pub use crossings::{ThresholdCrossings, ThresholdCrossingsSnapshot};
pub use ew_variance::{
    EW_VARIANCE_MAX_ALPHA_SHIFT, EwVariance, EwVarianceSnapshot, ew_variance_max_fractional_bits,
};
pub use ewma::{EW_MEAN_MAX_ALPHA_SHIFT, EwMean, EwMeanSnapshot, ew_mean_max_fractional_bits};
pub use extrema::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
pub use histogram::{HistogramSnapshot, LinearHistogram};
pub use moments::{RawMomentsSnapshot, RunningMoments};
pub use power_of_two::{
    PowerOfTwoHistogram, PowerOfTwoHistogramSnapshot, power_of_two_buckets_are_valid,
};
pub use runs::{ThresholdRuns, ThresholdRunsSnapshot};
pub use successive_difference::{SuccessiveDifference, SuccessiveDifferenceSnapshot};
