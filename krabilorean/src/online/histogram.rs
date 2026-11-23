//! Constant-state cumulative fixed-range linear histogram.

use core::marker::PhantomData;

use crate::distribution::{Bucket, BucketView, ExactBucketView};
use crate::histogram::{
    HistogramBounds, HistogramClass, HistogramMode, bins_are_valid, classify, linear_bin_bounds,
    linear_bucket, lowest_mode,
};
use crate::policy::OverflowPolicy;
use crate::sample::HistogramWord;
use crate::{Error, Quality};

/// Borrowed fixed linear histogram state with compile-time bin capacity `BINS`.
///
/// Every word implements [`BucketView`] in its own word: slot `0` is the below-range bucket,
/// slots `1..=BINS` are the in-range bins ([`HistogramSnapshot::bin_bounds`]), and slot
/// `BINS + 1` is the above-range bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramSnapshot<'a, const BINS: usize, S> {
    /// Inclusive configured bounds.
    pub bounds: HistogramBounds<S>,
    /// Per-bin counts.
    pub counts: &'a [u32; BINS],
    /// Samples smaller than the lower bound.
    pub below: u32,
    /// Samples greater than the upper bound.
    pub above: u32,
    /// Total accepted samples.
    pub samples: u32,
    /// Lowest-index mode, or `None` when no sample is in range.
    pub mode: Option<HistogramMode>,
    /// Saturation quality.
    pub quality: Quality,
}

c0nst::c0nst! {
    impl<const BINS: usize, S: HistogramWord> HistogramSnapshot<'_, BINS, S> {
        /// Returns the inclusive sample bounds of in-range bin `bin`, at any width.
        ///
        /// Returns `None` past the last bin, or for a bin that holds no integer value because the
        /// histogram has more bins than distinct values in its range.
        pub c0nst fn bin_bounds(&self, bin: usize) -> Option<HistogramBounds<S>>
        where
            S: [c0nst] HistogramWord,
        {
            linear_bin_bounds::<BINS, S>(self.bounds, bin)
        }
    }

    c0nst impl<const BINS: usize, S: [c0nst] HistogramWord> BucketView<S>
        for HistogramSnapshot<'_, BINS, S>
    {
        fn bucket_slots(&self) -> usize {
            BINS + 2
        }

        fn bucket(&self, index: usize) -> Option<Bucket<S>> {
            linear_bucket::<BINS, S>(self.bounds, self.counts, self.below, self.above, index)
        }

        fn samples(&self) -> u32 {
            self.samples
        }

        fn quality(&self) -> Quality {
            self.quality
        }
    }

    c0nst impl<const BINS: usize, S: [c0nst] HistogramWord> crate::sealed::ExactBucketView
        for HistogramSnapshot<'_, BINS, S>
    {
    }

    c0nst impl<const BINS: usize, S: [c0nst] HistogramWord> ExactBucketView<S>
        for HistogramSnapshot<'_, BINS, S>
    {
    }
}

/// Cumulative fixed-range linear histogram.
///
/// `S` is the sample word. Counters are `u32` at every width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearHistogram<const BINS: usize, O, S> {
    bounds: HistogramBounds<S>,
    counts: [u32; BINS],
    below: u32,
    above: u32,
    samples: u32,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective histogram counter update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HistogramCounter {
    Below(u32),
    Bin { index: usize, value: u32 },
    Above(u32),
}

/// Prospective histogram state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistogramDelta {
    counter: HistogramCounter,
    samples: u32,
    quality: Quality,
}

/// Prospective exact histogram concatenation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistogramMergeDelta<const BINS: usize> {
    counts: [u32; BINS],
    below: u32,
    above: u32,
    samples: u32,
    quality: Quality,
}

impl<const BINS: usize, O, S: HistogramWord> LinearHistogram<BINS, O, S> {
    /// Creates an empty fixed histogram.
    pub const fn new(bounds: HistogramBounds<S>) -> Result<Self, Error> {
        if bins_are_valid::<BINS>() {
            Ok(Self {
                bounds,
                counts: [0; BINS],
                below: 0,
                above: 0,
                samples: 0,
                quality: Quality::Complete,
                policy: PhantomData,
            })
        } else {
            Err(Error::InvalidConfiguration)
        }
    }

    /// Returns the configured inclusive bounds.
    pub const fn bounds(&self) -> HistogramBounds<S> {
        self.bounds
    }

    /// Borrows the current bin counts and copies scalar counters.
    pub const fn snapshot(&self) -> HistogramSnapshot<'_, BINS, S> {
        HistogramSnapshot {
            bounds: self.bounds,
            counts: &self.counts,
            below: self.below,
            above: self.above,
            samples: self.samples,
            mode: lowest_mode(&self.counts),
            quality: self.quality,
        }
    }

    /// Clears all counts without changing bounds.
    pub const fn reset(&mut self) {
        self.counts = [0; BINS];
        self.below = 0;
        self.above = 0;
        self.samples = 0;
        self.quality = Quality::Complete;
    }
}

c0nst::c0nst! {
    impl<const BINS: usize, O: OverflowPolicy, S: HistogramWord> LinearHistogram<BINS, O, S> {
        /// Accepts one sample transactionally in constant time.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            S: [c0nst] HistogramWord,
        {
            let delta = match self.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit(delta);
            Ok(())
        }

        /// Appends a newer histogram with identical bounds in `O(BINS)` work.
        #[allow(clippy::question_mark)]
        pub c0nst fn merge(&mut self, newer: &Self) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            S: [c0nst] HistogramWord,
        {
            let delta = match self.prepare_merge(newer) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_merge(delta);
            Ok(())
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<HistogramDelta, Error>
        where
            O: [c0nst] OverflowPolicy,
            S: [c0nst] HistogramWord,
        {
            let samples = match O::add_u32(self.samples, 1) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let (counter, counter_saturated) = match classify::<BINS, S>(self.bounds, sample) {
                HistogramClass::Below => {
                    let next = match O::add_u32(self.below, 1) {
                        Ok(next) => next,
                        Err(error) => return Err(error),
                    };
                    (HistogramCounter::Below(next.value), next.saturated)
                }
                HistogramClass::Above => {
                    let next = match O::add_u32(self.above, 1) {
                        Ok(next) => next,
                        Err(error) => return Err(error),
                    };
                    (HistogramCounter::Above(next.value), next.saturated)
                }
                HistogramClass::Bin(index) => {
                    let next = match O::add_u32(self.counts[index], 1) {
                        Ok(next) => next,
                        Err(error) => return Err(error),
                    };
                    (
                        HistogramCounter::Bin {
                            index,
                            value: next.value,
                        },
                        next.saturated,
                    )
                }
            };
            Ok(HistogramDelta {
                counter,
                samples: samples.value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated() || samples.saturated || counter_saturated,
                ),
            })
        }

        pub(crate) const fn commit(&mut self, delta: HistogramDelta) {
            match delta.counter {
                HistogramCounter::Below(value) => self.below = value,
                HistogramCounter::Above(value) => self.above = value,
                HistogramCounter::Bin { index, value } => {
                    self.counts[index] = value;
                }
            }
            self.samples = delta.samples;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare_merge(
            &self,
            newer: &Self,
        ) -> Result<HistogramMergeDelta<BINS>, Error>
        where
            O: [c0nst] OverflowPolicy,
            S: [c0nst] HistogramWord,
        {
            if !self.bounds.eq_bounds(newer.bounds) {
                return Err(Error::InvalidConfiguration);
            }
            let samples = match O::add_u32(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let below = match O::add_u32(self.below, newer.below) {
                Ok(below) => below,
                Err(error) => return Err(error),
            };
            let above = match O::add_u32(self.above, newer.above) {
                Ok(above) => above,
                Err(error) => return Err(error),
            };
            let mut counts = [0; BINS];
            let mut saturated = self.quality.is_saturated()
                || newer.quality.is_saturated()
                || samples.saturated
                || below.saturated
                || above.saturated;
            let mut index = 0_usize;
            while index < BINS {
                let merged = match O::add_u32(self.counts[index], newer.counts[index]) {
                    Ok(merged) => merged,
                    Err(error) => return Err(error),
                };
                counts[index] = merged.value;
                saturated = saturated || merged.saturated;
                index += 1;
            }
            Ok(HistogramMergeDelta {
                counts,
                below: below.value,
                above: above.value,
                samples: samples.value,
                quality: Quality::from_flags(false, saturated),
            })
        }

        pub(crate) const fn commit_merge(&mut self, delta: HistogramMergeDelta<BINS>) {
            self.counts = delta.counts;
            self.below = delta.below;
            self.above = delta.above;
            self.samples = delta.samples;
            self.quality = delta.quality;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::NonZeroU32;

    use super::LinearHistogram;
    use crate::distribution::{BucketView, ExactDistributionQueries, Quantile};
    use crate::histogram::{HistogramBounds, HistogramMode};
    use crate::policy::{Checked, Saturating};
    use crate::sample::HistogramWord;
    use crate::{Error, Quality};

    #[test]
    fn zero_bins_are_rejected_at_construction() {
        let bounds = HistogramBounds::<i16>::new(0, 7).expect("valid bounds");
        assert_eq!(
            LinearHistogram::<0, Checked, i16>::new(bounds),
            Err(Error::InvalidConfiguration)
        );
    }

    fn check_lifecycle_and_merges<S: HistogramWord>(lower: S, mid_low: S, mid_high: S, upper: S) {
        let bounds = HistogramBounds::new(lower, upper).expect("valid bounds");
        let mut histogram = LinearHistogram::<4, Checked, S>::new(bounds).expect("valid");
        assert_eq!(histogram.bounds(), bounds);
        let initial = histogram.snapshot();
        assert_eq!(initial.samples, 0);
        assert_eq!(initial.below, 0);
        assert_eq!(initial.above, 0);
        assert_eq!(initial.counts, &[0, 0, 0, 0]);
        assert_eq!(initial.mode, None);
        assert_eq!(initial.quality, Quality::Complete);
        assert_eq!(
            initial.bin_bounds(0).map(HistogramBounds::lower),
            Some(lower)
        );
        assert_eq!(
            initial.bin_bounds(3).map(HistogramBounds::upper),
            Some(upper)
        );
        assert_eq!(initial.bin_bounds(4), None);

        let stream = [S::MIN, lower, mid_low, mid_low, mid_high, upper, S::MAX];
        for &sample in &stream {
            histogram.update(sample).expect("representable");
        }
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.samples, 7);
        assert_eq!(snapshot.below, 1);
        assert_eq!(snapshot.above, 1);
        assert_eq!(snapshot.counts.iter().sum::<u32>(), 5);
        assert_eq!(snapshot.bucket_slots(), 6);
        assert_eq!(snapshot.samples(), 7);
        assert_eq!(snapshot.quality(), Quality::Complete);
        assert_eq!(snapshot.bucket(6), None);
        assert_eq!(snapshot.occupancy().expect("valid").occupied, 6);
        assert_eq!(snapshot.mode().expect("valid").map(|m| m.index), Some(2));
        let median = snapshot
            .quantile_bounds(Quantile::new(1, 2).expect("valid"))
            .expect("valid")
            .expect("non-empty");
        assert_eq!(median.rank, 3);
        assert!(mid_low >= median.values.lower() && mid_low <= median.values.upper());
        let tails = snapshot.tail_bounds(mid_low).expect("valid");
        assert!(tails.at_or_below.minimum <= 4 && 4 <= tails.at_or_below.maximum);
        assert!(tails.at_or_above.minimum <= 5 && 5 <= tails.at_or_above.maximum);
        assert_eq!(
            snapshot.mode,
            Some(HistogramMode {
                bin: 1,
                count: NonZeroU32::new(2).expect("nonzero"),
            })
        );

        for split in 0..=stream.len() {
            let mut left = LinearHistogram::<4, Checked, S>::new(bounds).expect("valid");
            let mut right = LinearHistogram::<4, Checked, S>::new(bounds).expect("valid");
            for &sample in &stream[..split] {
                left.update(sample).expect("representable");
            }
            for &sample in &stream[split..] {
                right.update(sample).expect("representable");
            }
            left.merge(&right).expect("representable");
            assert_eq!(left, histogram);
        }

        histogram.reset();
        assert_eq!(histogram.snapshot().samples, 0);
        assert_eq!(histogram.snapshot().counts, &[0, 0, 0, 0]);
        assert_eq!(histogram.snapshot().mode, None);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_word() {
        check_lifecycle_and_merges::<i8>(-8, -4, 0, 7);
        check_lifecycle_and_merges::<i16>(-80, -40, 0, 79);
        check_lifecycle_and_merges::<i32>(-800, -400, 0, 799);
        check_lifecycle_and_merges::<u8>(8, 10, 12, 15);
        check_lifecycle_and_merges::<u16>(80, 100, 120, 159);
        check_lifecycle_and_merges::<u32>(800, 1000, 1200, 1599);
    }

    #[test]
    fn checked_bin_overflow_is_transactional() {
        let bounds = HistogramBounds::<i16>::new(0, 7).expect("valid bounds");
        let mut histogram = LinearHistogram::<4, Checked, i16> {
            bounds,
            counts: [u32::MAX, 0, 0, 0],
            below: 0,
            above: 0,
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = histogram;
        assert_eq!(histogram.update(0), Err(Error::Overflow));
        assert_eq!(histogram, before);
    }

    #[test]
    fn saturating_bin_and_edges_are_visible_in_quality() {
        let bounds = HistogramBounds::<i16>::new(0, 7).expect("valid bounds");
        let mut histogram = LinearHistogram::<4, Saturating, i16> {
            bounds,
            counts: [u32::MAX, 0, 0, 0],
            below: u32::MAX,
            above: u32::MAX,
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        histogram.update(0).expect("saturating bin update succeeds");
        assert_eq!(histogram.snapshot().counts[0], u32::MAX);
        assert_eq!(histogram.snapshot().quality, Quality::Saturated);

        let mut samples_sat = LinearHistogram::<4, Saturating, i16> {
            bounds,
            counts: [0; 4],
            below: 0,
            above: 0,
            samples: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        samples_sat
            .update(0)
            .expect("saturating samples update succeeds");
        assert_eq!(samples_sat.snapshot().quality, Quality::Saturated);

        let mut below_sat = LinearHistogram::<4, Saturating, i16> {
            quality: Quality::Complete,
            ..histogram
        };
        below_sat
            .update(-1)
            .expect("saturating below update succeeds");
        assert_eq!(below_sat.snapshot().below, u32::MAX);
        assert_eq!(below_sat.snapshot().quality, Quality::Saturated);

        let mut above_sat = LinearHistogram::<4, Saturating, i16> {
            quality: Quality::Complete,
            ..histogram
        };
        above_sat
            .update(8)
            .expect("saturating above update succeeds");
        assert_eq!(above_sat.snapshot().above, u32::MAX);
        assert_eq!(above_sat.snapshot().quality, Quality::Saturated);

        let mut left = LinearHistogram::<4, Saturating, i16> {
            bounds,
            counts: [u32::MAX, 0, 0, 0],
            below: 0,
            above: 0,
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = LinearHistogram::<4, Saturating, i16>::new(bounds).expect("valid");
        right.update(0).expect("in-range");
        left.merge(&right).expect("saturating merge succeeds");
        assert_eq!(left.snapshot().quality, Quality::Saturated);
        left.merge(&right)
            .expect("already saturated merge succeeds");
        assert_eq!(left.snapshot().quality, Quality::Saturated);
        left.update(0).expect("already saturated update succeeds");
        assert_eq!(left.snapshot().quality, Quality::Saturated);
    }

    #[test]
    fn checked_merge_overflow_is_transactional() {
        let bounds = HistogramBounds::<i16>::new(0, 7).expect("valid bounds");
        let base = LinearHistogram::<4, Checked, i16> {
            bounds,
            counts: [1, 0, 0, 0],
            below: 1,
            above: 1,
            samples: 3,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        for full in [
            LinearHistogram {
                samples: u32::MAX,
                ..base
            },
            LinearHistogram {
                below: u32::MAX,
                ..base
            },
            LinearHistogram {
                above: u32::MAX,
                ..base
            },
            LinearHistogram {
                counts: [u32::MAX, 0, 0, 0],
                ..base
            },
        ] {
            let mut left = full;
            assert_eq!(left.merge(&base), Err(Error::Overflow));
            assert_eq!(left, full);
        }
    }

    fn histogram_fails_transactionally<S: HistogramWord>(lower: S, upper: S) {
        let bounds = HistogramBounds::<S>::new(lower, upper).expect("valid bounds");
        let full_samples = LinearHistogram::<4, Checked, S> {
            bounds,
            counts: [0; 4],
            below: 0,
            above: 0,
            samples: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_samples;
        assert_eq!(histogram.update(lower), Err(Error::Overflow));
        assert_eq!(histogram, full_samples);

        let full_counters = LinearHistogram::<4, Checked, S> {
            bounds,
            counts: [u32::MAX; 4],
            below: u32::MAX,
            above: u32::MAX,
            samples: 7,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_counters;
        for sample in [S::MIN, lower, upper, S::MAX] {
            assert_eq!(histogram.update(sample), Err(Error::Overflow));
        }
        assert_eq!(histogram, full_counters);

        let mut other = LinearHistogram::<4, Checked, S>::new(bounds).expect("valid");
        other.update(lower).expect("representable");
        assert_eq!(histogram.merge(&other), Err(Error::Overflow));
        assert_eq!(histogram, full_counters);

        let other_bounds = HistogramBounds::<S>::new(lower, lower).expect("valid bounds");
        let other = LinearHistogram::<4, Checked, S>::new(other_bounds).expect("valid");
        assert_eq!(histogram.merge(&other), Err(Error::InvalidConfiguration));
        assert_eq!(histogram, full_counters);
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        histogram_fails_transactionally::<i8>(-8, 7);
        histogram_fails_transactionally::<i16>(-8, 7);
        histogram_fails_transactionally::<i32>(-8, 7);
        histogram_fails_transactionally::<u8>(8, 15);
        histogram_fails_transactionally::<u16>(8, 15);
        histogram_fails_transactionally::<u32>(8, 15);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_full_histogram_lifecycle_is_const_evaluable_on_nightly() {
        const BOUNDS: HistogramBounds<i16> = match HistogramBounds::new(-8, 7) {
            Ok(bounds) => bounds,
            Err(_) => unreachable!(),
        };
        const UPDATED: LinearHistogram<4, Checked, i16> = {
            let mut left = match LinearHistogram::new(BOUNDS) {
                Ok(histogram) => histogram,
                Err(_) => unreachable!(),
            };
            let mut right = match LinearHistogram::new(BOUNDS) {
                Ok(histogram) => histogram,
                Err(_) => unreachable!(),
            };
            assert!(left.update(-9).is_ok());
            assert!(left.update(-8).is_ok());
            assert!(right.update(-7).is_ok());
            assert!(right.update(8).is_ok());
            assert!(left.merge(&right).is_ok());
            left
        };
        const SNAPSHOT: super::HistogramSnapshot<'static, 4, i16> = UPDATED.snapshot();
        const BIN_0: Option<HistogramBounds<i16>> = SNAPSHOT.bin_bounds(0);
        assert_eq!(SNAPSHOT.samples, 4);
        assert_eq!(SNAPSHOT.below, 1);
        assert_eq!(SNAPSHOT.above, 1);
        assert_eq!(SNAPSHOT.counts, &[2, 0, 0, 0]);
        assert_eq!(
            SNAPSHOT.mode,
            Some(HistogramMode {
                bin: 0,
                count: NonZeroU32::new(2).unwrap(),
            })
        );
        assert_eq!(BIN_0, HistogramBounds::new(-8, -5).ok());
    }
}
