//! Constant-state cumulative fixed-range linear histogram.

use core::marker::PhantomData;

use super::{MergeSlot, Slot};
use crate::distribution::{Bucket, BucketView, ExactBucketView};
use crate::histogram::{
    HistogramBounds, HistogramClass, HistogramMode, bins_are_valid, classify, linear_bin_bounds,
    linear_bucket, lowest_mode,
};
use crate::policy::{CounterWord, OverflowPolicy};
use crate::sample::HistogramWord;
use crate::{Error, Quality};

/// Borrowed fixed linear histogram state with compile-time bin capacity `BINS`.
///
/// Every word implements [`BucketView`] in its own word: slot `0` is the below-range bucket,
/// slots `1..=BINS` are the in-range bins ([`HistogramSnapshot::bin_bounds`]), and slot
/// `BINS + 1` is the above-range bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramSnapshot<'a, const BINS: usize, C: CounterWord, S> {
    /// Inclusive configured bounds.
    pub bounds: HistogramBounds<S>,
    /// Per-bin counts.
    pub counts: &'a [C; BINS],
    /// Samples smaller than the lower bound.
    pub below: C,
    /// Samples greater than the upper bound.
    pub above: C,
    /// Total accepted samples.
    pub samples: C,
    /// Lowest-index mode, or `None` when no sample is in range.
    pub mode: Option<HistogramMode<C>>,
    /// Saturation quality.
    pub quality: Quality,
}

c0nst::c0nst! {
    impl<const BINS: usize, C: CounterWord, S: HistogramWord> HistogramSnapshot<'_, BINS, C, S> {
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

    c0nst impl<const BINS: usize, C: [c0nst] CounterWord, S: [c0nst] HistogramWord>
        BucketView<C, S> for HistogramSnapshot<'_, BINS, C, S>
    {
        fn bucket_slots(&self) -> usize {
            BINS + 2
        }

        fn bucket(&self, index: usize) -> Option<Bucket<C, S>> {
            linear_bucket::<BINS, C, S>(self.bounds, self.counts, self.below, self.above, index)
        }

        fn samples(&self) -> C {
            self.samples
        }

        fn quality(&self) -> Quality {
            self.quality
        }
    }

    c0nst impl<const BINS: usize, C: [c0nst] CounterWord, S: [c0nst] HistogramWord>
        crate::sealed::ExactBucketView for HistogramSnapshot<'_, BINS, C, S>
    {
    }

    c0nst impl<const BINS: usize, C: [c0nst] CounterWord, S: [c0nst] HistogramWord>
        ExactBucketView<C, S> for HistogramSnapshot<'_, BINS, C, S>
    {
    }
}

/// Cumulative fixed-range linear histogram.
///
/// `C` is the counter word (`u8`, `u16`, `u32`, or `u64`): each bin, the below and above counts,
/// and the sample count are stored in `C`. `S` is the sample word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearHistogram<const BINS: usize, O, C, S> {
    bounds: HistogramBounds<S>,
    counts: [C; BINS],
    below: C,
    above: C,
    samples: C,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HistogramCounter<C> {
    Below(C),
    Bin { index: usize, value: C },
    Above(C),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistogramDelta<C> {
    counter: HistogramCounter<C>,
    samples: C,
    quality: Quality,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistogramMergeDelta<const BINS: usize, C> {
    counts: [C; BINS],
    below: C,
    above: C,
    samples: C,
    quality: Quality,
}

impl<const BINS: usize, O, C: CounterWord, S: HistogramWord> LinearHistogram<BINS, O, C, S> {
    /// Creates an empty fixed histogram.
    pub const fn new(bounds: HistogramBounds<S>) -> Result<Self, Error> {
        if bins_are_valid::<BINS>() {
            Ok(Self {
                bounds,
                counts: [C::ZERO; BINS],
                below: C::ZERO,
                above: C::ZERO,
                samples: C::ZERO,
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

    /// Clears all counts without changing bounds.
    pub const fn reset(&mut self) {
        self.counts = [C::ZERO; BINS];
        self.below = C::ZERO;
        self.above = C::ZERO;
        self.samples = C::ZERO;
        self.quality = Quality::Complete;
    }
}

c0nst::c0nst! {
    impl<const BINS: usize, O, C: CounterWord, S: HistogramWord> LinearHistogram<BINS, O, C, S> {
        /// Borrows the current bin counts and copies scalar counters.
        pub c0nst fn snapshot(&self) -> HistogramSnapshot<'_, BINS, C, S>
        where
            C: [c0nst] CounterWord,
        {
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
    }

    impl<const BINS: usize, O: OverflowPolicy, C: CounterWord, S: HistogramWord>
        LinearHistogram<BINS, O, C, S>
    {
        /// Accepts one sample transactionally in constant time.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
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
            C: [c0nst] CounterWord,
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
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<HistogramDelta<C>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] HistogramWord,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let (counter, counter_saturated) = match classify::<BINS, S>(self.bounds, sample) {
                HistogramClass::Below => {
                    let next = match C::add::<O>(self.below, C::ONE) {
                        Ok(next) => next,
                        Err(error) => return Err(error),
                    };
                    (HistogramCounter::Below(next.value), next.saturated)
                }
                HistogramClass::Above => {
                    let next = match C::add::<O>(self.above, C::ONE) {
                        Ok(next) => next,
                        Err(error) => return Err(error),
                    };
                    (HistogramCounter::Above(next.value), next.saturated)
                }
                HistogramClass::Bin(index) => {
                    let next = match C::add::<O>(self.counts[index], C::ONE) {
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

        pub(crate) const fn commit(&mut self, delta: HistogramDelta<C>) {
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
        ) -> Result<HistogramMergeDelta<BINS, C>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] HistogramWord,
        {
            if !self.bounds.eq_bounds(newer.bounds) {
                return Err(Error::InvalidConfiguration);
            }
            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let below = match C::add::<O>(self.below, newer.below) {
                Ok(below) => below,
                Err(error) => return Err(error),
            };
            let above = match C::add::<O>(self.above, newer.above) {
                Ok(above) => above,
                Err(error) => return Err(error),
            };
            let mut counts = [C::ZERO; BINS];
            let mut saturated = self.quality.is_saturated()
                || newer.quality.is_saturated()
                || samples.saturated
                || below.saturated
                || above.saturated;
            let mut index = 0_usize;
            while index < BINS {
                let merged = match C::add::<O>(self.counts[index], newer.counts[index]) {
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

        pub(crate) const fn commit_merge(&mut self, delta: HistogramMergeDelta<BINS, C>) {
            self.counts = delta.counts;
            self.below = delta.below;
            self.above = delta.above;
            self.samples = delta.samples;
            self.quality = delta.quality;
        }
    }

    c0nst impl<
        const BINS: usize,
        O: [c0nst] OverflowPolicy,
        C: [c0nst] CounterWord,
        S: [c0nst] HistogramWord,
    > Slot<S> for LinearHistogram<BINS, O, C, S>
    {
        type Delta = HistogramDelta<C>;
        type Snapshot<'a>
            = HistogramSnapshot<'a, BINS, C, S>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            LinearHistogram::snapshot(self)
        }

        fn reset(&mut self) {
            LinearHistogram::reset(self);
        }
    }

    c0nst impl<
        const BINS: usize,
        O: [c0nst] OverflowPolicy,
        C: [c0nst] CounterWord,
        S: [c0nst] HistogramWord,
    > MergeSlot<S> for LinearHistogram<BINS, O, C, S>
    {
        type MergeDelta = HistogramMergeDelta<BINS, C>;

        fn prepare_merge(&self, newer: &Self) -> Result<Self::MergeDelta, Error> {
            self.prepare_merge(newer)
        }

        fn commit_merge(&mut self, delta: Self::MergeDelta) {
            self.commit_merge(delta);
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
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::HistogramWord;
    use crate::{Error, Quality};

    fn count<C: CounterWord>(n: u64) -> C {
        C::from_u64(n).expect("small counts fit every counter word")
    }

    #[test]
    fn zero_bins_are_rejected_at_construction() {
        let bounds = HistogramBounds::<i16>::new(0, 7).expect("valid bounds");
        assert_eq!(
            LinearHistogram::<0, Checked, u32, i16>::new(bounds),
            Err(Error::InvalidConfiguration)
        );
    }

    fn check_lifecycle_and_merges<S: HistogramWord>(lower: S, mid_low: S, mid_high: S, upper: S) {
        let bounds = HistogramBounds::new(lower, upper).expect("valid bounds");
        let mut histogram = LinearHistogram::<4, Checked, u32, S>::new(bounds).expect("valid");
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
            let mut left = LinearHistogram::<4, Checked, u32, S>::new(bounds).expect("valid");
            let mut right = LinearHistogram::<4, Checked, u32, S>::new(bounds).expect("valid");
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
        let mut histogram = LinearHistogram::<4, Checked, u32, i16> {
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
        let mut histogram = LinearHistogram::<4, Saturating, u32, i16> {
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

        let mut samples_sat = LinearHistogram::<4, Saturating, u32, i16> {
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

        let mut below_sat = LinearHistogram::<4, Saturating, u32, i16> {
            quality: Quality::Complete,
            ..histogram
        };
        below_sat
            .update(-1)
            .expect("saturating below update succeeds");
        assert_eq!(below_sat.snapshot().below, u32::MAX);
        assert_eq!(below_sat.snapshot().quality, Quality::Saturated);

        let mut above_sat = LinearHistogram::<4, Saturating, u32, i16> {
            quality: Quality::Complete,
            ..histogram
        };
        above_sat
            .update(8)
            .expect("saturating above update succeeds");
        assert_eq!(above_sat.snapshot().above, u32::MAX);
        assert_eq!(above_sat.snapshot().quality, Quality::Saturated);

        let mut left = LinearHistogram::<4, Saturating, u32, i16> {
            bounds,
            counts: [u32::MAX, 0, 0, 0],
            below: 0,
            above: 0,
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = LinearHistogram::<4, Saturating, u32, i16>::new(bounds).expect("valid");
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
        let base = LinearHistogram::<4, Checked, u32, i16> {
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

    fn histogram_fails_transactionally<S: HistogramWord, C: CounterWord>(lower: S, upper: S) {
        let bounds = HistogramBounds::<S>::new(lower, upper).expect("valid bounds");
        let full_samples = LinearHistogram::<4, Checked, C, S> {
            bounds,
            counts: [C::ZERO; 4],
            below: C::ZERO,
            above: C::ZERO,
            samples: C::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_samples;
        assert_eq!(histogram.update(lower), Err(Error::Overflow));
        assert_eq!(histogram, full_samples);

        let full_counters = LinearHistogram::<4, Checked, C, S> {
            bounds,
            counts: [C::MAX; 4],
            below: C::MAX,
            above: C::MAX,
            samples: count(7),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_counters;
        for sample in [S::MIN, lower, upper, S::MAX] {
            assert_eq!(histogram.update(sample), Err(Error::Overflow));
        }
        assert_eq!(histogram, full_counters);

        let mut other = LinearHistogram::<4, Checked, C, S>::new(bounds).expect("valid");
        other.update(lower).expect("representable");
        assert_eq!(histogram.merge(&other), Err(Error::Overflow));
        assert_eq!(histogram, full_counters);

        let other_bounds = HistogramBounds::<S>::new(lower, lower).expect("valid bounds");
        let other = LinearHistogram::<4, Checked, C, S>::new(other_bounds).expect("valid");
        assert_eq!(histogram.merge(&other), Err(Error::InvalidConfiguration));
        assert_eq!(histogram, full_counters);
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        histogram_fails_transactionally::<i8, u8>(-8, 7);
        histogram_fails_transactionally::<i8, u16>(-8, 7);
        histogram_fails_transactionally::<i8, u32>(-8, 7);
        histogram_fails_transactionally::<i8, u64>(-8, 7);
        histogram_fails_transactionally::<i16, u8>(-8, 7);
        histogram_fails_transactionally::<i16, u16>(-8, 7);
        histogram_fails_transactionally::<i16, u32>(-8, 7);
        histogram_fails_transactionally::<i16, u64>(-8, 7);
        histogram_fails_transactionally::<i32, u8>(-8, 7);
        histogram_fails_transactionally::<i32, u16>(-8, 7);
        histogram_fails_transactionally::<i32, u32>(-8, 7);
        histogram_fails_transactionally::<i32, u64>(-8, 7);
        histogram_fails_transactionally::<u8, u8>(8, 15);
        histogram_fails_transactionally::<u8, u16>(8, 15);
        histogram_fails_transactionally::<u8, u32>(8, 15);
        histogram_fails_transactionally::<u8, u64>(8, 15);
        histogram_fails_transactionally::<u16, u8>(8, 15);
        histogram_fails_transactionally::<u16, u16>(8, 15);
        histogram_fails_transactionally::<u16, u32>(8, 15);
        histogram_fails_transactionally::<u16, u64>(8, 15);
        histogram_fails_transactionally::<u32, u8>(8, 15);
        histogram_fails_transactionally::<u32, u16>(8, 15);
        histogram_fails_transactionally::<u32, u32>(8, 15);
        histogram_fails_transactionally::<u32, u64>(8, 15);
    }

    #[test]
    fn u8_counters_fill_at_255_and_take_less_state_than_wider_words() {
        let bounds = HistogramBounds::<i16>::new(0, 15).expect("valid bounds");
        let mut checked = LinearHistogram::<4, Checked, u8, i16>::new(bounds).expect("valid");
        let mut saturating = LinearHistogram::<4, Saturating, u8, i16>::new(bounds).expect("valid");
        for _ in 0..u8::MAX {
            checked.update(3).expect("below u8::MAX");
            saturating.update(3).expect("below u8::MAX");
        }
        let full = checked;
        assert_eq!(checked.update(3), Err(Error::Overflow));
        assert_eq!(checked, full);
        assert_eq!(checked.snapshot().samples, u8::MAX);

        saturating.update(3).expect("saturating accepts");
        assert_eq!(saturating.snapshot().samples, u8::MAX);
        assert_eq!(saturating.snapshot().quality, Quality::Saturated);

        let u8_bytes = core::mem::size_of::<LinearHistogram<64, Checked, u8, i16>>();
        let u16_bytes = core::mem::size_of::<LinearHistogram<64, Checked, u16, i16>>();
        let u32_bytes = core::mem::size_of::<LinearHistogram<64, Checked, u32, i16>>();
        let u64_bytes = core::mem::size_of::<LinearHistogram<64, Checked, u64, i16>>();
        assert!(u8_bytes < u16_bytes && u16_bytes < u32_bytes && u32_bytes < u64_bytes);
        assert!(u32_bytes - u8_bytes >= 3 * 67);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_full_histogram_lifecycle_is_const_evaluable_on_nightly() {
        const BOUNDS: HistogramBounds<i16> = match HistogramBounds::new(-8, 7) {
            Ok(bounds) => bounds,
            Err(_) => unreachable!(),
        };
        const UPDATED: LinearHistogram<4, Checked, u32, i16> = {
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
        const SNAPSHOT: super::HistogramSnapshot<'static, 4, u32, i16> = UPDATED.snapshot();
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
