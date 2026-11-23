//! Constant-state complete-domain power-of-two logarithmic histogram.

use core::marker::PhantomData;

use crate::distribution::{Bucket, BucketView, ExactBucketView, ValueBounds};
use crate::histogram::HistogramBounds;
use crate::policy::{Checked, OverflowPolicy};
use crate::sample::HistogramWord;
use crate::{Error, Quality};

/// Returns whether `N` can partition the [`HistogramWord::POWER_OF_TWO_CELLS`] canonical cells of
/// the sample word `S`: 16 for `i8`, 32 for `i16`, 64 for `i32`, and 8, 16, and 32 for `u8`,
/// `u16`, and `u32`.
pub const fn power_of_two_buckets_are_valid<const N: usize, S: HistogramWord>() -> bool {
    N > 0 && N <= S::POWER_OF_TWO_CELLS
}

/// Borrowed power-of-two histogram state.
///
/// `S` is the sample word whose partition the counts use. Every word implements [`BucketView`] in
/// its own word, with the bounds of [`PowerOfTwoHistogramSnapshot::bucket_bounds`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowerOfTwoHistogramSnapshot<'a, const N: usize, S> {
    /// Per-bucket counts in increasing value order.
    pub counts: &'a [u32; N],
    /// Total accepted samples.
    pub samples: u32,
    /// Arithmetic quality.
    pub quality: Quality,
    word: PhantomData<fn() -> S>,
}

impl<'a, const N: usize, S> PowerOfTwoHistogramSnapshot<'a, N, S> {
    pub(crate) const fn from_parts(counts: &'a [u32; N], samples: u32, quality: Quality) -> Self {
        Self {
            counts,
            samples,
            quality,
            word: PhantomData,
        }
    }
}

c0nst::c0nst! {
    impl<const N: usize, S: HistogramWord> PowerOfTwoHistogramSnapshot<'_, N, S> {
        /// Returns the inclusive sample bounds of bucket `index` in the word's partition, or
        /// `None` past the last bucket.
        pub c0nst fn bucket_bounds(&self, index: usize) -> Option<HistogramBounds<S>>
        where
            S: [c0nst] HistogramWord,
        {
            PowerOfTwoHistogram::<N, Checked, S>::bucket_bounds(index)
        }
    }

    c0nst impl<const N: usize, S: [c0nst] HistogramWord> BucketView<S>
        for PowerOfTwoHistogramSnapshot<'_, N, S>
    {
        fn bucket_slots(&self) -> usize {
            N
        }

        fn bucket(&self, index: usize) -> Option<Bucket<S>> {
            let bounds = match self.bucket_bounds(index) {
                Some(bounds) => bounds,
                None => return None,
            };
            Some(Bucket {
                values: ValueBounds::from_histogram_bounds(bounds),
                count: self.counts[index],
            })
        }

        fn samples(&self) -> u32 {
            self.samples
        }

        fn quality(&self) -> Quality {
            self.quality
        }
    }

    c0nst impl<const N: usize, S: [c0nst] HistogramWord> crate::sealed::ExactBucketView
        for PowerOfTwoHistogramSnapshot<'_, N, S>
    {
    }

    c0nst impl<const N: usize, S: [c0nst] HistogramWord> ExactBucketView<S>
        for PowerOfTwoHistogramSnapshot<'_, N, S>
    {
    }
}

/// Exact division-free `N`-bucket logarithmic histogram over the complete domain of the sample
/// word `S`.
///
/// The full-resolution partition has `C =` [`HistogramWord::POWER_OF_TWO_CELLS`] cells:
///
/// - **Signed words, `C = 2 * S::BITS`:** `S::BITS` negative base-2 magnitude classes (the widest
///   holding only `S::MIN`), then zero, then `S::BITS - 1` positive classes.
/// - **Unsigned words, `C = S::BITS`:** `[0, 1]`, then the octaves `[2^k, 2^(k + 1) - 1]` for `k`
///   in `1..S::BITS`. Zero shares the lowest cell with one.
///
/// `N` in `1..=C` coarsens adjacent cells by `floor(cell * N / C)` while preserving
/// complete-domain coverage and exact counts; `C` is a power of two, so coarsening compiles to a
/// bit shift.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowerOfTwoHistogram<const N: usize, O, S> {
    counts: [u32; N],
    samples: u32,
    quality: Quality,
    policy: PhantomData<fn() -> (O, S)>,
}

/// Prospective power-of-two histogram update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PowerOfTwoDelta {
    index: usize,
    count: u32,
    samples: u32,
    quality: Quality,
}

/// Prospective exact power-of-two histogram concatenation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PowerOfTwoMergeDelta<const N: usize> {
    counts: [u32; N],
    samples: u32,
    quality: Quality,
}

impl<const N: usize, O, S: HistogramWord> PowerOfTwoHistogram<N, O, S> {
    /// Creates an empty power-of-two histogram.
    pub const fn new() -> Result<Self, Error> {
        if power_of_two_buckets_are_valid::<N, S>() {
            Ok(Self {
                counts: [0; N],
                samples: 0,
                quality: Quality::Complete,
                policy: PhantomData,
            })
        } else {
            Err(Error::InvalidConfiguration)
        }
    }

    /// Borrows the current bucket counts.
    pub const fn snapshot(&self) -> PowerOfTwoHistogramSnapshot<'_, N, S> {
        PowerOfTwoHistogramSnapshot::from_parts(&self.counts, self.samples, self.quality)
    }

    /// Clears all counts.
    pub const fn reset(&mut self) {
        self.counts = [0; N];
        self.samples = 0;
        self.quality = Quality::Complete;
    }
}

c0nst::c0nst! {
    impl<const N: usize, O, S: HistogramWord> PowerOfTwoHistogram<N, O, S> {
        /// Returns the bucket that `sample` falls in, without integer division.
        pub c0nst fn bucket_index(sample: S) -> Result<usize, Error>
        where
            S: [c0nst] HistogramWord,
        {
            if !power_of_two_buckets_are_valid::<N, S>() {
                return Err(Error::InvalidConfiguration);
            }
            Ok((sample.power_of_two_cell() * N) / S::POWER_OF_TWO_CELLS)
        }

        /// Returns the inclusive sample bounds of bucket `index`, or `None` past the last bucket.
        pub c0nst fn bucket_bounds(index: usize) -> Option<HistogramBounds<S>>
        where
            S: [c0nst] HistogramWord,
        {
            if !power_of_two_buckets_are_valid::<N, S>() || index >= N {
                return None;
            }
            // Bucket `index` holds the cells `c` with `floor(c * N / C) == index`, a contiguous
            // run from `ceil(index * C / N)` to `ceil((index + 1) * C / N) - 1`. `1 <= N <= C`, so
            // the run is non-empty and both cell indices lie in `0..C`.
            let cells = S::POWER_OF_TWO_CELLS;
            let first = (index * cells).div_ceil(N);
            let last = ((index + 1) * cells).div_ceil(N) - 1;
            let lower = match S::power_of_two_cell_bounds(first) {
                Some(bounds) => bounds.lower(),
                None => S::MIN,
            };
            let upper = match S::power_of_two_cell_bounds(last) {
                Some(bounds) => bounds.upper(),
                None => S::MAX,
            };
            Some(HistogramBounds::from_ordered(lower, upper))
        }
    }

    impl<const N: usize, O: OverflowPolicy, S: HistogramWord> PowerOfTwoHistogram<N, O, S> {
        /// Accepts one sample transactionally without integer division.
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

        /// Combines a newer histogram in `O(N)` work.
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
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<PowerOfTwoDelta, Error>
        where
            O: [c0nst] OverflowPolicy,
            S: [c0nst] HistogramWord,
        {
            // `new` validated `1 <= N <= S::POWER_OF_TWO_CELLS`, and `power_of_two_cell() < C`,
            // so `index < N`.
            let index = (sample.power_of_two_cell() * N) / S::POWER_OF_TWO_CELLS;
            let samples = match O::add_u32(self.samples, 1) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let count = match O::add_u32(self.counts[index], 1) {
                Ok(count) => count,
                Err(error) => return Err(error),
            };
            Ok(PowerOfTwoDelta {
                index,
                count: count.value,
                samples: samples.value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated() || count.saturated || samples.saturated,
                ),
            })
        }

        pub(crate) const fn commit(&mut self, delta: PowerOfTwoDelta) {
            self.counts[delta.index] = delta.count;
            self.samples = delta.samples;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare_merge(
            &self,
            newer: &Self,
        ) -> Result<PowerOfTwoMergeDelta<N>, Error>
        where
            O: [c0nst] OverflowPolicy,
        {
            let samples = match O::add_u32(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let mut counts = [0; N];
            let mut saturated =
                self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated;
            let mut index = 0_usize;
            while index < N {
                let count = match O::add_u32(self.counts[index], newer.counts[index]) {
                    Ok(count) => count,
                    Err(error) => return Err(error),
                };
                counts[index] = count.value;
                saturated = saturated || count.saturated;
                index += 1;
            }
            Ok(PowerOfTwoMergeDelta {
                counts,
                samples: samples.value,
                quality: Quality::from_flags(false, saturated),
            })
        }

        pub(crate) const fn commit_merge(&mut self, delta: PowerOfTwoMergeDelta<N>) {
            self.counts = delta.counts;
            self.samples = delta.samples;
            self.quality = delta.quality;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::NonZeroU32;

    use super::{PowerOfTwoHistogram, power_of_two_buckets_are_valid};
    use crate::distribution::{BucketMode, BucketView, ExactDistributionQueries, Quantile};
    use crate::policy::{Checked, Saturating};
    use crate::sample::HistogramWord;
    use crate::{Error, Quality};

    fn qualify_i16_partition<const N: usize>() {
        let mut previous_upper = None;
        for index in 0..N {
            let bounds =
                PowerOfTwoHistogram::<N, Checked, i16>::bucket_bounds(index).expect("bucket");
            if let Some(upper) = previous_upper {
                assert_eq!(i32::from(bounds.lower()), i32::from(upper) + 1);
            } else {
                assert_eq!(bounds.lower(), i16::MIN);
            }
            previous_upper = Some(bounds.upper());
        }
        assert_eq!(previous_upper, Some(i16::MAX));
        for value in i16::MIN..=i16::MAX {
            let index =
                PowerOfTwoHistogram::<N, Checked, i16>::bucket_index(value).expect("valid N");
            let bounds =
                PowerOfTwoHistogram::<N, Checked, i16>::bucket_bounds(index).expect("valid index");
            assert!(value >= bounds.lower() && value <= bounds.upper());
        }
    }

    #[test]
    fn every_supported_capacity_partitions_every_i16_value_exactly_once() {
        macro_rules! qualify {
            ($($n:literal),+ $(,)?) => { $(qualify_i16_partition::<$n>();)+ };
        }
        qualify!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
    }

    #[test]
    fn invalid_capacities_are_rejected_at_every_width() {
        assert!(!power_of_two_buckets_are_valid::<0, i8>());
        assert!(power_of_two_buckets_are_valid::<16, i8>());
        assert!(!power_of_two_buckets_are_valid::<17, i8>());
        assert!(power_of_two_buckets_are_valid::<8, u8>());
        assert!(!power_of_two_buckets_are_valid::<9, u8>());
        assert_eq!(
            PowerOfTwoHistogram::<0, Checked, i16>::new(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            PowerOfTwoHistogram::<33, Checked, i16>::new(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            PowerOfTwoHistogram::<0, Checked, i16>::bucket_index(0),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            PowerOfTwoHistogram::<33, Checked, i16>::bucket_bounds(0),
            None
        );
        assert_eq!(
            PowerOfTwoHistogram::<8, Checked, i16>::bucket_bounds(8),
            None
        );
    }

    fn check_word_lifecycle_and_queries<const N: usize, S: HistogramWord>(stream: [S; 6]) {
        let mut histogram = PowerOfTwoHistogram::<N, Checked, S>::new().expect("valid");
        let empty = histogram.snapshot();
        assert_eq!(empty.bucket_slots(), N);
        assert_eq!(empty.samples(), 0);
        assert_eq!(empty.quality(), Quality::Complete);
        assert_eq!(empty.bucket(N), None);
        assert_eq!(empty.occupancy().expect("valid").occupied, 0);
        assert_eq!(empty.mode(), Ok(None));

        for &sample in &stream {
            histogram.update(sample).expect("representable");
        }
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.samples(), stream.len() as u32);
        assert_eq!(snapshot.counts.iter().sum::<u32>(), stream.len() as u32);
        let mode: BucketMode<S> = snapshot.mode().expect("valid").expect("non-empty");
        assert!(mode.count >= NonZeroU32::MIN);
        assert_eq!(
            Some(mode.values),
            snapshot.bucket_bounds(mode.index).map(Into::into)
        );
        let median = snapshot
            .quantile_bounds(Quantile::new(1, 2).expect("valid"))
            .expect("valid")
            .expect("non-empty");
        let mut sorted = stream;
        sorted.sort_unstable();
        let expected_sample = sorted[median.rank as usize];
        assert!(
            expected_sample >= median.values.lower() && expected_sample <= median.values.upper()
        );
        let tails = snapshot.tail_bounds(S::ZERO).expect("valid");
        let below = stream.iter().filter(|&&v| v <= S::ZERO).count() as u32;
        let above = stream.iter().filter(|&&v| v >= S::ZERO).count() as u32;
        assert!(tails.at_or_below.minimum <= below && below <= tails.at_or_below.maximum);
        assert!(tails.at_or_above.minimum <= above && above <= tails.at_or_above.maximum);

        for split in 0..=stream.len() {
            let mut left = PowerOfTwoHistogram::<N, Checked, S>::new().expect("valid");
            let mut right = PowerOfTwoHistogram::<N, Checked, S>::new().expect("valid");
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
        assert_eq!(histogram.snapshot().counts, &[0; N]);
    }

    #[test]
    fn lifecycle_merges_and_queries_hold_at_every_word() {
        check_word_lifecycle_and_queries::<16, i8>([i8::MIN, -7, -1, 0, 1, i8::MAX]);
        check_word_lifecycle_and_queries::<32, i16>([i16::MIN, -256, -1, 0, 1, i16::MAX]);
        check_word_lifecycle_and_queries::<64, i32>([i32::MIN, -65_536, -1, 0, 1, i32::MAX]);
        check_word_lifecycle_and_queries::<8, u8>([0, 1, 2, 7, 64, u8::MAX]);
        check_word_lifecycle_and_queries::<16, u16>([0, 1, 2, 255, 1024, u16::MAX]);
        check_word_lifecycle_and_queries::<32, u32>([0, 1, 2, 65_535, 1 << 20, u32::MAX]);
    }

    fn power_of_two_fails_transactionally<const N: usize, S: HistogramWord>() {
        let full_samples = PowerOfTwoHistogram::<N, Checked, S> {
            counts: [0; N],
            samples: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_samples;
        assert_eq!(histogram.update(S::MIN), Err(Error::Overflow));
        assert_eq!(histogram, full_samples);

        let full_counts = PowerOfTwoHistogram::<N, Checked, S> {
            counts: [u32::MAX; N],
            samples: 3,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut histogram = full_counts;
        for sample in [S::MIN, S::ZERO, S::MAX] {
            assert_eq!(histogram.update(sample), Err(Error::Overflow));
        }
        assert_eq!(histogram, full_counts);

        let mut other = PowerOfTwoHistogram::<N, Checked, S>::new().expect("valid");
        other.update(S::MAX).expect("representable");
        assert_eq!(histogram.merge(&other), Err(Error::Overflow));
        assert_eq!(histogram, full_counts);

        let mut left_samples_full = full_samples;
        assert_eq!(left_samples_full.merge(&other), Err(Error::Overflow));
        assert_eq!(left_samples_full, full_samples);

        let mut saturating = PowerOfTwoHistogram::<N, Saturating, S> {
            counts: [u32::MAX; N],
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating.update(S::MIN).expect("saturating accepts");
        assert_eq!(saturating.snapshot().quality, Quality::Saturated);

        let mut samples_saturating = PowerOfTwoHistogram::<N, Saturating, S> {
            counts: [0; N],
            samples: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        samples_saturating
            .update(S::MIN)
            .expect("saturating samples accepts");
        assert_eq!(samples_saturating.snapshot().quality, Quality::Saturated);

        let mut sat_left = PowerOfTwoHistogram::<N, Saturating, S> {
            counts: [u32::MAX; N],
            samples: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut sat_right = PowerOfTwoHistogram::<N, Saturating, S>::new().expect("valid");
        sat_right.update(S::MIN).expect("representable");
        sat_left
            .merge(&sat_right)
            .expect("saturating merge accepts");
        assert_eq!(sat_left.snapshot().quality, Quality::Saturated);
        sat_left
            .merge(&sat_right)
            .expect("already saturated merge accepts");
        assert_eq!(sat_left.snapshot().quality, Quality::Saturated);
        sat_left
            .update(S::MIN)
            .expect("already saturated update accepts");
        assert_eq!(sat_left.snapshot().quality, Quality::Saturated);
    }

    #[test]
    fn checked_failure_is_transactional_and_saturation_is_explicit_at_every_width() {
        power_of_two_fails_transactionally::<16, i8>();
        power_of_two_fails_transactionally::<32, i16>();
        power_of_two_fails_transactionally::<64, i32>();
        power_of_two_fails_transactionally::<8, u8>();
        power_of_two_fails_transactionally::<16, u16>();
        power_of_two_fails_transactionally::<32, u32>();
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_power_of_two_histogram_is_const_evaluable_on_nightly() {
        const UPDATED: PowerOfTwoHistogram<8, Checked, i16> = {
            let mut left = match PowerOfTwoHistogram::new() {
                Ok(h) => h,
                Err(_) => unreachable!(),
            };
            let mut right = match PowerOfTwoHistogram::new() {
                Ok(h) => h,
                Err(_) => unreachable!(),
            };
            assert!(left.update(i16::MIN).is_ok());
            assert!(left.update(0).is_ok());
            assert!(right.update(0).is_ok());
            assert!(right.update(i16::MAX).is_ok());
            assert!(left.merge(&right).is_ok());
            left
        };
        const SNAPSHOT: super::PowerOfTwoHistogramSnapshot<'static, 8, i16> = UPDATED.snapshot();
        const ZERO_INDEX: Result<usize, Error> =
            PowerOfTwoHistogram::<8, Checked, i16>::bucket_index(0);
        const BOUNDS_4: Option<crate::histogram::HistogramBounds<i16>> = SNAPSHOT.bucket_bounds(4);
        const OCC: Result<crate::distribution::Occupancy, Error> = SNAPSHOT.occupancy();
        const MODE: Result<Option<BucketMode<i16>>, Error> = SNAPSHOT.mode();
        assert_eq!(ZERO_INDEX, Ok(4));
        assert_eq!(SNAPSHOT.samples, 4);
        assert_eq!(BOUNDS_4.unwrap().lower(), 0);
        assert_eq!(OCC.unwrap().occupied, 3);
        assert_eq!(MODE.unwrap().unwrap().index, 4);
    }
}
