//! Constant-state cumulative mean absolute successive difference.

use core::marker::PhantomData;
use core::num::NonZeroU64;

use crate::arithmetic::UnsignedRatio;
use crate::policy::{Arithmetic, OverflowPolicy};
use crate::sample::SampleWord;
use crate::{Error, Quality};

/// Whether `u64` holds the absolute sum of `u32::MAX` differences of `bits`-bit words.
///
/// Each absolute difference is at most `2^bits - 1`, and the difference count is a `u32`, so the
/// sum is at most `u32::MAX * (2^bits - 1)`. The product is computed in `u128`, which holds it
/// for every `bits <= 64`.
const fn absolute_sum_fits_u64(bits: u32) -> bool {
    bits <= 64 && (u32::MAX as u128) * ((1_u128 << bits) - 1) <= u64::MAX as u128
}

// At 32 bits the bound is `(2^32 - 1)^2 = 2^64 - 2^33 + 1 < 2^64`, the tightest implemented width.
const _: () = assert!(absolute_sum_fits_u64(8));
const _: () = assert!(absolute_sum_fits_u64(16));
const _: () = assert!(absolute_sum_fits_u64(32));
// A 33-bit word would need a wider sum: the proof is not vacuous.
const _: () = assert!(!absolute_sum_fits_u64(33));

/// Cumulative absolute successive-difference result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessiveDifferenceSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Number of defined successive differences.
    pub differences: u32,
    /// Sum of absolute successive differences.
    pub absolute_sum: u64,
    /// First accepted sample, retained for exact interval concatenation.
    pub first_sample: S,
    /// Latest accepted sample, retained for exact interval concatenation.
    pub last_sample: S,
    /// Saturation quality.
    pub quality: Quality,
}

impl<S: SampleWord> SuccessiveDifferenceSnapshot<S> {
    /// Returns the sum of the signed successive differences, `last_sample - first_sample`.
    ///
    /// The signed differences telescope, so their sum is exact without any accumulator. It is
    /// [`SampleWord::Difference`]: `i16`, `i32` or `i64` for 8-, 16- and 32-bit words. A difference
    /// of unsigned words can be negative, so for `u8`, `u16` and `u32` this is a signed type one
    /// size up, never the word itself.
    pub fn net_difference(self) -> S::Difference {
        self.last_sample.difference(self.first_sample)
    }
}

impl<S: Copy> SuccessiveDifferenceSnapshot<S> {
    /// Returns the mean absolute successive difference when at least one difference exists.
    pub const fn mean_absolute(self) -> Option<UnsignedRatio> {
        match NonZeroU64::new(self.differences as u64) {
            Some(denominator) => Some(UnsignedRatio {
                numerator: self.absolute_sum,
                denominator,
            }),
            None => None,
        }
    }
}

/// Cumulative mean-absolute-successive-difference sufficient state.
///
/// `S` may be signed or unsigned. Each absolute difference is computed exactly in the
/// same-width unsigned type and added to a `u64`. That sum cannot overflow before the `u32`
/// difference count does: at most `u32::MAX` differences of at most `2^BITS - 1` each is below
/// `2^64` at every implemented width, 32-bit words included. The bound is asserted at compile
/// time. Both additions still go through the overflow policy, so a checked update past the
/// count's range fails and leaves the state unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessiveDifference<O, S> {
    samples: u32,
    differences: u32,
    absolute_sum: u64,
    first: S,
    previous: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective successive-difference state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuccessiveDifferenceDelta<S> {
    samples: u32,
    differences: u32,
    absolute_sum: u64,
    first: S,
    previous: S,
    quality: Quality,
}

impl<O, S: SampleWord> SuccessiveDifference<O, S> {
    /// Creates empty successive-difference state.
    pub const fn new() -> Self {
        const { assert!(absolute_sum_fits_u64(S::BITS)) };
        Self {
            samples: 0,
            differences: 0,
            absolute_sum: 0,
            first: S::ZERO,
            previous: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }
}

impl<O, S: SampleWord> Default for SuccessiveDifference<O, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: OverflowPolicy, S: SampleWord> SuccessiveDifference<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<SuccessiveDifferenceSnapshot<S>> {
        (self.samples != 0).then_some(SuccessiveDifferenceSnapshot {
            samples: self.samples,
            differences: self.differences,
            absolute_sum: self.absolute_sum,
            first_sample: self.first,
            last_sample: self.previous,
            quality: self.quality,
        })
    }

    /// Clears all cumulative state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Exactly appends a newer sequence summary, including its boundary difference.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    fn prepare_word(&self, sample: S) -> Result<SuccessiveDifferenceDelta<S>, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        let (differences, absolute_sum) = if self.samples == 0 {
            (
                Arithmetic {
                    value: self.differences,
                    saturated: false,
                },
                Arithmetic {
                    value: self.absolute_sum,
                    saturated: false,
                },
            )
        } else {
            // Exact in the same-width unsigned type: at most `2^BITS - 1`.
            let difference = crate::sealed::SampleWord::absolute_difference(sample, self.previous);
            (
                O::add_u32(self.differences, 1)?,
                O::add_u64(self.absolute_sum, difference)?,
            )
        };
        Ok(SuccessiveDifferenceDelta {
            samples: samples.value,
            differences: differences.value,
            absolute_sum: absolute_sum.value,
            first: if self.samples == 0 {
                sample
            } else {
                self.first
            },
            previous: sample,
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || samples.saturated
                    || differences.saturated
                    || absolute_sum.saturated,
            ),
        })
    }

    fn commit_word(&mut self, delta: SuccessiveDifferenceDelta<S>) {
        self.samples = delta.samples;
        self.differences = delta.differences;
        self.absolute_sum = delta.absolute_sum;
        self.first = delta.first;
        self.previous = delta.previous;
        self.quality = delta.quality;
    }

    fn prepare_merge_word(&self, newer: &Self) -> Result<SuccessiveDifferenceDelta<S>, Error> {
        if newer.samples == 0 {
            return Ok(SuccessiveDifferenceDelta {
                samples: self.samples,
                differences: self.differences,
                absolute_sum: self.absolute_sum,
                first: self.first,
                previous: self.previous,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(SuccessiveDifferenceDelta {
                samples: newer.samples,
                differences: newer.differences,
                absolute_sum: newer.absolute_sum,
                first: newer.first,
                previous: newer.previous,
                quality: newer.quality,
            });
        }

        let samples = O::add_u32(self.samples, newer.samples)?;
        let joined_differences = O::add_u32(self.differences, newer.differences)?;
        let differences = O::add_u32(joined_differences.value, 1)?;
        let joined_sum = O::add_u64(self.absolute_sum, newer.absolute_sum)?;
        let boundary = crate::sealed::SampleWord::absolute_difference(newer.first, self.previous);
        let absolute_sum = O::add_u64(joined_sum.value, boundary)?;
        Ok(SuccessiveDifferenceDelta {
            samples: samples.value,
            differences: differences.value,
            absolute_sum: absolute_sum.value,
            first: self.first,
            previous: newer.previous,
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || newer.quality.is_saturated()
                    || samples.saturated
                    || joined_differences.saturated
                    || differences.saturated
                    || joined_sum.saturated
                    || absolute_sum.saturated,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::NonZeroU64;

    use super::{SuccessiveDifference, SuccessiveDifferenceSnapshot, absolute_sum_fits_u64};
    use crate::arithmetic::UnsignedRatio;
    use crate::policy::{Checked, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    #[test]
    fn absolute_sum_bound_is_exact_at_the_edge() {
        let i32_bound = u128::from(u32::MAX) * u128::from(u32::MAX);
        assert!(i32_bound <= u128::from(u64::MAX));
        assert_eq!(u128::from(u64::MAX) - i32_bound, (1_u128 << 33) - 2);
        assert!(absolute_sum_fits_u64(32) && !absolute_sum_fits_u64(33));
    }

    fn check_lifecycle_and_merges<S: SampleWord>(a: S, b: S, c: S) {
        let mut diff = SuccessiveDifference::<Checked, S>::default();
        assert_eq!(diff.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = SuccessiveDifference::<Checked, S>::new();
        diff.merge(&empty).expect("empty merge");
        assert_eq!(diff.snapshot(), None);

        // Single-sample summary has 0 differences and mean_absolute() == None.
        let mut head = SuccessiveDifference::<Checked, S>::new();
        head.update(a).expect("update a");
        diff.merge(&head).expect("merge into empty");
        let single = diff.snapshot().expect("single sample defined");
        assert_eq!(
            single,
            SuccessiveDifferenceSnapshot {
                samples: 1,
                differences: 0,
                absolute_sum: 0,
                first_sample: a,
                last_sample: a,
                quality: Quality::Complete,
            }
        );
        assert_eq!(single.mean_absolute(), None);
        assert_eq!(single.net_difference().into(), 0);

        // Merging an empty newer summary into a non-empty state preserves it.
        diff.merge(&empty).expect("merge empty newer");
        assert_eq!(diff.snapshot(), Some(single));

        // Second sample defines the first successive difference.
        diff.update(b).expect("update b");
        let ab = crate::sealed::SampleWord::absolute_difference(b, a);
        let two = diff.snapshot().expect("two samples defined");
        assert_eq!(
            two.mean_absolute(),
            Some(UnsignedRatio {
                numerator: ab,
                denominator: NonZeroU64::new(1).expect("nonzero"),
            })
        );
        assert_eq!(two.net_difference(), b.difference(a));

        // Merging a tail segment adds its internal differences plus the boundary difference.
        let mut tail = SuccessiveDifference::<Checked, S>::new();
        tail.update(c).expect("update c");
        tail.update(a).expect("update a");
        diff.merge(&tail).expect("merge tail");

        let bc = crate::sealed::SampleWord::absolute_difference(c, b);
        let ca = crate::sealed::SampleWord::absolute_difference(a, c);
        let expected_sum = ab + bc + ca;
        let merged = diff.snapshot().expect("merged defined");
        assert_eq!(
            merged,
            SuccessiveDifferenceSnapshot {
                samples: 4,
                differences: 3,
                absolute_sum: expected_sum,
                first_sample: a,
                last_sample: a,
                quality: Quality::Complete,
            }
        );
        assert_eq!(
            merged.mean_absolute(),
            Some(UnsignedRatio {
                numerator: expected_sum,
                denominator: NonZeroU64::new(3).expect("nonzero"),
            })
        );
        assert_eq!(merged.net_difference().into(), 0);

        diff.reset();
        assert_eq!(diff.snapshot(), None);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_word() {
        check_lifecycle_and_merges::<i8>(20, -40, 70);
        check_lifecycle_and_merges::<i16>(200, -400, 700);
        check_lifecycle_and_merges::<i32>(20_000, -40_000, 70_000);
        check_lifecycle_and_merges::<u8>(90, 10, 200);
        check_lifecycle_and_merges::<u16>(900, 100, 2_000);
        check_lifecycle_and_merges::<u32>(90_000, 10_000, 200_000);
    }

    fn count_fails_before_the_sum<S: SampleWord>() {
        let swing: i64 = S::MAX.difference(S::MIN).into();
        let swing = swing.unsigned_abs();
        let mut difference = SuccessiveDifference::<Checked, S> {
            samples: u32::MAX,
            differences: u32::MAX - 1,
            absolute_sum: swing * u64::from(u32::MAX - 1),
            first: S::MIN,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = difference;
        assert_eq!(difference.update(S::MAX), Err(Error::Overflow));
        assert_eq!(difference, before);

        difference.samples = u32::MAX - 1;
        difference
            .update(S::MAX)
            .expect("the bound holds at u32::MAX differences");
        let snapshot = difference.snapshot().expect("defined");
        assert_eq!(snapshot.differences, u32::MAX);
        assert_eq!(snapshot.absolute_sum, swing * u64::from(u32::MAX));
        assert_eq!(snapshot.quality, Quality::Complete);
    }

    #[test]
    fn count_fails_before_the_sum_at_every_width() {
        count_fails_before_the_sum::<i8>();
        count_fails_before_the_sum::<i16>();
        count_fails_before_the_sum::<i32>();
        count_fails_before_the_sum::<u8>();
        count_fails_before_the_sum::<u16>();
        count_fails_before_the_sum::<u32>();
    }

    fn every_width_fails_transactionally<S: SampleWord>() {
        let mut right_one = SuccessiveDifference::<Checked, S>::new();
        right_one.update(S::MAX).expect("representable");

        let mut right_two = SuccessiveDifference::<Checked, S>::new();
        right_two.update(S::MIN).expect("representable");
        right_two.update(S::MAX).expect("representable");

        // Sample-count overflow on update and merge.
        let samples_full = SuccessiveDifference::<Checked, S> {
            samples: u32::MAX,
            differences: 10,
            absolute_sum: 10,
            first: S::ZERO,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut state = samples_full;
        assert_eq!(state.update(S::MAX), Err(Error::Overflow));
        assert_eq!(state, samples_full);
        assert_eq!(state.merge(&right_one), Err(Error::Overflow));
        assert_eq!(state, samples_full);

        // Differences-count overflow on update, joined_differences, and boundary difference + 1.
        let diffs_full = SuccessiveDifference::<Checked, S> {
            samples: 10,
            differences: u32::MAX,
            absolute_sum: 10,
            first: S::ZERO,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut state = diffs_full;
        assert_eq!(state.update(S::MAX), Err(Error::Overflow));
        assert_eq!(state, diffs_full);
        assert_eq!(state.merge(&right_one), Err(Error::Overflow));
        assert_eq!(state, diffs_full);
        assert_eq!(state.merge(&right_two), Err(Error::Overflow));
        assert_eq!(state, diffs_full);

        // Sum overflow on update, joined_sum, and boundary sum addition.
        let sum_full = SuccessiveDifference::<Checked, S> {
            samples: 2,
            differences: 1,
            absolute_sum: u64::MAX,
            first: S::ZERO,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut state = sum_full;
        assert_eq!(state.update(S::MAX), Err(Error::Overflow));
        assert_eq!(state, sum_full);
        assert_eq!(state.merge(&right_one), Err(Error::Overflow));
        assert_eq!(state, sum_full);
        assert_eq!(state.merge(&right_two), Err(Error::Overflow));
        assert_eq!(state, sum_full);

        // Saturating update and merge each independently mark quality from Quality::Complete.
        let mut saturating_count = SuccessiveDifference::<Saturating, S> {
            samples: u32::MAX,
            differences: 1,
            absolute_sum: 0,
            first: S::ZERO,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_count.update(S::MIN).expect("saturating count");
        let snapshot = saturating_count.snapshot().expect("defined");
        assert_eq!(snapshot.samples, u32::MAX);
        assert_eq!(snapshot.differences, 2);
        assert_eq!(snapshot.absolute_sum, 0);
        assert_eq!(snapshot.quality, Quality::Saturated);

        let mut saturating_sum = SuccessiveDifference::<Saturating, S> {
            samples: 2,
            differences: 1,
            absolute_sum: u64::MAX,
            first: S::ZERO,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_sum.update(S::MAX).expect("saturating sum");
        let snapshot = saturating_sum.snapshot().expect("defined");
        assert_eq!(snapshot.samples, 3);
        assert_eq!(snapshot.differences, 2);
        assert_eq!(snapshot.absolute_sum, u64::MAX);
        assert_eq!(snapshot.quality, Quality::Saturated);

        let mut saturating_merge = SuccessiveDifference::<Saturating, S> {
            samples: 2,
            differences: 1,
            absolute_sum: u64::MAX,
            first: S::ZERO,
            previous: S::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut tail = SuccessiveDifference::<Saturating, S>::new();
        tail.update(S::MIN).expect("tail min");
        saturating_merge.merge(&tail).expect("saturating merge");
        assert_eq!(
            saturating_merge.snapshot().map(|s| (
                s.samples,
                s.differences,
                s.absolute_sum,
                s.last_sample,
                s.quality
            )),
            Some((3, 2, u64::MAX, S::MIN, Quality::Saturated))
        );
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        every_width_fails_transactionally::<i8>();
        every_width_fails_transactionally::<i16>();
        every_width_fails_transactionally::<i32>();
        every_width_fails_transactionally::<u8>();
        every_width_fails_transactionally::<u16>();
        every_width_fails_transactionally::<u32>();
    }
}
