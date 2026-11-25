//! Constant-state cumulative mean absolute successive difference.

use core::marker::PhantomData;
use core::num::NonZeroU64;

use crate::arithmetic::UnsignedRatio;
use crate::policy::{Arithmetic, CounterWord, OverflowPolicy};
use crate::sample::SampleWord;
use crate::{Error, Quality};

/// Returns whether the `u64` absolute sum holds every difference a `counter_bits` counter can
/// count for `bits`-bit words. [`SuccessiveDifference::new`] asserts this at compile time.
///
/// Each absolute difference is at most `2^bits - 1`, and the difference count is at most
/// `2^counter_bits - 1`, so the sum is at most their product. The product is computed in `u128`,
/// which holds it for every `bits` and `counter_bits` in `1..=64`. Zero-bit words and counters
/// are rejected first so the shift amounts stay below 64.
pub const fn successive_difference_holds(bits: u32, counter_bits: u32) -> bool {
    bits >= 1
        && bits <= 64
        && counter_bits >= 1
        && counter_bits <= 64
        && ((u64::MAX >> (64 - counter_bits)) as u128) * ((u64::MAX >> (64 - bits)) as u128)
            <= u64::MAX as u128
}

// At 32-bit words with a `u32` counter the bound is `(2^32 - 1)^2 = 2^64 - 2^33 + 1 < 2^64`, the
// tightest supported pair. Every word holds a `u8`, `u16`, or `u32` count.
const _: () = assert!(successive_difference_holds(8, 32));
const _: () = assert!(successive_difference_holds(16, 32));
const _: () = assert!(successive_difference_holds(32, 32));
// A 33-bit word or a 33-bit counter would need a wider sum: the proof is not vacuous.
const _: () = assert!(!successive_difference_holds(33, 32));
const _: () = assert!(!successive_difference_holds(32, 33));
// No word holds a `u64` count: even 8-bit words reach `2^72` in the sum.
const _: () = assert!(!successive_difference_holds(8, 64));

/// Cumulative absolute successive-difference result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessiveDifferenceSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Number of defined successive differences.
    pub differences: C,
    /// Sum of absolute successive differences.
    pub absolute_sum: u64,
    /// First accepted sample, retained for exact interval concatenation.
    pub first_sample: S,
    /// Latest accepted sample, retained for exact interval concatenation.
    pub last_sample: S,
    /// Saturation quality.
    pub quality: Quality,
}

c0nst::c0nst! {
    impl<C: Copy, S: SampleWord> SuccessiveDifferenceSnapshot<C, S> {
        /// Returns the sum of the signed successive differences, `last_sample - first_sample`.
        ///
        /// The signed differences telescope, so their sum is exact without any accumulator. It is
        /// [`SampleWord::Difference`]: `i16`, `i32` or `i64` for 8-, 16- and 32-bit words. A difference
        /// of unsigned words can be negative, so for `u8`, `u16` and `u32` this is a signed type one
        /// size up, never the word itself.
        pub c0nst fn net_difference(self) -> S::Difference
        where
            S: [c0nst] SampleWord,
        {
            self.last_sample.difference(self.first_sample)
        }
    }

    impl<C: CounterWord, S: Copy> SuccessiveDifferenceSnapshot<C, S> {
        /// Returns the mean absolute successive difference when at least one difference exists.
        pub c0nst fn mean_absolute(self) -> Option<UnsignedRatio>
        where
            C: [c0nst] CounterWord,
        {
            match NonZeroU64::new(self.differences.to_u64()) {
                Some(denominator) => Some(UnsignedRatio {
                    numerator: self.absolute_sum,
                    denominator,
                }),
                None => None,
            }
        }
    }
}

/// Cumulative mean-absolute-successive-difference sufficient state.
///
/// `C` is the counter word of the sample and difference counts, and `S` is the sample word
/// (signed or unsigned). Each absolute difference is computed exactly in the same-width unsigned
/// type and added to a `u64`. That sum cannot overflow before the difference count does when
/// [`successive_difference_holds`] accepts `S::BITS` and `C::BITS`: `C::MAX` differences of at
/// most `2^BITS - 1` each stay below `2^64`. Every word holds a `u8`, `u16`, or `u32` count, and
/// none holds a `u64` count, so [`SuccessiveDifference::new`] rejects `u64` counters at compile
/// time:
///
/// ```compile_fail,E0080
/// use krabilorean::online::SuccessiveDifference;
/// use krabilorean::policy::Checked;
///
/// // A `u64` count of `i16` differences can exceed the `u64` absolute sum.
/// let _ = SuccessiveDifference::<Checked, u64, i16>::new();
/// ```
///
/// Both additions still go through the overflow policy, so a checked update past the count's
/// range fails and leaves the state unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessiveDifference<O, C, S> {
    samples: C,
    differences: C,
    absolute_sum: u64,
    first: S,
    previous: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective successive-difference state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuccessiveDifferenceDelta<C, S> {
    samples: C,
    differences: C,
    absolute_sum: u64,
    first: S,
    previous: S,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleWord> SuccessiveDifference<O, C, S> {
    /// Creates empty successive-difference state.
    ///
    /// Fails to compile when [`successive_difference_holds`] rejects the word and counter widths.
    pub const fn new() -> Self {
        const { assert!(successive_difference_holds(S::BITS, C::BITS)) };
        Self {
            samples: C::ZERO,
            differences: C::ZERO,
            absolute_sum: 0,
            first: S::ZERO,
            previous: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Clears all cumulative state.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
}

impl<O, C: CounterWord, S: SampleWord> Default for SuccessiveDifference<O, C, S> {
    fn default() -> Self {
        Self::new()
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: SampleWord> SuccessiveDifference<O, C, S> {
        /// Accepts one sample transactionally.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            let delta = match self.prepare_word(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_word(delta);
            Ok(())
        }

        /// Returns `None` until the first sample is accepted.
        pub c0nst fn snapshot(&self) -> Option<SuccessiveDifferenceSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.non_zero().is_none() {
                None
            } else {
                Some(SuccessiveDifferenceSnapshot {
                    samples: self.samples,
                    differences: self.differences,
                    absolute_sum: self.absolute_sum,
                    first_sample: self.first,
                    last_sample: self.previous,
                    quality: self.quality,
                })
            }
        }

        /// Exactly appends a newer sequence summary, including its boundary difference.
        #[allow(clippy::question_mark)]
        pub c0nst fn merge(&mut self, newer: &Self) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            let delta = match self.prepare_merge_word(newer) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_word(delta);
            Ok(())
        }

        #[allow(clippy::question_mark)]
        c0nst fn prepare_word(&self, sample: S) -> Result<SuccessiveDifferenceDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let is_first = self.samples.non_zero().is_none();
            let (differences, absolute_sum) = if is_first {
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
                let difference =
                    crate::sealed::SampleWord::absolute_difference(sample, self.previous);
                let diffs = match C::add::<O>(self.differences, C::ONE) {
                    Ok(diffs) => diffs,
                    Err(error) => return Err(error),
                };
                let sum = match O::add_u64(self.absolute_sum, difference) {
                    Ok(sum) => sum,
                    Err(error) => return Err(error),
                };
                (diffs, sum)
            };
            Ok(SuccessiveDifferenceDelta {
                samples: samples.value,
                differences: differences.value,
                absolute_sum: absolute_sum.value,
                first: if is_first { sample } else { self.first },
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

        const fn commit_word(&mut self, delta: SuccessiveDifferenceDelta<C, S>) {
            self.samples = delta.samples;
            self.differences = delta.differences;
            self.absolute_sum = delta.absolute_sum;
            self.first = delta.first;
            self.previous = delta.previous;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<SuccessiveDifferenceDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            if newer.samples.non_zero().is_none() {
                return Ok(SuccessiveDifferenceDelta {
                    samples: self.samples,
                    differences: self.differences,
                    absolute_sum: self.absolute_sum,
                    first: self.first,
                    previous: self.previous,
                    quality: self.quality,
                });
            }
            if self.samples.non_zero().is_none() {
                return Ok(SuccessiveDifferenceDelta {
                    samples: newer.samples,
                    differences: newer.differences,
                    absolute_sum: newer.absolute_sum,
                    first: newer.first,
                    previous: newer.previous,
                    quality: newer.quality,
                });
            }

            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let joined_differences = match C::add::<O>(self.differences, newer.differences) {
                Ok(diffs) => diffs,
                Err(error) => return Err(error),
            };
            let differences = match C::add::<O>(joined_differences.value, C::ONE) {
                Ok(diffs) => diffs,
                Err(error) => return Err(error),
            };
            let joined_sum = match O::add_u64(self.absolute_sum, newer.absolute_sum) {
                Ok(sum) => sum,
                Err(error) => return Err(error),
            };
            let boundary =
                crate::sealed::SampleWord::absolute_difference(newer.first, self.previous);
            let absolute_sum = match O::add_u64(joined_sum.value, boundary) {
                Ok(sum) => sum,
                Err(error) => return Err(error),
            };
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
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::mem::size_of;
    use core::num::NonZeroU64;

    use super::{SuccessiveDifference, SuccessiveDifferenceSnapshot, successive_difference_holds};
    use crate::arithmetic::UnsignedRatio;
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    #[test]
    fn absolute_sum_bound_is_exact_at_the_edge() {
        let i32_bound = u128::from(u32::MAX) * u128::from(u32::MAX);
        assert!(i32_bound <= u128::from(u64::MAX));
        assert_eq!(u128::from(u64::MAX) - i32_bound, (1_u128 << 33) - 2);
        assert!(successive_difference_holds(32, 32));
        assert!(!successive_difference_holds(33, 32));
        assert!(!successive_difference_holds(32, 33));
        assert!(!successive_difference_holds(0, 32));
        assert!(!successive_difference_holds(32, 0));
        assert!(!successive_difference_holds(65, 32));
        assert!(!successive_difference_holds(32, 65));
    }

    fn check_lifecycle_and_merges<C: CounterWord, S: SampleWord>(a: S, b: S, c: S) {
        let mut diff = SuccessiveDifference::<Checked, C, S>::default();
        assert_eq!(diff.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = SuccessiveDifference::<Checked, C, S>::new();
        diff.merge(&empty).expect("empty merge");
        assert_eq!(diff.snapshot(), None);

        // Single-sample summary has 0 differences and mean_absolute() == None.
        let mut head = SuccessiveDifference::<Checked, C, S>::new();
        head.update(a).expect("update a");
        diff.merge(&head).expect("merge into empty");
        let single = diff.snapshot().expect("single sample defined");
        assert_eq!(
            single,
            SuccessiveDifferenceSnapshot {
                samples: C::ONE,
                differences: C::ZERO,
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
        let mut tail = SuccessiveDifference::<Checked, C, S>::new();
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
                samples: C::from_u64(4).expect("four fits"),
                differences: C::from_u64(3).expect("three fits"),
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

    fn lifecycle_for_counter<C: CounterWord>() {
        check_lifecycle_and_merges::<C, i8>(20, -40, 70);
        check_lifecycle_and_merges::<C, i16>(200, -400, 700);
        check_lifecycle_and_merges::<C, i32>(20_000, -40_000, 70_000);
        check_lifecycle_and_merges::<C, u8>(90, 10, 200);
        check_lifecycle_and_merges::<C, u16>(900, 100, 2_000);
        check_lifecycle_and_merges::<C, u32>(90_000, 10_000, 200_000);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_counter_and_sample_word() {
        lifecycle_for_counter::<u8>();
        lifecycle_for_counter::<u16>();
        lifecycle_for_counter::<u32>();
    }

    fn count_fails_before_the_sum<C: CounterWord, S: SampleWord>() {
        let swing: i64 = S::MAX.difference(S::MIN).into();
        let swing = swing.unsigned_abs();
        let max_less_one = C::MAX.checked_sub(C::ONE).expect("counter max >= 1");
        let mut difference = SuccessiveDifference::<Checked, C, S> {
            samples: C::MAX,
            differences: max_less_one,
            absolute_sum: swing * max_less_one.to_u64(),
            first: S::MIN,
            previous: S::MIN,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = difference;
        assert_eq!(difference.update(S::MAX), Err(Error::Overflow));
        assert_eq!(difference, before);

        difference.samples = max_less_one;
        difference
            .update(S::MAX)
            .expect("the bound holds at C::MAX differences");
        let snapshot = difference.snapshot().expect("defined");
        assert_eq!(snapshot.differences, C::MAX);
        assert_eq!(snapshot.absolute_sum, swing * C::MAX.to_u64());
        assert_eq!(snapshot.quality, Quality::Complete);
    }

    fn count_fails_for_counter<C: CounterWord>() {
        count_fails_before_the_sum::<C, i8>();
        count_fails_before_the_sum::<C, i16>();
        count_fails_before_the_sum::<C, i32>();
        count_fails_before_the_sum::<C, u8>();
        count_fails_before_the_sum::<C, u16>();
        count_fails_before_the_sum::<C, u32>();
    }

    #[test]
    fn count_fails_before_the_sum_at_every_counter_and_sample_word() {
        count_fails_for_counter::<u8>();
        count_fails_for_counter::<u16>();
        count_fails_for_counter::<u32>();
    }

    fn every_width_fails_transactionally<S: SampleWord>() {
        let mut right_one = SuccessiveDifference::<Checked, u32, S>::new();
        right_one.update(S::MAX).expect("representable");

        let mut right_two = SuccessiveDifference::<Checked, u32, S>::new();
        right_two.update(S::MIN).expect("representable");
        right_two.update(S::MAX).expect("representable");

        // Sample-count overflow on update and merge.
        let samples_full = SuccessiveDifference::<Checked, u32, S> {
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
        let diffs_full = SuccessiveDifference::<Checked, u32, S> {
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
        let sum_full = SuccessiveDifference::<Checked, u32, S> {
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
        let mut saturating_count = SuccessiveDifference::<Saturating, u32, S> {
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

        let mut saturating_sum = SuccessiveDifference::<Saturating, u32, S> {
            samples: 2,
            differences: 1,
            // One below the limit, so the next difference is what saturates the sum.
            absolute_sum: u64::MAX - 1,
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

        let mut saturating_merge = SuccessiveDifference::<Saturating, u32, S> {
            samples: 2,
            differences: 1,
            // One below the limit, so the next difference is what saturates the sum.
            absolute_sum: u64::MAX - 1,
            first: S::ZERO,
            previous: S::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut tail = SuccessiveDifference::<Saturating, u32, S>::new();
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

    #[test]
    fn narrower_counters_shrink_state_size() {
        assert!(
            size_of::<SuccessiveDifference<Checked, u8, i16>>()
                < size_of::<SuccessiveDifference<Checked, u32, i16>>()
        );
        assert!(
            size_of::<SuccessiveDifference<Checked, u16, i16>>()
                <= size_of::<SuccessiveDifference<Checked, u32, i16>>()
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_successive_difference_is_const_evaluable_on_nightly() {
        const SNAPSHOT: SuccessiveDifferenceSnapshot<u16, u16> = {
            let mut diff = SuccessiveDifference::<Checked, u16, u16>::new();
            assert!(diff.update(25).is_ok());
            assert!(diff.update(15).is_ok());
            let mut tail = SuccessiveDifference::<Checked, u16, u16>::new();
            assert!(tail.update(5).is_ok());
            assert!(diff.merge(&tail).is_ok());
            match diff.snapshot() {
                Some(s) => s,
                None => unreachable!(),
            }
        };
        const NET: i32 = SNAPSHOT.net_difference();
        const MEAN: Option<UnsignedRatio> = SNAPSHOT.mean_absolute();
        assert_eq!(NET, -20);
        assert_eq!(
            MEAN,
            Some(UnsignedRatio {
                numerator: 20,
                denominator: NonZeroU64::new(2).unwrap(),
            })
        );
    }
}
