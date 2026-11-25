//! Constant-state cumulative raw moments, mean, and exact population variance.

use core::marker::PhantomData;
use core::num::NonZeroU64;

use crate::arithmetic::UnsignedWideRatio;
use crate::policy::{Accumulator, CounterWord, OverflowPolicy};
use crate::sample::{
    MomentWord, square_sum_bits, sum_bits, unsigned_square_sum_bits, unsigned_sum_bits,
};
use crate::sealed::{SquareSumOps, SumOps};
use crate::{Error, Quality};

/// Widest counter, in bits, whose maximum count's square fits the `u64` variance denominator
/// `N^2`.
const MAX_VARIANCE_COUNTER_BITS: u32 = u64::BITS / 2;

/// Returns whether a counter of `counter_bits` bits keeps the raw moment sums of `S` samples and
/// the population variance's numerator and denominator inside their types.
/// [`RunningMoments::new`] asserts this at compile time.
///
/// The sums require `counter_bits <= S::MAX_COUNTER_BITS`, and the `u64` denominator `N^2`
/// requires `counter_bits <= 32`. The `u128` numerator is proven at 32 counter bits and needs
/// fewer bits for a narrower counter. Every word therefore accepts `u8`, `u16`, and `u32`
/// counters, and rejects `u64` counters and zero-bit counters.
pub const fn raw_moments_hold<S: MomentWord>(counter_bits: u32) -> bool {
    counter_bits >= 1
        && counter_bits <= S::MAX_COUNTER_BITS
        && counter_bits <= MAX_VARIANCE_COUNTER_BITS
}

/// Bits that hold the population-variance numerator `N * sum(x^2) - sum(x)^2` of
/// `2^counter_bits - 1` words of `bits` bits, and both operands of its subtraction.
///
/// `N < 2^counter_bits` and `sum(x^2) < 2^square_sum_bits(bits, counter_bits)`, so
/// `N * sum(x^2)` is below `2^(counter_bits + square_sum_bits(bits, counter_bits))`. `sum(x)^2`
/// has the same bound `2^(2 * (sum_bits(bits, counter_bits) - 1))`. The numerator is non-negative
/// and at most `N * sum(x^2)`.
const fn variance_numerator_bits(bits: u32, counter_bits: u32) -> u32 {
    counter_bits + square_sum_bits(bits, counter_bits)
}

// The numerator is computed in `u128` at every width. At `i32` with the widest accepted counter
// it needs 126 bits, the tightest implemented pair.
const _: () = assert!(variance_numerator_bits(8, 32) == 2 * (sum_bits(8, 32) - 1));
const _: () = assert!(variance_numerator_bits(16, 32) == 2 * (sum_bits(16, 32) - 1));
const _: () = assert!(variance_numerator_bits(32, 32) == 2 * (sum_bits(32, 32) - 1));
const _: () = assert!(variance_numerator_bits(8, MAX_VARIANCE_COUNTER_BITS) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(16, MAX_VARIANCE_COUNTER_BITS) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(32, MAX_VARIANCE_COUNTER_BITS) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(32, 32) == 126);
// A 34-bit word would not fit: the proof is not vacuous.
const _: () = assert!(variance_numerator_bits(34, 32) > u128::BITS);

/// Bits that hold the population-variance numerator of `2^counter_bits - 1` unsigned words of
/// `bits` bits, and both operands of its subtraction:
/// `N * sum(x^2) < 2^(counter_bits + unsigned_square_sum_bits(bits, counter_bits))` and
/// `sum(x)^2 < 2^(2 * unsigned_sum_bits(bits, counter_bits))`.
const fn unsigned_variance_numerator_bits(bits: u32, counter_bits: u32) -> u32 {
    counter_bits + unsigned_square_sum_bits(bits, counter_bits)
}

// `u32` with a `u32` counter needs all 128 bits of the `u128` numerator, the tightest unsigned
// pair.
const _: () = assert!(unsigned_variance_numerator_bits(8, 32) == 2 * unsigned_sum_bits(8, 32));
const _: () = assert!(unsigned_variance_numerator_bits(16, 32) == 2 * unsigned_sum_bits(16, 32));
const _: () = assert!(unsigned_variance_numerator_bits(32, 32) == 2 * unsigned_sum_bits(32, 32));
const _: () = assert!(unsigned_variance_numerator_bits(8, MAX_VARIANCE_COUNTER_BITS) <= u128::BITS);
const _: () =
    assert!(unsigned_variance_numerator_bits(16, MAX_VARIANCE_COUNTER_BITS) <= u128::BITS);
const _: () = assert!(unsigned_variance_numerator_bits(32, 32) == u128::BITS);
const _: () = assert!(unsigned_variance_numerator_bits(33, 32) > u128::BITS);

/// Cumulative count, sum, and sum of squares.
///
/// `C` is the counter word and `S` is the sample word. The sum is [`MomentWord::Sum`] and the
/// square sum [`MomentWord::SquareSum`], each proven to hold `C::MAX` samples: `i64` and `u64` at
/// `i8` and `i16`, `i64` and `u128` at `i32`, `u64` and `u64` at `u8` and `u16`, and `u64` and
/// `u128` at `u32`. `mean` is the word's [`MomentWord::Mean`]: a
/// [`SignedRatio`](crate::arithmetic::SignedRatio) for a signed word and an
/// [`UnsignedRatio`](crate::arithmetic::UnsignedRatio) for an unsigned one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawMomentsSnapshot<C: CounterWord, S: MomentWord> {
    /// Accepted sample count.
    pub samples: C::NonZero,
    /// Sum of accepted samples.
    pub sum: S::Sum,
    /// Sum of squared accepted samples.
    pub sum_of_squares: S::SquareSum,
    /// Saturation quality.
    pub quality: Quality,
}

const _: () = assert!(
    core::mem::size_of::<Option<RawMomentsSnapshot<u8, i16>>>()
        == core::mem::size_of::<RawMomentsSnapshot<u8, i16>>()
        && core::mem::size_of::<Option<RawMomentsSnapshot<u16, i16>>>()
            == core::mem::size_of::<RawMomentsSnapshot<u16, i16>>()
        && core::mem::size_of::<Option<RawMomentsSnapshot<u32, i16>>>()
            == core::mem::size_of::<RawMomentsSnapshot<u32, i16>>()
        && core::mem::size_of::<Option<RawMomentsSnapshot<u64, i16>>>()
            == core::mem::size_of::<RawMomentsSnapshot<u64, i16>>()
);

/// Cumulative raw moments with policy-controlled overflow.
///
/// `C` is the counter word of the sample count, and `O` rejects or saturates an update that `C`
/// cannot count. `S` is the sample word. Its sum and square sum are the word's proven
/// [`MomentWord::Sum`] and [`MomentWord::SquareSum`], so neither can overflow before the sample
/// count does when [`raw_moments_hold`] accepts the counter (`u8`, `u16`, or `u32`). No word's
/// sums hold a `u64` count, so [`RunningMoments::new`] rejects it at compile time:
///
/// ```compile_fail,E0080
/// use krabilorean::online::RunningMoments;
/// use krabilorean::policy::Checked;
///
/// // `u64::MAX` samples of `i16` overflow the `i64` sum.
/// let _ = RunningMoments::<Checked, u64, i16>::new();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunningMoments<O, C, S: MomentWord> {
    samples: C,
    sum: S::Sum,
    sum_of_squares: S::SquareSum,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective raw-moment state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawMomentsDelta<C, S: MomentWord> {
    samples: C,
    sum: S::Sum,
    sum_of_squares: S::SquareSum,
    quality: Quality,
}

impl<O, C: CounterWord, S: MomentWord> RunningMoments<O, C, S> {
    /// Creates empty cumulative raw moments.
    ///
    /// Fails to compile when [`raw_moments_hold`] rejects the counter word for `S`.
    pub const fn new() -> Self {
        const { assert!(raw_moments_hold::<S>(C::BITS)) };
        Self {
            samples: C::ZERO,
            sum: S::ZERO_SUM,
            sum_of_squares: S::ZERO_SQUARE_SUM,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Clears all cumulative state.
    pub const fn reset(&mut self) {
        self.samples = C::ZERO;
        self.sum = <S as MomentWord>::ZERO_SUM;
        self.sum_of_squares = <S as MomentWord>::ZERO_SQUARE_SUM;
        self.quality = Quality::Complete;
    }
}

impl<O, C: CounterWord, S: MomentWord> Default for RunningMoments<O, C, S> {
    fn default() -> Self {
        Self::new()
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: MomentWord> RunningMoments<O, C, S> {
        /// Accepts one sample transactionally.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] Accumulator + [c0nst] SumOps<S::SquareSum, S::Mean>,
            S::SquareSum: [c0nst] Accumulator,
        {
            let delta = match self.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit(delta);
            Ok(())
        }

        /// Returns `None` until the first sample is accepted.
        pub c0nst fn snapshot(&self) -> Option<RawMomentsSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            match self.samples.non_zero() {
                Some(samples) => Some(RawMomentsSnapshot {
                    samples,
                    sum: self.sum,
                    sum_of_squares: self.sum_of_squares,
                    quality: self.quality,
                }),
                None => None,
            }
        }

        /// Exactly appends newer raw moments, transactionally.
        #[allow(clippy::question_mark)]
        pub c0nst fn merge(&mut self, newer: &Self) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] Accumulator,
            S::SquareSum: [c0nst] Accumulator,
        {
            let delta = match self.prepare_merge(newer) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_merge(delta);
            Ok(())
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<RawMomentsDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] Accumulator + [c0nst] SumOps<S::SquareSum, S::Mean>,
            S::SquareSum: [c0nst] Accumulator,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let widened = <S as MomentWord>::to_sum(sample);
            let sum = match Accumulator::add::<O>(self.sum, widened) {
                Ok(sum) => sum,
                Err(error) => return Err(error),
            };
            // Signed: at most `MIN^2 = 2^(2 * BITS - 2) <= 2^62`, so the `i64` product is
            // exact. Unsigned: at most `(2^BITS - 1)^2 < 2^64`, so the `u64` product is exact.
            let square = <<S as MomentWord>::Sum as SumOps<_, _>>::square(
                widened,
                crate::sealed::Internal::new(),
            );
            let sum_of_squares = match Accumulator::add::<O>(self.sum_of_squares, square) {
                Ok(squares) => squares,
                Err(error) => return Err(error),
            };
            Ok(RawMomentsDelta {
                samples: samples.value,
                sum: sum.value,
                sum_of_squares: sum_of_squares.value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated()
                        || samples.saturated
                        || sum.saturated
                        || sum_of_squares.saturated,
                ),
            })
        }

        pub(crate) const fn commit(&mut self, delta: RawMomentsDelta<C, S>) {
            self.samples = delta.samples;
            self.sum = delta.sum;
            self.sum_of_squares = delta.sum_of_squares;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare_merge(&self, newer: &Self) -> Result<RawMomentsDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] Accumulator,
            S::SquareSum: [c0nst] Accumulator,
        {
            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let sum = match Accumulator::add::<O>(self.sum, newer.sum) {
                Ok(sum) => sum,
                Err(error) => return Err(error),
            };
            let sum_of_squares = match Accumulator::add::<O>(self.sum_of_squares, newer.sum_of_squares) {
                Ok(squares) => squares,
                Err(error) => return Err(error),
            };
            Ok(RawMomentsDelta {
                samples: samples.value,
                sum: sum.value,
                sum_of_squares: sum_of_squares.value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated()
                        || newer.quality.is_saturated()
                        || samples.saturated
                        || sum.saturated
                        || sum_of_squares.saturated,
                ),
            })
        }

        pub(crate) const fn commit_merge(&mut self, delta: RawMomentsDelta<C, S>) {
            self.commit(delta);
        }
    }

    impl<C: CounterWord, S: MomentWord> RawMomentsSnapshot<C, S> {
        /// Returns the cumulative mean as an exact ratio, or `None` unless the raw moments are
        /// complete.
        ///
        /// Under [`Saturating`](crate::policy::Saturating) the count and the sum saturate
        /// separately, so once either has, `sum / samples` is not the mean of the input.
        pub c0nst fn mean(self) -> Option<<S as MomentWord>::Mean>
        where
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] SumOps<S::SquareSum, S::Mean>,
        {
            if !self.quality.is_complete() {
                return None;
            }
            // Every accepted counter fits in `NonZeroU32`; a hand-built snapshot with a larger
            // `u64` count has no exact 32-bit denominator ratio.
            let Some(samples) = C::non_zero_u32(self.samples) else {
                return None;
            };
            Some(<<S as MomentWord>::Sum as SumOps<_, _>>::mean(
                self.sum,
                samples,
            ))
        }

        /// Returns exact cumulative population variance when raw moments are complete and
        /// consistent.
        ///
        /// The numerator is `N * sum(x^2) - sum(x)^2`, which fits `u128` at every width for
        /// any state the family produces; the denominator is `N^2`, which fits `u64`. An
        /// inconsistent hand-built snapshot whose terms don't fit returns `None`.
        /// Saturated or approximate raw moments return `None` because the exact variance cannot be
        /// recovered from non-exact sums.
        #[allow(clippy::question_mark)]
        pub c0nst fn population_variance(self) -> Option<UnsignedWideRatio>
        where
            C: [c0nst] CounterWord,
            S: [c0nst] MomentWord,
            S::Sum: [c0nst] SumOps<S::SquareSum, S::Mean>,
            S::SquareSum: [c0nst] SquareSumOps,
        {
            if !self.quality.is_complete() {
                return None;
            }
            // `N^2` must fit the `u64` denominator, so the count must fit `u32`.
            let samples = match C::non_zero_u32(self.samples) {
                Some(samples) => samples,
                None => return None,
            };
            let scaled_squares = match SquareSumOps::scaled(self.sum_of_squares, samples) {
                Some(value) => value,
                None => return None,
            };
            let magnitude = <<S as MomentWord>::Sum as SumOps<_, _>>::magnitude(self.sum);
            let sum_square = magnitude * magnitude;
            let numerator = match scaled_squares.checked_sub(sum_square) {
                Some(value) => value,
                None => return None,
            };
            // `samples <= u32::MAX`, so `samples^2 < 2^64` is exact in `u64` and non-zero.
            // Squaring in `u64` before `NonZeroU64::new` preserves the 32-bit upper bound so
            // the multiplication needs no 64-bit overflow check.
            let samples = samples.get() as u64;
            let denominator = match NonZeroU64::new(samples * samples) {
                Some(denominator) => denominator,
                None => NonZeroU64::MIN,
            };
            Some(UnsignedWideRatio {
                numerator,
                denominator,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::{NonZeroU32, NonZeroU64};

    use super::{
        RawMomentsSnapshot, RunningMoments, raw_moments_hold, unsigned_variance_numerator_bits,
        variance_numerator_bits,
    };
    use crate::arithmetic::{SignedRatio, UnsignedRatio};
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::MomentWord;
    use crate::{Error, Quality};

    /// `N * (N * sum(x^2) - sum(x)^2)` by the two-pass definition `sum((N * x - sum(x))^2)`,
    /// which shares no arithmetic with the library's one-pass `N * sum(x^2) - sum(x)^2`.
    fn two_pass_scaled_variance(values: &[i128]) -> i128 {
        let n = values.len() as i128;
        let sum: i128 = values.iter().sum();
        values.iter().map(|&x| (n * x - sum).pow(2)).sum()
    }

    fn check_lifecycle_and_merges<C: CounterWord, S: MomentWord>(
        values: [S; 4],
        mean_parts: impl Fn(S::Mean) -> (i128, u64),
    ) {
        let mut moments = RunningMoments::<Checked, C, S>::default();
        assert_eq!(moments.snapshot(), None);

        let wide: [i128; 4] = core::array::from_fn(|i| i128::from(values[i].to_i64()));
        let (mut sum, mut squares) = (0_i128, 0_i128);
        for (index, &value) in values.iter().enumerate() {
            moments.update(value).expect("proven not to overflow");
            sum += wide[index];
            squares += wide[index] * wide[index];
            let n = index as i128 + 1;
            let snapshot = moments.snapshot().expect("defined after a sample");
            assert_eq!(i128::from(C::get(snapshot.samples).to_u64()), n);
            assert_eq!(snapshot.sum.into(), sum);
            assert_eq!(
                snapshot.sum_of_squares.into(),
                u128::try_from(squares).expect("non-negative")
            );
            assert_eq!(snapshot.quality, Quality::Complete);
            assert_eq!(
                mean_parts(snapshot.mean().expect("complete moments")),
                (sum, n as u64)
            );
            let variance = snapshot.population_variance().expect("complete moments");
            assert_eq!(variance.denominator.get(), (n * n) as u64);
            assert_eq!(
                variance.numerator as i128 * n,
                two_pass_scaled_variance(&wide[..=index])
            );
        }

        // Merging the two sides of every split reproduces the whole.
        for split in 0..=values.len() {
            let mut left = RunningMoments::<Checked, C, S>::new();
            let mut right = RunningMoments::<Checked, C, S>::new();
            for &value in &values[..split] {
                left.update(value).expect("representable");
            }
            for &value in &values[split..] {
                right.update(value).expect("representable");
            }
            left.merge(&right).expect("representable");
            assert_eq!(left, moments);
        }

        moments.reset();
        assert_eq!(moments.snapshot(), None);
    }

    fn lifecycle_for_counter<C: CounterWord>() {
        let signed = |m: SignedRatio| (i128::from(m.numerator), u64::from(m.denominator.get()));
        let unsigned = |m: UnsignedRatio| (i128::from(m.numerator), m.denominator.get());

        check_lifecycle_and_merges::<C, i8>([i8::MIN, -1, 0, i8::MAX], signed);
        check_lifecycle_and_merges::<C, i16>([i16::MIN, -1, 0, i16::MAX], signed);
        check_lifecycle_and_merges::<C, i32>([i32::MIN, -1, 0, i32::MAX], signed);
        check_lifecycle_and_merges::<C, u8>([0, 1, 128, u8::MAX], unsigned);
        check_lifecycle_and_merges::<C, u16>([0, 1, 1024, u16::MAX], unsigned);
        check_lifecycle_and_merges::<C, u32>([0, 1, 65_536, u32::MAX], unsigned);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_counter_and_sample_word() {
        lifecycle_for_counter::<u8>();
        lifecycle_for_counter::<u16>();
        lifecycle_for_counter::<u32>();
    }

    #[test]
    fn variance_headroom_formulas_and_validators_match_proven_widths() {
        assert_eq!(variance_numerator_bits(8, 32), 78);
        assert_eq!(variance_numerator_bits(16, 32), 94);
        assert_eq!(variance_numerator_bits(32, 32), 126);
        assert_eq!(unsigned_variance_numerator_bits(8, 32), 80);
        assert_eq!(unsigned_variance_numerator_bits(16, 32), 96);
        assert_eq!(unsigned_variance_numerator_bits(32, 32), 128);
        assert!(!raw_moments_hold::<i16>(0));
        assert!(raw_moments_hold::<i16>(8));
        assert!(raw_moments_hold::<i16>(16));
        assert!(raw_moments_hold::<i16>(32));
        assert!(!raw_moments_hold::<i16>(33));
        assert!(!raw_moments_hold::<i16>(64));
    }

    #[test]
    fn checked_sum_overflow_is_transactional() {
        let mut moments = RunningMoments::<Checked, u32, i16> {
            samples: 7,
            sum: i64::MAX,
            sum_of_squares: 49,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = moments;
        assert_eq!(moments.update(1), Err(Error::Overflow));
        assert_eq!(moments, before);
    }

    #[test]
    fn saturating_sum_is_visible_in_quality() {
        let mut moments = RunningMoments::<Saturating, u32, i16> {
            samples: 1,
            sum: i64::MAX,
            sum_of_squares: 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        moments.update(1).expect("saturating update succeeds");
        let snapshot = moments.snapshot().expect("defined");
        assert_eq!(snapshot.sum, i64::MAX);
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.mean(), None);
        assert_eq!(snapshot.population_variance(), None);
    }

    #[test]
    fn exact_population_variance_uses_wide_headroom() {
        let mut moments = RunningMoments::<Checked, u32, i16>::new();
        for sample in [i16::MIN, i16::MAX] {
            moments.update(sample).expect("representable");
        }
        let variance = moments
            .snapshot()
            .expect("defined")
            .population_variance()
            .expect("complete moments");
        assert_eq!(variance.numerator, 4_294_836_225);
        assert_eq!(variance.denominator, NonZeroU64::new(4).expect("nonzero"));
    }

    #[test]
    fn checked_merge_overflow_is_transactional() {
        let mut left = RunningMoments::<Checked, u32, i16> {
            samples: u32::MAX,
            sum: 7,
            sum_of_squares: 49,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = RunningMoments::<Checked, u32, i16>::new();
        right.update(1).expect("representable");
        let before = left;
        assert_eq!(left.merge(&right), Err(Error::Overflow));
        assert_eq!(left, before);
    }

    #[test]
    fn saturating_merge_is_visible_in_quality() {
        let mut left = RunningMoments::<Saturating, u32, i16> {
            samples: u32::MAX,
            sum: 7,
            sum_of_squares: 49,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = RunningMoments::<Saturating, u32, i16>::new();
        right.update(1).expect("representable");
        left.merge(&right).expect("saturating merge succeeds");
        let snapshot = left.snapshot().expect("defined");
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.mean(), None);
        assert_eq!(snapshot.population_variance(), None);
    }

    #[test]
    fn u8_counter_fills_at_255_and_rejects_or_saturates_the_256th_sample() {
        let mut checked = RunningMoments::<Checked, u8, i16>::new();
        let mut saturating = RunningMoments::<Saturating, u8, i16>::new();
        for step in 0..u8::MAX {
            let sample = i16::from(step) - 100;
            checked.update(sample).expect("below u8::MAX");
            saturating.update(sample).expect("below u8::MAX");
        }
        let full = checked;
        assert_eq!(checked.update(7), Err(Error::Overflow));
        assert_eq!(checked, full);
        assert_eq!(checked.snapshot().expect("defined").samples.get(), u8::MAX);
        saturating.update(7).expect("saturates");
        let snap = saturating.snapshot().expect("defined");
        assert_eq!(snap.samples.get(), u8::MAX);
        assert_eq!(snap.quality, Quality::Saturated);
        assert_eq!(snap.mean(), None);
        assert_eq!(snap.population_variance(), None);
    }

    /// Every checked failure leaves the whole state unchanged, and saturation is visible, at
    /// every signed width, including the `i32` square sum's `u128`.
    macro_rules! every_width_fails_transactionally {
        ($($name:ident: $word:ty => $square_sum:ty;)*) => {$(
            #[test]
            fn $name() {
                let samples_full = RunningMoments::<Checked, u32, $word> {
                    samples: u32::MAX,
                    sum: 7,
                    sum_of_squares: 49,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                let sum_full = RunningMoments::<Checked, u32, $word> {
                    samples: 7,
                    sum: i64::MAX,
                    ..samples_full
                };
                let squares_full = RunningMoments::<Checked, u32, $word> {
                    samples: 7,
                    sum_of_squares: <$square_sum>::MAX,
                    ..samples_full
                };
                let sum_empty = RunningMoments::<Checked, u32, $word> {
                    samples: 7,
                    sum: i64::MIN,
                    ..samples_full
                };
                for (full, sample) in [
                    (samples_full, <$word>::MAX),
                    (samples_full, <$word>::MIN),
                    (sum_full, <$word>::MAX),
                    (sum_empty, <$word>::MIN),
                    (squares_full, <$word>::MAX),
                    (squares_full, <$word>::MIN),
                ] {
                    let mut moments = full;
                    assert_eq!(moments.update(sample), Err(Error::Overflow));
                    assert_eq!(moments, full);
                    let mut newer = RunningMoments::<Checked, u32, $word>::new();
                    newer.update(sample).expect("representable");
                    assert_eq!(moments.merge(&newer), Err(Error::Overflow));
                    assert_eq!(moments, full);
                }

                let mut saturating = RunningMoments::<Saturating, u32, $word> {
                    samples: 1,
                    sum: 1,
                    sum_of_squares: <$square_sum>::MAX,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                saturating.update(<$word>::MIN).expect("saturating accepts");
                let snapshot = saturating.snapshot().expect("defined");
                assert_eq!(snapshot.sum_of_squares, <$square_sum>::MAX);
                assert_eq!(snapshot.quality, Quality::Saturated);
                assert_eq!(snapshot.mean(), None);
                assert_eq!(snapshot.population_variance(), None);
            }
        )*};
    }

    every_width_fails_transactionally! {
        checked_failure_is_transactional_at_i8: i8 => u64;
        checked_failure_is_transactional_at_i16: i16 => u64;
        checked_failure_is_transactional_at_i32: i32 => u128;
    }

    /// The unsigned words: a full count, sum or square sum overflows on `MAX`, and checked
    /// failures leave the whole state unchanged. Zero still overflows a full count but adds
    /// nothing to the sums.
    macro_rules! every_unsigned_width_fails_transactionally {
        ($($name:ident: $word:ty => $square_sum:ty;)*) => {$(
            #[test]
            fn $name() {
                let samples_full = RunningMoments::<Checked, u32, $word> {
                    samples: u32::MAX,
                    sum: 7,
                    sum_of_squares: 49,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                let sum_full = RunningMoments::<Checked, u32, $word> {
                    samples: 7,
                    sum: u64::MAX,
                    ..samples_full
                };
                let squares_full = RunningMoments::<Checked, u32, $word> {
                    samples: 7,
                    sum_of_squares: <$square_sum>::MAX,
                    ..samples_full
                };
                for (full, sample) in [
                    (samples_full, <$word>::MAX),
                    (samples_full, 0),
                    (sum_full, <$word>::MAX),
                    (sum_full, 1),
                    (squares_full, <$word>::MAX),
                    (squares_full, 1),
                ] {
                    let mut moments = full;
                    assert_eq!(moments.update(sample), Err(Error::Overflow));
                    assert_eq!(moments, full);
                    let mut newer = RunningMoments::<Checked, u32, $word>::new();
                    newer.update(sample).expect("representable");
                    assert_eq!(moments.merge(&newer), Err(Error::Overflow));
                    assert_eq!(moments, full);
                }
                // Zero fits a full sum and a full square sum exactly.
                for full in [sum_full, squares_full] {
                    let mut moments = full;
                    moments.update(0).expect("zero adds nothing");
                    assert_eq!(moments.samples, 8);
                    assert_eq!((moments.sum, moments.sum_of_squares), (full.sum, full.sum_of_squares));
                }

                let mut saturating = RunningMoments::<Saturating, u32, $word> {
                    samples: 1,
                    sum: u64::MAX - 1,
                    sum_of_squares: 1,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                saturating.update(<$word>::MAX).expect("saturating accepts");
                let snapshot = saturating.snapshot().expect("defined");
                assert_eq!(snapshot.sum, u64::MAX);
                assert_eq!(snapshot.quality, Quality::Saturated);
                assert_eq!(snapshot.mean(), None);
                assert_eq!(snapshot.population_variance(), None);
            }
        )*};
    }

    every_unsigned_width_fails_transactionally! {
        checked_failure_is_transactional_at_u8: u8 => u64;
        checked_failure_is_transactional_at_u16: u16 => u64;
        checked_failure_is_transactional_at_u32: u32 => u128;
    }

    /// The `u32` variance operands at the proven bound need all 128 bits: `N * sum(x^2)` is largest,
    /// above `2^127`, when every sample is `MAX`, and the numerator is largest with half the
    /// samples at zero and half at `MAX`.
    #[test]
    fn u32_variance_at_the_proven_bound() {
        let n = u32::MAX;
        let samples = NonZeroU32::new(n).expect("nonzero");
        let highs = u64::from(n / 2);
        let squares = u128::from(highs) * u128::from(u32::MAX).pow(2);
        let mixed = RawMomentsSnapshot::<u32, u32> {
            samples,
            sum: highs * u64::from(u32::MAX),
            sum_of_squares: squares,
            quality: Quality::Complete,
        };
        let k = u128::from(highs);
        let expected = k * (u128::from(n) - k) * u128::from(u32::MAX).pow(2);
        assert!(u128::from(n) * squares > 1 << 126);
        assert!(expected > 1 << 125);
        assert_eq!(
            mixed.population_variance().map(|v| v.numerator),
            Some(expected)
        );
        let all_max = RawMomentsSnapshot::<u32, u32> {
            samples,
            sum: u64::from(n) * u64::from(u32::MAX),
            sum_of_squares: u128::from(n) * u128::from(u32::MAX).pow(2),
            quality: Quality::Complete,
        };
        assert!(u128::from(n) * all_max.sum_of_squares > 1 << 127);
        assert_eq!(all_max.population_variance().map(|v| v.numerator), Some(0));
    }

    #[test]
    fn i32_variance_at_the_proven_bound() {
        let n = u32::MAX;
        let samples = NonZeroU32::new(n).expect("nonzero");
        let all_min = RawMomentsSnapshot::<u32, i32> {
            samples,
            sum: i64::from(n) * i64::from(i32::MIN),
            sum_of_squares: u128::from(n) << 62,
            quality: Quality::Complete,
        };
        let variance = all_min.population_variance().expect("in range");
        assert_eq!(variance.numerator, 0);
        assert_eq!(variance.denominator.get(), u64::from(n) * u64::from(n));
        let lows = u64::from(n / 2 + 1);
        let highs = u64::from(n / 2);
        let squares = u128::from(lows) * (1 << 62) + u128::from(highs) * (i32::MAX as u128).pow(2);
        let mixed = RawMomentsSnapshot::<u32, i32> {
            samples,
            sum: lows as i64 * i64::from(i32::MIN) + highs as i64 * i64::from(i32::MAX),
            sum_of_squares: squares,
            quality: Quality::Complete,
        };
        let wide = |value: i128| value.unsigned_abs();
        let expected = u128::from(n) * squares - wide(i128::from(mixed.sum)).pow(2);
        assert!(u128::from(n) * squares > 1 << 125);
        assert_eq!(
            mixed.population_variance().map(|v| v.numerator),
            Some(expected)
        );
    }

    #[test]
    fn inconsistent_snapshots_return_none() {
        let approximate = RawMomentsSnapshot::<u32, i16> {
            samples: NonZeroU32::new(2).expect("nonzero"),
            sum: 4,
            sum_of_squares: 10,
            quality: Quality::Approximate,
        };
        assert_eq!(approximate.mean(), None);
        assert_eq!(approximate.population_variance(), None);

        let scaled_overflow = RawMomentsSnapshot::<u32, i32> {
            samples: NonZeroU32::new(2).expect("nonzero"),
            sum: 0,
            sum_of_squares: u128::MAX,
            quality: Quality::Complete,
        };
        assert_eq!(scaled_overflow.population_variance(), None);

        let negative_numerator = RawMomentsSnapshot::<u32, i16> {
            samples: NonZeroU32::new(2).expect("nonzero"),
            sum: 5,
            sum_of_squares: 0,
            quality: Quality::Complete,
        };
        assert_eq!(negative_numerator.population_variance(), None);

        let oversized_u64 = RawMomentsSnapshot::<u64, i16> {
            samples: NonZeroU64::new(u64::from(u32::MAX) + 1).expect("nonzero"),
            sum: 4,
            sum_of_squares: 10,
            quality: Quality::Complete,
        };
        assert_eq!(oversized_u64.mean(), None);
        assert_eq!(oversized_u64.population_variance(), None);
    }

    /// The `i32` square sum outgrows `u64` after four full-scale samples; in `u128` it does not.
    #[test]
    fn i32_square_sums_exceed_u64_and_stay_exact() {
        let mut moments = RunningMoments::<Checked, u32, i32>::new();
        for _ in 0..5 {
            moments.update(i32::MIN).expect("u128 holds the square sum");
        }
        let snapshot = moments.snapshot().expect("defined");
        assert_eq!(snapshot.sum_of_squares, 5 * (1u128 << 62));
        assert!(snapshot.sum_of_squares > u128::from(u64::MAX));
        assert_eq!(snapshot.sum, 5 * i64::from(i32::MIN));
        let variance = snapshot.population_variance().expect("complete");
        assert_eq!(variance.numerator, 0);
        assert_eq!(variance.denominator.get(), 25);
    }

    #[test]
    fn i32_full_swing_variance_is_exact() {
        let mut moments = RunningMoments::<Checked, u32, i32>::new();
        for sample in [i32::MIN, i32::MAX] {
            moments.update(sample).expect("representable");
        }
        let variance = moments
            .snapshot()
            .expect("defined")
            .population_variance()
            .expect("complete");
        let span = (i128::from(i32::MAX) - i128::from(i32::MIN)) as u128;
        assert_eq!(variance.numerator, span * span);
        assert_eq!(variance.denominator.get(), 4);
    }

    #[test]
    fn layouts_follow_the_square_sum_and_counter_word() {
        use core::mem::size_of;
        assert_eq!(
            size_of::<RunningMoments<Checked, u32, i16>>(),
            size_of::<RunningMoments<Checked, u32, i8>>()
        );
        assert!(
            size_of::<RunningMoments<Checked, u32, i32>>()
                > size_of::<RunningMoments<Checked, u32, i16>>()
        );
        assert!(
            size_of::<RunningMoments<Checked, u8, i16>>()
                <= size_of::<RunningMoments<Checked, u16, i16>>()
        );
        assert!(
            size_of::<RunningMoments<Checked, u16, i16>>()
                <= size_of::<RunningMoments<Checked, u32, i16>>()
        );
        assert_eq!(
            size_of::<Option<RawMomentsSnapshot<u32, i16>>>(),
            size_of::<RawMomentsSnapshot<u32, i16>>()
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_snapshot_queries_are_const_evaluable_on_nightly() {
        const SNAPSHOT: RawMomentsSnapshot<u16, i16> = {
            let mut moments = RunningMoments::<Checked, u16, i16>::new();
            assert!(moments.update(2).is_ok());
            assert!(moments.update(4).is_ok());
            match moments.snapshot() {
                Some(s) => s,
                None => unreachable!(),
            }
        };
        const MEAN: Option<SignedRatio> = SNAPSHOT.mean();
        const VARIANCE: Option<crate::arithmetic::UnsignedWideRatio> =
            SNAPSHOT.population_variance();
        assert_eq!(
            MEAN,
            Some(SignedRatio {
                numerator: 6,
                denominator: NonZeroU32::new(2).unwrap(),
            })
        );
        assert_eq!(
            VARIANCE,
            Some(UnsignedRatio {
                numerator: 4,
                denominator: NonZeroU64::new(4).unwrap(),
            })
        );
    }
}
