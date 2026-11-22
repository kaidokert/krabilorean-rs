//! Constant-state cumulative raw moments, mean, and exact population variance.

use core::marker::PhantomData;
use core::num::{NonZeroU32, NonZeroU64};

use crate::arithmetic::UnsignedWideRatio;
use crate::policy::{Accumulator, OverflowPolicy};
use crate::sample::{
    MomentWord, square_sum_bits, sum_bits, unsigned_square_sum_bits, unsigned_sum_bits,
};
use crate::sealed::{SquareSumOps, SumOps};
use crate::{Error, Quality};

/// Bits that hold the population-variance numerator `N * sum(x^2) - sum(x)^2` of `u32::MAX`
/// words of `bits` bits, and both operands of its subtraction.
///
/// `N <= u32::MAX < 2^32` and `sum(x^2) < 2^square_sum_bits(bits)`, so `N * sum(x^2)` is below
/// `2^(32 + square_sum_bits(bits))`. `sum(x)^2` has the same bound `2^(2 * (sum_bits(bits) - 1))`
/// `= 2^(2 * bits + 62)`. The numerator is non-negative and at most `N * sum(x^2)`.
const fn variance_numerator_bits(bits: u32) -> u32 {
    32 + square_sum_bits(bits)
}

// The numerator is computed in `u128` at every width, and `N^2 <= (2^32 - 1)^2` fits the `u64`
// denominator. At `i32` it needs 126 bits, the tightest implemented width.
const _: () = assert!(variance_numerator_bits(8) == 2 * (sum_bits(8) - 1));
const _: () = assert!(variance_numerator_bits(16) == 2 * (sum_bits(16) - 1));
const _: () = assert!(variance_numerator_bits(32) == 2 * (sum_bits(32) - 1));
const _: () = assert!(variance_numerator_bits(8) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(16) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(32) <= u128::BITS);
const _: () = assert!(variance_numerator_bits(32) == 126);
// A 34-bit word would not fit: the proof is not vacuous.
const _: () = assert!(variance_numerator_bits(34) > u128::BITS);

/// Bits that hold the population-variance numerator of `u32::MAX` unsigned words of `bits` bits,
/// and both operands of its subtraction: `N * sum(x^2) < 2^(32 + unsigned_square_sum_bits(bits))`
/// and `sum(x)^2 < 2^(2 * unsigned_sum_bits(bits))` `= 2^(2 * bits + 64)`.
const fn unsigned_variance_numerator_bits(bits: u32) -> u32 {
    32 + unsigned_square_sum_bits(bits)
}

// `u32` needs all 128 bits of the `u128` numerator, the tightest unsigned width.
const _: () = assert!(unsigned_variance_numerator_bits(8) == 2 * unsigned_sum_bits(8));
const _: () = assert!(unsigned_variance_numerator_bits(16) == 2 * unsigned_sum_bits(16));
const _: () = assert!(unsigned_variance_numerator_bits(32) == 2 * unsigned_sum_bits(32));
const _: () = assert!(unsigned_variance_numerator_bits(8) <= u128::BITS);
const _: () = assert!(unsigned_variance_numerator_bits(16) <= u128::BITS);
const _: () = assert!(unsigned_variance_numerator_bits(32) == u128::BITS);
const _: () = assert!(unsigned_variance_numerator_bits(33) > u128::BITS);

/// Cumulative count, sum, and sum of squares.
///
/// `S` is the sample word. The sum is [`MomentWord::Sum`] and the square sum
/// [`MomentWord::SquareSum`], each proven to hold `u32::MAX` samples: `i64` and `u64` at `i8` and
/// `i16`, `i64` and `u128` at `i32`, `u64` and `u64` at `u8` and `u16`, and `u64` and `u128` at
/// `u32`. `mean` is the word's [`MomentWord::Mean`]: a
/// [`SignedRatio`](crate::arithmetic::SignedRatio) for a signed word and an
/// [`UnsignedRatio`](crate::arithmetic::UnsignedRatio) for an unsigned one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawMomentsSnapshot<S: MomentWord> {
    /// Accepted sample count.
    pub samples: NonZeroU32,
    /// Sum of accepted samples.
    pub sum: S::Sum,
    /// Sum of squared accepted samples.
    pub sum_of_squares: S::SquareSum,
    /// Saturation quality.
    pub quality: Quality,
}

/// Cumulative raw moments with policy-controlled overflow.
///
/// `S` is the sample word. Its sum and square sum are the word's proven
/// [`MomentWord::Sum`] and [`MomentWord::SquareSum`], so neither can overflow before the `u32`
/// sample count does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunningMoments<O, S: MomentWord> {
    samples: u32,
    sum: S::Sum,
    sum_of_squares: S::SquareSum,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective raw-moment state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawMomentsDelta<S: MomentWord> {
    samples: u32,
    sum: S::Sum,
    sum_of_squares: S::SquareSum,
    quality: Quality,
}

impl<O, S: MomentWord> RunningMoments<O, S> {
    /// Creates empty cumulative raw moments.
    pub const fn new() -> Self {
        Self {
            samples: 0,
            sum: S::ZERO_SUM,
            sum_of_squares: S::ZERO_SQUARE_SUM,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }
}

impl<O, S: MomentWord> Default for RunningMoments<O, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: OverflowPolicy, S: MomentWord> RunningMoments<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare(sample)?;
        self.commit(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<RawMomentsSnapshot<S>> {
        NonZeroU32::new(self.samples).map(|samples| RawMomentsSnapshot {
            samples,
            sum: self.sum,
            sum_of_squares: self.sum_of_squares,
            quality: self.quality,
        })
    }

    /// Clears all cumulative state.
    pub fn reset(&mut self) {
        self.samples = 0;
        self.sum = <S as MomentWord>::ZERO_SUM;
        self.sum_of_squares = <S as MomentWord>::ZERO_SQUARE_SUM;
        self.quality = Quality::Complete;
    }

    /// Exactly appends newer raw moments, transactionally.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge(newer)?;
        self.commit_merge(delta);
        Ok(())
    }

    pub(crate) fn prepare(&self, sample: S) -> Result<RawMomentsDelta<S>, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        let widened = <S as MomentWord>::to_sum(sample);
        let sum = Accumulator::add::<O>(self.sum, widened)?;
        // Signed: at most `MIN^2 = 2^(2 * BITS - 2) <= 2^62`, so the `i64` product is
        // exact. Unsigned: at most `(2^BITS - 1)^2 < 2^64`, so the `u64` product is exact.
        let square = <<S as MomentWord>::Sum as SumOps<_, _>>::square(
            widened,
            crate::sealed::Internal::new(),
        );
        let sum_of_squares = Accumulator::add::<O>(self.sum_of_squares, square)?;
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

    pub(crate) fn commit(&mut self, delta: RawMomentsDelta<S>) {
        self.samples = delta.samples;
        self.sum = delta.sum;
        self.sum_of_squares = delta.sum_of_squares;
        self.quality = delta.quality;
    }

    pub(crate) fn prepare_merge(&self, newer: &Self) -> Result<RawMomentsDelta<S>, Error> {
        let samples = O::add_u32(self.samples, newer.samples)?;
        let sum = Accumulator::add::<O>(self.sum, newer.sum)?;
        let sum_of_squares = Accumulator::add::<O>(self.sum_of_squares, newer.sum_of_squares)?;
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

    pub(crate) fn commit_merge(&mut self, delta: RawMomentsDelta<S>) {
        self.commit(delta);
    }
}

impl<S: MomentWord> RawMomentsSnapshot<S> {
    /// Returns the cumulative mean as an exact ratio.
    pub fn mean(self) -> <S as MomentWord>::Mean {
        <<S as MomentWord>::Sum as SumOps<_, _>>::mean(self.sum, self.samples)
    }

    /// Returns exact cumulative population variance when raw moments are complete and
    /// consistent.
    ///
    /// The numerator is `N * sum(x^2) - sum(x)^2`, which fits `u128` at every width for
    /// any state the family produces; the denominator is `N^2`, which fits `u64`. An
    /// inconsistent hand-built snapshot whose terms don't fit returns `None`.
    /// Saturated raw moments return `None` because the exact variance cannot be recovered from
    /// saturated sums.
    #[allow(clippy::question_mark)]
    pub fn population_variance(self) -> Option<UnsignedWideRatio> {
        if self.quality.is_saturated() {
            return None;
        }
        let samples = self.samples.get() as u128;
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
        let samples_u64 = NonZeroU64::from(self.samples);
        // `samples <= u32::MAX`, so `samples^2 < 2^64` and the product never saturates.
        let denominator = samples_u64.saturating_mul(samples_u64);
        Some(UnsignedWideRatio {
            numerator,
            denominator,
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::{NonZeroU32, NonZeroU64};

    use super::{
        RawMomentsSnapshot, RunningMoments, unsigned_variance_numerator_bits,
        variance_numerator_bits,
    };
    use crate::arithmetic::{SignedRatio, UnsignedRatio};
    use crate::policy::{Checked, Saturating};
    use crate::sample::MomentWord;
    use crate::{Error, Quality};

    /// `N * (N * sum(x^2) - sum(x)^2)` by the two-pass definition `sum((N * x - sum(x))^2)`,
    /// which shares no arithmetic with the library's one-pass `N * sum(x^2) - sum(x)^2`.
    fn two_pass_scaled_variance(values: &[i128]) -> i128 {
        let n = values.len() as i128;
        let sum: i128 = values.iter().sum();
        values.iter().map(|&x| (n * x - sum).pow(2)).sum()
    }

    fn check_lifecycle_and_merges<S: MomentWord>(
        values: [S; 4],
        mean_parts: impl Fn(S::Mean) -> (i128, u64),
    ) {
        let mut moments = RunningMoments::<Checked, S>::default();
        assert_eq!(moments.snapshot(), None);

        let wide: [i128; 4] = core::array::from_fn(|i| i128::from(values[i].to_i64()));
        let (mut sum, mut squares) = (0_i128, 0_i128);
        for (index, &value) in values.iter().enumerate() {
            moments.update(value).expect("proven not to overflow");
            sum += wide[index];
            squares += wide[index] * wide[index];
            let n = index as i128 + 1;
            let snapshot = moments.snapshot().expect("defined after a sample");
            assert_eq!(i128::from(snapshot.samples.get()), n);
            assert_eq!(snapshot.sum.into(), sum);
            assert_eq!(
                snapshot.sum_of_squares.into(),
                u128::try_from(squares).expect("non-negative")
            );
            assert_eq!(snapshot.quality, Quality::Complete);
            assert_eq!(mean_parts(snapshot.mean()), (sum, n as u64));
            let variance = snapshot.population_variance().expect("complete moments");
            assert_eq!(variance.denominator.get(), (n * n) as u64);
            assert_eq!(
                variance.numerator as i128 * n,
                two_pass_scaled_variance(&wide[..=index])
            );
        }

        // Merging the two sides of every split reproduces the whole.
        for split in 0..=values.len() {
            let mut left = RunningMoments::<Checked, S>::new();
            let mut right = RunningMoments::<Checked, S>::new();
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

    #[test]
    fn lifecycle_and_merges_hold_at_every_word() {
        let signed = |m: SignedRatio| (i128::from(m.numerator), u64::from(m.denominator.get()));
        let unsigned = |m: UnsignedRatio| (i128::from(m.numerator), m.denominator.get());

        check_lifecycle_and_merges::<i8>([i8::MIN, -1, 0, i8::MAX], signed);
        check_lifecycle_and_merges::<i16>([i16::MIN, -1, 0, i16::MAX], signed);
        check_lifecycle_and_merges::<i32>([i32::MIN, -1, 0, i32::MAX], signed);
        check_lifecycle_and_merges::<u8>([0, 1, 128, u8::MAX], unsigned);
        check_lifecycle_and_merges::<u16>([0, 1, 1024, u16::MAX], unsigned);
        check_lifecycle_and_merges::<u32>([0, 1, 65_536, u32::MAX], unsigned);
    }

    #[test]
    fn variance_headroom_formulas_match_proven_widths() {
        assert_eq!(variance_numerator_bits(8), 78);
        assert_eq!(variance_numerator_bits(16), 94);
        assert_eq!(variance_numerator_bits(32), 126);
        assert_eq!(unsigned_variance_numerator_bits(8), 80);
        assert_eq!(unsigned_variance_numerator_bits(16), 96);
        assert_eq!(unsigned_variance_numerator_bits(32), 128);
    }

    #[test]
    fn checked_sum_overflow_is_transactional() {
        let mut moments = RunningMoments::<Checked, i16> {
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
        let mut moments = RunningMoments::<Saturating, i16> {
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
        assert_eq!(snapshot.population_variance(), None);
    }

    #[test]
    fn exact_population_variance_uses_wide_headroom() {
        let mut moments = RunningMoments::<Checked, i16>::new();
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
        let mut left = RunningMoments::<Checked, i16> {
            samples: u32::MAX,
            sum: 7,
            sum_of_squares: 49,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = RunningMoments::<Checked, i16>::new();
        right.update(1).expect("representable");
        let before = left;
        assert_eq!(left.merge(&right), Err(Error::Overflow));
        assert_eq!(left, before);
    }

    #[test]
    fn saturating_merge_is_visible_in_quality() {
        let mut left = RunningMoments::<Saturating, i16> {
            samples: u32::MAX,
            sum: 7,
            sum_of_squares: 49,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut right = RunningMoments::<Saturating, i16>::new();
        right.update(1).expect("representable");
        left.merge(&right).expect("saturating merge succeeds");
        assert_eq!(
            left.snapshot().expect("defined").quality,
            Quality::Saturated
        );
    }

    /// Every checked failure leaves the whole state unchanged, and saturation is visible, at
    /// every signed width, including the `i32` square sum's `u128`.
    macro_rules! every_width_fails_transactionally {
        ($($name:ident: $word:ty => $square_sum:ty;)*) => {$(
            #[test]
            fn $name() {
                let samples_full = RunningMoments::<Checked, $word> {
                    samples: u32::MAX,
                    sum: 7,
                    sum_of_squares: 49,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                let sum_full = RunningMoments::<Checked, $word> {
                    samples: 7,
                    sum: i64::MAX,
                    ..samples_full
                };
                let squares_full = RunningMoments::<Checked, $word> {
                    samples: 7,
                    sum_of_squares: <$square_sum>::MAX,
                    ..samples_full
                };
                let sum_empty = RunningMoments::<Checked, $word> {
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
                    let mut newer = RunningMoments::<Checked, $word>::new();
                    newer.update(sample).expect("representable");
                    assert_eq!(moments.merge(&newer), Err(Error::Overflow));
                    assert_eq!(moments, full);
                }

                let mut saturating = RunningMoments::<Saturating, $word> {
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
                let samples_full = RunningMoments::<Checked, $word> {
                    samples: u32::MAX,
                    sum: 7,
                    sum_of_squares: 49,
                    quality: Quality::Complete,
                    policy: PhantomData,
                };
                let sum_full = RunningMoments::<Checked, $word> {
                    samples: 7,
                    sum: u64::MAX,
                    ..samples_full
                };
                let squares_full = RunningMoments::<Checked, $word> {
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
                    let mut newer = RunningMoments::<Checked, $word>::new();
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

                let mut saturating = RunningMoments::<Saturating, $word> {
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
        let mixed = RawMomentsSnapshot::<u32> {
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
        let all_max = RawMomentsSnapshot::<u32> {
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
        let all_min = RawMomentsSnapshot::<i32> {
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
        let mixed = RawMomentsSnapshot::<i32> {
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
        let scaled_overflow = RawMomentsSnapshot::<i32> {
            samples: NonZeroU32::new(2).expect("nonzero"),
            sum: 0,
            sum_of_squares: u128::MAX,
            quality: Quality::Complete,
        };
        assert_eq!(scaled_overflow.population_variance(), None);

        let negative_numerator = RawMomentsSnapshot::<i16> {
            samples: NonZeroU32::new(2).expect("nonzero"),
            sum: 5,
            sum_of_squares: 0,
            quality: Quality::Complete,
        };
        assert_eq!(negative_numerator.population_variance(), None);
    }

    /// The `i32` square sum outgrows `u64` after four full-scale samples; in `u128` it does not.
    #[test]
    fn i32_square_sums_exceed_u64_and_stay_exact() {
        let mut moments = RunningMoments::<Checked, i32>::new();
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
        let mut moments = RunningMoments::<Checked, i32>::new();
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
    fn layouts_follow_the_square_sum() {
        use core::mem::size_of;
        assert_eq!(
            size_of::<RunningMoments<Checked, i16>>(),
            size_of::<RunningMoments<Checked, i8>>()
        );
        assert!(
            size_of::<RunningMoments<Checked, i32>>() > size_of::<RunningMoments<Checked, i16>>()
        );
        assert_eq!(
            size_of::<Option<RawMomentsSnapshot<i16>>>(),
            size_of::<RawMomentsSnapshot<i16>>()
        );
    }
}
