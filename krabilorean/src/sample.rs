//! Sample words: the integer types a statistic can consume, with their wider arithmetic types and
//! the width-dependent limits proven for each.
//!
//! The traits are split by what a family needs:
//!
//! - [`SampleValue`] holds the order and extremes (`MIN`, `MAX`, `ZERO`) that comparison-only
//!   families need.
//! - [`SampleWord`] adds the width facts every integer sample family may use: `BITS`, exact `i64`
//!   conversions, the Q-format headroom, and the exact signed difference type
//!   [`SampleWord::Difference`].
//! - [`MomentWord`] adds the raw-moment sum and square-sum types, signed or unsigned with the
//!   word, each proven for `u32::MAX` samples.
//!
//! All traits are sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`, `u16`,
//! `u32` implement them.
//!
//! Accumulator widths are formulas of the word's `BITS`, each checked against its type at compile
//! time for every word:
//!
//! | Accumulator | Bits needed for `u32::MAX` samples | `i8` | `i16` | `i32` |
//! |---|---|---|---|---|
//! | moments sum ([`MomentWord::Sum`]) | [`sum_bits`]: `BITS + 32`, signed | `i64` | `i64` | `i64` |
//! | moments sum of squares ([`MomentWord::SquareSum`]) | [`square_sum_bits`]: `2 * BITS + 30` | `u64` | `u64` | `u128` |
//! | variance numerator `N * sum(x^2) - sum(x)^2` | `32 + square_sum_bits = 2 * (sum_bits - 1)` | `u128` | `u128` | `u128` |
//! | variance denominator `N^2` | 64 | `u64` | `u64` | `u64` |
//!
//! The unsigned words have their own formulas, because their extreme is `MAX = 2^BITS - 1`, not
//! `|MIN| = 2^(BITS - 1)`:
//!
//! | Accumulator | Bits needed for `u32::MAX` samples | `u8` | `u16` | `u32` |
//! |---|---|---|---|---|
//! | moments sum ([`MomentWord::Sum`]) | [`unsigned_sum_bits`]: `BITS + 32`, unsigned | `u64` | `u64` | `u64` |
//! | moments sum of squares ([`MomentWord::SquareSum`]) | [`unsigned_square_sum_bits`]: `2 * BITS + 32` | `u64` | `u64` | `u128` |
//! | variance numerator | `32 + unsigned_square_sum_bits = 2 * unsigned_sum_bits` | `u128` | `u128` | `u128` |
//!
//! The square of a moment sum is exact only for a single widened sample, and a snapshot's `sum`
//! can be far larger. Another crate can't reach it at all: the sum operations live on the sum
//! type, on a crate-private trait, so a word bound does not bring them into scope. The square
//! also takes the crate-private call token, so a later public bound can't expose it:
//!
//! ```compile_fail,E0599
//! use krabilorean::sample::MomentWord;
//! // Overflows for `sum = 1 << 40` if callable; it must not compile outside the crate.
//! fn square<S: MomentWord>(sum: S::Sum) -> S::SquareSum {
//!     sum.square()
//! }
//! ```

use core::num::{NonZeroU32, NonZeroU64};

use crate::arithmetic::{SignedRatio, UnsignedRatio};
use crate::policy::Accumulator;

/// A totally ordered sample type: what the comparison-only families need.
///
/// The trait is sealed. `Ord` is required, not `PartialOrd`: every sample must compare with every
/// other, or a minimum, a crossing or a run could silently skip a value.
pub trait SampleValue: crate::sealed::SampleValue + Copy + Ord + core::fmt::Debug {
    /// Smallest value.
    const MIN: Self;
    /// Largest value.
    const MAX: Self;
    /// Zero.
    const ZERO: Self;
}

/// An integer sample type with its width-dependent limits.
///
/// The trait is sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`, `u16`,
/// `u32` implement it. Its limits `MIN`, `MAX` and `ZERO` come from [`SampleValue`].
pub trait SampleWord: crate::sealed::SampleWord + SampleValue {
    /// Width of the word in bits.
    const BITS: u32;
    /// Largest `F` for which `(MAX - MIN) << F` fits `i64`, so a Q-format sample and the
    /// difference of two Q-format samples cannot overflow `i64` at `F` fractional bits.
    const I64_Q_HEADROOM_BITS: u32;
    /// Fractional bits of the word's sample units: `0` for an integer word.
    const FRACTIONAL_BITS: u32;

    /// Exact widening to `i64`, which holds every word.
    fn to_i64(self) -> i64;
    /// Exact narrowing from `i64`, or `None` outside the word.
    fn try_from_i64(value: i64) -> Option<Self>;

    /// Signed type that holds every difference `a - b` of two words: `i16`, `i32` and `i64` for
    /// the 8-, 16- and 32-bit words, signed or unsigned. A difference of two unsigned words can be
    /// negative, so it is never the word itself.
    type Difference: Copy + Ord + core::fmt::Debug + Into<i64>;

    /// Exact signed difference `self - other`.
    fn difference(self, other: Self) -> Self::Difference;
}

/// A sample word with the sum and square-sum types that raw moments need.
///
/// The trait is sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`, `u16`,
/// `u32` implement it. A signed word's sums are signed and an unsigned word's are unsigned, each
/// proven from the word's extremes to hold [`Self::MAX_PROVEN_SAMPLES`] samples; see the module
/// table.
pub trait MomentWord: SampleWord + crate::sealed::MomentWord {
    /// Type holding the sum of [`Self::MAX_PROVEN_SAMPLES`] words: signed for a signed word,
    /// unsigned for an unsigned one.
    type Sum: Copy
        + Ord
        + core::fmt::Debug
        + Into<i128>
        + Accumulator
        + crate::sealed::IntegerName
        + crate::sealed::SumOps<Self::SquareSum, Self::Mean>;
    /// Unsigned type holding the sum of the squares of [`Self::MAX_PROVEN_SAMPLES`] words.
    type SquareSum: Copy
        + Ord
        + core::fmt::Debug
        + Into<u128>
        + From<u64>
        + Accumulator
        + crate::sealed::IntegerName
        + crate::sealed::SquareSumOps;
    /// The exact mean `sum / N`: [`SignedRatio`] for a signed word and [`UnsignedRatio`] for an
    /// unsigned one.
    type Mean: Copy + Eq + core::fmt::Debug;
    /// Largest sample count for which [`Self::Sum`] and [`Self::SquareSum`] are proven to hold
    /// their sums.
    const MAX_PROVEN_SAMPLES: u32;
    /// Bits the sum of [`Self::MAX_PROVEN_SAMPLES`] words needs: [`sum_bits`] of `BITS`, signed,
    /// for a signed word, and [`unsigned_sum_bits`] for an unsigned one.
    const SUM_BITS: u32;
    /// Bits the sum of [`Self::MAX_PROVEN_SAMPLES`] squares needs: [`square_sum_bits`] of `BITS`
    /// for a signed word, and [`unsigned_square_sum_bits`] for an unsigned one.
    const SQUARE_SUM_BITS: u32;
    /// Zero of [`Self::Sum`], usable in `const` code.
    const ZERO_SUM: Self::Sum;
    /// Zero of [`Self::SquareSum`], usable in `const` code.
    const ZERO_SQUARE_SUM: Self::SquareSum;

    /// Exact widening of one word into [`Self::Sum`].
    fn to_sum(self) -> Self::Sum;
}

/// Signed bits that hold the sum of `u32::MAX` words of `bits` bits.
///
/// Each word's magnitude is at most `|MIN| = 2^(bits - 1)`, so `u32::MAX` of them sum to less than
/// `2^(bits - 1 + 32)` in magnitude, which a signed type of `bits + 32` bits holds.
pub const fn sum_bits(bits: u32) -> u32 {
    bits.saturating_add(32)
}

/// Bits that hold the sum of the squares of `u32::MAX` words of `bits` bits.
///
/// The largest square is `MIN^2 = 2^(2 * bits - 2)`, so `u32::MAX` of them sum to less than
/// `2^(2 * bits - 2 + 32)`, which an unsigned type of `2 * bits + 30` bits holds.
pub const fn square_sum_bits(bits: u32) -> u32 {
    bits.saturating_mul(2).saturating_add(30)
}

/// Unsigned bits that hold the sum of `u32::MAX` unsigned words of `bits` bits.
///
/// Each word is at most `2^bits - 1`, so `u32::MAX` of them sum to less than `2^(bits + 32)`.
pub const fn unsigned_sum_bits(bits: u32) -> u32 {
    bits.saturating_add(32)
}

/// Bits that hold the sum of the squares of `u32::MAX` unsigned words of `bits` bits.
///
/// The largest square is `(2^bits - 1)^2 < 2^(2 * bits)`, so `u32::MAX` of them sum to less than
/// `2^(2 * bits + 32)`. That is two bits more than a signed word of the same width needs, because
/// the unsigned extreme `2^bits - 1` is about twice the signed one's magnitude `2^(bits - 1)`.
pub const fn unsigned_square_sum_bits(bits: u32) -> u32 {
    bits.saturating_mul(2).saturating_add(32)
}

/// `(MAX - MIN) = 2^BITS - 1`, and `(2^BITS - 1) << F <= i64::MAX` exactly when
/// `BITS + F <= 63`.
const fn i64_q_headroom_bits(bits: u32) -> u32 {
    63 - bits
}

macro_rules! integer_name {
    ($($type:ty),* $(,)?) => {$(
        impl crate::sealed::IntegerName for $type {
            const NAME: &'static str = stringify!($type);
        }
    )*};
}

integer_name!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

macro_rules! sum_ops {
    ($($sum:ty => $mean:ident: $($square:ty),*);* $(;)?) => {$($(
        impl crate::sealed::SumOps<$square, $mean> for $sum {
            #[inline(always)]
            fn square(self, _: crate::sealed::Internal) -> $square {
                (self * self) as $square
            }

            #[inline(always)]
            fn magnitude(self) -> u128 {
                sum_magnitude!($mean, self)
            }

            #[inline(always)]
            fn mean(self, samples: NonZeroU32) -> $mean {
                sum_mean!($mean, self, samples)
            }
        }
    )*)*};
}

/// `|sum|` in `u128`: the magnitude of a signed sum, or an unsigned sum as it is.
macro_rules! sum_magnitude {
    (SignedRatio, $sum:expr) => {
        $sum.unsigned_abs() as u128
    };
    (UnsignedRatio, $sum:expr) => {
        $sum as u128
    };
}

/// The exact mean `sum / N`.
macro_rules! sum_mean {
    (SignedRatio, $sum:expr, $samples:expr) => {
        SignedRatio {
            numerator: $sum,
            denominator: $samples,
        }
    };
    (UnsignedRatio, $sum:expr, $samples:expr) => {
        UnsignedRatio {
            numerator: $sum,
            denominator: NonZeroU64::from($samples),
        }
    };
}

sum_ops!(i64 => SignedRatio: u64, u128; u64 => UnsignedRatio: u64, u128);

impl crate::sealed::SquareSumOps for u64 {
    #[inline(always)]
    fn scaled(self, samples: NonZeroU32) -> Option<u128> {
        // Below `2^96`, so the product is exact.
        Some(u128::from(samples.get()) * u128::from(self))
    }
}

impl crate::sealed::SquareSumOps for u128 {
    #[inline(always)]
    fn scaled(self, samples: NonZeroU32) -> Option<u128> {
        u128::from(samples.get()).checked_mul(self)
    }
}

macro_rules! sample_word {
    (
        $word:ty, $difference:ty, $sum:ty, $square_sum:ty, $mean:ty,
        $sum_bits_fn:ident, $square_sum_bits_fn:ident
    ) => {
        const _: () = assert!(<$difference>::BITS > <$word>::BITS);
        const _: () = assert!(<$sum>::BITS >= $sum_bits_fn(<$word>::BITS));
        const _: () = assert!(<$square_sum>::BITS >= $square_sum_bits_fn(<$word>::BITS));

        impl crate::sealed::SampleValue for $word {
            #[inline(always)]
            fn above(self, threshold: Self) -> bool {
                self > threshold
            }

            #[inline(always)]
            fn at_or_above(self, threshold: Self) -> bool {
                self >= threshold
            }
        }

        impl crate::sealed::SampleWord for $word {
            #[inline(always)]
            fn absolute_difference(self, other: Self) -> u64 {
                u64::from(self.abs_diff(other))
            }
        }

        impl crate::sealed::MomentWord for $word {}

        impl SampleValue for $word {
            const MIN: Self = <$word>::MIN;
            const MAX: Self = <$word>::MAX;
            const ZERO: Self = 0;
        }

        impl SampleWord for $word {
            const BITS: u32 = <$word>::BITS;
            const I64_Q_HEADROOM_BITS: u32 = i64_q_headroom_bits(<$word>::BITS);
            const FRACTIONAL_BITS: u32 = 0;

            #[inline(always)]
            fn to_i64(self) -> i64 {
                i64::from(self)
            }

            #[inline(always)]
            fn try_from_i64(value: i64) -> Option<Self> {
                <$word>::try_from(value).ok()
            }

            type Difference = $difference;

            #[inline(always)]
            fn difference(self, other: Self) -> $difference {
                // Both operands fit `Difference`, and so does their difference.
                <$difference>::from(self) - <$difference>::from(other)
            }
        }

        impl MomentWord for $word {
            type Sum = $sum;
            type SquareSum = $square_sum;
            type Mean = $mean;
            const MAX_PROVEN_SAMPLES: u32 = u32::MAX;
            const SUM_BITS: u32 = $sum_bits_fn(<$word>::BITS);
            const SQUARE_SUM_BITS: u32 = $square_sum_bits_fn(<$word>::BITS);
            const ZERO_SUM: $sum = 0;
            const ZERO_SQUARE_SUM: $square_sum = 0;

            #[inline(always)]
            fn to_sum(self) -> $sum {
                <$sum>::from(self)
            }
        }
    };
}

sample_word!(i8, i16, i64, u64, SignedRatio, sum_bits, square_sum_bits);
sample_word!(i16, i32, i64, u64, SignedRatio, sum_bits, square_sum_bits);
sample_word!(i32, i64, i64, u128, SignedRatio, sum_bits, square_sum_bits);
sample_word!(
    u8,
    i16,
    u64,
    u64,
    UnsignedRatio,
    unsigned_sum_bits,
    unsigned_square_sum_bits
);
sample_word!(
    u16,
    i32,
    u64,
    u64,
    UnsignedRatio,
    unsigned_sum_bits,
    unsigned_square_sum_bits
);
sample_word!(
    u32,
    i64,
    u64,
    u128,
    UnsignedRatio,
    unsigned_sum_bits,
    unsigned_square_sum_bits
);

#[cfg(test)]
mod tests {
    use super::{MomentWord, SampleValue, SampleWord};

    trait Range {
        const LO: i128;
        const HI: i128;
    }

    macro_rules! range {
        ($($t:ty),*) => {$(
            impl Range for $t {
                const LO: i128 = <$t>::MIN as i128;
                const HI: i128 = <$t>::MAX as i128;
            }
        )*};
    }
    range!(i8, i16, i32, i64, u8, u16, u32);

    trait UnsignedMax {
        const HI: u128;
    }

    impl UnsignedMax for u64 {
        const HI: u128 = u64::MAX as u128;
    }

    impl UnsignedMax for u128 {
        const HI: u128 = u128::MAX;
    }

    fn check_signed_moment_bounds<S>()
    where
        S: MomentWord + Range,
        S::Sum: Range,
        S::SquareSum: UnsignedMax,
    {
        let (min, max) = (S::LO, S::HI);
        assert_eq!(S::BITS, (max - min + 1).ilog2());
        let n = i128::from(S::MAX_PROVEN_SAMPLES);
        assert_eq!(S::MAX_PROVEN_SAMPLES, u32::MAX);
        assert_eq!(S::SUM_BITS, super::sum_bits(S::BITS));
        assert_eq!(S::SQUARE_SUM_BITS, super::square_sum_bits(S::BITS));
        assert!(S::Sum::LO <= min * n && max * n <= S::Sum::HI);
        let square = (min * min).unsigned_abs();
        assert!(
            square
                .checked_mul(n.unsigned_abs())
                .is_some_and(|total| total <= S::SquareSum::HI)
        );
        assert!((min * n).unsigned_abs() <= 1u128 << (S::SUM_BITS - 1));
        assert!(square * n.unsigned_abs() < 1u128 << S::SQUARE_SUM_BITS);
        assert_eq!(S::MIN.to_sum().into(), min);
        assert_eq!(S::MAX.to_sum().into(), max);
    }

    fn check_unsigned_moment_bounds<S>()
    where
        S: MomentWord + Range,
        S::Sum: UnsignedMax,
        S::SquareSum: UnsignedMax,
        S::Difference: Range,
    {
        let max = S::HI.unsigned_abs();
        assert_eq!(S::LO, 0);
        assert_eq!(max + 1, 1u128 << S::BITS);
        let n = u128::from(S::MAX_PROVEN_SAMPLES);
        assert_eq!(S::MAX_PROVEN_SAMPLES, u32::MAX);
        assert_eq!(S::SUM_BITS, super::unsigned_sum_bits(S::BITS));
        assert_eq!(S::SQUARE_SUM_BITS, super::unsigned_square_sum_bits(S::BITS));
        assert!(max * n <= S::Sum::HI);
        assert!(max * n < 1u128 << S::SUM_BITS);
        let squares = max * max * n;
        assert!(squares <= S::SquareSum::HI);
        assert!(squares < 1u128 << S::SQUARE_SUM_BITS);
        assert!(max * n >= 1u128 << (S::SUM_BITS - 1));
        assert!(squares >= 1u128 << (S::SQUARE_SUM_BITS - 1));
        assert!(S::Difference::LO <= -S::HI && S::HI <= S::Difference::HI);
        assert_eq!(S::MIN.to_sum().into(), 0);
        assert_eq!(S::MAX.to_sum().into(), S::HI);
    }

    #[test]
    fn moment_associated_types_hold_their_bounds_at_the_extremes() {
        check_signed_moment_bounds::<i8>();
        check_signed_moment_bounds::<i16>();
        check_signed_moment_bounds::<i32>();
        check_unsigned_moment_bounds::<u8>();
        check_unsigned_moment_bounds::<u16>();
        check_unsigned_moment_bounds::<u32>();

        fn same<T>(_: T, _: T) {}
        same::<<i8 as MomentWord>::Sum>(0, 0_i64);
        same::<<i16 as MomentWord>::Sum>(0, 0_i64);
        same::<<i32 as MomentWord>::Sum>(0, 0_i64);
        same::<<i8 as MomentWord>::SquareSum>(0, 0_u64);
        same::<<i16 as MomentWord>::SquareSum>(0, 0_u64);
        same::<<i32 as MomentWord>::SquareSum>(0, 0_u128);
        same::<<u8 as MomentWord>::Sum>(0, 0_u64);
        same::<<u16 as MomentWord>::Sum>(0, 0_u64);
        same::<<u32 as MomentWord>::Sum>(0, 0_u64);
        same::<<u8 as MomentWord>::SquareSum>(0, 0_u64);
        same::<<u16 as MomentWord>::SquareSum>(0, 0_u64);
        same::<<u32 as MomentWord>::SquareSum>(0, 0_u128);
        assert_eq!(super::unsigned_square_sum_bits(16), 64);
        assert_eq!(super::unsigned_square_sum_bits(32), 96);
    }

    #[test]
    fn extremes_match_the_primitive_types() {
        assert_eq!(
            (<i8 as SampleValue>::MIN, <i8 as SampleValue>::MAX),
            (i8::MIN, i8::MAX)
        );
        assert_eq!(
            (<i16 as SampleValue>::MIN, <i16 as SampleValue>::MAX),
            (i16::MIN, i16::MAX)
        );
        assert_eq!(
            (<i32 as SampleValue>::MIN, <i32 as SampleValue>::MAX),
            (i32::MIN, i32::MAX)
        );
        assert_eq!(
            (<u8 as SampleValue>::MIN, <u8 as SampleValue>::MAX),
            (u8::MIN, u8::MAX)
        );
        assert_eq!(
            (<u16 as SampleValue>::MIN, <u16 as SampleValue>::MAX),
            (u16::MIN, u16::MAX)
        );
        assert_eq!(
            (<u32 as SampleValue>::MIN, <u32 as SampleValue>::MAX),
            (u32::MIN, u32::MAX)
        );
    }

    /// The headroom is the largest `F` for which the widest Q-format difference fits `i64`.
    fn check_headroom<S: SampleWord>(min: i128, max: i128) {
        let f = S::I64_Q_HEADROOM_BITS;
        assert_eq!(f, super::i64_q_headroom_bits(S::BITS));
        let widest = |bits: u32| (max - min) << bits;
        assert!(widest(f) <= i128::from(i64::MAX));
        assert!(widest(f + 1) > i128::from(i64::MAX));
        // A single Q-format sample fits as well.
        assert!(min << f >= i128::from(i64::MIN) && max << f <= i128::from(i64::MAX));
    }

    #[test]
    fn i64_q_headroom_is_tight() {
        check_headroom::<i8>(i8::MIN.into(), i8::MAX.into());
        check_headroom::<i16>(i16::MIN.into(), i16::MAX.into());
        check_headroom::<i32>(i32::MIN.into(), i32::MAX.into());
        assert_eq!(
            (
                <i8 as SampleWord>::I64_Q_HEADROOM_BITS,
                <i16 as SampleWord>::I64_Q_HEADROOM_BITS,
                <i32 as SampleWord>::I64_Q_HEADROOM_BITS,
            ),
            (55, 47, 31)
        );
    }

    #[test]
    fn unsigned_q_headroom_is_tight() {
        check_headroom::<u8>(0, u8::MAX.into());
        check_headroom::<u16>(0, u16::MAX.into());
        check_headroom::<u32>(0, u32::MAX.into());
        assert_eq!(
            (
                <u8 as SampleWord>::I64_Q_HEADROOM_BITS,
                <u16 as SampleWord>::I64_Q_HEADROOM_BITS,
                <u32 as SampleWord>::I64_Q_HEADROOM_BITS,
            ),
            (55, 47, 31)
        );
    }

    fn check_word_conversions_and_differences<S: SampleWord + Into<i64>>(expected_name: &str) {
        assert_eq!(S::NAME, expected_name);
        assert_eq!(S::FRACTIONAL_BITS, 0);
        let (lo, hi): (i64, i64) = (S::MIN.into(), S::MAX.into());
        assert_eq!(S::MIN.to_i64(), lo);
        assert_eq!(S::MAX.to_i64(), hi);
        assert_eq!(S::ZERO.to_i64(), 0);
        assert_eq!(S::try_from_i64(lo), Some(S::MIN));
        assert_eq!(S::try_from_i64(hi), Some(S::MAX));
        assert_eq!(S::try_from_i64(lo - 1), None);
        assert_eq!(S::try_from_i64(hi + 1), None);

        assert!(S::MAX.above(S::MIN));
        assert!(!S::MIN.above(S::MIN));
        assert!(S::MIN.at_or_above(S::MIN));
        assert!(!S::MIN.at_or_above(S::MAX));

        assert_eq!(S::MAX.difference(S::MIN).into(), hi - lo);
        assert_eq!(S::MIN.difference(S::MAX).into(), lo - hi);
        assert_eq!(S::MIN.difference(S::MIN).into(), 0);
        assert_eq!(S::MAX.absolute_difference(S::MIN), (hi - lo) as u64);
        assert_eq!(S::MIN.absolute_difference(S::MAX), (hi - lo) as u64);
    }

    #[test]
    fn conversions_and_differences_hold_at_every_word() {
        check_word_conversions_and_differences::<i8>("i8");
        check_word_conversions_and_differences::<i16>("i16");
        check_word_conversions_and_differences::<i32>("i32");
        check_word_conversions_and_differences::<u8>("u8");
        check_word_conversions_and_differences::<u16>("u16");
        check_word_conversions_and_differences::<u32>("u32");
    }
}
