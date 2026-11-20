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
//!
//! Both traits are sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`, `u16`,
//! `u32` implement them.

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

macro_rules! sample_word {
    ($word:ty, $difference:ty) => {
        const _: () = assert!(<$difference>::BITS > <$word>::BITS);

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
    };
}

sample_word!(i8, i16);
sample_word!(i16, i32);
sample_word!(i32, i64);
sample_word!(u8, i16);
sample_word!(u16, i32);
sample_word!(u32, i64);

#[cfg(test)]
mod tests {
    use super::{SampleValue, SampleWord};

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
