//! Sample words: the integer types a statistic can consume, with their wider arithmetic types and
//! the width-dependent limits proven for each.
//!
//! The traits are split by what a family needs:
//!
//! - [`SampleValue`] holds the order and extremes (`MIN`, `MAX`, `ZERO`) that comparison-only
//!   families need. Every [`SampleWord`] implements it, as do the NaN-free floats
//!   [`Finite<f32>`](crate::float::Finite) and `Finite<f64>`.
//! - [`SampleWord`] adds the width facts every integer sample family may use: `BITS`, exact `i64`
//!   conversions, the Q-format headroom, and the exact signed difference type
//!   [`SampleWord::Difference`].
//! - [`MomentWord`] adds the raw-moment sum and square-sum types, signed or unsigned with the
//!   word, each proven to hold as many samples as a counter of [`MomentWord::MAX_COUNTER_BITS`]
//!   bits can count.
//! - [`HistogramWord`] adds the histogram kernels: the canonical power-of-two partition and the
//!   linear-bin offset and span.
//!
//! All traits are sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`, `u16`,
//! `u32` implement every trait, and [`Finite<f32>`](crate::float::Finite) and `Finite<f64>`
//! implement [`SampleValue`].
//!
//! The offset and span kernels behind [`LinearHistogram`](crate::online::LinearHistogram) are
//! crate-private because they are exact only under a precondition (`lower <= upper`) that a caller
//! outside the crate could violate. Neither the sealed supertrait that holds them nor its call
//! token is reachable from safe code in another crate:
//!
//! ```compile_fail,E0603
//! use krabilorean::sample::HistogramWord;
//! // Wraps silently for `a < b` if callable; it must not compile outside the crate.
//! fn offset<S: HistogramWord>(a: S, b: S) -> u64 {
//!     a.offset_from(b, krabilorean::sealed::Internal::new())
//! }
//! ```
//!
//! ```compile_fail,E0061
//! use krabilorean::sample::HistogramWord;
//! fn offset<S: HistogramWord>(a: S, b: S) -> u64 {
//!     a.offset_from(b)
//! }
//! ```
//!
//! Accumulator widths are formulas of the word's `BITS` and the counter's `counter_bits`, each
//! checked against its type at compile time for every word:
//!
//! | Accumulator | Bits needed for `2^counter_bits - 1` samples | `i8` | `i16` | `i32` |
//! |---|---|---|---|---|
//! | moments sum ([`MomentWord::Sum`]) | [`sum_bits`]: `BITS + counter_bits`, signed | `i64` | `i64` | `i64` |
//! | moments sum of squares ([`MomentWord::SquareSum`]) | [`square_sum_bits`]: `2 * BITS - 2 + counter_bits` | `u64` | `u64` | `u128` |
//! | variance numerator `N * sum(x^2) - sum(x)^2` | `counter_bits + square_sum_bits = 2 * (sum_bits - 1)` | `u128` | `u128` | `u128` |
//! | variance denominator `N^2` | `2 * counter_bits` | `u64` | `u64` | `u64` |
//!
//! The unsigned words have their own formulas, because their extreme is `MAX = 2^BITS - 1`, not
//! `|MIN| = 2^(BITS - 1)`:
//!
//! | Accumulator | Bits needed for `2^counter_bits - 1` samples | `u8` | `u16` | `u32` |
//! |---|---|---|---|---|
//! | moments sum ([`MomentWord::Sum`]) | [`unsigned_sum_bits`]: `BITS + counter_bits`, unsigned | `u64` | `u64` | `u64` |
//! | moments sum of squares ([`MomentWord::SquareSum`]) | [`unsigned_square_sum_bits`]: `2 * BITS + counter_bits` | `u64` | `u64` | `u128` |
//! | variance numerator | `counter_bits + unsigned_square_sum_bits = 2 * unsigned_sum_bits` | `u128` | `u128` | `u128` |
//!
//! Every word's [`MomentWord::Sum`] and [`MomentWord::SquareSum`] hold any `u8`, `u16`, or `u32`
//! sample count; none holds a `u64` count:
//!
//! | Word | [`MomentWord::MAX_COUNTER_BITS`] |
//! |---|---|
//! | `i8` | 50 |
//! | `i16` | 34 |
//! | `i32` | 32 |
//! | `u8` | 48 |
//! | `u16` | 32 |
//! | `u32` | 32 |
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

c0nst::c0nst! {
    /// A totally ordered sample type: what the comparison-only families need.
    ///
    /// [`Minimum`](crate::online::Minimum), [`Maximum`](crate::online::Maximum),
    /// [`Extrema`](crate::online::Extrema), [`ThresholdCrossings`](crate::online::ThresholdCrossings),
    /// and [`ThresholdRuns`](crate::online::ThresholdRuns) only compare samples, so they take any
    /// `SampleValue`. Every [`SampleWord`] is one, as are the NaN-free floats
    /// [`Finite<f32>`](crate::float::Finite) and `Finite<f64>`.
    ///
    /// The trait is sealed. `Ord` is required, not `PartialOrd`: every sample must compare with every
    /// other, or a minimum, a crossing or a run could silently skip a value.
    pub c0nst trait SampleValue:
        [c0nst] crate::sealed::SampleValue + Copy + Ord + core::fmt::Debug + 'static
    {
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
    pub c0nst trait SampleWord: [c0nst] crate::sealed::SampleWord + [c0nst] SampleValue {
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
    /// proven from the word's extremes to hold as many samples as a counter of
    /// [`Self::MAX_COUNTER_BITS`] bits can count; see the module table.
    pub c0nst trait MomentWord: [c0nst] SampleWord + [c0nst] crate::sealed::MomentWord {
        /// Type holding the sum of `2^MAX_COUNTER_BITS - 1` words: signed for a signed word,
        /// unsigned for an unsigned one.
        type Sum: Copy
            + Ord
            + core::fmt::Debug
            + Into<i128>
            + [c0nst] Accumulator
            + crate::sealed::IntegerName
            + [c0nst] crate::sealed::SumOps<Self::SquareSum, Self::Mean>;
        /// Unsigned type holding the sum of the squares of `2^MAX_COUNTER_BITS - 1` words.
        type SquareSum: Copy
            + Ord
            + core::fmt::Debug
            + Into<u128>
            + From<u64>
            + [c0nst] Accumulator
            + crate::sealed::IntegerName
            + [c0nst] crate::sealed::SquareSumOps;
        /// The exact mean `sum / N`: [`SignedRatio`] for a signed word and [`UnsignedRatio`] for an
        /// unsigned one.
        type Mean: Copy + Eq + core::fmt::Debug;
        /// Widest counter, in bits, whose maximum count of words [`Self::Sum`] and
        /// [`Self::SquareSum`] are proven to hold.
        const MAX_COUNTER_BITS: u32;
        /// Zero of [`Self::Sum`], usable in `const` code.
        const ZERO_SUM: Self::Sum;
        /// Zero of [`Self::SquareSum`], usable in `const` code.
        const ZERO_SQUARE_SUM: Self::SquareSum;

        /// Exact widening of one word into [`Self::Sum`].
        fn to_sum(self) -> Self::Sum;
    }

    /// A sample word with the kernels the histograms need.
    ///
    /// The trait is sealed: the signed words `i8`, `i16`, `i32` and the unsigned words `u8`,
    /// `u16`, `u32` implement it. Its crate-private supertrait holds the linear-histogram offset
    /// and span kernels, which only the crate can call.
    pub c0nst trait HistogramWord: [c0nst] SampleWord + [c0nst] crate::sealed::HistogramWord {
        /// Number of canonical cells in the word's full-resolution power-of-two partition, a power
        /// of two so that coarsening is a shift.
        ///
        /// - Signed words, `2 * BITS` cells: `BITS` negative magnitude octaves (the widest holds
        ///   only `MIN`), zero, and `BITS - 1` positive octaves.
        /// - Unsigned words, `BITS` cells: `[0, 1]`, then the octaves `[2^k, 2^(k + 1) - 1]` for
        ///   `k` in `1..BITS`. Zero shares the lowest cell with one because a separate zero cell
        ///   would give `BITS + 1` cells and require a division to coarsen instead of a shift.
        const POWER_OF_TWO_CELLS: usize;

        /// Canonical full-resolution power-of-two cell of `self`, in increasing value order.
        /// Signed: negative octaves at `0..BITS`, zero at `BITS`, positive octaves above it.
        /// Unsigned: `floor(log2(max(self, 1)))`.
        fn power_of_two_cell(self) -> usize;
        /// Inclusive bounds of a canonical power-of-two cell, or `None` past the last cell.
        fn power_of_two_cell_bounds(cell: usize) -> Option<crate::histogram::HistogramBounds<Self>>;
    }
}

/// Signed bits that hold the sum of `2^counter_bits - 1` words of `bits` bits.
///
/// Each word's magnitude is at most `|MIN| = 2^(bits - 1)`, so `2^counter_bits - 1` of them sum
/// to less than `2^(bits - 1 + counter_bits)` in magnitude, which a signed type of
/// `bits + counter_bits` bits holds.
pub const fn sum_bits(bits: u32, counter_bits: u32) -> u32 {
    bits.saturating_add(counter_bits)
}

/// Bits that hold the sum of the squares of `2^counter_bits - 1` words of `bits` bits.
///
/// The largest square is `MIN^2 = 2^(2 * bits - 2)`, so `2^counter_bits - 1` of them sum to less
/// than `2^(2 * bits - 2 + counter_bits)`, which an unsigned type of
/// `2 * bits - 2 + counter_bits` bits holds.
pub const fn square_sum_bits(bits: u32, counter_bits: u32) -> u32 {
    bits.saturating_mul(2)
        .saturating_sub(2)
        .saturating_add(counter_bits)
}

/// Unsigned bits that hold the sum of `2^counter_bits - 1` unsigned words of `bits` bits.
///
/// Each word is at most `2^bits - 1`, so `2^counter_bits - 1` of them sum to less than
/// `2^(bits + counter_bits)`.
pub const fn unsigned_sum_bits(bits: u32, counter_bits: u32) -> u32 {
    bits.saturating_add(counter_bits)
}

/// Bits that hold the sum of the squares of `2^counter_bits - 1` unsigned words of `bits` bits.
///
/// The largest square is `(2^bits - 1)^2 < 2^(2 * bits)`, so `2^counter_bits - 1` of them sum to
/// less than `2^(2 * bits + counter_bits)`. That is two bits more than a signed word of the same
/// width needs, because the unsigned extreme `2^bits - 1` is about twice the signed one's
/// magnitude `2^(bits - 1)`.
pub const fn unsigned_square_sum_bits(bits: u32, counter_bits: u32) -> u32 {
    bits.saturating_mul(2).saturating_add(counter_bits)
}

/// Widest counter whose term count fits both a type of `first_type_bits` requiring `first_bits`
/// at zero counter bits and a type of `second_type_bits` requiring `second_bits`. Each required
/// width grows by one bit per counter bit.
const fn max_counter_bits(
    first_type_bits: u32,
    first_bits: u32,
    second_type_bits: u32,
    second_bits: u32,
) -> u32 {
    let first = first_type_bits.saturating_sub(first_bits);
    let second = second_type_bits.saturating_sub(second_bits);
    if first < second { first } else { second }
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
        c0nst::c0nst! {
            c0nst impl crate::sealed::SumOps<$square, $mean> for $sum {
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
            denominator: crate::arithmetic::widen_count($samples),
        }
    };
}

sum_ops!(i64 => SignedRatio: u64, u128; u64 => UnsignedRatio: u64, u128);

c0nst::c0nst! {
    c0nst impl crate::sealed::SquareSumOps for u64 {
        #[inline(always)]
        fn scaled(self, samples: NonZeroU32) -> Option<u128> {
            // Below `2^96`, so the product is exact.
            Some((samples.get() as u128) * (self as u128))
        }
    }

    c0nst impl crate::sealed::SquareSumOps for u128 {
        #[inline(always)]
        fn scaled(self, samples: NonZeroU32) -> Option<u128> {
            (samples.get() as u128).checked_mul(self)
        }
    }
}

macro_rules! sample_word {
    (
        $word:ty, $difference:ty, $sum:ty, $square_sum:ty, $mean:ty,
        $sum_bits_fn:ident, $square_sum_bits_fn:ident
    ) => {
        const _: () = assert!(<$difference>::BITS > <$word>::BITS);
        const _: () = assert!(<$sum>::BITS >= $sum_bits_fn(<$word>::BITS, u32::BITS));
        const _: () = assert!(<$square_sum>::BITS >= $square_sum_bits_fn(<$word>::BITS, u32::BITS));

        c0nst::c0nst! {
            c0nst impl crate::sealed::SampleValue for $word {
                #[inline(always)]
                fn above(self, threshold: Self) -> bool {
                    self > threshold
                }

                #[inline(always)]
                fn at_or_above(self, threshold: Self) -> bool {
                    self >= threshold
                }
            }

            c0nst impl crate::sealed::SampleWord for $word {
                #[inline(always)]
                fn absolute_difference(self, other: Self) -> u64 {
                    self.abs_diff(other) as u64
                }
            }

            c0nst impl crate::sealed::MomentWord for $word {}

            c0nst impl SampleValue for $word {
                const MIN: Self = <$word>::MIN;
                const MAX: Self = <$word>::MAX;
                const ZERO: Self = 0;
            }

            c0nst impl SampleWord for $word {
                const BITS: u32 = <$word>::BITS;
                const I64_Q_HEADROOM_BITS: u32 = i64_q_headroom_bits(<$word>::BITS);
                const FRACTIONAL_BITS: u32 = 0;

                #[inline(always)]
                fn to_i64(self) -> i64 {
                    self as i64
                }

                #[inline(always)]
                fn try_from_i64(value: i64) -> Option<Self> {
                    if value >= (<$word>::MIN as i64) && value <= (<$word>::MAX as i64) {
                        Some(value as Self)
                    } else {
                        None
                    }
                }

                type Difference = $difference;

                #[inline(always)]
                fn difference(self, other: Self) -> $difference {
                    // Both operands fit `Difference`, and so does their difference.
                    (self as $difference) - (other as $difference)
                }
            }

            c0nst impl MomentWord for $word {
                type Sum = $sum;
                type SquareSum = $square_sum;
                type Mean = $mean;
                const MAX_COUNTER_BITS: u32 = max_counter_bits(
                    <$sum>::BITS,
                    $sum_bits_fn(<$word>::BITS, 0),
                    <$square_sum>::BITS,
                    $square_sum_bits_fn(<$word>::BITS, 0),
                );
                const ZERO_SUM: $sum = 0;
                const ZERO_SQUARE_SUM: $square_sum = 0;

                #[inline(always)]
                fn to_sum(self) -> $sum {
                    self as $sum
                }
            }
        }
    };
}

/// `2 * BITS` canonical power-of-two cells: `BITS` negative octaves, zero, `BITS - 1` positive.
const fn power_of_two_cells(bits: u32) -> usize {
    2 * bits as usize
}

/// `BITS` canonical power-of-two cells of an unsigned word: `[0, 1]` and `BITS - 1` octaves.
const fn unsigned_power_of_two_cells(bits: u32) -> usize {
    bits as usize
}

macro_rules! signed_histogram_word {
    ($word:ty, $difference:ty) => {
        const _: () = assert!(power_of_two_cells(<$word>::BITS).is_power_of_two());

        c0nst::c0nst! {
            c0nst impl crate::sealed::HistogramWord for $word {
                #[inline(always)]
                fn offset_from(self, lower: Self, _: crate::sealed::Internal) -> u64 {
                    // Both operands fit `$difference`, and so does their difference, which is
                    // non-negative under the caller's `self >= lower`.
                    ((self as $difference) - (lower as $difference)) as u64
                }

                #[inline(always)]
                fn span_values(
                    lower: Self,
                    upper: Self,
                    _: crate::sealed::Internal,
                ) -> NonZeroU64 {
                    // `lower <= upper`, so the span is in `1..=2^BITS` and `NonZeroU64::MIN` is
                    // unreachable.
                    let raw = ((upper as $difference) - (lower as $difference) + 1) as u64;
                    match NonZeroU64::new(raw) {
                        Some(span) => span,
                        None => NonZeroU64::MIN,
                    }
                }
            }

            c0nst impl HistogramWord for $word {
                const POWER_OF_TWO_CELLS: usize = power_of_two_cells(<$word>::BITS);

                #[inline(always)]
                fn power_of_two_cell(self) -> usize {
                    const BITS: u32 = <$word>::BITS;
                    if self == 0 {
                        return BITS as usize;
                    }
                    let magnitude = (self as $difference).unsigned_abs();
                    let exponent = (<$difference>::BITS - 1) - magnitude.leading_zeros();
                    if self < 0 {
                        (BITS - 1 - exponent) as usize
                    } else {
                        (BITS + 1 + exponent) as usize
                    }
                }

                fn power_of_two_cell_bounds(
                    cell: usize,
                ) -> Option<crate::histogram::HistogramBounds<Self>> {
                    const BITS: u32 = <$word>::BITS;
                    if cell < BITS as usize {
                        let exponent = BITS - 1 - (cell as u32);
                        let minimum: $difference = 1 << exponent;
                        let maximum: $difference = if exponent == BITS - 1 {
                            1 << exponent
                        } else {
                            (1 << (exponent + 1)) - 1
                        };
                        return Some(crate::histogram::HistogramBounds::from_ordered(
                            (-maximum) as Self,
                            (-minimum) as Self,
                        ));
                    }
                    if cell == BITS as usize {
                        return Some(crate::histogram::HistogramBounds::from_ordered(0, 0));
                    }
                    if cell < (2 * BITS) as usize {
                        let exponent = (cell as u32) - BITS - 1;
                        let lower: $difference = 1 << exponent;
                        let upper: $difference = (1 << (exponent + 1)) - 1;
                        return Some(crate::histogram::HistogramBounds::from_ordered(
                            lower as Self,
                            upper as Self,
                        ));
                    }
                    None
                }
            }
        }
    };
}

macro_rules! unsigned_histogram_word {
    ($word:ty) => {
        const _: () = assert!(unsigned_power_of_two_cells(<$word>::BITS).is_power_of_two());

        c0nst::c0nst! {
            c0nst impl crate::sealed::HistogramWord for $word {
                #[inline(always)]
                fn offset_from(self, lower: Self, _: crate::sealed::Internal) -> u64 {
                    // Non-negative under the caller's `self >= lower`, and below `2^BITS`.
                    (self as u64) - (lower as u64)
                }

                #[inline(always)]
                fn span_values(
                    lower: Self,
                    upper: Self,
                    _: crate::sealed::Internal,
                ) -> NonZeroU64 {
                    // `lower <= upper`, so the span is in `1..=2^BITS <= 2^32` and
                    // `NonZeroU64::MIN` is unreachable.
                    let raw = (upper as u64) - (lower as u64) + 1;
                    match NonZeroU64::new(raw) {
                        Some(span) => span,
                        None => NonZeroU64::MIN,
                    }
                }
            }

            c0nst impl HistogramWord for $word {
                const POWER_OF_TWO_CELLS: usize = unsigned_power_of_two_cells(<$word>::BITS);

                #[inline(always)]
                fn power_of_two_cell(self) -> usize {
                    // `self | 1` puts zero in the cell of one; the exponent is below `BITS`.
                    (<$word>::BITS - 1 - (self | 1).leading_zeros()) as usize
                }

                fn power_of_two_cell_bounds(
                    cell: usize,
                ) -> Option<crate::histogram::HistogramBounds<Self>> {
                    if cell == 0 {
                        return Some(crate::histogram::HistogramBounds::from_ordered(0, 1));
                    }
                    if cell < <$word>::BITS as usize {
                        let lower: u64 = 1 << cell;
                        let upper: u64 = (1 << (cell + 1)) - 1;
                        return Some(crate::histogram::HistogramBounds::from_ordered(
                            lower as Self,
                            upper as Self,
                        ));
                    }
                    None
                }
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

signed_histogram_word!(i8, i16);
signed_histogram_word!(i16, i32);
signed_histogram_word!(i32, i64);
unsigned_histogram_word!(u8);
unsigned_histogram_word!(u16);
unsigned_histogram_word!(u32);

#[cfg(test)]
mod tests {
    use super::{HistogramWord, MomentWord, SampleValue, SampleWord};

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
        assert!(S::MAX_COUNTER_BITS >= 32 && S::MAX_COUNTER_BITS < 64);
        let square = (min * min).unsigned_abs();
        let sums_hold = |n: u128| {
            let signed = i128::try_from(n).ok();
            signed
                .and_then(|n| min.checked_mul(n))
                .is_some_and(|sum| S::Sum::LO <= sum)
                && signed
                    .and_then(|n| max.checked_mul(n))
                    .is_some_and(|sum| sum <= S::Sum::HI)
                && square
                    .checked_mul(n)
                    .is_some_and(|total| total <= S::SquareSum::HI)
        };
        let largest = |bits: u32| (1_u128 << bits) - 1;
        assert!(sums_hold(largest(S::MAX_COUNTER_BITS)));
        assert!(!sums_hold(largest(S::MAX_COUNTER_BITS + 1)));
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
        assert_eq!(max + 1, 1_u128 << S::BITS);
        assert!(S::MAX_COUNTER_BITS >= 32 && S::MAX_COUNTER_BITS < 64);
        let sums_hold = |n: u128| {
            max.checked_mul(n).is_some_and(|sum| sum <= S::Sum::HI)
                && (max * max)
                    .checked_mul(n)
                    .is_some_and(|total| total <= S::SquareSum::HI)
        };
        let largest = |bits: u32| (1_u128 << bits) - 1;
        assert!(sums_hold(largest(S::MAX_COUNTER_BITS)));
        assert!(!sums_hold(largest(S::MAX_COUNTER_BITS + 1)));
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

        assert_eq!(
            [
                <i8 as MomentWord>::MAX_COUNTER_BITS,
                <i16 as MomentWord>::MAX_COUNTER_BITS,
                <i32 as MomentWord>::MAX_COUNTER_BITS,
                <u8 as MomentWord>::MAX_COUNTER_BITS,
                <u16 as MomentWord>::MAX_COUNTER_BITS,
                <u32 as MomentWord>::MAX_COUNTER_BITS,
            ],
            [50, 34, 32, 48, 32, 32]
        );
        assert_eq!(super::max_counter_bits(64, 32, 128, 62), 32);
        assert_eq!(super::max_counter_bits(64, 16, 64, 30), 34);

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
        assert_eq!(super::sum_bits(16, 32), 48);
        assert_eq!(super::square_sum_bits(16, 32), 62);
        assert_eq!(super::unsigned_sum_bits(16, 32), 48);
        assert_eq!(super::unsigned_square_sum_bits(16, 32), 64);
        assert_eq!(super::unsigned_square_sum_bits(32, 32), 96);
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

    fn check_word_conversions_and_differences<S: HistogramWord + Into<i64>>(expected_name: &str) {
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

        let internal = crate::sealed::Internal::new();
        assert_eq!(S::MAX.offset_from(S::MIN, internal), (hi - lo) as u64);
        assert_eq!(S::MIN.offset_from(S::MIN, internal), 0);
        assert_eq!(
            S::span_values(S::MIN, S::MAX, internal).get(),
            1_u64 << S::BITS
        );
        assert_eq!(S::span_values(S::MAX, S::MAX, internal).get(), 1);

        assert!(S::POWER_OF_TWO_CELLS.is_power_of_two());
        assert_eq!(S::power_of_two_cell_bounds(S::POWER_OF_TWO_CELLS), None);
        let mut next = lo;
        for cell in 0..S::POWER_OF_TWO_CELLS {
            let bounds = S::power_of_two_cell_bounds(cell).expect("cell in range");
            let (first, last) = (bounds.lower(), bounds.upper());
            assert_eq!(first.to_i64(), next, "cell {cell}");
            assert!(first <= last);
            assert_eq!(first.power_of_two_cell(), cell);
            assert_eq!(last.power_of_two_cell(), cell);
            next = last.to_i64() + 1;
        }
        assert_eq!(next, hi + 1);
    }

    #[test]
    fn conversions_and_differences_hold_at_every_word() {
        check_word_conversions_and_differences::<i8>("i8");
        check_word_conversions_and_differences::<i16>("i16");
        check_word_conversions_and_differences::<i32>("i32");
        check_word_conversions_and_differences::<u8>("u8");
        check_word_conversions_and_differences::<u16>("u16");
        check_word_conversions_and_differences::<u32>("u32");
        assert_eq!(0_i16.power_of_two_cell(), 16);
        assert_eq!(0_u16.power_of_two_cell(), 0);
        assert_eq!(super::power_of_two_cells(16), 32);
        assert_eq!(super::unsigned_power_of_two_cells(16), 16);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_word_operations_are_const_evaluable_on_nightly() {
        const WIDE: i64 = <i16 as SampleWord>::to_i64(-12);
        const NARROW: Option<i16> = <i16 as SampleWord>::try_from_i64(WIDE);
        const DIFF: i32 = <i16 as SampleWord>::difference(10, -12);
        const SUM: i64 = <i16 as MomentWord>::to_sum(-12);
        const CELL: usize = <i16 as HistogramWord>::power_of_two_cell(-12);
        const BOUNDS: Option<crate::histogram::HistogramBounds<i16>> =
            <i16 as HistogramWord>::power_of_two_cell_bounds(CELL);
        assert_eq!(WIDE, -12);
        assert_eq!(NARROW, Some(-12));
        assert_eq!(DIFF, 22);
        assert_eq!(SUM, -12);
        assert_eq!(BOUNDS, crate::histogram::HistogramBounds::new(-15, -8).ok());
    }
}
