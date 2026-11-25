//! Fixed-point sample words, and conversions of raw-unit results back into fixed-point values.
//!
//! `const_fixed`'s signed [`FixedI8<Frac>`](const_fixed::FixedI8),
//! [`FixedI16<Frac>`](const_fixed::FixedI16), [`FixedI32<Frac>`](const_fixed::FixedI32) and
//! unsigned [`FixedU8<Frac>`](const_fixed::FixedU8), [`FixedU16<Frac>`](const_fixed::FixedU16),
//! [`FixedU32<Frac>`](const_fixed::FixedU32) implement [`SampleValue`](crate::sample::SampleValue),
//! [`SampleWord`], [`MomentWord`], and [`HistogramWord`](crate::sample::HistogramWord). Every
//! online family that accepts an integer sample word accepts these fixed-point types and computes
//! on their raw two's-complement or unsigned bits, so state layout, accumulators, overflow proofs,
//! and execution cost match the underlying integer word. Each raw bit represents `2^-Frac` sample
//! units:
//!
//! - **Values typed as the word** are in sample units as they stand: extrema, thresholds,
//!   histogram bounds, power-of-two cell bounds, and a successive-difference snapshot's first and
//!   last samples. A threshold of `I16F16::from_num(12.5)` (a `FixedI32<U16>`) is 12.5 sample
//!   units.
//! - **Values computed from raw bits** are in raw units. A raw-moments mean is a ratio of raw sum
//!   over sample count ([`SignedRatio`](crate::arithmetic::SignedRatio) for signed words,
//!   [`UnsignedRatio`] for unsigned words), and a population variance is a ratio in squared raw
//!   units (`2 * Frac` fractional bits). An [`EwMeanSnapshot`] or [`EwVarianceSnapshot`] records
//!   its configured Q-format bits plus `Frac` in `fractional_bits`.
//!
//! The functions in this module convert raw-unit results back into fixed-point values with an
//! explicit rounding rule. A mean returns the word type itself; a non-negative magnitude (such as
//! a standard deviation or mean absolute successive difference) returns the same-width unsigned
//! fixed-point type with the same fractional bits, [`FixedPointWord::Magnitude`].
//!
//! ```
//! use const_fixed::types::I16F16;
//! use krabilorean::fixed_point;
//! use krabilorean::online::{Extrema, RunningMoments};
//! use krabilorean::policy::Checked;
//!
//! let current = [1.25, -0.5, 3.75, 2.0].map(I16F16::from_num);
//! let mut extrema = Extrema::<Checked, u32, I16F16>::new();
//! let mut moments = RunningMoments::<Checked, u32, I16F16>::new();
//! for sample in current {
//!     extrema.update(sample)?;
//!     moments.update(sample)?;
//! }
//! let extrema = extrema.snapshot().expect("samples");
//! assert_eq!(extrema.maximum, I16F16::from_num(3.75));
//! let snapshot = moments.snapshot().expect("samples");
//! let mean: I16F16 = fixed_point::mean_nearest(snapshot.mean().expect("complete"))?;
//! assert_eq!(mean, I16F16::from_num(1.625));
//! # Ok::<(), krabilorean::Error>(())
//! ```

use core::num::NonZeroU64;

use crate::Error;
use crate::arithmetic::UnsignedRatio;
use crate::online::{EwMeanSnapshot, EwVarianceSnapshot};
use crate::policy::CounterWord;
use crate::sample::{MomentWord, SampleWord};

mod sealed {
    use core::num::NonZeroU64;

    use crate::arithmetic::{SignedRatio, UnsignedRatio, widen_count};

    c0nst::c0nst! {
        /// An exact mean ratio split into sign, magnitude, and denominator.
        pub c0nst trait MeanRatio: Copy {
            /// Sign, magnitude, and non-zero 64-bit denominator.
            fn parts(self) -> (bool, u64, NonZeroU64);
        }

        c0nst impl MeanRatio for SignedRatio {
            #[inline]
            fn parts(self) -> (bool, u64, NonZeroU64) {
                (
                    self.numerator < 0,
                    self.numerator.unsigned_abs(),
                    widen_count(self.denominator),
                )
            }
        }

        c0nst impl MeanRatio for UnsignedRatio {
            #[inline]
            fn parts(self) -> (bool, u64, NonZeroU64) {
                (false, self.numerator, self.denominator)
            }
        }

        /// Unsigned variance numerator widened to `u128`.
        pub c0nst trait VarianceNumerator: Copy {
            /// Exact widening to `u128`.
            fn to_u128(self) -> u128;
        }

        c0nst impl VarianceNumerator for u64 {
            #[inline]
            fn to_u128(self) -> u128 {
                self as u128
            }
        }

        c0nst impl VarianceNumerator for u128 {
            #[inline]
            fn to_u128(self) -> u128 {
                self
            }
        }
    }
}

c0nst::c0nst! {
    /// A fixed-point sample word, with the unsigned type its magnitudes convert to.
    ///
    /// Implemented by the signed words `FixedI8<Frac>`, `FixedI16<Frac>`, `FixedI32<Frac>`, whose
    /// mean is a [`SignedRatio`](crate::arithmetic::SignedRatio), and the unsigned words
    /// `FixedU8<Frac>`, `FixedU16<Frac>`, `FixedU32<Frac>`, whose mean is an [`UnsignedRatio`].
    /// The trait is sealed through [`SampleWord`].
    pub c0nst trait FixedPointWord:
        [c0nst] SampleWord + [c0nst] MomentWord<Mean: sealed::MeanRatio>
    {
        /// Same-width unsigned fixed-point type with the same fractional bits: `FixedU8<Frac>`,
        /// `FixedU16<Frac>`, or `FixedU32<Frac>`. An unsigned word is its own magnitude type.
        type Magnitude: Copy + Ord + core::fmt::Debug;

        /// The magnitude whose raw bits are `bits`, or `None` if they do not fit.
        fn magnitude_from_bits(bits: u64) -> Option<Self::Magnitude>;
    }
}

macro_rules! fixed_point_word {
    ($fixed:ident, $magnitude:ident, $leq:ident, $bits:ty) => {
        c0nst::c0nst! {
            c0nst impl<Frac: const_fixed::types::extra::$leq> FixedPointWord
                for const_fixed::$fixed<Frac>
            {
                type Magnitude = const_fixed::$magnitude<Frac>;

                #[inline]
                fn magnitude_from_bits(bits: u64) -> Option<Self::Magnitude> {
                    if bits <= (<$bits>::MAX as u64) {
                        Some(const_fixed::$magnitude::from_bits(bits as $bits))
                    } else {
                        None
                    }
                }
            }
        }
    };
}

fixed_point_word!(FixedI8, FixedU8, LeEqU8, u8);
fixed_point_word!(FixedI16, FixedU16, LeEqU16, u16);
fixed_point_word!(FixedI32, FixedU32, LeEqU32, u32);
fixed_point_word!(FixedU8, FixedU8, LeEqU8, u8);
fixed_point_word!(FixedU16, FixedU16, LeEqU16, u16);
fixed_point_word!(FixedU32, FixedU32, LeEqU32, u32);

/// `round(magnitude / denominator)`, ties away from zero, for a positive `denominator`.
const fn divide_nearest_ties_away(magnitude: u64, denominator: NonZeroU64) -> u64 {
    let divisor = denominator.get();
    let quotient = magnitude / divisor;
    let remainder = magnitude % divisor;
    // Round up when the remainder is at least half the denominator (`remainder < divisor`, so the
    // subtraction cannot underflow). A non-zero remainder requires `divisor >= 2`, so then
    // `quotient <= u64::MAX / 2` and the increment cannot overflow.
    quotient + ((remainder >= divisor - remainder) as u64)
}

/// Raw-unit shift between a snapshot's `fractional_bits` (which already include the word's own)
/// and the word's raw units. A snapshot with fewer fractional bits than the word came from a
/// different word.
const fn raw_shift<S: SampleWord>(fractional_bits: u32) -> Result<u32, Error> {
    match fractional_bits.checked_sub(S::FRACTIONAL_BITS) {
        Some(shift) => Ok(shift),
        None => Err(Error::InvalidConfiguration),
    }
}

c0nst::c0nst! {
    /// A signed raw value of `magnitude` with the given sign, narrowed into the word.
    c0nst fn signed_word<S: [c0nst] SampleWord>(
        negative: bool,
        magnitude: u64,
    ) -> Result<S, Error> {
        if magnitude > (i64::MAX as u64) {
            return Err(Error::Overflow);
        }
        let value = if negative {
            -(magnitude as i64)
        } else {
            magnitude as i64
        };
        match S::try_from_i64(value) {
            Some(word) => Ok(word),
            None => Err(Error::Overflow),
        }
    }

    /// Converts an exact raw-moments mean of a fixed-point word to the nearest representable
    /// value, ties away from zero.
    ///
    /// `mean` is [`RawMomentsSnapshot::mean`](crate::online::RawMomentsSnapshot::mean): raw sum
    /// over sample count, a [`SignedRatio`](crate::arithmetic::SignedRatio) for a signed word and
    /// an [`UnsignedRatio`] for an unsigned one. The mean of words lies between their extremes, so
    /// it always fits the word.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Overflow`] for a hand-built ratio outside the word. Neither ratio type can
    /// hold a zero denominator.
    pub c0nst fn mean_nearest<S: [c0nst] FixedPointWord>(mean: S::Mean) -> Result<S, Error>
    where
        S::Mean: [c0nst] sealed::MeanRatio,
    {
        let (negative, magnitude, denominator) = sealed::MeanRatio::parts(mean);
        signed_word(negative, divide_nearest_ties_away(magnitude, denominator))
    }

    /// Converts an exact non-negative ratio in raw units, such as
    /// [`SuccessiveDifferenceSnapshot::mean_absolute`](crate::online::SuccessiveDifferenceSnapshot::mean_absolute),
    /// to the nearest representable magnitude, ties away from zero.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Overflow`] when the result does not fit the magnitude type.
    pub c0nst fn magnitude_nearest<S: [c0nst] FixedPointWord>(
        ratio: UnsignedRatio,
    ) -> Result<S::Magnitude, Error> {
        let raw = divide_nearest_ties_away(ratio.numerator, ratio.denominator);
        match S::magnitude_from_bits(raw) {
            Some(magnitude) => Ok(magnitude),
            None => Err(Error::Overflow),
        }
    }

    /// Computes the population standard deviation `floor(sqrt(variance))` of a raw-unit population
    /// variance, such as
    /// [`RawMomentsSnapshot::population_variance`](crate::online::RawMomentsSnapshot::population_variance).
    ///
    /// Rounded down: `floor(sqrt(n / d)) = isqrt(floor(n / d))` for non-negative `n` and positive
    /// `d`, so the result is exact to the last raw bit, never above the true value. The standard
    /// deviation of words is at most half their span, so it fits the magnitude type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Overflow`] for a hand-built ratio whose root does not fit the magnitude
    /// type.
    pub c0nst fn standard_deviation_floor<S: [c0nst] FixedPointWord, N: [c0nst] sealed::VarianceNumerator>(
        variance: UnsignedRatio<N>,
    ) -> Result<S::Magnitude, Error> {
        let whole = variance.numerator.to_u128() / (variance.denominator.get() as u128);
        // `whole <= u128::MAX`, so `whole.isqrt() <= u64::MAX as u128`.
        let raw = whole.isqrt() as u64;
        match S::magnitude_from_bits(raw) {
            Some(magnitude) => Ok(magnitude),
            None => Err(Error::Overflow),
        }
    }

    /// Converts an [`EwMean`](crate::online::EwMean) snapshot's mean to the nearest representable
    /// value, ties away from zero.
    ///
    /// The snapshot's `fractional_bits` include the word's own, so the mean in raw units has
    /// `fractional_bits - S::FRACTIONAL_BITS` extra fractional bits that are rounded off.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] if the snapshot has fewer fractional bits than the
    /// word (from a different word), and [`Error::Overflow`] if the rounded mean does not fit.
    #[allow(clippy::question_mark)]
    pub c0nst fn ew_mean_nearest<C: CounterWord, S: [c0nst] FixedPointWord>(
        snapshot: &EwMeanSnapshot<C>,
    ) -> Result<S, Error> {
        let shift = match raw_shift::<S>(snapshot.fractional_bits) {
            Ok(shift) => shift,
            Err(error) => return Err(error),
        };
        let mean = snapshot.mean_q.unsigned_abs();
        let magnitude = if shift == 0 {
            mean
        } else if shift < u64::BITS {
            let half = 1_u64 << (shift - 1);
            (mean >> shift) + (((mean & ((1_u64 << shift) - 1)) >= half) as u64)
        } else if shift == u64::BITS {
            // `mean < 2^64 = 2^shift`: the quotient is below one, and rounds up only at one half
            // or above, which requires `mean >= 2^63`.
            (mean >= (1_u64 << 63)) as u64
        } else {
            0
        };
        signed_word(snapshot.mean_q < 0, magnitude)
    }

    /// Computes an [`EwVariance`](crate::online::EwVariance) snapshot's standard deviation,
    /// rounded down.
    ///
    /// The variance has twice the snapshot's `fractional_bits`, so
    /// `floor(sqrt(variance_q) / 2^shift) = isqrt(variance_q) >> shift`, where `shift` is the
    /// snapshot's fractional bits beyond the word's own.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for a snapshot from a word with fewer fractional
    /// bits, and [`Error::Overflow`] if the result does not fit the magnitude type.
    #[allow(clippy::question_mark)]
    pub c0nst fn ew_standard_deviation_floor<C: CounterWord, S: [c0nst] FixedPointWord>(
        snapshot: &EwVarianceSnapshot<C>,
    ) -> Result<S::Magnitude, Error> {
        let shift = match raw_shift::<S>(snapshot.fractional_bits) {
            Ok(shift) => shift,
            Err(error) => return Err(error),
        };
        // `variance_q <= u128::MAX`, so its integer square root fits `u64`.
        let root = snapshot.variance_q.isqrt() as u64;
        let raw = match root.checked_shr(shift) {
            Some(shifted) => shifted,
            None => 0,
        };
        match S::magnitude_from_bits(raw) {
            Some(magnitude) => Ok(magnitude),
            None => Err(Error::Overflow),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::num::{NonZeroU32, NonZeroU64};

    use const_fixed::types::extra::{U0, U4, U7, U8, U12, U15, U16, U24, U32};
    use const_fixed::{FixedI8, FixedI16, FixedI32, FixedU8, FixedU16, FixedU32};

    use super::{
        FixedPointWord, ew_mean_nearest, ew_standard_deviation_floor, magnitude_nearest,
        mean_nearest, standard_deviation_floor,
    };
    use crate::arithmetic::{NearestTiesAway, SignedRatio, UnsignedRatio, UnsignedWideRatio};
    use crate::histogram::HistogramBounds;
    use crate::online::{
        EwMean, EwMeanSnapshot, EwVariance, EwVarianceSnapshot, Extrema, LinearHistogram,
        PowerOfTwoHistogram, RunningMoments, SuccessiveDifference, ThresholdCrossings,
        ThresholdRuns,
    };
    use crate::policy::Checked;
    use crate::sample::{HistogramWord, MomentWord, SampleValue};
    use crate::{Error, Quality};

    type Amps = FixedI32<U16>;
    type Q15 = FixedI16<U15>;

    fn lcg(seed: u64, count: usize, bits: u32) -> std::vec::Vec<i64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> (64 - bits)) as i64 - (1_i64 << (bits - 1))
            })
            .collect()
    }

    fn raw_sets_for(bits: u32, signed: bool) -> std::vec::Vec<(&'static str, std::vec::Vec<i64>)> {
        let offset = if signed { 0 } else { 1_i64 << (bits - 1) };
        raw_sets(bits)
            .into_iter()
            .map(|(name, raw)| (name, raw.into_iter().map(|r| r + offset).collect()))
            .collect()
    }

    fn raw_sets(bits: u32) -> std::vec::Vec<(&'static str, std::vec::Vec<i64>)> {
        let min = -(1_i64 << (bits - 1));
        let max = (1_i64 << (bits - 1)) - 1;
        std::vec![
            ("constant zero", std::vec![0; 9]),
            ("constant max", std::vec![max; 7]),
            (
                "extremes alternating",
                (0..12)
                    .map(|i| if i % 2 == 0 { min } else { max })
                    .collect(),
            ),
            ("near constant", std::vec![5, 5, 6, 5, 5, 4, 5, 5]),
            (
                "ramp",
                (0..33).map(|i| min + i * ((max - min) / 32)).collect(),
            ),
            (
                "impulse",
                (0..16).map(|i| if i == 7 { max } else { -1 }).collect(),
            ),
            (
                "regime change",
                (0..20).map(|i| if i < 10 { -3 } else { max / 3 }).collect(),
            ),
            ("tiny alternating", (0..15).map(|i| i % 2).collect()),
            ("min then max", std::vec![min, max]),
            ("pseudo-random", lcg(bits as u64, 64, bits)),
        ]
    }

    #[derive(Debug, PartialEq)]
    struct RawState {
        extrema: (i64, i64),
        ew_mean: (i64, u32),
        ew_variance: (i64, u128, u32),
        moments: (i128, u128, std::string::String, Option<UnsignedWideRatio>),
        successive: (u64, u32, i64, i64, Option<UnsignedRatio>),
        crossings: (u32, u32, u32),
        runs: (u32, u32, u32),
        linear: (std::vec::Vec<u32>, u32, u32, std::vec::Vec<(i64, i64)>),
        power_of_two: (std::vec::Vec<u32>, std::vec::Vec<(i64, i64)>),
    }

    fn raw_state<S>(samples: &[S], threshold: i64) -> RawState
    where
        S: HistogramWord + MomentWord,
        S::Mean: core::fmt::Debug,
    {
        let word = |raw: i64| S::try_from_i64(raw).expect("raw value in the word");
        let (min, max) = (S::MIN.to_i64(), S::MAX.to_i64());
        let quarter = (max - min) / 4;
        let mut extrema = Extrema::<Checked, u32, S>::new();
        let mut ew_mean = EwMean::<3, 8, NearestTiesAway, Checked, u32, S>::new().expect("valid");
        let mut ew_variance =
            EwVariance::<3, 8, NearestTiesAway, Checked, u32, S>::new().expect("valid");
        let mut moments = RunningMoments::<Checked, u32, S>::new();
        let mut successive = SuccessiveDifference::<Checked, u32, S>::new();
        let mut crossings = ThresholdCrossings::<Checked, u32, S>::new(word(threshold));
        let mut runs = ThresholdRuns::<Checked, u32, S>::new(word(threshold));
        let bounds =
            HistogramBounds::<S>::new(word(min + quarter), word(max - quarter)).expect("bounds");
        let mut linear = LinearHistogram::<8, Checked, u32, S>::new(bounds).expect("valid");
        let mut power = PowerOfTwoHistogram::<8, Checked, u32, S>::new().expect("valid");
        for &sample in samples {
            extrema.update(sample).expect("extrema");
            ew_mean.update(sample).expect("ew mean");
            ew_variance.update(sample).expect("ew variance");
            moments.update(sample).expect("moments");
            successive.update(sample).expect("successive");
            crossings.update(sample).expect("crossings");
            runs.update(sample).expect("runs");
            linear.update(sample).expect("linear");
            power.update(sample).expect("power of two");
        }
        let extrema = extrema.snapshot().expect("samples");
        let ew_mean = ew_mean.snapshot().expect("samples");
        let ew_variance = ew_variance.snapshot().expect("samples");
        let moments = moments.snapshot().expect("samples");
        let successive = successive.snapshot().expect("samples");
        let crossings = crossings.snapshot().expect("samples");
        let runs = runs.snapshot().expect("samples");
        let histogram = linear.snapshot();
        let power_snapshot = power.snapshot();
        RawState {
            extrema: (extrema.minimum.to_i64(), extrema.maximum.to_i64()),
            ew_mean: (ew_mean.mean_q, ew_mean.fractional_bits - S::FRACTIONAL_BITS),
            ew_variance: (
                ew_variance.mean_q,
                ew_variance.variance_q,
                ew_variance.fractional_bits - S::FRACTIONAL_BITS,
            ),
            moments: (
                moments.sum.into(),
                moments.sum_of_squares.into(),
                std::format!("{:?}", moments.mean()),
                moments.population_variance(),
            ),
            successive: (
                successive.absolute_sum,
                successive.differences,
                successive.first_sample.to_i64(),
                successive.last_sample.to_i64(),
                successive.mean_absolute(),
            ),
            crossings: (crossings.crossings, crossings.rising, crossings.falling),
            runs: (runs.prefix_run, runs.current_run, runs.longest_run),
            linear: (
                histogram.counts.to_vec(),
                histogram.below,
                histogram.above,
                (0..8)
                    .map(|bin| {
                        let bounds = histogram.bin_bounds(bin).expect("bin");
                        (bounds.lower().to_i64(), bounds.upper().to_i64())
                    })
                    .collect(),
            ),
            power_of_two: (
                power_snapshot.counts.to_vec(),
                (0..8)
                    .map(|cell| {
                        let bounds = PowerOfTwoHistogram::<8, Checked, u32, S>::bucket_bounds(cell)
                            .expect("cell");
                        (bounds.lower().to_i64(), bounds.upper().to_i64())
                    })
                    .collect(),
            ),
        }
    }

    fn any_bit_identity<F, I>(bits: u32, signed: bool)
    where
        F: HistogramWord + MomentWord,
        I: HistogramWord + MomentWord,
        F::Mean: core::fmt::Debug,
        I::Mean: core::fmt::Debug,
    {
        let threshold = if signed { 3 } else { (1 << (bits - 1)) + 3 };
        for (name, raw) in raw_sets_for(bits, signed) {
            let fixed: std::vec::Vec<F> = raw
                .iter()
                .map(|&r| F::try_from_i64(r).expect("raw"))
                .collect();
            let integer: std::vec::Vec<I> = raw
                .iter()
                .map(|&r| I::try_from_i64(r).expect("raw"))
                .collect();
            assert_eq!(
                raw_state(&fixed, threshold),
                raw_state(&integer, threshold),
                "{name}"
            );
        }
    }

    #[test]
    fn fixed_point_words_match_their_integer_words_bit_for_bit() {
        any_bit_identity::<FixedI32<U16>, i32>(32, true);
        any_bit_identity::<FixedI32<U0>, i32>(32, true);
        any_bit_identity::<FixedI32<U32>, i32>(32, true);
        any_bit_identity::<FixedI16<U15>, i16>(16, true);
        any_bit_identity::<FixedI16<U8>, i16>(16, true);
        any_bit_identity::<FixedI16<U0>, i16>(16, true);
        any_bit_identity::<FixedI8<U4>, i8>(8, true);
        any_bit_identity::<FixedI8<U7>, i8>(8, true);
        any_bit_identity::<FixedI8<U8>, i8>(8, true);
        any_bit_identity::<FixedU8<U0>, u8>(8, false);
        any_bit_identity::<FixedU8<U8>, u8>(8, false);
        any_bit_identity::<FixedU16<U12>, u16>(16, false);
        any_bit_identity::<FixedU16<U16>, u16>(16, false);
        any_bit_identity::<FixedU32<U24>, u32>(32, false);
        any_bit_identity::<FixedU32<U32>, u32>(32, false);
    }

    #[test]
    fn snapshots_add_the_words_fractional_bits() {
        let mut ew_mean =
            EwMean::<3, 8, NearestTiesAway, Checked, u32, Amps>::new().expect("valid");
        let mut ew_variance =
            EwVariance::<3, 8, NearestTiesAway, Checked, u32, Amps>::new().expect("valid");
        ew_mean.update(Amps::from_num(1.5)).expect("update");
        ew_variance.update(Amps::from_num(1.5)).expect("update");
        let mean = ew_mean.snapshot().expect("sample");
        assert_eq!(mean.fractional_bits, 24);
        assert_eq!(mean.mean_q, 3 << 23);
        assert_eq!(ew_variance.snapshot().expect("sample").fractional_bits, 24);
    }

    /// Nearest integer to `numerator / denominator`, ties away from zero, by comparing the two
    /// candidates around the exact quotient.
    fn oracle_nearest(numerator: i128, denominator: i128) -> i128 {
        let floor = numerator.div_euclid(denominator);
        let candidates = [floor, floor + 1];
        let mut best = candidates[0];
        for candidate in candidates {
            let distance = (2 * (candidate * denominator - numerator)).abs();
            let best_distance = (2 * (best * denominator - numerator)).abs();
            if distance < best_distance
                || (distance == best_distance && candidate.abs() > best.abs())
            {
                best = candidate;
            }
        }
        best
    }

    /// Largest `r` with `r^2 * scale <= value`, by binary search.
    fn oracle_floor_root(value: u128, scale: u128) -> u128 {
        let (mut low, mut high) = (0_u128, 1_u128 << 64);
        while low < high {
            let middle = (low + high).div_ceil(2);
            if middle
                .checked_mul(middle)
                .and_then(|m| m.checked_mul(scale))
                .is_some_and(|m| m <= value)
            {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        low
    }

    struct MagnitudeBits(u64);

    impl<Frac: const_fixed::types::extra::LeEqU32> From<const_fixed::FixedU32<Frac>> for MagnitudeBits {
        fn from(value: const_fixed::FixedU32<Frac>) -> Self {
            Self(u64::from(value.to_bits()))
        }
    }

    impl<Frac: const_fixed::types::extra::LeEqU8> From<const_fixed::FixedU8<Frac>> for MagnitudeBits {
        fn from(value: const_fixed::FixedU8<Frac>) -> Self {
            Self(u64::from(value.to_bits()))
        }
    }

    impl<Frac: const_fixed::types::extra::LeEqU16> From<const_fixed::FixedU16<Frac>> for MagnitudeBits {
        fn from(value: const_fixed::FixedU16<Frac>) -> Self {
            Self(u64::from(value.to_bits()))
        }
    }

    fn conversions_against_oracles<S>(bits: u32)
    where
        S: FixedPointWord,
        S::Magnitude: Into<MagnitudeBits>,
    {
        let signed = S::MIN.to_i64() < 0;
        for (name, raw) in raw_sets_for(bits, signed) {
            let samples: std::vec::Vec<S> = raw
                .iter()
                .map(|&r| S::try_from_i64(r).expect("raw"))
                .collect();
            let mut moments = RunningMoments::<Checked, u32, S>::new();
            let mut successive = SuccessiveDifference::<Checked, u32, S>::new();
            let mut ew_mean =
                EwMean::<3, 8, NearestTiesAway, Checked, u32, S>::new().expect("valid");
            let mut ew_variance =
                EwVariance::<3, 8, NearestTiesAway, Checked, u32, S>::new().expect("valid");
            for &sample in &samples {
                moments.update(sample).expect("moments");
                successive.update(sample).expect("successive");
                ew_mean.update(sample).expect("ew mean");
                ew_variance.update(sample).expect("ew variance");
            }
            let n = raw.len() as i128;
            let sum: i128 = raw.iter().map(|&r| i128::from(r)).sum();

            let snapshot = moments.snapshot().expect("samples");
            let mean: S = mean_nearest(snapshot.mean().expect("complete")).expect("mean");
            assert_eq!(
                i128::from(mean.to_i64()),
                oracle_nearest(sum, n),
                "{name}: mean"
            );

            let spread: u128 = raw
                .iter()
                .map(|&r| (n * i128::from(r) - sum).unsigned_abs().pow(2))
                .sum();
            let deviation =
                standard_deviation_floor::<S, _>(snapshot.population_variance().expect("variance"))
                    .expect("deviation");
            assert_eq!(
                u128::from(deviation.into().0),
                oracle_floor_root(spread, (n as u128).pow(3)),
                "{name}: standard deviation"
            );

            let ratio = successive
                .snapshot()
                .expect("samples")
                .mean_absolute()
                .expect("differences");
            let absolute: i128 = raw
                .windows(2)
                .map(|w| (i128::from(w[1]) - i128::from(w[0])).abs())
                .sum();
            let masd = magnitude_nearest::<S>(ratio).expect("masd");
            assert_eq!(
                i128::from(masd.into().0),
                oracle_nearest(absolute, n - 1),
                "{name}: mean absolute successive difference"
            );

            let ew = ew_mean.snapshot().expect("samples");
            let ew_value: S = ew_mean_nearest(&ew).expect("ew mean");
            assert_eq!(
                i128::from(ew_value.to_i64()),
                oracle_nearest(i128::from(ew.mean_q), 1 << 8),
                "{name}: ew mean"
            );

            let ew = ew_variance.snapshot().expect("samples");
            let ew_deviation = ew_standard_deviation_floor::<_, S>(&ew).expect("ew deviation");
            assert_eq!(
                u128::from(ew_deviation.into().0),
                oracle_floor_root(ew.variance_q, 1 << 16),
                "{name}: ew standard deviation"
            );
        }
    }

    #[test]
    fn conversions_match_exact_oracles() {
        conversions_against_oracles::<Amps>(32);
        conversions_against_oracles::<FixedI32<U0>>(32);
        conversions_against_oracles::<FixedI32<U32>>(32);
        conversions_against_oracles::<Q15>(16);
        conversions_against_oracles::<FixedI16<U0>>(16);
        conversions_against_oracles::<FixedI8<U4>>(8);
        conversions_against_oracles::<FixedI8<U8>>(8);
        conversions_against_oracles::<FixedU8<U8>>(8);
        conversions_against_oracles::<FixedU16<U12>>(16);
        conversions_against_oracles::<FixedU16<U0>>(16);
        conversions_against_oracles::<FixedU32<U24>>(32);
        conversions_against_oracles::<FixedU32<U32>>(32);
    }

    #[test]
    fn conversions_agree_with_physical_units() {
        let amps = [12.5, -3.25, 7.0, 0.015625, 9.75, -11.5, 4.0, 2.5];
        let samples = amps.map(Amps::from_num);
        let mut moments = RunningMoments::<Checked, u32, Amps>::new();
        for sample in samples {
            moments.update(sample).expect("update");
        }
        let snapshot = moments.snapshot().expect("samples");
        let mean: Amps = mean_nearest(snapshot.mean().expect("complete")).expect("mean");
        let deviation =
            standard_deviation_floor::<Amps, _>(snapshot.population_variance().expect("variance"))
                .expect("deviation");
        let n = amps.len() as f64;
        let mean_f = amps.iter().sum::<f64>() / n;
        let deviation_f = (amps.iter().map(|a| (a - mean_f).powi(2)).sum::<f64>() / n).sqrt();
        let lsb = 1.0 / 65_536.0;
        assert!((mean.to_num::<f64>() - mean_f).abs() <= lsb / 2.0);
        assert!(deviation.to_num::<f64>() <= deviation_f);
        assert!(deviation_f - deviation.to_num::<f64>() < lsb);
    }

    #[test]
    fn conversions_handle_edge_shifts_and_reject_invalid_inputs() {
        let wide_signed = SignedRatio {
            numerator: i64::MAX,
            denominator: NonZeroU32::MIN,
        };
        assert_eq!(mean_nearest::<Amps>(wide_signed), Err(Error::Overflow));

        let wide_unsigned = UnsignedRatio {
            numerator: u64::MAX,
            denominator: NonZeroU64::MIN,
        };
        assert_eq!(
            mean_nearest::<FixedU32<U16>>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<Q15>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<FixedI8<U4>>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<Amps>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<FixedU8<U4>>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<FixedU16<U12>>(wide_unsigned),
            Err(Error::Overflow)
        );
        assert_eq!(
            magnitude_nearest::<FixedU32<U16>>(wide_unsigned),
            Err(Error::Overflow)
        );

        let wide_variance = UnsignedWideRatio {
            numerator: u128::MAX,
            denominator: NonZeroU64::MIN,
        };
        assert_eq!(
            standard_deviation_floor::<Q15, _>(wide_variance),
            Err(Error::Overflow)
        );
        let u64_variance = UnsignedRatio::<u64> {
            numerator: 144,
            denominator: NonZeroU64::new(9).expect("non-zero"),
        };
        assert_eq!(
            standard_deviation_floor::<Amps, _>(u64_variance),
            Ok(FixedU32::<U16>::from_bits(4))
        );

        // `shift == 0` when `EwMean` has `FRACTIONAL_BITS == 0`.
        let mut q0_mean =
            EwMean::<3, 0, NearestTiesAway, Checked, u32, Amps>::new().expect("valid");
        q0_mean.update(Amps::from_num(2.25)).expect("update");
        let q0_snapshot = q0_mean.snapshot().expect("sample");
        assert_eq!(
            ew_mean_nearest::<_, Amps>(&q0_snapshot),
            Ok(Amps::from_num(2.25))
        );

        // `shift == 64` rounds `mean >= 2^63` (`mean_q == i64::MIN`) away from zero to `-1` raw bit,
        // and `shift > 64` rounds to `0`.
        let shift_64 = EwMeanSnapshot::<u32> {
            samples: NonZeroU32::MIN,
            mean_q: i64::MIN,
            fractional_bits: 16 + 64,
            alpha_shift: 3,
            quality: Quality::Approximate,
        };
        assert_eq!(
            ew_mean_nearest::<_, Amps>(&shift_64),
            Ok(Amps::from_bits(-1))
        );
        let shift_65 = EwMeanSnapshot::<u32> {
            fractional_bits: 16 + 65,
            ..shift_64
        };
        assert_eq!(
            ew_mean_nearest::<_, Amps>(&shift_65),
            Ok(<Amps as SampleValue>::ZERO)
        );

        // A snapshot from an integer word has fewer fractional bits than the fixed-point word.
        let mut integer = EwMean::<3, 8, NearestTiesAway, Checked, u32, i32>::new().expect("valid");
        integer.update(5).expect("update");
        let snapshot = integer.snapshot().expect("sample");
        assert_eq!(
            ew_mean_nearest::<_, Amps>(&snapshot),
            Err(Error::InvalidConfiguration)
        );

        let bad_variance = EwVarianceSnapshot::<u32> {
            samples: NonZeroU32::MIN,
            mean_q: 0,
            variance_q: 0,
            fractional_bits: 8,
            alpha_shift: 3,
            quality: Quality::Approximate,
        };
        assert_eq!(
            ew_standard_deviation_floor::<_, Amps>(&bad_variance),
            Err(Error::InvalidConfiguration)
        );

        let huge_variance = EwVarianceSnapshot::<u32> {
            variance_q: u128::MAX,
            fractional_bits: 16,
            ..bad_variance
        };
        assert_eq!(
            ew_standard_deviation_floor::<_, Amps>(&huge_variance),
            Err(Error::Overflow)
        );

        let wide_shift_variance = EwVarianceSnapshot::<u32> {
            variance_q: u128::MAX,
            fractional_bits: 16 + 64,
            ..bad_variance
        };
        assert_eq!(
            ew_standard_deviation_floor::<_, Amps>(&wide_shift_variance),
            Ok(FixedU32::<U16>::from_bits(0))
        );
    }

    #[test]
    fn word_typed_configuration_is_in_physical_units() {
        let limit = Amps::from_num(10);
        let mut crossings = ThresholdCrossings::<Checked, u32, Amps>::new(limit);
        let bounds =
            HistogramBounds::<Amps>::new(Amps::from_num(-16), Amps::from_num(16) - Amps::DELTA)
                .expect("bounds");
        let mut histogram = LinearHistogram::<4, Checked, u32, Amps>::new(bounds).expect("valid");
        for amps in [9.5, 10.5, 9.0, 11.0, -20.0, 15.999] {
            let sample = Amps::from_num(amps);
            crossings.update(sample).expect("update");
            histogram.update(sample).expect("update");
        }
        let crossings = crossings.snapshot().expect("samples");
        assert_eq!((crossings.rising, crossings.falling), (3, 2));
        let histogram = histogram.snapshot();
        assert_eq!(histogram.below, 1);
        assert_eq!(histogram.counts, &[0, 0, 0, 5]);
        assert_eq!(
            histogram.bin_bounds(3),
            HistogramBounds::new(Amps::from_num(8), Amps::from_num(16) - Amps::DELTA).ok()
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_fixed_point_conversions_are_const_evaluable_on_nightly() {
        const MEAN: Result<Amps, Error> = mean_nearest::<Amps>(SignedRatio {
            numerator: 3 << 16,
            denominator: NonZeroU32::new(2).expect("non-zero"),
        });
        const MAGNITUDE: Result<FixedU32<U16>, Error> = magnitude_nearest::<Amps>(UnsignedRatio {
            numerator: 5 << 16,
            denominator: NonZeroU64::new(2).expect("non-zero"),
        });
        const STD_DEV: Result<FixedU32<U16>, Error> =
            standard_deviation_floor::<Amps, u128>(UnsignedWideRatio {
                numerator: 9 << 32,
                denominator: NonZeroU64::new(1).expect("non-zero"),
            });
        const EW_MEAN: Result<Amps, Error> = ew_mean_nearest::<u32, Amps>(&EwMeanSnapshot {
            samples: NonZeroU32::MIN,
            mean_q: 3 << 23,
            fractional_bits: 24,
            alpha_shift: 3,
            quality: Quality::Approximate,
        });
        const EW_STD: Result<FixedU32<U16>, Error> =
            ew_standard_deviation_floor::<u32, Amps>(&EwVarianceSnapshot {
                samples: NonZeroU32::MIN,
                mean_q: 0,
                variance_q: 9 << 48,
                fractional_bits: 24,
                alpha_shift: 3,
                quality: Quality::Approximate,
            });
        assert_eq!(MEAN, Ok(Amps::from_bits(3 << 15)));
        assert_eq!(MAGNITUDE, Ok(FixedU32::<U16>::from_bits(5 << 15)));
        assert_eq!(STD_DEV, Ok(FixedU32::<U16>::from_bits(3 << 16)));
        assert_eq!(EW_MEAN, Ok(Amps::from_bits(3 << 15)));
        assert_eq!(EW_STD, Ok(FixedU32::<U16>::from_bits(3 << 16)));
    }
}
