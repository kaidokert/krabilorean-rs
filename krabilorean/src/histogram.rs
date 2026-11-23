//! Shared fixed-range linear histogram semantics.

use core::num::{NonZeroU32, NonZeroU64};

use crate::Error;
use crate::distribution::{Bucket, ValueBounds};
use crate::sample::HistogramWord;

/// Inclusive sample bounds `lower..=upper` with `lower <= upper` guaranteed by construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramBounds<S> {
    /// Lowest in-range sample, included in bin zero.
    lower: S,
    /// Highest in-range sample, included in the final bin.
    upper: S,
}

c0nst::c0nst! {
    impl<S: HistogramWord> HistogramBounds<S> {
        /// Creates inclusive bounds with a non-empty range (`lower <= upper`).
        pub c0nst fn new(lower: S, upper: S) -> Result<Self, Error>
        where
            S: [c0nst] HistogramWord,
        {
            if lower.to_i64() > upper.to_i64() {
                Err(Error::InvalidConfiguration)
            } else {
                Ok(Self { lower, upper })
            }
        }

        /// Returns the number of distinct integer sample values in the range, in `1..=2^S::BITS`.
        pub c0nst fn span(self) -> NonZeroU64
        where
            S: [c0nst] HistogramWord,
        {
            S::span_values(self.lower, self.upper, crate::sealed::Internal::new())
        }

        /// Returns whether `self` and `other` have identical lower and upper bounds.
        pub c0nst fn eq_bounds(self, other: Self) -> bool
        where
            S: [c0nst] HistogramWord,
        {
            self.lower.to_i64() == other.lower.to_i64()
                && self.upper.to_i64() == other.upper.to_i64()
        }
    }
}

impl<S: Copy> HistogramBounds<S> {
    /// Crate-internal constructor for callers that have already established `lower <= upper`.
    pub(crate) const fn from_ordered(lower: S, upper: S) -> Self {
        Self { lower, upper }
    }

    /// Returns the inclusive lower bound.
    pub const fn lower(self) -> S {
        self.lower
    }

    /// Returns the inclusive upper bound.
    pub const fn upper(self) -> S {
        self.upper
    }
}

/// The lowest-index bin holding the most samples, with its count.
///
/// A histogram has a mode exactly when it holds at least one in-range sample, so the count is
/// never zero. `B` is the bin index (`u32` by default, since configurable histograms cap bin counts
/// at `u32::MAX`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramMode<B = u32> {
    /// Lowest index among the bins with the largest count.
    pub bin: B,
    /// Samples in that bin.
    pub count: NonZeroU32,
}

const _: () =
    assert!(core::mem::size_of::<Option<HistogramMode>>() == core::mem::size_of::<HistogramMode>());

/// The lowest-index mode of `counts`, or `None` when every count is zero.
pub(crate) const fn lowest_mode(counts: &[u32]) -> Option<HistogramMode> {
    let mut mode: Option<HistogramMode> = None;
    let mut largest = 0_u32;
    let mut index = 0_usize;
    while index < counts.len() {
        let count = counts[index];
        if count > largest {
            largest = count;
            // `largest >= 0` and `count > largest`, so `count >= 1` and `NonZeroU32::MIN` is
            // unreachable.
            let nonzero = match NonZeroU32::new(count) {
                Some(nonzero) => nonzero,
                None => NonZeroU32::MIN,
            };
            mode = Some(HistogramMode {
                bin: index as u32,
                count: nonzero,
            });
        }
        index += 1;
    }
    mode
}

/// Classification of one sample relative to histogram bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HistogramClass {
    Below,
    Bin(usize),
    Above,
}

/// Validates the compile-time bin count.
pub const fn bins_are_valid<const BINS: usize>() -> bool {
    BINS > 0 && BINS <= u32::MAX as usize && BINS <= usize::MAX - 2
}

/// Width in bits of the unsigned arithmetic that computes a linear bin for a `word_bits`-bit
/// sample word and `bins` bins: 16, 32, or 64.
///
/// The bin is `floor(offset * bins / span)` with `offset = sample - lower`. For a `w`-bit word,
/// `offset <= 2^w - 1` and `1 <= span <= 2^w`. An `n`-bit unsigned type with `n > w` holds the
/// span, and holds the product whenever `bins <= 2^(n - w)`, because then
/// `offset * bins <= (2^w - 1) * 2^(n - w) < 2^n`. The narrowest `n` in `{16, 32}` that meets both
/// is chosen, and 64 otherwise. Every width gives the same bin; a narrower one avoids a wider
/// software division on cores without a hardware divide instruction.
///
/// With `bins` up to the [`bins_are_valid`] cap of `u32::MAX`:
///
/// | Word | 16 bits | 32 bits | 64 bits |
/// |---|---|---|---|
/// | `i8`, `u8` | `bins <= 2^8` | `bins <= 2^24` | above |
/// | `i16`, `u16` | never | `bins <= 2^16` | above |
/// | `i32`, `u32` | never | never | always |
pub(crate) const fn bin_arithmetic_bits(word_bits: u32, bins: u64) -> u32 {
    if word_bits < 16 && bins <= 1 << (16 - word_bits) {
        16
    } else if word_bits < 32 && bins <= 1 << (32 - word_bits) {
        32
    } else {
        64
    }
}

c0nst::c0nst! {
    /// Maps a sample to its inclusive integer-domain linear bin or out-of-range side.
    ///
    /// Callers must validate `BINS` with [`bins_are_valid`]. The bin is
    /// `floor((sample - lower) * BINS / span)`, computed in the width [`bin_arithmetic_bits`]
    /// selects for the word and `BINS`. In the 64-bit arm `offset < 2^S::BITS <= 2^32` and
    /// `BINS <= u32::MAX`, so the product is below `2^64` and `1 <= span <= 2^32`.
    pub(crate) c0nst fn classify<const BINS: usize, S: [c0nst] HistogramWord>(
        bounds: HistogramBounds<S>,
        sample: S,
    ) -> HistogramClass {
        if !sample.at_or_above(bounds.lower) {
            return HistogramClass::Below;
        }
        if sample.above(bounds.upper) {
            return HistogramClass::Above;
        }

        let offset = sample.offset_from(bounds.lower, crate::sealed::Internal::new());
        let span = bounds.span().get();
        // Evaluated at compile time for each word and `BINS`, so an optimized build keeps one arm.
        // The narrowing casts are exact: `bin_arithmetic_bits` proves that `offset`, `span >= 1`,
        // and `offset * BINS` fit the selected width, and `offset < span` implies `index < BINS`.
        let index = match const { bin_arithmetic_bits(S::BITS, BINS as u64) } {
            16 => ((offset as u16 * BINS as u16) / (span as u16)) as usize,
            32 => ((offset as u32 * BINS as u32) / (span as u32)) as usize,
            _ => ((offset * BINS as u64) / span) as usize,
        };
        HistogramClass::Bin(index)
    }

    /// Returns the inclusive sample bounds of linear bin `bin`, at any width.
    ///
    /// Bin `k` holds the offsets `ceil(k * span / BINS) ..= ceil((k + 1) * span / BINS) - 1`.
    /// Returns `None` for an empty bin (more bins than distinct values in `bounds`) or an index
    /// past the last bin.
    pub(crate) c0nst fn linear_bin_bounds<const BINS: usize, S: [c0nst] HistogramWord>(
        bounds: HistogramBounds<S>,
        bin: usize,
    ) -> Option<HistogramBounds<S>> {
        if bin >= BINS {
            return None;
        }
        let width = bounds.span().get();
        let bins_u64 = BINS as u64;
        let bin_u64 = bin as u64;
        let first_offset = (bin_u64 * width).div_ceil(bins_u64);
        let after_offset = ((bin_u64 + 1) * width).div_ceil(bins_u64);
        if first_offset >= after_offset {
            return None;
        }
        let base = bounds.lower.to_i64();
        // `0 <= first_offset < after_offset <= width`, so `lower <= upper` and both endpoints lie
        // within `bounds.lower..=bounds.upper` and always fit `S`.
        let lower = match S::try_from_i64(base + first_offset as i64) {
            Some(value) => value,
            None => bounds.lower,
        };
        let upper = match S::try_from_i64(base + (after_offset - 1) as i64) {
            Some(value) => value,
            None => bounds.upper,
        };
        Some(HistogramBounds::from_ordered(lower, upper))
    }

    /// Returns one complete-domain bucket from a linear histogram snapshot: the below-range
    /// bucket `S::MIN..=lower - 1`, the in-range bins, then the above-range bucket
    /// `upper + 1..=S::MAX`. An edge bucket is `None` when the bounds reach that end of the word.
    pub(crate) c0nst fn linear_bucket<const BINS: usize, S: [c0nst] HistogramWord>(
        bounds: HistogramBounds<S>,
        counts: &[u32; BINS],
        below: u32,
        above: u32,
        index: usize,
    ) -> Option<Bucket<S>> {
        if index == 0 {
            if !bounds.lower.above(S::MIN) {
                return None;
            }
            // `bounds.lower > S::MIN`, so `bounds.lower - 1` lies within `S::MIN..=S::MAX`.
            let upper = match S::try_from_i64(bounds.lower.to_i64() - 1) {
                Some(value) => value,
                None => S::MIN,
            };
            return Some(Bucket {
                values: ValueBounds::from_ordered(S::MIN, upper),
                count: below,
            });
        }
        if index == BINS + 1 {
            if !S::MAX.above(bounds.upper) {
                return None;
            }
            // `bounds.upper < S::MAX`, so `bounds.upper + 1` lies within `S::MIN..=S::MAX`.
            let lower = match S::try_from_i64(bounds.upper.to_i64() + 1) {
                Some(value) => value,
                None => S::MAX,
            };
            return Some(Bucket {
                values: ValueBounds::from_ordered(lower, S::MAX),
                count: above,
            });
        }
        let bin = index - 1;
        let bin_bounds = match linear_bin_bounds::<BINS, S>(bounds, bin) {
            Some(bin_bounds) => bin_bounds,
            None => return None,
        };
        Some(Bucket {
            values: ValueBounds::from_histogram_bounds(bin_bounds),
            count: counts[bin],
        })
    }
}

#[cfg(test)]
mod tests {
    use core::num::{NonZeroU32, NonZeroU64};

    use super::{
        HistogramBounds, HistogramClass, HistogramMode, bin_arithmetic_bits, bins_are_valid,
        classify, linear_bin_bounds, linear_bucket, lowest_mode,
    };
    use crate::Error;
    use crate::distribution::ValueBounds;
    use crate::sample::HistogramWord;

    #[test]
    fn invalid_bounds_and_bin_counts_are_rejected() {
        assert_eq!(
            HistogramBounds::<i16>::new(10, -10),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            HistogramBounds::<u8>::new(20, 19),
            Err(Error::InvalidConfiguration)
        );
        let a = HistogramBounds::<i16>::new(-4, 3).expect("valid");
        let b = HistogramBounds::<i16>::new(-4, 4).expect("valid");
        assert!(a.eq_bounds(a));
        assert!(!a.eq_bounds(b));
        assert!(!bins_are_valid::<0>());
        assert!(bins_are_valid::<1>());
        assert!(bins_are_valid::<16>());
    }

    #[test]
    fn lowest_mode_picks_first_highest_nonzero_bin() {
        assert_eq!(lowest_mode(&[0, 0, 0]), None);
        assert_eq!(
            lowest_mode(&[0, 3, 5, 5, 2]),
            Some(HistogramMode {
                bin: 2,
                count: NonZeroU32::new(5).expect("nonzero"),
            })
        );
    }

    #[test]
    fn linear_bin_bounds_partition_the_range_and_handle_empty_bins() {
        let bounds = HistogramBounds::<i16>::new(-4, 3).expect("valid bounds");
        assert_eq!(
            linear_bin_bounds::<4, _>(bounds, 0),
            HistogramBounds::new(-4, -3).ok()
        );
        assert_eq!(
            linear_bin_bounds::<4, _>(bounds, 1),
            HistogramBounds::new(-2, -1).ok()
        );
        assert_eq!(
            linear_bin_bounds::<4, _>(bounds, 2),
            HistogramBounds::new(0, 1).ok()
        );
        assert_eq!(
            linear_bin_bounds::<4, _>(bounds, 3),
            HistogramBounds::new(2, 3).ok()
        );
        assert_eq!(linear_bin_bounds::<4, _>(bounds, 4), None);

        // More bins (4) than distinct values in the range (2): two bins are empty (`None`).
        let narrow = HistogramBounds::<i8>::new(10, 11).expect("valid bounds");
        assert_eq!(
            linear_bin_bounds::<4, _>(narrow, 0),
            HistogramBounds::new(10, 10).ok()
        );
        assert_eq!(linear_bin_bounds::<4, _>(narrow, 1), None);
        assert_eq!(
            linear_bin_bounds::<4, _>(narrow, 2),
            HistogramBounds::new(11, 11).ok()
        );
        assert_eq!(linear_bin_bounds::<4, _>(narrow, 3), None);

        let counts = [3_u32, 0, 5, 0];
        assert_eq!(
            linear_bucket::<4, i8>(narrow, &counts, 2, 7, 0).map(|b| (b.values, b.count)),
            Some((ValueBounds::new(i8::MIN, 9).expect("valid"), 2))
        );
        assert_eq!(
            linear_bucket::<4, i8>(narrow, &counts, 2, 7, 1).map(|b| (b.values, b.count)),
            Some((ValueBounds::new(10, 10).expect("valid"), 3))
        );
        assert_eq!(linear_bucket::<4, i8>(narrow, &counts, 2, 7, 2), None);
        assert_eq!(
            linear_bucket::<4, i8>(narrow, &counts, 2, 7, 3).map(|b| (b.values, b.count)),
            Some((ValueBounds::new(11, 11).expect("valid"), 5))
        );
        assert_eq!(linear_bucket::<4, i8>(narrow, &counts, 2, 7, 4), None);
        assert_eq!(
            linear_bucket::<4, i8>(narrow, &counts, 2, 7, 5).map(|b| (b.values, b.count)),
            Some((ValueBounds::new(12, i8::MAX).expect("valid"), 7))
        );
        assert_eq!(linear_bucket::<4, i8>(narrow, &counts, 2, 7, 6), None);

        let full = HistogramBounds::<i8>::new(i8::MIN, i8::MAX).expect("valid");
        assert_eq!(linear_bucket::<4, i8>(full, &counts, 0, 0, 0), None);
        assert_eq!(linear_bucket::<4, i8>(full, &counts, 0, 0, 5), None);
    }

    /// Every `(word_bits, bins)` pair that selects a 16- or 32-bit division holds the full-range
    /// span `2^word_bits` and the largest product `(2^word_bits - 1) * bins` in `u128` without
    /// exceeding that width's `MAX`.
    #[test]
    fn the_selected_bin_width_holds_every_intermediate() {
        for word_bits in [8_u32, 16, 32] {
            let max_span = 1_u128 << word_bits;
            let max_offset = max_span - 1;
            for bins in [
                1_u64,
                2,
                16,
                255,
                256,
                257,
                65_535,
                65_536,
                65_537,
                (1 << 24) - 1,
                1 << 24,
                (1 << 24) + 1,
                u64::from(u32::MAX),
            ] {
                let selected = bin_arithmetic_bits(word_bits, bins);
                let limit = match selected {
                    16 => u128::from(u16::MAX),
                    32 => u128::from(u32::MAX),
                    64 => u128::from(u64::MAX),
                    other => panic!("unexpected arithmetic width {other}"),
                };
                assert!(max_span <= limit, "word {word_bits} bins {bins}");
                assert!(
                    max_offset * u128::from(bins) <= limit,
                    "word {word_bits} bins {bins} selected {selected}"
                );
            }
        }
        assert_eq!(bin_arithmetic_bits(8, 256), 16);
        assert_eq!(bin_arithmetic_bits(8, 257), 32);
        assert_eq!(bin_arithmetic_bits(8, 1 << 24), 32);
        assert_eq!(bin_arithmetic_bits(8, (1 << 24) + 1), 64);
        assert_eq!(bin_arithmetic_bits(16, 1), 32);
        assert_eq!(bin_arithmetic_bits(16, 65_536), 32);
        assert_eq!(bin_arithmetic_bits(16, 65_537), 64);
        assert_eq!(bin_arithmetic_bits(32, 1), 64);
    }

    /// Independent oracle: `floor((sample - lower) * bins / (upper - lower + 1))` in `u128`.
    fn bin_oracle(lower: i64, upper: i64, bins: u128, sample: i64) -> HistogramClass {
        if sample < lower {
            return HistogramClass::Below;
        }
        if sample > upper {
            return HistogramClass::Above;
        }
        let offset = (sample - lower) as u128;
        let span = (upper - lower + 1) as u128;
        HistogramClass::Bin((offset * bins / span) as usize)
    }

    fn agrees_with_oracle<const BINS: usize, S: HistogramWord>(lower: S, upper: S) {
        let bounds = HistogramBounds::new(lower, upper).expect("valid bounds");
        let (lo, hi) = (lower.to_i64(), upper.to_i64());
        let span = hi - lo + 1;
        let mut samples = std::vec![S::MIN, S::MAX, S::ZERO, lower, upper];
        for bin in 0..=BINS.min(4096) as i64 {
            let edge = lo + (bin as u128 * span as u128).div_ceil(BINS as u128) as i64;
            samples.extend((edge - 1..=edge + 1).filter_map(S::try_from_i64));
        }
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..2_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            samples.extend(S::try_from_i64(
                S::MIN.to_i64() + (state % (1 << S::BITS)) as i64,
            ));
        }
        for sample in samples {
            assert_eq!(
                classify::<BINS, S>(bounds, sample),
                bin_oracle(lo, hi, BINS as u128, sample.to_i64()),
                "bins {BINS} bounds {lo}..={hi} sample {}",
                sample.to_i64()
            );
        }
    }

    #[test]
    fn every_bin_width_matches_the_wide_oracle() {
        agrees_with_oracle::<256, i8>(i8::MIN, i8::MAX);
        agrees_with_oracle::<257, i8>(i8::MIN, i8::MAX);
        agrees_with_oracle::<{ 1 << 24 }, i8>(i8::MIN, i8::MAX);
        agrees_with_oracle::<{ (1 << 24) + 1 }, i8>(i8::MIN, i8::MAX);
        agrees_with_oracle::<7, i8>(-3, 99);
        agrees_with_oracle::<16, i16>(i16::MIN, i16::MAX);
        agrees_with_oracle::<16, i16>(-20_000, 19_999);
        agrees_with_oracle::<65_536, i16>(i16::MIN, i16::MAX);
        agrees_with_oracle::<65_537, i16>(i16::MIN, i16::MAX);
        agrees_with_oracle::<65_536, i16>(-7, 12_345);
        agrees_with_oracle::<65_538, i16>(i16::MIN, i16::MAX);
        agrees_with_oracle::<16, i32>(i32::MIN, i32::MAX);
        agrees_with_oracle::<1_000, i32>(-1_000_003, 77);
        agrees_with_oracle::<256, u8>(u8::MIN, u8::MAX);
        agrees_with_oracle::<257, u8>(u8::MIN, u8::MAX);
        agrees_with_oracle::<{ 1 << 24 }, u8>(u8::MIN, u8::MAX);
        agrees_with_oracle::<{ (1 << 24) + 1 }, u8>(u8::MIN, u8::MAX);
        agrees_with_oracle::<7, u8>(3, 99);
        agrees_with_oracle::<16, u16>(u16::MIN, u16::MAX);
        agrees_with_oracle::<65_536, u16>(u16::MIN, u16::MAX);
        agrees_with_oracle::<65_537, u16>(u16::MIN, u16::MAX);
        agrees_with_oracle::<65_536, u16>(7, 12_345);
        agrees_with_oracle::<65_538, u16>(u16::MIN, u16::MAX);
        agrees_with_oracle::<16, u32>(u32::MIN, u32::MAX);
        agrees_with_oracle::<1_000, u32>(1_000_003, 4_000_000_077);
    }

    #[test]
    fn inclusive_extrema_map_to_edge_bins() {
        let bounds = HistogramBounds::<i16>::new(-8, 7).expect("valid test bounds");
        assert_eq!(classify::<4, _>(bounds, -8), HistogramClass::Bin(0));
        assert_eq!(classify::<4, _>(bounds, 7), HistogramClass::Bin(3));
        assert_eq!(classify::<4, _>(bounds, -9), HistogramClass::Below);
        assert_eq!(classify::<4, _>(bounds, 8), HistogramClass::Above);
    }

    #[test]
    fn a_single_value_range_is_valid() {
        let bounds = HistogramBounds::<i16>::new(5, 5).expect("one included value is non-empty");
        assert_eq!(bounds.span(), NonZeroU64::MIN);
        assert_eq!(classify::<4, _>(bounds, 5), HistogramClass::Bin(0));
        assert_eq!(classify::<4, _>(bounds, 4), HistogramClass::Below);
        assert_eq!(classify::<4, _>(bounds, 6), HistogramClass::Above);
    }

    #[test]
    fn complete_i16_domain_always_maps_or_counts_outside() {
        let bounds = HistogramBounds::<i16>::new(-1000, 1000).expect("valid bounds");
        for value in i16::MIN..=i16::MAX {
            let class = classify::<17, _>(bounds, value);
            match class {
                HistogramClass::Below => assert!(value < bounds.lower()),
                HistogramClass::Above => assert!(value > bounds.upper()),
                HistogramClass::Bin(index) => {
                    assert!(value >= bounds.lower());
                    assert!(value <= bounds.upper());
                    assert!(index < 17);
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_histogram_bounds_and_classify_are_const_evaluable_on_nightly() {
        const BOUNDS: HistogramBounds<i16> = match HistogramBounds::new(-8, 7) {
            Ok(bounds) => bounds,
            Err(_) => unreachable!(),
        };
        const SPAN: NonZeroU64 = BOUNDS.span();
        const SAME: bool = BOUNDS.eq_bounds(BOUNDS);
        const BIN: HistogramClass = classify::<4, i16>(BOUNDS, 0);
        const BIN_BOUNDS: Option<HistogramBounds<i16>> = linear_bin_bounds::<4, i16>(BOUNDS, 2);
        assert_eq!((SPAN.get(), SAME), (16, true));
        assert_eq!(BIN, HistogramClass::Bin(2));
        assert_eq!(BIN_BOUNDS, HistogramBounds::new(0, 3).ok());
    }
}
