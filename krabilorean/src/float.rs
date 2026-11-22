//! Finite floating-point samples: `f32` and `f64` without NaN or infinity.
//!
//! [`Finite<f32>`](Finite) and `Finite<f64>` implement [`SampleValue`], so the comparison-only
//! online families accept them: [`Minimum`](crate::online::Minimum),
//! [`Maximum`](crate::online::Maximum), [`Extrema`](crate::online::Extrema),
//! [`ThresholdCrossings`](crate::online::ThresholdCrossings), and
//! [`ThresholdRuns`](crate::online::ThresholdRuns). Because those families only compare samples,
//! they perform no floating-point arithmetic and incur no rounding error.
//!
//! Comparison-based statistics require a total order over samples, which raw IEEE floats do not
//! provide: NaN is unordered with respect to every value, so a minimum, threshold crossing, or run
//! could silently ignore or be poisoned by a NaN sample. `Finite` enforces total ordering at the
//! boundary where a raw reading becomes a sample:
//!
//! - **NaN and ±infinity are rejected.** [`Finite::new`] returns `None` for non-finite inputs, just
//!   as [`NonZero::new`](core::num::NonZero::new) does for zero. Handling non-finite readings
//!   (such as sensor faults or upstream overflow) belongs at the application boundary.
//! - **Negative zero is canonicalized to positive zero**, so `-0.0` and `0.0` have identical bits,
//!   compare equal, hash identically, and fall on the same side of every threshold.
//! - **Ordering is numeric**, implemented with `total_cmp`, which agrees with IEEE `<` on finite
//!   values once negative zero is folded to `+0.0`.
//!
//! Accumulating families ([`RunningMoments`](crate::online::RunningMoments),
//! [`SuccessiveDifference`](crate::online::SuccessiveDifference)) do not accept `Finite` floats:
//! their contracts rely on exact integer accumulation.
//!
//! ```
//! use krabilorean::float::Finite;
//! use krabilorean::online::{Extrema, ThresholdCrossings};
//! use krabilorean::policy::Checked;
//!
//! const LIMIT: Finite<f32> = match Finite::from_f32(9.81) {
//!     Some(limit) => limit,
//!     None => panic!("limit is finite"),
//! };
//! let mut extrema = Extrema::<Checked, Finite<f32>>::new();
//! let mut crossings = ThresholdCrossings::<Checked, Finite<f32>>::new(LIMIT);
//! for reading in [9.7_f32, 9.9, f32::NAN, 9.6, 10.2] {
//!     let Some(sample) = Finite::new(reading) else { continue };
//!     extrema.update(sample)?;
//!     crossings.update(sample)?;
//! }
//! let extrema = extrema.snapshot().expect("samples");
//! assert_eq!((extrema.minimum.get(), extrema.maximum.get()), (9.6, 10.2));
//! assert_eq!(crossings.snapshot().expect("samples").rising, 2);
//! # Ok::<(), krabilorean::Error>(())
//! ```
//!
//! A raw float is rejected at compile time because NaN makes its order partial:
//!
//! ```compile_fail
//! use krabilorean::online::Extrema;
//! use krabilorean::policy::Checked;
//! let _ = Extrema::<Checked, f32>::new();
//! ```
//!
//! and an accumulating family rejects `Finite` floats:
//!
//! ```compile_fail
//! use krabilorean::float::Finite;
//! use krabilorean::online::RunningMoments;
//! use krabilorean::policy::Checked;
//! let _ = RunningMoments::<Checked, Finite<f32>>::new();
//! ```

use crate::sample::SampleValue;

/// A finite floating-point sample: never NaN, never infinite, never negative zero.
///
/// Constructed with [`Finite::new`], or in constant evaluation with [`Finite::from_f32`] or
/// [`Finite::from_f64`]; read back with [`Finite::get`]. Implemented for `f32` and `f64`.
#[derive(Clone, Copy, Debug)]
pub struct Finite<T>(T);

mod sealed {
    c0nst::c0nst! {
        /// The float types [`Finite`](super::Finite) wraps.
        pub c0nst trait Float: Copy {
            /// `self` with negative zero folded to positive zero, or `None` if NaN or infinite.
            fn finite(self) -> Option<Self>;
        }
    }
}

c0nst::c0nst! {
    impl<T: sealed::Float> Finite<T> {
        /// `value` as a sample, or `None` if it is NaN or infinite. Negative zero becomes positive
        /// zero.
        #[inline]
        pub c0nst fn new(value: T) -> Option<Self>
        where
            T: [c0nst] sealed::Float,
        {
            match value.finite() {
                Some(finite) => Some(Self(finite)),
                None => None,
            }
        }

        /// The underlying finite floating-point value.
        #[inline]
        pub const fn get(self) -> T {
            self.0
        }
    }
}

macro_rules! finite {
    ($float:ty, $from:ident, $name:literal) => {
        c0nst::c0nst! {
            c0nst impl sealed::Float for $float {
                #[inline]
                fn finite(self) -> Option<Self> {
                    match Finite::<$float>::$from(self) {
                        Some(finite) => Some(finite.get()),
                        None => None,
                    }
                }
            }
        }

        impl Finite<$float> {
            /// The most negative finite value.
            pub const MIN: Self = Self(<$float>::MIN);
            /// The largest finite value.
            pub const MAX: Self = Self(<$float>::MAX);
            /// Positive zero.
            pub const ZERO: Self = Self(0.0);

            /// [`Finite::new`] in constant evaluation: `value` as a sample, or `None` if it is
            /// NaN or infinite. Negative zero becomes positive zero.
            #[inline]
            pub const fn $from(value: $float) -> Option<Self> {
                if !value.is_finite() {
                    return None;
                }
                // `-0.0 == 0.0`, so both store positive zero.
                Some(Self(if value == 0.0 { 0.0 } else { value }))
            }
        }

        impl Default for Finite<$float> {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl From<Finite<$float>> for $float {
            #[inline]
            fn from(value: Finite<$float>) -> Self {
                value.0
            }
        }

        // Without NaN and negative zero, equal values have equal bits, so equality, order, and
        // hashing all agree.
        impl PartialEq for Finite<$float> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }

        impl Eq for Finite<$float> {}

        impl PartialOrd for Finite<$float> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Finite<$float> {
            #[inline]
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.total_cmp(&other.0)
            }
        }

        impl core::hash::Hash for Finite<$float> {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.0.to_bits().hash(state);
            }
        }

        impl core::fmt::Display for Finite<$float> {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(&self.0, formatter)
            }
        }

        impl crate::sealed::IntegerName for Finite<$float> {
            const NAME: &'static str = $name;
        }

        c0nst::c0nst! {
            c0nst impl crate::sealed::SampleValue for Finite<$float> {
                #[inline(always)]
                fn above(self, threshold: Self) -> bool {
                    self.0 > threshold.0
                }

                #[inline(always)]
                fn at_or_above(self, threshold: Self) -> bool {
                    self.0 >= threshold.0
                }
            }

            c0nst impl SampleValue for Finite<$float> {
                const MIN: Self = Self::MIN;
                const MAX: Self = Self::MAX;
                const ZERO: Self = Self::ZERO;
            }
        }
    };
}

finite!(f32, from_f32, "f32");
finite!(f64, from_f64, "f64");

#[cfg(test)]
mod tests {
    use core::hash::{Hash, Hasher};

    use super::Finite;
    use crate::online::{Extrema, Maximum, Minimum, ThresholdCrossings, ThresholdRuns};
    use crate::policy::Checked;
    use crate::sample::SampleValue;
    use crate::sealed::{IntegerName, SampleValue as SealedSampleValue};

    #[derive(Default)]
    struct WordHasher(u64);

    impl Hasher for WordHasher {
        fn finish(&self) -> u64 {
            self.0
        }

        fn write(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                self.0 = self
                    .0
                    .wrapping_mul(0x100_0000_01b3)
                    .wrapping_add(u64::from(byte));
            }
        }
    }

    fn hash_of<T: Hash>(value: T) -> u64 {
        let mut hasher = WordHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn construction_rejects_nan_and_infinity_and_folds_negative_zero() {
        assert!(Finite::<f32>::new(f32::NAN).is_none());
        assert!(Finite::<f32>::new(f32::INFINITY).is_none());
        assert!(Finite::<f32>::new(f32::NEG_INFINITY).is_none());
        assert!(Finite::<f32>::from_f32(f32::NAN).is_none());
        assert!(Finite::<f64>::new(f64::NAN).is_none());
        assert!(Finite::<f64>::new(-f64::NAN).is_none());
        assert!(Finite::<f64>::new(f64::INFINITY).is_none());
        assert!(Finite::<f64>::new(f64::NEG_INFINITY).is_none());
        assert!(Finite::<f64>::from_f64(f64::NAN).is_none());

        let negative32 = Finite::<f32>::new(-0.0).expect("finite");
        let positive32 = Finite::<f32>::default();
        assert_eq!(negative32, positive32);
        assert_eq!(negative32.get().to_bits(), 0.0_f32.to_bits());
        assert_eq!(f32::from(negative32).to_bits(), 0.0_f32.to_bits());
        assert_eq!(negative32.cmp(&positive32), core::cmp::Ordering::Equal);
        assert_eq!(
            negative32.partial_cmp(&positive32),
            Some(core::cmp::Ordering::Equal)
        );
        assert_eq!(hash_of(negative32), hash_of(positive32));
        assert_eq!(std::format!("{negative32}"), "0");

        let negative64 = Finite::<f64>::new(-0.0).expect("finite");
        let positive64 = Finite::<f64>::default();
        assert_eq!(negative64, positive64);
        assert_eq!(negative64.get().to_bits(), 0.0_f64.to_bits());
        assert_eq!(f64::from(negative64).to_bits(), 0.0_f64.to_bits());
        assert_eq!(negative64.cmp(&positive64), core::cmp::Ordering::Equal);
        assert_eq!(
            negative64.partial_cmp(&positive64),
            Some(core::cmp::Ordering::Equal)
        );
        assert_eq!(hash_of(negative64), hash_of(positive64));
        assert_eq!(std::format!("{negative64}"), "0");

        assert_eq!(
            Finite::<f32>::new(f32::MIN),
            Some(<Finite<f32> as SampleValue>::MIN)
        );
        assert_eq!(
            Finite::<f32>::new(f32::MAX),
            Some(<Finite<f32> as SampleValue>::MAX)
        );
        assert_eq!(<Finite<f32> as SampleValue>::ZERO.get(), 0.0);
        assert_eq!(
            Finite::<f64>::new(f64::MIN),
            Some(<Finite<f64> as SampleValue>::MIN)
        );
        assert_eq!(
            Finite::<f64>::new(f64::MAX),
            Some(<Finite<f64> as SampleValue>::MAX)
        );
        assert_eq!(<Finite<f64> as SampleValue>::ZERO.get(), 0.0);
        assert_eq!(<Finite<f32> as IntegerName>::NAME, "f32");
        assert_eq!(<Finite<f64> as IntegerName>::NAME, "f64");

        assert!(Finite::<f32>::MAX.above(Finite::<f32>::MIN));
        assert!(!Finite::<f32>::MIN.above(Finite::<f32>::MIN));
        assert!(Finite::<f32>::MIN.at_or_above(Finite::<f32>::MIN));
        assert!(!Finite::<f32>::MIN.at_or_above(Finite::<f32>::MAX));

        assert!(Finite::<f64>::MAX.above(Finite::<f64>::MIN));
        assert!(!Finite::<f64>::MIN.above(Finite::<f64>::MIN));
        assert!(Finite::<f64>::MIN.at_or_above(Finite::<f64>::MIN));
        assert!(!Finite::<f64>::MIN.at_or_above(Finite::<f64>::MAX));

        let subnormal = f32::from_bits(1);
        assert_eq!(
            Finite::<f32>::new(subnormal).map(Finite::get),
            Some(subnormal)
        );
    }

    /// Deterministic finite `f32` values from across the bit space, with NaN and infinity skipped.
    fn scattered(count: usize) -> std::vec::Vec<f32> {
        let mut state = 0x2545_f491_u32;
        let mut values = std::vec::Vec::new();
        while values.len() < count {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = f32::from_bits(state);
            if value.is_finite() {
                values.push(value);
            }
        }
        values
    }

    #[test]
    fn order_is_the_numeric_order() {
        let values = scattered(400);
        for &a in &values {
            for &b in values.iter().take(40) {
                let (x, y) = (
                    Finite::new(a).expect("finite"),
                    Finite::new(b).expect("finite"),
                );
                // IEEE comparison of the raw floats, with -0.0 == 0.0.
                assert_eq!(x.cmp(&y), a.partial_cmp(&b).expect("no NaN"), "{a} vs {b}");
                assert_eq!(x == y, a == b, "{a} == {b}");
            }
        }
    }

    /// Raw float sequences with the threshold each one is checked against.
    fn cases() -> std::vec::Vec<(&'static str, std::vec::Vec<f32>, f32)> {
        let next_up = |x: f32| f32::from_bits(x.to_bits() + 1);
        std::vec![
            (
                "signed zeros at a zero threshold",
                std::vec![-0.0, 0.0, -0.0, -1e-30, 0.0],
                0.0,
            ),
            (
                "subnormals around zero",
                std::vec![
                    f32::from_bits(1),
                    -f32::from_bits(1),
                    f32::from_bits(0x007f_ffff),
                    0.0,
                ],
                0.0,
            ),
            (
                "finite extremes",
                std::vec![f32::MIN, f32::MAX, f32::MIN, f32::MAX],
                0.0,
            ),
            (
                "exact threshold ties",
                std::vec![9.81, 9.81, 9.8, 9.81, 9.82, 9.81],
                9.81,
            ),
            (
                "neighbouring values",
                std::vec![1.0, next_up(1.0), 1.0, next_up(1.0)],
                next_up(1.0),
            ),
            (
                "alternating around the threshold",
                (0..21)
                    .map(|i| if i % 2 == 0 { 2.5 } else { -2.5 })
                    .collect(),
                0.0,
            ),
            ("constant below", std::vec![-3.0; 9], 1.0),
            ("constant at", std::vec![1.0; 9], 1.0),
            (
                "impulse",
                (0..17).map(|i| if i == 8 { 1e30 } else { 0.5 }).collect(),
                1.0,
            ),
            (
                "regime change",
                (0..20).map(|i| if i < 10 { -7.25 } else { 7.25 }).collect(),
                0.0,
            ),
            ("scattered", scattered(257), 0.0),
        ]
    }

    /// Crossing and run counts from an independent state machine over raw floats: crossings use
    /// `x >= threshold` (inclusive high), runs use `x > threshold`.
    fn oracle_counts(values: &[f32], threshold: f32) -> ((u32, u32, u32), (u32, u32, u32)) {
        let (mut rising, mut falling) = (0, 0);
        let mut previous_high = None;
        for &x in values {
            let high = x >= threshold;
            match previous_high {
                Some(false) if high => rising += 1,
                Some(true) if !high => falling += 1,
                _ => {}
            }
            previous_high = Some(high);
        }
        let (mut prefix, mut current, mut longest, mut in_prefix) = (0, 0, 0, true);
        for &x in values {
            if x > threshold {
                current += 1;
                if in_prefix {
                    prefix += 1;
                }
            } else {
                current = 0;
                in_prefix = false;
            }
            longest = longest.max(current);
        }
        (
            (rising + falling, rising, falling),
            (prefix, current, longest),
        )
    }

    #[test]
    fn families_match_independent_oracles() {
        for (name, raw, threshold) in cases() {
            let samples: std::vec::Vec<Finite<f32>> = raw
                .iter()
                .map(|&x| Finite::new(x).expect("finite"))
                .collect();
            let limit = Finite::new(threshold).expect("finite");
            let mut minimum = Minimum::<Checked, Finite<f32>>::new();
            let mut maximum = Maximum::<Checked, Finite<f32>>::new();
            let mut extrema = Extrema::<Checked, Finite<f32>>::new();
            let mut crossings = ThresholdCrossings::<Checked, Finite<f32>>::new(limit);
            let mut runs = ThresholdRuns::<Checked, Finite<f32>>::new(limit);
            for &sample in &samples {
                minimum.update(sample).expect("minimum");
                maximum.update(sample).expect("maximum");
                extrema.update(sample).expect("extrema");
                crossings.update(sample).expect("crossings");
                runs.update(sample).expect("runs");
            }

            let low = raw
                .iter()
                .copied()
                .fold(f32::INFINITY, |a, x| if x < a { x } else { a });
            let high = raw
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, x| if x > a { x } else { a });
            let extrema = extrema.snapshot().expect("samples");
            // Compare as values: a -0.0 minimum is stored as 0.0.
            assert_eq!(extrema.minimum.get(), low, "{name}: minimum");
            assert_eq!(extrema.maximum.get(), high, "{name}: maximum");
            assert_eq!(
                minimum.snapshot().expect("samples").value,
                extrema.minimum,
                "{name}"
            );
            assert_eq!(
                maximum.snapshot().expect("samples").value,
                extrema.maximum,
                "{name}"
            );

            let (expected_crossings, expected_runs) = oracle_counts(&raw, threshold);
            let crossings = crossings.snapshot().expect("samples");
            assert_eq!(
                (crossings.crossings, crossings.rising, crossings.falling),
                expected_crossings,
                "{name}: crossings"
            );
            let runs = runs.snapshot().expect("samples");
            assert_eq!(
                (runs.prefix_run, runs.current_run, runs.longest_run),
                expected_runs,
                "{name}: runs"
            );
        }
    }

    #[test]
    fn merge_equals_one_pass() {
        for (name, raw, threshold) in cases() {
            let samples: std::vec::Vec<Finite<f64>> = raw
                .iter()
                .map(|&x| Finite::new(f64::from(x)).expect("finite"))
                .collect();
            let limit = Finite::new(f64::from(threshold)).expect("finite");
            for split in [0, 1, samples.len() / 2, samples.len()] {
                let (left, right) = samples.split_at(split);
                let mut whole = Extrema::<Checked, Finite<f64>>::new();
                let mut first = Extrema::<Checked, Finite<f64>>::new();
                let mut second = Extrema::<Checked, Finite<f64>>::new();
                let mut whole_crossings = ThresholdCrossings::<Checked, Finite<f64>>::new(limit);
                let mut first_crossings = ThresholdCrossings::<Checked, Finite<f64>>::new(limit);
                let mut second_crossings = ThresholdCrossings::<Checked, Finite<f64>>::new(limit);
                let mut whole_runs = ThresholdRuns::<Checked, Finite<f64>>::new(limit);
                let mut first_runs = ThresholdRuns::<Checked, Finite<f64>>::new(limit);
                let mut second_runs = ThresholdRuns::<Checked, Finite<f64>>::new(limit);
                for &sample in &samples {
                    whole.update(sample).expect("update");
                    whole_crossings.update(sample).expect("update");
                    whole_runs.update(sample).expect("update");
                }
                for &sample in left {
                    first.update(sample).expect("update");
                    first_crossings.update(sample).expect("update");
                    first_runs.update(sample).expect("update");
                }
                for &sample in right {
                    second.update(sample).expect("update");
                    second_crossings.update(sample).expect("update");
                    second_runs.update(sample).expect("update");
                }
                first.merge(&second).expect("merge");
                first_crossings.merge(&second_crossings).expect("merge");
                first_runs.merge(&second_runs).expect("merge");
                assert_eq!(first.snapshot(), whole.snapshot(), "{name} at {split}");
                assert_eq!(
                    first_crossings.snapshot(),
                    whole_crossings.snapshot(),
                    "{name} at {split}"
                );
                assert_eq!(
                    first_runs.snapshot(),
                    whole_runs.snapshot(),
                    "{name} at {split}"
                );
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_float_operations_are_const_evaluable_on_nightly() {
        const SAMPLE: Option<Finite<f32>> = Finite::new(-0.0_f32);
        const THRESHOLD: Finite<f32> = Finite::from_f32(1.5).unwrap();
        const ABOVE: bool = <Finite<f32> as SealedSampleValue>::above(THRESHOLD, SAMPLE.unwrap());
        const AT_OR_ABOVE: bool =
            <Finite<f32> as SealedSampleValue>::at_or_above(SAMPLE.unwrap(), Finite::<f32>::ZERO);
        assert_eq!(SAMPLE.unwrap().get().to_bits(), 0.0_f32.to_bits());
        assert_eq!((ABOVE, AT_OR_ABOVE), (true, true));
    }
}
