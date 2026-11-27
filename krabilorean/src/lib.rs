#![cfg_attr(not(test), no_std)]
#![cfg_attr(feature = "nightly", feature(const_trait_impl))]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Bounded time-series statistics and feature extraction for embedded systems.
//!
//! # Online extrema
//!
//! ```
//! use krabilorean::online::Extrema;
//! use krabilorean::policy::Checked;
//!
//! let mut extrema = Extrema::<Checked, u32, i16>::new();
//! for sample in [4_i16, -2, 7, 3] {
//!     extrema.update(sample)?;
//! }
//!
//! let snapshot = extrema.snapshot().expect("at least one sample");
//! assert_eq!((snapshot.samples, snapshot.minimum, snapshot.maximum), (4, -2, 7));
//! # Ok::<(), krabilorean::Error>(())
//! ```

pub mod arithmetic;
pub mod distribution;
pub mod error;
pub mod fixed_point;
pub mod float;
pub mod histogram;
pub mod online;
pub mod policy;
pub mod quality;
pub mod sample;
pub mod view;

pub use error::Error;
pub use quality::Quality;

mod sealed {
    /// Proof of a call from inside the crate. Its field is private, so safe code in another crate
    /// can't build one, and so can't call a method that takes it. (`unsafe` code can conjure
    /// any zero-sized value; that is outside this guarantee.)
    #[derive(Clone, Copy)]
    pub struct Internal(());

    impl Internal {
        pub(crate) const fn new() -> Self {
            Self(())
        }
    }

    /// The name of an integer type as it appears in semantic ids: `i8` to `i128` and `u8` to
    /// `u128`. A sample word supplies its own name and its accumulator types' names this way.
    pub trait IntegerName {
        /// The type's name, such as `i16` or `u128`.
        const NAME: &'static str;
    }

    c0nst::c0nst! {
        pub c0nst trait OverflowPolicy {}

        /// Seal of `arithmetic::RoundingPolicy`, with the shift-validated power-of-two division
        /// kernels. Callers must hold `shift < 63` for `i64` and `shift < 127` for `i128`.
        pub c0nst trait RoundingPolicy {
            /// `round(value / 2^shift)` for `shift < 63`.
            fn divide_power_of_two_checked_shift(
                value: i64,
                shift: u32,
                internal: Internal,
            ) -> i64;
            /// `round(value / 2^shift)` for `shift < 127`.
            fn divide_power_of_two_i128_checked_shift(
                value: i128,
                shift: u32,
                internal: Internal,
            ) -> i128;
        }

        /// Seal of `policy::Accumulator`.
        pub c0nst trait Accumulator {}

        /// Seal of `policy::CounterWord`.
        pub c0nst trait CounterWord {}

        /// Seal of the public `ExactBucketView`.
        pub c0nst trait ExactBucketView {}

        /// Seal of `online::MergeableOnline`.
        pub c0nst trait MergeableOnline {}

        /// Seal of the public `SampleValue`, with the comparisons that comparison-only families need.
        /// Families read these instead of matching on a type, so a new sample type changes no family.
        pub c0nst trait SampleValue: Sized + IntegerName {
            /// `self > threshold`, compared by value in the type's own order.
            fn above(self, threshold: Self) -> bool;

            /// `self >= threshold`, compared by value in the type's own order.
            fn at_or_above(self, threshold: Self) -> bool;
        }

        /// Seal of the public `SampleWord`, with the exact operations families need from any integer
        /// word.
        pub c0nst trait SampleWord: [c0nst] SampleValue {
            /// Exact `|self - other|`, at most `2^BITS - 1`, widened to `u64`.
            fn absolute_difference(self, other: Self) -> u64;
        }

        /// Seal of the public `MomentWord`.
        pub c0nst trait MomentWord: [c0nst] SampleWord {}

        /// Crate-internal kernels on a histogram word. They have preconditions the type system does
        /// not carry (`self >= lower` for `offset_from`, `lower <= upper` for `span_values`).
        /// Sealing stops other crates implementing them, and the [`Internal`] argument stops safe
        /// code in other crates calling them.
        pub c0nst trait HistogramWord: [c0nst] SampleWord {
            /// `self - lower` for `self >= lower`, exact in `u64`. Callers must hold
            /// `self >= lower`.
            fn offset_from(self, lower: Self, internal: Internal) -> u64;
            /// Number of distinct words in `lower..=upper`, in `1..=2^BITS`. Callers must hold
            /// `lower <= upper`.
            fn span_values(lower: Self, upper: Self, internal: Internal) -> core::num::NonZeroU64;
        }

        /// Exact operations on a moment sum type (`i64` for signed words, `u64` for unsigned ones)
        /// that the moment families need, given the word's square-sum type `SquareSum` and its mean
        /// type `Mean`. A word states them as bounds on its `Sum`.
        pub c0nst trait SumOps<SquareSum, Mean>: Copy {
            /// `self * self` for a sample widened into the sum type. Exact only under that
            /// precondition: a word's square is at most `MIN^2 <= 2^62` signed or `MAX^2 < 2^64`
            /// unsigned, and fits `SquareSum`. A larger sum, such as a snapshot's `sum`, would
            /// overflow, so the [`Internal`] argument stops safe code in other crates calling it.
            fn square(self, internal: Internal) -> SquareSum;
            /// `|self|` in `u128`.
            fn magnitude(self) -> u128;
            /// The exact mean `self / samples`.
            fn mean(self, samples: core::num::NonZeroU32) -> Mean;
        }

        /// Exact operations on a moment square-sum type (`u64` or `u128`).
        pub c0nst trait SquareSumOps: Copy {
            /// `samples * self`, or `None` when it doesn't fit `u128`. A `u64` square sum times a
            /// `u32` count is below `2^96`, so it always fits; a `u128` one is checked, because a
            /// hand-built snapshot can hold any value.
            fn scaled(self, samples: core::num::NonZeroU32) -> Option<u128>;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unit_tests_may_use_std() {
        let values = std::vec![1_u8, 2, 3];
        assert_eq!(values.len(), 3);
    }
}
