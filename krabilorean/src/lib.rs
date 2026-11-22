#![cfg_attr(not(test), no_std)]
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
//! let mut extrema = Extrema::<Checked, i16>::new();
//! for sample in [4_i16, -2, 7, 3] {
//!     extrema.update(sample)?;
//! }
//!
//! let snapshot = extrema.snapshot().expect("at least one sample");
//! assert_eq!((snapshot.samples, snapshot.minimum, snapshot.maximum), (4, -2, 7));
//! # Ok::<(), krabilorean::Error>(())
//! ```

pub mod arithmetic;
pub mod error;
pub mod online;
pub mod policy;
pub mod quality;
pub mod sample;
pub mod view;

pub use error::Error;
pub use quality::Quality;

mod sealed {
    pub trait OverflowPolicy {}

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

    /// Seal of `policy::Accumulator`.
    pub trait Accumulator {}

    /// Seal of the public `SampleValue`, with the comparisons that comparison-only families need.
    /// Families read these instead of matching on a type, so a new sample type changes no family.
    pub trait SampleValue: Sized + IntegerName {
        /// `self > threshold`, compared by value in the type's own order.
        fn above(self, threshold: Self) -> bool;

        /// `self >= threshold`, compared by value in the type's own order.
        fn at_or_above(self, threshold: Self) -> bool;
    }

    /// Seal of the public `SampleWord`, with the exact operations families need from any integer
    /// word.
    pub trait SampleWord: SampleValue {
        /// Exact `|self - other|`, at most `2^BITS - 1`, widened to `u64`.
        fn absolute_difference(self, other: Self) -> u64;
    }

    /// Seal of the public `MomentWord`.
    pub trait MomentWord: SampleWord {}

    /// Exact operations on a moment sum type (`i64` for signed words, `u64` for unsigned ones)
    /// that the moment families need, given the word's square-sum type `SquareSum` and its mean
    /// type `Mean`. A word states them as bounds on its `Sum`.
    pub trait SumOps<SquareSum, Mean>: Copy {
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
    pub trait SquareSumOps: Copy {
        /// `samples * self`, or `None` when it doesn't fit `u128`. A `u64` square sum times a
        /// `u32` count is below `2^96`, so it always fits; a `u128` one is checked, because a
        /// hand-built snapshot can hold any value.
        fn scaled(self, samples: core::num::NonZeroU32) -> Option<u128>;
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
