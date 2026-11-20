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

    /// The name of an integer type as it appears in semantic ids: `i8` to `i128` and `u8` to
    /// `u128`. A sample word supplies its own name and its accumulator types' names this way, so
    /// a family composes its id without naming a width.
    pub trait IntegerName {
        /// The type's name, such as `i16` or `u128`.
        const NAME: &'static str;
    }

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
}

#[cfg(test)]
mod tests {
    #[test]
    fn unit_tests_may_use_std() {
        let values = std::vec![1_u8, 2, 3];
        assert_eq!(values.len(), 3);
    }
}
