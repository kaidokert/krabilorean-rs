//! Explicit integer representations.

use core::num::NonZeroU64;

/// An exact unsigned ratio. No implicit division or rounding is performed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsignedRatio {
    /// Ratio numerator.
    pub numerator: u64,
    /// Strictly positive ratio denominator.
    pub denominator: NonZeroU64,
}

const _: () =
    assert!(core::mem::size_of::<Option<UnsignedRatio>>() == core::mem::size_of::<UnsignedRatio>());
