//! Explicit integer representations.

use core::num::{NonZeroU32, NonZeroU64};

/// An exact signed ratio. No implicit division or rounding is performed.
///
/// The numerator is `i64` by default; a statistic that needs more headroom uses a wider integer
/// type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedRatio<N = i64> {
    /// Ratio numerator.
    pub numerator: N,
    /// Strictly positive ratio denominator.
    pub denominator: NonZeroU32,
}

const _: () =
    assert!(core::mem::size_of::<Option<SignedRatio>>() == core::mem::size_of::<SignedRatio>());

/// An exact unsigned ratio. No implicit division or rounding is performed.
///
/// The numerator is `u64` by default; a statistic that needs more headroom uses a wider integer
/// type, such as [`UnsignedWideRatio`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsignedRatio<N = u64> {
    /// Ratio numerator.
    pub numerator: N,
    /// Strictly positive ratio denominator.
    pub denominator: NonZeroU64,
}

const _: () =
    assert!(core::mem::size_of::<Option<UnsignedRatio>>() == core::mem::size_of::<UnsignedRatio>());

/// An exact unsigned ratio with a `u128` numerator, for complete cumulative variance headroom.
pub type UnsignedWideRatio = UnsignedRatio<u128>;
