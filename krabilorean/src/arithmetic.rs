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

/// `count` as a `NonZeroU64`: what `NonZeroU64::from` returns, but callable in a `const fn`.
pub(crate) const fn widen_count(count: NonZeroU32) -> NonZeroU64 {
    // A non-zero `u32` widens to a non-zero `u64`, so the `MIN` arm is dead and folds away.
    match NonZeroU64::new(count.get() as u64) {
        Some(widened) => widened,
        None => NonZeroU64::MIN,
    }
}

#[cfg(test)]
mod tests {
    use core::num::{NonZeroU32, NonZeroU64};

    use super::widen_count;

    #[test]
    fn widened_counts_are_exact_at_the_extremes() {
        for count in [1, 2, u32::MAX - 1, u32::MAX] {
            let count = NonZeroU32::new(count).expect("nonzero");
            assert_eq!(widen_count(count), NonZeroU64::from(count));
        }
    }
}
