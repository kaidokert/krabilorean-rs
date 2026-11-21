//! Explicit integer representations.

/// An exact unsigned ratio. No implicit division or rounding is performed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsignedRatio<N = u64> {
    /// Ratio numerator.
    pub numerator: N,
    /// Strictly positive ratio denominator.
    pub denominator: u64,
}
