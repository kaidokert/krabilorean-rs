//! Explicit integer representations and rounding behavior.

use core::num::{NonZeroU32, NonZeroU64};

use crate::Error;
use crate::sealed::RoundingPolicy as _;

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

c0nst::c0nst! {
    /// Static policy for division by a power of two.
    pub c0nst trait RoundingPolicy: [c0nst] crate::sealed::RoundingPolicy + Copy {
        /// Divides by `2^shift` according to the selected rounding rule.
        ///
        /// Returns [`Error::Overflow`] when `2^shift` is not a positive `i64`, that is for
        /// `shift >= 63`.
        fn divide_power_of_two(value: i64, shift: u32) -> Result<i64, Error>;
        /// Divides a wide value by `2^shift` according to the selected rounding rule.
        ///
        /// Returns [`Error::Overflow`] when `2^shift` is not a positive `i128`, that is for
        /// `shift >= 127`.
        fn divide_power_of_two_i128(value: i128, shift: u32) -> Result<i128, Error>;
    }
}

/// Round to the nearest integer, resolving exact ties away from zero.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NearestTiesAway;

c0nst::c0nst! {
    c0nst impl crate::sealed::RoundingPolicy for NearestTiesAway {
        #[inline(always)]
        fn divide_power_of_two_checked_shift(
            value: i64,
            shift: u32,
            _: crate::sealed::Internal,
        ) -> i64 {
            if shift == 0 {
                return value;
            }
            // The floor quotient and non-negative remainder come from an arithmetic shift and a
            // mask, so no target links a software division routine. `fraction` is
            // `value - floor * 2^shift`, in `0..2^shift`.
            let floor = value >> shift;
            let fraction = (value as u64) & (u64::MAX >> (u64::BITS - shift));
            let half = 1_u64 << (shift - 1);
            // Above half rounds up. An exact half rounds up for a non-negative value and stays at
            // the floor for a negative one, which is away from zero in both cases.
            let away = fraction > half || (fraction == half && value >= 0);
            // `shift >= 1` implies `floor <= i64::MAX >> 1`, so adding one cannot overflow.
            floor + (away as i64)
        }

        #[inline(always)]
        fn divide_power_of_two_i128_checked_shift(
            value: i128,
            shift: u32,
            _: crate::sealed::Internal,
        ) -> i128 {
            if shift == 0 {
                return value;
            }
            // Computing the floor quotient and non-negative remainder with an arithmetic shift
            // and mask avoids `i128` software division and remainder routines. `fraction` is
            // `value - floor * 2^shift`, in `0..2^shift`.
            let floor = value >> shift;
            let fraction = (value as u128) & (u128::MAX >> (u128::BITS - shift));
            let half = 1_u128 << (shift - 1);
            let away = fraction > half || (fraction == half && value >= 0);
            // `shift >= 1` implies `floor <= i128::MAX >> 1`, so adding one cannot overflow.
            floor + (away as i128)
        }
    }

    c0nst impl RoundingPolicy for NearestTiesAway {
        fn divide_power_of_two(value: i64, shift: u32) -> Result<i64, Error> {
            if shift >= i64::BITS - 1 {
                return Err(Error::Overflow);
            }
            Ok(Self::divide_power_of_two_checked_shift(
                value,
                shift,
                crate::sealed::Internal::new(),
            ))
        }

        fn divide_power_of_two_i128(value: i128, shift: u32) -> Result<i128, Error> {
            if shift >= i128::BITS - 1 {
                return Err(Error::Overflow);
            }
            Ok(Self::divide_power_of_two_i128_checked_shift(
                value,
                shift,
                crate::sealed::Internal::new(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use core::num::{NonZeroU32, NonZeroU64};

    use super::{NearestTiesAway, RoundingPolicy, widen_count};
    use crate::Error;

    #[test]
    fn widened_counts_are_exact_at_the_extremes() {
        for count in [1, 2, u32::MAX - 1, u32::MAX] {
            let count = NonZeroU32::new(count).expect("nonzero");
            assert_eq!(widen_count(count), NonZeroU64::from(count));
        }
    }

    #[test]
    fn ties_round_away_from_zero() {
        assert_eq!(NearestTiesAway::divide_power_of_two(5, 1), Ok(3));
        assert_eq!(NearestTiesAway::divide_power_of_two(-5, 1), Ok(-3));
        assert_eq!(NearestTiesAway::divide_power_of_two(4, 1), Ok(2));
        assert_eq!(NearestTiesAway::divide_power_of_two(4, 0), Ok(4));
        assert_eq!(NearestTiesAway::divide_power_of_two_i128(4, 0), Ok(4));
    }

    /// Independent oracle: the exact quotient `value / 2^shift` rounded to nearest, ties away
    /// from zero, from the magnitude in `u128`.
    fn ties_away_oracle(value: i128, shift: u32) -> i128 {
        if shift == 0 {
            return value;
        }
        let magnitude = value.unsigned_abs();
        let whole = magnitude >> shift;
        let rest = magnitude - (whole << shift);
        let rounded = whole + u128::from((rest >> (shift - 1)) != 0);
        let rounded = i128::try_from(rounded).expect("rounded magnitude fits");
        if value < 0 { -rounded } else { rounded }
    }

    fn wide_values() -> impl Iterator<Item = i128> {
        let edges = [
            i128::MIN,
            i128::MIN + 1,
            i128::MAX,
            i128::MAX - 1,
            0,
            1,
            -1,
            2,
            -2,
            3,
            -3,
            0xFFFF,
            -0xFFFF,
            0x1_0000,
            -0x1_0000,
            0xAF_0000_0000,
            -0xAF_0000_0000,
            375_809_638_400,
            -375_809_638_400,
            1 << 64,
            -(1 << 64),
            (1 << 126) - 1,
            -(1 << 126),
        ];
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mixed = (0..500).map(move |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let high = state;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((u128::from(high) << 64) | u128::from(state)) as i128 >> (high % 128)
        });
        edges.into_iter().chain(mixed)
    }

    #[test]
    fn wide_rounding_matches_the_magnitude_oracle_at_every_shift() {
        for value in wide_values() {
            for shift in 0..127 {
                let tie = if shift == 0 { 0 } else { 1_i128 << (shift - 1) };
                let base = value >> shift << shift;
                for candidate in [
                    value,
                    base.wrapping_add(tie),
                    base.wrapping_add(tie).wrapping_sub(1),
                    base.wrapping_add(tie).wrapping_add(1),
                ] {
                    assert_eq!(
                        NearestTiesAway::divide_power_of_two_i128(candidate, shift),
                        Ok(ties_away_oracle(candidate, shift)),
                        "value {candidate:#x} shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn rounding_matches_the_magnitude_oracle_at_every_shift() {
        let narrow = wide_values().map(|value| value as i64).chain([
            i64::MIN,
            i64::MIN + 1,
            i64::MAX,
            i64::MAX - 1,
        ]);
        for value in narrow {
            for shift in 0..63 {
                let tie = if shift == 0 { 0 } else { 1_i64 << (shift - 1) };
                let base = value >> shift << shift;
                for candidate in [
                    value,
                    base.wrapping_add(tie),
                    base.wrapping_add(tie).wrapping_sub(1),
                    base.wrapping_add(tie).wrapping_add(1),
                ] {
                    assert_eq!(
                        NearestTiesAway::divide_power_of_two(candidate, shift),
                        Ok(ties_away_oracle(i128::from(candidate), shift) as i64),
                        "value {candidate:#x} shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn rounding_rejects_a_divisor_outside_type() {
        for shift in [63, 64, 100, u32::MAX] {
            for value in [0, 1, -1, i64::MIN, i64::MAX] {
                assert_eq!(
                    NearestTiesAway::divide_power_of_two(value, shift),
                    Err(Error::Overflow)
                );
            }
        }
        assert_eq!(NearestTiesAway::divide_power_of_two(i64::MIN, 62), Ok(-2));
        assert_eq!(NearestTiesAway::divide_power_of_two(i64::MAX, 62), Ok(2));

        for shift in [127, 128, 200, u32::MAX] {
            for value in [0, 1, -1, i128::MIN, i128::MAX] {
                assert_eq!(
                    NearestTiesAway::divide_power_of_two_i128(value, shift),
                    Err(Error::Overflow)
                );
            }
        }
        assert_eq!(NearestTiesAway::divide_power_of_two_i128(-5, 1), Ok(-3));
        assert_eq!(NearestTiesAway::divide_power_of_two_i128(5, 1), Ok(3));
        assert_eq!(NearestTiesAway::divide_power_of_two_i128(-6, 2), Ok(-2));
        assert_eq!(
            NearestTiesAway::divide_power_of_two_i128(i128::MIN, 126),
            Ok(-2)
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_rounding_is_const_evaluable_on_nightly() {
        const ROUNDED_64: Result<i64, Error> = NearestTiesAway::divide_power_of_two(-5, 1);
        const ROUNDED_128: Result<i128, Error> = NearestTiesAway::divide_power_of_two_i128(-5, 1);
        assert_eq!(ROUNDED_64, Ok(-3));
        assert_eq!(ROUNDED_128, Ok(-3));
    }
}
