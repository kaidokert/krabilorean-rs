//! Zero-sized arithmetic policies.

use crate::Error;

/// The result of one policy-controlled operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arithmetic<T> {
    /// The representable result.
    pub value: T,
    /// Whether the operation had to saturate.
    pub saturated: bool,
}

impl<T> Arithmetic<T> {
    const fn complete(value: T) -> Self {
        Self {
            value,
            saturated: false,
        }
    }

    const fn saturated(value: T) -> Self {
        Self {
            value,
            saturated: true,
        }
    }
}

c0nst::c0nst! {
    /// Static policy for counter and accumulator arithmetic that may overflow.
    pub c0nst trait OverflowPolicy: [c0nst] crate::sealed::OverflowPolicy + Copy {
        /// Whether counter/addition overflow is accepted with visible saturation quality.
        const SATURATING: bool;
        /// Adds two `u32` counter values.
        fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error>;
        /// Adds two `u64` accumulator values.
        fn add_u64(left: u64, right: u64) -> Result<Arithmetic<u64>, Error>;
        /// Adds two `i64` accumulator values.
        fn add_i64(left: i64, right: i64) -> Result<Arithmetic<i64>, Error>;
        /// Adds two `u128` accumulator values.
        fn add_u128(left: u128, right: u128) -> Result<Arithmetic<u128>, Error>;
        /// Adds two `i128` accumulator values.
        fn add_i128(left: i128, right: i128) -> Result<Arithmetic<i128>, Error>;
    }

    /// An integer accumulator whose addition an [`OverflowPolicy`] controls: `u32`, `u64`, `i64`,
    /// `u128`, and `i128`.
    ///
    /// A family that accumulates in a type chosen by its sample word, such as
    /// [`MomentWord::Sum`](crate::sample::MomentWord::Sum), adds through this trait without
    /// specializing per accumulator width. The trait is sealed.
    pub c0nst trait Accumulator: [c0nst] crate::sealed::Accumulator + Copy {
        /// Adds two values under the overflow policy `O`.
        fn add<O: [c0nst] OverflowPolicy>(left: Self, right: Self) -> Result<Arithmetic<Self>, Error>;
    }
}

macro_rules! accumulator {
    ($($type:ty => $add:ident),* $(,)?) => {$(
        c0nst::c0nst! {
            c0nst impl crate::sealed::Accumulator for $type {}

            c0nst impl Accumulator for $type {
                #[inline(always)]
                fn add<O: [c0nst] OverflowPolicy>(left: Self, right: Self) -> Result<Arithmetic<Self>, Error> {
                    O::$add(left, right)
                }
            }
        }
    )*};
}

accumulator!(
    u32 => add_u32,
    u64 => add_u64,
    i64 => add_i64,
    u128 => add_u128,
    i128 => add_i128,
);

/// Reject arithmetic overflow and preserve the previous state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Checked;

c0nst::c0nst! {
    c0nst impl crate::sealed::OverflowPolicy for Checked {}

    c0nst impl OverflowPolicy for Checked {
        const SATURATING: bool = false;

        fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

        fn add_u64(left: u64, right: u64) -> Result<Arithmetic<u64>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

        fn add_i64(left: i64, right: i64) -> Result<Arithmetic<i64>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

        fn add_u128(left: u128, right: u128) -> Result<Arithmetic<u128>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

        fn add_i128(left: i128, right: i128) -> Result<Arithmetic<i128>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }
    }
}

/// Accept arithmetic overflow by saturating and marking result quality.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Saturating;

c0nst::c0nst! {
    c0nst impl crate::sealed::OverflowPolicy for Saturating {}

    c0nst impl OverflowPolicy for Saturating {
        const SATURATING: bool = true;

        fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

        fn add_u64(left: u64, right: u64) -> Result<Arithmetic<u64>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

        fn add_i64(left: i64, right: i64) -> Result<Arithmetic<i64>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

        fn add_u128(left: u128, right: u128) -> Result<Arithmetic<u128>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

        fn add_i128(left: i128, right: i128) -> Result<Arithmetic<i128>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Accumulator, Arithmetic, Checked, OverflowPolicy, Saturating};
    use crate::Error;

    #[test]
    fn checked_and_saturating_addition() {
        const { assert!(!<Checked as OverflowPolicy>::SATURATING) };
        const { assert!(<Saturating as OverflowPolicy>::SATURATING) };

        assert_eq!(
            <u32 as Accumulator>::add::<Checked>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u32(u32::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            <u32 as Accumulator>::add::<Saturating>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_u32(u32::MAX, 1),
            Ok(Arithmetic {
                value: u32::MAX,
                saturated: true,
            })
        );

        assert_eq!(
            <u64 as Accumulator>::add::<Checked>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u64(u64::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            <u64 as Accumulator>::add::<Saturating>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_u64(u64::MAX, 1),
            Ok(Arithmetic {
                value: u64::MAX,
                saturated: true,
            })
        );

        assert_eq!(
            <i64 as Accumulator>::add::<Checked>(-10, 20),
            Ok(Arithmetic {
                value: 10,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_i64(i64::MAX, 1), Err(Error::Overflow));
        assert_eq!(Checked::add_i64(i64::MIN, -1), Err(Error::Overflow));
        assert_eq!(
            <i64 as Accumulator>::add::<Saturating>(-10, 20),
            Ok(Arithmetic {
                value: 10,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_i64(i64::MAX, 1),
            Ok(Arithmetic {
                value: i64::MAX,
                saturated: true,
            })
        );
        assert_eq!(
            Saturating::add_i64(i64::MIN, -1),
            Ok(Arithmetic {
                value: i64::MIN,
                saturated: true,
            })
        );

        assert_eq!(
            <u128 as Accumulator>::add::<Checked>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u128(u128::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            <u128 as Accumulator>::add::<Saturating>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_u128(u128::MAX, 1),
            Ok(Arithmetic {
                value: u128::MAX,
                saturated: true,
            })
        );

        assert_eq!(
            <i128 as Accumulator>::add::<Checked>(-10, 20),
            Ok(Arithmetic {
                value: 10,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_i128(i128::MAX, 1), Err(Error::Overflow));
        assert_eq!(Checked::add_i128(i128::MIN, -1), Err(Error::Overflow));
        assert_eq!(
            <i128 as Accumulator>::add::<Saturating>(-10, 20),
            Ok(Arithmetic {
                value: 10,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_i128(i128::MAX, 1),
            Ok(Arithmetic {
                value: i128::MAX,
                saturated: true,
            })
        );
        assert_eq!(
            Saturating::add_i128(i128::MIN, -1),
            Ok(Arithmetic {
                value: i128::MIN,
                saturated: true,
            })
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_overflow_policy_and_accumulator_are_const_evaluable_on_nightly() {
        const CHECKED_U32: Result<Arithmetic<u32>, Error> =
            <u32 as Accumulator>::add::<Checked>(20, 22);
        const SATURATED_I64: Result<Arithmetic<i64>, Error> =
            <i64 as Accumulator>::add::<Saturating>(i64::MAX, 1);
        assert_eq!(
            CHECKED_U32,
            Ok(Arithmetic {
                value: 42,
                saturated: false,
            })
        );
        assert_eq!(
            SATURATED_I64,
            Ok(Arithmetic {
                value: i64::MAX,
                saturated: true,
            })
        );
    }
}
