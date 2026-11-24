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
        /// Adds two `u8` counter values.
        fn add_u8(left: u8, right: u8) -> Result<Arithmetic<u8>, Error>;
        /// Adds two `u16` counter values.
        fn add_u16(left: u16, right: u16) -> Result<Arithmetic<u16>, Error>;
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

    /// An integer accumulator whose addition an [`OverflowPolicy`] controls: `u8`, `u16`, `u32`,
    /// `u64`, `i64`, `u128`, and `i128`.
    ///
    /// A family that accumulates in a type chosen by its sample or counter word, such as
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
    u8 => add_u8,
    u16 => add_u16,
    u32 => add_u32,
    u64 => add_u64,
    i64 => add_i64,
    u128 => add_u128,
    i128 => add_i128,
);

c0nst::c0nst! {
    /// The unsigned integer word in which a counting family stores its counters: `u8`, `u16`,
    /// `u32`, or `u64`.
    ///
    /// The counter word determines both storage layout (for example, one byte per bin in a
    /// `LinearHistogram` with `u8` counters) and the count ceiling before the family's
    /// [`OverflowPolicy`] rejects ([`Checked`]) or saturates ([`Saturating`]) an update. The
    /// trait is sealed.
    pub c0nst trait CounterWord:
        [c0nst] Accumulator
        + [c0nst] crate::sealed::CounterWord
        + Ord
        + core::fmt::Debug
    {
        /// The word's identifier in semantic metadata (`"u8"`, `"u16"`, `"u32"`, or `"u64"`).
        const NAME: &'static str;
        /// Zero samples.
        const ZERO: Self;
        /// One sample.
        const ONE: Self;
        /// Largest representable count in this word.
        const MAX: Self;
        /// Matching non-zero integer type (`NonZeroU8`, `NonZeroU16`, `NonZeroU32`, or
        /// `NonZeroU64`).
        type NonZero: Copy + Ord + core::fmt::Debug;
        /// Wraps `self` when non-zero.
        fn non_zero(self) -> Option<Self::NonZero>;
        /// Unwraps a non-zero count to its primitive value.
        fn get(count: Self::NonZero) -> Self;
        /// Saturating addition for derived count bounds (sample accumulation always goes through
        /// [`OverflowPolicy`]).
        fn saturating_add(self, other: Self) -> Self;
        /// Checked subtraction returning `None` when `other > self`.
        fn checked_sub(self, other: Self) -> Option<Self>;
        /// Exact widening conversion to `u64`.
        fn to_u64(self) -> u64;
        /// Narrowing conversion from `u64`, returning `None` when `value > Self::MAX`.
        fn from_u64(value: u64) -> Option<Self>;
        /// Zero-based order-statistic rank `floor(quantile * (samples - 1))`, which is at most
        /// `samples - 1` and always fits `Self`.
        fn rank(quantile: crate::distribution::Quantile, samples: Self::NonZero) -> Self;
    }
}

macro_rules! counter_word {
    ($($type:ident, $non_zero:ident, $name:literal, |$quantile:ident, $samples:ident| $rank:expr);* $(;)?) => {$(
        c0nst::c0nst! {
            c0nst impl crate::sealed::CounterWord for $type {}

            c0nst impl CounterWord for $type {
                const NAME: &'static str = $name;
                const ZERO: Self = 0;
                const ONE: Self = 1;
                const MAX: Self = $type::MAX;
                type NonZero = core::num::$non_zero;

                #[inline(always)]
                fn non_zero(self) -> Option<Self::NonZero> {
                    core::num::$non_zero::new(self)
                }

                #[inline(always)]
                fn get(count: Self::NonZero) -> Self {
                    count.get()
                }

                #[inline(always)]
                fn saturating_add(self, other: Self) -> Self {
                    $type::saturating_add(self, other)
                }

                #[inline(always)]
                fn checked_sub(self, other: Self) -> Option<Self> {
                    $type::checked_sub(self, other)
                }

                #[inline(always)]
                fn to_u64(self) -> u64 {
                    self as u64
                }

                #[inline(always)]
                fn from_u64(value: u64) -> Option<Self> {
                    if value <= $type::MAX as u64 {
                        Some(value as $type)
                    } else {
                        None
                    }
                }

                #[inline(always)]
                fn rank($quantile: crate::distribution::Quantile, $samples: Self::NonZero) -> Self {
                    $rank
                }
            }
        }
    )*};
}

/// Reject arithmetic overflow and preserve the previous state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Checked;

c0nst::c0nst! {
    c0nst impl crate::sealed::OverflowPolicy for Checked {}

    c0nst impl OverflowPolicy for Checked {
        const SATURATING: bool = false;

        fn add_u8(left: u8, right: u8) -> Result<Arithmetic<u8>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

        fn add_u16(left: u16, right: u16) -> Result<Arithmetic<u16>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Err(Error::Overflow),
            }
        }

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

        fn add_u8(left: u8, right: u8) -> Result<Arithmetic<u8>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

        fn add_u16(left: u16, right: u16) -> Result<Arithmetic<u16>, Error> {
            match left.checked_add(right) {
                Some(value) => Ok(Arithmetic::complete(value)),
                None => Ok(Arithmetic::saturated(left.saturating_add(right))),
            }
        }

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

// Narrow counter words widen `samples - 1` into `u32` and narrow the rank back: the rank is at
// most `samples - 1`, so it fits the word. `u64` computes the rank in `u128`.
counter_word!(
    u8, NonZeroU8, "u8", |quantile, samples| quantile.rank_less_one((samples.get() - 1) as u32) as u8;
    u16, NonZeroU16, "u16", |quantile, samples| quantile.rank_less_one((samples.get() - 1) as u32) as u16;
    u32, NonZeroU32, "u32", |quantile, samples| quantile.rank(samples);
    u64, NonZeroU64, "u64", |quantile, samples| quantile.rank_u64(samples);
);

#[cfg(test)]
mod tests {
    use super::{Accumulator, Arithmetic, Checked, CounterWord, OverflowPolicy, Saturating};
    use crate::Error;
    use crate::distribution::Quantile;

    #[test]
    fn checked_and_saturating_addition() {
        const { assert!(!<Checked as OverflowPolicy>::SATURATING) };
        const { assert!(<Saturating as OverflowPolicy>::SATURATING) };

        assert_eq!(
            <u8 as Accumulator>::add::<Checked>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u8(u8::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            <u8 as Accumulator>::add::<Saturating>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_u8(u8::MAX, 1),
            Ok(Arithmetic {
                value: u8::MAX,
                saturated: true,
            })
        );

        assert_eq!(
            <u16 as Accumulator>::add::<Checked>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u16(u16::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            <u16 as Accumulator>::add::<Saturating>(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(
            Saturating::add_u16(u16::MAX, 1),
            Ok(Arithmetic {
                value: u16::MAX,
                saturated: true,
            })
        );

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

    fn wide_rank(numerator: u32, denominator: u32, samples: u64) -> u128 {
        u128::from(numerator) * u128::from(samples - 1) / u128::from(denominator)
    }

    #[test]
    fn ranks_match_wide_arithmetic_at_every_counter_word() {
        let quantiles = [
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (u32::MAX - 1, u32::MAX),
            (1, 1),
        ];
        for (numerator, denominator) in quantiles {
            let quantile = Quantile::new(numerator, denominator).expect("valid");
            for samples in [1_u64, 2, 3, 255, 256, 65_535, u64::from(u32::MAX), u64::MAX] {
                let expected = wide_rank(numerator, denominator, samples);
                let rank = u64::rank(
                    quantile,
                    core::num::NonZeroU64::new(samples).expect("non-zero"),
                );
                assert_eq!(u128::from(rank), expected);
                if let Ok(narrow) = u32::try_from(samples) {
                    let narrow = core::num::NonZeroU32::new(narrow).expect("non-zero");
                    assert_eq!(u128::from(u32::rank(quantile, narrow)), expected);
                }
                if let Ok(narrow) = u16::try_from(samples) {
                    let narrow = core::num::NonZeroU16::new(narrow).expect("non-zero");
                    assert_eq!(u128::from(u16::rank(quantile, narrow)), expected);
                }
                if let Ok(narrow) = u8::try_from(samples) {
                    let narrow = core::num::NonZeroU8::new(narrow).expect("non-zero");
                    assert_eq!(u128::from(u8::rank(quantile, narrow)), expected);
                }
            }
        }
    }

    #[test]
    fn counter_conversions_are_exact_at_the_word_edges() {
        assert_eq!(u8::from_u64(255), Some(u8::MAX));
        assert_eq!(u8::from_u64(256), None);
        assert_eq!(u16::from_u64(65_535), Some(u16::MAX));
        assert_eq!(u16::from_u64(65_536), None);
        assert_eq!(u32::from_u64(u64::from(u32::MAX)), Some(u32::MAX));
        assert_eq!(u32::from_u64(u64::from(u32::MAX) + 1), None);
        assert_eq!(u64::from_u64(u64::MAX), Some(u64::MAX));
        assert_eq!((u8::MAX.to_u64(), u16::MAX.to_u64()), (255, 65_535));
        assert_eq!(
            (u32::MAX.to_u64(), u64::MAX.to_u64()),
            (u64::from(u32::MAX), u64::MAX)
        );
        assert_eq!(CounterWord::saturating_add(250_u8, 9), u8::MAX);
        assert_eq!(CounterWord::saturating_add(65_530_u16, 9), u16::MAX);
        assert_eq!(CounterWord::saturating_add(u32::MAX - 2, 9), u32::MAX);
        assert_eq!(CounterWord::saturating_add(u64::MAX - 2, 9), u64::MAX);
        assert_eq!(CounterWord::checked_sub(4_u8, 3), Some(1));
        assert_eq!(CounterWord::checked_sub(3_u16, 4), None);
        assert_eq!(CounterWord::checked_sub(4_u32, 3), Some(1));
        assert_eq!(CounterWord::checked_sub(4_u64, 3), Some(1));
        assert_eq!(u8::ZERO.non_zero(), None);
        assert_eq!(u16::ZERO.non_zero(), None);
        assert_eq!(u32::ZERO.non_zero(), None);
        assert_eq!(u64::ZERO.non_zero(), None);
        assert_eq!(u8::get(u8::ONE.non_zero().expect("one")), 1);
        assert_eq!(u16::get(u16::ONE.non_zero().expect("one")), 1);
        assert_eq!(u32::get(u32::ONE.non_zero().expect("one")), 1);
        assert_eq!(u64::get(u64::ONE.non_zero().expect("one")), 1);
        assert_eq!(
            [u8::NAME, u16::NAME, u32::NAME, u64::NAME],
            ["u8", "u16", "u32", "u64"]
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_overflow_policy_and_accumulator_are_const_evaluable_on_nightly() {
        const CHECKED_U32: Result<Arithmetic<u32>, Error> =
            <u32 as Accumulator>::add::<Checked>(20, 22);
        const SATURATED_I64: Result<Arithmetic<i64>, Error> =
            <i64 as Accumulator>::add::<Saturating>(i64::MAX, 1);
        const COUNTER_SUB: Option<u8> = <u8 as CounterWord>::checked_sub(200, 55);
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
        assert_eq!(COUNTER_SUB, Some(145));
    }
}
