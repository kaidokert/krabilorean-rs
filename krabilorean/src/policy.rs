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

/// Static policy for counter and accumulator arithmetic that may overflow.
pub trait OverflowPolicy: crate::sealed::OverflowPolicy + Copy {
    /// Whether counter/addition overflow is accepted with visible saturation quality.
    const SATURATING: bool;
    /// Adds two `u32` counter values.
    fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error>;
}

/// Reject arithmetic overflow and preserve the previous state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Checked;

impl crate::sealed::OverflowPolicy for Checked {}

impl OverflowPolicy for Checked {
    const SATURATING: bool = false;

    fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error> {
        match left.checked_add(right) {
            Some(value) => Ok(Arithmetic::complete(value)),
            None => Err(Error::Overflow),
        }
    }
}

/// Accept arithmetic overflow by saturating and marking result quality.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Saturating;

impl crate::sealed::OverflowPolicy for Saturating {}

impl OverflowPolicy for Saturating {
    const SATURATING: bool = true;

    fn add_u32(left: u32, right: u32) -> Result<Arithmetic<u32>, Error> {
        match left.checked_add(right) {
            Some(value) => Ok(Arithmetic::complete(value)),
            None => Ok(Arithmetic::saturated(left.saturating_add(right))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Arithmetic, Checked, OverflowPolicy, Saturating};
    use crate::Error;

    #[test]
    fn checked_and_saturating_u32_addition() {
        const { assert!(!<Checked as OverflowPolicy>::SATURATING) };
        const { assert!(<Saturating as OverflowPolicy>::SATURATING) };

        assert_eq!(
            Checked::add_u32(10, 20),
            Ok(Arithmetic {
                value: 30,
                saturated: false,
            })
        );
        assert_eq!(Checked::add_u32(u32::MAX, 1), Err(Error::Overflow));
        assert_eq!(
            Saturating::add_u32(10, 20),
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
    }
}
