//! Errors returned by configuration and numerical operations.

/// A failure that callers may need to handle differently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Static or runtime configuration values do not form a valid processor.
    InvalidConfiguration,
    /// A checked arithmetic or counter operation could not be represented.
    Overflow,
    /// A windowed operation requires at least one sample.
    EmptyInput,
    /// The supplied input exceeds a compile-time capacity.
    CapacityExceeded,
    /// A safe arithmetic plan exists, but exceeds the caller's declared precision loss.
    PrecisionBudgetExceeded,
    /// A [`crate::view::SampleView`] did not honor its declared length.
    ViewContractViolation,
}

impl Error {
    /// A short, stable, human-readable description of the variant.
    ///
    /// This is the text [`Display`](core::fmt::Display) writes. It is one fixed string per variant
    /// and does not change between patch releases.
    pub const fn description(self) -> &'static str {
        match self {
            Self::InvalidConfiguration => "invalid configuration",
            Self::Overflow => "arithmetic overflow",
            Self::EmptyInput => "empty input",
            Self::CapacityExceeded => "capacity exceeded",
            Self::PrecisionBudgetExceeded => "precision budget exceeded",
            Self::ViewContractViolation => "sample view contract violated",
        }
    }
}

/// Writes [`Error::description`], with no allocation or formatting machinery beyond `write_str`.
///
/// ```
/// assert_eq!(format!("{}", krabilorean::Error::Overflow), "arithmetic overflow");
/// ```
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn display_strings_are_stable() {
        let expected = [
            (Error::InvalidConfiguration, "invalid configuration"),
            (Error::Overflow, "arithmetic overflow"),
            (Error::EmptyInput, "empty input"),
            (Error::CapacityExceeded, "capacity exceeded"),
            (Error::PrecisionBudgetExceeded, "precision budget exceeded"),
            (
                Error::ViewContractViolation,
                "sample view contract violated",
            ),
        ];
        for (error, text) in expected {
            assert_eq!(std::format!("{error}"), text);
            assert_eq!(error.description(), text);
        }
    }
}
