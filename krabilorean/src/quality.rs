//! Quality annotations for results that may saturate or approximate.

/// Numerical quality carried by a result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Quality {
    /// The selected arithmetic completed without saturation or approximation.
    Complete,
    /// At least one selected arithmetic operation saturated.
    Saturated,
    /// The value has documented approximation or quantization semantics.
    Approximate,
    /// The value is both approximate and affected by saturation.
    ApproximateSaturated,
}

impl Quality {
    /// Returns a quality with the requested approximation and saturation flags.
    pub const fn from_flags(approximate: bool, saturated: bool) -> Self {
        match (approximate, saturated) {
            (false, false) => Self::Complete,
            (false, true) => Self::Saturated,
            (true, false) => Self::Approximate,
            (true, true) => Self::ApproximateSaturated,
        }
    }

    /// Returns whether any arithmetic operation saturated.
    pub const fn is_saturated(self) -> bool {
        matches!(self, Self::Saturated | Self::ApproximateSaturated)
    }

    /// Returns whether the result has approximation semantics.
    pub const fn is_approximate(self) -> bool {
        matches!(self, Self::Approximate | Self::ApproximateSaturated)
    }

    /// Combines independent result qualities.
    pub const fn combine(self, other: Self) -> Self {
        Self::from_flags(
            self.is_approximate() || other.is_approximate(),
            self.is_saturated() || other.is_saturated(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Quality;

    #[test]
    fn flags_and_combinations_round_trip() {
        let cases = [
            (false, false, Quality::Complete),
            (false, true, Quality::Saturated),
            (true, false, Quality::Approximate),
            (true, true, Quality::ApproximateSaturated),
        ];
        for (approximate, saturated, expected) in cases {
            let quality = Quality::from_flags(approximate, saturated);
            assert_eq!(quality, expected);
            assert_eq!(quality.is_approximate(), approximate);
            assert_eq!(quality.is_saturated(), saturated);
            assert_eq!(Quality::Complete.combine(quality), quality);
            assert_eq!(quality.combine(Quality::Complete), quality);
        }
        assert_eq!(
            Quality::Approximate.combine(Quality::Saturated),
            Quality::ApproximateSaturated
        );
        assert_eq!(
            Quality::Saturated.combine(Quality::Approximate),
            Quality::ApproximateSaturated
        );
    }
}
