//! Constant-state cumulative threshold crossings.

use core::marker::PhantomData;

use crate::policy::{Arithmetic, OverflowPolicy};
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// Cumulative threshold-crossing result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdCrossingsSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Runtime threshold; equality belongs to the high side.
    pub threshold: S,
    /// Total low-to-high and high-to-low transitions.
    pub crossings: u32,
    /// Low-to-high transitions.
    pub rising: u32,
    /// High-to-low transitions.
    pub falling: u32,
    /// Whether the first sample is on the high side, retained for exact concatenation.
    pub first_high: bool,
    /// Whether the latest sample is on the high side.
    pub current_high: bool,
    /// Saturation quality.
    pub quality: Quality,
}

/// Cumulative crossings of one runtime threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdCrossings<O, S> {
    samples: u32,
    threshold: S,
    crossings: u32,
    rising: u32,
    falling: u32,
    first_high: bool,
    previous_high: bool,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective threshold-crossing state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThresholdCrossingsDelta {
    samples: u32,
    crossings: u32,
    rising: u32,
    falling: u32,
    first_high: bool,
    previous_high: bool,
    quality: Quality,
}

impl<O, S: SampleValue> ThresholdCrossings<O, S> {
    /// Creates an empty crossing summary for one threshold.
    pub const fn new(threshold: S) -> Self {
        Self {
            samples: 0,
            threshold,
            crossings: 0,
            rising: 0,
            falling: 0,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Returns the configured threshold.
    pub const fn threshold(&self) -> S {
        self.threshold
    }
}

impl<O: OverflowPolicy, S: SampleValue> ThresholdCrossings<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<ThresholdCrossingsSnapshot<S>> {
        (self.samples != 0).then_some(ThresholdCrossingsSnapshot {
            samples: self.samples,
            threshold: self.threshold,
            crossings: self.crossings,
            rising: self.rising,
            falling: self.falling,
            first_high: self.first_high,
            current_high: self.previous_high,
            quality: self.quality,
        })
    }

    /// Clears counts while retaining the configured threshold.
    pub fn reset(&mut self) {
        self.samples = 0;
        self.crossings = 0;
        self.rising = 0;
        self.falling = 0;
        self.first_high = false;
        self.previous_high = false;
        self.quality = Quality::Complete;
    }

    /// Exactly appends newer crossings with the same threshold and includes the boundary.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    fn prepare_word(&self, sample: S) -> Result<ThresholdCrossingsDelta, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        let current_high = crate::sealed::SampleValue::at_or_above(sample, self.threshold);
        let (crossings, rising, falling) =
            if self.samples == 0 || current_high == self.previous_high {
                (
                    Arithmetic {
                        value: self.crossings,
                        saturated: false,
                    },
                    Arithmetic {
                        value: self.rising,
                        saturated: false,
                    },
                    Arithmetic {
                        value: self.falling,
                        saturated: false,
                    },
                )
            } else if current_high {
                (
                    O::add_u32(self.crossings, 1)?,
                    O::add_u32(self.rising, 1)?,
                    Arithmetic {
                        value: self.falling,
                        saturated: false,
                    },
                )
            } else {
                (
                    O::add_u32(self.crossings, 1)?,
                    Arithmetic {
                        value: self.rising,
                        saturated: false,
                    },
                    O::add_u32(self.falling, 1)?,
                )
            };
        Ok(ThresholdCrossingsDelta {
            samples: samples.value,
            crossings: crossings.value,
            rising: rising.value,
            falling: falling.value,
            first_high: if self.samples == 0 {
                current_high
            } else {
                self.first_high
            },
            previous_high: current_high,
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || samples.saturated
                    || crossings.saturated
                    || rising.saturated
                    || falling.saturated,
            ),
        })
    }

    fn commit_word(&mut self, delta: ThresholdCrossingsDelta) {
        self.samples = delta.samples;
        self.crossings = delta.crossings;
        self.rising = delta.rising;
        self.falling = delta.falling;
        self.first_high = delta.first_high;
        self.previous_high = delta.previous_high;
        self.quality = delta.quality;
    }

    fn prepare_merge_word(&self, newer: &Self) -> Result<ThresholdCrossingsDelta, Error> {
        if self.threshold != newer.threshold {
            return Err(Error::InvalidConfiguration);
        }
        if newer.samples == 0 {
            return Ok(ThresholdCrossingsDelta {
                samples: self.samples,
                crossings: self.crossings,
                rising: self.rising,
                falling: self.falling,
                first_high: self.first_high,
                previous_high: self.previous_high,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(ThresholdCrossingsDelta {
                samples: newer.samples,
                crossings: newer.crossings,
                rising: newer.rising,
                falling: newer.falling,
                first_high: newer.first_high,
                previous_high: newer.previous_high,
                quality: newer.quality,
            });
        }

        let samples = O::add_u32(self.samples, newer.samples)?;
        let joined_crossings = O::add_u32(self.crossings, newer.crossings)?;
        let joined_rising = O::add_u32(self.rising, newer.rising)?;
        let joined_falling = O::add_u32(self.falling, newer.falling)?;
        let is_boundary = self.previous_high != newer.first_high;
        let crossings = if is_boundary {
            O::add_u32(joined_crossings.value, 1)?
        } else {
            joined_crossings
        };
        let rising = if is_boundary && newer.first_high {
            O::add_u32(joined_rising.value, 1)?
        } else {
            joined_rising
        };
        let falling = if is_boundary && !newer.first_high {
            O::add_u32(joined_falling.value, 1)?
        } else {
            joined_falling
        };
        Ok(ThresholdCrossingsDelta {
            samples: samples.value,
            crossings: crossings.value,
            rising: rising.value,
            falling: falling.value,
            first_high: self.first_high,
            previous_high: newer.previous_high,
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || newer.quality.is_saturated()
                    || samples.saturated
                    || joined_crossings.saturated
                    || joined_rising.saturated
                    || joined_falling.saturated
                    || crossings.saturated
                    || rising.saturated
                    || falling.saturated,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::{ThresholdCrossings, ThresholdCrossingsSnapshot};
    use crate::policy::{Checked, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn check_lifecycle_and_merges<S: SampleWord>(low: S, threshold: S, high: S) {
        let mut crossings = ThresholdCrossings::<Checked, S>::new(threshold);
        assert_eq!(crossings.threshold(), threshold);
        assert_eq!(crossings.snapshot(), None);

        // Merging a summary with a different threshold fails transactionally.
        let different = ThresholdCrossings::<Checked, S>::new(low);
        assert_eq!(
            crossings.merge(&different),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(crossings.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = ThresholdCrossings::<Checked, S>::new(threshold);
        crossings.merge(&empty).expect("empty merge");
        assert_eq!(crossings.snapshot(), None);

        // Merging a non-empty summary into an empty state adopts the newer summary.
        let mut first = ThresholdCrossings::<Checked, S>::new(threshold);
        for sample in [low, threshold, high, low] {
            first.update(sample).expect("update");
        }
        crossings.merge(&first).expect("merge into empty");
        assert_eq!(
            crossings.snapshot(),
            Some(ThresholdCrossingsSnapshot {
                samples: 4,
                threshold,
                crossings: 2,
                rising: 1,
                falling: 1,
                first_high: false,
                current_high: false,
                quality: Quality::Complete,
            })
        );

        // Merging an empty newer summary into a non-empty state preserves it.
        crossings.merge(&empty).expect("merge empty newer");
        assert_eq!(crossings.snapshot().map(|s| s.samples), Some(4));

        // Merging a segment that starts high creates a rising boundary crossing.
        let mut second = ThresholdCrossings::<Checked, S>::new(threshold);
        second.update(threshold).expect("update high");
        crossings.merge(&second).expect("rising boundary merge");
        assert_eq!(
            crossings.snapshot().map(|s| (
                s.samples,
                s.crossings,
                s.rising,
                s.falling,
                s.current_high
            )),
            Some((5, 3, 2, 1, true))
        );

        // Merging another segment that starts high creates no boundary crossing.
        crossings.merge(&second).expect("same-side boundary merge");
        assert_eq!(
            crossings.snapshot().map(|s| (
                s.samples,
                s.crossings,
                s.rising,
                s.falling,
                s.current_high
            )),
            Some((6, 3, 2, 1, true))
        );

        // Merging a segment that starts low creates a falling boundary crossing.
        let mut third = ThresholdCrossings::<Checked, S>::new(threshold);
        third.update(low).expect("update low");
        crossings.merge(&third).expect("falling boundary merge");
        assert_eq!(
            crossings.snapshot().map(|s| (
                s.samples,
                s.crossings,
                s.rising,
                s.falling,
                s.current_high
            )),
            Some((7, 4, 2, 2, false))
        );

        crossings.reset();
        assert_eq!(crossings.threshold(), threshold);
        assert_eq!(crossings.snapshot(), None);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_word() {
        check_lifecycle_and_merges::<i8>(-4, 5, 9);
        check_lifecycle_and_merges::<i16>(-40, 50, 90);
        check_lifecycle_and_merges::<i32>(-400, 500, 900);
        check_lifecycle_and_merges::<u8>(2, 5, 9);
        check_lifecycle_and_merges::<u16>(20, 50, 90);
        check_lifecycle_and_merges::<u32>(200, 500, 900);
    }

    fn every_width_fails_transactionally<S: SampleWord>(low: S, threshold: S, high: S) {
        let full = ThresholdCrossings::<Checked, S> {
            samples: u32::MAX,
            threshold,
            crossings: 7,
            rising: 4,
            falling: 3,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut crossings = full;
        assert_eq!(crossings.update(high), Err(Error::Overflow));
        assert_eq!(crossings.update(low), Err(Error::Overflow));
        assert_eq!(crossings, full);

        let mut newer = ThresholdCrossings::<Checked, S>::new(threshold);
        newer.update(high).expect("representable");
        assert_eq!(crossings.merge(&newer), Err(Error::Overflow));
        assert_eq!(crossings, full);

        let mut newer_rising = ThresholdCrossings::<Checked, S>::new(threshold);
        newer_rising.update(low).expect("representable");
        newer_rising.update(high).expect("representable");

        let mut newer_falling = ThresholdCrossings::<Checked, S>::new(threshold);
        newer_falling.update(high).expect("representable");
        newer_falling.update(low).expect("representable");

        // Later counter overflow points while sample-count addition fits.
        let mut rising_overflow = ThresholdCrossings::<Checked, S> {
            samples: 10,
            threshold,
            crossings: u32::MAX,
            rising: 4,
            falling: 3,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_rising = rising_overflow;
        assert_eq!(rising_overflow.update(high), Err(Error::Overflow));
        assert_eq!(rising_overflow, before_rising);
        assert_eq!(rising_overflow.merge(&newer), Err(Error::Overflow));
        assert_eq!(rising_overflow, before_rising);
        assert_eq!(rising_overflow.merge(&newer_rising), Err(Error::Overflow));
        assert_eq!(rising_overflow, before_rising);

        let mut rising_only_overflow = ThresholdCrossings::<Checked, S> {
            samples: 10,
            threshold,
            crossings: 7,
            rising: u32::MAX,
            falling: 3,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_rising_only = rising_only_overflow;
        assert_eq!(rising_only_overflow.update(high), Err(Error::Overflow));
        assert_eq!(rising_only_overflow, before_rising_only);
        assert_eq!(rising_only_overflow.merge(&newer), Err(Error::Overflow));
        assert_eq!(rising_only_overflow, before_rising_only);
        assert_eq!(
            rising_only_overflow.merge(&newer_rising),
            Err(Error::Overflow)
        );
        assert_eq!(rising_only_overflow, before_rising_only);

        let mut falling_overflow = ThresholdCrossings::<Checked, S> {
            samples: 10,
            threshold,
            crossings: u32::MAX,
            rising: 4,
            falling: 3,
            first_high: true,
            previous_high: true,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_falling = falling_overflow;
        assert_eq!(falling_overflow.update(low), Err(Error::Overflow));
        assert_eq!(falling_overflow, before_falling);

        let mut falling_only_overflow = ThresholdCrossings::<Checked, S> {
            samples: 10,
            threshold,
            crossings: 7,
            rising: 4,
            falling: u32::MAX,
            first_high: true,
            previous_high: true,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_falling_only = falling_only_overflow;
        assert_eq!(falling_only_overflow.update(low), Err(Error::Overflow));
        assert_eq!(falling_only_overflow, before_falling_only);
        let mut newer_low = ThresholdCrossings::<Checked, S>::new(threshold);
        newer_low.update(low).expect("representable");
        assert_eq!(
            falling_only_overflow.merge(&newer_low),
            Err(Error::Overflow)
        );
        assert_eq!(falling_only_overflow, before_falling_only);
        assert_eq!(
            falling_only_overflow.merge(&newer_falling),
            Err(Error::Overflow)
        );
        assert_eq!(falling_only_overflow, before_falling_only);

        let mut saturating = ThresholdCrossings::<Saturating, S> {
            samples: u32::MAX,
            threshold,
            crossings: u32::MAX,
            rising: u32::MAX,
            falling: 0,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating.update(high).expect("saturating update accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!((snapshot.crossings, snapshot.rising), (u32::MAX, u32::MAX));
        assert!(snapshot.current_high);

        let mut tail = ThresholdCrossings::<Saturating, S>::new(threshold);
        tail.update(low).expect("tail low");
        saturating.merge(&tail).expect("saturating merge accepts");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.falling, s.current_high, s.quality)),
            Some((1, false, Quality::Saturated))
        );
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        every_width_fails_transactionally::<i8>(-1, 0, 1);
        every_width_fails_transactionally::<i16>(-1, 0, 1);
        every_width_fails_transactionally::<i32>(-1, 0, 1);
        every_width_fails_transactionally::<u8>(0, 5, 9);
        every_width_fails_transactionally::<u16>(0, 5, 9);
        every_width_fails_transactionally::<u32>(0, 5, 9);
    }
}
