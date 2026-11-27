//! Constant-state cumulative threshold crossings.

use core::marker::PhantomData;

use super::{MergeSlot, Slot};
use crate::policy::{Arithmetic, CounterWord, OverflowPolicy};
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// Cumulative threshold-crossing result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdCrossingsSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Runtime threshold; equality belongs to the high side.
    pub threshold: S,
    /// Total low-to-high and high-to-low transitions.
    pub crossings: C,
    /// Low-to-high transitions.
    pub rising: C,
    /// High-to-low transitions.
    pub falling: C,
    /// Whether the first sample is on the high side, retained for exact concatenation.
    pub first_high: bool,
    /// Whether the latest sample is on the high side.
    pub current_high: bool,
    /// Saturation quality.
    pub quality: Quality,
}

/// Cumulative crossings of one runtime threshold.
///
/// `C` is the counter word (`u8`, `u16`, `u32`, or `u64`) and `S` is the sample value type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdCrossings<O, C, S> {
    samples: C,
    threshold: S,
    crossings: C,
    rising: C,
    falling: C,
    first_high: bool,
    previous_high: bool,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdCrossingsDelta<C> {
    samples: C,
    crossings: C,
    rising: C,
    falling: C,
    first_high: bool,
    previous_high: bool,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleValue> ThresholdCrossings<O, C, S> {
    /// Creates an empty crossing summary for one threshold.
    pub const fn new(threshold: S) -> Self {
        Self {
            samples: C::ZERO,
            threshold,
            crossings: C::ZERO,
            rising: C::ZERO,
            falling: C::ZERO,
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

    /// Clears counts while retaining the configured threshold.
    pub const fn reset(&mut self) {
        self.samples = C::ZERO;
        self.crossings = C::ZERO;
        self.rising = C::ZERO;
        self.falling = C::ZERO;
        self.first_high = false;
        self.previous_high = false;
        self.quality = Quality::Complete;
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: SampleValue> ThresholdCrossings<O, C, S> {
        /// Accepts one sample transactionally.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let delta = match self.prepare_word(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_word(delta);
            Ok(())
        }

        /// Returns `None` until the first sample is accepted.
        pub c0nst fn snapshot(&self) -> Option<ThresholdCrossingsSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.to_u64() == 0 {
                None
            } else {
                Some(ThresholdCrossingsSnapshot {
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
        }

        /// Exactly appends newer crossings with the same threshold and includes the boundary.
        #[allow(clippy::question_mark)]
        pub c0nst fn merge(&mut self, newer: &Self) -> Result<(), Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let delta = match self.prepare_merge_word(newer) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit_word(delta);
            Ok(())
        }

        #[allow(clippy::question_mark)]
        c0nst fn prepare_word(&self, sample: S) -> Result<ThresholdCrossingsDelta<C>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let current_high = crate::sealed::SampleValue::at_or_above(sample, self.threshold);
            let empty = self.samples.to_u64() == 0;
            let (crossings, rising, falling) = if empty || current_high == self.previous_high {
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
                let crossings = match C::add::<O>(self.crossings, C::ONE) {
                    Ok(crossings) => crossings,
                    Err(error) => return Err(error),
                };
                let rising = match C::add::<O>(self.rising, C::ONE) {
                    Ok(rising) => rising,
                    Err(error) => return Err(error),
                };
                (
                    crossings,
                    rising,
                    Arithmetic {
                        value: self.falling,
                        saturated: false,
                    },
                )
            } else {
                let crossings = match C::add::<O>(self.crossings, C::ONE) {
                    Ok(crossings) => crossings,
                    Err(error) => return Err(error),
                };
                let falling = match C::add::<O>(self.falling, C::ONE) {
                    Ok(falling) => falling,
                    Err(error) => return Err(error),
                };
                (
                    crossings,
                    Arithmetic {
                        value: self.rising,
                        saturated: false,
                    },
                    falling,
                )
            };
            Ok(ThresholdCrossingsDelta {
                samples: samples.value,
                crossings: crossings.value,
                rising: rising.value,
                falling: falling.value,
                first_high: if empty { current_high } else { self.first_high },
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

        const fn commit_word(&mut self, delta: ThresholdCrossingsDelta<C>) {
            self.samples = delta.samples;
            self.crossings = delta.crossings;
            self.rising = delta.rising;
            self.falling = delta.falling;
            self.first_high = delta.first_high;
            self.previous_high = delta.previous_high;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<ThresholdCrossingsDelta<C>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            if crate::sealed::SampleValue::above(self.threshold, newer.threshold)
                || crate::sealed::SampleValue::above(newer.threshold, self.threshold)
            {
                return Err(Error::InvalidConfiguration);
            }
            if newer.samples.to_u64() == 0 {
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
            if self.samples.to_u64() == 0 {
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

            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let joined_crossings = match C::add::<O>(self.crossings, newer.crossings) {
                Ok(joined) => joined,
                Err(error) => return Err(error),
            };
            let joined_rising = match C::add::<O>(self.rising, newer.rising) {
                Ok(joined) => joined,
                Err(error) => return Err(error),
            };
            let joined_falling = match C::add::<O>(self.falling, newer.falling) {
                Ok(joined) => joined,
                Err(error) => return Err(error),
            };
            let is_boundary = self.previous_high != newer.first_high;
            let crossings = if is_boundary {
                match C::add::<O>(joined_crossings.value, C::ONE) {
                    Ok(crossings) => crossings,
                    Err(error) => return Err(error),
                }
            } else {
                joined_crossings
            };
            let rising = if is_boundary && newer.first_high {
                match C::add::<O>(joined_rising.value, C::ONE) {
                    Ok(rising) => rising,
                    Err(error) => return Err(error),
                }
            } else {
                joined_rising
            };
            let falling = if is_boundary && !newer.first_high {
                match C::add::<O>(joined_falling.value, C::ONE) {
                    Ok(falling) => falling,
                    Err(error) => return Err(error),
                }
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

    c0nst impl<O, C, S> crate::sealed::OnlineSlot for ThresholdCrossings<O, C, S> {}

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue> Slot<S>
        for ThresholdCrossings<O, C, S>
    {
        type Delta = ThresholdCrossingsDelta<C>;
        type Snapshot<'a>
            = Option<ThresholdCrossingsSnapshot<C, S>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare_word(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit_word(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            ThresholdCrossings::snapshot(self)
        }

        fn reset(&mut self) {
            ThresholdCrossings::reset(self);
        }
    }

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue>
        MergeSlot<S> for ThresholdCrossings<O, C, S>
    {
        type MergeDelta = ThresholdCrossingsDelta<C>;

        fn prepare_merge(&self, newer: &Self) -> Result<Self::MergeDelta, Error> {
            self.prepare_merge_word(newer)
        }

        fn commit_merge(&mut self, delta: Self::MergeDelta) {
            self.commit_word(delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::{ThresholdCrossings, ThresholdCrossingsSnapshot};
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn count<C: CounterWord>(n: u64) -> C {
        C::from_u64(n).expect("small counts fit every counter word")
    }

    fn check_lifecycle_and_merges<S: SampleWord>(low: S, threshold: S, high: S) {
        let mut crossings = ThresholdCrossings::<Checked, u32, S>::new(threshold);
        assert_eq!(crossings.threshold(), threshold);
        assert_eq!(crossings.snapshot(), None);

        // Merging a summary with a different threshold fails transactionally.
        let different = ThresholdCrossings::<Checked, u32, S>::new(low);
        assert_eq!(
            crossings.merge(&different),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(crossings.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = ThresholdCrossings::<Checked, u32, S>::new(threshold);
        crossings.merge(&empty).expect("empty merge");
        assert_eq!(crossings.snapshot(), None);

        // Merging a non-empty summary into an empty state adopts the newer summary.
        let mut first = ThresholdCrossings::<Checked, u32, S>::new(threshold);
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
        let mut second = ThresholdCrossings::<Checked, u32, S>::new(threshold);
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
        let mut third = ThresholdCrossings::<Checked, u32, S>::new(threshold);
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

    fn every_width_fails_transactionally<S: SampleWord, C: CounterWord>(
        low: S,
        threshold: S,
        high: S,
    ) {
        let full = ThresholdCrossings::<Checked, C, S> {
            samples: C::MAX,
            threshold,
            crossings: count(7),
            rising: count(4),
            falling: count(3),
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut crossings = full;
        assert_eq!(crossings.update(high), Err(Error::Overflow));
        assert_eq!(crossings.update(low), Err(Error::Overflow));
        assert_eq!(crossings, full);

        let mut newer = ThresholdCrossings::<Checked, C, S>::new(threshold);
        newer.update(high).expect("representable");
        assert_eq!(crossings.merge(&newer), Err(Error::Overflow));
        assert_eq!(crossings, full);

        let mut newer_rising = ThresholdCrossings::<Checked, C, S>::new(threshold);
        newer_rising.update(low).expect("representable");
        newer_rising.update(high).expect("representable");

        let mut newer_falling = ThresholdCrossings::<Checked, C, S>::new(threshold);
        newer_falling.update(high).expect("representable");
        newer_falling.update(low).expect("representable");

        // Later counter overflow points while sample-count addition fits.
        let mut rising_overflow = ThresholdCrossings::<Checked, C, S> {
            samples: count(10),
            threshold,
            crossings: C::MAX,
            rising: count(4),
            falling: count(3),
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

        let mut rising_only_overflow = ThresholdCrossings::<Checked, C, S> {
            samples: count(10),
            threshold,
            crossings: count(7),
            rising: C::MAX,
            falling: count(3),
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

        let mut falling_overflow = ThresholdCrossings::<Checked, C, S> {
            samples: count(10),
            threshold,
            crossings: C::MAX,
            rising: count(4),
            falling: count(3),
            first_high: true,
            previous_high: true,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_falling = falling_overflow;
        assert_eq!(falling_overflow.update(low), Err(Error::Overflow));
        assert_eq!(falling_overflow, before_falling);

        let mut falling_only_overflow = ThresholdCrossings::<Checked, C, S> {
            samples: count(10),
            threshold,
            crossings: count(7),
            rising: count(4),
            falling: C::MAX,
            first_high: true,
            previous_high: true,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_falling_only = falling_only_overflow;
        assert_eq!(falling_only_overflow.update(low), Err(Error::Overflow));
        assert_eq!(falling_only_overflow, before_falling_only);
        let mut newer_low = ThresholdCrossings::<Checked, C, S>::new(threshold);
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

        let mut saturating = ThresholdCrossings::<Saturating, C, S> {
            samples: C::MAX,
            threshold,
            crossings: C::MAX,
            rising: C::MAX,
            falling: C::ZERO,
            first_high: false,
            previous_high: false,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating.update(high).expect("saturating update accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!((snapshot.crossings, snapshot.rising), (C::MAX, C::MAX));
        assert!(snapshot.current_high);

        let mut tail = ThresholdCrossings::<Saturating, C, S>::new(threshold);
        tail.update(low).expect("tail low");
        saturating.merge(&tail).expect("saturating merge accepts");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.falling, s.current_high, s.quality)),
            Some((C::ONE, false, Quality::Saturated))
        );
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        every_width_fails_transactionally::<i8, u8>(-1, 0, 1);
        every_width_fails_transactionally::<i8, u16>(-1, 0, 1);
        every_width_fails_transactionally::<i8, u32>(-1, 0, 1);
        every_width_fails_transactionally::<i8, u64>(-1, 0, 1);
        every_width_fails_transactionally::<i16, u8>(-1, 0, 1);
        every_width_fails_transactionally::<i16, u16>(-1, 0, 1);
        every_width_fails_transactionally::<i16, u32>(-1, 0, 1);
        every_width_fails_transactionally::<i16, u64>(-1, 0, 1);
        every_width_fails_transactionally::<i32, u8>(-1, 0, 1);
        every_width_fails_transactionally::<i32, u16>(-1, 0, 1);
        every_width_fails_transactionally::<i32, u32>(-1, 0, 1);
        every_width_fails_transactionally::<i32, u64>(-1, 0, 1);
        every_width_fails_transactionally::<u8, u8>(0, 5, 9);
        every_width_fails_transactionally::<u8, u16>(0, 5, 9);
        every_width_fails_transactionally::<u8, u32>(0, 5, 9);
        every_width_fails_transactionally::<u8, u64>(0, 5, 9);
        every_width_fails_transactionally::<u16, u8>(0, 5, 9);
        every_width_fails_transactionally::<u16, u16>(0, 5, 9);
        every_width_fails_transactionally::<u16, u32>(0, 5, 9);
        every_width_fails_transactionally::<u16, u64>(0, 5, 9);
        every_width_fails_transactionally::<u32, u8>(0, 5, 9);
        every_width_fails_transactionally::<u32, u16>(0, 5, 9);
        every_width_fails_transactionally::<u32, u32>(0, 5, 9);
        every_width_fails_transactionally::<u32, u64>(0, 5, 9);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_threshold_crossings_are_const_evaluable_on_nightly() {
        const SNAPSHOT: ThresholdCrossingsSnapshot<u8, i16> = {
            let mut left = ThresholdCrossings::<Checked, u8, i16>::new(0);
            let mut right = ThresholdCrossings::<Checked, u8, i16>::new(0);
            assert!(left.update(-1).is_ok());
            assert!(left.update(1).is_ok());
            assert!(right.update(-1).is_ok());
            assert!(left.merge(&right).is_ok());
            match left.snapshot() {
                Some(snapshot) => snapshot,
                None => unreachable!(),
            }
        };
        assert_eq!(SNAPSHOT.samples, 3);
        assert_eq!(
            (SNAPSHOT.crossings, SNAPSHOT.rising, SNAPSHOT.falling),
            (2, 1, 1)
        );
    }
}
