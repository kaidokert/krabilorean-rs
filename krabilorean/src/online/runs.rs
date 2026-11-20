//! Constant-state cumulative strict threshold runs.

use core::marker::PhantomData;

use crate::policy::{Arithmetic, OverflowPolicy};
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// Cumulative strict threshold-run result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdRunsSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Runtime threshold; only samples strictly greater belong to a run.
    pub threshold: S,
    /// Length of the strict-high prefix, retained for exact concatenation.
    pub prefix_run: u32,
    /// Length of the current strict-high run.
    pub current_run: u32,
    /// Longest strict-high run since reset.
    pub longest_run: u32,
    /// Saturation quality.
    pub quality: Quality,
}

/// Cumulative current and longest runs strictly above one threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdRuns<O, S> {
    samples: u32,
    threshold: S,
    prefix_run: u32,
    current_run: u32,
    longest_run: u32,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective threshold-run state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThresholdRunsDelta {
    samples: u32,
    prefix_run: u32,
    current_run: u32,
    longest_run: u32,
    quality: Quality,
}

impl<O, S: SampleValue> ThresholdRuns<O, S> {
    /// Creates an empty strict-high run summary.
    pub const fn new(threshold: S) -> Self {
        Self {
            samples: 0,
            threshold,
            prefix_run: 0,
            current_run: 0,
            longest_run: 0,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Returns the configured threshold.
    pub const fn threshold(&self) -> S {
        self.threshold
    }
}

impl<O: OverflowPolicy, S: SampleValue> ThresholdRuns<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<ThresholdRunsSnapshot<S>> {
        (self.samples != 0).then_some(ThresholdRunsSnapshot {
            samples: self.samples,
            threshold: self.threshold,
            prefix_run: self.prefix_run,
            current_run: self.current_run,
            longest_run: self.longest_run,
            quality: self.quality,
        })
    }

    /// Clears run state while retaining the configured threshold.
    pub fn reset(&mut self) {
        self.samples = 0;
        self.prefix_run = 0;
        self.current_run = 0;
        self.longest_run = 0;
        self.quality = Quality::Complete;
    }

    /// Exactly appends newer run state with the same threshold, including a spanning run.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    fn prepare_word(&self, sample: S) -> Result<ThresholdRunsDelta, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        let high = crate::sealed::SampleValue::above(sample, self.threshold);
        let current_run = if high {
            O::add_u32(self.current_run, 1)?
        } else {
            Arithmetic {
                value: 0,
                saturated: false,
            }
        };
        // Under `Saturating`, `samples` and `prefix_run` can both saturate at `u32::MAX` before a
        // low sample resets `current_run`, so `current_run == prefix_run` distinguishes an unbroken
        // all-high prefix from a saturated prefix followed by a low sample.
        let all_high = (self.samples == self.prefix_run)
            & (!O::SATURATING | (self.current_run == self.prefix_run));
        let prefix_run = if all_high & high {
            current_run
        } else {
            Arithmetic {
                value: self.prefix_run,
                saturated: false,
            }
        };
        let longest_run = self.longest_run.max(current_run.value);
        Ok(ThresholdRunsDelta {
            samples: samples.value,
            prefix_run: prefix_run.value,
            current_run: current_run.value,
            longest_run,
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || samples.saturated
                    || prefix_run.saturated
                    || current_run.saturated,
            ),
        })
    }

    fn commit_word(&mut self, delta: ThresholdRunsDelta) {
        self.samples = delta.samples;
        self.prefix_run = delta.prefix_run;
        self.current_run = delta.current_run;
        self.longest_run = delta.longest_run;
        self.quality = delta.quality;
    }

    fn prepare_merge_word(&self, newer: &Self) -> Result<ThresholdRunsDelta, Error> {
        if self.threshold != newer.threshold {
            return Err(Error::InvalidConfiguration);
        }
        if newer.samples == 0 {
            return Ok(ThresholdRunsDelta {
                samples: self.samples,
                prefix_run: self.prefix_run,
                current_run: self.current_run,
                longest_run: self.longest_run,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(ThresholdRunsDelta {
                samples: newer.samples,
                prefix_run: newer.prefix_run,
                current_run: newer.current_run,
                longest_run: newer.longest_run,
                quality: newer.quality,
            });
        }

        let samples = O::add_u32(self.samples, newer.samples)?;
        let bridge = O::add_u32(self.current_run, newer.prefix_run)?;
        // A saturated summary that ended low has `prefix_run == samples == u32::MAX` but a shorter
        // `current_run`; see `prepare_word`.
        let self_all_high = if O::SATURATING {
            self.prefix_run == self.samples && self.current_run == self.prefix_run
        } else {
            self.prefix_run == self.samples
        };
        let newer_all_high = if O::SATURATING {
            newer.prefix_run == newer.samples && newer.current_run == newer.prefix_run
        } else {
            newer.prefix_run == newer.samples
        };
        let prefix_run = if self_all_high {
            O::add_u32(self.samples, newer.prefix_run)?
        } else {
            Arithmetic {
                value: self.prefix_run,
                saturated: false,
            }
        };
        let current_run = if newer_all_high {
            O::add_u32(self.current_run, newer.samples)?
        } else {
            Arithmetic {
                value: newer.current_run,
                saturated: false,
            }
        };
        Ok(ThresholdRunsDelta {
            samples: samples.value,
            prefix_run: prefix_run.value,
            current_run: current_run.value,
            longest_run: self.longest_run.max(newer.longest_run).max(bridge.value),
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated()
                    || newer.quality.is_saturated()
                    || samples.saturated
                    || bridge.saturated
                    || prefix_run.saturated
                    || current_run.saturated,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::{ThresholdRuns, ThresholdRunsSnapshot};
    use crate::policy::{Checked, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn check_lifecycle_and_merges<S: SampleWord>(low: S, threshold: S, high: S) {
        let mut runs = ThresholdRuns::<Checked, S>::new(threshold);
        assert_eq!(runs.threshold(), threshold);
        assert_eq!(runs.snapshot(), None);

        // Merging a summary with a different threshold fails transactionally.
        let different = ThresholdRuns::<Checked, S>::new(low);
        assert_eq!(runs.merge(&different), Err(Error::InvalidConfiguration));
        assert_eq!(runs.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = ThresholdRuns::<Checked, S>::new(threshold);
        runs.merge(&empty).expect("empty merge");
        assert_eq!(runs.snapshot(), None);

        // Merging an all-high segment into an empty state adopts it.
        let mut all_high = ThresholdRuns::<Checked, S>::new(threshold);
        all_high.update(high).expect("update high");
        all_high.update(high).expect("update high");
        runs.merge(&all_high).expect("merge into empty");
        assert_eq!(
            runs.snapshot(),
            Some(ThresholdRunsSnapshot {
                samples: 2,
                threshold,
                prefix_run: 2,
                current_run: 2,
                longest_run: 2,
                quality: Quality::Complete,
            })
        );

        // Merging an empty newer summary into a non-empty state preserves it.
        runs.merge(&empty).expect("merge empty newer");
        assert_eq!(runs.snapshot().map(|s| s.samples), Some(2));

        // Merging another all-high segment extends both prefix_run and current_run.
        runs.merge(&all_high).expect("merge all-high into all-high");
        assert_eq!(
            runs.snapshot()
                .map(|s| (s.samples, s.prefix_run, s.current_run, s.longest_run)),
            Some((4, 4, 4, 4))
        );

        // Equality breaks a strict-high run; subsequent high samples start a new trailing run.
        let mut mixed = ThresholdRuns::<Checked, S>::new(threshold);
        for sample in [high, threshold, low, high, high] {
            mixed.update(sample).expect("update mixed");
        }
        assert_eq!(
            mixed
                .snapshot()
                .map(|s| (s.samples, s.prefix_run, s.current_run, s.longest_run)),
            Some((5, 1, 2, 2))
        );

        // Merging mixed into all-high extends prefix_run by mixed.prefix_run (4 + 1 = 5) and adopts
        // mixed.current_run (2).
        runs.merge(&mixed).expect("merge mixed into all-high");
        assert_eq!(
            runs.snapshot()
                .map(|s| (s.samples, s.prefix_run, s.current_run, s.longest_run)),
            Some((9, 5, 2, 5))
        );

        // Merging an all-high segment into a non-all-high segment keeps prefix_run (5) and extends
        // current_run (2 + 2 = 4).
        runs.merge(&all_high).expect("merge all-high into mixed");
        assert_eq!(
            runs.snapshot()
                .map(|s| (s.samples, s.prefix_run, s.current_run, s.longest_run)),
            Some((11, 5, 4, 5))
        );

        // Merging mixed again bridges current_run (4) + mixed.prefix_run (1) = 5 while keeping
        // prefix_run (5) and adopting mixed.current_run (2).
        runs.merge(&mixed).expect("merge mixed into non-all-high");
        assert_eq!(
            runs.snapshot()
                .map(|s| (s.samples, s.prefix_run, s.current_run, s.longest_run)),
            Some((16, 5, 2, 5))
        );

        runs.reset();
        assert_eq!(runs.threshold(), threshold);
        assert_eq!(runs.snapshot(), None);
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
        let full = ThresholdRuns::<Checked, S> {
            samples: u32::MAX,
            threshold,
            prefix_run: 0,
            current_run: 7,
            longest_run: 7,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut runs = full;
        assert_eq!(runs.update(high), Err(Error::Overflow));
        assert_eq!(runs.update(low), Err(Error::Overflow));
        assert_eq!(runs, full);

        let mut newer = ThresholdRuns::<Checked, S>::new(threshold);
        newer.update(high).expect("representable");
        assert_eq!(runs.merge(&newer), Err(Error::Overflow));
        assert_eq!(runs, full);

        // Later counter overflow points while sample-count addition fits.
        let mut current_max = ThresholdRuns::<Checked, S> {
            samples: 10,
            threshold,
            prefix_run: 3,
            current_run: u32::MAX,
            longest_run: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_current_max = current_max;
        assert_eq!(current_max.update(high), Err(Error::Overflow));
        assert_eq!(current_max, before_current_max);
        assert_eq!(current_max.merge(&newer), Err(Error::Overflow));
        assert_eq!(current_max, before_current_max);

        let mut prefix_overflow = ThresholdRuns::<Checked, S> {
            samples: u32::MAX - 2,
            threshold,
            prefix_run: u32::MAX - 2,
            current_run: 1,
            longest_run: u32::MAX - 2,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_prefix = prefix_overflow;
        let newer_high_three = ThresholdRuns::<Checked, S> {
            samples: 2,
            threshold,
            prefix_run: 3,
            current_run: 2,
            longest_run: 3,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        assert_eq!(
            prefix_overflow.merge(&newer_high_three),
            Err(Error::Overflow)
        );
        assert_eq!(prefix_overflow, before_prefix);

        let mut trailing_overflow = ThresholdRuns::<Checked, S> {
            samples: 2,
            threshold,
            prefix_run: 1,
            current_run: u32::MAX - 1,
            longest_run: u32::MAX - 1,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_trailing = trailing_overflow;
        let newer_all_high = ThresholdRuns::<Checked, S> {
            samples: 2,
            threshold,
            prefix_run: 1,
            current_run: 2,
            longest_run: 2,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut newer_all_high_valid = newer_all_high;
        newer_all_high_valid.prefix_run = 2;
        assert_eq!(
            trailing_overflow.merge(&newer_all_high_valid),
            Err(Error::Overflow)
        );
        assert_eq!(trailing_overflow, before_trailing);

        // Saturated all-high stream followed by a low sample and then a high sample must preserve
        // prefix_run == u32::MAX and must not be treated as all-high when merged.
        let mut saturating = ThresholdRuns::<Saturating, S> {
            samples: u32::MAX,
            threshold,
            prefix_run: u32::MAX,
            current_run: u32::MAX,
            longest_run: u32::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating.update(high).expect("saturating update accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.prefix_run, u32::MAX);
        assert_eq!(snapshot.current_run, u32::MAX);

        saturating.update(low).expect("low breaks all-high");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((u32::MAX, 0, u32::MAX, Quality::Saturated))
        );

        // Merging a saturated summary that ended low onto a high prefix must yield current_run == 0.
        let mut head = ThresholdRuns::<Saturating, S>::new(threshold);
        head.update(high).expect("head high");
        head.merge(&saturating)
            .expect("merge low-ending saturated summary");
        assert_eq!(
            head.snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((u32::MAX, 0, u32::MAX, Quality::Saturated))
        );

        // Subsequent high sample on the saturated summary starts current_run == 1 without
        // overwriting prefix_run == u32::MAX.
        saturating
            .update(high)
            .expect("high after low on saturated summary");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((u32::MAX, 1, u32::MAX, Quality::Saturated))
        );

        let mut tail_high = ThresholdRuns::<Saturating, S>::new(threshold);
        tail_high.update(high).expect("tail high");
        saturating
            .merge(&tail_high)
            .expect("merge all-high tail onto broken saturated summary");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((u32::MAX, 2, u32::MAX, Quality::Saturated))
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
