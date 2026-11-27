//! Constant-state cumulative strict threshold runs.

use core::marker::PhantomData;

use super::{MergeSlot, Slot};
use crate::policy::{Arithmetic, CounterWord, OverflowPolicy};
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// Cumulative strict threshold-run result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdRunsSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Runtime threshold; only samples strictly greater belong to a run.
    pub threshold: S,
    /// Length of the strict-high prefix, retained for exact concatenation.
    pub prefix_run: C,
    /// Length of the current strict-high run.
    pub current_run: C,
    /// Longest strict-high run since reset.
    pub longest_run: C,
    /// Saturation quality.
    pub quality: Quality,
}

/// Cumulative current and longest runs strictly above one threshold.
///
/// `C` is the counter word (`u8`, `u16`, `u32`, or `u64`) and `S` is the sample value type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdRuns<O, C, S> {
    samples: C,
    threshold: S,
    prefix_run: C,
    current_run: C,
    longest_run: C,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ThresholdRunsDelta<C> {
    samples: C,
    prefix_run: C,
    current_run: C,
    longest_run: C,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleValue> ThresholdRuns<O, C, S> {
    /// Creates an empty strict-high run summary.
    pub const fn new(threshold: S) -> Self {
        Self {
            samples: C::ZERO,
            threshold,
            prefix_run: C::ZERO,
            current_run: C::ZERO,
            longest_run: C::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Returns the configured threshold.
    pub const fn threshold(&self) -> S {
        self.threshold
    }

    /// Clears run state while retaining the configured threshold.
    pub const fn reset(&mut self) {
        self.samples = C::ZERO;
        self.prefix_run = C::ZERO;
        self.current_run = C::ZERO;
        self.longest_run = C::ZERO;
        self.quality = Quality::Complete;
    }
}

c0nst::c0nst! {
    c0nst fn max_counter<C: [c0nst] CounterWord>(left: C, right: C) -> C {
        if left.to_u64() >= right.to_u64() {
            left
        } else {
            right
        }
    }

    impl<O: OverflowPolicy, C: CounterWord, S: SampleValue> ThresholdRuns<O, C, S> {
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
        pub c0nst fn snapshot(&self) -> Option<ThresholdRunsSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.to_u64() == 0 {
                None
            } else {
                Some(ThresholdRunsSnapshot {
                    samples: self.samples,
                    threshold: self.threshold,
                    prefix_run: self.prefix_run,
                    current_run: self.current_run,
                    longest_run: self.longest_run,
                    quality: self.quality,
                })
            }
        }

        /// Exactly appends newer run state with the same threshold, including a spanning run.
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
        c0nst fn prepare_word(&self, sample: S) -> Result<ThresholdRunsDelta<C>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let high = crate::sealed::SampleValue::above(sample, self.threshold);
            let current_run = if high {
                match C::add::<O>(self.current_run, C::ONE) {
                    Ok(run) => run,
                    Err(error) => return Err(error),
                }
            } else {
                Arithmetic {
                    value: C::ZERO,
                    saturated: false,
                }
            };
            // Under `Saturating`, `samples` and `prefix_run` can both saturate at `C::MAX` before a
            // low sample resets `current_run`, so `current_run == prefix_run` distinguishes an
            // unbroken all-high prefix from a saturated prefix followed by a low sample.
            let prefix_u64 = self.prefix_run.to_u64();
            let all_high = (self.samples.to_u64() == prefix_u64)
                & (!O::SATURATING | (self.current_run.to_u64() == prefix_u64));
            let prefix_run = if all_high & high {
                current_run
            } else {
                Arithmetic {
                    value: self.prefix_run,
                    saturated: false,
                }
            };
            let longest_run = max_counter(self.longest_run, current_run.value);
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

        const fn commit_word(&mut self, delta: ThresholdRunsDelta<C>) {
            self.samples = delta.samples;
            self.prefix_run = delta.prefix_run;
            self.current_run = delta.current_run;
            self.longest_run = delta.longest_run;
            self.quality = delta.quality;
        }

        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<ThresholdRunsDelta<C>, Error>
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
                return Ok(ThresholdRunsDelta {
                    samples: self.samples,
                    prefix_run: self.prefix_run,
                    current_run: self.current_run,
                    longest_run: self.longest_run,
                    quality: self.quality,
                });
            }
            if self.samples.to_u64() == 0 {
                return Ok(ThresholdRunsDelta {
                    samples: newer.samples,
                    prefix_run: newer.prefix_run,
                    current_run: newer.current_run,
                    longest_run: newer.longest_run,
                    quality: newer.quality,
                });
            }

            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let bridge = match C::add::<O>(self.current_run, newer.prefix_run) {
                Ok(bridge) => bridge,
                Err(error) => return Err(error),
            };
            // A saturated summary that ended low has `prefix_run == samples == C::MAX` but a
            // shorter `current_run`; see `prepare_word`.
            let self_prefix = self.prefix_run.to_u64();
            let self_all_high = if O::SATURATING {
                self_prefix == self.samples.to_u64() && self.current_run.to_u64() == self_prefix
            } else {
                self_prefix == self.samples.to_u64()
            };
            let newer_prefix = newer.prefix_run.to_u64();
            let newer_all_high = if O::SATURATING {
                newer_prefix == newer.samples.to_u64() && newer.current_run.to_u64() == newer_prefix
            } else {
                newer_prefix == newer.samples.to_u64()
            };
            let prefix_run = if self_all_high {
                match C::add::<O>(self.samples, newer.prefix_run) {
                    Ok(prefix) => prefix,
                    Err(error) => return Err(error),
                }
            } else {
                Arithmetic {
                    value: self.prefix_run,
                    saturated: false,
                }
            };
            let current_run = if newer_all_high {
                match C::add::<O>(self.current_run, newer.samples) {
                    Ok(current) => current,
                    Err(error) => return Err(error),
                }
            } else {
                Arithmetic {
                    value: newer.current_run,
                    saturated: false,
                }
            };
            let longest_run = max_counter(
                max_counter(self.longest_run, newer.longest_run),
                bridge.value,
            );
            Ok(ThresholdRunsDelta {
                samples: samples.value,
                prefix_run: prefix_run.value,
                current_run: current_run.value,
                longest_run,
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

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue> Slot<S>
        for ThresholdRuns<O, C, S>
    {
        type Delta = ThresholdRunsDelta<C>;
        type Snapshot<'a>
            = Option<ThresholdRunsSnapshot<C, S>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare_word(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit_word(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            ThresholdRuns::snapshot(self)
        }

        fn reset(&mut self) {
            ThresholdRuns::reset(self);
        }
    }

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue>
        MergeSlot<S> for ThresholdRuns<O, C, S>
    {
        type MergeDelta = ThresholdRunsDelta<C>;

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

    use super::{ThresholdRuns, ThresholdRunsSnapshot};
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn count<C: CounterWord>(n: u64) -> C {
        C::from_u64(n).expect("small counts fit every counter word")
    }

    fn less<C: CounterWord>(max: C, n: u64) -> C {
        max.checked_sub(count(n))
            .expect("n is below every counter maximum")
    }

    fn check_lifecycle_and_merges<S: SampleWord>(low: S, threshold: S, high: S) {
        let mut runs = ThresholdRuns::<Checked, u32, S>::new(threshold);
        assert_eq!(runs.threshold(), threshold);
        assert_eq!(runs.snapshot(), None);

        // Merging a summary with a different threshold fails transactionally.
        let different = ThresholdRuns::<Checked, u32, S>::new(low);
        assert_eq!(runs.merge(&different), Err(Error::InvalidConfiguration));
        assert_eq!(runs.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty = ThresholdRuns::<Checked, u32, S>::new(threshold);
        runs.merge(&empty).expect("empty merge");
        assert_eq!(runs.snapshot(), None);

        // Merging an all-high segment into an empty state adopts it.
        let mut all_high = ThresholdRuns::<Checked, u32, S>::new(threshold);
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
        let mut mixed = ThresholdRuns::<Checked, u32, S>::new(threshold);
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

    fn every_width_fails_transactionally<S: SampleWord, C: CounterWord>(
        low: S,
        threshold: S,
        high: S,
    ) {
        let full = ThresholdRuns::<Checked, C, S> {
            samples: C::MAX,
            threshold,
            prefix_run: C::ZERO,
            current_run: count(7),
            longest_run: count(7),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let mut runs = full;
        assert_eq!(runs.update(high), Err(Error::Overflow));
        assert_eq!(runs.update(low), Err(Error::Overflow));
        assert_eq!(runs, full);

        let mut newer = ThresholdRuns::<Checked, C, S>::new(threshold);
        newer.update(high).expect("representable");
        assert_eq!(runs.merge(&newer), Err(Error::Overflow));
        assert_eq!(runs, full);

        // Later counter overflow points while sample-count addition fits.
        let mut current_max = ThresholdRuns::<Checked, C, S> {
            samples: count(10),
            threshold,
            prefix_run: count(3),
            current_run: C::MAX,
            longest_run: C::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_current_max = current_max;
        assert_eq!(current_max.update(high), Err(Error::Overflow));
        assert_eq!(current_max, before_current_max);
        assert_eq!(current_max.merge(&newer), Err(Error::Overflow));
        assert_eq!(current_max, before_current_max);

        let mut prefix_overflow = ThresholdRuns::<Checked, C, S> {
            samples: less(C::MAX, 2),
            threshold,
            prefix_run: less(C::MAX, 2),
            current_run: C::ONE,
            longest_run: less(C::MAX, 2),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_prefix = prefix_overflow;
        let newer_high_three = ThresholdRuns::<Checked, C, S> {
            samples: count(2),
            threshold,
            prefix_run: count(3),
            current_run: count(2),
            longest_run: count(3),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        assert_eq!(
            prefix_overflow.merge(&newer_high_three),
            Err(Error::Overflow)
        );
        assert_eq!(prefix_overflow, before_prefix);

        let mut trailing_overflow = ThresholdRuns::<Checked, C, S> {
            samples: count(2),
            threshold,
            prefix_run: C::ONE,
            current_run: less(C::MAX, 1),
            longest_run: less(C::MAX, 1),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_trailing = trailing_overflow;
        let newer_all_high = ThresholdRuns::<Checked, C, S> {
            samples: count(2),
            threshold,
            prefix_run: count(2),
            current_run: count(2),
            longest_run: count(2),
            quality: Quality::Complete,
            policy: PhantomData,
        };
        assert_eq!(
            trailing_overflow.merge(&newer_all_high),
            Err(Error::Overflow)
        );
        assert_eq!(trailing_overflow, before_trailing);

        // Saturated all-high stream followed by a low sample and then a high sample must preserve
        // prefix_run == C::MAX and must not be treated as all-high when merged.
        let mut saturating = ThresholdRuns::<Saturating, C, S> {
            samples: C::MAX,
            threshold,
            prefix_run: C::MAX,
            current_run: C::MAX,
            longest_run: C::MAX,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating.update(high).expect("saturating update accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.prefix_run, C::MAX);
        assert_eq!(snapshot.current_run, C::MAX);

        saturating.update(low).expect("low breaks all-high");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((C::MAX, C::ZERO, C::MAX, Quality::Saturated))
        );

        // Merging a saturated summary that ended low onto a high prefix must yield current_run == 0.
        let mut head = ThresholdRuns::<Saturating, C, S>::new(threshold);
        head.update(high).expect("head high");
        head.merge(&saturating)
            .expect("merge low-ending saturated summary");
        assert_eq!(
            head.snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((C::MAX, C::ZERO, C::MAX, Quality::Saturated))
        );

        // Subsequent high sample on the saturated summary starts current_run == 1 without
        // overwriting prefix_run == C::MAX.
        saturating
            .update(high)
            .expect("high after low on saturated summary");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((C::MAX, C::ONE, C::MAX, Quality::Saturated))
        );

        let mut tail_high = ThresholdRuns::<Saturating, C, S>::new(threshold);
        tail_high.update(high).expect("tail high");
        saturating
            .merge(&tail_high)
            .expect("merge all-high tail onto broken saturated summary");
        assert_eq!(
            saturating
                .snapshot()
                .map(|s| (s.prefix_run, s.current_run, s.longest_run, s.quality)),
            Some((C::MAX, count(2), C::MAX, Quality::Saturated))
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
    fn c0nst_threshold_runs_are_const_evaluable_on_nightly() {
        const SNAPSHOT: ThresholdRunsSnapshot<u8, i16> = {
            let mut left = ThresholdRuns::<Checked, u8, i16>::new(0);
            let mut right = ThresholdRuns::<Checked, u8, i16>::new(0);
            assert!(left.update(1).is_ok());
            assert!(left.update(2).is_ok());
            assert!(right.update(3).is_ok());
            assert!(left.merge(&right).is_ok());
            match left.snapshot() {
                Some(snapshot) => snapshot,
                None => unreachable!(),
            }
        };
        assert_eq!(
            (
                SNAPSHOT.samples,
                SNAPSHOT.prefix_run,
                SNAPSHOT.current_run,
                SNAPSHOT.longest_run
            ),
            (3, 3, 3, 3)
        );
    }
}
