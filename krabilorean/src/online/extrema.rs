//! Constant-state cumulative minimum, maximum, and fused extrema.

use core::marker::PhantomData;

use crate::policy::OverflowPolicy;
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// A defined cumulative minimum result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimumSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Smallest accepted sample.
    pub value: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative minimum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Minimum<O, S> {
    samples: u32,
    value: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective minimum state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MinimumDelta<S> {
    samples: u32,
    value: S,
    quality: Quality,
}

impl<O, S: SampleValue> Minimum<O, S> {
    /// Creates an empty cumulative minimum.
    pub const fn new() -> Self {
        Self {
            samples: 0,
            value: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }
}

impl<O, S: SampleValue> Default for Minimum<O, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: OverflowPolicy, S: SampleValue> Minimum<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<MinimumSnapshot<S>> {
        (self.samples != 0).then_some(MinimumSnapshot {
            samples: self.samples,
            value: self.value,
            quality: self.quality,
        })
    }

    /// Clears all cumulative state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Exactly appends a newer minimum summary, transactionally.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    #[inline(always)]
    fn prepare_word(&self, sample: S) -> Result<MinimumDelta<S>, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        Ok(MinimumDelta {
            samples: samples.value,
            value: if self.samples == 0 {
                sample
            } else {
                self.value.min(sample)
            },
            quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
        })
    }

    #[inline(always)]
    fn commit_word(&mut self, delta: MinimumDelta<S>) {
        self.samples = delta.samples;
        self.value = delta.value;
        self.quality = delta.quality;
    }

    #[inline(always)]
    fn prepare_merge_word(&self, newer: &Self) -> Result<MinimumDelta<S>, Error> {
        if newer.samples == 0 {
            return Ok(MinimumDelta {
                samples: self.samples,
                value: self.value,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(MinimumDelta {
                samples: newer.samples,
                value: newer.value,
                quality: newer.quality,
            });
        }
        let samples = O::add_u32(self.samples, newer.samples)?;
        Ok(MinimumDelta {
            samples: samples.value,
            value: self.value.min(newer.value),
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
            ),
        })
    }
}

/// A defined cumulative maximum result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaximumSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Largest accepted sample.
    pub value: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Maximum<O, S> {
    samples: u32,
    value: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective maximum state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaximumDelta<S> {
    samples: u32,
    value: S,
    quality: Quality,
}

impl<O, S: SampleValue> Maximum<O, S> {
    /// Creates an empty cumulative maximum.
    pub const fn new() -> Self {
        Self {
            samples: 0,
            value: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }
}

impl<O, S: SampleValue> Default for Maximum<O, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: OverflowPolicy, S: SampleValue> Maximum<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<MaximumSnapshot<S>> {
        (self.samples != 0).then_some(MaximumSnapshot {
            samples: self.samples,
            value: self.value,
            quality: self.quality,
        })
    }

    /// Clears all cumulative state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Exactly appends a newer maximum summary, transactionally.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    #[inline(always)]
    fn prepare_word(&self, sample: S) -> Result<MaximumDelta<S>, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        Ok(MaximumDelta {
            samples: samples.value,
            value: if self.samples == 0 {
                sample
            } else {
                self.value.max(sample)
            },
            quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
        })
    }

    #[inline(always)]
    fn commit_word(&mut self, delta: MaximumDelta<S>) {
        self.samples = delta.samples;
        self.value = delta.value;
        self.quality = delta.quality;
    }

    #[inline(always)]
    fn prepare_merge_word(&self, newer: &Self) -> Result<MaximumDelta<S>, Error> {
        if newer.samples == 0 {
            return Ok(MaximumDelta {
                samples: self.samples,
                value: self.value,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(MaximumDelta {
                samples: newer.samples,
                value: newer.value,
                quality: newer.quality,
            });
        }
        let samples = O::add_u32(self.samples, newer.samples)?;
        Ok(MaximumDelta {
            samples: samples.value,
            value: self.value.max(newer.value),
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
            ),
        })
    }
}

/// A defined cumulative extrema result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtremaSnapshot<S> {
    /// Accepted sample count.
    pub samples: u32,
    /// Smallest accepted sample.
    pub minimum: S,
    /// Largest accepted sample.
    pub maximum: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative minimum and maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Extrema<O, S> {
    samples: u32,
    minimum: S,
    maximum: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective extrema state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExtremaDelta<S> {
    samples: u32,
    minimum: S,
    maximum: S,
    quality: Quality,
}

impl<O, S: SampleValue> Extrema<O, S> {
    /// Creates empty cumulative extrema.
    pub const fn new() -> Self {
        Self {
            samples: 0,
            minimum: S::ZERO,
            maximum: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }
}

impl<O, S: SampleValue> Default for Extrema<O, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: OverflowPolicy, S: SampleValue> Extrema<O, S> {
    /// Accepts one sample transactionally.
    pub fn update(&mut self, sample: S) -> Result<(), Error> {
        let delta = self.prepare_word(sample)?;
        self.commit_word(delta);
        Ok(())
    }

    /// Returns `None` until the first sample is accepted.
    pub fn snapshot(&self) -> Option<ExtremaSnapshot<S>> {
        (self.samples != 0).then_some(ExtremaSnapshot {
            samples: self.samples,
            minimum: self.minimum,
            maximum: self.maximum,
            quality: self.quality,
        })
    }

    /// Clears all cumulative state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Exactly appends a newer extrema summary, transactionally.
    pub fn merge(&mut self, newer: &Self) -> Result<(), Error> {
        let delta = self.prepare_merge_word(newer)?;
        self.commit_word(delta);
        Ok(())
    }

    #[inline(always)]
    fn prepare_word(&self, sample: S) -> Result<ExtremaDelta<S>, Error> {
        let samples = O::add_u32(self.samples, 1)?;
        let (minimum, maximum) = if self.samples == 0 {
            (sample, sample)
        } else {
            (self.minimum.min(sample), self.maximum.max(sample))
        };
        Ok(ExtremaDelta {
            samples: samples.value,
            minimum,
            maximum,
            quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
        })
    }

    #[inline(always)]
    fn commit_word(&mut self, delta: ExtremaDelta<S>) {
        self.samples = delta.samples;
        self.minimum = delta.minimum;
        self.maximum = delta.maximum;
        self.quality = delta.quality;
    }

    #[inline(always)]
    fn prepare_merge_word(&self, newer: &Self) -> Result<ExtremaDelta<S>, Error> {
        if newer.samples == 0 {
            return Ok(ExtremaDelta {
                samples: self.samples,
                minimum: self.minimum,
                maximum: self.maximum,
                quality: self.quality,
            });
        }
        if self.samples == 0 {
            return Ok(ExtremaDelta {
                samples: newer.samples,
                minimum: newer.minimum,
                maximum: newer.maximum,
                quality: newer.quality,
            });
        }
        let samples = O::add_u32(self.samples, newer.samples)?;
        Ok(ExtremaDelta {
            samples: samples.value,
            minimum: self.minimum.min(newer.minimum),
            maximum: self.maximum.max(newer.maximum),
            quality: Quality::from_flags(
                false,
                self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use super::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
    use crate::policy::{Checked, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn check_lifecycle_and_merges<S: SampleWord>(a: S, b: S, c: S) {
        let mut min = Minimum::<Checked, S>::default();
        let mut max = Maximum::<Checked, S>::default();
        let mut ext = Extrema::<Checked, S>::default();
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty_min = Minimum::<Checked, S>::new();
        let empty_max = Maximum::<Checked, S>::new();
        let empty_ext = Extrema::<Checked, S>::new();
        min.merge(&empty_min).expect("empty merge");
        max.merge(&empty_max).expect("empty merge");
        ext.merge(&empty_ext).expect("empty merge");
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);

        // Merging a non-empty summary into an empty state adopts the newer state.
        let mut right_min = Minimum::<Checked, S>::new();
        let mut right_max = Maximum::<Checked, S>::new();
        let mut right_ext = Extrema::<Checked, S>::new();
        right_min.update(b).expect("update");
        right_max.update(b).expect("update");
        right_ext.update(b).expect("update");

        min.merge(&right_min).expect("merge into empty");
        max.merge(&right_max).expect("merge into empty");
        ext.merge(&right_ext).expect("merge into empty");
        assert_eq!(
            min.snapshot(),
            Some(MinimumSnapshot {
                samples: 1,
                value: b,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            max.snapshot(),
            Some(MaximumSnapshot {
                samples: 1,
                value: b,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            ext.snapshot(),
            Some(ExtremaSnapshot {
                samples: 1,
                minimum: b,
                maximum: b,
                quality: Quality::Complete,
            })
        );

        // Merging an empty newer summary into a non-empty state preserves it.
        min.merge(&empty_min).expect("merge empty newer");
        max.merge(&empty_max).expect("merge empty newer");
        ext.merge(&empty_ext).expect("merge empty newer");
        assert_eq!(min.snapshot().map(|s| s.samples), Some(1));
        assert_eq!(max.snapshot().map(|s| s.samples), Some(1));
        assert_eq!(ext.snapshot().map(|s| s.samples), Some(1));

        // Additional updates and non-empty merges combine exactly.
        min.update(a).expect("update a");
        max.update(a).expect("update a");
        ext.update(a).expect("update a");

        let mut tail_min = Minimum::<Checked, S>::new();
        let mut tail_max = Maximum::<Checked, S>::new();
        let mut tail_ext = Extrema::<Checked, S>::new();
        tail_min.update(c).expect("update c");
        tail_max.update(c).expect("update c");
        tail_ext.update(c).expect("update c");

        min.merge(&tail_min).expect("merge non-empty");
        max.merge(&tail_max).expect("merge non-empty");
        ext.merge(&tail_ext).expect("merge non-empty");

        let expected_min = a.min(b).min(c);
        let expected_max = a.max(b).max(c);
        assert_eq!(
            min.snapshot(),
            Some(MinimumSnapshot {
                samples: 3,
                value: expected_min,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            max.snapshot(),
            Some(MaximumSnapshot {
                samples: 3,
                value: expected_max,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            ext.snapshot(),
            Some(ExtremaSnapshot {
                samples: 3,
                minimum: expected_min,
                maximum: expected_max,
                quality: Quality::Complete,
            })
        );

        min.reset();
        max.reset();
        ext.reset();
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_word() {
        check_lifecycle_and_merges::<i8>(-40, 7, 99);
        check_lifecycle_and_merges::<i16>(-400, 7, 990);
        check_lifecycle_and_merges::<i32>(-40_000, 7, 99_000);
        check_lifecycle_and_merges::<u8>(4, 70, 19);
        check_lifecycle_and_merges::<u16>(40, 700, 190);
        check_lifecycle_and_merges::<u32>(400, 70_000, 1_900);
    }

    fn every_width_fails_transactionally<S: SampleWord>(low: S, high: S) {
        let mut extrema = Extrema::<Checked, S> {
            samples: u32::MAX,
            minimum: low,
            maximum: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = extrema;
        assert_eq!(extrema.update(S::MIN), Err(Error::Overflow));
        assert_eq!(extrema.update(S::MAX), Err(Error::Overflow));
        assert_eq!(extrema, before);
        let mut other = Extrema::<Checked, S>::new();
        other.update(S::MIN).expect("representable");
        assert_eq!(extrema.merge(&other), Err(Error::Overflow));
        assert_eq!(extrema, before);

        let mut minimum = Minimum::<Checked, S> {
            samples: u32::MAX,
            value: low,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_minimum = minimum;
        assert_eq!(minimum.update(S::MIN), Err(Error::Overflow));
        assert_eq!(minimum, before_minimum);
        let mut other_min = Minimum::<Checked, S>::new();
        other_min.update(S::MIN).expect("representable");
        assert_eq!(minimum.merge(&other_min), Err(Error::Overflow));
        assert_eq!(minimum, before_minimum);

        let mut maximum = Maximum::<Checked, S> {
            samples: u32::MAX,
            value: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_maximum = maximum;
        assert_eq!(maximum.update(S::MAX), Err(Error::Overflow));
        assert_eq!(maximum, before_maximum);
        let mut other_max = Maximum::<Checked, S>::new();
        other_max.update(S::MAX).expect("representable");
        assert_eq!(maximum.merge(&other_max), Err(Error::Overflow));
        assert_eq!(maximum, before_maximum);

        let mut saturating_ext = Extrema::<Saturating, S> {
            samples: u32::MAX,
            minimum: low,
            maximum: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_ext
            .update(S::MAX)
            .expect("saturating update accepts");
        let snapshot = saturating_ext.snapshot().expect("defined");
        assert_eq!(snapshot.samples, u32::MAX);
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.maximum, S::MAX);

        let mut tail_ext = Extrema::<Saturating, S>::new();
        tail_ext.update(S::MIN).expect("tail update");
        saturating_ext
            .merge(&tail_ext)
            .expect("saturating merge accepts");
        assert_eq!(
            saturating_ext.snapshot().map(|s| (s.minimum, s.quality)),
            Some((S::MIN, Quality::Saturated))
        );

        let mut saturating_min = Minimum::<Saturating, S> {
            samples: u32::MAX,
            value: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_min.update(low).expect("saturating min");
        assert_eq!(
            saturating_min.snapshot().map(|s| (s.value, s.quality)),
            Some((low, Quality::Saturated))
        );
        let mut tail_min = Minimum::<Saturating, S>::new();
        tail_min.update(S::MIN).expect("tail min");
        saturating_min.merge(&tail_min).expect("saturating merge");
        assert_eq!(
            saturating_min.snapshot().map(|s| (s.value, s.quality)),
            Some((S::MIN, Quality::Saturated))
        );

        let mut saturating_max = Maximum::<Saturating, S> {
            samples: u32::MAX,
            value: low,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_max.update(high).expect("saturating max");
        assert_eq!(
            saturating_max.snapshot().map(|s| (s.value, s.quality)),
            Some((high, Quality::Saturated))
        );
        let mut tail_max = Maximum::<Saturating, S>::new();
        tail_max.update(S::MAX).expect("tail max");
        saturating_max.merge(&tail_max).expect("saturating merge");
        assert_eq!(
            saturating_max.snapshot().map(|s| (s.value, s.quality)),
            Some((S::MAX, Quality::Saturated))
        );
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        every_width_fails_transactionally::<i8>(-3, 4);
        every_width_fails_transactionally::<i16>(-3, 4);
        every_width_fails_transactionally::<i32>(-3, 4);
        every_width_fails_transactionally::<u8>(3, 9);
        every_width_fails_transactionally::<u16>(3, 9);
        every_width_fails_transactionally::<u32>(3, 9);
    }
}
