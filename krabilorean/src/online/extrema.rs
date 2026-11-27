//! Constant-state cumulative minimum, maximum, and fused extrema.

use core::marker::PhantomData;

use super::{MergeSlot, Slot};
use crate::policy::{CounterWord, OverflowPolicy};
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// A defined cumulative minimum result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimumSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Smallest accepted sample.
    pub value: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative minimum.
///
/// `C` is the counter word of the sample count, and `O` rejects or saturates an update that `C`
/// cannot count. `S` is the sample type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Minimum<O, C, S> {
    samples: C,
    value: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective minimum state.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimumDelta<C, S> {
    samples: C,
    value: S,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleValue> Minimum<O, C, S> {
    /// Creates an empty cumulative minimum.
    pub const fn new() -> Self {
        Self {
            samples: C::ZERO,
            value: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Clears all cumulative state.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
}

impl<O, C: CounterWord, S: SampleValue> Default for Minimum<O, C, S> {
    fn default() -> Self {
        Self::new()
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: SampleValue> Minimum<O, C, S> {
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
        pub c0nst fn snapshot(&self) -> Option<MinimumSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.non_zero().is_none() {
                None
            } else {
                Some(MinimumSnapshot {
                    samples: self.samples,
                    value: self.value,
                    quality: self.quality,
                })
            }
        }

        /// Exactly appends a newer minimum summary, transactionally.
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

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_word(&self, sample: S) -> Result<MinimumDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let value = if self.samples.non_zero().is_none()
                || crate::sealed::SampleValue::above(self.value, sample)
            {
                sample
            } else {
                self.value
            };
            Ok(MinimumDelta {
                samples: samples.value,
                value,
                quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
            })
        }

        #[inline(always)]
        const fn commit_word(&mut self, delta: MinimumDelta<C, S>) {
            self.samples = delta.samples;
            self.value = delta.value;
            self.quality = delta.quality;
        }

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<MinimumDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            if newer.samples.non_zero().is_none() {
                return Ok(MinimumDelta {
                    samples: self.samples,
                    value: self.value,
                    quality: self.quality,
                });
            }
            if self.samples.non_zero().is_none() {
                return Ok(MinimumDelta {
                    samples: newer.samples,
                    value: newer.value,
                    quality: newer.quality,
                });
            }
            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let value = if crate::sealed::SampleValue::above(self.value, newer.value) {
                newer.value
            } else {
                self.value
            };
            Ok(MinimumDelta {
                samples: samples.value,
                value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
                ),
            })
        }
    }
}

/// A defined cumulative maximum result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaximumSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Largest accepted sample.
    pub value: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative maximum.
///
/// `C` is the counter word of the sample count, and `O` rejects or saturates an update that `C`
/// cannot count. `S` is the sample type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Maximum<O, C, S> {
    samples: C,
    value: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective maximum state.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaximumDelta<C, S> {
    samples: C,
    value: S,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleValue> Maximum<O, C, S> {
    /// Creates an empty cumulative maximum.
    pub const fn new() -> Self {
        Self {
            samples: C::ZERO,
            value: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Clears all cumulative state.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
}

impl<O, C: CounterWord, S: SampleValue> Default for Maximum<O, C, S> {
    fn default() -> Self {
        Self::new()
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: SampleValue> Maximum<O, C, S> {
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
        pub c0nst fn snapshot(&self) -> Option<MaximumSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.non_zero().is_none() {
                None
            } else {
                Some(MaximumSnapshot {
                    samples: self.samples,
                    value: self.value,
                    quality: self.quality,
                })
            }
        }

        /// Exactly appends a newer maximum summary, transactionally.
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

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_word(&self, sample: S) -> Result<MaximumDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let value = if self.samples.non_zero().is_none()
                || crate::sealed::SampleValue::above(sample, self.value)
            {
                sample
            } else {
                self.value
            };
            Ok(MaximumDelta {
                samples: samples.value,
                value,
                quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
            })
        }

        #[inline(always)]
        const fn commit_word(&mut self, delta: MaximumDelta<C, S>) {
            self.samples = delta.samples;
            self.value = delta.value;
            self.quality = delta.quality;
        }

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<MaximumDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            if newer.samples.non_zero().is_none() {
                return Ok(MaximumDelta {
                    samples: self.samples,
                    value: self.value,
                    quality: self.quality,
                });
            }
            if self.samples.non_zero().is_none() {
                return Ok(MaximumDelta {
                    samples: newer.samples,
                    value: newer.value,
                    quality: newer.quality,
                });
            }
            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let value = if crate::sealed::SampleValue::above(newer.value, self.value) {
                newer.value
            } else {
                self.value
            };
            Ok(MaximumDelta {
                samples: samples.value,
                value,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
                ),
            })
        }
    }
}

/// A defined cumulative extrema result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtremaSnapshot<C, S> {
    /// Accepted sample count.
    pub samples: C,
    /// Smallest accepted sample.
    pub minimum: S,
    /// Largest accepted sample.
    pub maximum: S,
    /// Saturation quality.
    pub quality: Quality,
}

/// Constant-state cumulative minimum and maximum.
///
/// `C` is the counter word of the sample count, and `O` rejects or saturates an update that `C`
/// cannot count. `S` is the sample type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Extrema<O, C, S> {
    samples: C,
    minimum: S,
    maximum: S,
    quality: Quality,
    policy: PhantomData<fn() -> O>,
}

/// Prospective extrema state.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtremaDelta<C, S> {
    samples: C,
    minimum: S,
    maximum: S,
    quality: Quality,
}

impl<O, C: CounterWord, S: SampleValue> Extrema<O, C, S> {
    /// Creates empty cumulative extrema.
    pub const fn new() -> Self {
        Self {
            samples: C::ZERO,
            minimum: S::ZERO,
            maximum: S::ZERO,
            quality: Quality::Complete,
            policy: PhantomData,
        }
    }

    /// Clears all cumulative state.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
}

impl<O, C: CounterWord, S: SampleValue> Default for Extrema<O, C, S> {
    fn default() -> Self {
        Self::new()
    }
}

c0nst::c0nst! {
    impl<O: OverflowPolicy, C: CounterWord, S: SampleValue> Extrema<O, C, S> {
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
        pub c0nst fn snapshot(&self) -> Option<ExtremaSnapshot<C, S>>
        where
            C: [c0nst] CounterWord,
        {
            if self.samples.non_zero().is_none() {
                None
            } else {
                Some(ExtremaSnapshot {
                    samples: self.samples,
                    minimum: self.minimum,
                    maximum: self.maximum,
                    quality: self.quality,
                })
            }
        }

        /// Exactly appends a newer extrema summary, transactionally.
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

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_word(&self, sample: S) -> Result<ExtremaDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let (minimum, maximum) = if self.samples.non_zero().is_none() {
                (sample, sample)
            } else {
                (
                    if crate::sealed::SampleValue::above(self.minimum, sample) {
                        sample
                    } else {
                        self.minimum
                    },
                    if crate::sealed::SampleValue::above(sample, self.maximum) {
                        sample
                    } else {
                        self.maximum
                    },
                )
            };
            Ok(ExtremaDelta {
                samples: samples.value,
                minimum,
                maximum,
                quality: Quality::from_flags(false, self.quality.is_saturated() || samples.saturated),
            })
        }

        #[inline(always)]
        const fn commit_word(&mut self, delta: ExtremaDelta<C, S>) {
            self.samples = delta.samples;
            self.minimum = delta.minimum;
            self.maximum = delta.maximum;
            self.quality = delta.quality;
        }

        #[inline(always)]
        #[allow(clippy::question_mark)]
        c0nst fn prepare_merge_word(&self, newer: &Self) -> Result<ExtremaDelta<C, S>, Error>
        where
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleValue,
        {
            if newer.samples.non_zero().is_none() {
                return Ok(ExtremaDelta {
                    samples: self.samples,
                    minimum: self.minimum,
                    maximum: self.maximum,
                    quality: self.quality,
                });
            }
            if self.samples.non_zero().is_none() {
                return Ok(ExtremaDelta {
                    samples: newer.samples,
                    minimum: newer.minimum,
                    maximum: newer.maximum,
                    quality: newer.quality,
                });
            }
            let samples = match C::add::<O>(self.samples, newer.samples) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            let minimum = if crate::sealed::SampleValue::above(self.minimum, newer.minimum) {
                newer.minimum
            } else {
                self.minimum
            };
            let maximum = if crate::sealed::SampleValue::above(newer.maximum, self.maximum) {
                newer.maximum
            } else {
                self.maximum
            };
            Ok(ExtremaDelta {
                samples: samples.value,
                minimum,
                maximum,
                quality: Quality::from_flags(
                    false,
                    self.quality.is_saturated() || newer.quality.is_saturated() || samples.saturated,
                ),
            })
        }
    }

    c0nst impl<O, C, S> crate::sealed::OnlineSlot for Minimum<O, C, S> {}

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue> Slot<S>
        for Minimum<O, C, S>
    {
        type Delta = MinimumDelta<C, S>;
        type Snapshot<'a>
            = Option<MinimumSnapshot<C, S>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare_word(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit_word(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            Minimum::snapshot(self)
        }

        fn reset(&mut self) {
            Minimum::reset(self);
        }
    }

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue>
        MergeSlot<S> for Minimum<O, C, S>
    {
        type MergeDelta = MinimumDelta<C, S>;

        fn prepare_merge(&self, newer: &Self) -> Result<Self::MergeDelta, Error> {
            self.prepare_merge_word(newer)
        }

        fn commit_merge(&mut self, delta: Self::MergeDelta) {
            self.commit_word(delta);
        }
    }

    c0nst impl<O, C, S> crate::sealed::OnlineSlot for Maximum<O, C, S> {}

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue> Slot<S>
        for Maximum<O, C, S>
    {
        type Delta = MaximumDelta<C, S>;
        type Snapshot<'a>
            = Option<MaximumSnapshot<C, S>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare_word(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit_word(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            Maximum::snapshot(self)
        }

        fn reset(&mut self) {
            Maximum::reset(self);
        }
    }

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue>
        MergeSlot<S> for Maximum<O, C, S>
    {
        type MergeDelta = MaximumDelta<C, S>;

        fn prepare_merge(&self, newer: &Self) -> Result<Self::MergeDelta, Error> {
            self.prepare_merge_word(newer)
        }

        fn commit_merge(&mut self, delta: Self::MergeDelta) {
            self.commit_word(delta);
        }
    }

    c0nst impl<O, C, S> crate::sealed::OnlineSlot for Extrema<O, C, S> {}

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue> Slot<S>
        for Extrema<O, C, S>
    {
        type Delta = ExtremaDelta<C, S>;
        type Snapshot<'a>
            = Option<ExtremaSnapshot<C, S>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare_word(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit_word(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            Extrema::snapshot(self)
        }

        fn reset(&mut self) {
            Extrema::reset(self);
        }
    }

    c0nst impl<O: [c0nst] OverflowPolicy, C: [c0nst] CounterWord, S: [c0nst] SampleValue>
        MergeSlot<S> for Extrema<O, C, S>
    {
        type MergeDelta = ExtremaDelta<C, S>;

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
    use core::mem::size_of;

    use super::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    fn check_lifecycle_and_merges<C: CounterWord, S: SampleWord>(a: S, b: S, c: S) {
        let mut min = Minimum::<Checked, C, S>::default();
        let mut max = Maximum::<Checked, C, S>::default();
        let mut ext = Extrema::<Checked, C, S>::default();
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);

        // Merging an empty newer summary into an empty state stays empty.
        let empty_min = Minimum::<Checked, C, S>::new();
        let empty_max = Maximum::<Checked, C, S>::new();
        let empty_ext = Extrema::<Checked, C, S>::new();
        min.merge(&empty_min).expect("empty merge");
        max.merge(&empty_max).expect("empty merge");
        ext.merge(&empty_ext).expect("empty merge");
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);

        // Merging a non-empty summary into an empty state adopts the newer state.
        let mut right_min = Minimum::<Checked, C, S>::new();
        let mut right_max = Maximum::<Checked, C, S>::new();
        let mut right_ext = Extrema::<Checked, C, S>::new();
        right_min.update(b).expect("update");
        right_max.update(b).expect("update");
        right_ext.update(b).expect("update");

        min.merge(&right_min).expect("merge into empty");
        max.merge(&right_max).expect("merge into empty");
        ext.merge(&right_ext).expect("merge into empty");
        assert_eq!(
            min.snapshot(),
            Some(MinimumSnapshot {
                samples: C::ONE,
                value: b,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            max.snapshot(),
            Some(MaximumSnapshot {
                samples: C::ONE,
                value: b,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            ext.snapshot(),
            Some(ExtremaSnapshot {
                samples: C::ONE,
                minimum: b,
                maximum: b,
                quality: Quality::Complete,
            })
        );

        // Merging an empty newer summary into a non-empty state preserves it.
        min.merge(&empty_min).expect("merge empty newer");
        max.merge(&empty_max).expect("merge empty newer");
        ext.merge(&empty_ext).expect("merge empty newer");
        assert_eq!(min.snapshot().map(|s| s.samples), Some(C::ONE));
        assert_eq!(max.snapshot().map(|s| s.samples), Some(C::ONE));
        assert_eq!(ext.snapshot().map(|s| s.samples), Some(C::ONE));

        // Additional updates and non-empty merges combine exactly.
        min.update(a).expect("update a");
        max.update(a).expect("update a");
        ext.update(a).expect("update a");

        let mut tail_min = Minimum::<Checked, C, S>::new();
        let mut tail_max = Maximum::<Checked, C, S>::new();
        let mut tail_ext = Extrema::<Checked, C, S>::new();
        tail_min.update(c).expect("update c");
        tail_max.update(c).expect("update c");
        tail_ext.update(c).expect("update c");

        min.merge(&tail_min).expect("merge non-empty");
        max.merge(&tail_max).expect("merge non-empty");
        ext.merge(&tail_ext).expect("merge non-empty");

        let expected_min = a.min(b).min(c);
        let expected_max = a.max(b).max(c);
        let three = C::from_u64(3).expect("three fits");
        assert_eq!(
            min.snapshot(),
            Some(MinimumSnapshot {
                samples: three,
                value: expected_min,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            max.snapshot(),
            Some(MaximumSnapshot {
                samples: three,
                value: expected_max,
                quality: Quality::Complete,
            })
        );
        assert_eq!(
            ext.snapshot(),
            Some(ExtremaSnapshot {
                samples: three,
                minimum: expected_min,
                maximum: expected_max,
                quality: Quality::Complete,
            })
        );

        // Merge where `self` holds the more extreme value (both branches of `above` in merge).
        let mut mid_min = Minimum::<Checked, C, S>::new();
        let mut mid_max = Maximum::<Checked, C, S>::new();
        let mut mid_ext = Extrema::<Checked, C, S>::new();
        mid_min.update(b).expect("update b");
        mid_max.update(b).expect("update b");
        mid_ext.update(b).expect("update b");
        min.merge(&mid_min).expect("merge mid");
        max.merge(&mid_max).expect("merge mid");
        ext.merge(&mid_ext).expect("merge mid");
        assert_eq!(min.snapshot().map(|s| s.value), Some(expected_min));
        assert_eq!(max.snapshot().map(|s| s.value), Some(expected_max));
        assert_eq!(
            ext.snapshot().map(|s| (s.minimum, s.maximum)),
            Some((expected_min, expected_max))
        );

        min.reset();
        max.reset();
        ext.reset();
        assert_eq!(min.snapshot(), None);
        assert_eq!(max.snapshot(), None);
        assert_eq!(ext.snapshot(), None);
    }

    fn lifecycle_for_counter<C: CounterWord>() {
        check_lifecycle_and_merges::<C, i8>(-40, 7, 99);
        check_lifecycle_and_merges::<C, i16>(-400, 7, 990);
        check_lifecycle_and_merges::<C, i32>(-40_000, 7, 99_000);
        check_lifecycle_and_merges::<C, u8>(4, 70, 19);
        check_lifecycle_and_merges::<C, u16>(40, 700, 190);
        check_lifecycle_and_merges::<C, u32>(400, 70_000, 1_900);
    }

    #[test]
    fn lifecycle_and_merges_hold_at_every_counter_and_sample_word() {
        lifecycle_for_counter::<u8>();
        lifecycle_for_counter::<u16>();
        lifecycle_for_counter::<u32>();
        lifecycle_for_counter::<u64>();
    }

    fn every_width_fails_transactionally<C: CounterWord, S: SampleWord>(low: S, high: S) {
        let mut extrema = Extrema::<Checked, C, S> {
            samples: C::MAX,
            minimum: low,
            maximum: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before = extrema;
        assert_eq!(extrema.update(S::MIN), Err(Error::Overflow));
        assert_eq!(extrema.update(S::MAX), Err(Error::Overflow));
        assert_eq!(extrema, before);
        let mut other = Extrema::<Checked, C, S>::new();
        other.update(S::MIN).expect("representable");
        assert_eq!(extrema.merge(&other), Err(Error::Overflow));
        assert_eq!(extrema, before);

        let mut minimum = Minimum::<Checked, C, S> {
            samples: C::MAX,
            value: low,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_minimum = minimum;
        assert_eq!(minimum.update(S::MIN), Err(Error::Overflow));
        assert_eq!(minimum, before_minimum);
        let mut other_min = Minimum::<Checked, C, S>::new();
        other_min.update(S::MIN).expect("representable");
        assert_eq!(minimum.merge(&other_min), Err(Error::Overflow));
        assert_eq!(minimum, before_minimum);

        let mut maximum = Maximum::<Checked, C, S> {
            samples: C::MAX,
            value: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        let before_maximum = maximum;
        assert_eq!(maximum.update(S::MAX), Err(Error::Overflow));
        assert_eq!(maximum, before_maximum);
        let mut other_max = Maximum::<Checked, C, S>::new();
        other_max.update(S::MAX).expect("representable");
        assert_eq!(maximum.merge(&other_max), Err(Error::Overflow));
        assert_eq!(maximum, before_maximum);

        let mut saturating_ext = Extrema::<Saturating, C, S> {
            samples: C::MAX,
            minimum: low,
            maximum: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_ext
            .update(S::MAX)
            .expect("saturating update accepts");
        let snapshot = saturating_ext.snapshot().expect("defined");
        assert_eq!(snapshot.samples, C::MAX);
        assert_eq!(snapshot.quality, Quality::Saturated);
        assert_eq!(snapshot.maximum, S::MAX);

        let mut tail_ext = Extrema::<Saturating, C, S>::new();
        tail_ext.update(S::MIN).expect("tail update");
        saturating_ext
            .merge(&tail_ext)
            .expect("saturating merge accepts");
        assert_eq!(
            saturating_ext.snapshot().map(|s| (s.minimum, s.quality)),
            Some((S::MIN, Quality::Saturated))
        );

        let mut saturating_min = Minimum::<Saturating, C, S> {
            samples: C::MAX,
            value: high,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_min.update(low).expect("saturating min");
        assert_eq!(
            saturating_min.snapshot().map(|s| (s.value, s.quality)),
            Some((low, Quality::Saturated))
        );
        let mut tail_min = Minimum::<Saturating, C, S>::new();
        tail_min.update(S::MIN).expect("tail min");
        saturating_min.merge(&tail_min).expect("saturating merge");
        assert_eq!(
            saturating_min.snapshot().map(|s| (s.value, s.quality)),
            Some((S::MIN, Quality::Saturated))
        );

        let mut saturating_max = Maximum::<Saturating, C, S> {
            samples: C::MAX,
            value: low,
            quality: Quality::Complete,
            policy: PhantomData,
        };
        saturating_max.update(high).expect("saturating max");
        assert_eq!(
            saturating_max.snapshot().map(|s| (s.value, s.quality)),
            Some((high, Quality::Saturated))
        );
        let mut tail_max = Maximum::<Saturating, C, S>::new();
        tail_max.update(S::MAX).expect("tail max");
        saturating_max.merge(&tail_max).expect("saturating merge");
        assert_eq!(
            saturating_max.snapshot().map(|s| (s.value, s.quality)),
            Some((S::MAX, Quality::Saturated))
        );
    }

    fn checked_failure_for_counter<C: CounterWord>() {
        every_width_fails_transactionally::<C, i8>(-3, 4);
        every_width_fails_transactionally::<C, i16>(-3, 4);
        every_width_fails_transactionally::<C, i32>(-3, 4);
        every_width_fails_transactionally::<C, u8>(3, 9);
        every_width_fails_transactionally::<C, u16>(3, 9);
        every_width_fails_transactionally::<C, u32>(3, 9);
    }

    #[test]
    fn checked_failure_is_transactional_at_every_counter_and_sample_word() {
        checked_failure_for_counter::<u8>();
        checked_failure_for_counter::<u16>();
        checked_failure_for_counter::<u32>();
        checked_failure_for_counter::<u64>();
    }

    #[test]
    fn narrower_counters_shrink_state_size() {
        assert!(size_of::<Extrema<Checked, u8, i8>>() <= size_of::<Extrema<Checked, u16, i8>>());
        assert!(size_of::<Extrema<Checked, u16, i16>>() < size_of::<Extrema<Checked, u32, i16>>());
        assert!(size_of::<Extrema<Checked, u32, i16>>() < size_of::<Extrema<Checked, u64, i16>>());
        assert!(size_of::<Minimum<Checked, u16, i16>>() < size_of::<Minimum<Checked, u32, i16>>());
        assert!(size_of::<Maximum<Checked, u16, i16>>() < size_of::<Maximum<Checked, u32, i16>>());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_extrema_are_const_evaluable_on_nightly() {
        const SNAPSHOT: ExtremaSnapshot<u16, i16> = {
            let mut left = Extrema::<Checked, u16, i16>::new();
            assert!(left.update(4).is_ok());
            assert!(left.update(-7).is_ok());
            let mut right = Extrema::<Checked, u16, i16>::new();
            assert!(right.update(12).is_ok());
            assert!(left.merge(&right).is_ok());
            match left.snapshot() {
                Some(s) => s,
                None => unreachable!(),
            }
        };
        assert_eq!(
            (SNAPSHOT.samples, SNAPSHOT.minimum, SNAPSHOT.maximum),
            (3, -7, 12)
        );
    }
}
