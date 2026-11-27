//! Bounded processors that update once for each arriving sample.

mod crossings;
mod ew_variance;
mod ewma;
mod extrema;
mod histogram;
mod moments;
mod power_of_two;
mod runs;
mod successive_difference;

pub use crossings::{ThresholdCrossings, ThresholdCrossingsSnapshot};
pub use ew_variance::{
    EW_VARIANCE_MAX_ALPHA_SHIFT, EwVariance, EwVarianceSnapshot, ew_variance_max_fractional_bits,
};
pub use ewma::{EW_MEAN_MAX_ALPHA_SHIFT, EwMean, EwMeanSnapshot, ew_mean_max_fractional_bits};
pub use extrema::{Extrema, ExtremaSnapshot, Maximum, MaximumSnapshot, Minimum, MinimumSnapshot};
pub use histogram::{HistogramSnapshot, LinearHistogram};
pub use moments::{RawMomentsSnapshot, RunningMoments, raw_moments_hold};
pub use power_of_two::{
    PowerOfTwoHistogram, PowerOfTwoHistogramSnapshot, power_of_two_buckets_are_valid,
};
pub use runs::{ThresholdRuns, ThresholdRunsSnapshot};
pub use successive_difference::{
    SuccessiveDifference, SuccessiveDifferenceSnapshot, successive_difference_holds,
};

use core::marker::PhantomData;

use crate::Error;
use crate::arithmetic::{NearestTiesAway, RoundingPolicy};
use crate::histogram::HistogramBounds;
use crate::policy::{CounterWord, OverflowPolicy};
use crate::sample::{HistogramWord, MomentWord, SampleValue, SampleWord};

c0nst::c0nst! {
    /// Crate-internal capability implemented by every selectable online feature.
    pub(crate) c0nst trait Slot<S> {
        /// A prospective constant-sized update.
        type Delta: Copy;
        /// Snapshot produced by the slot.
        type Snapshot<'a>
        where
            Self: 'a;

        /// Computes an update without changing state.
        fn prepare(&self, sample: S) -> Result<Self::Delta, Error>;
        /// Commits a prepared update without failure.
        fn commit(&mut self, delta: Self::Delta);
        /// Returns the current snapshot.
        fn snapshot(&self) -> Self::Snapshot<'_>;
        /// Returns the slot to its initial state.
        fn reset(&mut self);
    }

    /// Crate-internal bounded concatenation capability for online summaries.
    ///
    /// Implementations represent `self` followed by `newer`; sequence-sensitive slots therefore
    /// retain boundary state. Exact slots reproduce raw-stream concatenation. Quantized
    /// exponentially weighted estimators do not implement this trait because per-sample rounding
    /// makes their sufficient state path-dependent.
    pub(crate) c0nst trait MergeSlot<S>: [c0nst] Slot<S> {
        /// A prospective constant- or capacity-bounded merge.
        type MergeDelta: Copy;

        /// Computes the slot's documented concatenation without changing state.
        fn prepare_merge(&self, newer: &Self) -> Result<Self::MergeDelta, Error>;
        /// Commits a prepared merge without failure.
        fn commit_merge(&mut self, delta: Self::MergeDelta);
    }
}

/// A disabled online capability with zero state and zero output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Disabled;

/// A freshly constructed online profile whose history has never been updated.
///
/// Created by [`OnlineBuilder::build_empty`]. The private field prevents relabeling an updated
/// profile as empty, enabling `const` interval construction without a runtime reset.
///
/// ```compile_fail,E0451
/// use krabilorean::online::{EmptyOnlineProfile, OnlineBuilder};
/// use krabilorean::policy::Checked;
///
/// let mut profile = OnlineBuilder::<i16>::new().extrema::<Checked, u32>().build();
/// profile.update(42).unwrap();
/// // Updated states cannot be relabeled as fresh: the `profile` field is private.
/// let _ = EmptyOnlineProfile { profile };
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmptyOnlineProfile<P: Copy> {
    profile: P,
}

impl<P: Copy> EmptyOnlineProfile<P> {
    /// Releases the configured empty profile for updates or composition.
    pub const fn into_profile(self) -> P {
        self.profile
    }
}

c0nst::c0nst! {
    c0nst impl<S: Copy> Slot<S> for Disabled {
        type Delta = ();
        type Snapshot<'a> = ();

        fn prepare(&self, _sample: S) -> Result<Self::Delta, Error> {
            Ok(())
        }

        fn commit(&mut self, _delta: Self::Delta) {}

        fn snapshot(&self) -> Self::Snapshot<'_> {}

        fn reset(&mut self) {}
    }

    c0nst impl<S: Copy> MergeSlot<S> for Disabled {
        type MergeDelta = ();

        fn prepare_merge(&self, _newer: &Self) -> Result<Self::MergeDelta, Error> {
            Ok(())
        }

        fn commit_merge(&mut self, _delta: Self::MergeDelta) {}
    }
}

/// A consuming builder whose type records which online state is selected.
///
/// `S` is the sample word every selected feature accepts. Each selection method requires the
/// bound its feature needs, such as [`MomentWord`] for raw moments or [`HistogramWord`] for the
/// histograms. Unselected slots remain [`Disabled`] (zero-sized) and compile out of the resulting
/// [`OnlineProcessor`] and [`OnlineSnapshot`].
///
/// ```
/// use krabilorean::histogram::HistogramBounds;
/// use krabilorean::online::OnlineBuilder;
/// use krabilorean::policy::Checked;
///
/// let mut profile = OnlineBuilder::<i32>::new()
///     .extrema::<Checked, u32>()
///     .linear_histogram::<4, Checked, u32>(HistogramBounds::new(-100_000, 99_999)?)?
///     .build();
/// profile.update(-70_000)?;
/// profile.update(65_000)?;
/// let snapshot = profile.snapshot();
/// let extrema = snapshot.extrema.map(|e| (e.minimum, e.maximum));
/// assert_eq!(extrema, Some((-70_000, 65_000)));
/// assert_eq!(snapshot.histogram.counts, &[1, 0, 0, 1]);
/// # Ok::<(), krabilorean::Error>(())
/// ```
///
/// Selection and finishing are `const`-callable on stable Rust. Consuming methods require `Copy`
/// slots so their generic destructors need not run during `const` evaluation; all supported slots
/// satisfy this.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineBuilder<
    S,
    E = Disabled,
    W = Disabled,
    M = Disabled,
    D = Disabled,
    H = Disabled,
    C = Disabled,
    R = Disabled,
> {
    extrema: E,
    ew_mean: W,
    moments: M,
    difference: D,
    histogram: H,
    crossings: C,
    runs: R,
    word: PhantomData<fn() -> S>,
}

/// Builder type after selecting the standard Q16 EWMA.
pub type WithQ16EwMean<const ALPHA_SHIFT: u32, S, E, M, D, H, C, U, O, Count> =
    OnlineBuilder<S, E, EwMean<ALPHA_SHIFT, 16, NearestTiesAway, O, Count, S>, M, D, H, C, U>;

/// Builder type after selecting an explicitly represented EWMA.
pub type WithEwMean<
    const ALPHA_SHIFT: u32,
    const FRACTIONAL_BITS: u32,
    S,
    E,
    M,
    D,
    H,
    C,
    U,
    R,
    O,
    Count,
> = OnlineBuilder<S, E, EwMean<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, Count, S>, M, D, H, C, U>;

/// Builder type after selecting the standard Q16 exponentially weighted variance.
pub type WithQ16EwVariance<const ALPHA_SHIFT: u32, S, E, M, D, H, C, U, O, Count> =
    OnlineBuilder<S, E, EwVariance<ALPHA_SHIFT, 16, NearestTiesAway, O, Count, S>, M, D, H, C, U>;

/// Builder type after selecting an explicitly represented exponentially weighted variance.
pub type WithEwVariance<
    const ALPHA_SHIFT: u32,
    const FRACTIONAL_BITS: u32,
    S,
    E,
    M,
    D,
    H,
    C,
    U,
    R,
    O,
    Count,
> = OnlineBuilder<S, E, EwVariance<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, Count, S>, M, D, H, C, U>;

impl<S: SampleValue> OnlineBuilder<S> {
    /// Starts a builder for the sample word `S` with every feature disabled.
    ///
    /// A selection method is available only where its family supports `S`. Raw moments require a
    /// [`MomentWord`], so a float sample type cannot select them:
    ///
    /// ```compile_fail,E0277
    /// use krabilorean::float::Finite;
    /// use krabilorean::online::OnlineBuilder;
    /// use krabilorean::policy::Checked;
    ///
    /// let _ = OnlineBuilder::<Finite<f32>>::new().moments::<Checked, u32>();
    /// ```
    ///
    /// The same builder selects the comparison families:
    ///
    /// ```
    /// use krabilorean::float::Finite;
    /// use krabilorean::online::OnlineBuilder;
    /// use krabilorean::policy::Checked;
    ///
    /// let _ = OnlineBuilder::<Finite<f32>>::new().extrema::<Checked, u32>().build();
    /// ```
    pub const fn new() -> Self {
        Self {
            extrema: Disabled,
            ew_mean: Disabled,
            moments: Disabled,
            difference: Disabled,
            histogram: Disabled,
            crossings: Disabled,
            runs: Disabled,
            word: PhantomData,
        }
    }
}

impl<S: SampleValue> Default for OnlineBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: SampleValue, W: Copy, M: Copy, D: Copy, H: Copy, C: Copy, R: Copy>
    OnlineBuilder<S, Disabled, W, M, D, H, C, R>
{
    /// Selects only the cumulative minimum with the chosen overflow policy and counter word.
    #[allow(clippy::type_complexity)]
    pub const fn minimum<O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> OnlineBuilder<S, Minimum<O, Count, S>, W, M, D, H, C, R> {
        OnlineBuilder {
            extrema: Minimum::new(),
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }

    /// Selects only the cumulative maximum with the chosen overflow policy and counter word.
    #[allow(clippy::type_complexity)]
    pub const fn maximum<O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> OnlineBuilder<S, Maximum<O, Count, S>, W, M, D, H, C, R> {
        OnlineBuilder {
            extrema: Maximum::new(),
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }

    /// Selects cumulative extrema with the chosen overflow policy and counter word.
    #[allow(clippy::type_complexity)]
    pub const fn extrema<O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> OnlineBuilder<S, Extrema<O, Count, S>, W, M, D, H, C, R> {
        OnlineBuilder {
            extrema: Extrema::new(),
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }
}

impl<S: SampleValue, E: Copy, M: Copy, D: Copy, H: Copy, C: Copy, U: Copy>
    OnlineBuilder<S, E, Disabled, M, D, H, C, U>
{
    /// Selects a Q16 EWMA with `alpha = 1 / 2^ALPHA_SHIFT`, counting samples in `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn ew_mean<const ALPHA_SHIFT: u32, O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> Result<WithQ16EwMean<ALPHA_SHIFT, S, E, M, D, H, C, U, O, Count>, Error>
    where
        S: SampleWord,
    {
        match EwMean::new() {
            Ok(ew_mean) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram: self.histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }

    /// Selects an EWMA with explicit Q format and rounding policy, counting samples in `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn ew_mean_with<
        const ALPHA_SHIFT: u32,
        const FRACTIONAL_BITS: u32,
        R: RoundingPolicy,
        O: OverflowPolicy,
        Count: CounterWord,
    >(
        self,
    ) -> Result<WithEwMean<ALPHA_SHIFT, FRACTIONAL_BITS, S, E, M, D, H, C, U, R, O, Count>, Error>
    where
        S: SampleWord,
    {
        match EwMean::new() {
            Ok(ew_mean) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram: self.histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }

    /// Selects a Q16 exponentially weighted variance with `alpha = 1 / 2^ALPHA_SHIFT`, counting
    /// samples in `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn ew_variance<const ALPHA_SHIFT: u32, O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> Result<WithQ16EwVariance<ALPHA_SHIFT, S, E, M, D, H, C, U, O, Count>, Error>
    where
        S: SampleWord,
    {
        match EwVariance::new() {
            Ok(ew_mean) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram: self.histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }

    /// Selects an exponentially weighted variance with explicit Q format and rounding policy,
    /// counting samples in `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn ew_variance_with<
        const ALPHA_SHIFT: u32,
        const FRACTIONAL_BITS: u32,
        R: RoundingPolicy,
        O: OverflowPolicy,
        Count: CounterWord,
    >(
        self,
    ) -> Result<WithEwVariance<ALPHA_SHIFT, FRACTIONAL_BITS, S, E, M, D, H, C, U, R, O, Count>, Error>
    where
        S: SampleWord,
    {
        match EwVariance::new() {
            Ok(ew_mean) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram: self.histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, D: Copy, H: Copy, C: Copy, R: Copy>
    OnlineBuilder<S, E, W, Disabled, D, H, C, R>
{
    /// Selects cumulative raw moments with the chosen overflow policy, counting samples in
    /// `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn moments<O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> OnlineBuilder<S, E, W, RunningMoments<O, Count, S>, D, H, C, R>
    where
        S: MomentWord,
    {
        OnlineBuilder {
            extrema: self.extrema,
            ew_mean: self.ew_mean,
            moments: RunningMoments::new(),
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, M: Copy, H: Copy, C: Copy, R: Copy>
    OnlineBuilder<S, E, W, M, Disabled, H, C, R>
{
    /// Selects cumulative absolute successive differences, counting them in `Count`.
    #[allow(clippy::type_complexity)]
    pub const fn successive_difference<O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> OnlineBuilder<S, E, W, M, SuccessiveDifference<O, Count, S>, H, C, R>
    where
        S: SampleWord,
    {
        OnlineBuilder {
            extrema: self.extrema,
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: SuccessiveDifference::new(),
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, M: Copy, D: Copy, C: Copy, R: Copy>
    OnlineBuilder<S, E, W, M, D, Disabled, C, R>
{
    /// Selects a fixed-range inclusive linear histogram.
    #[allow(clippy::type_complexity)]
    pub const fn linear_histogram<const BINS: usize, O: OverflowPolicy, Count: CounterWord>(
        self,
        bounds: HistogramBounds<S>,
    ) -> Result<OnlineBuilder<S, E, W, M, D, LinearHistogram<BINS, O, Count, S>, C, R>, Error>
    where
        S: HistogramWord,
    {
        match LinearHistogram::new(bounds) {
            Ok(histogram) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean: self.ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }

    /// Selects the exact division-free power-of-two histogram.
    #[allow(clippy::type_complexity)]
    pub const fn power_of_two_histogram<const N: usize, O: OverflowPolicy, Count: CounterWord>(
        self,
    ) -> Result<OnlineBuilder<S, E, W, M, D, PowerOfTwoHistogram<N, O, Count, S>, C, R>, Error>
    where
        S: HistogramWord,
    {
        match PowerOfTwoHistogram::new() {
            Ok(histogram) => Ok(OnlineBuilder {
                extrema: self.extrema,
                ew_mean: self.ew_mean,
                moments: self.moments,
                difference: self.difference,
                histogram,
                crossings: self.crossings,
                runs: self.runs,
                word: PhantomData,
            }),
            Err(error) => Err(error),
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, M: Copy, D: Copy, H: Copy, R: Copy>
    OnlineBuilder<S, E, W, M, D, H, Disabled, R>
{
    /// Selects cumulative crossings of a runtime threshold.
    #[allow(clippy::type_complexity)]
    pub const fn threshold_crossings<O: OverflowPolicy, Count: CounterWord>(
        self,
        threshold: S,
    ) -> OnlineBuilder<S, E, W, M, D, H, ThresholdCrossings<O, Count, S>, R> {
        OnlineBuilder {
            extrema: self.extrema,
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: ThresholdCrossings::new(threshold),
            runs: self.runs,
            word: PhantomData,
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, M: Copy, D: Copy, H: Copy, C: Copy>
    OnlineBuilder<S, E, W, M, D, H, C, Disabled>
{
    /// Selects cumulative strict runs above a runtime threshold.
    #[allow(clippy::type_complexity)]
    pub const fn threshold_runs<O: OverflowPolicy, Count: CounterWord>(
        self,
        threshold: S,
    ) -> OnlineBuilder<S, E, W, M, D, H, C, ThresholdRuns<O, Count, S>> {
        OnlineBuilder {
            extrema: self.extrema,
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: ThresholdRuns::new(threshold),
            word: PhantomData,
        }
    }
}

impl<S: SampleValue, E: Copy, W: Copy, M: Copy, D: Copy, H: Copy, C: Copy, R: Copy>
    OnlineBuilder<S, E, W, M, D, H, C, R>
{
    /// Finishes the type-selected processor.
    pub const fn build(self) -> OnlineProcessor<S, E, W, M, D, H, C, R> {
        OnlineProcessor {
            extrema: self.extrema,
            ew_mean: self.ew_mean,
            moments: self.moments,
            difference: self.difference,
            histogram: self.histogram,
            crossings: self.crossings,
            runs: self.runs,
            word: PhantomData,
        }
    }

    /// Finishes an empty template suitable for `const` interval-history construction.
    ///
    /// Builder slots are private and can only be selected through fresh constructors.
    #[allow(clippy::type_complexity)]
    pub const fn build_empty(self) -> EmptyOnlineProfile<OnlineProcessor<S, E, W, M, D, H, C, R>> {
        EmptyOnlineProfile {
            profile: self.build(),
        }
    }
}

/// A statically selected group of online processors over the sample word `S`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineProcessor<
    S,
    E = Disabled,
    W = Disabled,
    M = Disabled,
    D = Disabled,
    H = Disabled,
    C = Disabled,
    R = Disabled,
> {
    extrema: E,
    ew_mean: W,
    moments: M,
    difference: D,
    histogram: H,
    crossings: C,
    runs: R,
    word: PhantomData<fn() -> S>,
}

c0nst::c0nst! {
    #[allow(private_bounds)]
    impl<
        S: Copy,
        E: Slot<S>,
        W: Slot<S>,
        M: Slot<S>,
        D: Slot<S>,
        H: Slot<S>,
        C: Slot<S>,
        R: Slot<S>,
    > OnlineProcessor<S, E, W, M, D, H, C, R>
    {
        /// Updates every selected feature transactionally.
        ///
        /// Every prospective update is calculated before any slot is committed, so an error leaves
        /// the entire composed processor bit-identical. The two-phase slot trait is crate-private
        /// so external callers cannot split prepare and commit across slots:
        ///
        /// ```compile_fail,E0603
        /// use krabilorean::online::Slot;
        /// ```
        #[allow(clippy::question_mark, private_bounds)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            E: [c0nst] Slot<S>,
            W: [c0nst] Slot<S>,
            M: [c0nst] Slot<S>,
            D: [c0nst] Slot<S>,
            H: [c0nst] Slot<S>,
            C: [c0nst] Slot<S>,
            R: [c0nst] Slot<S>,
        {
            let extrema = match self.extrema.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let ew_mean = match self.ew_mean.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let moments = match self.moments.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let difference = match self.difference.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let histogram = match self.histogram.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let crossings = match self.crossings.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let runs = match self.runs.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };

            self.extrema.commit(extrema);
            self.ew_mean.commit(ew_mean);
            self.moments.commit(moments);
            self.difference.commit(difference);
            self.histogram.commit(histogram);
            self.crossings.commit(crossings);
            self.runs.commit(runs);
            Ok(())
        }

        /// Borrows or copies each selected feature's result.
        #[allow(private_bounds)]
        pub c0nst fn snapshot(&self) -> OnlineSnapshot<'_, S, E, W, M, D, H, C, R>
        where
            E: [c0nst] Slot<S>,
            W: [c0nst] Slot<S>,
            M: [c0nst] Slot<S>,
            D: [c0nst] Slot<S>,
            H: [c0nst] Slot<S>,
            C: [c0nst] Slot<S>,
            R: [c0nst] Slot<S>,
        {
            OnlineSnapshot {
                extrema: self.extrema.snapshot(),
                ew_mean: self.ew_mean.snapshot(),
                moments: self.moments.snapshot(),
                difference: self.difference.snapshot(),
                histogram: self.histogram.snapshot(),
                crossings: self.crossings.snapshot(),
                runs: self.runs.snapshot(),
            }
        }

        /// Resets every selected feature.
        #[allow(private_bounds)]
        pub c0nst fn reset(&mut self)
        where
            E: [c0nst] Slot<S>,
            W: [c0nst] Slot<S>,
            M: [c0nst] Slot<S>,
            D: [c0nst] Slot<S>,
            H: [c0nst] Slot<S>,
            C: [c0nst] Slot<S>,
            R: [c0nst] Slot<S>,
        {
            self.extrema.reset();
            self.ew_mean.reset();
            self.moments.reset();
            self.difference.reset();
            self.histogram.reset();
            self.crossings.reset();
            self.runs.reset();
        }
    }

    #[allow(private_bounds)]
    impl<
        S: Copy,
        E: MergeSlot<S>,
        W: MergeSlot<S>,
        M: MergeSlot<S>,
        D: MergeSlot<S>,
        H: MergeSlot<S>,
        C: MergeSlot<S>,
        R: MergeSlot<S>,
    > OnlineProcessor<S, E, W, M, D, H, C, R>
    {
        /// Appends a newer compatible summary transactionally.
        ///
        /// Exact slots reproduce raw-stream concatenation. Configuration-bearing slots reject
        /// different bounds or thresholds. Every prospective slot merge is prepared before any
        /// state is committed, so an error leaves the complete processor bit-identical. Quantized
        /// exponentially weighted profiles do not implement this method because per-sample
        /// rounding makes their sufficient state non-mergeable:
        ///
        /// ```compile_fail,E0599
        /// use krabilorean::online::OnlineBuilder;
        /// use krabilorean::policy::Checked;
        ///
        /// let builder = match OnlineBuilder::<i16>::new().ew_mean::<2, Checked, u32>() {
        ///     Ok(value) => value,
        ///     Err(_) => return,
        /// };
        /// let mut left = builder.build();
        /// let right = left;
        /// left.merge(&right);
        /// ```
        #[allow(clippy::question_mark, private_bounds)]
        pub c0nst fn merge(&mut self, newer: &Self) -> Result<(), Error>
        where
            E: [c0nst] MergeSlot<S>,
            W: [c0nst] MergeSlot<S>,
            M: [c0nst] MergeSlot<S>,
            D: [c0nst] MergeSlot<S>,
            H: [c0nst] MergeSlot<S>,
            C: [c0nst] MergeSlot<S>,
            R: [c0nst] MergeSlot<S>,
        {
            let extrema = match self.extrema.prepare_merge(&newer.extrema) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let ew_mean = match self.ew_mean.prepare_merge(&newer.ew_mean) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let moments = match self.moments.prepare_merge(&newer.moments) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let difference = match self.difference.prepare_merge(&newer.difference) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let histogram = match self.histogram.prepare_merge(&newer.histogram) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let crossings = match self.crossings.prepare_merge(&newer.crossings) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            let runs = match self.runs.prepare_merge(&newer.runs) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };

            self.extrema.commit_merge(extrema);
            self.ew_mean.commit_merge(ew_mean);
            self.moments.commit_merge(moments);
            self.difference.commit_merge(difference);
            self.histogram.commit_merge(histogram);
            self.crossings.commit_merge(crossings);
            self.runs.commit_merge(runs);
            Ok(())
        }
    }

    /// A statically selected online profile whose state has a bounded merge operation.
    ///
    /// The trait is sealed to profiles assembled from merge slots.
    pub c0nst trait MergeableOnline: [c0nst] crate::sealed::MergeableOnline + Copy {
        /// The sample word the profile accepts.
        type Sample: Copy;

        /// Accepts one sample transactionally.
        fn update(&mut self, sample: Self::Sample) -> Result<(), Error>;
        /// Returns every selected slot to its configured empty state.
        fn reset(&mut self);
        /// Appends `newer` under every selected slot's merge contract, transactionally.
        fn merge(&mut self, newer: &Self) -> Result<(), Error>;
    }

    c0nst impl<
        S: Copy,
        E: [c0nst] MergeSlot<S> + Copy,
        W: [c0nst] MergeSlot<S> + Copy,
        M: [c0nst] MergeSlot<S> + Copy,
        D: [c0nst] MergeSlot<S> + Copy,
        H: [c0nst] MergeSlot<S> + Copy,
        C: [c0nst] MergeSlot<S> + Copy,
        R: [c0nst] MergeSlot<S> + Copy,
    > crate::sealed::MergeableOnline for OnlineProcessor<S, E, W, M, D, H, C, R>
    {
    }

    #[allow(private_bounds)]
    c0nst impl<
        S: Copy,
        E: [c0nst] MergeSlot<S> + Copy,
        W: [c0nst] MergeSlot<S> + Copy,
        M: [c0nst] MergeSlot<S> + Copy,
        D: [c0nst] MergeSlot<S> + Copy,
        H: [c0nst] MergeSlot<S> + Copy,
        C: [c0nst] MergeSlot<S> + Copy,
        R: [c0nst] MergeSlot<S> + Copy,
    > MergeableOnline for OnlineProcessor<S, E, W, M, D, H, C, R>
    {
        type Sample = S;

        fn update(&mut self, sample: S) -> Result<(), Error> {
            OnlineProcessor::update(self, sample)
        }

        fn reset(&mut self) {
            OnlineProcessor::reset(self);
        }

        fn merge(&mut self, newer: &Self) -> Result<(), Error> {
            OnlineProcessor::merge(self, newer)
        }
    }
}

/// Output of a composed online processor over the sample word `S`.
#[allow(private_bounds, private_interfaces)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineSnapshot<
    'a,
    S,
    E: Slot<S> + 'a = Disabled,
    W: Slot<S> + 'a = Disabled,
    M: Slot<S> + 'a = Disabled,
    D: Slot<S> + 'a = Disabled,
    H: Slot<S> + 'a = Disabled,
    C: Slot<S> + 'a = Disabled,
    R: Slot<S> + 'a = Disabled,
> {
    /// Extrema output, or `()` when disabled.
    pub extrema: E::Snapshot<'a>,
    /// EWMA or EW variance output, or `()` when disabled.
    pub ew_mean: W::Snapshot<'a>,
    /// Raw moments output, or `()` when disabled.
    pub moments: M::Snapshot<'a>,
    /// Successive-difference output, or `()` when disabled.
    pub difference: D::Snapshot<'a>,
    /// Histogram output, or `()` when disabled.
    pub histogram: H::Snapshot<'a>,
    /// Threshold-crossing output, or `()` when disabled.
    pub crossings: C::Snapshot<'a>,
    /// Threshold-run output, or `()` when disabled.
    pub runs: R::Snapshot<'a>,
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::mem::size_of;

    use super::{
        Disabled, Extrema, MergeSlot, MergeableOnline, OnlineBuilder, OnlineProcessor, Slot,
        ThresholdCrossings,
    };
    use crate::Error;
    use crate::arithmetic::NearestTiesAway;
    use crate::histogram::HistogramBounds;
    use crate::policy::{Checked, Saturating};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Failing;

    c0nst::c0nst! {
        c0nst impl<S: Copy> Slot<S> for Failing {
            type Delta = ();
            type Snapshot<'a> = ();

            fn prepare(&self, _sample: S) -> Result<Self::Delta, Error> {
                Err(Error::Overflow)
            }

            fn commit(&mut self, _delta: Self::Delta) {}

            fn snapshot(&self) -> Self::Snapshot<'_> {}

            fn reset(&mut self) {}
        }

        c0nst impl<S: Copy> MergeSlot<S> for Failing {
            type MergeDelta = ();

            fn prepare_merge(&self, _newer: &Self) -> Result<Self::MergeDelta, Error> {
                Err(Error::Overflow)
            }

            fn commit_merge(&mut self, _delta: Self::MergeDelta) {}
        }
    }

    #[test]
    fn unselected_slots_add_no_storage_overhead() {
        type SingleSlot = OnlineProcessor<i16, Extrema<Checked, u32, i16>>;
        assert_eq!(
            size_of::<SingleSlot>(),
            size_of::<Extrema<Checked, u32, i16>>()
        );
        assert_eq!(size_of::<OnlineProcessor<i16>>(), 0);
    }

    #[test]
    fn builder_rejects_invalid_configurations() {
        assert_eq!(
            OnlineBuilder::<i16>::default().ew_mean::<63, Checked, u32>(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            OnlineBuilder::<i16>::new().ew_mean_with::<3, 32, NearestTiesAway, Checked, u32>(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            OnlineBuilder::<i16>::new().ew_variance::<63, Checked, u32>(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            OnlineBuilder::<i16>::new().ew_variance_with::<3, 32, NearestTiesAway, Checked, u32>(),
            Err(Error::InvalidConfiguration)
        );
        let bounds = HistogramBounds::<i16>::new(-10, 10).expect("valid");
        assert_eq!(
            OnlineBuilder::<i16>::new().linear_histogram::<0, Checked, u32>(bounds),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            OnlineBuilder::<i16>::new().power_of_two_histogram::<0, Checked, u32>(),
            Err(Error::InvalidConfiguration)
        );
    }

    #[test]
    fn every_slot_position_fails_transactionally_on_update_and_merge() {
        macro_rules! check_slot_failure {
            (
                ($e1:expr, $w1:expr, $m1:expr, $d1:expr, $h1:expr, $c1:expr, $r1:expr),
                ($e2:expr, $w2:expr, $m2:expr, $d2:expr, $h2:expr, $c2:expr, $r2:expr)
            ) => {{
                let mut processor = OnlineProcessor::<i32, _, _, _, _, _, _, _> {
                    extrema: $e1,
                    ew_mean: $w1,
                    moments: $m1,
                    difference: $d1,
                    histogram: $h1,
                    crossings: $c1,
                    runs: $r1,
                    word: PhantomData,
                };
                let newer = OnlineProcessor::<i32, _, _, _, _, _, _, _> {
                    extrema: $e2,
                    ew_mean: $w2,
                    moments: $m2,
                    difference: $d2,
                    histogram: $h2,
                    crossings: $c2,
                    runs: $r2,
                    word: PhantomData,
                };
                let before = processor;
                assert_eq!(processor.update(100_000), Err(Error::Overflow));
                assert_eq!(processor, before);
                assert_eq!(processor.merge(&newer), Err(Error::Overflow));
                assert_eq!(processor, before);
            }};
        }

        let mut ext_left = Extrema::<Checked, u32, i32>::new();
        assert_eq!(ext_left.update(10), Ok(()));
        let mut ext_right = Extrema::<Checked, u32, i32>::new();
        assert_eq!(ext_right.update(90_000), Ok(()));
        let mut cross_left = ThresholdCrossings::<Checked, u32, i32>::new(70_000);
        assert_eq!(cross_left.update(10), Ok(()));
        let mut cross_right = ThresholdCrossings::<Checked, u32, i32>::new(70_000);
        assert_eq!(cross_right.update(90_000), Ok(()));
        assert_ne!(ext_left, ext_right);
        assert_ne!(cross_left, cross_right);

        let mut failing = Failing;
        <Failing as Slot<i32>>::commit(&mut failing, ());
        <Failing as Slot<i32>>::snapshot(&failing);
        <Failing as Slot<i32>>::reset(&mut failing);
        <Failing as MergeSlot<i32>>::commit_merge(&mut failing, ());
        check_slot_failure!(
            (
                Failing, Disabled, Disabled, Disabled, Disabled, cross_left, Disabled
            ),
            (
                Failing,
                Disabled,
                Disabled,
                Disabled,
                Disabled,
                cross_right,
                Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Failing, Disabled, Disabled, Disabled, cross_left, Disabled
            ),
            (
                ext_right,
                Failing,
                Disabled,
                Disabled,
                Disabled,
                cross_right,
                Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Disabled, Failing, Disabled, Disabled, cross_left, Disabled
            ),
            (
                ext_right,
                Disabled,
                Failing,
                Disabled,
                Disabled,
                cross_right,
                Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Disabled, Disabled, Failing, Disabled, cross_left, Disabled
            ),
            (
                ext_right,
                Disabled,
                Disabled,
                Failing,
                Disabled,
                cross_right,
                Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Disabled, Disabled, Disabled, Failing, cross_left, Disabled
            ),
            (
                ext_right,
                Disabled,
                Disabled,
                Disabled,
                Failing,
                cross_right,
                Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Disabled, Disabled, Disabled, Disabled, Failing, Disabled
            ),
            (
                ext_right, Disabled, Disabled, Disabled, Disabled, Failing, Disabled
            )
        );
        check_slot_failure!(
            (
                ext_left, Disabled, Disabled, Disabled, Disabled, cross_left, Failing
            ),
            (
                ext_right,
                Disabled,
                Disabled,
                Disabled,
                Disabled,
                cross_right,
                Failing
            )
        );
    }

    #[test]
    fn every_family_composes_updates_merges_and_resets() {
        let bounds = HistogramBounds::<i16>::new(-20, 20).expect("valid");
        let template = OnlineBuilder::<i16>::new()
            .extrema::<Checked, u32>()
            .moments::<Checked, u32>()
            .successive_difference::<Checked, u32>()
            .linear_histogram::<4, Checked, u32>(bounds)
            .expect("valid")
            .threshold_crossings::<Checked, u32>(0)
            .threshold_runs::<Checked, u32>(0)
            .build_empty();

        let mut single = template.into_profile();
        let mut left = template.into_profile();
        let mut right = template.into_profile();

        let samples = [-15_i16, 5, 12, -3, 18, 0, -25, 30];
        for &s in &samples {
            MergeableOnline::update(&mut single, s).expect("update");
        }
        for &s in &samples[..4] {
            MergeableOnline::update(&mut left, s).expect("left");
        }
        for &s in &samples[4..] {
            MergeableOnline::update(&mut right, s).expect("right");
        }
        MergeableOnline::merge(&mut left, &right).expect("merge");
        assert_eq!(left, single);
        assert_eq!(left.snapshot(), single.snapshot());

        MergeableOnline::reset(&mut left);
        assert_eq!(left, template.into_profile());
        assert_eq!(left.snapshot().extrema, None);

        // Minimum, maximum, power-of-two histogram, and EW estimators also compose and reset.
        let mut min_pow_ew = OnlineBuilder::<i16>::new()
            .minimum::<Saturating, u16>()
            .ew_mean::<3, Saturating, u16>()
            .expect("valid")
            .power_of_two_histogram::<8, Saturating, u16>()
            .expect("valid")
            .build();
        for &s in &samples {
            min_pow_ew.update(s).expect("update");
        }
        assert_eq!(min_pow_ew.snapshot().extrema.expect("min").value, -25);
        assert!(min_pow_ew.snapshot().ew_mean.is_some());
        assert_eq!(min_pow_ew.snapshot().histogram.samples, 8);
        min_pow_ew.reset();
        assert!(min_pow_ew.snapshot().extrema.is_none());
        assert!(min_pow_ew.snapshot().ew_mean.is_none());
        assert_eq!(min_pow_ew.snapshot().histogram.samples, 0);

        let mut max_var = OnlineBuilder::<i16>::new()
            .maximum::<Checked, u16>()
            .ew_variance::<3, Checked, u16>()
            .expect("valid")
            .build();
        let mut min_pow_merge_left = OnlineBuilder::<i16>::new()
            .minimum::<Checked, u16>()
            .power_of_two_histogram::<8, Checked, u16>()
            .expect("valid")
            .build();
        let mut min_pow_merge_right = min_pow_merge_left;
        let mut max_merge_left = OnlineBuilder::<i16>::new()
            .maximum::<Checked, u16>()
            .build();
        let mut max_merge_right = max_merge_left;
        for &s in &samples[..4] {
            max_var.update(s).expect("update");
            min_pow_merge_left.update(s).expect("update");
            max_merge_left.update(s).expect("update");
        }
        for &s in &samples[4..] {
            max_var.update(s).expect("update");
            min_pow_merge_right.update(s).expect("update");
            max_merge_right.update(s).expect("update");
        }
        min_pow_merge_left
            .merge(&min_pow_merge_right)
            .expect("merge");
        max_merge_left.merge(&max_merge_right).expect("merge");
        assert_eq!(
            min_pow_merge_left.snapshot().extrema.expect("min").value,
            -25
        );
        assert_eq!(min_pow_merge_left.snapshot().histogram.samples, 8);
        assert_eq!(max_merge_left.snapshot().extrema.expect("max").value, 30);
        assert_eq!(max_var.snapshot().extrema.expect("max").value, 30);
        assert!(max_var.snapshot().ew_mean.is_some());
        max_var.reset();
        assert!(max_var.snapshot().extrema.is_none());
        assert!(max_var.snapshot().ew_mean.is_none());

        let mut custom_ew = OnlineBuilder::<i16>::new()
            .ew_mean_with::<2, 8, NearestTiesAway, Checked, u16>()
            .expect("valid")
            .build();
        let mut custom_var = OnlineBuilder::<i16>::new()
            .ew_variance_with::<2, 8, NearestTiesAway, Checked, u16>()
            .expect("valid")
            .build();
        assert!(custom_ew.snapshot().ew_mean.is_none());
        assert!(custom_var.snapshot().ew_mean.is_none());
        custom_ew.update(12).expect("update");
        custom_var.update(12).expect("update");
        let ew_snap = custom_ew.snapshot().ew_mean.expect("defined");
        let var_snap = custom_var.snapshot().ew_mean.expect("defined");
        assert_eq!((ew_snap.mean_q, ew_snap.fractional_bits), (12 << 8, 8));
        assert_eq!(
            (
                var_snap.mean_q,
                var_snap.variance_q,
                var_snap.fractional_bits
            ),
            (12 << 8, 0, 8)
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_online_processor_is_const_evaluable_on_nightly() {
        const SUMMARY: (Option<(i16, i16)>, u32, u32) = {
            let bounds = match HistogramBounds::<i16>::new(-20, 20) {
                Ok(b) => b,
                Err(_) => panic!("valid bounds"),
            };
            let builder = match OnlineBuilder::<i16>::new()
                .extrema::<Checked, u32>()
                .moments::<Checked, u32>()
                .successive_difference::<Checked, u32>()
                .linear_histogram::<4, Checked, u32>(bounds)
            {
                Ok(b) => b,
                Err(_) => panic!("valid histogram"),
            };
            let template = builder
                .threshold_crossings::<Checked, u32>(0)
                .threshold_runs::<Checked, u32>(0)
                .build_empty();
            let mut left = template.into_profile();
            let mut right = template.into_profile();
            assert!(left.update(-10).is_ok());
            assert!(left.update(5).is_ok());
            assert!(right.update(15).is_ok());
            assert!(right.update(-2).is_ok());
            assert!(left.merge(&right).is_ok());
            let snap = left.snapshot();
            let ext = match snap.extrema {
                Some(e) => Some((e.minimum, e.maximum)),
                None => None,
            };
            let crossings = match snap.crossings {
                Some(c) => c.crossings,
                None => 0,
            };
            let runs = match snap.runs {
                Some(r) => r.longest_run,
                None => 0,
            };
            let mut cleared = left;
            cleared.reset();
            assert!(cleared.snapshot().extrema.is_none());
            (ext, crossings, runs)
        };
        assert_eq!(SUMMARY, (Some((-10, 15)), 2, 2));
    }
}
