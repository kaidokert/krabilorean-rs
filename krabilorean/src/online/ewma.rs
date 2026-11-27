//! Constant-state exponentially weighted mean.

use core::marker::PhantomData;

use super::Slot;
use crate::arithmetic::RoundingPolicy;
use crate::policy::{CounterWord, OverflowPolicy};
use crate::sample::SampleWord;
use crate::{Error, Quality};

/// A defined exponentially weighted mean in a declared Q format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EwMeanSnapshot<C: CounterWord> {
    /// Accepted sample count, which may saturate under a saturating policy.
    pub samples: C::NonZero,
    /// Signed fixed-point mean in Q`fractional_bits`.
    pub mean_q: i64,
    /// Number of fractional bits in `mean_q`, in sample units: the configured bits plus the
    /// word's [`SampleWord::FRACTIONAL_BITS`].
    pub fractional_bits: u32,
    /// Power-of-two smoothing shift where `alpha = 1 / 2^alpha_shift`.
    pub alpha_shift: u32,
    /// Approximation and saturation quality.
    pub quality: Quality,
}

const _: () = assert!(
    core::mem::size_of::<Option<EwMeanSnapshot<u8>>>()
        == core::mem::size_of::<EwMeanSnapshot<u8>>()
        && core::mem::size_of::<Option<EwMeanSnapshot<u16>>>()
            == core::mem::size_of::<EwMeanSnapshot<u16>>()
        && core::mem::size_of::<Option<EwMeanSnapshot<u32>>>()
            == core::mem::size_of::<EwMeanSnapshot<u32>>()
        && core::mem::size_of::<Option<EwMeanSnapshot<u64>>>()
            == core::mem::size_of::<EwMeanSnapshot<u64>>()
);

type Policies<R, O, S> = PhantomData<fn() -> (R, O, S)>;

/// Exponentially weighted mean with explicit smoothing, Q format, rounding, and overflow policy.
///
/// `C` is the counter word of the sample count, and `O` rejects or saturates an update that `C`
/// cannot count. `S` is the sample word. The mean is held in `i64` Q format at every sample
/// width; [`EwMean::new`] accepts only formats in which every sample, mean, and innovation
/// difference are proven to fit `i64`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EwMean<const ALPHA_SHIFT: u32, const FRACTIONAL_BITS: u32, R, O, C, S> {
    samples: C,
    mean_q: i64,
    quality: Quality,
    policies: Policies<R, O, S>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EwMeanDelta<C> {
    samples: C,
    mean_q: i64,
    quality: Quality,
}

/// Largest accepted `ALPHA_SHIFT` at every width: the rounding policies divide the `i64` mean
/// delta by `2^ALPHA_SHIFT` for any shift up to 62.
pub const EW_MEAN_MAX_ALPHA_SHIFT: u32 = 62;

/// Shared 31-bit ceiling on `FRACTIONAL_BITS` so 8-, 16-, and 32-bit words share the `i32`
/// `i64` Q-headroom limit (`63 - 32 = 31`) and match [`EwVariance`](super::EwVariance)'s
/// `Q(2 * FRACTIONAL_BITS)` bound.
const FAMILY_MAX_FRACTIONAL_BITS: u32 = 31;

/// Largest accepted `FRACTIONAL_BITS` for a word of `bits` bits: the family cap 31, or the word's
/// `i64` Q headroom `63 - bits` when that is smaller.
///
/// With `bits + FRACTIONAL_BITS <= 63`, every Q-format sample, the mean, and their difference are
/// below `2^(bits + FRACTIONAL_BITS) <= 2^63` in magnitude and fit `i64`. The rounded adjustment
/// moves the mean toward the sample without overshooting it, so the mean stays inside
/// `[S::MIN << FRACTIONAL_BITS, S::MAX << FRACTIONAL_BITS]`.
pub const fn ew_mean_max_fractional_bits(bits: u32) -> u32 {
    let headroom = 63_u32.saturating_sub(bits);
    if headroom < FAMILY_MAX_FRACTIONAL_BITS {
        headroom
    } else {
        FAMILY_MAX_FRACTIONAL_BITS
    }
}

const _: () = assert!(ew_mean_max_fractional_bits(8) == 31);
const _: () = assert!(ew_mean_max_fractional_bits(16) == 31);
const _: () = assert!(ew_mean_max_fractional_bits(32) == 31);
const _: () = assert!(ew_mean_max_fractional_bits(32) == <i32 as SampleWord>::I64_Q_HEADROOM_BITS);
const _: () = assert!(ew_mean_max_fractional_bits(33) < FAMILY_MAX_FRACTIONAL_BITS);

impl<const ALPHA_SHIFT: u32, const FRACTIONAL_BITS: u32, R, O, C: CounterWord, S: SampleWord>
    EwMean<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, C, S>
{
    /// Creates an empty EWMA after validating its shift and Q-format headroom.
    ///
    /// `ALPHA_SHIFT` must be at most [`EW_MEAN_MAX_ALPHA_SHIFT`], and `FRACTIONAL_BITS` at most
    /// [`ew_mean_max_fractional_bits`] of `S::BITS` (31 at every implemented word).
    pub const fn new() -> Result<Self, Error> {
        if ALPHA_SHIFT <= EW_MEAN_MAX_ALPHA_SHIFT
            && FRACTIONAL_BITS <= ew_mean_max_fractional_bits(S::BITS)
        {
            Ok(Self {
                samples: C::ZERO,
                mean_q: 0,
                quality: Quality::Approximate,
                policies: PhantomData,
            })
        } else {
            Err(Error::InvalidConfiguration)
        }
    }

    /// Clears all weighted state.
    pub const fn reset(&mut self) {
        self.samples = C::ZERO;
        self.mean_q = 0;
        self.quality = Quality::Approximate;
    }
}

c0nst::c0nst! {
    impl<
        const ALPHA_SHIFT: u32,
        const FRACTIONAL_BITS: u32,
        R,
        O,
        C: CounterWord,
        S: SampleWord,
    > EwMean<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, C, S>
    {
        /// Returns `None` until the first sample is accepted.
        pub c0nst fn snapshot(&self) -> Option<EwMeanSnapshot<C>>
        where
            C: [c0nst] CounterWord,
        {
            match self.samples.non_zero() {
                Some(samples) => Some(EwMeanSnapshot {
                    samples,
                    mean_q: self.mean_q,
                    fractional_bits: FRACTIONAL_BITS.saturating_add(S::FRACTIONAL_BITS),
                    alpha_shift: ALPHA_SHIFT,
                    quality: self.quality,
                }),
                None => None,
            }
        }
    }

    impl<
        const ALPHA_SHIFT: u32,
        const FRACTIONAL_BITS: u32,
        R: RoundingPolicy,
        O: OverflowPolicy,
        C: CounterWord,
        S: SampleWord,
    > EwMean<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, C, S>
    {
        /// Accepts one sample transactionally in constant time.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            R: [c0nst] RoundingPolicy,
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            let delta = match self.prepare(sample) {
                Ok(delta) => delta,
                Err(error) => return Err(error),
            };
            self.commit(delta);
            Ok(())
        }

        #[allow(clippy::question_mark)]
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<EwMeanDelta<C>, Error>
        where
            R: [c0nst] RoundingPolicy,
            O: [c0nst] OverflowPolicy,
            C: [c0nst] CounterWord,
            S: [c0nst] SampleWord,
        {
            let samples = match C::add::<O>(self.samples, C::ONE) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            // `new` validated `ALPHA_SHIFT <= 62` and `S::BITS + FRACTIONAL_BITS <= 63`, so
            // `sample_q`, `self.mean_q`, their difference, and the updated mean fit `i64`.
            let sample_q = sample.to_i64() << FRACTIONAL_BITS;
            let mean_q = if self.samples.non_zero().is_none() {
                sample_q
            } else {
                let delta = sample_q - self.mean_q;
                let adjustment = R::divide_power_of_two_checked_shift(
                    delta,
                    ALPHA_SHIFT,
                    crate::sealed::Internal::new(),
                );
                self.mean_q + adjustment
            };
            Ok(EwMeanDelta {
                samples: samples.value,
                mean_q,
                quality: Quality::from_flags(true, self.quality.is_saturated() || samples.saturated),
            })
        }

        pub(crate) const fn commit(&mut self, delta: EwMeanDelta<C>) {
            self.samples = delta.samples;
            self.mean_q = delta.mean_q;
            self.quality = delta.quality;
        }
    }

    c0nst impl<
        const ALPHA_SHIFT: u32,
        const FRACTIONAL_BITS: u32,
        R: [c0nst] RoundingPolicy,
        O: [c0nst] OverflowPolicy,
        C: [c0nst] CounterWord,
        S: [c0nst] SampleWord,
    > Slot<S> for EwMean<ALPHA_SHIFT, FRACTIONAL_BITS, R, O, C, S>
    {
        type Delta = EwMeanDelta<C>;
        type Snapshot<'a>
            = Option<EwMeanSnapshot<C>>
        where
            Self: 'a;

        fn prepare(&self, sample: S) -> Result<Self::Delta, Error> {
            self.prepare(sample)
        }

        fn commit(&mut self, delta: Self::Delta) {
            self.commit(delta);
        }

        fn snapshot(&self) -> Self::Snapshot<'_> {
            EwMean::snapshot(self)
        }

        fn reset(&mut self) {
            EwMean::reset(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::mem::size_of;
    use core::num::NonZeroU32;

    use super::{EwMean, ew_mean_max_fractional_bits};
    use crate::arithmetic::NearestTiesAway;
    use crate::policy::{Checked, CounterWord, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    #[test]
    fn q_zero_ties_round_away_from_zero() {
        let mut positive =
            EwMean::<1, 0, NearestTiesAway, Checked, u32, i16>::new().expect("valid format");
        assert_eq!(positive.snapshot(), None);
        positive.update(0).expect("representable");
        positive.update(1).expect("representable");
        let snap = positive.snapshot().expect("defined");
        assert_eq!(snap.samples, NonZeroU32::new(2).expect("nonzero"));
        assert_eq!(snap.mean_q, 1);
        assert_eq!(snap.fractional_bits, 0);
        assert_eq!(snap.alpha_shift, 1);
        assert_eq!(snap.quality, Quality::Approximate);

        let mut negative =
            EwMean::<1, 0, NearestTiesAway, Saturating, u32, i16>::new().expect("valid format");
        negative.update(0).expect("representable");
        negative.update(-1).expect("representable");
        assert_eq!(negative.snapshot().expect("defined").mean_q, -1);
        negative.reset();
        assert_eq!(negative.snapshot(), None);
    }

    fn formats_follow_the_validator<S: SampleWord>() {
        assert_eq!(ew_mean_max_fractional_bits(S::BITS), 31);
        assert!(EwMean::<0, 31, NearestTiesAway, Checked, u32, S>::new().is_ok());
        assert!(EwMean::<62, 0, NearestTiesAway, Checked, u32, S>::new().is_ok());
        assert_eq!(
            EwMean::<4, 32, NearestTiesAway, Checked, u32, S>::new(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            EwMean::<63, 4, NearestTiesAway, Checked, u32, S>::new(),
            Err(Error::InvalidConfiguration)
        );
    }

    #[test]
    fn accepted_formats_follow_the_validator_at_every_width() {
        assert_eq!(ew_mean_max_fractional_bits(40), 23);
        formats_follow_the_validator::<i8>();
        formats_follow_the_validator::<i16>();
        formats_follow_the_validator::<i32>();
        formats_follow_the_validator::<u8>();
        formats_follow_the_validator::<u16>();
        formats_follow_the_validator::<u32>();
    }

    fn ewma_fails_transactionally<C: CounterWord, S: SampleWord>() {
        let full = EwMean::<4, 31, NearestTiesAway, Checked, C, S> {
            samples: C::MAX,
            mean_q: -12_345,
            quality: Quality::Approximate,
            policies: PhantomData,
        };
        let mut mean = full;
        assert_eq!(mean.update(S::MAX), Err(Error::Overflow));
        assert_eq!(mean.update(S::MIN), Err(Error::Overflow));
        assert_eq!(mean, full);

        let mut saturating = EwMean::<4, 31, NearestTiesAway, Saturating, C, S> {
            samples: C::MAX,
            mean_q: 0,
            quality: Quality::Approximate,
            policies: PhantomData,
        };
        saturating.update(S::MAX).expect("saturating accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(C::get(snapshot.samples), C::MAX);
        assert_eq!(snapshot.quality, Quality::ApproximateSaturated);
        saturating
            .update(S::MIN)
            .expect("already saturated update accepts");
        assert_eq!(
            saturating.snapshot().expect("defined").quality,
            Quality::ApproximateSaturated
        );
    }

    fn checked_failure_for_counter<C: CounterWord>() {
        ewma_fails_transactionally::<C, i8>();
        ewma_fails_transactionally::<C, i16>();
        ewma_fails_transactionally::<C, i32>();
        ewma_fails_transactionally::<C, u8>();
        ewma_fails_transactionally::<C, u16>();
        ewma_fails_transactionally::<C, u32>();
    }

    #[test]
    fn checked_failure_is_transactional_at_every_counter_and_sample_word() {
        checked_failure_for_counter::<u8>();
        checked_failure_for_counter::<u16>();
        checked_failure_for_counter::<u32>();
        checked_failure_for_counter::<u64>();
        assert!(
            size_of::<EwMean<4, 8, NearestTiesAway, Checked, u8, i16>>()
                <= size_of::<EwMean<4, 8, NearestTiesAway, Checked, u32, i16>>()
        );
    }

    #[test]
    fn i32_full_swing_at_q31_never_overflows() {
        for index in 0..64 {
            let mut mean =
                EwMean::<1, 31, NearestTiesAway, Checked, u32, i32>::new().expect("valid format");
            let first = if index % 2 == 0 { i32::MIN } else { i32::MAX };
            let second = if index % 2 == 0 { i32::MAX } else { i32::MIN };
            mean.update(first).expect("proven");
            mean.update(second).expect("proven");
            let snap = mean.snapshot().expect("defined");
            assert!(snap.mean_q >= (i32::MIN as i64) << 31);
            assert!(snap.mean_q <= (i32::MAX as i64) << 31);
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_ew_mean_is_const_evaluable_on_nightly() {
        const SNAPSHOT: super::EwMeanSnapshot<u16> = {
            let mut mean = match EwMean::<1, 4, NearestTiesAway, Checked, u16, i16>::new() {
                Ok(m) => m,
                Err(_) => unreachable!(),
            };
            assert!(mean.update(10).is_ok());
            assert!(mean.update(20).is_ok());
            match mean.snapshot() {
                Some(s) => s,
                None => unreachable!(),
            }
        };
        assert_eq!(SNAPSHOT.samples.get(), 2);
        assert_eq!(SNAPSHOT.mean_q, 15 << 4);
    }
}
