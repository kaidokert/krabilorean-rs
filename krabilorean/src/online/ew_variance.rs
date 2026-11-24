//! Constant-state exponentially weighted mean and central variance.

use core::marker::PhantomData;
use core::num::NonZeroU32;

use crate::arithmetic::RoundingPolicy;
use crate::policy::OverflowPolicy;
use crate::sample::SampleWord;
use crate::{Error, Quality};

type Policies<R, O, S> = PhantomData<fn() -> (R, O, S)>;

/// Exponentially weighted mean and central variance in declared fixed-point formats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EwVarianceSnapshot {
    /// Accepted sample count, which may saturate under a saturating policy.
    pub samples: NonZeroU32,
    /// Weighted mean in Q`fractional_bits`.
    pub mean_q: i64,
    /// Non-negative weighted central variance in Q`2 * fractional_bits` squared sample units.
    pub variance_q: u128,
    /// Fractional bits in the mean, in sample units: the configured bits plus the word's
    /// [`SampleWord::FRACTIONAL_BITS`]. The variance has twice this count.
    pub fractional_bits: u32,
    /// Power-of-two smoothing shift where `alpha = 1 / 2^alpha_shift`.
    pub alpha_shift: u32,
    /// Approximation and saturation quality.
    pub quality: Quality,
}

const _: () = assert!(
    core::mem::size_of::<Option<EwVarianceSnapshot>>()
        == core::mem::size_of::<EwVarianceSnapshot>()
);

/// Exponentially weighted mean and variance using the West central-moment recurrence.
///
/// The first sample initializes the mean and zero variance. Later samples update the mean first,
/// then use `(x - old_mean) * (x - new_mean)` as the variance innovation. Both adjustments use
/// the selected power-of-two rounding policy. The quantized recurrence is path-dependent and
/// models an exponentially weighted history.
///
/// The mean is held in `i64` and the variance in `u128` at every sample width;
/// [`EwVariance::new`] accepts only formats in which every intermediate and state value is proven
/// to fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EwVariance<const ALPHA_SHIFT: u32, const FRACTIONAL_BITS: u32, R, O, S> {
    samples: u32,
    mean_q: i64,
    variance_q: u128,
    quality: Quality,
    policies: Policies<R, O, S>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EwVarianceDelta {
    samples: u32,
    mean_q: i64,
    variance_q: u128,
    quality: Quality,
}

/// Cap on `FRACTIONAL_BITS` so the variance Q format (`2 * FRACTIONAL_BITS`) is at most Q62 and
/// matches `i32`'s `i64` Q headroom (`63 - 32 = 31`).
const FAMILY_MAX_FRACTIONAL_BITS: u32 = 31;

/// Largest accepted `ALPHA_SHIFT` at every width: the rounding policies divide the `i64` mean
/// and `i128` variance deltas by `2^ALPHA_SHIFT` for any shift up to 62.
pub const EW_VARIANCE_MAX_ALPHA_SHIFT: u32 = 62;

/// Largest accepted `FRACTIONAL_BITS` for a word of `bits` bits: the family cap 31, or the word's
/// `i64` Q headroom `63 - bits` when that is smaller.
///
/// With `bits + F <= 63`, every Q-format sample, the mean, and the difference of any two of them
/// is below `2^(bits + F) <= 2^63` in magnitude, so all fit `i64`. The mean stays between its old
/// value and the sample, never leaving the sample range. The West innovation
/// `(x - old_mean) * (x - new_mean)` is a product of two differences of the same sign, so it lies
/// in `0..2^(2 * (bits + F))`, below `2^126`. The variance stays between its old value and the
/// innovation, so it lies in `0..2^126` and its signed delta fits `i128`.
pub const fn ew_variance_max_fractional_bits(bits: u32) -> u32 {
    let headroom = 63_u32.saturating_sub(bits);
    if headroom < FAMILY_MAX_FRACTIONAL_BITS {
        headroom
    } else {
        FAMILY_MAX_FRACTIONAL_BITS
    }
}

/// Whether the `i64` mean and `u128`/`i128` variance hold every value of a `bits`-bit word at `F`.
const fn ew_variance_fits(bits: u32, fractional_bits: u32) -> bool {
    match bits.checked_add(fractional_bits) {
        Some(total) => total <= 63,
        None => false,
    }
}

// The largest accepted `F` fits at every implemented width. At `i32` the word's headroom equals
// the family cap, so the proof is tight there; a 33-bit word would accept fewer bits.
const _: () = assert!(ew_variance_fits(8, ew_variance_max_fractional_bits(8)));
const _: () = assert!(ew_variance_fits(16, ew_variance_max_fractional_bits(16)));
const _: () = assert!(ew_variance_fits(32, ew_variance_max_fractional_bits(32)));
const _: () = assert!(!ew_variance_fits(
    32,
    ew_variance_max_fractional_bits(32) + 1
));
const _: () = assert!(ew_variance_max_fractional_bits(32) == FAMILY_MAX_FRACTIONAL_BITS);
const _: () = assert!(ew_variance_max_fractional_bits(33) < FAMILY_MAX_FRACTIONAL_BITS);
const _: () = assert!(ew_variance_max_fractional_bits(16) == 31);
const _: () =
    assert!(ew_variance_max_fractional_bits(16) <= <i16 as SampleWord>::I64_Q_HEADROOM_BITS);

impl<const A: u32, const F: u32, R, O, S: SampleWord> EwVariance<A, F, R, O, S> {
    /// Creates an empty weighted state after proving its shifts and products fit.
    ///
    /// `ALPHA_SHIFT` must be at most [`EW_VARIANCE_MAX_ALPHA_SHIFT`], and `FRACTIONAL_BITS` at
    /// most [`ew_variance_max_fractional_bits`] of `S::BITS` (31 at every implemented word).
    pub const fn new() -> Result<Self, Error> {
        if A <= EW_VARIANCE_MAX_ALPHA_SHIFT && F <= ew_variance_max_fractional_bits(S::BITS) {
            Ok(Self {
                samples: 0,
                mean_q: 0,
                variance_q: 0,
                quality: Quality::Approximate,
                policies: PhantomData,
            })
        } else {
            Err(Error::InvalidConfiguration)
        }
    }

    /// Returns `None` until the first sample is accepted.
    pub const fn snapshot(&self) -> Option<EwVarianceSnapshot> {
        let Some(samples) = NonZeroU32::new(self.samples) else {
            return None;
        };
        Some(EwVarianceSnapshot {
            samples,
            mean_q: self.mean_q,
            variance_q: self.variance_q,
            fractional_bits: F.saturating_add(S::FRACTIONAL_BITS),
            alpha_shift: A,
            quality: self.quality,
        })
    }

    /// Clears all weighted state.
    pub const fn reset(&mut self) {
        self.samples = 0;
        self.mean_q = 0;
        self.variance_q = 0;
        self.quality = Quality::Approximate;
    }
}

c0nst::c0nst! {
    impl<const A: u32, const F: u32, R: RoundingPolicy, O: OverflowPolicy, S: SampleWord>
        EwVariance<A, F, R, O, S>
    {
        /// Accepts one sample transactionally in constant time.
        #[allow(clippy::question_mark)]
        pub c0nst fn update(&mut self, sample: S) -> Result<(), Error>
        where
            R: [c0nst] RoundingPolicy,
            O: [c0nst] OverflowPolicy,
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
        pub(crate) c0nst fn prepare(&self, sample: S) -> Result<EwVarianceDelta, Error>
        where
            R: [c0nst] RoundingPolicy,
            O: [c0nst] OverflowPolicy,
            S: [c0nst] SampleWord,
        {
            let samples = match O::add_u32(self.samples, 1) {
                Ok(samples) => samples,
                Err(error) => return Err(error),
            };
            // `new` validated `A <= 62` and `S::BITS + F <= 63`, so `sample_q`, `self.mean_q`,
            // `mean_delta`, and `residual_after` fit `i64`, and `innovation`, `self.variance_q`,
            // and the updated variance lie in `0..2^126`.
            let sample_q = sample.to_i64() << F;
            let (mean_q, variance_q) = if self.samples == 0 {
                (sample_q, 0_u128)
            } else {
                let internal = crate::sealed::Internal::new();
                let mean_delta = sample_q - self.mean_q;
                let mean_adjustment =
                    R::divide_power_of_two_checked_shift(mean_delta, A, internal);
                let mean_q = self.mean_q + mean_adjustment;
                let residual_after = sample_q - mean_q;
                let innovation = (mean_delta as i128) * (residual_after as i128);
                let variance_delta = innovation - (self.variance_q as i128);
                let variance_adjustment =
                    R::divide_power_of_two_i128_checked_shift(variance_delta, A, internal);
                let variance_q = ((self.variance_q as i128) + variance_adjustment) as u128;
                (mean_q, variance_q)
            };
            Ok(EwVarianceDelta {
                samples: samples.value,
                mean_q,
                variance_q,
                quality: Quality::from_flags(true, self.quality.is_saturated() || samples.saturated),
            })
        }

        pub(crate) const fn commit(&mut self, delta: EwVarianceDelta) {
            self.samples = delta.samples;
            self.mean_q = delta.mean_q;
            self.variance_q = delta.variance_q;
            self.quality = delta.quality;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use core::num::NonZeroU32;

    use super::{EwVariance, ew_variance_fits, ew_variance_max_fractional_bits};
    use crate::arithmetic::NearestTiesAway;
    use crate::policy::{Checked, Saturating};
    use crate::sample::SampleWord;
    use crate::{Error, Quality};

    #[test]
    fn lifecycle_and_constant_signal_variance() {
        assert!(ew_variance_fits(32, 31));
        assert!(!ew_variance_fits(u32::MAX, 1));
        assert_eq!(ew_variance_max_fractional_bits(40), 23);
        let mut variance = EwVariance::<2, 8, NearestTiesAway, Checked, i16>::new().expect("valid");
        assert_eq!(variance.snapshot(), None);
        for _ in 0..8 {
            variance.update(42).expect("representable");
        }
        let snap = variance.snapshot().expect("defined");
        assert_eq!(snap.samples, NonZeroU32::new(8).expect("nonzero"));
        assert_eq!(snap.mean_q, 42 << 8);
        assert_eq!(snap.variance_q, 0);
        assert_eq!(snap.fractional_bits, 8);
        assert_eq!(snap.alpha_shift, 2);
        assert_eq!(snap.quality, Quality::Approximate);
        variance.reset();
        assert_eq!(variance.snapshot(), None);
    }

    fn every_width_fails_transactionally<S: SampleWord>() {
        let full = EwVariance::<2, 31, NearestTiesAway, Checked, S> {
            samples: u32::MAX,
            mean_q: 3,
            variance_q: 9,
            quality: Quality::Approximate,
            policies: PhantomData,
        };
        let mut variance = full;
        assert_eq!(variance.update(S::MAX), Err(Error::Overflow));
        assert_eq!(variance.update(S::MIN), Err(Error::Overflow));
        assert_eq!(variance, full);

        let mut saturating = EwVariance::<2, 31, NearestTiesAway, Saturating, S> {
            samples: u32::MAX,
            mean_q: 0,
            variance_q: 0,
            quality: Quality::Approximate,
            policies: PhantomData,
        };
        saturating.update(S::MAX).expect("saturating accepts");
        let snapshot = saturating.snapshot().expect("defined");
        assert_eq!(snapshot.samples.get(), u32::MAX);
        assert_eq!(snapshot.quality, Quality::ApproximateSaturated);
        saturating
            .update(S::MIN)
            .expect("already saturated update accepts");
        assert_eq!(
            saturating.snapshot().expect("defined").quality,
            Quality::ApproximateSaturated
        );
    }

    #[test]
    fn checked_failure_is_transactional_at_every_width() {
        every_width_fails_transactionally::<i8>();
        every_width_fails_transactionally::<i16>();
        every_width_fails_transactionally::<i32>();
        every_width_fails_transactionally::<u8>();
        every_width_fails_transactionally::<u16>();
        every_width_fails_transactionally::<u32>();
    }

    fn formats_follow_the_validator<S: SampleWord>() {
        assert_eq!(ew_variance_max_fractional_bits(S::BITS), 31);
        assert!(EwVariance::<0, 31, NearestTiesAway, Checked, S>::new().is_ok());
        assert!(EwVariance::<62, 0, NearestTiesAway, Checked, S>::new().is_ok());
        assert_eq!(
            EwVariance::<4, 32, NearestTiesAway, Checked, S>::new(),
            Err(Error::InvalidConfiguration)
        );
        assert_eq!(
            EwVariance::<63, 4, NearestTiesAway, Checked, S>::new(),
            Err(Error::InvalidConfiguration)
        );
    }

    #[test]
    fn accepted_formats_follow_the_validator_at_every_width() {
        formats_follow_the_validator::<i8>();
        formats_follow_the_validator::<i16>();
        formats_follow_the_validator::<i32>();
        formats_follow_the_validator::<u8>();
        formats_follow_the_validator::<u16>();
        formats_follow_the_validator::<u32>();
    }

    fn full_swing<const A: u32>() {
        let mut variance =
            EwVariance::<A, 31, NearestTiesAway, Checked, i32>::new().expect("valid");
        for index in 0..64 {
            let sample = if index % 2 == 0 { i32::MIN } else { i32::MAX };
            variance.update(sample).expect("proven not to overflow");
        }
        let snapshot = variance.snapshot().expect("defined");
        assert!(snapshot.variance_q < 1_u128 << 126);
    }

    #[test]
    fn i32_full_swing_at_q31_never_overflows() {
        full_swing::<0>();
        full_swing::<1>();
        full_swing::<62>();
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_ew_variance_is_const_evaluable_on_nightly() {
        const SNAPSHOT: super::EwVarianceSnapshot = {
            let mut variance = match EwVariance::<1, 4, NearestTiesAway, Checked, i16>::new() {
                Ok(v) => v,
                Err(_) => unreachable!(),
            };
            assert!(variance.update(10).is_ok());
            assert!(variance.update(20).is_ok());
            match variance.snapshot() {
                Some(s) => s,
                None => unreachable!(),
            }
        };
        assert_eq!(SNAPSHOT.samples.get(), 2);
        assert_eq!(SNAPSHOT.mean_q, 15 << 4);
        assert_eq!(SNAPSHOT.variance_q, (50 << 8) / 2);
    }
}
