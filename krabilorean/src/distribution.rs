//! Shared borrowed views and queries for bounded ordered distributions.

use core::num::NonZeroU32;

use crate::histogram::HistogramBounds;
use crate::sample::SampleValue;
use crate::{Error, Quality};

/// Inclusive sample-value bounds `lower..=upper` with `lower <= upper` guaranteed by construction.
///
/// `S` is the sample value type, and every bucket view reports bounds in its own value type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueBounds<S> {
    lower: S,
    upper: S,
}

c0nst::c0nst! {
    impl<S: SampleValue> ValueBounds<S> {
        /// Creates inclusive bounds when `lower <= upper`.
        pub c0nst fn new(lower: S, upper: S) -> Result<Self, Error>
        where
            S: [c0nst] SampleValue,
        {
            if lower.above(upper) {
                Err(Error::InvalidConfiguration)
            } else {
                Ok(Self { lower, upper })
            }
        }
    }
}

impl<S: Copy> ValueBounds<S> {
    /// Creates single-value bounds `value..=value`.
    pub const fn point(value: S) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Converts validated [`HistogramBounds`] into [`ValueBounds`].
    pub const fn from_histogram_bounds(bounds: HistogramBounds<S>) -> Self {
        Self {
            lower: bounds.lower(),
            upper: bounds.upper(),
        }
    }

    /// Crate-internal constructor for callers that have already established `lower <= upper`.
    pub(crate) const fn from_ordered(lower: S, upper: S) -> Self {
        Self { lower, upper }
    }

    /// Returns the inclusive lower bound.
    pub const fn lower(self) -> S {
        self.lower
    }

    /// Returns the inclusive upper bound.
    pub const fn upper(self) -> S {
        self.upper
    }
}

impl<S: Copy> From<HistogramBounds<S>> for ValueBounds<S> {
    fn from(bounds: HistogramBounds<S>) -> Self {
        Self::from_histogram_bounds(bounds)
    }
}

/// One borrowed-view bucket copied out of a bounded distribution of `S` samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bucket<S> {
    /// Inclusive values represented by this bucket.
    pub values: ValueBounds<S>,
    /// Accepted samples represented by this bucket.
    pub count: u32,
}

/// Inclusive lower and upper bounds on a sample count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountBounds {
    /// Count guaranteed by the summary.
    pub minimum: u32,
    /// Largest count consistent with the summary.
    pub maximum: u32,
}

/// An exact rational quantile in the closed interval `0..=1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quantile {
    numerator: u32,
    // Storing `denominator - 1` keeps every bit pattern a valid non-zero denominator while
    // avoiding a `NonZeroU32` layout niche that perturbs surrounding estimator layouts on
    // thumbv6m.
    denominator_less_one: u32,
}

impl Quantile {
    /// Creates `numerator / denominator` when it lies in the closed unit interval.
    pub const fn new(numerator: u32, denominator: u32) -> Result<Self, Error> {
        let Some(denominator) = NonZeroU32::new(denominator) else {
            return Err(Error::InvalidConfiguration);
        };
        if numerator > denominator.get() {
            return Err(Error::InvalidConfiguration);
        }
        Ok(Self {
            numerator,
            denominator_less_one: denominator.get() - 1,
        })
    }

    /// Returns the zero-based order-statistic rank `floor(q * (samples - 1))`.
    ///
    /// Infallible because the denominator is non-zero, the product of two `u32` values fits `u64`,
    /// and `q <= 1` keeps the rank at most `samples - 1`.
    pub const fn rank(self, samples: NonZeroU32) -> u32 {
        let scaled = self.numerator as u64 * (samples.get() - 1) as u64;
        (scaled / self.denominator().get() as u64) as u32
    }

    /// Returns the numerator.
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// Returns the non-zero denominator.
    pub const fn denominator(self) -> NonZeroU32 {
        // The stored value is at most `u32::MAX - 1`, so adding it to `1` cannot saturate.
        NonZeroU32::MIN.saturating_add(self.denominator_less_one)
    }
}

/// Value bounds containing one requested quantile rank, in `S` sample units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantileBounds<S> {
    /// Zero-based order-statistic rank, `floor(q * (samples - 1))`.
    pub rank: u32,
    /// Inclusive value bounds containing that rank.
    pub values: ValueBounds<S>,
}

/// The most populated bucket, with the lowest bucket index winning ties, in `S` sample units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketMode<S> {
    /// Stable index in the borrowed bucket view.
    pub index: usize,
    /// Inclusive values represented by the modal bucket.
    pub values: ValueBounds<S>,
    /// Non-zero count in the modal bucket.
    pub count: NonZeroU32,
}

const _: () = assert!(
    core::mem::size_of::<Option<BucketMode<i16>>>() == core::mem::size_of::<BucketMode<i16>>()
);

/// Used and available bucket slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Occupancy {
    /// Slots whose count is non-zero.
    pub occupied: usize,
    /// Slots exposed by this view.
    pub slots: usize,
}

/// Conservative counts on both sides of a threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TailBounds {
    /// Samples less than or equal to the threshold.
    pub at_or_below: CountBounds,
    /// Samples greater than or equal to the threshold.
    pub at_or_above: CountBounds,
}

c0nst::c0nst! {
    /// Borrowed, ordered buckets from a bounded distribution of `S` samples.
    ///
    /// Implementations expose buckets in increasing value order. Exact distributions cover every
    /// accepted sample once; a bucket may represent a value range or a single value.
    pub c0nst trait BucketView<S> {
        /// Number of stable bucket positions exposed by [`Self::bucket`].
        fn bucket_slots(&self) -> usize;
        /// Returns the bucket at `index`, or `None` when that slot holds no representable values
        /// (such as an empty interior linear bin or an edge slot outside the word's domain) or
        /// when `index >= self.bucket_slots()`. Callers must scan all indices in
        /// `0..self.bucket_slots()` because `None` can appear between populated slots.
        fn bucket(&self, index: usize) -> Option<Bucket<S>>;
        /// Total accepted samples represented by the distribution.
        fn samples(&self) -> u32;
        /// Arithmetic and approximation quality of the distribution state.
        fn quality(&self) -> Quality;
    }

    /// Marker for bucket views whose algorithmic policy is exact while quality is complete.
    ///
    /// Saturated arithmetic is visible in [`BucketView::quality`], and shared queries reject
    /// bucket counts that do not sum to `samples`.
    pub c0nst trait ExactBucketView<S>:
        [c0nst] BucketView<S> + [c0nst] crate::sealed::ExactBucketView
    {
    }

    /// Queries shared by exact linear, power-of-two, and categorical distributions of `S` samples.
    pub c0nst trait ExactDistributionQueries<S: [c0nst] SampleValue>:
        [c0nst] ExactBucketView<S>
    {
        /// Returns occupied bucket slots after validating that counts sum to `samples`.
        #[allow(clippy::question_mark)]
        fn occupancy(&self) -> Result<Occupancy, Error> {
            let slots = self.bucket_slots();
            let mut occupied = 0_usize;
            let mut total = 0_u64;
            let mut index = 0_usize;
            while index < slots {
                if let Some(bucket) = self.bucket(index) {
                    if bucket.count > 0 {
                        occupied += 1;
                    }
                    total = total.saturating_add(bucket.count as u64);
                }
                index += 1;
            }
            if let Err(error) = validate_total(total, self.samples()) {
                return Err(error);
            }
            Ok(Occupancy { occupied, slots })
        }

        /// Returns the lowest-index populated bucket having the largest count.
        #[allow(clippy::question_mark)]
        fn mode(&self) -> Result<Option<BucketMode<S>>, Error> {
            let slots = self.bucket_slots();
            let mut mode: Option<BucketMode<S>> = None;
            let mut largest = 0_u32;
            let mut total = 0_u64;
            let mut index = 0_usize;
            while index < slots {
                if let Some(bucket) = self.bucket(index) {
                    total = total.saturating_add(bucket.count as u64);
                    if bucket.count > largest {
                        largest = bucket.count;
                        // `largest >= 0` and `bucket.count > largest`, so `bucket.count >= 1` and
                        // `NonZeroU32::MIN` is unreachable.
                        let count = match NonZeroU32::new(bucket.count) {
                            Some(nonzero) => nonzero,
                            None => NonZeroU32::MIN,
                        };
                        mode = Some(BucketMode {
                            index,
                            values: bucket.values,
                            count,
                        });
                    }
                }
                index += 1;
            }
            if let Err(error) = validate_total(total, self.samples()) {
                return Err(error);
            }
            Ok(mode)
        }

        /// Returns inclusive value bounds containing the order-statistic rank
        /// `floor(q * (samples - 1))`.
        #[allow(clippy::question_mark)]
        fn quantile_bounds(&self, quantile: Quantile) -> Result<Option<QuantileBounds<S>>, Error> {
            let rank = match NonZeroU32::new(self.samples()) {
                Some(samples) => Some(quantile.rank(samples)),
                None => None,
            };
            let slots = self.bucket_slots();
            let mut cumulative = 0_u64;
            let mut result = None;
            let mut index = 0_usize;
            while index < slots {
                if let Some(bucket) = self.bucket(index) {
                    cumulative = cumulative.saturating_add(bucket.count as u64);
                    if let Some(rank) = rank
                        && result.is_none()
                        && cumulative > rank as u64
                    {
                        result = Some(QuantileBounds {
                            rank,
                            values: bucket.values,
                        });
                    }
                }
                index += 1;
            }
            // When `cumulative == self.samples() as u64`: if `self.samples() == 0`, `result` is
            // `None`; if `self.samples() > 0`, `cumulative > rank as u64` became true on the
            // bucket covering `rank`, so `result` is `Some`.
            if let Err(error) = validate_total(cumulative, self.samples()) {
                return Err(error);
            }
            Ok(result)
        }

        /// Returns conservative inclusive tail counts around `threshold`.
        ///
        /// A bucket wholly on one side contributes to both bounds. A bucket straddling the
        /// threshold contributes only to the maximum because its within-bucket ordering is not
        /// retained. Returns [`Error::Overflow`] when [`BucketView::quality`] is saturated,
        /// because saturated counters cannot establish a `u32` upper bound on the accepted count.
        #[allow(clippy::question_mark)]
        fn tail_bounds(&self, threshold: S) -> Result<TailBounds, Error> {
            if self.quality().is_saturated() {
                return Err(Error::Overflow);
            }
            let slots = self.bucket_slots();
            let mut below_min = 0_u64;
            let mut below_max = 0_u64;
            let mut above_min = 0_u64;
            let mut above_max = 0_u64;
            let mut total = 0_u64;
            let mut index = 0_usize;
            while index < slots {
                if let Some(bucket) = self.bucket(index) {
                    let count = bucket.count as u64;
                    total = total.saturating_add(count);
                    if threshold.at_or_above(bucket.values.upper()) {
                        below_min = below_min.saturating_add(count);
                        below_max = below_max.saturating_add(count);
                    } else if threshold.at_or_above(bucket.values.lower()) {
                        below_max = below_max.saturating_add(count);
                    }
                    if bucket.values.lower().at_or_above(threshold) {
                        above_min = above_min.saturating_add(count);
                        above_max = above_max.saturating_add(count);
                    } else if bucket.values.upper().at_or_above(threshold) {
                        above_max = above_max.saturating_add(count);
                    }
                }
                index += 1;
            }
            // Each partial sum is at most `total`; once `total == self.samples() as u64`, all four
            // bounds fit `u32`.
            if let Err(error) = validate_total(total, self.samples()) {
                return Err(error);
            }
            Ok(TailBounds {
                at_or_below: CountBounds {
                    minimum: below_min as u32,
                    maximum: below_max as u32,
                },
                at_or_above: CountBounds {
                    minimum: above_min as u32,
                    maximum: above_max as u32,
                },
            })
        }
    }

    c0nst impl<S: [c0nst] SampleValue, T: [c0nst] ExactBucketView<S> + ?Sized>
        ExactDistributionQueries<S> for T
    {
    }
}

const fn validate_total(total: u64, samples: u32) -> Result<(), Error> {
    if total == samples as u64 {
        Ok(())
    } else {
        Err(Error::ViewContractViolation)
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::{
        Bucket, BucketMode, BucketView, CountBounds, ExactBucketView, ExactDistributionQueries,
        Occupancy, Quantile, QuantileBounds, TailBounds, ValueBounds,
    };
    use crate::histogram::HistogramBounds;
    use crate::{Error, Quality};

    fn rank(numerator: u32, denominator: u32, samples: u32) -> u32 {
        let quantile = Quantile::new(numerator, denominator).expect("valid quantile");
        quantile.rank(NonZeroU32::new(samples).expect("non-zero samples"))
    }

    #[test]
    fn value_bounds_reject_inverted_endpoints_and_convert_from_histogram_bounds() {
        assert_eq!(
            ValueBounds::<i16>::new(5, -5),
            Err(Error::InvalidConfiguration)
        );
        let bounds = ValueBounds::<i16>::new(-5, 5).expect("valid");
        assert_eq!((bounds.lower(), bounds.upper()), (-5, 5));
        let point = ValueBounds::<i16>::point(3);
        assert_eq!((point.lower(), point.upper()), (3, 3));
        let hist = HistogramBounds::<i16>::new(-2, 4).expect("valid");
        let converted = ValueBounds::from(hist);
        assert_eq!((converted.lower(), converted.upper()), (-2, 4));
    }

    #[test]
    fn rank_matches_wide_arithmetic_at_the_extremes() {
        let cases = [
            (0, 1, 1),
            (1, 1, 1),
            (1, 2, 5),
            (1, 3, 10),
            (0, u32::MAX, u32::MAX),
            (1, 1, u32::MAX),
            (u32::MAX - 1, u32::MAX, u32::MAX),
            (u32::MAX, u32::MAX, u32::MAX),
            (1, u32::MAX, u32::MAX),
        ];
        for (numerator, denominator, samples) in cases {
            let wide = u128::from(numerator) * u128::from(samples - 1) / u128::from(denominator);
            assert_eq!(u128::from(rank(numerator, denominator, samples)), wide);
        }
    }

    #[test]
    fn zero_denominator_and_values_above_one_are_rejected() {
        assert_eq!(Quantile::new(0, 0), Err(Error::InvalidConfiguration));
        assert_eq!(Quantile::new(2, 1), Err(Error::InvalidConfiguration));
        let q = Quantile::new(3, 4).expect("valid");
        assert_eq!(q.numerator(), 3);
        assert_eq!(q.denominator().get(), 4);
        assert_eq!(
            Quantile::new(1, u32::MAX).map(|quantile| quantile.denominator().get()),
            Ok(u32::MAX)
        );
    }

    struct DummyView {
        buckets: [Option<Bucket<i16>>; 3],
        samples: u32,
        quality: Quality,
    }

    c0nst::c0nst! {
        c0nst impl BucketView<i16> for DummyView {
            fn bucket_slots(&self) -> usize {
                self.buckets.len()
            }

            fn bucket(&self, index: usize) -> Option<Bucket<i16>> {
                if index < self.buckets.len() {
                    self.buckets[index]
                } else {
                    None
                }
            }

            fn samples(&self) -> u32 {
                self.samples
            }

            fn quality(&self) -> Quality {
                self.quality
            }
        }

        c0nst impl crate::sealed::ExactBucketView for DummyView {}
        c0nst impl ExactBucketView<i16> for DummyView {}
    }

    #[test]
    fn queries_handle_empty_and_populated_views_and_reject_mismatched_totals() {
        let empty = DummyView {
            buckets: [None, None, None],
            samples: 0,
            quality: Quality::Complete,
        };
        assert_eq!(empty.quality(), Quality::Complete);
        assert_eq!(empty.bucket(99), None);
        assert_eq!(
            empty.occupancy(),
            Ok(Occupancy {
                occupied: 0,
                slots: 3,
            })
        );
        assert_eq!(empty.mode(), Ok(None));
        assert_eq!(
            empty.quantile_bounds(Quantile::new(1, 2).expect("valid")),
            Ok(None)
        );
        assert_eq!(
            empty.tail_bounds(0),
            Ok(TailBounds {
                at_or_below: CountBounds {
                    minimum: 0,
                    maximum: 0,
                },
                at_or_above: CountBounds {
                    minimum: 0,
                    maximum: 0,
                },
            })
        );

        let b0 = ValueBounds::new(-10, -1).expect("valid");
        let b1 = ValueBounds::new(0, 5).expect("valid");
        let b2 = ValueBounds::new(6, 10).expect("valid");
        let populated = DummyView {
            buckets: [
                Some(Bucket {
                    values: b0,
                    count: 2,
                }),
                Some(Bucket {
                    values: b1,
                    count: 5,
                }),
                Some(Bucket {
                    values: b2,
                    count: 5,
                }),
            ],
            samples: 12,
            quality: Quality::Complete,
        };
        assert_eq!(
            populated.occupancy(),
            Ok(Occupancy {
                occupied: 3,
                slots: 3,
            })
        );
        assert_eq!(
            populated.mode(),
            Ok(Some(BucketMode {
                index: 1,
                values: b1,
                count: NonZeroU32::new(5).expect("nonzero"),
            }))
        );
        assert_eq!(
            populated.quantile_bounds(Quantile::new(1, 2).expect("valid")),
            Ok(Some(QuantileBounds {
                rank: 5,
                values: b1,
            }))
        );
        assert_eq!(
            populated.tail_bounds(3),
            Ok(TailBounds {
                at_or_below: CountBounds {
                    minimum: 2,
                    maximum: 7,
                },
                at_or_above: CountBounds {
                    minimum: 5,
                    maximum: 10,
                },
            })
        );

        let mismatched = DummyView {
            samples: 11,
            ..populated
        };
        assert_eq!(mismatched.occupancy(), Err(Error::ViewContractViolation));
        assert_eq!(mismatched.mode(), Err(Error::ViewContractViolation));
        assert_eq!(
            mismatched.quantile_bounds(Quantile::new(1, 2).expect("valid")),
            Err(Error::ViewContractViolation)
        );
        assert_eq!(mismatched.tail_bounds(3), Err(Error::ViewContractViolation));

        let zero_samples_with_nonzero_bucket = DummyView {
            samples: 0,
            ..populated
        };
        assert_eq!(
            zero_samples_with_nonzero_bucket.quantile_bounds(Quantile::new(1, 2).expect("valid")),
            Err(Error::ViewContractViolation)
        );

        let saturated_view = DummyView {
            quality: Quality::Saturated,
            ..populated
        };
        assert_eq!(saturated_view.tail_bounds(3), Err(Error::Overflow));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn c0nst_distribution_queries_are_const_evaluable_on_nightly() {
        const VIEW: DummyView = DummyView {
            buckets: [
                Some(Bucket {
                    values: match ValueBounds::new(-4, -1) {
                        Ok(bounds) => bounds,
                        Err(_) => unreachable!(),
                    },
                    count: 1,
                }),
                Some(Bucket {
                    values: match ValueBounds::new(0, 3) {
                        Ok(bounds) => bounds,
                        Err(_) => unreachable!(),
                    },
                    count: 3,
                }),
                None,
            ],
            samples: 4,
            quality: Quality::Complete,
        };
        const MEDIAN: Quantile = match Quantile::new(1, 2) {
            Ok(q) => q,
            Err(_) => unreachable!(),
        };
        const OCC: Result<Occupancy, Error> = VIEW.occupancy();
        const MODE: Result<Option<BucketMode<i16>>, Error> = VIEW.mode();
        const QBOUNDS: Result<Option<QuantileBounds<i16>>, Error> = VIEW.quantile_bounds(MEDIAN);
        const TAILS: Result<TailBounds, Error> = VIEW.tail_bounds(1);
        assert_eq!(
            OCC,
            Ok(Occupancy {
                occupied: 2,
                slots: 3,
            })
        );
        assert_eq!(MODE.unwrap().unwrap().count, NonZeroU32::new(3).unwrap());
        assert_eq!(QBOUNDS.unwrap().unwrap().rank, 1);
        assert_eq!(
            TAILS,
            Ok(TailBounds {
                at_or_below: CountBounds {
                    minimum: 1,
                    maximum: 4,
                },
                at_or_above: CountBounds {
                    minimum: 0,
                    maximum: 3,
                },
            })
        );
    }
}
