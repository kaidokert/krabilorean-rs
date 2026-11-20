//! Stable, storage-independent indexed sample views.

use core::marker::PhantomData;

use crate::Error;

/// A stable indexed view of samples of type `S`.
///
/// Implementations must return `Some` for every index smaller than [`Self::len`]. A consumer
/// reports [`Error::ViewContractViolation`] if an implementation breaks that contract.
///
/// Slices, arrays, [`SubView`] and [`FunctionView`] are views of any `Copy` sample type.
pub trait SampleView<S> {
    /// Number of samples visible through the view.
    fn len(&self) -> usize;

    /// Loads one sample, or `None` when the index is outside the view.
    fn sample(&self, index: usize) -> Option<S>;

    /// Returns whether the view contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S: Copy> SampleView<S> for [S] {
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    fn sample(&self, index: usize) -> Option<S> {
        self.get(index).copied()
    }
}

impl<S: Copy, const N: usize> SampleView<S> for [S; N] {
    fn len(&self) -> usize {
        N
    }

    fn sample(&self, index: usize) -> Option<S> {
        self.get(index).copied()
    }
}

/// A validated subrange of another stable view of `S` samples.
#[derive(Debug)]
pub struct SubView<'a, V: SampleView<S> + ?Sized, S> {
    source: &'a V,
    start: usize,
    len: usize,
    sample: PhantomData<fn() -> S>,
}

impl<V: SampleView<S> + ?Sized, S> Clone for SubView<'_, V, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V: SampleView<S> + ?Sized, S> Copy for SubView<'_, V, S> {}

impl<'a, V: SampleView<S> + ?Sized, S> SubView<'a, V, S> {
    /// Creates a subview after validating the requested range.
    pub fn new(source: &'a V, start: usize, len: usize) -> Result<Self, Error> {
        let end = start.checked_add(len).ok_or(Error::InvalidConfiguration)?;
        if end > source.len() {
            return Err(Error::InvalidConfiguration);
        }
        Ok(Self {
            source,
            start,
            len,
            sample: PhantomData,
        })
    }
}

impl<V: SampleView<S> + ?Sized, S> SampleView<S> for SubView<'_, V, S> {
    fn len(&self) -> usize {
        self.len
    }

    fn sample(&self, index: usize) -> Option<S> {
        if index >= self.len {
            return None;
        }
        self.start
            .checked_add(index)
            .and_then(|source_index| self.source.sample(source_index))
    }
}

/// A view backed by a caller-provided loading function.
#[derive(Clone, Copy)]
pub struct FunctionView<F> {
    len: usize,
    load: F,
}

impl<F> FunctionView<F> {
    /// Creates a function-backed view with a declared length.
    pub const fn new(len: usize, load: F) -> Self {
        Self { len, load }
    }
}

impl<F, S> SampleView<S> for FunctionView<F>
where
    F: Fn(usize) -> Option<S>,
{
    fn len(&self) -> usize {
        self.len
    }

    fn sample(&self, index: usize) -> Option<S> {
        if index >= self.len {
            None
        } else {
            (self.load)(index)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::{FunctionView, SampleView, SubView};
    use crate::Error;

    #[test]
    fn views_work_across_sample_words() {
        let samples = [i32::MIN, -1, 0, 1, i32::MAX];
        assert!(!SampleView::is_empty(&samples));
        let slice: &[i32] = &samples;
        assert_eq!((slice.sample(0), slice.sample(5)), (Some(i32::MIN), None));
        assert!(SampleView::is_empty(&[0_u8; 0][..]));

        let sub = SubView::new(&samples, 1, 3).expect("in range");
        assert!(!std::format!("{sub:?}").is_empty());
        #[allow(clippy::clone_on_copy)]
        let copied = sub.clone();
        assert_eq!(SampleView::len(&copied), 3);
        assert_eq!(
            (copied.sample(0), copied.sample(2), copied.sample(3)),
            (Some(-1), Some(1), None)
        );

        let function = FunctionView::new(2, |index: usize| u32::try_from(index).ok());
        assert_eq!(SampleView::len(&function), 2);
        assert_eq!(
            (function.sample(0), function.sample(1), function.sample(2)),
            (Some(0_u32), Some(1_u32), None)
        );
    }

    #[test]
    fn subview_rejects_out_of_bounds_and_never_reads_outside_declared_range() {
        let values = [10_i16, 20, 30, 40, 50];
        assert_eq!(
            SubView::new(&values, 3, 3).unwrap_err(),
            Error::InvalidConfiguration
        );
        assert_eq!(
            SubView::new(&values, usize::MAX, 1).unwrap_err(),
            Error::InvalidConfiguration
        );

        let seen_mask = Cell::new(0_u32);
        let source = FunctionView::new(values.len(), |index| {
            seen_mask.set(seen_mask.get() | (1_u32 << index));
            values.get(index).copied()
        });
        let sub = SubView::new(&source, 1, 3).expect("valid subrange");
        assert_eq!(sub.sample(0), Some(20));
        assert_eq!(sub.sample(1), Some(30));
        assert_eq!(sub.sample(2), Some(40));
        assert_eq!(sub.sample(3), None);
        assert_eq!(source.sample(values.len()), None);
        assert_eq!(seen_mask.get(), 0b01110);
    }
}
