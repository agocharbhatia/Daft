use common_error::DaftResult;

use crate::{datatypes::DataType, series::Series};

impl Series {
    pub fn cast(&self, datatype: &DataType) -> DaftResult<Self> {
        let casted = self.inner.cast(datatype)?;
        if casted.name() == self.name() {
            Ok(casted)
        } else {
            Ok(casted.rename(self.name()))
        }
    }

    /// Safe type cast: returns null instead of raising an error when conversion fails
    pub fn try_cast(&self, datatype: &DataType) -> DaftResult<Self> {
        // If the target type is the same as the current type, return directly
        if self.data_type() == datatype {
            return Ok(self.clone());
        }

        // Try bulk cast first, return directly if successful
        match self.cast(datatype) {
            Ok(casted) => Ok(casted),
            Err(_) => {
                // Bulk cast failed, try element-wise.
                // NOTE: Known performance limitation - this is O(n) slice+cast+concat which
                // can be very slow at scale. For columns where all rows fail the cast, this
                // creates N single-element Series objects. A future optimization could use a
                // validity bitmap to perform a vectorized cast. See: https://github.com/Eventual-Inc/Daft/issues/6959
                let len = self.len();
                let mut results: Vec<Self> = Vec::with_capacity(len);

                for i in 0..len {
                    let element = self.slice(i, i + 1)?;
                    match element.cast(datatype) {
                        Ok(casted) => results.push(casted),
                        Err(_) => {
                            results.push(Self::full_null(self.name(), datatype, 1));
                        }
                    }
                }

                // Concatenate all single-element Series
                if results.is_empty() {
                    Ok(Self::empty(self.name(), datatype))
                } else {
                    let ref_results: Vec<&Self> = results.iter().collect();
                    Self::concat(&ref_results)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::buffer::{NullBuffer, OffsetBuffer};

    use super::*;
    use crate::{array::ListArray, datatypes::Field, prelude::BinaryArray, series::IntoSeries};

    // --- Test fixture helpers ---

    /// Keep child storage shared so row slices exercise values outside their visible range.
    fn binary_lists(
        values: &[Option<&[u8]>],
        lengths: &[usize],
        nulls: Option<NullBuffer>,
    ) -> Series {
        let child = BinaryArray::from_iter("item", values.iter().copied()).into_series();
        ListArray::new(
            Field::new("a", DataType::List(Box::new(DataType::Binary))),
            child,
            OffsetBuffer::from_lengths(lengths.iter().copied()),
            nulls,
        )
        .into_series()
    }

    // --- End test fixture helpers ---

    #[test]
    fn test_list_cast_visible_children() {
        let values = [Some(b"x".as_slice()), Some(b"ab"), Some(b"cd"), Some(b"z")];
        let source = binary_lists(&values, &[1, 1, 1, 1], None);
        let target = DataType::List(Box::new(DataType::FixedSizeBinary(2)));

        for (start, end) in [(1, 2), (2, 3), (1, 3)] {
            let result = source.slice(start, end).unwrap().cast(&target).unwrap();
            let expected = binary_lists(&values[start..end], &vec![1; end - start], None)
                .cast(&target)
                .unwrap();
            assert_eq!(
                result.to_arrow().unwrap().to_data(),
                expected.to_arrow().unwrap().to_data()
            );
            assert_eq!(result.name(), source.name());
            assert_eq!(result.data_type(), &target);
        }
        assert!(source.cast(&target).is_err());
    }

    #[test]
    fn test_fixed_size_list_cast_visible_children() {
        let source = binary_lists(
            &[Some(b"x"), Some(b"ab"), Some(b"cd"), Some(b"z")],
            &[1, 1, 0, 1, 1],
            Some(NullBuffer::from_iter([true, true, false, true, true])),
        );
        let target = DataType::FixedSizeList(Box::new(DataType::FixedSizeBinary(2)), 1);
        let result = source.slice(1, 4).unwrap().cast(&target).unwrap();
        let expected = binary_lists(
            &[Some(b"ab"), Some(b"cd")],
            &[1, 0, 1],
            Some(NullBuffer::from_iter([true, false, true])),
        )
        .cast(&target)
        .unwrap();
        assert_eq!(
            result.to_arrow().unwrap().to_data(),
            expected.to_arrow().unwrap().to_data()
        );
        assert_eq!(result.name(), source.name());
        assert_eq!(result.data_type(), &target);
        assert!(source.cast(&target).is_err());

        let source = binary_lists(
            &[Some(b"x"), Some(b"ab"), Some(b"cd"), Some(b"z")],
            &[1, 1, 1, 1],
            None,
        );
        let result = source.slice(1, 3).unwrap().cast(&target).unwrap();
        let expected = binary_lists(&[Some(b"ab"), Some(b"cd")], &[1, 1], None)
            .cast(&target)
            .unwrap();
        assert_eq!(
            result.to_arrow().unwrap().to_data(),
            expected.to_arrow().unwrap().to_data()
        );
    }

    #[test]
    fn test_try_cast_list_preserves_valid_rows() {
        let target = DataType::List(Box::new(DataType::FixedSizeBinary(2)));
        for bad_row in 0..3 {
            let mut values = [Some(b"ab".as_slice()), Some(b"cd"), Some(b"ef")];
            values[bad_row] = Some(b"x");
            let source = binary_lists(&values, &[1, 1, 1], None);
            let result = source.try_cast(&target).unwrap();
            let expected_values = values
                .iter()
                .enumerate()
                .filter_map(|(i, value)| (i != bad_row).then_some(*value))
                .collect::<Vec<_>>();
            let lengths = (0..3)
                .map(|i| usize::from(i != bad_row))
                .collect::<Vec<_>>();
            let expected = binary_lists(
                &expected_values,
                &lengths,
                Some(NullBuffer::from_iter((0..3).map(|i| i != bad_row))),
            )
            .cast(&target)
            .unwrap();
            assert_eq!(
                result.to_arrow().unwrap().to_data(),
                expected.to_arrow().unwrap().to_data()
            );
            assert_eq!(result.name(), source.name());
            assert_eq!(result.data_type(), &target);
        }
    }

    #[test]
    fn test_list_cast_visible_null_and_empty_rows() {
        let source = binary_lists(
            &[Some(b"x"), Some(b"ab"), None, Some(b"cd"), Some(b"z")],
            &[1, 1, 0, 0, 2, 1],
            Some(NullBuffer::from_iter([true, true, false, true, true, true])),
        );
        let target = DataType::List(Box::new(DataType::FixedSizeBinary(2)));
        let result = source.slice(1, 5).unwrap().cast(&target).unwrap();
        let expected = binary_lists(
            &[Some(b"ab"), None, Some(b"cd")],
            &[1, 0, 0, 2],
            Some(NullBuffer::from_iter([true, false, true, true])),
        )
        .cast(&target)
        .unwrap();
        assert_eq!(
            result.to_arrow().unwrap().to_data(),
            expected.to_arrow().unwrap().to_data()
        );
    }

    #[test]
    fn test_list_cast_empty_slice_ignores_hidden_children() {
        let source = binary_lists(&[Some(b"x")], &[1], None);
        let target = DataType::List(Box::new(DataType::FixedSizeBinary(2)));
        for index in [0, 1] {
            let result = source.slice(index, index).unwrap().cast(&target).unwrap();
            assert_eq!(result.len(), 0);
            assert_eq!(result.name(), source.name());
            assert_eq!(result.data_type(), &target);
            assert!(result.list().unwrap().flat_child.is_empty());
        }
    }

    #[test]
    fn test_try_cast_nested_list_preserves_valid_rows() {
        let child = ListArray::from_vec(
            "item",
            vec![Some(vec![1i64, 2]), Some(vec![3]), Some(vec![4, 5])],
        )
        .into_series();
        let source = ListArray::new(
            child.field().to_list_field(),
            child,
            OffsetBuffer::from_lengths([1, 1, 1]),
            None,
        )
        .into_series();
        let inner_target = DataType::FixedSizeList(Box::new(DataType::Int64), 2);
        let target = DataType::List(Box::new(inner_target.clone()));
        let result = source.try_cast(&target).unwrap();

        let child = ListArray::from_vec("item", vec![Some(vec![1i64, 2]), Some(vec![4, 5])])
            .cast(&inner_target)
            .unwrap();
        let expected = ListArray::new(
            child.field().to_list_field(),
            child,
            OffsetBuffer::from_lengths([1, 0, 1]),
            Some(NullBuffer::from_iter([true, false, true])),
        )
        .into_series();
        assert_eq!(
            result.to_arrow().unwrap().to_data(),
            expected.to_arrow().unwrap().to_data()
        );
    }
}
