
import pytest
import numpy as np
import pandas as pd

from brainmaze_hdf.utils import get_data_segments, create_block_indexes, reinterpolate, get_involved_intervals

def test_get_data_segments_no_nans():
    """Test with data containing no NaN values."""
    data = np.array([1, 2, 3, 4, 5])
    expected_df = pd.DataFrame({'start': [0], 'end': [4], 'type': ['data']})

    data_segments = get_data_segments(data)

    assert data_segments.equals(expected_df)

def test_get_data_segments_only_nans():
    """Test with data containing no NaN values."""
    data = np.array([np.nan]*5)
    expected_df = pd.DataFrame({'start': [0], 'end': [4], 'type': ['nan']})

    data_segments = get_data_segments(data)

    assert data_segments.equals(expected_df)

def test_get_data_segments_with_nans():
    """Test with data containing NaN values."""
    data = np.array([1, 2, np.nan, np.nan, 5, 6, 7, np.nan, 9, 10])
    expected_df = pd.DataFrame({
        'start': [0, 2, 4, 7, 8],
        'end': [1, 3, 6, 7, 9],
        'type': ['data', 'nan', 'data', 'nan', 'data']
    })

    data_segments = get_data_segments(data)
    assert data_segments.equals(expected_df)

def test_get_data_segments_nans_beginning():
    """Test with data starting with NaN values."""
    data = np.array([np.nan, np.nan, 1, 2, 3, 4, 5])
    expected_df = pd.DataFrame({'start': [0, 2], 'end': [1, 6], 'type': ['nan', 'data']})

    data_segments = get_data_segments(data)
    assert data_segments.equals(expected_df)

def test_get_data_segments_nans_end():
    """Test with data ending with NaN values."""
    data = np.array([1, 2, 3, 4, 5, np.nan, np.nan])
    expected_df = pd.DataFrame({'start': [0, 5], 'end': [4, 6], 'type': ['data', 'nan']})

    data_segments = get_data_segments(data)
    assert data_segments.equals(expected_df)



def test_create_block_indexes_no_nans():
    """Test with data containing no NaN values."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    block_size = 3
    expected_indexes = pd.DataFrame([
        {'start_idx': 0, 'end_idx': 2},
        {'start_idx': 3, 'end_idx': 5},
        {'start_idx': 6, 'end_idx': 8},
        {'start_idx': 9, 'end_idx': 9}
    ])
    block_indexes = create_block_indexes(data, block_size)
    assert block_indexes.equals(expected_indexes)

def test_create_block_indexes_with_nans():
    """Test with data containing NaN values."""
    data = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan, 9, 10, np.nan, np.nan, np.nan, 14])
    block_size = 3
    expected_indexes = pd.DataFrame([
        {'start_idx': 0, 'end_idx': 2},
        {'start_idx': 4, 'end_idx': 6},
        {'start_idx': 8, 'end_idx': 9},
        {'start_idx': 13, 'end_idx': 13}
    ])
    block_indexes = create_block_indexes(data, block_size)
    assert block_indexes.equals(expected_indexes)

def test_create_block_indexes_with_nans_at_boundaries():
    """Test with NaN values at the beginning and end."""
    data = np.array([np.nan, 1, 2, 3, 4, np.nan])
    block_size = 3
    expected_indexes = pd.DataFrame([
        {'start_idx': 1, 'end_idx': 3},
        {'start_idx': 4, 'end_idx': 4}
    ])
    block_indexes = create_block_indexes(data, block_size)
    assert block_indexes.equals(expected_indexes)

def test_create_block_indexes_with_consecutive_nans():
    """Test with consecutive NaN values."""
    data = np.array([1, 2, np.nan, np.nan, 5, 6, 7, 8, 9, 10])
    block_size = 3
    expected_indexes = pd.DataFrame([
        {'start_idx': 0, 'end_idx': 1},
        {'start_idx': 4, 'end_idx': 6},
        {'start_idx': 7, 'end_idx': 9}
    ])
    block_indexes = create_block_indexes(data, block_size)
    assert block_indexes.equals(expected_indexes)

def test_create_block_indexes_with_all_nans():
    """Test with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    block_size = 3
    expected_indexes = pd.DataFrame([], columns=['start_idx', 'end_idx'])
    block_indexes = create_block_indexes(data, block_size)
    assert block_indexes.equals(expected_indexes)


def test_reinterpolate_no_nans():
    """Test reinterpolation with no NaN values."""
    x = np.arange(0, 20, 0.5)
    # y = np.sin(x*2*np.pi*0.1)
    y = np.array([0,5,10,5]*10)

    x_in = x[::2]
    y_in = y[::2]

    x_out = x[1:-1:2]
    y_ref = y[1:-1:2]

    x_out_valid, y_out = reinterpolate(x_in, y_in, x_out)

    assert np.all(y_out == y_ref)



def test_get_involved_intervals():
    # Test data
    intervals = np.array([
        [0, 9],  # Interval 1
        [10, 19], # Interval 2
        [20, 29], # Interval 3
    ])

    start_uutc = 12
    end_uutc = 14
    expected = np.array([False, True, False])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)


    start_uutc = 5
    end_uutc = 15
    expected = np.array([True, True, False])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)

    start_uutc = 5
    end_uutc = 25
    expected = np.array([True, True, True])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)

    start_uutc = 0
    end_uutc = 30
    expected = np.array([True, True, True])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)

    start_uutc = 0
    end_uutc = 9
    expected = np.array([True, False, False])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)

    start_uutc = 0
    end_uutc = 10
    expected = np.array([True, True, False])
    result = get_involved_intervals(intervals, start_uutc, end_uutc)
    assert np.array_equal(result, expected)

