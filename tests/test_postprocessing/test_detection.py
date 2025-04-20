import numpy as np
import pytest
import xarray as xr
from soundevent.arrays import Dimensions

from batdetect2.postprocess.detection import extract_detections_from_array


@pytest.fixture
def sample_data_array():
    """Provides a basic 3x3 DataArray.
    Top values: 0.9 (f=300, t=20), 0.8 (f=200, t=10), 0.7 (f=300, t=30)
    """
    array = xr.DataArray(
        np.zeros([3, 3]),
        coords={
            Dimensions.frequency.value: [100, 200, 300],
            Dimensions.time.value: [10, 20, 30],
        },
        dims=[
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
    )

    array.loc[dict(time=10, frequency=100)] = 0.005
    array.loc[dict(time=10, frequency=200)] = 0.5
    array.loc[dict(time=10, frequency=300)] = 0.03
    array.loc[dict(time=20, frequency=100)] = 0.8
    array.loc[dict(time=20, frequency=200)] = 0.02
    array.loc[dict(time=20, frequency=300)] = 0.6
    array.loc[dict(time=30, frequency=100)] = 0.04
    array.loc[dict(time=30, frequency=200)] = 0.9
    array.loc[dict(time=30, frequency=300)] = 0.7
    return array


@pytest.fixture
def data_array_with_nans(sample_data_array: xr.DataArray):
    """Provides a 2D DataArray containing NaN values."""
    array = sample_data_array.copy()
    array.loc[dict(time=10, frequency=300)] = np.nan
    array.loc[dict(time=30, frequency=100)] = np.nan
    return array


def test_basic_extraction(sample_data_array: xr.DataArray):
    threshold = 0.1
    max_detections = 3

    actual_result = extract_detections_from_array(
        sample_data_array,
        threshold=threshold,
        max_detections=max_detections,
    )

    expected_values = np.array([0.9, 0.8, 0.7])
    expected_times = np.array([30, 20, 30])
    expected_freqs = np.array([200, 100, 300])
    expected_coords = {
        Dimensions.frequency.value: ("detection", expected_freqs),
        Dimensions.time.value: ("detection", expected_times),
    }
    expected_result = xr.DataArray(
        expected_values,
        coords=expected_coords,
        dims="detection",
        name="score",
    )

    xr.testing.assert_equal(actual_result, expected_result)


def test_threshold_only(sample_data_array):
    input_array = sample_data_array
    threshold = 0.5
    actual_result = extract_detections_from_array(
        input_array, threshold=threshold
    )
    expected_values = np.array([0.9, 0.8, 0.7, 0.6])
    expected_times = np.array([30, 20, 30, 20])
    expected_freqs = np.array([200, 100, 300, 300])
    expected_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }
    expected_result = xr.DataArray(
        expected_values,
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)


def test_max_detections_only(sample_data_array):
    input_array = sample_data_array
    max_detections = 4
    actual_result = extract_detections_from_array(
        input_array, max_detections=max_detections
    )
    expected_values = np.array([0.9, 0.8, 0.7, 0.6])
    expected_times = np.array([30, 20, 30, 20])
    expected_freqs = np.array([200, 100, 300, 300])
    expected_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }
    expected_result = xr.DataArray(
        expected_values,
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)


def test_no_optional_args(sample_data_array):
    input_array = sample_data_array
    actual_result = extract_detections_from_array(input_array)
    expected_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.04, 0.03, 0.02])
    expected_times = np.array([30, 20, 30, 20, 10, 30, 10, 20])
    expected_freqs = np.array([200, 100, 300, 300, 200, 100, 300, 200])
    expected_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }
    expected_result = xr.DataArray(
        expected_values,
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)


def test_no_values_above_threshold(sample_data_array):
    input_array = sample_data_array
    threshold = 1.0
    actual_result = extract_detections_from_array(
        input_array, threshold=threshold
    )
    expected_coords = {
        Dimensions.time.value: ("detection", np.array([], dtype=np.int64)),
        Dimensions.frequency.value: (
            "detection",
            np.array([], dtype=np.int64),
        ),
    }
    expected_result = xr.DataArray(
        np.array([], dtype=np.float64),
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)
    assert actual_result.sizes["detection"] == 0


def test_max_detections_zero(sample_data_array):
    input_array = sample_data_array
    max_detections = 0
    with pytest.raises(ValueError):
        extract_detections_from_array(
            input_array,
            max_detections=max_detections,
        )


def test_empty_input_array():
    empty_array = xr.DataArray(
        np.empty((0, 0)),
        coords={Dimensions.time.value: [], Dimensions.frequency.value: []},
        dims=[Dimensions.time.value, Dimensions.frequency.value],
    )
    actual_result = extract_detections_from_array(empty_array)
    expected_coords = {
        Dimensions.time.value: ("detection", np.array([], dtype=np.int64)),
        Dimensions.frequency.value: (
            "detection",
            np.array([], dtype=np.int64),
        ),
    }
    expected_result = xr.DataArray(
        np.array([], dtype=np.float64),
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)
    assert actual_result.sizes["detection"] == 0


def test_nan_handling(data_array_with_nans):
    input_array = data_array_with_nans
    threshold = 0.1
    max_detections = 3
    actual_result = extract_detections_from_array(
        input_array, threshold=threshold, max_detections=max_detections
    )
    expected_values = np.array([0.9, 0.8, 0.7])
    expected_times = np.array([30, 20, 30])
    expected_freqs = np.array([200, 100, 300])
    expected_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }
    expected_result = xr.DataArray(
        expected_values,
        coords=expected_coords,
        dims="detection",
        name="detection_value",
    )
    xr.testing.assert_equal(actual_result, expected_result)
