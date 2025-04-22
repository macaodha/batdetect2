import numpy as np
import pytest
import xarray as xr
from soundevent.arrays import Dimensions

from batdetect2.postprocess.detection import extract_detections_from_array
from batdetect2.postprocess.extraction import (
    extract_detection_xr_dataset,
    extract_values_at_positions,
)


@pytest.fixture
def sample_data_array():
    """Provides a basic 3x3 DataArray.
    Top values: 0.9 (f=300, t=20), 0.8 (f=200, t=10), 0.7 (f=300, t=30)
    """
    coords = {
        Dimensions.frequency.value: [100, 200, 300],
        Dimensions.time.value: [10, 20, 30],
    }
    array = xr.DataArray(
        np.zeros([3, 3]),
        coords=coords,
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
def sample_array_for_extraction():
    """Provides a simple array (1-9) for value extraction tests."""
    data = np.arange(1, 10).reshape(3, 3)
    coords = {
        Dimensions.frequency.value: [100, 200, 300],
        Dimensions.time.value: [10, 20, 30],
    }
    return xr.DataArray(
        data,
        coords=coords,
        dims=[
            Dimensions.frequency.value,
            Dimensions.time.value,
        ],
        name="test_values",
    )


@pytest.fixture
def sample_positions_top3(sample_data_array):
    """Get top 3 detection positions from sample_data_array."""

    return extract_detections_from_array(
        sample_data_array,
        max_detections=3,
        threshold=None,
    )


@pytest.fixture
def sample_positions_top2(sample_data_array):
    """Get top 2 detection positions from sample_data_array."""
    return extract_detections_from_array(
        sample_data_array,
        max_detections=2,
        threshold=None,
    )


@pytest.fixture
def empty_positions(sample_data_array):
    """Get an empty positions array (high threshold)."""
    return extract_detections_from_array(
        sample_data_array,
        threshold=1.0,
    )


@pytest.fixture
def sample_sizes_array(sample_data_array):
    """Provides a sample sizes array matching sample_data_array coords."""
    coords = sample_data_array.coords
    data = np.array(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            [
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17],
            ],
        ],
        dtype=np.float32,
    )

    return xr.DataArray(
        data,
        coords={
            "dimension": ["width", "height"],
            Dimensions.frequency.value: coords[Dimensions.frequency.value],
            Dimensions.time.value: coords[Dimensions.time.value],
        },
        dims=["dimension", Dimensions.frequency.value, Dimensions.time.value],
        name="sizes",
    )


@pytest.fixture
def sample_classes_array(sample_data_array):
    """Provides a sample classes array matching sample_data_array coords."""
    coords = sample_data_array.coords
    data = np.linspace(0.1, 0.9, 18, dtype=np.float32).reshape(2, 3, 3)
    return xr.DataArray(
        data,
        coords={
            "category": ["bat", "noise"],
            Dimensions.frequency.value: coords[Dimensions.frequency.value],
            Dimensions.time.value: coords[Dimensions.time.value],
        },
        dims=["category", Dimensions.frequency.value, Dimensions.time.value],
        name="class_scores",
    )


@pytest.fixture
def sample_features_array(sample_data_array):
    """Provides a sample features array matching sample_data_array coords."""
    coords = sample_data_array.coords
    data = np.arange(0, 36, dtype=np.float32).reshape(4, 3, 3)
    return xr.DataArray(
        data,
        coords={
            "feature": ["f0", "f1", "f2", "f3"],
            Dimensions.frequency.value: coords[Dimensions.frequency.value],
            Dimensions.time.value: coords[Dimensions.time.value],
        },
        dims=["feature", Dimensions.frequency.value, Dimensions.time.value],
        name="features",
    )


def test_extract_values_at_positions_correct(
    sample_array_for_extraction,
    sample_positions_top3,
):
    """Verify correct values are extracted based on positions coords."""
    expected_values = np.array(
        [
            sample_array_for_extraction.sel(time=30, frequency=200).values,
            sample_array_for_extraction.sel(time=20, frequency=100).values,
            sample_array_for_extraction.sel(time=30, frequency=300).values,
        ]
    )

    expected = xr.DataArray(
        expected_values,
        coords=sample_positions_top3.coords,
        dims="detection",
        name="test_values",
    )

    extracted = extract_values_at_positions(
        sample_array_for_extraction, sample_positions_top3
    )

    xr.testing.assert_allclose(extracted, expected)


def test_extract_values_at_positions_extra_dims(
    sample_sizes_array,
    sample_positions_top2,
):
    """Test extraction preserves other dimensions in the source array."""
    times = np.array([30, 20])
    freqs = np.array([200, 100])

    expected_values = np.array(
        [
            sample_sizes_array.sel(time=30, frequency=200).values,
            sample_sizes_array.sel(time=20, frequency=100).values,
        ],
        dtype=np.float32,
    )

    expected = xr.DataArray(
        expected_values,
        coords={
            "dimension": ["width", "height"],
            Dimensions.frequency.value: ("detection", freqs),
            Dimensions.time.value: ("detection", times),
        },
        dims=["detection", "dimension"],
        name="sizes",
    )

    extracted = extract_values_at_positions(
        sample_sizes_array,
        sample_positions_top2,
    )

    xr.testing.assert_equal(extracted, expected)


def test_extract_values_at_positions_empty(
    sample_array_for_extraction, empty_positions
):
    """Test extraction with empty positions returns empty array."""
    extracted = extract_values_at_positions(
        sample_array_for_extraction, empty_positions
    )
    assert extracted.sizes["detection"] == 0
    assert Dimensions.time.value in extracted.coords
    assert Dimensions.frequency.value in extracted.coords
    assert extracted.coords[Dimensions.time.value].size == 0
    assert extracted.coords[Dimensions.frequency.value].size == 0
    assert extracted.name == sample_array_for_extraction.name


def test_extract_values_at_positions_missing_coord_in_array(
    sample_array_for_extraction, sample_positions_top2
):
    """Test error if source array misses required coordinates."""
    array_no_time = sample_array_for_extraction.copy()
    del array_no_time.coords[Dimensions.time.value]
    with pytest.raises(IndexError):
        extract_values_at_positions(array_no_time, sample_positions_top2)

    array_no_freq = sample_array_for_extraction.copy()
    del array_no_freq.coords[Dimensions.frequency.value]
    with pytest.raises(IndexError):
        extract_values_at_positions(array_no_freq, sample_positions_top2)


def test_extract_values_at_positions_missing_coord_in_positions(
    sample_array_for_extraction, sample_positions_top2
):
    """Test error if positions array misses required coordinates."""
    positions_no_time = sample_positions_top2.copy()
    del positions_no_time.coords[Dimensions.time.value]
    with pytest.raises(KeyError):
        extract_values_at_positions(
            sample_array_for_extraction, positions_no_time
        )

    positions_no_freq = sample_positions_top2.copy()
    del positions_no_freq.coords[Dimensions.frequency.value]
    with pytest.raises(KeyError):
        extract_values_at_positions(
            sample_array_for_extraction, positions_no_freq
        )


def test_extract_values_at_positions_mismatched_coords(
    sample_array_for_extraction, sample_positions_top2
):
    """Test error if positions requests coords not in source array."""
    bad_positions = sample_positions_top2.copy()
    bad_positions.coords[Dimensions.time.value] = (
        "detection",
        np.array([40, 10]),
    )
    with pytest.raises(KeyError):
        extract_values_at_positions(sample_array_for_extraction, bad_positions)


def test_extract_detection_xr_dataset_correct(
    sample_positions_top2,
    sample_sizes_array,
    sample_classes_array,
    sample_features_array,
):
    """Tests extracting and bundling info for top 2 detections."""
    actual_dataset = extract_detection_xr_dataset(
        sample_positions_top2,
        sample_sizes_array,
        sample_classes_array,
        sample_features_array,
    )

    expected_times = np.array([30, 20])
    expected_freqs = np.array([200, 100])
    detection_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }

    expected_score = sample_positions_top2

    expected_dimensions_data = np.array(
        [
            sample_sizes_array.sel(time=30, frequency=200).values,
            sample_sizes_array.sel(time=20, frequency=100).values,
        ],
        dtype=np.float32,
    )
    expected_dimensions = xr.DataArray(
        expected_dimensions_data,
        coords={**detection_coords, "dimension": ["width", "height"]},
        dims=["detection", "dimension"],
        name="dimensions",
    )

    expected_classes_data = np.array(
        [
            sample_classes_array.sel(time=30, frequency=200).values,
            sample_classes_array.sel(time=20, frequency=100).values,
        ],
        dtype=np.float32,
    )
    expected_classes = xr.DataArray(
        expected_classes_data,
        coords={**detection_coords, "category": ["bat", "noise"]},
        dims=["detection", "category"],
        name="classes",
    )

    expected_features_data = np.array(
        [
            sample_features_array.sel(time=30, frequency=200).values,
            sample_features_array.sel(time=20, frequency=100).values,
        ],
        dtype=np.float32,
    )
    expected_features = xr.DataArray(
        expected_features_data,
        coords={**detection_coords, "feature": ["f0", "f1", "f2", "f3"]},
        dims=["detection", "feature"],
        name="features",
    )

    expected_dataset = xr.Dataset(
        {
            "scores": expected_score,
            "dimensions": expected_dimensions,
            "classes": expected_classes,
            "features": expected_features,
        }
    )
    expected_dataset = expected_dataset.assign_coords(detection_coords)

    xr.testing.assert_allclose(actual_dataset, expected_dataset)


def test_extract_detection_xr_dataset_empty(
    empty_positions,
    sample_sizes_array,
    sample_classes_array,
    sample_features_array,
):
    """Test extraction with empty positions yields an empty dataset."""
    actual_dataset = extract_detection_xr_dataset(
        empty_positions,
        sample_sizes_array,
        sample_classes_array,
        sample_features_array,
    )

    assert isinstance(actual_dataset, xr.Dataset)
    assert "detection" in actual_dataset.dims
    assert actual_dataset.sizes["detection"] == 0

    assert "scores" in actual_dataset
    assert actual_dataset["scores"].dims == ("detection",)
    assert actual_dataset["scores"].size == 0

    assert "dimensions" in actual_dataset
    assert actual_dataset["dimensions"].dims == ("detection", "dimension")
    assert actual_dataset["dimensions"].shape == (0, 2)

    assert "classes" in actual_dataset
    assert actual_dataset["classes"].dims == ("detection", "category")
    assert actual_dataset["classes"].shape == (0, 2)

    assert "features" in actual_dataset
    assert actual_dataset["features"].dims == ("detection", "feature")
    assert actual_dataset["features"].shape == (0, 4)

    assert Dimensions.time.value in actual_dataset.coords
    assert Dimensions.frequency.value in actual_dataset.coords
    assert actual_dataset.coords[Dimensions.time.value].size == 0
    assert actual_dataset.coords[Dimensions.frequency.value].size == 0
