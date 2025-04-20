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
    # Expected: (f=300, t=20, s=0.9), (f=200, t=10, s=0.8), (f=300, t=30, s=0.7)
    return extract_detections_from_array(
        sample_data_array, max_detections=3, threshold=None
    )


@pytest.fixture
def sample_positions_top2(sample_data_array):
    """Get top 2 detection positions from sample_data_array."""
    # Expected: (f=300, t=20, s=0.9), (f=200, t=10, s=0.8)
    return extract_detections_from_array(
        sample_data_array, max_detections=2, threshold=None
    )


@pytest.fixture
def empty_positions(sample_data_array):
    """Get an empty positions array (high threshold)."""
    return extract_detections_from_array(
        sample_data_array,
        threshold=1.0,  # No values > 1.0
    )


@pytest.fixture
def sample_sizes_array(sample_data_array):
    """Provides a sample sizes array matching sample_data_array coords."""
    coords = sample_data_array.coords
    # Data: [[0, 1, 2], [3, 4, 5]] # Dim 0 (width)
    #       [[9,10,11], [12,13,14]] # Dim 1 (height)
    # Reshaped: (2, 3, 3) -> (dim, freq, time)
    data = np.array(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],  # width (freq increases down, time across)
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],  # height
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
    # Example: (2 cats, 3 freqs, 3 times)
    data = np.linspace(0.1, 0.9, 18, dtype=np.float32).reshape(2, 3, 3)
    # data[0, 2, 1] -> cat=0, f=300, t=20 -> val for 0.9 detection
    # data[0, 1, 0] -> cat=0, f=200, t=10 -> val for 0.8 detection
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
    # Example: (4 features, 3 freqs, 3 times)
    data = np.arange(0, 36, dtype=np.float32).reshape(4, 3, 3)
    # data[:, 2, 1] -> feats, f=300, t=20 -> vals for 0.9 detection
    # data[:, 1, 0] -> feats, f=200, t=10 -> vals for 0.8 detection
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


# --- Tests for extract_values_at_positions ---


def test_extract_values_at_positions_correct(
    sample_array_for_extraction, sample_positions_top3
):
    """Verify correct values are extracted based on positions coords."""
    # Positions: (f=300, t=20), (f=200, t=10), (f=300, t=30)
    # Corresponding values in sample_array_for_extraction (1-9):
    # f=300, t=20 -> index (2, 1) -> value 8
    # f=200, t=10 -> index (1, 0) -> value 4
    # f=300, t=30 -> index (2, 2) -> value 9
    expected_values = np.array([8, 4, 9])

    print(sample_positions_top3)

    expected = xr.DataArray(
        expected_values,
        coords=sample_positions_top3.coords,  # Should inherit coords
        dims="detection",
        name="test_values",  # Should inherit name
    )

    extracted = extract_values_at_positions(
        sample_array_for_extraction, sample_positions_top3
    )

    xr.testing.assert_allclose(extracted, expected)


def test_extract_values_at_positions_extra_dims(
    sample_sizes_array, sample_positions_top2
):
    """Test extraction preserves other dimensions in the source array."""
    # Positions: (f=300, t=20), (f=200, t=10)
    # Extract from sample_sizes_array (dim, freq, time)
    # Det 1 (f=300, t=20) -> index (:, 2, 1) -> values [7, 16]
    # Det 2 (f=200, t=10) -> index (:, 1, 0) -> values [3, 12]
    # Expected shape: (dimension, detection)
    expected_values = np.array([[7.0, 3.0], [16.0, 12.0]], dtype=np.float32)

    expected = xr.DataArray(
        expected_values,
        coords={
            "dimension": ["width", "height"],
            Dimensions.frequency.value: sample_positions_top2.coords[
                Dimensions.frequency.value
            ],
            Dimensions.time.value: sample_positions_top2.coords[
                Dimensions.time.value
            ],
        },
        dims=["dimension", "detection"],
        name="sizes",  # Inherits name
    )

    extracted = extract_values_at_positions(
        sample_sizes_array, sample_positions_top2
    )
    xr.testing.assert_allclose(extracted, expected)


def test_extract_values_at_positions_empty(
    sample_array_for_extraction, empty_positions
):
    """Test extraction with empty positions returns empty array."""
    extracted = extract_values_at_positions(
        sample_array_for_extraction, empty_positions
    )
    assert extracted.sizes["detection"] == 0
    # Check coordinates are also empty but defined
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
    # Create positions requesting a time=40 not present in sample_array
    bad_positions = sample_positions_top2.copy()
    bad_positions.coords[Dimensions.time.value] = (
        "detection",
        np.array([40, 10]),  # First time is invalid
    )
    with pytest.raises(
        KeyError
    ):  # xarray.sel raises KeyError for missing labels
        extract_values_at_positions(sample_array_for_extraction, bad_positions)


# --- Tests for extract_detection_xr_dataset ---


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

    # Expected positions (top 2):
    # 1. Score 0.9, Time 20, Freq 300. Indices (freq=2, time=1)
    # 2. Score 0.8, Time 10, Freq 200. Indices (freq=1, time=0)
    expected_times = np.array([20, 10])
    expected_freqs = np.array([300, 200])
    detection_coords = {
        Dimensions.time.value: ("detection", expected_times),
        Dimensions.frequency.value: ("detection", expected_freqs),
    }

    # --- Manually Calculate Expected Data ---

    # Scores (already correct in sample_positions_top2)
    expected_score = sample_positions_top2.rename(
        "scores"
    )  # Rename to match output

    # Dimensions Data (width, height) -> Transposed to (detection, dimension)
    # sample_sizes_array data: (dim, freq, time)
    # Det 1 (f=300, t=20): index (:, 2, 1) -> values [ 7., 16.]
    # Det 2 (f=200, t=10): index (:, 1, 0) -> values [ 3., 12.]
    expected_dimensions_data = np.array(
        [
            [7.0, 16.0],  # Detection 1 [width, height]
            [3.0, 12.0],
        ],  # Detection 2 [width, height]
        dtype=np.float32,
    )
    expected_dimensions = xr.DataArray(
        expected_dimensions_data,
        coords={**detection_coords, "dimension": ["width", "height"]},
        dims=["detection", "dimension"],
        name="dimensions",
    )

    # Classes Data (bat, noise) -> Transposed to (detection, category)
    # sample_classes_array data: np.linspace(0.1, 0.9, 18).reshape(2, 3, 3)
    # linspace vals: [0.1, 0.147, 0.194, 0.241, 0.288, 0.335, 0.382, 0.429, 0.476, # cat 0
    #                 0.523, 0.570, 0.617, 0.664, 0.711, 0.758, 0.805, 0.852, 0.9]  # cat 1
    # Det 1 (cat, f=2, t=1): index (:, 2, 1) -> values [idx 7=0.429, idx 16=0.852]
    # Det 2 (cat, f=1, t=0): index (:, 1, 0) -> values [idx 3=0.241, idx 12=0.664]
    expected_classes_data = np.array(
        [
            [0.42941177, 0.85294118],  # Detection 1 [bat_prob, noise_prob]
            [0.24117647, 0.66470588],
        ],  # Detection 2 [bat_prob, noise_prob]
        dtype=np.float32,
    )
    expected_classes = xr.DataArray(
        expected_classes_data,
        coords={**detection_coords, "category": ["bat", "noise"]},
        dims=["detection", "category"],
        name="classes",
    )

    # Features Data (f0..f3) -> Transposed to (detection, feature)
    # sample_features_array data: np.arange(36).reshape(4, 3, 3)
    # Det 1 (feat, f=2, t=1): index (:, 2, 1) -> values [ 7, 16, 25, 34]
    # Det 2 (feat, f=1, t=0): index (:, 1, 0) -> values [ 3, 12, 21, 30]
    expected_features_data = np.array(
        [
            [7.0, 16.0, 25.0, 34.0],  # Detection 1 [f0, f1, f2, f3]
            [3.0, 12.0, 21.0, 30.0],
        ],  # Detection 2 [f0, f1, f2, f3]
        dtype=np.float32,
    )
    expected_features = xr.DataArray(
        expected_features_data,
        coords={**detection_coords, "feature": ["f0", "f1", "f2", "f3"]},
        dims=["detection", "feature"],
        name="features",
    )

    # Construct Expected Dataset
    expected_dataset = xr.Dataset(
        {
            "scores": expected_score,
            "dimensions": expected_dimensions,
            "classes": expected_classes,
            "features": expected_features,
        }
    )
    # Add coords explicitly to ensure they match
    expected_dataset = expected_dataset.assign_coords(detection_coords)

    # --- Assert Equality ---
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
    assert actual_dataset.dims["detection"] == 0

    # Check variables exist and have 0 size along detection dim
    assert "scores" in actual_dataset
    assert actual_dataset["scores"].dims == ("detection",)
    assert actual_dataset["scores"].size == 0

    assert "dimensions" in actual_dataset
    assert actual_dataset["dimensions"].dims == ("detection", "dimension")
    assert actual_dataset["dimensions"].shape == (0, 2)  # Check both dims

    assert "classes" in actual_dataset
    assert actual_dataset["classes"].dims == ("detection", "category")
    assert actual_dataset["classes"].shape == (0, 2)

    assert "features" in actual_dataset
    assert actual_dataset["features"].dims == ("detection", "feature")
    assert actual_dataset["features"].shape == (0, 4)

    # Check coordinates exist and are empty
    assert Dimensions.time.value in actual_dataset.coords
    assert Dimensions.frequency.value in actual_dataset.coords
    assert actual_dataset.coords[Dimensions.time.value].size == 0
    assert actual_dataset.coords[Dimensions.frequency.value].size == 0
