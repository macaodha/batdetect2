import numpy as np
import pytest
from soundevent import data

from batdetect2.targets.rois import (
    DEFAULT_FREQUENCY_SCALE,
    DEFAULT_POSITION,
    DEFAULT_TIME_SCALE,
    SIZE_HEIGHT,
    SIZE_WIDTH,
    BBoxEncoder,
    ROIConfig,
    _build_bounding_box,
    build_roi_mapper,
    load_roi_mapper,
)


@pytest.fixture
def sample_bbox() -> data.BoundingBox:
    """A standard bounding box for testing."""
    return data.BoundingBox(coordinates=[10.0, 100.0, 20.0, 200.0])


@pytest.fixture
def zero_bbox() -> data.BoundingBox:
    """A bounding box with zero duration and bandwidth."""
    return data.BoundingBox(coordinates=[15.0, 150.0, 15.0, 150.0])


@pytest.fixture
def default_encoder() -> BBoxEncoder:
    """A BBoxEncoder with default settings."""
    return BBoxEncoder()


@pytest.fixture
def custom_encoder() -> BBoxEncoder:
    """A BBoxEncoder with custom settings."""
    return BBoxEncoder(position="center", time_scale=1.0, frequency_scale=10.0)


def test_roi_config_defaults():
    """Test ROIConfig default values."""
    config = ROIConfig()
    assert config.position == DEFAULT_POSITION
    assert config.time_scale == DEFAULT_TIME_SCALE
    assert config.frequency_scale == DEFAULT_FREQUENCY_SCALE


def test_roi_config_custom():
    """Test creating ROIConfig with custom values."""
    config = ROIConfig(position="center", time_scale=1.0, frequency_scale=10.0)
    assert config.position == "center"
    assert config.time_scale == 1.0
    assert config.frequency_scale == 10.0


def test_bbox_encoder_init_defaults(default_encoder):
    """Test BBoxEncoder initialization with default arguments."""
    assert default_encoder.position == DEFAULT_POSITION
    assert default_encoder.time_scale == DEFAULT_TIME_SCALE
    assert default_encoder.frequency_scale == DEFAULT_FREQUENCY_SCALE
    assert default_encoder.dimension_names == [SIZE_WIDTH, SIZE_HEIGHT]


def test_bbox_encoder_init_custom(custom_encoder):
    """Test BBoxEncoder initialization with custom arguments."""
    assert custom_encoder.position == "center"
    assert custom_encoder.time_scale == 1.0
    assert custom_encoder.frequency_scale == 10.0
    assert custom_encoder.dimension_names == [SIZE_WIDTH, SIZE_HEIGHT]


POSITION_TEST_CASES = [
    ("bottom-left", (10.0, 100.0)),
    ("bottom-right", (20.0, 100.0)),
    ("top-left", (10.0, 200.0)),
    ("top-right", (20.0, 200.0)),
    ("center-left", (10.0, 150.0)),
    ("center-right", (20.0, 150.0)),
    ("top-center", (15.0, 200.0)),
    ("bottom-center", (15.0, 100.0)),
    ("center", (15.0, 150.0)),
    ("centroid", (15.0, 150.0)),
    ("point_on_surface", (15.0, 150.0)),
]


@pytest.mark.parametrize("position_type, expected_pos", POSITION_TEST_CASES)
def test_bbox_encoder_get_roi_position(
    sample_bbox, position_type, expected_pos
):
    """Test get_roi_position for various position types."""
    encoder = BBoxEncoder(position=position_type)
    actual_pos = encoder.get_roi_position(sample_bbox)
    assert actual_pos == pytest.approx(expected_pos)


def test_bbox_encoder_get_roi_position_zero_box(zero_bbox):
    """Test get_roi_position for a zero-sized box."""
    encoder = BBoxEncoder(position="center")
    assert encoder.get_roi_position(zero_bbox) == pytest.approx((15.0, 150.0))


def test_bbox_encoder_get_roi_size_defaults(sample_bbox, default_encoder):
    """Test get_roi_size with default scaling."""
    expected_size = np.array(
        [
            10.0 * DEFAULT_TIME_SCALE,
            100.0 * DEFAULT_FREQUENCY_SCALE,
        ]
    )
    actual_size = default_encoder.get_roi_size(sample_bbox)
    np.testing.assert_allclose(actual_size, expected_size)
    assert actual_size.shape == (2,)


def test_bbox_encoder_get_roi_size_custom(sample_bbox, custom_encoder):
    """Test get_roi_size with custom scaling."""
    expected_size = np.array(
        [
            10.0 * 1.0,
            100.0 * 10.0,
        ]
    )
    actual_size = custom_encoder.get_roi_size(sample_bbox)
    np.testing.assert_allclose(actual_size, expected_size)
    assert actual_size.shape == (2,)


def test_bbox_encoder_get_roi_size_zero_box(zero_bbox, default_encoder):
    """Test get_roi_size for a zero-sized box."""
    expected_size = np.array([0.0, 0.0])
    actual_size = default_encoder.get_roi_size(zero_bbox)
    np.testing.assert_allclose(actual_size, expected_size)


BUILD_BOX_TEST_CASES = [
    ("bottom-left", [50.0, 500.0, 60.0, 600.0]),
    ("bottom-right", [40.0, 500.0, 50.0, 600.0]),
    ("top-left", [50.0, 400.0, 60.0, 500.0]),
    ("top-right", [40.0, 400.0, 50.0, 500.0]),
    ("center-left", [50.0, 450.0, 60.0, 550.0]),
    ("center-right", [40.0, 450.0, 50.0, 550.0]),
    ("top-center", [45.0, 400.0, 55.0, 500.0]),
    ("bottom-center", [45.0, 500.0, 55.0, 600.0]),
    ("center", [45.0, 450.0, 55.0, 550.0]),
    ("centroid", [45.0, 450.0, 55.0, 550.0]),
    ("point_on_surface", [45.0, 450.0, 55.0, 550.0]),
]


@pytest.mark.parametrize(
    "position_type, expected_coords", BUILD_BOX_TEST_CASES
)
def test_build_bounding_box(position_type, expected_coords):
    """Test _build_bounding_box for various position types."""
    ref_pos = (50.0, 500.0)
    duration = 10.0
    bandwidth = 100.0
    bbox = _build_bounding_box(
        ref_pos, duration, bandwidth, position=position_type
    )
    assert isinstance(bbox, data.BoundingBox)
    np.testing.assert_allclose(bbox.coordinates, expected_coords)


def test_build_bounding_box_invalid_position():
    """Test _build_bounding_box raises error for invalid position."""
    with pytest.raises(ValueError, match="Invalid position"):
        _build_bounding_box(
            (0, 0),
            1,
            1,
            position="invalid-spot",  # type: ignore
        )


@pytest.mark.parametrize("position_type, ref_pos", POSITION_TEST_CASES)
def test_bbox_encoder_recover_roi(sample_bbox, position_type, ref_pos):
    """Test recover_roi correctly reconstructs the original bbox."""
    encoder = BBoxEncoder(position=position_type)
    scaled_dims = encoder.get_roi_size(sample_bbox)

    recovered_bbox = encoder.recover_roi(ref_pos, scaled_dims)

    assert isinstance(recovered_bbox, data.BoundingBox)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, sample_bbox.coordinates, atol=1e-6
    )


def test_bbox_encoder_recover_roi_custom_scale(sample_bbox, custom_encoder):
    """Test recover_roi with custom scaling factors."""
    ref_pos = custom_encoder.get_roi_position(sample_bbox)
    scaled_dims = custom_encoder.get_roi_size(sample_bbox)

    recovered_bbox = custom_encoder.recover_roi(ref_pos, scaled_dims)

    assert isinstance(recovered_bbox, data.BoundingBox)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, sample_bbox.coordinates, atol=1e-6
    )


def test_bbox_encoder_recover_roi_zero_box(zero_bbox, default_encoder):
    """Test recover_roi for a zero-sized box."""
    ref_pos = default_encoder.get_roi_position(zero_bbox)
    scaled_dims = default_encoder.get_roi_size(zero_bbox)
    recovered_bbox = default_encoder.recover_roi(ref_pos, scaled_dims)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, zero_bbox.coordinates, atol=1e-6
    )


def test_bbox_encoder_recover_roi_invalid_dims_shape(default_encoder):
    """Test recover_roi raises ValueError for incorrect dims shape."""
    ref_pos = (10, 100)
    with pytest.raises(ValueError):
        default_encoder.recover_roi(ref_pos, np.array([1.0]))
    with pytest.raises(ValueError):
        default_encoder.recover_roi(ref_pos, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        default_encoder.recover_roi(ref_pos, np.array([[1.0], [2.0]]))


def test_build_roi_mapper():
    """Test build_roi_mapper creates a configured BBoxEncoder."""
    config = ROIConfig(
        position="top-right", time_scale=2.0, frequency_scale=20.0
    )
    mapper = build_roi_mapper(config)

    assert isinstance(mapper, BBoxEncoder)
    assert mapper.position == config.position
    assert mapper.time_scale == config.time_scale
    assert mapper.frequency_scale == config.frequency_scale


@pytest.fixture
def sample_config_yaml_content() -> str:
    """YAML content for a sample ROIConfig."""
    return f"""
position: center
time_scale: 500.0
frequency_scale: {1 / 1000.0}
"""


@pytest.fixture
def nested_config_yaml_content() -> str:
    """YAML content with ROIConfig nested under a field."""
    return f"""
model_settings:
  preprocessing:
    whatever: true
  roi_mapping:
    position: bottom-right
    time_scale: {DEFAULT_TIME_SCALE}
    frequency_scale: 0.01
other_stuff: 123
"""


def test_load_roi_mapper_simple(tmp_path, sample_config_yaml_content):
    """Test loading a simple ROIConfig from YAML."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(sample_config_yaml_content)

    mapper = load_roi_mapper(config_path)

    assert isinstance(mapper, BBoxEncoder)
    assert mapper.position == "center"
    assert mapper.time_scale == 500.0
    assert mapper.frequency_scale == pytest.approx(1 / 1000.0)


def test_load_roi_mapper_nested(tmp_path, nested_config_yaml_content):
    """Test loading a nested ROIConfig from YAML using 'field'."""
    config_path = tmp_path / "nested_config.yaml"
    config_path.write_text(nested_config_yaml_content)

    mapper = load_roi_mapper(config_path, field="model_settings.roi_mapping")

    assert isinstance(mapper, BBoxEncoder)
    assert mapper.position == "bottom-right"
    assert mapper.time_scale == DEFAULT_TIME_SCALE
    assert mapper.frequency_scale == 0.01


def test_load_roi_mapper_file_not_found(tmp_path):
    """Test load_roi_mapper raises error if file doesn't exist."""
    non_existent_path = tmp_path / "not_real.yaml"
    with pytest.raises(FileNotFoundError):
        load_roi_mapper(non_existent_path)


def test_load_roi_mapper_invalid_field(tmp_path, sample_config_yaml_content):
    """Test load_roi_mapper raises error for invalid field."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(sample_config_yaml_content)
    with pytest.raises(KeyError):
        load_roi_mapper(config_path, field="invalid.path")
