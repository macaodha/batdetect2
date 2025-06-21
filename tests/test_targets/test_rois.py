import numpy as np
import pytest
from soundevent import data

from batdetect2.targets.rois import (
    DEFAULT_ANCHOR,
    DEFAULT_FREQUENCY_SCALE,
    DEFAULT_TIME_SCALE,
    SIZE_HEIGHT,
    SIZE_WIDTH,
    AnchorBBoxMapper,
    BBoxAnchorMapperConfig,
    _build_bounding_box,
    build_roi_mapper,
)


@pytest.fixture
def sample_bbox() -> data.BoundingBox:
    """A standard bounding box for testing."""
    return data.BoundingBox(coordinates=[10.0, 100.0, 20.0, 200.0])


@pytest.fixture
def sample_recording(create_recording) -> data.Recording:
    return create_recording(duration=30, samplerate=4_000)


@pytest.fixture
def sample_sound_event(sample_bbox, sample_recording) -> data.SoundEvent:
    return data.SoundEvent(geometry=sample_bbox, recording=sample_recording)


@pytest.fixture
def zero_bbox() -> data.BoundingBox:
    """A bounding box with zero duration and bandwidth."""
    return data.BoundingBox(coordinates=[15.0, 150.0, 15.0, 150.0])


@pytest.fixture
def zero_sound_event(zero_bbox, sample_recording) -> data.SoundEvent:
    """A sample sound event with a zero-sized bounding box."""
    return data.SoundEvent(geometry=zero_bbox, recording=sample_recording)


@pytest.fixture
def default_mapper() -> AnchorBBoxMapper:
    """A BBoxEncoder with default settings."""
    return AnchorBBoxMapper()


@pytest.fixture
def custom_encoder() -> AnchorBBoxMapper:
    """A BBoxEncoder with custom settings."""
    return AnchorBBoxMapper(
        anchor="center", time_scale=1.0, frequency_scale=10.0
    )


@pytest.fixture
def custom_mapper() -> AnchorBBoxMapper:
    """An AnchorBBoxMapper with custom settings."""
    return AnchorBBoxMapper(
        anchor="center", time_scale=1.0, frequency_scale=10.0
    )


def test_bbox_encoder_init_defaults(default_mapper):
    """Test BBoxEncoder initialization with default arguments."""
    assert default_mapper.anchor == DEFAULT_ANCHOR
    assert default_mapper.time_scale == DEFAULT_TIME_SCALE
    assert default_mapper.frequency_scale == DEFAULT_FREQUENCY_SCALE
    assert default_mapper.dimension_names == [SIZE_WIDTH, SIZE_HEIGHT]


def test_bbox_encoder_init_custom(custom_encoder):
    """Test BBoxEncoder initialization with custom arguments."""
    assert custom_encoder.anchor == "center"
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


@pytest.mark.parametrize("anchor, expected_pos", POSITION_TEST_CASES)
def test_anchor_bbox_mapper_encode_position(
    sample_sound_event, anchor, expected_pos
):
    """Test encode returns the correct position for various anchors."""
    encoder = AnchorBBoxMapper(anchor=anchor)
    actual_pos, _ = encoder.encode(sample_sound_event)
    assert actual_pos == pytest.approx(expected_pos)


def test_anchor_bbox_mapper_encode_defaults(
    sample_sound_event, default_mapper
):
    """Test encode with default settings returns correct position and size."""
    expected_pos = (10.0, 100.0)  # bottom-left
    expected_size = np.array(
        [
            10.0 * DEFAULT_TIME_SCALE,
            100.0 * DEFAULT_FREQUENCY_SCALE,
        ]
    )
    actual_pos, actual_size = default_mapper.encode(sample_sound_event)
    assert actual_pos == pytest.approx(expected_pos)
    np.testing.assert_allclose(actual_size, expected_size)
    assert actual_size.shape == (2,)


def test_anchor_bbox_mapper_encode_custom(sample_sound_event, custom_mapper):
    """Test encode with custom settings returns correct position and size."""
    expected_pos = (15.0, 150.0)  # center
    expected_size = np.array([10.0 * 1.0, 100.0 * 10.0])

    actual_pos, actual_size = custom_mapper.encode(sample_sound_event)
    assert actual_pos == pytest.approx(expected_pos)
    np.testing.assert_allclose(actual_size, expected_size)
    assert actual_size.shape == (2,)


def test_anchor_bbox_mapper_encode_zero_box(zero_sound_event, default_mapper):
    """Test encode for a zero-sized box."""
    expected_pos = (15.0, 150.0)
    expected_size = np.array([0.0, 0.0])
    actual_pos, actual_size = default_mapper.encode(zero_sound_event)
    assert actual_pos == pytest.approx(expected_pos)
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
        ref_pos, duration, bandwidth, anchor=position_type
    )
    assert isinstance(bbox, data.BoundingBox)
    np.testing.assert_allclose(bbox.coordinates, expected_coords)


def test_build_bounding_box_invalid_anchor():
    """Test _build_bounding_box raises error for invalid position."""
    with pytest.raises(ValueError, match="Invalid anchor"):
        _build_bounding_box(
            (0, 0),
            1,
            1,
            anchor="invalid-spot",  # type: ignore
        )


@pytest.mark.parametrize(
    "anchor", [anchor for anchor, _ in POSITION_TEST_CASES]
)
def test_anchor_bbox_mapper_encode_decode_roundtrip(
    sample_sound_event, sample_bbox, anchor
):
    """Test encode-decode roundtrip reconstructs the original bbox."""
    mapper = AnchorBBoxMapper(anchor=anchor)
    position, size = mapper.encode(sample_sound_event)
    recovered_bbox = mapper.decode(position, size)

    assert isinstance(recovered_bbox, data.BoundingBox)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, sample_bbox.coordinates, atol=1e-6
    )


def test_anchor_bbox_mapper_roundtrip_custom_scale(
    sample_sound_event, sample_bbox, custom_mapper
):
    """Test encode-decode roundtrip with custom scaling factors."""
    position, size = custom_mapper.encode(sample_sound_event)
    recovered_bbox = custom_mapper.decode(position, size)

    assert isinstance(recovered_bbox, data.BoundingBox)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, sample_bbox.coordinates, atol=1e-6
    )


def test_anchor_bbox_mapper_roundtrip_zero_box(
    zero_sound_event, zero_bbox, default_mapper
):
    """Test encode-decode roundtrip for a zero-sized box."""
    position, size = default_mapper.encode(zero_sound_event)
    recovered_bbox = default_mapper.decode(position, size)
    np.testing.assert_allclose(
        recovered_bbox.coordinates, zero_bbox.coordinates, atol=1e-6
    )


def test_anchor_bbox_mapper_decode_invalid_size_shape(default_mapper):
    """Test decode raises ValueError for incorrect size shape."""
    ref_pos = (10, 100)
    with pytest.raises(ValueError, match="does not have the expected shape"):
        default_mapper.decode(ref_pos, np.array([1.0]))
    with pytest.raises(ValueError, match="does not have the expected shape"):
        default_mapper.decode(ref_pos, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="does not have the expected shape"):
        default_mapper.decode(ref_pos, np.array([[1.0], [2.0]]))


def test_build_roi_mapper():
    """Test build_roi_mapper creates a configured BBoxEncoder."""
    config = BBoxAnchorMapperConfig(
        anchor="top-right", time_scale=2.0, frequency_scale=20.0
    )
    mapper = build_roi_mapper(config)

    assert isinstance(mapper, AnchorBBoxMapper)
    assert mapper.anchor == config.anchor
    assert mapper.time_scale == config.time_scale
    assert mapper.frequency_scale == config.frequency_scale
