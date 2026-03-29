from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.preprocess import (
    PreprocessingConfig,
    build_preprocessor,
)
from batdetect2.preprocess.spectrogram import (
    ScaleAmplitudeConfig,
    SpectralMeanSubtractionConfig,
)
from batdetect2.targets.rois import (
    DEFAULT_ANCHOR,
    DEFAULT_FREQUENCY_SCALE,
    DEFAULT_TIME_SCALE,
    SIZE_HEIGHT,
    SIZE_WIDTH,
    AnchorBBoxMapper,
    AnchorBBoxMapperConfig,
    PeakEnergyBBoxMapper,
    PeakEnergyBBoxMapperConfig,
    ROIMappingConfig,
    _build_bounding_box,
    build_roi_mapper,
    build_roi_mapping,
    get_peak_energy_coordinates,
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
    config = AnchorBBoxMapperConfig(
        anchor="top-right", time_scale=2.0, frequency_scale=20.0
    )
    mapper = build_roi_mapper(config)

    assert isinstance(mapper, AnchorBBoxMapper)
    assert mapper.anchor == config.anchor
    assert mapper.time_scale == config.time_scale
    assert mapper.frequency_scale == config.frequency_scale


def test_get_peak_energy_coordinates(generate_whistle):
    whistle_time = 0.5
    whistle_frequency = 40_000
    duration = 1.0
    samplerate = 256_000

    # Generate a WAV file with a whistle
    whistle_path = generate_whistle(
        time=whistle_time,
        frequency=whistle_frequency,
        duration=duration,
        samplerate=samplerate,
        whistle_duration=0.01,
    )

    # Create a recording object from the generated WAV
    recording = data.Recording.from_file(path=whistle_path)

    # Build a preprocessor (default config should be fine for this test)
    preprocessor = build_preprocessor()

    audio_loader = build_audio_loader()

    # Define a region of interest that contains the whistle
    start_time = 0.2
    end_time = 0.7
    low_freq = 20_000
    high_freq = 60_000

    # Get the peak energy coordinates
    peak_time, peak_freq = get_peak_energy_coordinates(
        recording=recording,
        preprocessor=preprocessor,
        audio_loader=audio_loader,
        start_time=start_time,
        end_time=end_time,
        low_freq=low_freq,
        high_freq=high_freq,
        loading_buffer=0.05,
    )

    # Assert that the peak coordinates are close to the expected values
    assert peak_time == pytest.approx(whistle_time, abs=0.01)
    assert peak_freq == pytest.approx(whistle_frequency, abs=1000)


def test_get_peak_energy_coordinates_with_two_whistles(generate_whistle):
    # Parameters for the first (stronger) whistle
    strong_whistle_time = 0.2
    strong_whistle_frequency = 30_000
    strong_whistle_amplitude = 1.0  # Full amplitude

    # Parameters for the second (weaker) whistle
    weak_whistle_time = 0.8
    weak_whistle_frequency = 50_000
    weak_whistle_amplitude = 0.1  # Weaker amplitude

    # Recording parameters
    duration = 1.0
    samplerate = 256_000

    # Generate WAV files for each whistle
    strong_whistle_path = generate_whistle(
        time=strong_whistle_time,
        frequency=strong_whistle_frequency,
        duration=duration,
        samplerate=samplerate,
        whistle_duration=0.01,
    )
    weak_whistle_path = generate_whistle(
        time=weak_whistle_time,
        frequency=weak_whistle_frequency,
        duration=duration,
        samplerate=samplerate,
        whistle_duration=0.01,
    )

    # Load audio data
    strong_audio, _ = sf.read(strong_whistle_path)
    weak_audio, _ = sf.read(weak_whistle_path)

    # Mix the audio files
    mixed_audio = (
        strong_audio * strong_whistle_amplitude
        + weak_audio * weak_whistle_amplitude
    )
    mixed_audio_path = strong_whistle_path.parent / "mixed_whistles.wav"
    sf.write(str(mixed_audio_path), mixed_audio, samplerate)

    # Create a recording object from the mixed WAV
    recording = data.Recording.from_file(path=mixed_audio_path)

    # Build a preprocessor
    preprocessor = build_preprocessor()

    # Define a region of interest that contains only the weaker whistle
    start_time = 0.7
    end_time = 0.9
    low_freq = 45_000
    high_freq = 55_000

    # Get the peak energy coordinates within the bounding box
    peak_time, peak_freq = get_peak_energy_coordinates(
        recording=recording,
        preprocessor=preprocessor,
        audio_loader=build_audio_loader(),
        start_time=start_time,
        end_time=end_time,
        low_freq=low_freq,
        high_freq=high_freq,
        loading_buffer=0.05,
    )

    # Assert that the peak coordinates are close to the weaker whistle's values
    assert peak_time == pytest.approx(weak_whistle_time, abs=0.01)
    assert peak_freq == pytest.approx(weak_whistle_frequency, abs=1000)


def test_get_peak_energy_coordinates_silent_region(create_recording):
    # Parameters for a silent recording
    duration = 2.0  # seconds
    samplerate = 44_100  # Hz

    # Create a silent recording
    recording = create_recording(duration=duration, samplerate=samplerate)

    # Build a preprocessor
    preprocessor = build_preprocessor()

    # Define a region of interest within the silent recording
    start_time = 0.5
    end_time = 1.5
    low_freq = 10_000
    high_freq = 20_000

    # Get the peak energy coordinates from the silent region
    peak_time, peak_freq = get_peak_energy_coordinates(
        recording=recording,
        preprocessor=preprocessor,
        audio_loader=build_audio_loader(),
        start_time=start_time,
        end_time=end_time,
        low_freq=low_freq,
        high_freq=high_freq,
        loading_buffer=0.05,
    )

    # Assert that the peak coordinates are within the defined ROI bounds
    assert start_time <= peak_time <= end_time
    assert low_freq <= peak_freq <= high_freq

    # Since there's no actual peak, the exact values might vary depending on
    # argmax behavior with all-zero or very low, uniform energy. We just need
    # to ensure they are within the search bounds.


def test_peak_energy_bbox_mapper_encode(generate_whistle):
    """
    Tests the 'happy path' for PeakEnergyBBoxMapper.encode.

    It verifies that the method correctly identifies a known peak within a
    bounding box and calculates the four scaled distances to the box edges.
    """
    # 1. SETUP
    samplerate = 256_000
    time_scale = 100.0
    freq_scale = 0.1

    bbox_start_time, bbox_low_freq = 1.0, 10000
    bbox_end_time, bbox_high_freq = 2.0, 30000
    bbox = data.BoundingBox(
        coordinates=[
            bbox_start_time,
            bbox_low_freq,
            bbox_end_time,
            bbox_high_freq,
        ]
    )

    # Define the known location of the peak energy inside the bbox
    peak_time, peak_freq = 1.6, 25000

    # Create a recording with a whistle at the defined peak location
    recording_path = generate_whistle(
        time=peak_time,
        frequency=peak_freq,
        duration=3.0,
        samplerate=samplerate,
    )
    recording = data.Recording.from_file(path=recording_path)
    sound_event = data.SoundEvent(geometry=bbox, recording=recording)

    # Instantiate the mapper with a preprocessor
    preprocessor = build_preprocessor(
        PreprocessingConfig(spectrogram_transforms=[])
    )
    mapper = PeakEnergyBBoxMapper(
        preprocessor=preprocessor,
        audio_loader=build_audio_loader(),
        time_scale=time_scale,
        frequency_scale=freq_scale,
    )

    # Encode the sound event to get the position and size
    actual_pos, actual_size = mapper.encode(sound_event)

    # Then
    assert actual_pos[0] == pytest.approx(peak_time, abs=0.01)
    assert actual_pos[1] == pytest.approx(peak_freq, abs=1000)

    # Assert that the calculated scaled distances are correct
    identified_time, identified_freq = actual_pos
    expected_left = (identified_time - bbox_start_time) * time_scale
    expected_bottom = (identified_freq - bbox_low_freq) * freq_scale
    expected_right = (bbox_end_time - identified_time) * time_scale
    expected_top = (bbox_high_freq - identified_freq) * freq_scale
    expected_size = np.array(
        [expected_left, expected_bottom, expected_right, expected_top]
    )

    assert actual_size.shape == (4,)
    np.testing.assert_allclose(actual_size, expected_size, rtol=1e-5)


def test_peak_energy_bbox_mapper_decode():
    """
    Tests that PeakEnergyBBoxMapper.decode correctly reconstructs a BoundingBox.
    """
    # Given
    time_scale = 100.0
    freq_scale = 0.1

    # Define a known peak position and scaled distances.
    peak_position = (1.5, 15000)
    scaled_size = np.array([50.0, 500.0, 50.0, 500.0])

    mapper = PeakEnergyBBoxMapper(
        preprocessor=build_preprocessor(),
        audio_loader=build_audio_loader(),
        time_scale=time_scale,
        frequency_scale=freq_scale,
    )

    # When
    reconstructed_bbox = mapper.decode(peak_position, scaled_size)

    # Then
    # Calculate the expected coordinates based on the decode logic.
    expected_start_time = peak_position[0] - scaled_size[0] / time_scale
    expected_low_freq = peak_position[1] - scaled_size[1] / freq_scale
    expected_end_time = peak_position[0] + scaled_size[2] / time_scale
    expected_high_freq = peak_position[1] + scaled_size[3] / freq_scale

    expected_coordinates = [
        expected_start_time,
        expected_low_freq,
        expected_end_time,
        expected_high_freq,
    ]

    assert isinstance(reconstructed_bbox, data.BoundingBox)
    np.testing.assert_allclose(
        reconstructed_bbox.coordinates, expected_coordinates
    )


def test_peak_energy_bbox_mapper_encode_decode_roundtrip(generate_whistle):
    """
    Tests that encoding and then decoding a SoundEvent with the
    PeakEnergyBBoxMapper results in the original BoundingBox.
    """
    # Given
    samplerate = 256_000

    # Define the original geometry and the peak location within it.
    original_bbox = data.BoundingBox(coordinates=[1.0, 10000, 2.0, 30000])
    peak_time, peak_freq = 1.6, 25000

    # Create the recording and sound event.
    recording_path = generate_whistle(
        time=peak_time,
        frequency=peak_freq,
        duration=3.0,
        samplerate=samplerate,
    )
    recording = data.Recording.from_file(path=recording_path)
    sound_event = data.SoundEvent(geometry=original_bbox, recording=recording)

    # Instantiate the mapper.
    preprocessor = build_preprocessor(
        PreprocessingConfig(spectrogram_transforms=[])
    )
    audio_loader = build_audio_loader()
    mapper = PeakEnergyBBoxMapper(
        preprocessor=preprocessor,
        audio_loader=audio_loader,
    )

    # When
    # Encode the sound event, then immediately decode the result.
    position, size = mapper.encode(sound_event)
    reconstructed_bbox = mapper.decode(position, size)

    # Then
    # Verify the reconstructed bounding box is identical to the original.
    np.testing.assert_allclose(
        reconstructed_bbox.coordinates,
        original_bbox.coordinates,
        rtol=1e-5,
    )


def test_build_roi_mapper_for_anchor_bbox():
    # Given
    config = AnchorBBoxMapperConfig(
        anchor="center",
        time_scale=123.0,
        frequency_scale=456.0,
    )

    # When
    mapper = build_roi_mapper(config)

    # Then
    assert isinstance(mapper, AnchorBBoxMapper)
    assert mapper.anchor == "center"
    assert mapper.time_scale == 123.0
    assert mapper.frequency_scale == 456.0


def test_build_roi_mapper_for_peak_energy_bbox():
    # Given
    preproc_config = PreprocessingConfig(
        spectrogram_transforms=[
            ScaleAmplitudeConfig(scale="db"),
            SpectralMeanSubtractionConfig(),
        ]
    )
    config = PeakEnergyBBoxMapperConfig(
        loading_buffer=0.99,
        time_scale=789.0,
        frequency_scale=123.0,
        preprocessing=preproc_config,
    )

    # When
    mapper = build_roi_mapper(config)

    # Then
    assert isinstance(mapper, PeakEnergyBBoxMapper)
    assert mapper.loading_buffer == 0.99
    assert mapper.time_scale == 789.0
    assert mapper.frequency_scale == 123.0


def test_build_roi_mapper_raises_error_for_unknown_name():
    """
    Tests that the factory raises a NotImplementedError when given a
    config with an unrecognized mapper name.
    """

    # Given
    class DummyConfig:
        name = "non_existent_mapper"

    # Then
    with pytest.raises(NotImplementedError):
        build_roi_mapper(DummyConfig())  # type: ignore


def test_build_roi_mapping_applies_class_override():
    config = ROIMappingConfig(
        default=AnchorBBoxMapperConfig(anchor="bottom-left"),
        overrides={
            "myomyo": AnchorBBoxMapperConfig(anchor="top-left"),
        },
    )

    mapper = build_roi_mapping(config=config)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])
    sound_event = data.SoundEvent(
        recording=data.Recording(
            path=Path("x.wav"),
            samplerate=256_000,
            channels=1,
            duration=1.0,
        ),
        geometry=geometry,
    )

    default_position, _ = mapper.encode(sound_event, class_name="pippip")
    override_position, _ = mapper.encode(sound_event, class_name="myomyo")

    assert default_position == pytest.approx((0.1, 12_000))
    assert override_position == pytest.approx((0.1, 18_000))


def test_build_roi_mapping_rejects_dimension_mismatch():
    config = ROIMappingConfig(
        default=AnchorBBoxMapperConfig(),
        overrides={
            "myomyo": PeakEnergyBBoxMapperConfig(),
        },
    )

    with pytest.raises(ValueError, match="same dimension order"):
        build_roi_mapping(config=config)
