"""Test bat detect module API."""

import os
from glob import glob

import numpy as np
import torch
from torch import nn

from bat_detect.api import (
    generate_spectrogram,
    get_config,
    list_audio_files,
    load_audio,
    load_model,
    process_audio,
    process_file,
    process_spectrogram,
)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PKG_DIR, "example_data", "audio")
TEST_DATA = glob(os.path.join(TEST_DATA_DIR, "*.wav"))


def test_load_model_with_default_params():
    """Test loading model with default parameters."""
    model, params = load_model()

    assert model is not None
    assert isinstance(model, nn.Module)

    assert params is not None
    assert isinstance(params, dict)

    assert "model_name" in params
    assert "num_filters" in params
    assert "emb_dim" in params
    assert "ip_height" in params
    assert "resize_factor" in params
    assert "class_names" in params

    assert params["model_name"] == "Net2DFast"
    assert params["num_filters"] == 128
    assert params["emb_dim"] == 0
    assert params["ip_height"] == 128
    assert params["resize_factor"] == 0.5
    assert len(params["class_names"]) == 17


def test_list_audio_files():
    """Test listing audio files."""
    audio_files = list_audio_files(TEST_DATA_DIR)

    assert len(audio_files) == 3
    assert all(path.endswith((".wav", ".WAV")) for path in audio_files)


def test_load_audio():
    """Test loading audio."""
    samplerate, audio = load_audio(TEST_DATA[0])

    assert audio is not None
    assert samplerate == 256000
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (128000,)


def test_generate_spectrogram():
    """Test generating spectrogram."""
    samplerate, audio = load_audio(TEST_DATA[0])
    spectrogram = generate_spectrogram(audio, samplerate)

    assert spectrogram is not None
    assert isinstance(spectrogram, torch.Tensor)
    assert spectrogram.shape == (1, 1, 128, 512)


def test_get_default_config():
    """Test getting default configuration."""
    config = get_config()

    assert config is not None
    assert isinstance(config, dict)

    assert config["target_samp_rate"] == 256000
    assert config["fft_win_length"] == 0.002
    assert config["fft_overlap"] == 0.75
    assert config["resize_factor"] == 0.5
    assert config["spec_divide_factor"] == 32
    assert config["spec_height"] == 256
    assert config["spec_scale"] == "pcen"
    assert config["denoise_spec_avg"] is True
    assert config["max_scale_spec"] is False
    assert config["scale_raw_audio"] is False
    assert len(config["class_names"]) == 0
    assert config["detection_threshold"] == 0.01
    assert config["time_expansion"] == 1
    assert config["top_n"] == 3
    assert config["return_raw_preds"] is False
    assert config["max_duration"] is None
    assert config["nms_kernel_size"] == 9
    assert config["max_freq"] == 120000
    assert config["min_freq"] == 10000
    assert config["nms_top_k_per_sec"] == 200
    assert config["quiet"] is True
    assert config["chunk_size"] == 3
    assert config["cnn_features"] is False
    assert config["spec_features"] is False
    assert config["spec_slices"] is False


def test_process_file_with_model():
    """Test processing file with model."""
    model, params = load_model()
    config = get_config(**params)
    predictions = process_file(TEST_DATA[0], model, config=config)

    assert predictions is not None
    assert isinstance(predictions, dict)

    assert "pred_dict" in predictions
    assert "spec_feats" in predictions
    assert "spec_feat_names" in predictions
    assert "cnn_feats" in predictions
    assert "cnn_feat_names" in predictions
    assert "spec_slices" in predictions

    # By default will not return spectrogram features
    assert predictions["spec_feats"] is None
    assert predictions["spec_feat_names"] is None
    assert predictions["cnn_feats"] is None
    assert predictions["cnn_feat_names"] is None
    assert predictions["spec_slices"] is None

    # Check that predictions are returned
    assert isinstance(predictions["pred_dict"], dict)
    pred_dict = predictions["pred_dict"]
    assert pred_dict["id"] == os.path.basename(TEST_DATA[0])
    assert pred_dict["annotated"] is False
    assert pred_dict["issues"] is False
    assert pred_dict["notes"] == "Automatically generated."
    assert pred_dict["time_exp"] == 1
    assert pred_dict["duration"] == 0.5
    assert pred_dict["class_name"] is not None
    assert len(pred_dict["annotation"]) > 0


def test_process_spectrogram_with_model():
    """Test processing spectrogram with model."""
    model, params = load_model()
    config = get_config(**params)
    samplerate, audio = load_audio(TEST_DATA[0])
    spectrogram = generate_spectrogram(audio, samplerate)
    predictions, features = process_spectrogram(
        spectrogram,
        samplerate,
        model,
        config=config,
    )

    assert predictions is not None
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    sample_pred = predictions[0]
    assert isinstance(sample_pred, dict)
    assert "class" in sample_pred
    assert "class_prob" in sample_pred
    assert "det_prob" in sample_pred
    assert "start_time" in sample_pred
    assert "end_time" in sample_pred
    assert "low_freq" in sample_pred
    assert "high_freq" in sample_pred

    assert features is not None
    assert isinstance(features, list)
    # By default will not return cnn features
    assert len(features) == 0


def test_process_audio_with_model():
    """Test processing audio with model."""
    model, params = load_model()
    config = get_config(**params)
    samplerate, audio = load_audio(TEST_DATA[0])
    predictions, features, spec = process_audio(
        audio,
        samplerate,
        model,
        config=config,
    )

    assert predictions is not None
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    sample_pred = predictions[0]
    assert isinstance(sample_pred, dict)
    assert "class" in sample_pred
    assert "class_prob" in sample_pred
    assert "det_prob" in sample_pred
    assert "start_time" in sample_pred
    assert "end_time" in sample_pred
    assert "low_freq" in sample_pred
    assert "high_freq" in sample_pred

    assert features is not None
    assert isinstance(features, list)
    # By default will not return cnn features
    assert len(features) == 0

    assert spec is not None
    assert isinstance(spec, torch.Tensor)
    assert spec.shape == (1, 1, 128, 512)
