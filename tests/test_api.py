"""Test bat detect module API."""

from pathlib import Path

import os
from glob import glob

import numpy as np
import torch
from torch import nn
import soundfile as sf

from batdetect2 import api

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PKG_DIR, "example_data", "audio")
TEST_DATA = glob(os.path.join(TEST_DATA_DIR, "*.wav"))


def test_load_model_with_default_params():
    """Test loading model with default parameters."""
    model, params = api.load_model()

    assert model is not None
    assert isinstance(model, nn.Module)

    assert params is not None
    assert isinstance(params, dict)

    assert "model_name" in params
    assert "num_filters" in params
    assert "emb_dim" in params
    assert "ip_height" in params

    assert params["model_name"] == "Net2DFast"
    assert params["num_filters"] == 128
    assert params["emb_dim"] == 0
    assert params["ip_height"] == 128
    assert params["resize_factor"] == 0.5
    assert len(params["class_names"]) == 17


def test_list_audio_files():
    """Test listing audio files."""
    audio_files = api.list_audio_files(TEST_DATA_DIR)

    assert len(audio_files) == 3
    assert all(path.endswith((".wav", ".WAV")) for path in audio_files)


def test_load_audio():
    """Test loading audio."""
    audio = api.load_audio(TEST_DATA[0])

    assert audio is not None
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (128000,)


def test_generate_spectrogram():
    """Test generating spectrogram."""
    audio = api.load_audio(TEST_DATA[0])
    spectrogram = api.generate_spectrogram(audio)

    assert spectrogram is not None
    assert isinstance(spectrogram, torch.Tensor)
    assert spectrogram.shape == (1, 1, 128, 512)


def test_get_default_config():
    """Test getting default configuration."""
    config = api.get_config()

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
    assert len(config["class_names"]) == 17
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


def test_api_exposes_default_model():
    """Test that API exposes default model."""
    assert hasattr(api, "model")
    assert isinstance(api.model, nn.Module)
    assert type(api.model).__name__ == "Net2DFast"

    # Check that model has expected attributes
    assert api.model.num_classes == 17
    assert api.model.num_filts == 128
    assert api.model.emb_dim == 0
    assert api.model.ip_height_rs == 128
    assert api.model.resize_factor == 0.5


def test_api_exposes_default_config():
    """Test that API exposes default configuration."""
    assert hasattr(api, "config")
    assert isinstance(api.config, dict)

    assert api.config["target_samp_rate"] == 256000
    assert api.config["fft_win_length"] == 0.002
    assert api.config["fft_overlap"] == 0.75
    assert api.config["resize_factor"] == 0.5
    assert api.config["spec_divide_factor"] == 32
    assert api.config["spec_height"] == 256
    assert api.config["spec_scale"] == "pcen"
    assert api.config["denoise_spec_avg"] is True
    assert api.config["max_scale_spec"] is False
    assert api.config["scale_raw_audio"] is False
    assert len(api.config["class_names"]) == 17
    assert api.config["detection_threshold"] == 0.01
    assert api.config["time_expansion"] == 1
    assert api.config["top_n"] == 3
    assert api.config["return_raw_preds"] is False
    assert api.config["max_duration"] is None
    assert api.config["nms_kernel_size"] == 9
    assert api.config["max_freq"] == 120000
    assert api.config["min_freq"] == 10000
    assert api.config["nms_top_k_per_sec"] == 200
    assert api.config["quiet"] is True
    assert api.config["chunk_size"] == 3
    assert api.config["cnn_features"] is False
    assert api.config["spec_features"] is False
    assert api.config["spec_slices"] is False


def test_process_file_with_default_model():
    """Test processing file with model."""
    predictions = api.process_file(TEST_DATA[0])

    assert predictions is not None
    assert isinstance(predictions, dict)

    assert "pred_dict" in predictions

    # By default will not return other features
    assert "spec_feats" not in predictions
    assert "spec_feat_names" not in predictions
    assert "cnn_feats" not in predictions
    assert "cnn_feat_names" not in predictions
    assert "spec_slices" not in predictions

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


def test_process_spectrogram_with_default_model():
    """Test processing spectrogram with model."""
    audio = api.load_audio(TEST_DATA[0])
    spectrogram = api.generate_spectrogram(audio)
    predictions, features = api.process_spectrogram(spectrogram)

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
    assert isinstance(features, np.ndarray)
    assert len(features) == len(predictions)


def test_process_audio_with_default_model():
    """Test processing audio with model."""
    audio = api.load_audio(TEST_DATA[0])
    predictions, features, spec = api.process_audio(audio)

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
    assert isinstance(features, np.ndarray)
    assert len(features) == len(predictions)

    assert spec is not None
    assert isinstance(spec, torch.Tensor)
    assert spec.shape == (1, 1, 128, 512)


def test_postprocess_model_outputs():
    """Test postprocessing model outputs."""
    # Load model outputs
    audio = api.load_audio(TEST_DATA[1])
    spec = api.generate_spectrogram(audio)
    model_outputs = api.model(spec)

    # Postprocess outputs
    predictions, features = api.postprocess(model_outputs)

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
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(predictions)
    assert features.shape[1] == 32


def test_process_file_with_spec_slices():
    """Test process file returns spec slices."""
    config = api.get_config(spec_slices=True)
    results = api.process_file(TEST_DATA[0], config=config)
    detections = results["pred_dict"]["annotation"]

    assert "spec_slices" in results
    assert isinstance(results["spec_slices"], list)
    assert len(results["spec_slices"]) == len(detections)



def test_process_file_with_empty_predictions_does_not_fail(
    tmp_path: Path,
):
    """Test process file with empty predictions does not fail."""
    # Create empty file
    empty_file = tmp_path / "empty.wav"
    empty_wav = np.zeros((0, 1), dtype=np.float32)
    sf.write(empty_file, empty_wav, 256000)

    # Process file
    results = api.process_file(str(empty_file))

    assert results is not None
    assert len(results["pred_dict"]["annotation"]) == 0
