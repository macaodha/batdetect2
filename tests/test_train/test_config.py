from batdetect2.audio import AudioConfig
from batdetect2.evaluate import EvaluationConfig
from batdetect2.inference import InferenceConfig
from batdetect2.logging import AppLoggingConfig
from batdetect2.models import ModelConfig
from batdetect2.outputs import OutputsConfig
from batdetect2.targets import TargetConfig
from batdetect2.train import TrainingConfig


def test_example_split_configs_are_valid(example_data_dir):
    configs_dir = example_data_dir / "configs"

    assert isinstance(
        AudioConfig.load(configs_dir / "audio.yaml"), AudioConfig
    )
    assert isinstance(
        ModelConfig.load(configs_dir / "model.yaml"), ModelConfig
    )
    assert isinstance(
        TargetConfig.load(example_data_dir / "targets.yaml"),
        TargetConfig,
    )
    assert isinstance(
        TrainingConfig.load(configs_dir / "training.yaml"),
        TrainingConfig,
    )
    assert isinstance(
        EvaluationConfig.load(configs_dir / "evaluation.yaml"),
        EvaluationConfig,
    )
    assert isinstance(
        InferenceConfig.load(configs_dir / "inference.yaml"),
        InferenceConfig,
    )
    assert isinstance(
        OutputsConfig.load(configs_dir / "outputs.yaml"),
        OutputsConfig,
    )
    assert isinstance(
        AppLoggingConfig.load(configs_dir / "logging.yaml"),
        AppLoggingConfig,
    )
