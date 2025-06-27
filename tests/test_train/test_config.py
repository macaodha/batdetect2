from batdetect2.configs import load_config
from batdetect2.train import FullTrainingConfig


def test_example_config_is_valid(example_data_dir):
    conf = load_config(
        example_data_dir / "config.yaml",
        schema=FullTrainingConfig,
    )
    assert isinstance(conf, FullTrainingConfig)
