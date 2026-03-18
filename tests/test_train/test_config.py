from batdetect2.config import BatDetect2Config
from batdetect2.core import load_config


def test_example_config_is_valid(example_data_dir):
    conf = load_config(
        example_data_dir / "config.yaml",
        schema=BatDetect2Config,
        extra="forbid",
        strict=True,
    )
    assert isinstance(conf, BatDetect2Config)
