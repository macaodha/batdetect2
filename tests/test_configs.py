from pathlib import Path

from batdetect2.configs import load_config
from batdetect2.data import DatasetsConfig, load_datasets


def test_can_load_dataset_configs():
    root = Path(__file__).parent.parent
    path = root / "conf.yaml"
    config = load_config(path, schema=DatasetsConfig, field="datasets")
    load_datasets(config)
