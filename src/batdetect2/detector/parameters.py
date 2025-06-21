import datetime
import os
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, computed_field

from batdetect2.train.legacy.train_utils import (
    get_genus_mapping,
    get_short_class_names,
)
from batdetect2.types import ProcessingConfiguration, SpectrogramParameters

TARGET_SAMPLERATE_HZ = 256000
FFT_WIN_LENGTH_S = 512 / 256000.0
FFT_OVERLAP = 0.75
MAX_FREQ_HZ = 120000
MIN_FREQ_HZ = 10000
RESIZE_FACTOR = 0.5
SPEC_DIVIDE_FACTOR = 32
SPEC_HEIGHT = 256
SCALE_RAW_AUDIO = False
DETECTION_THRESHOLD = 0.01
NMS_KERNEL_SIZE = 9
NMS_TOP_K_PER_SEC = 200
SPEC_SCALE = "pcen"
DENOISE_SPEC_AVG = True
MAX_SCALE_SPEC = False


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "checkpoints",
    "Net2DFast_UK_same.pth.tar",
)


DEFAULT_SPECTROGRAM_PARAMETERS: SpectrogramParameters = {
    "fft_win_length": FFT_WIN_LENGTH_S,
    "fft_overlap": FFT_OVERLAP,
    "spec_height": SPEC_HEIGHT,
    "resize_factor": RESIZE_FACTOR,
    "spec_divide_factor": SPEC_DIVIDE_FACTOR,
    "max_freq": MAX_FREQ_HZ,
    "min_freq": MIN_FREQ_HZ,
    "spec_scale": SPEC_SCALE,
    "denoise_spec_avg": DENOISE_SPEC_AVG,
    "max_scale_spec": MAX_SCALE_SPEC,
}


DEFAULT_PROCESSING_CONFIGURATIONS: ProcessingConfiguration = {
    "detection_threshold": DETECTION_THRESHOLD,
    "spec_slices": False,
    "chunk_size": 3,
    "spec_features": False,
    "cnn_features": False,
    "quiet": True,
    "target_samp_rate": TARGET_SAMPLERATE_HZ,
    "fft_win_length": FFT_WIN_LENGTH_S,
    "fft_overlap": FFT_OVERLAP,
    "resize_factor": RESIZE_FACTOR,
    "spec_divide_factor": SPEC_DIVIDE_FACTOR,
    "spec_height": SPEC_HEIGHT,
    "scale_raw_audio": SCALE_RAW_AUDIO,
    "class_names": [],
    "time_expansion": 1,
    "top_n": 3,
    "return_raw_preds": False,
    "max_duration": None,
    "nms_kernel_size": NMS_KERNEL_SIZE,
    "max_freq": MAX_FREQ_HZ,
    "min_freq": MIN_FREQ_HZ,
    "nms_top_k_per_sec": NMS_TOP_K_PER_SEC,
    "spec_scale": SPEC_SCALE,
    "denoise_spec_avg": DENOISE_SPEC_AVG,
    "max_scale_spec": MAX_SCALE_SPEC,
}


def mk_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


AUG_SAMPLING_RATES = [
    220500,
    256000,
    300000,
    312500,
    384000,
    441000,
    500000,
]
CLASSES_TO_IGNORE = ["", " ", "Unknown", "Not Bat"]
GENERIC_CLASSES = ["Bat"]
EVENTS_OF_INTEREST = ["Echolocation"]


class TrainingParameters(BaseModel):
    # Net2DFast, Net2DSkip, Net2DSimple, Net2DSkipDS, Net2DRN
    model_name: str = "Net2DFast"
    num_filters: int = 128

    experiment: Path
    model_file_name: Path

    op_im_dir: Path
    op_im_dir_test: Path

    notes: str = ""

    target_samp_rate: int = TARGET_SAMPLERATE_HZ
    fft_win_length: float = FFT_WIN_LENGTH_S
    fft_overlap: float = FFT_OVERLAP

    max_freq: int = MAX_FREQ_HZ
    min_freq: int = MIN_FREQ_HZ

    resize_factor: float = RESIZE_FACTOR
    spec_height: int = SPEC_HEIGHT
    spec_train_width: int = 512
    spec_divide_factor: int = SPEC_DIVIDE_FACTOR

    denoise_spec_avg: bool = DENOISE_SPEC_AVG
    scale_raw_audio: bool = SCALE_RAW_AUDIO
    max_scale_spec: bool = MAX_SCALE_SPEC
    spec_scale: str = SPEC_SCALE

    detection_overlap: float = 0.01
    ignore_start_end: float = 0.01
    detection_threshold: float = DETECTION_THRESHOLD
    nms_kernel_size: int = NMS_KERNEL_SIZE
    nms_top_k_per_sec: int = NMS_TOP_K_PER_SEC

    aug_prob: float = 0.20
    augment_at_train: bool = True
    augment_at_train_combine: bool = True
    echo_max_delay: float = 0.005
    stretch_squeeze_delta: float = 0.04
    mask_max_time_perc: float = 0.05
    mask_max_freq_perc: float = 0.10
    spec_amp_scaling: float = 2.0
    aug_sampling_rates: List[int] = AUG_SAMPLING_RATES

    train_loss: str = "focal"
    det_loss_weight: float = 1.0
    size_loss_weight: float = 0.1
    class_loss_weight: float = 2.0
    individual_loss_weight: float = 0.0

    lr: float = 0.001
    batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 200
    num_eval_epochs: int = 5
    device: str = "cuda"
    save_test_image_during_train: bool = False
    save_test_image_after_train: bool = True

    convert_to_genus: bool = False
    class_names: List[str] = Field(default_factory=list)
    classes_to_ignore: List[str] = Field(
        default_factory=lambda: CLASSES_TO_IGNORE
    )
    generic_class: List[str] = Field(default_factory=lambda: GENERIC_CLASSES)
    events_of_interest: List[str] = Field(
        default_factory=lambda: EVENTS_OF_INTEREST
    )
    standardize_classs_names: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def emb_dim(self) -> int:
        if self.individual_loss_weight == 0.0:
            return 0
        return 3

    @computed_field
    @property
    def genus_mapping(self) -> List[int]:
        _, mapping = get_genus_mapping(self.class_names)
        return mapping

    @computed_field
    @property
    def genus_classes(self) -> List[str]:
        names, _ = get_genus_mapping(self.class_names)
        return names

    @computed_field
    @property
    def class_names_short(self) -> List[str]:
        return get_short_class_names(self.class_names)


def get_params(
    make_dirs: bool = False,
    exps_dir: str = "../../experiments/",
    model_name: Optional[str] = None,
    experiment: Union[Path, str, None] = None,
    **kwargs,
) -> TrainingParameters:
    experiments_dir = Path(exps_dir)

    now_str = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if model_name is None:
        model_name = f"{now_str}.pth.tar"

    if experiment is None:
        experiment = experiments_dir / now_str
    experiment = Path(experiment)

    model_file_name = experiment / model_name
    op_ims_dir = experiment / "op_ims"
    op_ims_test_dir = experiment / "op_ims_test"

    params = TrainingParameters(
        model_name=model_name,
        experiment=experiment,
        model_file_name=model_file_name,
        op_im_dir=op_ims_dir,
        op_im_dir_test=op_ims_test_dir,
        **kwargs,
    )

    if make_dirs:
        mk_dir(experiment)
        mk_dir(params.model_file_name.parent)
        if params.save_test_image_during_train:
            mk_dir(params.op_im_dir)
        if params.save_test_image_after_train:
            mk_dir(params.op_im_dir_test)

    return params
