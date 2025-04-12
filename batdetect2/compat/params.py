from batdetect2.preprocess import (
    AmplitudeScaleConfig,
    AudioConfig,
    FrequencyConfig,
    LogScaleConfig,
    PcenScaleConfig,
    PreprocessingConfig,
    ResampleConfig,
    Scales,
    SpecSizeConfig,
    SpectrogramConfig,
    STFTConfig,
)
from batdetect2.preprocess.spectrogram import get_spectrogram_resolution
from batdetect2.targets import (
    HeatmapsConfig,
    TagInfo,
    TargetConfig,
)
from batdetect2.targets.labels import LabelConfig
from batdetect2.train.preprocess import (
    TrainPreprocessingConfig,
)


def get_spectrogram_scale(scale: str) -> Scales:
    if scale == "pcen":
        return PcenScaleConfig()
    if scale == "log":
        return LogScaleConfig()
    return AmplitudeScaleConfig()


def get_preprocessing_config(params: dict) -> PreprocessingConfig:
    return PreprocessingConfig(
        audio=AudioConfig(
            resample=ResampleConfig(
                samplerate=params["target_samp_rate"],
                mode="poly",
            ),
            scale=params["scale_raw_audio"],
            center=params["scale_raw_audio"],
            duration=None,
        ),
        spectrogram=SpectrogramConfig(
            stft=STFTConfig(
                window_duration=params["fft_win_length"],
                window_overlap=params["fft_overlap"],
                window_fn="hann",
            ),
            frequencies=FrequencyConfig(
                min_freq=params["min_freq"],
                max_freq=params["max_freq"],
            ),
            scale=get_spectrogram_scale(params["spec_scale"]),
            denoise=params["denoise_spec_avg"],
            size=SpecSizeConfig(
                height=params["spec_height"],
                resize_factor=params["resize_factor"],
            ),
            max_scale=params["max_scale_spec"],
        ),
    )


def get_training_preprocessing_config(
    params: dict,
) -> TrainPreprocessingConfig:
    generic = params["generic_class"][0]
    preprocessing = get_preprocessing_config(params)

    freq_bin_width, time_bin_width = get_spectrogram_resolution(
        preprocessing.spectrogram
    )

    return TrainPreprocessingConfig(
        preprocessing=preprocessing,
        target=TargetConfig(
            classes=[
                TagInfo(key="class", value=class_name, label=class_name)
                for class_name in params["class_names"]
            ],
            generic_class=TagInfo(
                key="class",
                value=generic,
                label=generic,
            ),
            include=[
                TagInfo(key="event", value=event)
                for event in params["events_of_interest"]
            ],
            exclude=[
                TagInfo(key="class", value=value)
                for value in params["classes_to_ignore"]
            ],
        ),
        labels=LabelConfig(
            heatmaps=HeatmapsConfig(
                position="bottom-left",
                time_scale=1 / time_bin_width,
                frequency_scale=1 / freq_bin_width,
                sigma=params["target_sigma"],
            )
        ),
    )


#  'standardize_classs_names_ip',
#  'convert_to_genus',
#  'genus_mapping',
#  'standardize_classs_names',
#  'genus_names',

# ['data_dir',
#  'ann_dir',
#  'train_split',
#  'model_name',
#  'num_filters',
#  'experiment',
#  'model_file_name',
#  'op_im_dir',
#  'op_im_dir_test',
#  'notes',
#  'spec_divide_factor',
#  'detection_overlap',
#  'ignore_start_end',
#  'detection_threshold',
#  'nms_kernel_size',
#  'nms_top_k_per_sec',
#  'aug_prob',
#  'augment_at_train',
#  'augment_at_train_combine',
#  'echo_max_delay',
#  'stretch_squeeze_delta',
#  'mask_max_time_perc',
#  'mask_max_freq_perc',
#  'spec_amp_scaling',
#  'aug_sampling_rates',
#  'train_loss',
#  'det_loss_weight',
#  'size_loss_weight',
#  'class_loss_weight',
#  'individual_loss_weight',
#  'emb_dim',
#  'lr',
#  'batch_size',
#  'num_workers',
#  'num_epochs',
#  'num_eval_epochs',
#  'device',
#  'save_test_image_during_train',
#  'save_test_image_after_train',
#  'train_sets',
#  'test_sets',
#  'class_inv_freq',
#  'ip_height']
