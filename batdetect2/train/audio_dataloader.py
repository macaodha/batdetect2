"""Functions and dataloaders for training and testing the model."""
import copy
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio

import batdetect2.utils.audio_utils as au
from batdetect2.types import (
    Annotation,
    AnnotationGroup,
    AudioLoaderAnnotationGroup,
    FileAnnotations,
    HeatmapParameters,
)


def generate_gt_heatmaps(
    spec_op_shape: Tuple[int, int],
    sampling_rate: int,
    ann: AnnotationGroup,
    params: HeatmapParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AnnotationGroup]:
    """Generate ground truth heatmaps from annotations.

    Parameters
    ----------
    spec_op_shape : Tuple[int, int]
        Shape of the input spectrogram.
    sampling_rate : int
        Sampling rate of the input audio in Hz.
    ann : AnnotationGroup
        Dictionary containing the annotation information.
    params : HeatmapParameters
        Parameters controlling the generation of the heatmaps.

    Returns
    -------
    y_2d_det : np.ndarray
        2D heatmap of the presence of an event.
    y_2d_size : np.ndarray
        2D heatmap of the size of the bounding box associated to event.
    y_2d_classes : np.ndarray
        3D array containing the ground-truth class probabilities for each
        pixel.
    ann_aug : AnnotationGroup
        A dictionary containing the annotation information of the
        annotations that are within the input spectrogram, augmented with
        the x and y indices of their pixel location in the input spectrogram.
    """
    # spec may be resized on input into the network
    num_classes = len(params["class_names"])
    op_height = spec_op_shape[0]
    op_width = spec_op_shape[1]
    freq_per_bin = (params["max_freq"] - params["min_freq"]) / op_height

    # start and end times
    x_pos_start = au.time_to_x_coords(
        ann["start_times"],
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
    )
    x_pos_start = (params["resize_factor"] * x_pos_start).astype(np.int32)
    x_pos_end = au.time_to_x_coords(
        ann["end_times"],
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
    )
    x_pos_end = (params["resize_factor"] * x_pos_end).astype(np.int32)

    # location on y axis i.e. frequency
    y_pos_low = (ann["low_freqs"] - params["min_freq"]) / freq_per_bin
    y_pos_low = (op_height - y_pos_low).astype(np.int32)
    y_pos_high = (ann["high_freqs"] - params["min_freq"]) / freq_per_bin
    y_pos_high = (op_height - y_pos_high).astype(np.int32)
    bb_widths = x_pos_end - x_pos_start
    bb_heights = y_pos_low - y_pos_high

    # Only include annotations that are within the input spectrogram
    valid_inds = np.where(
        (x_pos_start >= 0)
        & (x_pos_start < op_width)
        & (y_pos_low >= 0)
        & (y_pos_low < (op_height - 1))
    )[0]

    ann_aug: AnnotationGroup = {
        "start_times": ann["start_times"][valid_inds],
        "end_times": ann["end_times"][valid_inds],
        "high_freqs": ann["high_freqs"][valid_inds],
        "low_freqs": ann["low_freqs"][valid_inds],
        "class_ids": ann["class_ids"][valid_inds],
        "individual_ids": ann["individual_ids"][valid_inds],
    }
    ann_aug["x_inds"] = x_pos_start[valid_inds]
    ann_aug["y_inds"] = y_pos_low[valid_inds]
    # keys = [
    #     "start_times",
    #     "end_times",
    #     "high_freqs",
    #     "low_freqs",
    #     "class_ids",
    #     "individual_ids",
    # ]
    # for kk in keys:
    #     ann_aug[kk] = ann[kk][valid_inds]

    # if the number of calls is only 1, then it is unique
    # TODO would be better if we found these unique calls at the merging stage
    if len(ann_aug["individual_ids"]) == 1:
        ann_aug["individual_ids"][0] = 0

    y_2d_det = np.zeros((1, op_height, op_width), dtype=np.float32)
    y_2d_size = np.zeros((2, op_height, op_width), dtype=np.float32)
    # num classes and "background" class
    y_2d_classes: np.ndarray = np.zeros(
        (num_classes + 1, op_height, op_width), dtype=np.float32
    )

    # create 2D ground truth heatmaps
    for ii in valid_inds:
        draw_gaussian(
            y_2d_det[0, :],
            (x_pos_start[ii], y_pos_low[ii]),
            params["target_sigma"],
        )
        # draw_gaussian(
        #     y_2d_det[0, :],
        #     (x_pos_start[ii], y_pos_low[ii]),
        #     params["target_sigma"],
        #     params["target_sigma"] * 2,
        # )
        y_2d_size[0, y_pos_low[ii], x_pos_start[ii]] = bb_widths[ii]
        y_2d_size[1, y_pos_low[ii], x_pos_start[ii]] = bb_heights[ii]

        cls_id = ann["class_ids"][ii]
        if cls_id > -1:
            draw_gaussian(
                y_2d_classes[cls_id, :],
                (x_pos_start[ii], y_pos_low[ii]),
                params["target_sigma"],
            )
            # draw_gaussian(
            #     y_2d_classes[cls_id, :],
            #     (x_pos_start[ii], y_pos_low[ii]),
            #     params["target_sigma"],
            #     params["target_sigma"] * 2,
            # )

    # be careful as this will have a 1.0 places where we have event but
    # dont know gt class this will be masked in training anyway
    y_2d_classes[num_classes, :] = 1.0 - y_2d_classes.sum(0)
    y_2d_classes = y_2d_classes / y_2d_classes.sum(0)[np.newaxis, ...]
    y_2d_classes[np.isnan(y_2d_classes)] = 0.0

    return y_2d_det, y_2d_size, y_2d_classes, ann_aug


def draw_gaussian(
    heatmap: np.ndarray,
    center: Tuple[int, int],
    sigmax: float,
    sigmay: Optional[float] = None,
) -> bool:
    """Draw a 2D gaussian into the heatmap.

    If the gaussian center is outside the heatmap, then the gaussian is not
    drawn.

    Parameters
    ----------
    heatmap : np.ndarray
        The heatmap to draw into. Should be of shape (height, width).
    center : Tuple[int, int]
        The center of the gaussian in (x, y) format.
    sigmax : float
        The standard deviation of the gaussian in the x direction.
    sigmay : Optional[float], optional
        The standard deviation of the gaussian in the y direction. If None,
        then sigmay = sigmax, by default None.

    Returns
    -------
    bool
        True if the gaussian was drawn, False if it was not (because
        the center was outside the heatmap).


    """
    # center is (x, y)
    # this edits the heatmap inplace

    if sigmay is None:
        sigmay = sigmax
    tmp_size = np.maximum(sigmax, sigmay) * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return False

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = np.exp(
        -((x - x0) ** 2) / (2 * sigmax**2)
        - ((y - y0) ** 2) / (2 * sigmay**2)
    )
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return True


def pad_aray(ip_array: np.ndarray, pad_size: int) -> np.ndarray:
    """Pad array with -1s."""
    return np.hstack((ip_array, np.ones(pad_size, dtype=np.int32) * -1))


def warp_spec_aug(
    spec: torch.Tensor,
    ann: AnnotationGroup,
    params: dict,
) -> torch.Tensor:
    """Warp spectrogram by randomly stretching and squeezing.

    Parameters
    ----------
    spec: torch.Tensor
        Spectrogram to warp.
    ann: AnnotationGroup
        Annotation group for the spectrogram. Must be provided to sync
        the start and stop times with the spectrogram after warping.
    params: dict
        Parameters for the augmentation.

    Returns
    -------
    torch.Tensor
        Warped spectrogram.

    Notes
    -----
    This function modifies the annotation group in place.
    """
    # This is messy
    # Augment spectrogram by randomly stretch and squeezing
    # NOTE this also changes the start and stop time in place

    delta = params["stretch_squeeze_delta"]
    op_size = (spec.shape[1], spec.shape[2])
    resize_fract_r = np.random.rand() * delta * 2 - delta + 1.0
    resize_amt = int(spec.shape[2] * resize_fract_r)

    if resize_amt >= spec.shape[2]:
        spec_r = torch.cat(
            (
                spec,
                torch.zeros(
                    (1, spec.shape[1], resize_amt - spec.shape[2]),
                    dtype=spec.dtype,
                ),
            ),
            2,
        )
    else:
        spec_r = spec[:, :, :resize_amt]

    # Resize the spectrogram
    spec = F.interpolate(
        spec_r.unsqueeze(0),
        size=op_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Update the start and stop times
    ann["start_times"] *= 1.0 / resize_fract_r
    ann["end_times"] *= 1.0 / resize_fract_r

    return spec


def mask_time_aug(spec: torch.Tensor, params: dict) -> torch.Tensor:
    """Mask out random blocks of time.

    Will randomly mask out a block of time in the spectrogram. The block
    will be between 0.0 and `mask_max_time_perc` of the total time.
    A random number of blocks will be masked out between 1 and 3.

    Parameters
    ----------
    spec: torch.Tensor
        Spectrogram to mask.
    params: dict
        Parameters for the augmentation.

    Returns
    -------
    torch.Tensor
        Spectrogram with masked out time blocks.

    Notes
    -----
    This function is based on the implementation in::

        SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition
    """
    fm = torchaudio.transforms.TimeMasking(
        int(spec.shape[1] * params["mask_max_time_perc"])
    )
    for _ in range(np.random.randint(1, 4)):
        spec = fm(spec)
    return spec


def mask_freq_aug(spec: torch.Tensor, params: dict) -> torch.Tensor:
    """Mask out random blocks of frequency.

    Will randomly mask out a block of frequency in the spectrogram. The block
    will be between 0.0 and `mask_max_freq_perc` of the total frequency.
    A random number of blocks will be masked out between 1 and 3.

    Parameters
    ----------
    spec: torch.Tensor
        Spectrogram to mask.
    params: dict
        Parameters for the augmentation.

    Returns
    -------
    torch.Tensor
        Spectrogram with masked out frequency blocks.

    Notes
    -----
    This function is based on the implementation in::

        SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition
    """
    fm = torchaudio.transforms.FrequencyMasking(
        int(spec.shape[1] * params["mask_max_freq_perc"])
    )
    for _ in range(np.random.randint(1, 4)):
        spec = fm(spec)
    return spec


def scale_vol_aug(spec: torch.Tensor, params: dict) -> torch.Tensor:
    """Scale the volume of the spectrogram.

    Parameters
    ----------
    spec: torch.Tensor
        Spectrogram to scale.
    params: dict
        Parameters for the augmentation.

    Returns
    -------
    torch.Tensor
    """
    return spec * np.random.random() * params["spec_amp_scaling"]


def echo_aug(audio: np.ndarray, sampling_rate: int, params: dict) -> np.ndarray:
    """Add echo to audio.

    Parameters
    ----------
    audio: np.ndarray
        Audio to add echo to.
    sampling_rate: int
        Sampling rate of the audio.
    params: dict
        Parameters for the augmentation.

    Returns
    -------
    np.ndarray
        Audio with echo added.
    """
    sample_offset = (
        int(params["echo_max_delay"] * np.random.random() * sampling_rate) + 1
    )
    audio[:-sample_offset] += np.random.random() * audio[sample_offset:]
    return audio


def resample_aug(
    audio: np.ndarray,
    sampling_rate: int,
    params: dict,
) -> Tuple[np.ndarray, int, float]:
    """Resample audio augmentation.

    Will resample the audio to a random sampling rate from the list of
    sampling rates in `aug_sampling_rates`.

    Parameters
    ----------
    audio: np.ndarray
        Audio to resample.
    sampling_rate: int
        Original sampling rate of the audio.
    params: dict
        Parameters for the augmentation. Includes the list of sampling rates
        to choose from for resampling in `aug_sampling_rates`.

    Returns
    -------
    audio : np.ndarray
        Resampled audio.
    sampling_rate : int
        New sampling rate.
    duration : float
        Duration of the audio in seconds.
    """
    sampling_rate_old = sampling_rate
    sampling_rate = np.random.choice(params["aug_sampling_rates"])
    audio = librosa.resample(
        audio,
        orig_sr=sampling_rate_old,
        target_sr=sampling_rate,
        res_type="polyphase",
    )

    audio = au.pad_audio(
        audio,
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
        params["resize_factor"],
        params["spec_divide_factor"],
        params["spec_train_width"],
    )
    duration = audio.shape[0] / float(sampling_rate)
    return audio, sampling_rate, duration


def resample_audio(
    num_samples: int,
    sampling_rate: int,
    audio2: np.ndarray,
    sampling_rate2: int,
) -> Tuple[np.ndarray, int]:
    """Resample audio.

    Parameters
    ----------
    num_samples: int
        Expected number of samples for the output audio.
    sampling_rate: int
        Original sampling rate of the audio.
    audio2: np.ndarray
        Audio to resample.
    sampling_rate2: int
        Target sampling rate of the audio.

    Returns
    -------
    audio2 : np.ndarray
        Resampled audio.
    sampling_rate2 : int
        New sampling rate.
    """
    # resample to target sampling rate
    if sampling_rate != sampling_rate2:
        audio2 = librosa.resample(
            audio2,
            orig_sr=sampling_rate2,
            target_sr=sampling_rate,
            res_type="polyphase",
        )
        sampling_rate2 = sampling_rate

    # pad or trim to the correct length
    if audio2.shape[0] < num_samples:
        audio2 = np.hstack(
            (
                audio2,
                np.zeros((num_samples - audio2.shape[0]), dtype=audio2.dtype),
            )
        )
    elif audio2.shape[0] > num_samples:
        audio2 = audio2[:num_samples]

    return audio2, sampling_rate2


def combine_audio_aug(
    audio: np.ndarray,
    sampling_rate: int,
    ann: AnnotationGroup,
    audio2: np.ndarray,
    sampling_rate2: int,
    ann2: AnnotationGroup,
) -> Tuple[np.ndarray, AnnotationGroup]:
    """Combine two audio files.

    Will combine two audio files by resampling them to the same sampling rate
    and then combining them with a random weight. The annotations will be
    combined by taking the union of the two sets of annotations.

    Parameters
    ----------
    audio: np.ndarray
        First Audio to combine.
    sampling_rate: int
        Sampling rate of the first audio.
    ann: AnnotationGroup
        Annotations for the first audio.
    audio2: np.ndarray
        Second Audio to combine.
    sampling_rate2: int
        Sampling rate of the second audio.
    ann2: AnnotationGroup
        Annotations for the second audio.

    Returns
    -------
    audio : np.ndarray
        Combined audio.
    ann : AnnotationGroup
        Combined annotations.
    """
    # resample so they are the same
    audio2, sampling_rate2 = resample_audio(
        audio.shape[0],
        sampling_rate,
        audio2,
        sampling_rate2,
    )

    # # set mean and std to be the same
    # audio2 = (audio2 - audio2.mean())
    # audio2 = (audio2/audio2.std())*audio.std()
    # audio2 = audio2 + audio.mean()

    if (
        ann.get("annotated", False)
        and (ann2.get("annotated", False))
        and (sampling_rate2 == sampling_rate)
        and (audio.shape[0] == audio2.shape[0])
    ):
        comb_weight = 0.3 + np.random.random() * 0.4
        audio = comb_weight * audio + (1 - comb_weight) * audio2
        inds = np.argsort(np.hstack((ann["start_times"], ann2["start_times"])))
        for kk in ann.keys():
            # when combining calls from different files, assume they come
            # from different individuals
            if kk == "individual_ids":
                if (ann[kk] > -1).sum() > 0:
                    ann2[kk][ann2[kk] > -1] += np.max(ann[kk][ann[kk] > -1]) + 1

            if (kk != "class_id_file") and (kk != "annotated"):
                ann[kk] = np.hstack((ann[kk], ann2[kk]))[inds]

    return audio, ann


def _prepare_annotation(
    annotation: Annotation, class_names: List[str]
) -> Annotation:
    try:
        class_id = class_names.index(annotation["class"])
    except ValueError:
        class_id = -1

    ann: Annotation = {
        **annotation,
        "class_id": class_id,
    }

    if "individual" in ann:
        ann["individual"] = int(ann["individual"])  # type: ignore

    return ann


def _prepare_file_annotation(
    annotation: FileAnnotations,
    class_names: List[str],
    classes_to_ignore: List[str],
) -> AudioLoaderAnnotationGroup:
    annotations = [
        _prepare_annotation(ann, class_names)
        for ann in annotation["annotation"]
        if ann["class"] not in classes_to_ignore
    ]

    try:
        class_id_file = class_names.index(annotation["class_name"])
    except ValueError:
        class_id_file = -1

    ret: AudioLoaderAnnotationGroup = {
        "id": annotation["id"],
        "annotated": annotation["annotated"],
        "duration": annotation["duration"],
        "issues": annotation["issues"],
        "time_exp": annotation["time_exp"],
        "class_name": annotation["class_name"],
        "notes": annotation["notes"],
        "annotation": annotations,
        "start_times": np.array([ann["start_time"] for ann in annotations]),
        "end_times": np.array([ann["end_time"] for ann in annotations]),
        "high_freqs": np.array([ann["high_freq"] for ann in annotations]),
        "low_freqs": np.array([ann["low_freq"] for ann in annotations]),
        "class_ids": np.array([ann.get("class_id", -1) for ann in annotations]),
        "individual_ids": np.array([ann["individual"] for ann in annotations]),
        "class_id_file": class_id_file,
    }

    return ret


class AudioLoader(torch.utils.data.Dataset):
    """Main AudioLoader for training and testing."""

    def __init__(
        self,
        data_anns_ip: List[FileAnnotations],
        params,
        dataset_name: Optional[str] = None,
        is_train: bool = False,
    ):
        self.is_train: bool = is_train
        self.params: dict = params
        self.return_spec_for_viz: bool = False

        self.data_anns: List[AudioLoaderAnnotationGroup] = [
            _prepare_file_annotation(
                ann,
                params["class_names"],
                params["classes_to_ignore"],
            )
            for ann in data_anns_ip
        ]

        # for ii in range(len(data_anns_ip)):
        #     dd = copy.deepcopy(data_anns_ip[ii])
        #
        #     # filter out unused annotation here
        #     filtered_annotations = []
        #     for ii, aa in enumerate(dd["annotation"]):
        #         if "individual" in aa.keys():
        #             aa["individual"] = int(aa["individual"])
        #
        #             # if only one call labeled it has to be from the same
        #             # individual
        #             if len(dd["annotation"]) == 1:
        #                 aa["individual"] = 0
        #
        #         # convert class name into class label
        #         if aa["class"] in self.params["class_names"]:
        #             aa["class_id"] = self.params["class_names"].index(
        #                 aa["class"]
        #             )
        #         else:
        #             aa["class_id"] = -1
        #
        #         if aa["class"] not in self.params["classes_to_ignore"]:
        #             filtered_annotations.append(aa)
        #
        #     dd["annotation"] = filtered_annotations
        #     dd["start_times"] = np.array(
        #         [aa["start_time"] for aa in dd["annotation"]]
        #     )
        #     dd["end_times"] = np.array(
        #         [aa["end_time"] for aa in dd["annotation"]]
        #     )
        #     dd["high_freqs"] = np.array(
        #         [float(aa["high_freq"]) for aa in dd["annotation"]]
        #     )
        #     dd["low_freqs"] = np.array(
        #         [float(aa["low_freq"]) for aa in dd["annotation"]]
        #     )
        #     dd["class_ids"] = np.array(
        #         [aa["class_id"] for aa in dd["annotation"]]
        #     ).astype(np.int32)
        #     dd["individual_ids"] = np.array(
        #         [aa["individual"] for aa in dd["annotation"]]
        #     ).astype(np.int32)
        #
        #     # file level class name
        #     dd["class_id_file"] = -1
        #     if "class_name" in dd.keys():
        #         if dd["class_name"] in self.params["class_names"]:
        #             dd["class_id_file"] = self.params["class_names"].index(
        #                 dd["class_name"]
        #             )
        #
        #     self.data_anns.append(dd)

        ann_cnt = [len(aa["annotation"]) for aa in self.data_anns]
        self.max_num_anns = 2 * np.max(
            ann_cnt
        )  # x2 because we may be combining files during training

        print("\n")
        if dataset_name is not None:
            print("Dataset     : " + dataset_name)
        if self.is_train:
            print("Split type  : train")
        else:
            print("Split type  : test")
        print("Num files   : " + str(len(self.data_anns)))
        print("Num calls   : " + str(np.sum(ann_cnt)))

    def get_file_and_anns(
        self,
        index: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, AudioLoaderAnnotationGroup]:
        """Get an audio file and its annotations.

        Parameters
        ----------
        index : int, optional
            Index of the file to be loaded. If None, a random file is chosen.

        Returns
        -------
        audio_raw : np.ndarray
            Loaded audio file.
        sampling_rate : int
            Sampling rate of the audio file.
        duration : float
            Duration of the audio file in seconds.
        ann : AnnotationGroup
            AnnotationGroup object containing the annotations for the audio file.
        """
        # if no file specified, choose random one
        if index is None:
            index = np.random.randint(0, len(self.data_anns))

        audio_file = self.data_anns[index]["file_path"]
        sampling_rate, audio_raw = au.load_audio(
            audio_file,
            self.data_anns[index]["time_exp"],
            self.params["target_samp_rate"],
            self.params["scale_raw_audio"],
        )

        # copy annotation
        ann = copy.deepcopy(self.data_anns[index])
        # ann["annotated"] = self.data_anns[index]["annotated"]
        # ann["class_id_file"] = self.data_anns[index]["class_id_file"]
        # keys = [
        #     "start_times",
        #     "end_times",
        #     "high_freqs",
        #     "low_freqs",
        #     "class_ids",
        #     "individual_ids",
        # ]
        # for kk in keys:
        #     ann[kk] = self.data_anns[index][kk].copy()

        # if train then grab a random crop
        if self.is_train:
            nfft = int(self.params["fft_win_length"] * sampling_rate)
            noverlap = int(self.params["fft_overlap"] * nfft)
            length_samples = (
                self.params["spec_train_width"] * (nfft - noverlap) + noverlap
            )

            if audio_raw.shape[0] - length_samples > 0:
                sample_crop = np.random.randint(
                    audio_raw.shape[0] - length_samples
                )
            else:
                sample_crop = 0
            audio_raw = audio_raw[sample_crop : sample_crop + length_samples]
            ann["start_times"] = ann["start_times"] - sample_crop / float(
                sampling_rate
            )
            ann["end_times"] = ann["end_times"] - sample_crop / float(
                sampling_rate
            )

        # pad audio
        if self.is_train:
            op_spec_target_size = self.params["spec_train_width"]
        else:
            op_spec_target_size = None
        audio_raw = au.pad_audio(
            audio_raw,
            sampling_rate,
            self.params["fft_win_length"],
            self.params["fft_overlap"],
            self.params["resize_factor"],
            self.params["spec_divide_factor"],
            op_spec_target_size,
        )
        duration = audio_raw.shape[0] / float(sampling_rate)

        # sort based on time
        inds = np.argsort(ann["start_times"])
        for kk in ann.keys():
            if (kk != "class_id_file") and (kk != "annotated"):
                ann[kk] = ann[kk][inds]

        return audio_raw, sampling_rate, duration, ann

    def __getitem__(self, index):
        """Get an item from the dataset."""
        # load audio file
        audio, sampling_rate, duration, ann = self.get_file_and_anns(index)

        # augment on raw audio
        if self.is_train and self.params["augment_at_train"]:
            # augment - combine with random audio file
            if (
                self.params["augment_at_train_combine"]
                and np.random.random() < self.params["aug_prob"]
            ):
                (
                    audio2,
                    sampling_rate2,
                    duration2,
                    ann2,
                ) = self.get_file_and_anns()
                audio, ann = combine_audio_aug(
                    audio, sampling_rate, ann, audio2, sampling_rate2, ann2
                )

            # simulate echo by adding delayed copy of the file
            if np.random.random() < self.params["aug_prob"]:
                audio = echo_aug(audio, sampling_rate, self.params)

            # resample the audio
            # if np.random.random() < self.params["aug_prob"]:
            #     audio, sampling_rate, duration = resample_aug(
            #         audio, sampling_rate, self.params
            #     )

        # create spectrogram
        spec, spec_for_viz = au.generate_spectrogram(
            audio,
            sampling_rate,
            self.params,
            self.return_spec_for_viz,
        )
        rsf = self.params["resize_factor"]
        spec_op_shape = (
            int(self.params["spec_height"] * rsf),
            int(spec.shape[1] * rsf),
        )

        # resize the spec
        spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(
            spec,
            size=spec_op_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # augment spectrogram
        if self.is_train and self.params["augment_at_train"]:
            if np.random.random() < self.params["aug_prob"]:
                spec = scale_vol_aug(spec, self.params)

            if np.random.random() < self.params["aug_prob"]:
                spec = warp_spec_aug(
                    spec,
                    ann,
                    self.params,
                )

            if np.random.random() < self.params["aug_prob"]:
                spec = mask_time_aug(spec, self.params)

            if np.random.random() < self.params["aug_prob"]:
                spec = mask_freq_aug(spec, self.params)

        outputs = {}
        outputs["spec"] = spec
        if self.return_spec_for_viz:
            outputs["spec_for_viz"] = torch.from_numpy(spec_for_viz).unsqueeze(
                0
            )

        # create ground truth heatmaps
        (
            outputs["y_2d_det"],
            outputs["y_2d_size"],
            outputs["y_2d_classes"],
            ann_aug,
        ) = generate_gt_heatmaps(
            spec_op_shape,
            sampling_rate,
            ann,
            self.params,
        )

        # hack to get around requirement that all vectors are the same length
        # in the output batch
        pad_size = self.max_num_anns - len(ann_aug["individual_ids"])
        outputs["is_valid"] = pad_aray(
            np.ones(len(ann_aug["individual_ids"])), pad_size
        )
        keys = [
            "class_ids",
            "individual_ids",
            "x_inds",
            "y_inds",
            "start_times",
            "end_times",
            "low_freqs",
            "high_freqs",
        ]
        for kk in keys:
            outputs[kk] = pad_aray(ann_aug[kk], pad_size)

        # convert to pytorch
        for kk in outputs.keys():
            if type(outputs[kk]) != torch.Tensor:
                outputs[kk] = torch.from_numpy(outputs[kk])

        # scalars
        outputs["class_id_file"] = ann["class_id_file"]
        outputs["annotated"] = ann["annotated"]
        outputs["duration"] = duration
        outputs["sampling_rate"] = sampling_rate
        outputs["file_id"] = index

        return outputs

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data_anns)
