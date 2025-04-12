from typing import List

import numpy as np
import torch
import xarray as xr
from soundevent import data

from batdetect2.modules import DetectorModel
from batdetect2.postprocess.arrays import to_xarray
from batdetect2.preprocess import preprocess_audio_clip


def test_this(clip: data.Clip, class_names: List[str]):
    spec = xr.DataArray(
        data=np.random.rand(100, 100),
        dims=["time", "frequency"],
        coords={
            "time": np.linspace(0, 100, 100, endpoint=False),
            "frequency": np.linspace(0, 100, 100, endpoint=False),
        },
    )

    model = DetectorModel()

    spec = preprocess_audio_clip(
        clip,
        config=model.config.preprocessing,
    )

    tensor = torch.from_numpy(spec.data).unsqueeze(0).unsqueeze(0)

    outputs = model(tensor)

    arrays = to_xarray(
        outputs,
        start_time=clip.start_time,
        end_time=clip.end_time,
        class_names=class_names,
    )

    print(arrays)

    assert False
