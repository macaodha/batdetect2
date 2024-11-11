"""Test suite for model functions."""

import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from batdetect2 import api
from batdetect2.detector import parameters


def test_can_import_model_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        api.load_model()


@settings(deadline=None, max_examples=5)
@given(duration=st.floats(min_value=0.1, max_value=2))
def test_can_import_model_without_pickle(duration: float):
    # NOTE: remove this test once no other issues are found This is a temporary
    # test to check that change in model loading did not impact model behaviour
    # in any way.

    samplerate = parameters.TARGET_SAMPLERATE_HZ
    audio = np.random.rand(int(duration * samplerate))

    model_without_pickle, model_params_without_pickle = api.load_model(
        weights_only=True
    )
    model_with_pickle, model_params_with_pickle = api.load_model(
        weights_only=False
    )

    assert model_params_without_pickle == model_params_with_pickle

    predictions_without_pickle, _, _ = api.process_audio(
        audio,
        model=model_without_pickle,
    )
    predictions_with_pickle, _, _ = api.process_audio(
        audio,
        model=model_with_pickle,
    )

    assert predictions_without_pickle == predictions_with_pickle
