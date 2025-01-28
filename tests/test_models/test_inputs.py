import torch
from hypothesis import given
from hypothesis import strategies as st

from batdetect2.models import ModelConfig, ModelType, get_backbone


@given(
    input_width=st.integers(min_value=50, max_value=1500),
    input_height=st.integers(min_value=1, max_value=16),
    model_type=st.sampled_from(ModelType),
)
def test_model_can_process_spectrograms_of_any_width(
    input_width,
    input_height,
    model_type,
):
    # Input height must be divisible by 8
    input_height = 8 * input_height

    input = torch.rand([1, 1, input_height, input_width])

    model = get_backbone(
        config=ModelConfig(
            name=model_type,  # type: ignore
            input_height=input_height,
        ),
    )

    output = model(input)
    assert output.shape[0] == 1
    assert output.shape[1] == model.out_channels
    assert output.shape[2] == input_height
    assert output.shape[3] == input_width
