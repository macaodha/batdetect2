import pytest
import torch

from batdetect2.models.blocks import (
    ConvBlock,
    ConvConfig,
    FreqCoordConvDownBlock,
    FreqCoordConvDownConfig,
    FreqCoordConvUpBlock,
    FreqCoordConvUpConfig,
    LayerGroup,
    LayerGroupConfig,
    SelfAttention,
    SelfAttentionConfig,
    StandardConvDownBlock,
    StandardConvDownConfig,
    StandardConvUpBlock,
    StandardConvUpConfig,
    VerticalConv,
    VerticalConvConfig,
    build_layer,
)


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Provides a standard (B, C, H, W) tensor for testing blocks."""
    batch_size, in_channels, height, width = 2, 16, 32, 32
    return torch.randn(batch_size, in_channels, height, width)


@pytest.fixture
def dummy_bottleneck_input() -> torch.Tensor:
    """Provides an input typical for the Bottleneck/SelfAttention (H=1)."""
    return torch.randn(2, 64, 1, 32)


@pytest.mark.parametrize(
    "block_class, expected_h_scale",
    [
        (ConvBlock, 1.0),
        (StandardConvDownBlock, 0.5),
        (StandardConvUpBlock, 2.0),
    ],
)
def test_standard_block_protocol_methods(
    block_class, expected_h_scale, dummy_input
):
    """Test get_output_channels and get_output_height for standard blocks."""
    in_channels = dummy_input.size(1)
    input_height = dummy_input.size(2)
    out_channels = 32

    block = block_class(in_channels=in_channels, out_channels=out_channels)

    assert block.out_channels == out_channels
    assert block.get_output_height(input_height) == int(
        input_height * expected_h_scale
    )


@pytest.mark.parametrize(
    "block_class, expected_h_scale",
    [
        (FreqCoordConvDownBlock, 0.5),
        (FreqCoordConvUpBlock, 2.0),
    ],
)
def test_coord_block_protocol_methods(
    block_class, expected_h_scale, dummy_input
):
    """Test get_output_channels and get_output_height for coord blocks."""
    in_channels = dummy_input.size(1)
    input_height = dummy_input.size(2)
    out_channels = 32

    block = block_class(
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
    )

    assert block.out_channels == out_channels
    assert block.get_output_height(input_height) == int(
        input_height * expected_h_scale
    )


def test_vertical_conv_forward_shape(dummy_input):
    """Test that VerticalConv correctly collapses the height dimension to 1."""
    in_channels = dummy_input.size(1)
    input_height = dummy_input.size(2)
    out_channels = 32

    block = VerticalConv(in_channels, out_channels, input_height)
    output = block(dummy_input)

    assert output.shape == (2, out_channels, 1, 32)
    assert block.out_channels == out_channels


def test_self_attention_forward_shape(dummy_bottleneck_input):
    """Test that SelfAttention maintains the exact shape."""
    in_channels = dummy_bottleneck_input.size(1)
    attention_channels = 32

    block = SelfAttention(
        in_channels=in_channels, attention_channels=attention_channels
    )
    output = block(dummy_bottleneck_input)

    assert output.shape == dummy_bottleneck_input.shape
    assert block.out_channels == in_channels


def test_self_attention_weights(dummy_bottleneck_input):
    """Test that attention weights sum to 1 over the time sequence."""
    in_channels = dummy_bottleneck_input.size(1)
    block = SelfAttention(in_channels=in_channels, attention_channels=32)

    weights = block.compute_attention_weights(dummy_bottleneck_input)

    # Weights shape should be (B, T, T) where T is time (width)
    batch_size = dummy_bottleneck_input.size(0)
    time_steps = dummy_bottleneck_input.size(3)

    assert weights.shape == (batch_size, time_steps, time_steps)

    # Summing across the keys (dim=1) for each query should equal 1.0
    sum_weights = weights.sum(dim=1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)


@pytest.mark.parametrize(
    "layer_config, expected_type",
    [
        (ConvConfig(out_channels=32), ConvBlock),
        (StandardConvDownConfig(out_channels=32), StandardConvDownBlock),
        (StandardConvUpConfig(out_channels=32), StandardConvUpBlock),
        (FreqCoordConvDownConfig(out_channels=32), FreqCoordConvDownBlock),
        (FreqCoordConvUpConfig(out_channels=32), FreqCoordConvUpBlock),
        (SelfAttentionConfig(attention_channels=32), SelfAttention),
        (VerticalConvConfig(channels=32), VerticalConv),
    ],
)
def test_build_layer_factory(layer_config, expected_type):
    """Test that the factory dynamically builds the correct block."""
    input_height = 32
    in_channels = 16

    layer = build_layer(
        input_height=input_height,
        in_channels=in_channels,
        config=layer_config,
    )

    assert isinstance(layer, expected_type)


def test_layer_group_from_config_and_forward(dummy_input):
    """Test that LayerGroup successfully chains multiple blocks."""
    in_channels = dummy_input.size(1)
    input_height = dummy_input.size(2)

    config = LayerGroupConfig(
        layers=[
            ConvConfig(out_channels=32),
            StandardConvDownConfig(out_channels=64),
        ]
    )

    layer_group = build_layer(
        input_height=input_height,
        in_channels=in_channels,
        config=config,
    )

    assert isinstance(layer_group, LayerGroup)
    assert len(layer_group.layers) == 2

    # The group should report the output channels of the LAST block
    assert layer_group.out_channels == 64

    # The group should report the accumulated height changes
    assert layer_group.get_output_height(input_height) == input_height // 2

    output = layer_group(dummy_input)

    # Shape should reflect: Conv (stays 32x32) -> DownConv (halves to 16x16)
    assert output.shape == (2, 64, 16, 16)
