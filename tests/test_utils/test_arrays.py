import torch

from batdetect2.utils.arrays import adjust_width, extend_width


def test_extend_width():
    array = torch.rand([1, 1, 128, 100])
    extended = extend_width(array, 100)
    assert extended.shape == (1, 1, 128, 200)

    extended = extend_width(array, 100, axis=0)
    assert extended.shape == (101, 1, 128, 100)

    extended = extend_width(array, 100, axis=1)
    assert extended.shape == (1, 101, 128, 100)

    extended = extend_width(array, 100, axis=2)
    assert extended.shape == (1, 1, 228, 100)

    extended = extend_width(array, 100, axis=3)
    assert extended.shape == (1, 1, 128, 200)

    extended = extend_width(array, 100, axis=-2)
    assert extended.shape == (1, 1, 228, 100)


def test_extends_with_value():
    array = torch.rand([1, 1, 128, 100])
    extended = extend_width(array, 100, value=-1)
    torch.testing.assert_close(
        extended[:, :, :, 100:],
        torch.ones_like(array) * -1,
        rtol=0,
        atol=0,
    )


def test_can_adjust_short_width():
    array = torch.rand([1, 1, 128, 100])
    extended = adjust_width(array, 512)
    assert extended.shape == (1, 1, 128, 512)

    extended = adjust_width(array, 512, axis=0)
    assert extended.shape == (512, 1, 128, 100)

    extended = adjust_width(array, 512, axis=1)
    assert extended.shape == (1, 512, 128, 100)

    extended = adjust_width(array, 512, axis=2)
    assert extended.shape == (1, 1, 512, 100)

    extended = adjust_width(array, 512, axis=3)
    assert extended.shape == (1, 1, 128, 512)


def test_can_adjust_long_width():
    array = torch.rand([1, 1, 128, 512])
    extended = adjust_width(array, 256)
    assert extended.shape == (1, 1, 128, 256)
