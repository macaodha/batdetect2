import numpy as np

from batdetect2.utils.arrays import adjust_width, extend_width


def test_extend_width():
    array = np.random.random([1, 1, 128, 100])

    extended = extend_width(array, 100)

    assert extended.shape == (1, 1, 128, 200)


def test_can_adjust_short_width():
    array = np.random.random([1, 1, 128, 100])
    extended = adjust_width(array, 512)
    assert extended.shape == (1, 1, 128, 512)


def test_can_adjust_long_width():
    array = np.random.random([1, 1, 128, 512])
    extended = adjust_width(array, 256)
    assert extended.shape == (1, 1, 128, 256)
