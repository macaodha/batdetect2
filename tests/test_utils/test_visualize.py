import numpy as np

from batdetect2.utils.visualize import InteractivePlotter


def test_interactive_plotter_init_builds_integer_label_arrays():
    feats_ds = np.zeros((2, 2))
    spec_slices = [np.zeros((4, 6)), np.zeros((4, 8))]
    call_info = [{"class": "a"}, {"class": "b"}]

    plotter = InteractivePlotter(
        feats_ds=feats_ds,
        feats=feats_ds,
        spec_slices=spec_slices,
        call_info=call_info,
        freq_lims=[0, 1],
        allow_training=False,
    )

    assert plotter.labels.shape == (2,)
    assert np.issubdtype(plotter.labels.dtype, np.integer)
    assert np.issubdtype(plotter.annotated.dtype, np.integer)
