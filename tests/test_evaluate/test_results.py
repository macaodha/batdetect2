import json

from matplotlib.figure import Figure

from batdetect2.evaluate.results import save_evaluation_results


def test_save_evaluation_results_writes_metrics_and_plots(tmp_path) -> None:
    metrics = {"mAP": 0.5}
    figure = Figure()

    save_evaluation_results(
        metrics=metrics,
        plots=[("plots/example.png", figure)],
        output_dir=tmp_path,
    )

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text()) == metrics
    assert (tmp_path / "plots" / "example.png").exists()
