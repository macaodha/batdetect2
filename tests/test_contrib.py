"""Test suite to ensure user provided files are correctly processed."""

from pathlib import Path

from click.testing import CliRunner

from batdetect2.cli import cli

runner = CliRunner()


def test_can_process_jeff37_files(
    contrib_dir: Path,
    tmp_path: Path,
):
    """This test stems from issue #31.

    A user provided a set of files which which batdetect2 cli failed and
    generated the following error message:

        [2272] "Error processing file!: negative dimensions are not allowed"

    This test ensures that the error message is not generated when running
    batdetect2 cli with the same set of files.
    """
    path = contrib_dir / "jeff37"
    assert path.exists()

    results_dir = tmp_path / "results"
    result = runner.invoke(
        cli,
        [
            "detect",
            str(path),
            str(results_dir),
            "0.3",
        ],
    )
    assert result.exit_code == 0
    assert results_dir.exists()
    assert len(list(results_dir.glob("*.csv"))) == 5
    assert len(list(results_dir.glob("*.json"))) == 5


def test_can_process_padpadpadpad_files(
    contrib_dir: Path,
    tmp_path: Path,
):
    """This test stems from issue #29.

    Batdetect2 cli failed on the files provided by the user @padpadpadpad
    with the following error message:

        AttributeError: module 'numpy' has no attribute 'AxisError'

    This test ensures that the files are processed without any error.
    """
    path = contrib_dir / "padpadpadpad"
    assert path.exists()
    results_dir = tmp_path / "results"
    result = runner.invoke(
        cli,
        [
            "detect",
            str(path),
            str(results_dir),
            "0.3",
        ],
    )
    assert result.exit_code == 0
    assert results_dir.exists()
    assert len(list(results_dir.glob("*.csv"))) == 2
    assert len(list(results_dir.glob("*.json"))) == 2
