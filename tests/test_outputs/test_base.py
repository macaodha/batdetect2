from pathlib import Path

from batdetect2.outputs.formats.base import make_path_relative


def test_make_path_relative_strips_shared_relative_prefix() -> None:
    audio_dir = Path("example_data/audio")
    path = Path("example_data/audio/subdir/clip.wav")

    relative = make_path_relative(path, audio_dir)

    assert relative == Path("subdir/clip.wav")


def test_make_path_relative_returns_dot_for_matching_relative_dir() -> None:
    audio_dir = Path("example_data/audio")
    path = Path("example_data/audio")

    relative = make_path_relative(path, audio_dir)

    assert relative == Path(".")
