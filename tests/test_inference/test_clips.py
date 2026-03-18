from soundevent import data

from batdetect2.inference.clips import get_recording_clips


def test_get_recording_clips_uses_requested_duration(create_recording) -> None:
    recording = create_recording(duration=2.0, samplerate=256_000)

    clips = get_recording_clips(
        recording,
        duration=0.5,
        overlap=0.0,
        discard_empty=False,
    )

    assert len(clips) == 4
    assert all(isinstance(clip, data.Clip) for clip in clips)
    assert clips[0].start_time == 0.0
    assert clips[0].end_time == 0.5
    assert clips[1].start_time == 0.5
    assert clips[1].end_time == 1.0
    assert clips[2].start_time == 1.0
    assert clips[2].end_time == 1.5
    assert clips[3].start_time == 1.5
    assert clips[3].end_time == 2.0
