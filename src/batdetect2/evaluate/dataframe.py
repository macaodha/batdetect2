from typing import List

import pandas as pd
from soundevent.geometry import compute_bounds

from batdetect2.typing.evaluate import MatchEvaluation


def extract_matches_dataframe(matches: List[MatchEvaluation]) -> pd.DataFrame:
    data = []

    for match in matches:
        gt_start_time = gt_low_freq = gt_end_time = gt_high_freq = None
        pred_start_time = pred_low_freq = pred_end_time = pred_high_freq = None

        sound_event_annotation = match.sound_event_annotation

        if sound_event_annotation is not None:
            geometry = sound_event_annotation.sound_event.geometry
            assert geometry is not None
            gt_start_time, gt_low_freq, gt_end_time, gt_high_freq = (
                compute_bounds(geometry)
            )

        if match.pred_geometry is not None:
            pred_start_time, pred_low_freq, pred_end_time, pred_high_freq = (
                compute_bounds(match.pred_geometry)
            )

        data.append(
            {
                ("recording", "uuid"): match.clip.recording.uuid,
                ("clip", "uuid"): match.clip.uuid,
                ("clip", "start_time"): match.clip.start_time,
                ("clip", "end_time"): match.clip.end_time,
                ("gt", "uuid"): match.sound_event_annotation.uuid
                if match.sound_event_annotation is not None
                else None,
                ("gt", "class"): match.gt_class,
                ("gt", "det"): match.gt_det,
                ("gt", "start_time"): gt_start_time,
                ("gt", "end_time"): gt_end_time,
                ("gt", "low_freq"): gt_low_freq,
                ("gt", "high_freq"): gt_high_freq,
                ("pred", "score"): match.pred_score,
                ("pred", "class"): match.pred_class,
                ("pred", "class_score"): match.pred_class_score,
                ("pred", "start_time"): pred_start_time,
                ("pred", "end_time"): pred_end_time,
                ("pred", "low_freq"): pred_low_freq,
                ("pred", "high_freq"): pred_high_freq,
                ("match", "affinity"): match.affinity,
                **{
                    ("pred_class_score", key): value
                    for key, value in match.pred_class_scores.items()
                },
            }
        )

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)  # type: ignore
    return df
