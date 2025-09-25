from typing import Optional

import numpy as np


def average_precision(
    y_true,
    y_score,
    num_positives: Optional[int] = None,
) -> float:
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if num_positives is None:
        num_positives = y_true.sum()

    # Remove non-detections
    valid_inds = y_score > 0
    y_true = y_true[valid_inds]
    y_score = y_score[valid_inds]

    # Sort by score
    sort_ind = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_ind]

    false_pos_c = np.cumsum(1 - y_true_sorted)
    true_pos_c = np.cumsum(y_true_sorted)

    recall = true_pos_c / num_positives
    precision = true_pos_c / np.maximum(
        true_pos_c + false_pos_c,
        np.finfo(np.float64).eps,
    )

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    # pascal 12 way
    mprec = np.hstack((0, precision, 0))
    mrec = np.hstack((0, recall, 1))
    for ii in range(mprec.shape[0] - 2, -1, -1):
        mprec[ii] = np.maximum(mprec[ii], mprec[ii + 1])
    inds = np.where(np.not_equal(mrec[1:], mrec[:-1]))[0] + 1
    ave_prec = ((mrec[inds] - mrec[inds - 1]) * mprec[inds]).sum()

    return ave_prec
