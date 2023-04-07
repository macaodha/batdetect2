import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    roc_curve,
)


def compute_error_auc(op_str, gt, pred, prob):

    # classification error
    pred_int = (pred > prob).astype(np.int)
    class_acc = (pred_int == gt).mean() * 100.0

    # ROC - area under curve
    fpr, tpr, thresholds = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    print(
        op_str
        + ", class acc = {:.3f}, ROC AUC = {:.3f}".format(class_acc, roc_auc)
    )
    # return class_acc, roc_auc


def calc_average_precision(recall, precision):

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    # pascal 12 way
    mprec = np.hstack((0, precision, 0))
    mrec = np.hstack((0, recall, 1))
    for ii in range(mprec.shape[0] - 2, -1, -1):
        mprec[ii] = np.maximum(mprec[ii], mprec[ii + 1])
    inds = np.where(np.not_equal(mrec[1:], mrec[:-1]))[0] + 1
    ave_prec = ((mrec[inds] - mrec[inds - 1]) * mprec[inds]).sum()

    return float(ave_prec)


def calc_recall_at_x(recall, precision, x=0.95):
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    inds = np.where(precision[::-1] > x)[0]
    if len(inds) > 0:
        return float(recall[::-1][inds[0]])
    else:
        return 0.0


def compute_affinity_1d(pred_box, gt_boxes, threshold):
    # first entry is start time
    score = np.abs(pred_box[0] - gt_boxes[:, 0])
    valid_detection = np.min(score) <= threshold
    return valid_detection, np.argmin(score)


def compute_pre_rec(
    gts,
    preds,
    eval_mode,
    class_of_interest,
    num_classes,
    threshold,
    ignore_start_end,
):
    """
    Computes precision and recall. Assumes that each file has been exhaustively
    annotated. Will not count predicted detection with a start time that is within
    ignore_start_end miliseconds of the start or end of the file.

    eval_mode == 'detection'
      Returns overall detection results (not per class)

    eval_mode == 'per_class'
      Filters ground truth based on class of interest. This will ignore predictions
      assigned to gt with unknown class.

    eval_mode = 'top_class'
       Turns the problem into a binary one and selects the top predicted class
       for each predicted detection

    """

    # get predictions and put in array
    pred_boxes = []
    confidence = []
    pred_class = []
    file_ids = []
    for pid, pp in enumerate(preds):

        # filter predicted calls that are too near the start or end of the file
        file_dur = gts[pid]["duration"]
        valid_inds = (pp["start_times"] >= ignore_start_end) & (
            pp["start_times"] <= (file_dur - ignore_start_end)
        )

        pred_boxes.append(
            np.vstack(
                (
                    pp["start_times"][valid_inds],
                    pp["end_times"][valid_inds],
                    pp["low_freqs"][valid_inds],
                    pp["high_freqs"][valid_inds],
                )
            ).T
        )

        if eval_mode == "detection":
            # overall detection
            confidence.append(pp["det_probs"][valid_inds])
        elif eval_mode == "per_class":
            # per class
            confidence.append(
                pp["class_probs"].T[valid_inds, class_of_interest]
            )
        elif eval_mode == "top_class":
            # per class - note that sometimes 'class_probs' can be num_classes+1 in size
            top_class = np.argmax(
                pp["class_probs"].T[valid_inds, :num_classes], 1
            )
            confidence.append(pp["class_probs"].T[valid_inds, top_class])
            pred_class.append(top_class)

        # be careful, assuming the order in the list is same as GT
        file_ids.append([pid] * valid_inds.sum())

    confidence = np.hstack(confidence)
    file_ids = np.hstack(file_ids).astype(np.int)
    pred_boxes = np.vstack(pred_boxes)
    if len(pred_class) > 0:
        pred_class = np.hstack(pred_class)

    # extract relevant ground truth boxes
    gt_boxes = []
    gt_assigned = []
    gt_class = []
    gt_generic_class = []
    num_positives = 0
    for gg in gts:

        # filter ground truth calls that are too near the start or end of the file
        file_dur = gg["duration"]
        valid_inds = (gg["start_times"] >= ignore_start_end) & (
            gg["start_times"] <= (file_dur - ignore_start_end)
        )

        # note, files with the incorrect duration will cause a problem
        if (gg["start_times"] > file_dur).sum() > 0:
            print("Error: file duration incorrect for", gg["id"])
            assert False

        boxes = np.vstack(
            (
                gg["start_times"][valid_inds],
                gg["end_times"][valid_inds],
                gg["low_freqs"][valid_inds],
                gg["high_freqs"][valid_inds],
            )
        ).T
        gen_class = gg["class_ids"][valid_inds] == -1
        class_ids = gg["class_ids"][valid_inds]

        # keep track of the number of relevant ground truth calls
        if eval_mode == "detection":
            # all valid ones
            num_positives += len(gg["start_times"][valid_inds])
        elif eval_mode == "per_class":
            # all valid ones with class of interest
            num_positives += (
                gg["class_ids"][valid_inds] == class_of_interest
            ).sum()
        elif eval_mode == "top_class":
            # all valid ones with non generic class
            num_positives += (gg["class_ids"][valid_inds] > -1).sum()

        # find relevant classes (i.e. class_of_interest) and events without known class (i.e. generic class, -1)
        if eval_mode == "per_class":
            class_inds = (class_ids == class_of_interest) | (class_ids == -1)
            boxes = boxes[class_inds, :]
            gen_class = gen_class[class_inds]
            class_ids = class_ids[class_inds]

        gt_assigned.append(np.zeros(boxes.shape[0]))
        gt_boxes.append(boxes)
        gt_generic_class.append(gen_class)
        gt_class.append(class_ids)

    # loop through detections and keep track of those that have been assigned
    true_pos = np.zeros(confidence.shape[0])
    valid_inds = np.ones(confidence.shape[0]) == 1  # intialize to True
    sorted_inds = np.argsort(confidence)[::-1]  # sort high to low
    for ii, ind in enumerate(sorted_inds):
        gt_id = file_ids[ind]

        valid_det = False
        if gt_boxes[gt_id].shape[0] > 0:
            # compute overlap
            valid_det, det_ind = compute_affinity_1d(
                pred_boxes[ind], gt_boxes[gt_id], threshold
            )

        # valid detection that has not already been assigned
        if valid_det and (gt_assigned[gt_id][det_ind] == 0):

            count_as_true_pos = True
            if eval_mode == "top_class" and (
                gt_class[gt_id][det_ind] != pred_class[ind]
            ):
                # needs to be the same class
                count_as_true_pos = False

            if count_as_true_pos:
                true_pos[ii] = 1

            gt_assigned[gt_id][det_ind] = 1

            # if event is generic class (i.e. gt_generic_class[gt_id][det_ind] is True)
            # and eval_mode != 'detection', then ignore it
            if gt_generic_class[gt_id][det_ind]:
                if eval_mode == "per_class" or eval_mode == "top_class":
                    valid_inds[ii] = False

    # store threshold values - used for plotting
    conf_sorted = np.sort(confidence)[::-1][valid_inds]
    thresholds = np.linspace(0.1, 0.9, 9)
    thresholds_inds = np.zeros(len(thresholds), dtype=np.int)
    for ii, tt in enumerate(thresholds):
        thresholds_inds[ii] = np.argmin(conf_sorted > tt)
    thresholds_inds[thresholds_inds == 0] = -1

    # compute precision and recall
    true_pos = true_pos[valid_inds]
    false_pos_c = np.cumsum(1 - true_pos)
    true_pos_c = np.cumsum(true_pos)

    recall = true_pos_c / num_positives
    precision = true_pos_c / np.maximum(
        true_pos_c + false_pos_c, np.finfo(np.float64).eps
    )

    results = {}
    results["recall"] = recall
    results["precision"] = precision
    results["num_gt"] = num_positives

    results["thresholds"] = thresholds
    results["thresholds_inds"] = thresholds_inds

    if num_positives == 0:
        results["avg_prec"] = np.nan
        results["rec_at_x"] = np.nan
    else:
        results["avg_prec"] = np.round(
            calc_average_precision(recall, precision), 5
        )
        results["rec_at_x"] = np.round(calc_recall_at_x(recall, precision), 5)

    return results


def compute_file_accuracy_simple(gts, preds, num_classes):
    """
    Evaluates the prediction accuracy at a file level.
    Does not include files that have more than one class (or the generic class).

    Simply chooses the class per file that has the highest probability overall.
    """

    gt_valid = []
    pred_valid = []
    for ii in range(len(gts)):
        gt_class = np.unique(gts[ii]["class_ids"])
        if len(gt_class) == 1 and gt_class[0] != -1:
            gt_valid.append(gt_class[0])
            pred = preds[ii]["class_probs"][:num_classes, :].T
            pred_valid.append(np.argmax(pred.mean(0)))
    acc = (np.array(gt_valid) == np.array(pred_valid)).mean()

    res = {}
    res["num_valid_files"] = len(gt_valid)
    res["num_total_files"] = len(gts)
    res["gt_valid_file"] = gt_valid
    res["pred_valid_file"] = pred_valid
    res["file_acc"] = np.round(acc, 5)
    return res


def compute_file_accuracy(gts, preds, num_classes):
    """
    Evaluates the prediction accuracy at a file level.
    Does not include files that have more than one class (or the unknown class).

    Tries several different detection thresholds and picks the best one.
    """

    # compute min and max scoring range - then threshold
    min_val = 0
    mins = [
        pp["class_probs"].min()
        for pp in preds
        if pp["class_probs"].shape[1] > 0
    ]
    if len(mins) > 0:
        min_val = np.min(mins)

    max_val = 1.0
    maxes = [
        pp["class_probs"].max()
        for pp in preds
        if pp["class_probs"].shape[1] > 0
    ]
    if len(maxes) > 0:
        max_val = np.max(maxes)

    thresh = np.linspace(min_val, max_val, 11)[:10]

    # loop over the files and store the accuracy at different prediction thresholds
    # only include gt files that have one valid species
    gt_valid = []
    pred_valid_all = []
    for ii in range(len(gts)):
        gt_class = np.unique(gts[ii]["class_ids"])
        if len(gt_class) == 1 and gt_class[0] != -1:
            gt_valid.append(gt_class[0])
            pred = preds[ii]["class_probs"][:num_classes, :].T
            p_class = np.zeros(len(thresh))
            for tt in range(len(thresh)):
                p_class[tt] = (pred * (pred >= thresh[tt])).sum(0).argmax()
            pred_valid_all.append(p_class)

    # pick the result corresponding to the overall best threshold
    pred_valid_all = np.vstack(pred_valid_all)
    acc_per_thresh = (
        np.array(gt_valid)[..., np.newaxis] == pred_valid_all
    ).mean(0)
    best_thresh = np.argmax(acc_per_thresh)
    best_acc = acc_per_thresh[best_thresh]
    pred_valid = pred_valid_all[:, best_thresh].astype(np.int).tolist()

    res = {}
    res["num_valid_files"] = len(gt_valid)
    res["num_total_files"] = len(gts)
    res["gt_valid_file"] = gt_valid
    res["pred_valid_file"] = pred_valid
    res["file_acc"] = np.round(best_acc, 5)

    return res


def evaluate_predictions(
    gts, preds, class_names, detection_overlap, ignore_start_end=0.0
):
    """
    Computes metrics derived from the precision and recall.
    Assumes that gts and preds are both lists of the same lengths, with ground
    truth and predictions contained within.

    Returns the overall detection results, and per class results
    """

    assert len(gts) == len(preds)
    num_classes = len(class_names)

    # evaluate detection on its own i.e. ignoring class
    det_results = compute_pre_rec(
        gts,
        preds,
        "detection",
        None,
        num_classes,
        detection_overlap,
        ignore_start_end,
    )
    top_class = compute_pre_rec(
        gts,
        preds,
        "top_class",
        None,
        num_classes,
        detection_overlap,
        ignore_start_end,
    )
    det_results["top_class"] = top_class

    # per class evaluation
    det_results["class_pr"] = []
    for cc in range(num_classes):
        res = compute_pre_rec(
            gts,
            preds,
            "per_class",
            cc,
            num_classes,
            detection_overlap,
            ignore_start_end,
        )
        res["name"] = class_names[cc]
        det_results["class_pr"].append(res)

    # ignores classes that are not present in the test set
    det_results["avg_prec_class"] = np.mean(
        [rs["avg_prec"] for rs in det_results["class_pr"] if rs["num_gt"] > 0]
    )
    det_results["avg_prec_class"] = np.round(det_results["avg_prec_class"], 5)

    # file level evaluation
    res_file = compute_file_accuracy(gts, preds, num_classes)
    det_results.update(res_file)

    return det_results
