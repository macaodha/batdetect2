import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def x_coords_to_time(x_pos, sampling_rate, fft_win_length, fft_overlap):
    nfft = int(fft_win_length*sampling_rate)
    noverlap = int(fft_overlap*nfft)
    return ((x_pos*(nfft - noverlap)) + noverlap) / sampling_rate
    #return (1.0 - fft_overlap) * fft_win_length * (x_pos + 0.5)  # 0.5 is for center of temporal window


def overall_class_pred(det_prob, class_prob):
    weighted_pred = (class_prob*det_prob).sum(1)
    return weighted_pred / weighted_pred.sum()


def run_nms(outputs, params, sampling_rate):

    pred_det  = outputs['pred_det']   # probability of box
    pred_size = outputs['pred_size']  # box size

    pred_det_nms = non_max_suppression(pred_det, params['nms_kernel_size'])
    freq_rescale = (params['max_freq'] - params['min_freq']) /pred_det.shape[-2]

    # NOTE there will be small differences depending on which sampling rate is chosen
    # as we are choosing the same sampling rate for the entire batch
    duration = x_coords_to_time(pred_det.shape[-1], sampling_rate[0].item(),
                                params['fft_win_length'], params['fft_overlap'])
    top_k = int(duration * params['nms_top_k_per_sec'])
    scores, y_pos, x_pos = get_topk_scores(pred_det_nms, top_k)

    # loop over batch to save outputs
    preds = []
    feats = []
    for ii in range(pred_det_nms.shape[0]):
        # get valid indices
        inds_ord = torch.argsort(x_pos[ii, :])
        valid_inds = scores[ii, inds_ord] > params['detection_threshold']
        valid_inds = inds_ord[valid_inds]

        # create result dictionary
        pred = {}
        pred['det_probs']   = scores[ii, valid_inds]
        pred['x_pos']       = x_pos[ii, valid_inds]
        pred['y_pos']       = y_pos[ii, valid_inds]
        pred['bb_width']    = pred_size[ii, 0, pred['y_pos'], pred['x_pos']]
        pred['bb_height']   = pred_size[ii, 1, pred['y_pos'], pred['x_pos']]
        pred['start_times'] = x_coords_to_time(pred['x_pos'].float() / params['resize_factor'],
                                sampling_rate[ii].item(), params['fft_win_length'], params['fft_overlap'])
        pred['end_times']   = x_coords_to_time((pred['x_pos'].float()+pred['bb_width']) / params['resize_factor'],
                                sampling_rate[ii].item(), params['fft_win_length'], params['fft_overlap'])
        pred['low_freqs']   = (pred_size[ii].shape[1] - pred['y_pos'].float())*freq_rescale + params['min_freq']
        pred['high_freqs']  = pred['low_freqs'] + pred['bb_height']*freq_rescale

        # extract the per class votes
        if 'pred_class' in outputs:
            pred['class_probs'] = outputs['pred_class'][ii, :, y_pos[ii, valid_inds], x_pos[ii, valid_inds]]

        # extract the model features
        if 'features' in outputs:
            feat = outputs['features'][ii, :, y_pos[ii, valid_inds], x_pos[ii, valid_inds]].transpose(0, 1)
            feat = feat.cpu().numpy().astype(np.float32)
            feats.append(feat)

        # convert to numpy
        for kk in pred.keys():
            pred[kk] = pred[kk].cpu().numpy().astype(np.float32)
        preds.append(pred)

    return preds, feats


def non_max_suppression(heat, kernel_size):
    # kernel can be an int or list/tuple    
    if type(kernel_size) is int:
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size

    pad_h = (kernel_size_h - 1) // 2
    pad_w = (kernel_size_w - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel_size_h, kernel_size_w), stride=1, padding=(pad_h, pad_w))
    keep = (hmax == heat).float()

    return heat * keep


def get_topk_scores(scores, K):
    # expects input of size:  batch x 1 x height x width
    batch, _, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds // width).long()
    topk_xs   = (topk_inds % width).long()

    return topk_scores, topk_ys, topk_xs
