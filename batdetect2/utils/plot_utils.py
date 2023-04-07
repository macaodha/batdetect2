import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import PatchCollection
from sklearn.metrics import confusion_matrix


def create_box_image(
    spec,
    fig,
    detections_ip,
    start_time,
    end_time,
    duration,
    params,
    max_val,
    hide_axis=True,
    plot_class_names=False,
):
    # filter detections
    stop_time = start_time + duration
    detections = []
    for bb in detections_ip:
        if (bb["start_time"] >= start_time) and (
            bb["start_time"] < stop_time - 0.02
        ):  # (bb['end_time'] < end_time):
            detections.append(bb)

    # create figure
    freq_scale = 1000  # turn Hz to kHz
    min_freq = params["min_freq"] // freq_scale
    max_freq = params["max_freq"] // freq_scale
    y_extent = [0, duration, min_freq, max_freq]

    if hide_axis:
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax = plt.gca()

    plt.imshow(
        spec,
        aspect="auto",
        cmap="plasma",
        extent=y_extent,
        vmin=0,
        vmax=max_val,
    )
    boxes = plot_bounding_box_patch_ann(detections, freq_scale, start_time)
    ax.add_collection(PatchCollection(boxes, match_original=True))
    plt.grid(False)

    if plot_class_names:
        for ii, bb in enumerate(boxes):
            txt = " ".join(
                [sp[:3] for sp in detections_ip[ii]["class"].split(" ")]
            )
            font_info = {
                "color": "white",
                "size": 10,
                "weight": "bold",
                "alpha": bb.get_alpha(),
            }
            y_pos = bb.get_xy()[1] + bb.get_height()
            if y_pos > (max_freq - 10):
                y_pos = max_freq - 10
            plt.gca().text(bb.get_xy()[0], y_pos, txt, fontdict=font_info)


def save_ann_spec(
    op_path,
    spec,
    min_freq,
    max_freq,
    duration,
    start_time,
    title_text="",
    anns=None,
):
    # create figure and plot boxes
    freq_scale = 1000  # turn Hz to kHz
    min_freq = min_freq // freq_scale
    max_freq = max_freq // freq_scale
    y_extent = [0, duration, min_freq, max_freq]

    plt.close("all")
    fig = plt.figure(
        0, figsize=(spec.shape[1] / 100, spec.shape[0] / 100), dpi=100
    )
    plt.imshow(
        spec,
        aspect="auto",
        cmap="plasma",
        extent=y_extent,
        vmin=0,
        vmax=spec.max() * 1.1,
    )

    plt.ylabel("Freq - kHz")
    plt.xlabel("Time - secs")
    if title_text != "":
        plt.title(title_text)
    plt.tight_layout()

    if anns is not None:
        # drawing bounding boxes and class names
        boxes = plot_bounding_box_patch_ann(anns, freq_scale, start_time)
        plt.gca().add_collection(PatchCollection(boxes, match_original=True))
        for ii, bb in enumerate(boxes):
            txt = " ".join([sp[:3] for sp in anns[ii]["class"].split(" ")])
            font_info = {
                "color": "white",
                "size": 10,
                "weight": "bold",
                "alpha": bb.get_alpha(),
            }
            y_pos = bb.get_xy()[1] + bb.get_height()
            if y_pos > (max_freq - 10):
                y_pos = max_freq - 10
            plt.gca().text(bb.get_xy()[0], y_pos, txt, fontdict=font_info)

    print("Saving figure to:", op_path)
    plt.savefig(op_path)


def plot_pts(
    fig_id, feats, class_names, colors, marker_size=4.0, plot_legend=False
):
    plt.figure(fig_id)
    un_class, labels = np.unique(class_names, return_inverse=True)
    un_labels = np.unique(labels)
    if un_labels.shape[0] > len(colors):
        colors = [
            plt.cm.jet(float(ii) / un_labels.shape[0]) for ii in un_labels
        ]

    for ii, u in enumerate(un_labels):
        inds = np.where(labels == u)[0]
        plt.scatter(
            feats[inds, 0],
            feats[inds, 1],
            c=colors[ii],
            label=str(un_class[ii]),
            s=marker_size,
        )
    if plot_legend:
        plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title("downsampled features")


def plot_bounding_box_patch(pred, freq_scale, ecolor="w"):
    patch_collect = []
    for bb in range(len(pred["start_times"])):
        xx = pred["start_times"][bb]
        ww = pred["end_times"][bb] - pred["start_times"][bb]
        yy = pred["low_freqs"][bb] / freq_scale
        hh = (pred["high_freqs"][bb] - pred["low_freqs"][bb]) / freq_scale

        if "det_probs" in pred.keys():
            alpha_val = pred["det_probs"][bb]
        else:
            alpha_val = 1.0
        patch_collect.append(
            patches.Rectangle(
                (xx, yy),
                ww,
                hh,
                linewidth=1,
                edgecolor=ecolor,
                facecolor="none",
                alpha=alpha_val,
            )
        )
    return patch_collect


def plot_bounding_box_patch_ann(anns, freq_scale, start_time):
    patch_collect = []
    for aa in range(len(anns)):
        xx = anns[aa]["start_time"] - start_time
        ww = anns[aa]["end_time"] - anns[aa]["start_time"]
        yy = anns[aa]["low_freq"] / freq_scale
        hh = (anns[aa]["high_freq"] - anns[aa]["low_freq"]) / freq_scale
        if "det_prob" in anns[aa]:
            alpha = anns[aa]["det_prob"]
        else:
            alpha = 1.0
        patch_collect.append(
            patches.Rectangle(
                (xx, yy),
                ww,
                hh,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
                alpha=alpha,
            )
        )
    return patch_collect


def plot_spec(
    spec,
    sampling_rate,
    duration,
    gt,
    pred,
    params,
    plot_title,
    op_file_name,
    pred_2d_hm,
    plot_boxes=True,
    fixed_aspect=True,
):
    if fixed_aspect:
        # ouptut image will be this width irrespective of the duration of the audio file
        width = 12
    else:
        width = 12 * duration

    fig = plt.figure(1, figsize=(width, 8))
    ax0 = plt.axes([0.05, 0.65, 0.9, 0.30])  # l b w h
    ax1 = plt.axes([0.05, 0.33, 0.9, 0.30])
    ax2 = plt.axes([0.05, 0.01, 0.9, 0.30])

    freq_scale = 1000  # turn Hz in kHz
    # duration = au.x_coords_to_time(spec.shape[1], sampling_rate, params['fft_win_length'], params['fft_overlap'])
    y_extent = [
        0,
        duration,
        params["min_freq"] // freq_scale,
        params["max_freq"] // freq_scale,
    ]

    # plot gt boxes
    ax0.imshow(spec, aspect="auto", cmap="plasma", extent=y_extent)
    ax0.xaxis.set_ticklabels([])
    font_info = {"color": "white", "size": 12, "weight": "bold"}
    ax0.text(
        0, params["min_freq"] // freq_scale, "Ground Truth", fontdict=font_info
    )

    plt.grid(False)
    if plot_boxes:
        boxes = plot_bounding_box_patch(gt, freq_scale)
        ax0.add_collection(PatchCollection(boxes, match_original=True))
        for ii, bb in enumerate(boxes):
            class_id = int(gt["class_ids"][ii])
            if class_id < 0:
                txt = params["generic_class"][0]
            else:
                txt = params["class_names_short"][class_id]
            font_info = {
                "color": "white",
                "size": 10,
                "weight": "bold",
                "alpha": bb.get_alpha(),
            }
            y_pos = bb.get_xy()[1] + bb.get_height()
            ax0.text(bb.get_xy()[0], y_pos, txt, fontdict=font_info)

    # plot predicted boxes
    ax1.imshow(spec, aspect="auto", cmap="plasma", extent=y_extent)
    ax1.xaxis.set_ticklabels([])
    font_info = {"color": "white", "size": 12, "weight": "bold"}
    ax1.text(
        0, params["min_freq"] // freq_scale, "Prediction", fontdict=font_info
    )

    plt.grid(False)
    if plot_boxes:
        boxes = plot_bounding_box_patch(pred, freq_scale)
        ax1.add_collection(PatchCollection(boxes, match_original=True))
        for ii, bb in enumerate(boxes):
            if pred["class_probs"].shape[0] > len(params["class_names_short"]):
                class_id = pred["class_probs"][:-1, ii].argmax()
            else:
                class_id = pred["class_probs"][:, ii].argmax()
            txt = params["class_names_short"][class_id]
            font_info = {
                "color": "white",
                "size": 10,
                "weight": "bold",
                "alpha": bb.get_alpha(),
            }
            y_pos = bb.get_xy()[1] + bb.get_height()
            ax1.text(bb.get_xy()[0], y_pos, txt, fontdict=font_info)

    # plot 2D heatmap
    if pred_2d_hm is not None:
        min_val = 0.0 if pred_2d_hm.min() > 0.0 else pred_2d_hm.min()
        max_val = 1.0 if pred_2d_hm.max() < 1.0 else pred_2d_hm.max()

        ax2.imshow(
            pred_2d_hm,
            aspect="auto",
            cmap="plasma",
            extent=y_extent,
            clim=[min_val, max_val],
        )
        # ax2.xaxis.set_ticklabels([])
        font_info = {"color": "white", "size": 12, "weight": "bold"}
        ax2.text(
            0, params["min_freq"] // freq_scale, "Heatmap", fontdict=font_info
        )

        plt.grid(False)

    plt.suptitle(plot_title)
    if op_file_name is not None:
        fig.savefig(op_file_name)

    plt.close(1)


def plot_pr_curve(
    op_dir, plt_title, file_name, results, file_type="png", title_text=""
):
    precision = results["precision"]
    recall = results["recall"]
    avg_prec = results["avg_prec"]

    plt.figure(0, figsize=(10, 8))
    plt.plot(recall, precision)
    plt.ylabel("Precision", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    if title_text != "":
        plt.title(title_text, fontdict={"fontsize": 28})
    else:
        plt.title(plt_title + " {:.3f}\n".format(avg_prec))
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(op_dir + file_name + "." + file_type)
    plt.close(0)


def plot_pr_curve_class(
    op_dir, plt_title, file_name, results, file_type="png", title_text=""
):
    plt.figure(0, figsize=(10, 8))
    plt.ylabel("Precision", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(True)
    linestyles = ["-", ":", "--"]
    markers = ["o", "v", ">", "^", "<", "s", "P", "X", "*"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot the PR curves
    for ii, rr in enumerate(results["class_pr"]):
        class_name = " ".join([sp[:3] for sp in rr["name"].split(" ")])
        cur_color = colors[int(ii % 10)]
        plt.plot(
            rr["recall"],
            rr["precision"],
            label=class_name,
            color=cur_color,
            linestyle=linestyles[int(ii // 10)],
            lw=2.5,
        )

        # print(class_name)
        # plot the location of the confidence threshold values
        for jj, tt in enumerate(rr["thresholds"]):
            ind = rr["thresholds_inds"][jj]
            if ind > -1:
                plt.plot(
                    rr["recall"][ind],
                    rr["precision"][ind],
                    markers[jj],
                    color=cur_color,
                    ms=10,
                )
                # print(np.round(tt,2), np.round(rr['recall'][ind],3), np.round(rr['precision'][ind],3))

    if title_text != "":
        plt.title(title_text, fontdict={"fontsize": 28})
    else:
        plt.title(plt_title + " {:.3f}\n".format(results["avg_prec_class"]))
    plt.legend(loc="lower left", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(op_dir + file_name + "." + file_type)
    plt.close(0)


def plot_confusion_matrix(
    op_dir,
    op_file,
    gt,
    pred,
    file_acc,
    class_names_long,
    verbose=False,
    file_type="png",
    title_text="",
):
    # shorten the class names for plotting
    class_names = []
    for cc in class_names_long:
        class_name_sm = "".join([cc_sm[:3] + " " for cc_sm in cc.split(" ")])[
            :-1
        ]
        class_names.append(class_name_sm)

    num_classes = len(class_names)
    cm = confusion_matrix(gt, pred, labels=np.arange(num_classes)).astype(
        np.float32
    )
    cm_norm = cm.sum(1)

    valid_inds = np.where(cm_norm > 0)[0]
    cm[valid_inds, :] = cm[valid_inds, :] / cm_norm[valid_inds][..., np.newaxis]
    cm[np.where(cm_norm == -0)[0], :] = np.nan

    if verbose:
        print("Per class accuracy:")
        str_len = np.max([len(cc) for cc in class_names_long]) + 5
        accs = np.diag(cm)
        for ii, cc in enumerate(class_names_long):
            if np.isnan(accs[ii]):
                print(str(ii).ljust(5) + cc.ljust(str_len))
            else:
                print(
                    str(ii).ljust(5)
                    + cc.ljust(str_len)
                    + "{:.2f}".format(accs[ii] * 100)
                )

    plt.figure(0, figsize=(10, 8))
    plt.imshow(cm, vmin=0, vmax=1, cmap="plasma")
    plt.colorbar()
    plt.xticks(np.arange(cm.shape[1]), class_names, rotation="vertical")
    plt.yticks(np.arange(cm.shape[0]), class_names)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Ground Truth", fontsize=20)
    if title_text != "":
        plt.title(title_text, fontdict={"fontsize": 28})
    else:
        plt.title(op_file + " {:.3f}\n".format(file_acc))
    plt.tight_layout()
    plt.savefig(op_dir + op_file + "." + file_type)
    plt.close("all")


class LossPlotter(object):
    def __init__(
        self,
        op_file_name,
        duration,
        labels,
        ylim,
        class_names,
        axis_labels=None,
        logy=False,
    ):
        self.reset()
        self.op_file_name = op_file_name
        self.duration = duration  # length of x axis
        self.labels = labels
        self.ylim = ylim
        self.class_names = class_names
        self.axis_labels = axis_labels
        self.logy = logy

    def reset(self):
        self.epochs = []
        self.vals = []

    def update_and_save(self, epoch, val, gt=None, pred=None):
        self.epochs.append(epoch)
        self.vals.append(val)
        self.save_plot()
        self.save_json()
        if gt is not None:
            self.save_confusion_matrix(gt, pred)

    def save_plot(self):
        linestyles = ["-", ":", "--"]
        plt.figure(0, figsize=(8, 5))
        for ii in range(len(self.vals[0])):
            l_vals = [vv[ii] for vv in self.vals]
            plt.plot(
                self.epochs,
                l_vals,
                label=self.labels[ii],
                linestyle=linestyles[int(ii // 10)],
            )
        plt.xlim(0, np.maximum(self.duration, len(self.vals)))
        if self.ylim is not None:
            plt.ylim(self.ylim[0], self.ylim[1])
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])
            plt.ylabel(self.axis_labels[1])
        if self.logy:
            plt.gca().set_yscale("log")
        plt.grid(True)
        plt.legend(
            bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0
        )
        plt.tight_layout()
        plt.savefig(self.op_file_name)
        plt.close(0)

    def save_json(self):
        data = {}
        data["epochs"] = self.epochs
        for ii in range(len(self.vals[0])):
            data[self.labels[ii]] = [round(vv[ii], 4) for vv in self.vals]
        with open(self.op_file_name[:-4] + ".json", "w") as da:
            json.dump(data, da, indent=2)

    def save_confusion_matrix(self, gt, pred):
        plt.figure(0)
        cm = confusion_matrix(
            gt, pred, labels=np.arange(len(self.class_names))
        ).astype(np.float32)
        cm_norm = cm.sum(1)
        valid_inds = np.where(cm_norm > 0)[0]
        cm[valid_inds, :] = (
            cm[valid_inds, :] / cm_norm[valid_inds][..., np.newaxis]
        )
        plt.imshow(cm, vmin=0, vmax=1, cmap="plasma")
        plt.colorbar()
        plt.xticks(
            np.arange(cm.shape[1]), self.class_names, rotation="vertical"
        )
        plt.yticks(np.arange(cm.shape[0]), self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(self.op_file_name[:-4] + "_cm.png")
        plt.close(0)
