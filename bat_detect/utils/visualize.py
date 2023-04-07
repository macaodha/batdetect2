import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.svm import LinearSVC

matplotlib_axes_logger.setLevel("ERROR")


colors = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabebe",
    "#469990",
    "#e6beff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]


class InteractivePlotter:
    def __init__(
        self,
        feats_ds,
        feats,
        spec_slices,
        call_info,
        freq_lims,
        allow_training,
    ):
        """
        Plots 2D low dimensional features on left and corresponding spectgrams on
        the right.
        """
        self.feats_ds = feats_ds
        self.feats = feats
        self.clf = None

        self.spec_slices = spec_slices
        self.call_info = call_info
        # _, self.labels = np.unique([cc['class'] for cc in call_info], return_inverse=True)
        self.labels = np.zeros(len(call_info), dtype=np.int)
        self.annotated = np.zeros(
            self.labels.shape[0], dtype=np.int
        )  # can populate this with 1's where we have labels
        self.labels_cols = [
            colors[self.labels[ii]] for ii in range(len(self.labels))
        ]
        self.freq_lims = freq_lims

        self.allow_training = allow_training
        self.pt_size = 5.0
        self.spec_pad = (
            0.2  # this much padding has been applied to the spec slices
        )
        self.fig_width = 12
        self.fig_height = 8

        self.current_id = 0
        max_ind = np.argmax([ss.shape[1] for ss in self.spec_slices])
        self.max_width = self.spec_slices[max_ind].shape[1]
        self.blank_spec = np.zeros(
            (self.spec_slices[0].shape[0], self.max_width)
        )

    def plot(self, fig_id):
        self.fig, self.ax = plt.subplots(
            nrows=1,
            ncols=2,
            num=fig_id,
            figsize=(self.fig_width, self.fig_height),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        plt.tight_layout()

        # plot 2D TNSE features
        self.low_dim_plt = self.ax[0].scatter(
            self.feats_ds[:, 0],
            self.feats_ds[:, 1],
            c=self.labels_cols,
            s=self.pt_size,
            picker=5,
        )
        self.ax[0].set_title("TSNE of Call Features")
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])

        # plot clip from spectrogram
        spec_min_max = (
            0,
            self.blank_spec.shape[1],
            self.freq_lims[0],
            self.freq_lims[1],
        )
        self.ax[1].imshow(
            self.blank_spec, extent=spec_min_max, cmap="plasma", aspect="auto"
        )
        self.spec_im = self.ax[1].get_images()[0]
        self.ax[1].set_title("Spectrogram")
        self.ax[1].grid(color="w", linewidth=0.5)
        self.ax[1].set_xticks([])
        self.ax[1].set_ylabel("kHz")

        bbox_orig = patches.Rectangle(
            (0, 0), 0, 0, edgecolor="w", linewidth=0, fill=False
        )
        self.ax[1].add_patch(bbox_orig)

        self.annot = self.ax[0].annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)

        self.fig.canvas.mpl_connect("motion_notify_event", self.mouse_hover)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)

    def mouse_hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax[0]:
            cont, ind = self.low_dim_plt.contains(event)
            if cont:
                self.current_id = ind["ind"][0]

                # copy spec into full window - probably a better way of doing this
                new_spec = self.blank_spec.copy()
                w_diff = (
                    self.blank_spec.shape[1]
                    - self.spec_slices[self.current_id].shape[1]
                ) // 2
                new_spec[
                    :,
                    w_diff : self.spec_slices[self.current_id].shape[1]
                    + w_diff,
                ] = self.spec_slices[self.current_id]
                self.spec_im.set_data(new_spec)
                self.spec_im.set_clim(vmin=0, vmax=new_spec.max())

                # draw bounding box around call
                self.ax[1].patches[0].remove()
                spec_width_orig = self.spec_slices[self.current_id].shape[1] / (
                    1.0 + 2.0 * self.spec_pad
                )
                xx = w_diff + self.spec_pad * spec_width_orig
                ww = spec_width_orig
                yy = self.call_info[self.current_id]["low_freq"] / 1000
                hh = (
                    self.call_info[self.current_id]["high_freq"]
                    - self.call_info[self.current_id]["low_freq"]
                ) / 1000
                bbox = patches.Rectangle(
                    (xx, yy), ww, hh, edgecolor="r", linewidth=0.5, fill=False
                )
                self.ax[1].add_patch(bbox)

                # update annotation arrow
                pos = self.low_dim_plt.get_offsets()[self.current_id]
                self.annot.xy = pos
                self.annot.set_visible(True)

                # write call info
                info_str = (
                    self.call_info[self.current_id]["file_name"]
                    + ", time="
                    + str(
                        round(self.call_info[self.current_id]["start_time"], 3)
                    )
                    + ", prob="
                    + str(round(self.call_info[self.current_id]["det_prob"], 3))
                )
                self.ax[0].set_xlabel(info_str)

                # redraw
                self.fig.canvas.draw_idle()

    def key_press(self, event):
        if event.key.isdigit():
            self.labels_cols[self.current_id] = colors[int(event.key)]
            self.labels[self.current_id] = int(event.key)
            self.annotated[self.current_id] = 1
        elif event.key == "enter" and self.allow_training:
            self.train_classifier()
        elif event.key == "x" and self.allow_training:
            self.get_classifier_params()

        self.ax[0].scatter(
            self.feats_ds[:, 0],
            self.feats_ds[:, 1],
            c=self.labels_cols,
            s=self.pt_size,
        )
        self.fig.canvas.draw_idle()

    def train_classifier(self):
        # TODO maybe it's better to classify in 2D space - but then can't be linear ...
        inds = np.where(self.annotated == 1)[0]
        labs_un, labs_inds = np.unique(self.labels[inds], return_inverse=True)

        if labs_un.shape[0] > 1:  # needs at least 2 classes
            self.clf = LinearSVC(
                C=1.0,
                penalty="l2",
                loss="squared_hinge",
                tol=0.0001,
                intercept_scaling=1.0,
                max_iter=2000,
            )

            self.clf.fit(self.feats[inds, :], self.labels[inds])

            # update labels
            inds_unlab = np.where(self.annotated == 0)[0]
            self.labels[inds_unlab] = self.clf.predict(self.feats[inds_unlab])
            for ii in inds_unlab:
                self.labels_cols[ii] = colors[self.labels[ii]]
        else:
            print("Not enough data - please label more classes.")

    def get_classifier_params(self):
        res = {}
        if self.clf is None:
            print("Model not trained!")
        else:
            res["weights"] = self.clf.coef_.astype(np.float32)
            res["biases"] = self.clf.intercept_.astype(np.float32)
        return res
