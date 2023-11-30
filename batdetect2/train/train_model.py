import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import batdetect2.detector.post_process as pp
import batdetect2.train.audio_dataloader as adl
import batdetect2.train.evaluate as evl
import batdetect2.train.train_split as ts
import batdetect2.train.train_utils as tu
import batdetect2.utils.plot_utils as pu
from batdetect2.detector import models, parameters
from batdetect2.train import losses

warnings.filterwarnings("ignore", category=UserWarning)


def save_images_batch(model, data_loader, params):
    print("\nsaving images ...")

    is_train_state = data_loader.dataset.is_train
    data_loader.dataset.is_train = False
    data_loader.dataset.return_spec_for_viz = True
    model.eval()

    ind = 0  # first image in each batch
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            data = inputs["spec"].to(params["device"])
            outputs = model(data)

            spec_viz = inputs["spec_for_viz"].data.cpu().numpy()
            orig_index = inputs["file_id"][ind]
            plot_title = data_loader.dataset.data_anns[orig_index]["id"]
            op_file_name = (
                params["op_im_dir_test"]
                + data_loader.dataset.data_anns[orig_index]["id"]
                + ".jpg"
            )
            save_image(
                spec_viz,
                outputs,
                ind,
                inputs,
                params,
                op_file_name,
                plot_title,
            )

    data_loader.dataset.is_train = is_train_state
    data_loader.dataset.return_spec_for_viz = False


def save_image(
    spec_viz, outputs, ind, inputs, params, op_file_name, plot_title
):
    pred_nms, _ = pp.run_nms(outputs, params, inputs["sampling_rate"].float())
    pred_hm = outputs["pred_det"][ind, 0, :].data.cpu().numpy()
    spec_viz = spec_viz[ind, 0, :]
    gt = parse_gt_data(inputs)[ind]
    sampling_rate = inputs["sampling_rate"][ind].item()
    duration = inputs["duration"][ind].item()

    pu.plot_spec(
        spec_viz,
        sampling_rate,
        duration,
        gt,
        pred_nms[ind],
        params,
        plot_title,
        op_file_name,
        pred_hm,
        plot_boxes=True,
        fixed_aspect=False,
    )


def loss_fun(
    outputs, gt_det, gt_size, gt_class, det_criterion, params, class_inv_freq
):
    # detection loss
    loss = params["det_loss_weight"] * det_criterion(
        outputs["pred_det"], gt_det
    )

    # bounding box size loss
    loss += params["size_loss_weight"] * losses.bbox_size_loss(
        outputs["pred_size"], gt_size
    )

    # classification loss
    valid_mask = (gt_class[:, :-1, :, :].sum(1) > 0).float().unsqueeze(1)
    p_class = outputs["pred_class"][:, :-1, :]
    loss += params["class_loss_weight"] * det_criterion(
        p_class, gt_class[:, :-1, :], valid_mask=valid_mask
    )

    return loss


def train(
    model, epoch, data_loader, det_criterion, optimizer, scheduler, params
):
    model.train()

    train_loss = tu.AverageMeter()
    class_inv_freq = torch.from_numpy(
        np.array(params["class_inv_freq"], dtype=np.float32)
    ).to(params["device"])
    class_inv_freq = class_inv_freq.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    print("\nEpoch", epoch)
    for batch_idx, inputs in enumerate(data_loader):
        data = inputs["spec"].to(params["device"])
        gt_det = inputs["y_2d_det"].to(params["device"])
        gt_size = inputs["y_2d_size"].to(params["device"])
        gt_class = inputs["y_2d_classes"].to(params["device"])

        optimizer.zero_grad()
        outputs = model(data)

        loss = loss_fun(
            outputs,
            gt_det,
            gt_size,
            gt_class,
            det_criterion,
            params,
            class_inv_freq,
        )

        train_loss.update(loss.item(), data.shape[0])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0 and batch_idx != 0:
            print(
                "[{}/{}]\tLoss: {:.4f}".format(
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    train_loss.avg,
                )
            )

    print("Train loss          : {:.4f}".format(train_loss.avg))

    res = {}
    res["train_loss"] = float(train_loss.avg)
    return res


def test(model, epoch, data_loader, det_criterion, params):
    model.eval()
    predictions = []
    ground_truths = []
    test_loss = tu.AverageMeter()

    class_inv_freq = torch.from_numpy(
        np.array(params["class_inv_freq"], dtype=np.float32)
    ).to(params["device"])
    class_inv_freq = class_inv_freq.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            data = inputs["spec"].to(params["device"])
            gt_det = inputs["y_2d_det"].to(params["device"])
            gt_size = inputs["y_2d_size"].to(params["device"])
            gt_class = inputs["y_2d_classes"].to(params["device"])

            outputs = model(data)

            # if the model needs a fixed sized intput run this
            # data = torch.cat(torch.split(data, int(params['spec_train_width']*params['resize_factor']), 3), 0)
            # outputs = model(data)
            # for kk in ['pred_det', 'pred_size', 'pred_class']:
            #     outputs[kk] = torch.cat([oo for oo in outputs[kk]], 2).unsqueeze(0)

            if params["save_test_image_during_train"] and batch_idx == 0:
                # for visualization - save the first prediction
                ind = 0
                orig_index = inputs["file_id"][ind]
                plot_title = data_loader.dataset.data_anns[orig_index]["id"]
                op_file_name = (
                    params["op_im_dir"]
                    + str(orig_index.item()).zfill(4)
                    + "_"
                    + str(epoch).zfill(4)
                    + "_pred.jpg"
                )
                save_image(
                    data,
                    outputs,
                    ind,
                    inputs,
                    params,
                    op_file_name,
                    plot_title,
                )

            loss = loss_fun(
                outputs,
                gt_det,
                gt_size,
                gt_class,
                det_criterion,
                params,
                class_inv_freq,
            )
            test_loss.update(loss.item(), data.shape[0])

            # do NMS
            pred_nms, _ = pp.run_nms(
                outputs, params, inputs["sampling_rate"].float()
            )
            predictions.extend(pred_nms)

            ground_truths.extend(parse_gt_data(inputs))

    res_det = evl.evaluate_predictions(
        ground_truths,
        predictions,
        params["class_names"],
        params["detection_overlap"],
        params["ignore_start_end"],
    )

    print("\nTest loss          : {:.4f}".format(test_loss.avg))
    print("Rec at 0.95  (det) : {:.4f}".format(res_det["rec_at_x"]))
    print("Avg prec     (cls) : {:.4f}".format(res_det["avg_prec"]))
    print(
        "File acc     (cls) : {:.2f} - for {} out of {}".format(
            res_det["file_acc"],
            res_det["num_valid_files"],
            res_det["num_total_files"],
        )
    )
    print("Cls Avg prec (cls) : {:.4f}".format(res_det["avg_prec_class"]))

    print("\nPer class average precision")
    str_len = np.max([len(rs["name"]) for rs in res_det["class_pr"]]) + 5
    for cc, rs in enumerate(res_det["class_pr"]):
        if rs["num_gt"] > 0:
            print(
                str(cc).ljust(5)
                + rs["name"].ljust(str_len)
                + "{:.4f}".format(rs["avg_prec"])
            )

    res = {}
    res["test_loss"] = float(test_loss.avg)

    return res_det, res


def parse_gt_data(inputs):
    # reads the torch arrays into a dictionary of numpy arrays, taking care to
    # remove padding data i.e. not valid ones
    keys = [
        "start_times",
        "end_times",
        "low_freqs",
        "high_freqs",
        "class_ids",
        "individual_ids",
    ]
    batch_data = []
    for ind in range(inputs["start_times"].shape[0]):
        is_valid = inputs["is_valid"][ind] == 1
        gt = {}
        for kk in keys:
            gt[kk] = inputs[kk][ind][is_valid].cpu().numpy().astype(np.float32)
        gt["duration"] = inputs["duration"][ind].item()
        gt["file_id"] = inputs["file_id"][ind].item()
        gt["class_id_file"] = inputs["class_id_file"][ind].item()
        batch_data.append(gt)
    return batch_data


def select_model(params):
    num_classes = len(params["class_names"])
    if params["model_name"] == "Net2DFast":
        model = models.Net2DFast(
            params["num_filters"],
            num_classes=num_classes,
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoAttn":
        model = models.Net2DFastNoAttn(
            params["num_filters"],
            num_classes=num_classes,
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoCoordConv":
        model = models.Net2DFastNoCoordConv(
            params["num_filters"],
            num_classes=num_classes,
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    else:
        print("No valid network specified")
    return model


def main():
    plt.close("all")

    params = parameters.get_params(True)

    if torch.cuda.is_available():
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

    # setup arg parser and populate it with exiting parameters - will not work with lists
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to root of datasets")
    parser.add_argument(
        "ann_dir", type=str, help="Path to extracted annotations"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="diff",  # diff, same
        help="Which train split to use",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="Notes to save in text file"
    )
    parser.add_argument(
        "--do_not_save_images",
        action="store_false",
        help="Do not save images at the end of training",
    )
    parser.add_argument(
        "--standardize_classs_names_ip",
        type=str,
        default="Rhinolophus ferrumequinum;Rhinolophus hipposideros",
        help='Will set low and high frequency the same for these classes. Separate names with ";"',
    )
    for key, val in params.items():
        parser.add_argument("--" + key, type=type(val), default=val)
    params = vars(parser.parse_args())

    # save notes file
    if params["notes"] != "":
        tu.write_notes_file(params["experiment"] + "notes.txt", params["notes"])

    # load the training and test meta data - there are different splits defined
    train_sets, test_sets = ts.get_train_test_data(
        params["ann_dir"], params["data_dir"], params["train_split"]
    )
    train_sets_no_path, test_sets_no_path = ts.get_train_test_data(
        "", "", params["train_split"]
    )

    # keep track of what we have trained on
    params["train_sets"] = train_sets_no_path
    params["test_sets"] = test_sets_no_path

    # load train annotations - merge them all together
    print("\nTraining on:")
    for tt in train_sets:
        print(tt["ann_path"])
    classes_to_ignore = params["classes_to_ignore"] + params["generic_class"]
    (
        data_train,
        params["class_names"],
        params["class_inv_freq"],
    ) = tu.load_set_of_anns(
        train_sets,
        classes_to_ignore,
        params["events_of_interest"],
        params["convert_to_genus"],
    )
    params["genus_names"], params["genus_mapping"] = tu.get_genus_mapping(
        params["class_names"]
    )
    params["class_names_short"] = tu.get_short_class_names(
        params["class_names"]
    )

    # standardize the low and high frequency value for specified classes
    params["standardize_classs_names"] = params[
        "standardize_classs_names_ip"
    ].split(";")
    for cc in params["standardize_classs_names"]:
        if cc in params["class_names"]:
            data_train = tu.standardize_low_freq(data_train, cc)
        else:
            print(cc, "not found")

    # train loader
    train_dataset = adl.AudioLoader(data_train, params, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    # test set
    print("\nTesting on:")
    for tt in test_sets:
        print(tt["ann_path"])
    data_test, _, _ = tu.load_set_of_anns(
        test_sets,
        classes_to_ignore,
        params["events_of_interest"],
        params["convert_to_genus"],
    )
    data_train = tu.remove_dupes(data_train, data_test)
    test_dataset = adl.AudioLoader(data_test, params, is_train=False)
    # batch size of 1 because of variable file length
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    inputs_train = next(iter(train_loader))
    # TODO remove params['ip_height'], this is just legacy
    params["ip_height"] = int(params["spec_height"] * params["resize_factor"])
    print("\ntrain batch spec size :", inputs_train["spec"].shape)
    print("class target size     :", inputs_train["y_2d_classes"].shape)

    # select network
    model = select_model(params)
    model = model.to(params["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    scheduler = CosineAnnealingLR(
        optimizer, params["num_epochs"] * len(train_loader)
    )
    if params["train_loss"] == "mse":
        det_criterion = losses.mse_loss
    elif params["train_loss"] == "focal":
        det_criterion = losses.focal_loss

    # save parameters to file
    with open(params["experiment"] + "params.json", "w") as da:
        json.dump(params, da, indent=2, sort_keys=True)

    # plotting
    train_plt_ls = pu.LossPlotter(
        params["experiment"] + "train_loss.png",
        params["num_epochs"] + 1,
        ["train_loss"],
        None,
        None,
        ["epoch", "train_loss"],
        logy=True,
    )
    test_plt_ls = pu.LossPlotter(
        params["experiment"] + "test_loss.png",
        params["num_epochs"] + 1,
        ["test_loss"],
        None,
        None,
        ["epoch", "test_loss"],
        logy=True,
    )
    test_plt = pu.LossPlotter(
        params["experiment"] + "test.png",
        params["num_epochs"] + 1,
        ["avg_prec", "rec_at_x", "avg_prec_class", "file_acc", "top_class"],
        [0, 1],
        None,
        ["epoch", ""],
    )
    test_plt_class = pu.LossPlotter(
        params["experiment"] + "test_avg_prec.png",
        params["num_epochs"] + 1,
        params["class_names_short"],
        [0, 1],
        params["class_names_short"],
        ["epoch", "avg_prec"],
    )

    #
    # main train loop
    for epoch in range(0, params["num_epochs"] + 1):
        train_loss = train(
            model,
            epoch,
            train_loader,
            det_criterion,
            optimizer,
            scheduler,
            params,
        )
        train_plt_ls.update_and_save(epoch, [train_loss["train_loss"]])

        if epoch % params["num_eval_epochs"] == 0:
            # detection accuracy on test set
            test_res, test_loss = test(
                model, epoch, test_loader, det_criterion, params
            )
            test_plt_ls.update_and_save(epoch, [test_loss["test_loss"]])
            test_plt.update_and_save(
                epoch,
                [
                    test_res["avg_prec"],
                    test_res["rec_at_x"],
                    test_res["avg_prec_class"],
                    test_res["file_acc"],
                    test_res["top_class"]["avg_prec"],
                ],
            )
            test_plt_class.update_and_save(
                epoch, [rs["avg_prec"] for rs in test_res["class_pr"]]
            )
            pu.plot_pr_curve_class(
                params["experiment"], "test_pr", "test_pr", test_res
            )

            # save trained model
            print("saving model to: " + params["model_file_name"])
            op_state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                #'optimizer' : optimizer.state_dict(),
                "params": params,
            }
            torch.save(op_state, params["model_file_name"])

    # save an image with associated prediction for each batch in the test set
    # TODO: args variable does not exist
    # if not args["do_not_save_images"]:
    #     save_images_batch(model, test_loader, params)


if __name__ == "__main__":
    main()
