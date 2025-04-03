import argparse
import os
import warnings
from typing import List, Optional

import torch
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR

import batdetect2.detector.parameters as parameters
import batdetect2.train.legacy.audio_dataloader as adl
import batdetect2.train.legacy.train_model as tm
import batdetect2.train.legacy.train_utils as tu
import batdetect2.train.losses as losses
import batdetect2.utils.detector_utils as du
import batdetect2.utils.plot_utils as pu
from batdetect2 import types
from batdetect2.detector.models import Net2DFast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_arugments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        type=str,
        help="Input directory for audio",
    )
    parser.add_argument(
        "train_ann_path",
        type=str,
        help="Path to where train annotation file is stored",
    )
    parser.add_argument(
        "test_ann_path",
        type=str,
        help="Path to where test annotation file is stored",
    )
    parser.add_argument(
        "model_path", type=str, help="Path to pretrained model"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=os.path.join(BASE_DIR, "experiments"),
        help="Path to where experiment files are stored",
    )
    parser.add_argument(
        "--op_model_name",
        type=str,
        default="",
        help="Path and name for finetuned model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        dest="num_epochs",
        help="Number of finetuning epochs",
    )
    parser.add_argument(
        "--finetune_only_last_layer",
        action="store_true",
        help="Only train final layers",
    )
    parser.add_argument(
        "--train_from_scratch",
        action="store_true",
        help="Do not use pretrained weights",
    )
    parser.add_argument(
        "--do_not_save_images",
        action="store_false",
        help="Do not save images at the end of training",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="Notes to save in text file"
    )
    args = parser.parse_args()
    return args


def select_device(warn=True) -> str:
    if torch.cuda.is_available():
        return "cuda"

    if warn:
        warnings.warn(
            "No GPU available, using the CPU instead. Please consider using a GPU "
            "to speed up training."
        )

    return "cpu"


def load_annotations(
    dataset_name: str,
    ann_path: str,
    audio_path: str,
    classes_to_ignore: Optional[List[str]] = None,
    events_of_interest: Optional[List[str]] = None,
) -> List[types.FileAnnotation]:
    train_sets: List[types.DatasetDict] = []
    train_sets.append(
        tu.get_blank_dataset_dict(
            dataset_name,
            is_test=False,
            ann_path=ann_path,
            wav_path=audio_path,
        )
    )

    return tu.load_set_of_anns(
        train_sets,
        events_of_interest=events_of_interest,
        classes_to_ignore=classes_to_ignore,
    )


def finetune_model(
    model: types.DetectionModel,
    data_train: List[types.FileAnnotation],
    data_test: List[types.FileAnnotation],
    params: parameters.TrainingParameters,
    model_params: types.ModelParameters,
    finetune_only_last_layer: bool = False,
    save_images: bool = True,
):
    # train loader
    train_dataset = adl.AudioLoader(data_train, params, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    # test loader - batch size of one because of variable file length
    test_dataset = adl.AudioLoader(data_test, params, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    inputs_train = next(iter(train_loader))
    params.ip_height = inputs_train["spec"].shape[2]
    print("\ntrain batch size :", inputs_train["spec"].shape)

    # Check that the model is the same as the one used to train the pretrained
    # weights
    assert model_params["model_name"] == "Net2DFast"
    assert isinstance(model, Net2DFast)
    print(
        "\n\nSOME hyperparams need to be the same as the loaded model "
        "(e.g. FFT) - currently they are getting overwritten.\n\n"
    )

    # set the number of output classes
    num_filts = model.conv_classes_op.in_channels
    (k_size,) = model.conv_classes_op.kernel_size
    (pad,) = model.conv_classes_op.padding
    model.conv_classes_op = torch.nn.Conv2d(
        num_filts,
        len(params.class_names) + 1,
        kernel_size=k_size,
        padding=pad,
    )
    model.conv_classes_op.to(params.device)

    if finetune_only_last_layer:
        print("\nOnly finetuning the final layers.\n")
        train_layers_i = [
            "conv_classes",
            "conv_classes_op",
            "conv_size",
            "conv_size_op",
        ]
        train_layers = [tt + ".weight" for tt in train_layers_i] + [
            tt + ".bias" for tt in train_layers_i
        ]
        for name, param in model.named_parameters():
            if name in train_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        params.num_epochs * len(train_loader),
    )

    if params.train_loss == "mse":
        det_criterion = losses.mse_loss
    elif params.train_loss == "focal":
        det_criterion = losses.focal_loss
    else:
        raise ValueError("Unknown loss function")

    # plotting
    train_plt_ls = pu.LossPlotter(
        params.experiment / "train_loss.png",
        params.num_epochs + 1,
        ["train_loss"],
        None,
        None,
        ["epoch", "train_loss"],
        logy=True,
    )
    test_plt_ls = pu.LossPlotter(
        params.experiment / "test_loss.png",
        params.num_epochs + 1,
        ["test_loss"],
        None,
        None,
        ["epoch", "test_loss"],
        logy=True,
    )
    test_plt = pu.LossPlotter(
        params.experiment / "test.png",
        params.num_epochs + 1,
        ["avg_prec", "rec_at_x", "avg_prec_class", "file_acc", "top_class"],
        [0, 1],
        None,
        ["epoch", ""],
    )
    test_plt_class = pu.LossPlotter(
        params.experiment / "test_avg_prec.png",
        params.num_epochs + 1,
        params.class_names_short,
        [0, 1],
        params.class_names_short,
        ["epoch", "avg_prec"],
    )

    # main train loop
    for epoch in range(0, params.num_epochs + 1):
        train_loss = tm.train(
            model,
            epoch,
            train_loader,
            det_criterion,
            optimizer,
            scheduler,
            params,
        )
        train_plt_ls.update_and_save(epoch, [train_loss["train_loss"]])

        if epoch % params.num_eval_epochs == 0:
            # detection accuracy on test set
            test_res, test_loss = tm.test(
                model,
                epoch,
                test_loader,
                det_criterion,
                params,
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
                params.experiment, "test_pr", "test_pr", test_res
            )

            # save finetuned model
            print(f"saving model to: {params.model_file_name}")
            op_state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "params": params,
            }
            torch.save(op_state, params.model_file_name)

    # save an image with associated prediction for each batch in the test set
    if save_images:
        tm.save_images_batch(model, test_loader, params)


def main():
    info_str = "\nBatDetect - Finetune Model\n"
    print(info_str)

    args = parse_arugments()

    # Load experiment parameters
    params = parameters.get_params(
        make_dirs=True,
        exps_dir=args.experiment_dir,
        device=select_device(),
        num_epochs=args.num_epochs,
        notes=args.notes,
    )

    print("\nAudio directory:      " + args.audio_path)
    print("Train file:           " + args.train_ann_path)
    print("Test file:            " + args.test_ann_path)
    print("Loading model:        " + args.model_path)

    if args.train_from_scratch:
        print(
            "\nTraining model from scratch i.e. not using pretrained weights"
        )

    model, model_params = du.load_model(
        args.model_path,
        load_weights=not args.train_from_scratch,
        device=params.device,
    )

    if args.op_model_name != "":
        params.model_file_name = args.op_model_name

    classes_to_ignore = params.classes_to_ignore + params.generic_class

    # save notes file
    if params.notes:
        tu.write_notes_file(
            params.experiment / "notes.txt",
            args.notes,
        )

    # NOTE:??
    dataset_name = (
        os.path.basename(args.train_ann_path)
        .replace(".json", "")
        .replace("_TRAIN", "")
    )

    # ==== LOAD DATA ====

    # load train annotations
    data_train = load_annotations(
        dataset_name,
        args.train_ann_path,
        args.audio_path,
        params.events_of_interest,
    )
    print("\nTrain set:")
    print("Number of files", len(data_train))

    # load test annotations
    data_test = load_annotations(
        dataset_name,
        args.test_ann_path,
        args.audio_path,
        classes_to_ignore,
        params.events_of_interest,
    )
    print("\nTrain set:")
    print("Number of files", len(data_train))

    finetune_model(
        model,
        data_train,
        data_test,
        params,
        model_params,
        finetune_only_last_layer=args.finetune_only_last_layer,
        save_images=args.do_not_save_images,
    )


if __name__ == "__main__":
    main()
