from typing import NamedTuple, Optional

import torch
from soundevent import data
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from batdetect2.models.types import DetectionModel
from batdetect2.train.dataset import LabeledDataset


class TrainInputs(NamedTuple):
    spec: torch.Tensor
    detection_heatmap: torch.Tensor
    class_heatmap: torch.Tensor
    size_heatmap: torch.Tensor


def train_loop(
    model: DetectionModel,
    train_dataset: LabeledDataset[TrainInputs],
    validation_dataset: LabeledDataset[TrainInputs],
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        num_epochs * len(train_loader),
    )

    for epoch in range(num_epochs):
        train_loss = train_single_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scheduler,
        )


def train_single_epoch(
    model: DetectionModel,
    train_loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    scheduler: CosineAnnealingLR,
):
    model.train()
    train_loss = tu.AverageMeter()

    for batch in train_loader:
        optimizer.zero_grad()

        spec = batch.spec.to(device)
        detection_heatmap = batch.detection_heatmap.to(device)
        class_heatmap = batch.class_heatmap.to(device)
        size_heatmap = batch.size_heatmap.to(device)

        outputs = model(spec)

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
