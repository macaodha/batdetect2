# Training config reference

`TrainingConfig` controls the training loop, optimisation, data loading, losses,
and validation tasks.

Defined in `batdetect2.train.config`.

## Top-level fields

- `train_loader`
  - training data loading and clipping settings.
- `val_loader`
  - validation data loading and clipping settings.
- `optimizer`
  - optimiser type and learning rate settings.
- `scheduler`
  - learning-rate schedule settings.
- `loss`
  - detection, classification, and size loss settings.
- `trainer`
  - PyTorch Lightning trainer settings such as `max_epochs`.
- `labels`
  - target label generation settings.
- `validation`
  - evaluation tasks used during validation.
- `checkpoints`
  - checkpoint saving settings.

## What this config controls

Use `TrainingConfig` when you want to change things like:

- batch size,
- augmentation,
- optimiser and scheduler settings,
- number of epochs,
- validation frequency,
- checkpoint behaviour.

Example files live under `example_data/configs/`, including
`example_data/configs/training.yaml`.

## Related pages

- Evaluation config:
  {doc}`evaluation-config`
- Train command reference:
  {doc}`cli/train`
- Fine-tune from a checkpoint:
  {doc}`../how_to/fine-tune-from-a-checkpoint`
