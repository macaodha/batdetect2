# FAQ

## Installation and setup

### Do I need Python knowledge to use batdetect2?

Not much.
If you only want to run the model on your own recordings, you can use the CLI
and follow the steps in {doc}`getting_started`.

Some command-line familiarity helps, but you do not need to write Python code
for standard inference workflows.

### Are there plans for an R version?

Not currently.
Output files are plain formats (for example CSV/JSON), so you can read and
analyze them in R or other environments.

### I cannot get installation working. What should I do?

First, re-check {doc}`getting_started` and confirm your environment is active.
If it still fails, open an issue with your OS, install method, and full error
output:
[GitHub Issues](https://github.com/macaodha/batdetect2/issues).

## Model behavior and performance

### The model does not perform well on my data

This usually means your data distribution differs from training data.
The best next step is to validate on reviewed local data and then
fine-tune/train on your own annotations if needed.

### The model confuses insects/noise with bats

This can happen, especially when recording conditions differ from training
conditions.
Threshold tuning and training with local annotations can improve results.

See {doc}`how_to/inference/tune-detection-threshold`.

### The model struggles with feeding buzzes or social calls

This is a known limitation of available training data in some settings.
If you have high-quality annotated examples, they are valuable for improving
models.

### Calls in the same sequence are predicted as different species

Currently we do not do any sophisticated post processing on the results output
by the model.
We return a probability associated with each species for each call.
You can use these predictions to clean up the noisy predictions for sequences of
calls.

### Can I trust model outputs for biodiversity conclusions?

The models developed and shared as part of this repository should be used with
caution.
While they have been evaluated on held out audio data, great care should be
taken when using the model outputs for any form of biodiversity assessment.
Your data may differ, and as a result it is very strongly recommended that you
validate the model first using data with known species to ensure that the
outputs can be trusted.

### The pipeline is slow

Runtime depends on hardware and recording duration.
GPU inference is often much faster than CPU.

## Training and scope

### Can I train on my own species set?

Yes.
You can train/fine-tune with your own annotated data and species labels.

### Does this work on frequency-division or zero-crossing recordings?

Not directly.
The workflow assumes audio can be converted to spectrograms from the raw
waveform.

### Can this be used for non-bat bioacoustics (for example insects or birds)?

Potentially yes, but expect retraining and configuration changes.
Open an issue if you want guidance for a specific use case.

## Usage and licensing

### Can I use this for commercial purposes?

No.
This project is currently for non-commercial use.
See the repository license for details.
