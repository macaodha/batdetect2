# FAQ

## Installation and setup

### Do I need Python knowledge to use batdetect2?

Not much. If you only want to run the model on your own recordings, you can
use the CLI and follow the steps in {doc}`getting_started`.

Some command-line familiarity helps, but you do not need to write Python code
for standard inference workflows.

### Are there plans for an R version?

Not currently. Output files are plain formats (for example CSV/JSON), so you
can read and analyze them in R or other environments.

### I cannot get installation working. What should I do?

First, re-check {doc}`getting_started` and confirm your environment is active.
If it still fails, open an issue with your OS, install method, and full error
output: [GitHub Issues](https://github.com/macaodha/batdetect2/issues).

## Model behavior and performance

### The model does not perform well on my data

This usually means your data distribution differs from training data. The best
next step is to validate on reviewed local data and then fine-tune/train on
your own annotations if needed.

### The model confuses insects/noise with bats

This can happen, especially when recording conditions differ from training
conditions. Threshold tuning and training with local annotations can improve
results.

See {doc}`how_to/tune-detection-threshold`.

### The model struggles with feeding buzzes or social calls

This is a known limitation of available training data in some settings. If you
have high-quality annotated examples, they are valuable for improving models.

### Calls in the same sequence are predicted as different species

batdetect2 returns per-call probabilities and does not apply heavy sequence-
level smoothing by default. You can apply sequence-aware postprocessing in your
own analysis workflow.

### Can I trust model outputs for biodiversity conclusions?

Use caution. Always validate model behavior on local, reviewed data before
using outputs for ecological inference or biodiversity assessment.

### The pipeline is slow

Runtime depends on hardware and recording duration. GPU inference is often much
faster than CPU. If files are very long, splitting them into shorter clips can
help throughput.

If you need a clipping workflow, see the annotation GUI repository:
[batdetect2_GUI](https://github.com/macaodha/batdetect2_GUI).

## Training and scope

### Can I train on my own species set?

Yes. You can train/fine-tune with your own annotated data and species labels.

### Does this work on frequency-division or zero-crossing recordings?

Not directly. The workflow assumes audio can be converted to spectrograms from
the raw waveform.

### Can this be used for non-bat bioacoustics (for example insects or birds)?

Potentially yes, but expect retraining and configuration changes. Open an issue
if you want guidance for a specific use case.

## Usage and licensing

### Can I use this for commercial purposes?

No. This project is currently for non-commercial use. See the repository
license for details.
