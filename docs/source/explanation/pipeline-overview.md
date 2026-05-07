# Pipeline overview

batdetect2 processes recordings as a sequence of modules. Each stage has a
clear role and configuration surface.

## End-to-end flow

1. Audio loading
2. Preprocessing (waveform -> spectrogram)
3. Detector forward pass
4. Postprocessing (peaks, decoding, thresholds)
5. Output formatting and export

## Why the modular design matters

The model, preprocessing, postprocessing, targets, and output formatting are
configured separately. That makes it easier to:

- swap components without rewriting the whole pipeline,
- keep experiments reproducible,
- adapt workflows to new datasets.

## Core objects in the stack

- `BatDetect2API` orchestrates training, inference, and evaluation workflows.
- `ModelConfig` defines architecture, preprocessing, postprocessing, and
  targets.
- `Targets` controls event filtering, class encoding/decoding, and ROI mapping.

## Related pages

- Preprocessing rationale: {doc}`preprocessing-consistency`
- Postprocessing rationale: {doc}`postprocessing-and-thresholds`
- Target rationale: {doc}`target-encoding-and-decoding`
