# Legacy feature extraction outputs

The previous BatDetect2 workflow exposed several output concepts that users may still rely on.

These included:

- `cnn_feats`
- `spec_features`
- `spec_slices`

## Why this matters

Users exploring older notebooks or downstream analysis code often encounter these names first.

The current stack exposes a different surface centered on per-detection `features` plus configurable output formatters.

## Migration note

There is not always a strict one-to-one replacement.

When migrating, validate which part of the old workflow you actually need:

- low-level exported features,
- spectrogram slices,
- model-internal feature vectors,
- legacy JSON output shape.

Then map that need onto the current API and output format configuration.

## Related pages

- Migration guide: {doc}`migration-guide`
- Current features explanation: {doc}`../explanation/extracted-features-and-embeddings`
- Output formats reference: {doc}`../reference/output-formats`
