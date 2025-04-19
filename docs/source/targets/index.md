# Defining Training Targets

A crucial aspect of training any supervised machine learning model, including BatDetect2, is clearly defining the **training targets**.
This process determines precisely what the model should learn to detect, localize, classify, and characterize from the input data (in this case, spectrograms).
The choices made here directly influence the model's focus, its performance, and how its predictions should be interpreted.

For BatDetect2, defining targets involves specifying:

- Which sounds in your annotated dataset are relevant for training.
- How these sounds should be categorized into distinct **classes** (e.g., different species).
- How the geometric **Region of Interest (ROI)** (e.g., bounding box) of each sound maps to the specific **position** and **size** targets the model predicts.
- How these classes and geometric properties relate back to the detailed information stored in your annotation **tags** (using a consistent **vocabulary/terms**).
- How the model's output (predicted class names, positions, sizes) should be translated back into meaningful tags and geometries.

## Sound Event Annotations: The Starting Point

BatDetect2 assumes your training data consists of audio recordings where relevant sound events have been **annotated**.
A typical annotation for a single sound event provides two key pieces of information:

1.  **Location & Extent:** Information defining _where_ the sound occurs in time and frequency, usually represented as a **bounding box** (the ROI) drawn on a spectrogram.
2.  **Description (Tags):** Information _about_ the sound event, provided as a set of descriptive **tags** (key-value pairs).

For example, an annotation might have a bounding box and tags like:

- `species: Myotis daubentonii`
- `quality: Good`
- `call_type: Echolocation`

A single sound event can have **multiple tags**, allowing for rich descriptions.
This richness requires a structured process to translate the annotation (both tags and geometry) into the precise targets needed for model training.
The **target definition process** provides clear rules to:

- Interpret the meaning of different tag keys (**Terms**).
- Select only the relevant annotations (**Filtering**).
- Potentially standardize or modify the tags (**Transforming**).
- Map the geometric ROI to specific position and size targets (**ROI Mapping**).
- Map the final set of tags on each selected annotation to a single, definitive **target class** label (**Classes**).

## Configuration-Driven Workflow

BatDetect2 is designed so that researchers can configure this entire target definition process primarily through **configuration files** (typically written in YAML format), minimizing the need for direct programming for standard workflows.
These settings are usually grouped under a main `targets:` key within your overall training configuration file.

## The Target Definition Steps

Defining the targets involves several sequential steps, each configurable and building upon the previous one:

1.  **Defining Vocabulary (Terms & Tags):** Understand how annotations use tags (key-value pairs).
    This step involves defining the meaning (**Terms**) behind the tag keys (e.g., `species`, `call_type`).
    Often, default terms are sufficient, but understanding this is key to using tags in later steps.
    (See: {doc}`tags_and_terms`})
2.  **Filtering Sound Events:** Select only the relevant sound event annotations based on their tags (e.g., keeping only high-quality calls).
    (See: {doc}`filtering`})
3.  **Transforming Tags (Optional):** Modify tags on selected annotations for standardization, correction, grouping (e.g., species to genus), or deriving new tags.
    (See: {doc}`transform`})
4.  **Defining Classes & Decoding Rules:** Map the final tags to specific target **class names** (like `pippip` or `myodau`).
    Define priorities for overlap and specify how predicted names map back to tags (decoding).
    (See: {doc}`classes`})
5.  **Mapping ROIs (Position & Size):** Define how the geometric ROI (e.g., bounding box) of each sound event maps to the specific reference **point** (e.g., center, corner) and scaled **size** values (width, height) used as targets by the model.
    (See: {doc}`rois`})
6.  **The `Targets` Object:** Understand the outcome of configuring steps 1-5 – a functional object used internally by BatDetect2 that encapsulates all your defined rules for filtering, transforming, ROI mapping, encoding, and decoding.
    (See: {doc}`use`)

The result of this configuration process is a clear set of instructions that BatDetect2 uses during training data preparation to determine the correct "answer" (the ground truth label and geometry representation) for each relevant sound event.

Explore the detailed steps using the links below:

```{toctree}
:maxdepth: 1
:caption: Target Definition Steps:

tags_and_terms
filtering
transform
classes
rois
use
```
