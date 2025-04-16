# Defining Training Targets

A crucial aspect of training any supervised machine learning model, including BatDetect2, is clearly defining the **training targets**.
This process determines precisely what the model should learn to detect and recognize from the input data (in this case, spectrograms).
The choices made here directly influence the model's focus, its performance, and how its predictions should be interpreted.

For BatDetect2, defining targets involves specifying:

- Which sounds in your annotated dataset are relevant for training.
- How these sounds should be categorized into distinct **classes** (e.g., different species, types of calls, noise categories).
- How the model's output (predicted class names) should be translated back into meaningful tags.

## Sound Event Annotations: The Starting Point

BatDetect2 assumes your training data consists of audio recordings where relevant sound events have been **annotated**.
A typical annotation for a single sound event provides two key pieces of information:

1.  **Location & Extent:** Information defining _where_ the sound occurs in time and frequency, usually represented as a **bounding box** drawn on a spectrogram.
2.  **Description (Tags):** Information _about_ the sound event, provided as a set of descriptive **tags**.

Tags are fundamental to how BatDetect2 understands annotations.
Each tag is a **key-value pair**, much like labels used in many data systems.
For example, a single echolocation pulse annotation might have tags like:

- `species: Myotis daubentonii`
- `quality: Good`
- `call_type: Echolocation`
- `verified_by: ExpertA`

A key aspect is that a single sound event can (and often does) have **multiple tags** associated with it, allowing for rich, detailed descriptions capturing various facets of information (taxonomy, signal quality, functional type, verification status, etc.).

While this detailed, multi-tag approach is powerful for data representation, standard classification models typically require a single target class label for each training example.
Therefore, the core task of the **target definition process** described in the following sections is to provide BatDetect2 with clear rules to:

- Interpret the meaning of different tag keys (**Terms**).
- Select only the relevant annotations (**Filtering**).
- Potentially standardize or modify the tags (**Transforming**).
- Ultimately map the rich set of tags on each selected annotation to a single, definitive **target class** label for training (**Classes**).

## Configuration-Driven Workflow

BatDetect2 is designed so that researchers can configure this entire target definition process primarily through **configuration files** (typically written in YAML format), minimizing the need for direct programming for standard workflows.
These settings are usually grouped under a main `targets:` key within your overall training configuration file.

## The Target Definition Steps

Defining the targets involves several sequential steps, each configurable and building upon the previous one:

1.  **Defining Vocabulary:** Understand how annotations use tags (key-value pairs like `species: Myotis daubentonii`).
    This first step involves defining the meaning (**Terms**) behind the keys used in your tags (e.g., establishing what `species` or `call_type` represents using standard or custom definitions).
    Often, the default vocabulary provided by BatDetect2 is sufficient, so you may not need to configure this step explicitly.
    However, reading through this section is encouraged to understand how tags are formally defined via keys, which is essential for using them effectively in subsequent steps like filtering and class definition.
2.  **Filtering Sound Events:** Select only the sound event annotations that are relevant for your specific training goal, based on their tags (e.g., keeping only high-quality echolocation calls).
3.  **Transforming Tags (Optional):** Modify the tags on the selected annotations.
    This is useful for standardizing inconsistent labels, correcting errors, grouping related concepts (like mapping species to genus), or deriving new information.
4.  **Defining Classes & Decoding Rules:** Map the final set of tags on each annotation to a specific target **class name** (like `pippip` or `myodau`) that the model will learn.
    This step also includes setting priorities for overlapping definitions and specifying how predicted class names should be translated back into tags (decoding).
5.  **The `Targets` Object:** Understand the outcome of this configuration process – a functional object used internally by BatDetect2 that encapsulates all your defined rules for filtering, transforming, encoding, and decoding.

The result of this entire configuration process is a clear set of instructions that BatDetect2 uses during training data preparation to determine the correct "answer" (the ground truth label or target) for each relevant sound event.

Explore the detailed steps using the links below:

```{toctree}
:maxdepth: 1
:caption: Contents:

tags_and_terms
filtering
transform
classes
labels
use
```
