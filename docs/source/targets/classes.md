# Step 4: Defining Target Classes for Training

## Purpose and Context

You've prepared your data by defining your annotation vocabulary (Step 1: Terms), removing irrelevant sounds (Step 2: Filtering), and potentially cleaning up or modifying tags (Step 3: Transforming Tags).
Now, it's time to tell `batdetect2` **exactly what categories (classes) your model should learn to identify**.

This step involves defining rules that map the final tags on your sound event annotations to specific **class names** (like `pippip`, `myodau`, or `noise`).
These class names are the labels the machine learning model will be trained to predict.
Getting this definition right is essential for successful model training.

## How it Works: Defining Classes with Rules

You define your target classes in your main configuration file (e.g., your `.yaml` training config), typically under a section named `classes`.
This section contains a **list** of class definitions.
Each item in the list defines one specific class your model should learn.

## Defining a Single Class

Each class definition rule requires a few key pieces of information:

1.  `name`: **(Required)** This is the unique, simple name you want to give this class (e.g., `pipistrellus_pipistrellus`, `myotis_daubentonii`, `echolocation_noise`).
    This is the label the model will actually use.
    Choose names that are clear and distinct.
    **Each class name must be unique.**
2.  `tags`: **(Required)** This is a list containing one or more specific tags that identify annotations belonging to this class.
    Remember, each tag is specified using its term `key` (like `species` or `sound_type`, defaulting to `class` if omitted) and its specific `value` (like `Pipistrellus pipistrellus` or `Echolocation`).
3.  `match_type`: **(Optional, defaults to `"all"`)** This tells the system how to use the list of tags you provided in the `tag` field:
    - `"all"`: An annotation must have **ALL** of the tags listed in the `tags` section to be considered part of this class.
      (This is the default if you don't specify `match_type`).
    - `"any"`: An annotation only needs to have **AT LEAST ONE** of the tags listed in the `tags` section to be considered part of this class.

**Example: Defining two specific bat species classes**

```yaml
# In your main configuration file
classes:
  # Definition for the first class
  - name: pippip # Simple name for Pipistrellus pipistrellus
    tags:
      - key: species # Term key (could also default to 'class')
        value: Pipistrellus pipistrellus # Specific tag value
    # match_type defaults to "all" (which is fine for a single tag)

  # Definition for the second class
  - name: myodau # Simple name for Myotis daubentonii
    tags:
      - key: species
        value: Myotis daubentonii
```

**Example: Defining a class requiring multiple conditions (`match_type: "all"`)**

```yaml
classes:
  - name: high_quality_pippip # Name for high-quality P. pip calls
    match_type: all # Annotation must match BOTH tags below
    tags:
      - key: species
        value: Pipistrellus pipistrellus
      - key: quality # Assumes 'quality' term key exists
        value: Good
```

**Example: Defining a class matching multiple alternative tags (`match_type: "any"`)**

```yaml
classes:
  - name: pipistrelle # Name for any Pipistrellus species in this list
    match_type: any # Annotation must match AT LEAST ONE tag below
    tags:
      - key: species
        value: Pipistrellus pipistrellus
      - key: species
        value: Pipistrellus pygmaeus
      - key: species
        value: Pipistrellus nathusii
```

## Handling Overlap: Priority Order Matters

Sometimes, an annotation might have tags that match the rules for _more than one_ class definition.
For example, an annotation tagged `species: Pipistrellus pipistrellus` would match both a specific `'pippip'` class rule and a broader `'pipistrelle'` genus rule (like the examples above) if both were defined.

How does `batdetect2` decide which class name to assign? It uses the **order of the class definitions in your configuration list**.

- The system checks an annotation against your class rules one by one, starting from the **top** of the `classes` list and moving down.
- As soon as it finds a rule that the annotation matches, it assigns that rule's `name` to the annotation and **stops checking** further rules for that annotation.
- **The first match wins!**

Therefore, you should generally place your **most specific rules before more general rules** if you want the specific category to take precedence.

**Example: Prioritizing Species over Noise**

```yaml
classes:
  # --- Specific Species Rules (Checked First) ---
  - name: pippip
    tags:
      - key: species
        value: Pipistrellus pipistrellus

  - name: myodau
    tags:
      - key: species
        value: Myotis daubentonii

  # --- General Noise Rule (Checked Last) ---
  - name: noise # Catch-all for anything tagged as Noise
    match_type: any # Match if any noise tag is present
    tags:
      - key: sound_type # Assume 'sound_type' term key exists
        value: Noise
      - key: quality # Assume 'quality' term key exists
        value: Low # Maybe low quality is also considered noise for training
```

In this example, an annotation tagged with `species: Myotis daubentonii` _and_ `quality: Low` would be assigned the class name `myodau` because that rule comes first in the list.
It would not be assigned `noise`, even though it also matches the second condition of the noise rule.

Okay, that's a very important clarification about how BatDetect2 handles sounds that don't match specific class definitions.
Let's refine that section to accurately reflect this behavior.

## What if No Class Matches?

It's important to understand what happens if a sound event annotation passes through the filtering (Step 2) and transformation (Step 3) steps, but its final set of tags doesn't match _any_ of the specific class definitions you've listed in this section.

These annotations are **not ignored** during training.
Instead, they are typically assigned to a **generic "relevant sound" class**.
Think of this as a category for sounds that you considered important enough to keep after filtering, but which don't fit into one of your specific target classes for detailed classification (like a particular species).
This generic class is distinct from background noise.

In BatDetect2, this default generic class is often referred to as the **"Bat"** class.
The goal is generally that all relevant bat echolocation calls that pass the initial filtering should fall into _either_ one of your specific defined classes (like `pippip` or `myodau`) _or_ this generic "Bat" class.

**In summary:**

- Sounds passing **filtering** are considered relevant.
- If a relevant sound matches one of your **specific class rules** (in priority order), it gets that specific class label.
- If a relevant sound does **not** match any specific class rule, it gets the **generic "Bat" class** label.

**Crucially:** If you want certain types of sounds (even if they are bat calls) to be **completely excluded** from the training process altogether (not even included in the generic "Bat" class), you **must remove them using rules in the Filtering step (Step 2)**.
Any sound annotation that makes it past filtering _will_ be used in training, either under one of your specific classes or the generic one.

## Outcome

By defining this list of prioritized class rules, you provide `batdetect2` with a clear procedure to assign a specific target label (your class `name`) to each relevant sound event annotation based on its tags.
This labelled data is exactly what the model needs for training (Step 5).
