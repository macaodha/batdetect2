# Step 4: Defining Target Classes and Decoding Rules

## Purpose and Context

You've prepared your data by defining your annotation vocabulary (Step 1: Terms), removing irrelevant sounds (Step 2: Filtering), and potentially cleaning up or modifying tags (Step 3: Transforming Tags).
Now, it's time for a crucial step with two related goals:

1.  Telling `batdetect2` **exactly what categories (classes) your model should learn to identify** by defining rules that map annotation tags to class names (like `pippip`, `myodau`, or `noise`).
    This process is often called **encoding**.
2.  Defining how the model's predictions (those same class names) should be translated back into meaningful, structured **annotation tags** when you use the trained model.
    This is often called **decoding**.

These definitions are essential for both training the model correctly and interpreting its output later.

## How it Works: Defining Classes with Rules

You define your target classes and their corresponding decoding rules in your main configuration file (e.g., your `.yaml` training config), typically under a section named `classes`.
This section contains:

1.  A **list** of specific class definitions.
2.  A definition for the **generic class** tags.

Each item in the `classes` list defines one specific class your model should learn.

## Defining a Single Class

Each specific class definition rule requires the following information:

1.  `name`: **(Required)** This is the unique, simple name for this class (e.g., `pipistrellus_pipistrellus`, `myotis_daubentonii`, `noise`).
    This label is used during training and is what the model predicts.
    Choose clear, distinct names.
    **Each class name must be unique.**
2.  `tags`: **(Required)** This list contains one or more specific tags (using `key` and `value`) used to identify if an _existing_ annotation belongs to this class during the _encoding_ phase (preparing training data).
3.  `match_type`: **(Optional, defaults to `"all"`)** Determines how the `tags` list is evaluated during _encoding_:
    - `"all"`: The annotation must have **ALL** listed tags to match.
      (Default).
    - `"any"`: The annotation needs **AT LEAST ONE** listed tag to match.
4.  `output_tags`: **(Optional)** This list specifies the tags that should be assigned to an annotation when the model _predicts_ this class `name`.
    This is used during the _decoding_ phase (interpreting model output).
    - **If you omit `output_tags` (or set it to `null`/~), the system will default to using the same tags listed in the `tags` field for decoding.** This is often what you want.
    - Providing `output_tags` allows you to specify a different, potentially more canonical or detailed, set of tags to represent the class upon prediction.
      For example, you could match based on simplified tags but output standardized tags.

**Example: Defining Species Classes (Encoding & Default Decoding)**

Here, the `tags` used for matching during encoding will also be used for decoding, as `output_tags` is omitted.

```yaml
# In your main configuration file
classes:
  # Definition for the first class
  - name: pippip # Simple name for Pipistrellus pipistrellus
    tags: # Used for BOTH encoding match and decoding output
      - key: species
        value: Pipistrellus pipistrellus
    # match_type defaults to "all"
    # output_tags is omitted, defaults to using 'tags' above

  # Definition for the second class
  - name: myodau # Simple name for Myotis daubentonii
    tags: # Used for BOTH encoding match and decoding output
      - key: species
        value: Myotis daubentonii
```

**Example: Defining a Class with Separate Encoding and Decoding Tags**

Here, we match based on _either_ of two tags (`match_type: any`), but when the model predicts `'pipistrelle'`, we decode it _only_ to the specific `Pipistrellus pipistrellus` tag plus a genus tag.

```yaml
classes:
  - name: pipistrelle # Name for a Pipistrellus group
    match_type: any # Match if EITHER tag below is present during encoding
    tags:
      - key: species
        value: Pipistrellus pipistrellus
      - key: species
        value: Pipistrellus pygmaeus # Match pygmaeus too
    output_tags: # BUT, when decoding 'pipistrelle', assign THESE tags:
      - key: species
        value: Pipistrellus pipistrellus # Canonical species
      - key: genus # Assumes 'genus' key exists
        value: Pipistrellus # Add genus tag
```

## Handling Overlap During Encoding: Priority Order Matters

As before, when preparing training data (encoding), if an annotation matches the `tags` and `match_type` rules for multiple class definitions, the **order of the class definitions in the configuration list determines the priority**.

- The system checks rules from the **top** of the `classes` list down.
- The annotation gets assigned the `name` of the **first class rule it matches**.
- **Place more specific rules before more general rules.**

_(The YAML example for prioritizing Species over Noise remains the same as the previous version)_

## Handling Non-Matches & Decoding the Generic Class

What happens if an annotation passes filtering/transformation but doesn't match any specific class rule during encoding?

- **Encoding:** As explained previously, these annotations are **not ignored**.
  They are typically assigned to a generic "relevant sound" category, often called the **"Bat"** class in BatDetect2, intended for all relevant bat calls not specifically classified.
- **Decoding:** When the model predicts this generic "Bat" category (or when processing sounds that weren't assigned a specific class during encoding), we need a way to represent this generic status with tags.
  This is defined by the `generic_class` list directly within the main `classes` configuration section.

**Defining the Generic Class Tags:**

You specify the tags for the generic class like this:

```yaml
# In your main configuration file
classes: # Main configuration section for classes
  # --- List of specific class definitions ---
  classes:
    - name: pippip
      tags:
        - key: species
          value: Pipistrellus pipistrellus
    # ... other specific classes ...

  # --- Definition of the generic class tags ---
  generic_class: # Define tags for the generic 'Bat' category
    - key: call_type
      value: Echolocation
    - key: order
      value: Chiroptera
    # These tags will be assigned when decoding the generic category
```

This `generic_class` list provides the standard tags assigned when a sound is identified as relevant (passed filtering) but doesn't belong to one of the specific target classes you defined.
Like the specific classes, sensible defaults are often provided if you don't explicitly define `generic_class`.

**Crucially:** Remember, if sounds should be **completely excluded** from training (not even considered "generic"), use **Filtering rules (Step 2)**.

### Outcome

By defining this list of prioritized class rules (including their `name`, matching `tags`, `match_type`, and optional `output_tags`) and the `generic_class` tags, you provide `batdetect2` with:

1.  A clear procedure to assign a target label (`name`) to each relevant annotation for training.
2.  A clear mapping to convert predicted class names (including the generic case) back into meaningful annotation tags.

This complete definition prepares your data for the final heatmap generation (Step 5) and enables interpretation of the model's results.
