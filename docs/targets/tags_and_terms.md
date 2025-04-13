## Managing Annotation Vocabulary: Terms and Tags

### Purpose

To train `batdetect2`, you will need sound events that have been carefully annotated. We annotate sound events using **tags**. A tag is simply a piece of information attached to an annotation, often describing what the sound is or its characteristics. Common examples include `Species: Myotis daubentonii` or `Quality: Good`.

Each tag fundamentally has two parts:

* **Value:** The specific information (e.g., "Myotis daubentonii", "Good").
* **Term:** The *type* of information (e.g., "Species", "Quality"). This defines the context or meaning of the value.

We use this flexible **Term: Value** approach because it allows you to annotate your data with any kind of information relevant to your project, while still providing a structure that makes the meaning clear.

While simple terms like "Species" are easy to understand, sometimes the underlying definition needs to be more precise to ensure everyone interprets it the same way (e.g., using a standard scientific definition for "Species" or clarifying what "Call Type" specifically refers to).

This `terms` module is designed to help manage these definitions effectively:

1.  It provides **standard definitions** for common terms used in bioacoustics, ensuring consistency.
2.  It lets you **define your own custom terms** if you need concepts specific to your project.
3.  Crucially, it allows you to use simple **"keys"** (like shortcuts) in your configuration files to refer to these potentially complex term definitions, making configuration much easier and less error-prone.

### The Problem: Why We Need Defined Terms (and Why It Gets Complicated)

Imagine you have a tag that simply says `"Myomyo"`.
If you created this tag, you might know it's a shortcut for the species _Myotis myotis_.
But what about someone else using your data or model later? Does `"Myomyo"` refer to the species? Or maybe it's the name of an individual bat, or even the location where it was recorded? Simple tags like this can be ambiguous.

To make things clearer, it's good practice to provide context.
We can do this by pairing the specific information (the **Value**) with the _type_ of information (the **Term**).
For example, writing the tag as `species: Myomyo` is much less ambiguous.
Here, `species` is the **Term**, explaining that `Myomyo` is a **Value** representing a species.

However, another challenge often comes up when sharing data or collaborating.
You might use the term `species`, while a colleague uses `Species`, and someone else uses the more formal `Scientific Name`.
Even though you all mean the same thing, these inconsistencies make it hard to combine data or reuse analysis pipelines automatically.

This is where standardized **Terms** become very helpful.
Several groups work to create standard definitions for common concepts.
For instance, the Darwin Core standard provides widely accepted terms for biological data, like `dwc:scientificName` for a species name.
Using standard Terms whenever possible makes your data clearer, easier for others (and machines!) to understand correctly, and much more reusable across different projects.

**But here's the practical problem:** While using standard, well-defined Terms is important for clarity and reusability, writing out full definitions or long standard names (like `dwc:scientificName` or "Scientific Name according to Darwin Core standard") every single time you need to refer to a species tag in a configuration file would be extremely tedious and prone to typing errors.

### The Solution: Keys (Shortcuts) and the Registry

This module uses a central **Registry** that stores the full definitions of various Terms.
Each Term in the registry is assigned a unique, short **key** (a simple string).

Think of the **key** as shortcut.

Instead of using the full Term definition in your configuration files, you just use its **key**.
The system automatically looks up the full definition in the registry using the key when needed.

**Example:**

- **Full Term Definition:** Represents the scientific name of the organism.
- **Key:** `species`
- **In Config:** You just write `species`.

### Available Keys

The registry comes pre-loaded with keys for many standard terms used in bioacoustics, including those from the `soundevent` package and some specific to `batdetect2`. This means you can often use these common concepts without needing to define them yourself.

Common examples of pre-defined keys might include:

* `species`: For scientific species names (e.g., *Myotis daubentonii*).
* `common_name`: For the common name of a species (e.g., "Daubenton's bat").
* `genus`, `family`, `order`: For higher levels of biological taxonomy.
* `call_type`: For functional call types (e.g., 'Echolocation', 'Social').
* `individual`: For identifying specific individuals if tracked.
* `class`: **(Special Key)** This key is often used **by default** in configurations when defining the target classes for your model (e.g., the different species you want the model to classify). If you are specifying a tag that represents a target class label, you often only need to provide the `value`, and the system assumes the `key` is `class`.

This is not an exhaustive list. To discover all the term keys currently available in the registry (including any standard ones loaded automatically and any custom ones you've added in your configuration), you can:

1.  Use the function `batdetect2.terms.get_term_keys()` if you are working directly with Python code.
2.  Refer to the main `batdetect2` API documentation for a list of commonly included standard terms.

Okay, let's refine the "Defining Your Own Terms" section to incorporate the explanation about namespacing within the `name` field description, keeping the style clear and researcher-focused.

### Defining Your Own Terms

While many common terms have pre-defined keys, you might need a term specific to your project or data that isn't already available (e.g., "Recording Setup", "Weather Condition", "Project Phase", "Noise Source"). You can easily define these custom terms directly within a configuration file (usually your main `.yaml` file).

Typically, you define custom terms under a dedicated section (often named `terms`). Inside this section, you create a list, where each item in the list defines one new term using the following fields:

* `key`: **(Required)** This is the unique shortcut key or nickname you will use to refer to this term throughout your configuration (e.g., `weather`, `setup_id`, `noise_src`). Choose something short and memorable.
* `label`: (Optional) A user-friendly label for the term, which might be used in reports or visualizations (e.g., "Weather Condition", "Setup ID"). If you don't provide one, it defaults to using the `key`.
* `name`: (Optional) A more formal or technical name for the term.
    * It's good practice, especially if defining terms that might overlap with standard vocabularies, to use a **namespaced format** like `<namespace>:<term_name>`. The `namespace` part helps avoid clashes with terms defined elsewhere. For example, the standard Darwin Core term for scientific name is `dwc:scientificName`, where `dwc` is the namespace for Darwin Core. Using namespaces makes your custom terms more specific and reduces potential confusion.
    * If you don't provide a `name`, it defaults to using the `key`.
* `definition`: (Optional) A brief text description explaining what this term represents (e.g., "The primary source of background noise identified", "General weather conditions during recording"). If omitted, it defaults to "Unknown".
* `uri`: (Optional) If your term definition comes directly from a standard online vocabulary (like Darwin Core), you can include its unique web identifier (URI) here.

**Example YAML Configuration for Custom Terms:**

```yaml
# In your main configuration file

# (Optional section to define custom terms)
terms:
  - key: weather              # Your chosen shortcut
    label: Weather Condition
    name: myproj:weather      # Formal namespaced name
    definition: General weather conditions during recording (e.g., Clear, Rain, Fog).

  - key: setup_id             # Another shortcut
    label: Recording Setup ID
    name: myproj:setupID      # Formal namespaced name
    definition: The unique identifier for the specific hardware setup used.

  - key: species              # Defining a term with a standard URI
    label: Scientific Name
    name: dwc:scientificName
    uri: http://rs.tdwg.org/dwc/terms/scientificName # Example URI
    definition: The full scientific name according to Darwin Core.

# ... other configuration sections ...
```

When `batdetect2` loads your configuration, it reads this `terms` section and adds your custom definitions (linked to their unique keys) to the central registry. These keys (`weather`, `setup_id`, etc.) are then ready to be used in other parts of your configuration, like defining filters or target classes.

### Using Keys to Specify Tags (in Filters, Class Definitions, etc.)

Now that you have keys for all the terms you need (both pre-defined and custom), you can easily refer to specific **tags** in other parts of your configuration, such as:

- Filtering rules (as seen in the `filtering` module documentation).
- Defining which tags represent your target classes.
- Associating extra information with your classes.

When you need to specify a tag, you typically use a structure with two fields:

- `key`: The **key** (shortcut) for the _Term_ part of the tag (e.g., `species`, `quality`, `weather`).
  **It defaults to `class`** if you omit it, which is common when defining the main target classes.
- `value`: The specific _value_ of the tag (e.g., `Myotis daubentonii`, `Good`, `Rain`).

**Example YAML Configuration using TagInfo (e.g., inside a filter rule):**

```yaml
# ... inside a filtering configuration section ...
rules:
  # Rule: Exclude events recorded in 'Rain'
  - match_type: exclude
    tags:
      - key: weather # Use the custom term key defined earlier
        value: Rain
  # Rule: Keep only 'Myotis daubentonii' (using the default 'class' key implicitly)
  - match_type: any # Or 'all' depending on logic
    tags:
      - value: Myotis daubentonii # 'key: class' is assumed by default here
        # key: class # Explicitly writing this is also fine
  # Rule: Keep only 'Good' quality events
  - match_type: any # Or 'all' depending on logic
    tags:
      - key: quality # Use a likely pre-defined key
        value: Good
```

### Summary

- Annotations have **tags** (Term + Value).
- This module uses short **keys** as shortcuts for Term definitions, stored in a **registry**.
- Many **common keys are pre-defined**.
- You can define **custom terms and keys** in your configuration file (using `key`, `label`, `definition`).
- You use these **keys** along with specific **values** to refer to tags in other configuration sections (like filters or class definitions), often defaulting to the `class` key.

This system makes your configurations cleaner, more readable, and less prone to errors by avoiding repetition of complex term definitions.
