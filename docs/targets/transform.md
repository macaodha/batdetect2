## Step 3: Transforming Annotation Tags (Optional)

### Purpose and Context

After defining your vocabulary (Step 1: Terms) and filtering out irrelevant sound events (Step 2: Filtering), you have a dataset of annotations ready for the next stages.
Before you select the final target classes for training (Step 4), you might want or need to **modify the tags** associated with your annotations.
This optional step allows you to clean up, standardize, or derive new information from your existing tags.

**Why transform tags?**

- **Correcting Mistakes:** Fix typos or incorrect values in specific tags (e.g., changing an incorrect species label).
- **Standardizing Labels:** Ensure consistency if the same information was tagged using slightly different values (e.g., mapping "echolocation", "Echoloc.", and "Echolocation Call" all to a single standard value: "Echolocation").
- **Grouping Related Concepts:** Combine different specific tags into a broader category (e.g., mapping several different species tags like _Myotis daubentonii_ and _Myotis nattereri_ to a single `genus: Myotis` tag).
- **Deriving New Information:** Automatically create new tags based on existing ones (e.g., automatically generating a `genus: Myotis` tag whenever a `species: Myotis daubentonii` tag is present).

This step uses the `batdetect2.targets.transform` module to apply these changes based on rules you define.

### How it Works: Transformation Rules

You control how tags are transformed by defining a list of **rules** in your configuration file (e.g., your main `.yaml` file, often under a section named `transform`).

Each rule specifies a particular type of transformation to perform.
Importantly, the rules are applied **sequentially**, in the exact order they appear in your configuration list.
The output annotation from one rule becomes the input for the next rule in the list.
This means the order can matter!

### Types of Transformation Rules

Here are the main types of rules you can define:

1.  **Replace an Exact Tag (`replace`)**

    - **Use Case:** Fixing a specific, known incorrect tag.
    - **How it works:** You specify the _exact_ original tag (both its term key and value) and the _exact_ tag you want to replace it with.
    - **Example Config:** Replace the informal tag `species: Pip pip` with the correct scientific name tag.
      ```yaml
      transform:
        rules:
          - rule_type: replace
            original:
              key: species # Term key of the tag to find
              value: "Pip pip" # Value of the tag to find
            replacement:
              key: species # Term key of the replacement tag
              value: "Pipistrellus pipistrellus" # Value of the replacement tag
      ```

2.  **Map Values (`map_value`)**

    - **Use Case:** Standardizing different values used for the same concept, or grouping multiple specific values into one category.
    - **How it works:** You specify a `source_term_key` (the type of tag to look at, e.g., `call_type`).
      Then you provide a `value_mapping` dictionary listing original values and the new values they should be mapped to.
      Only tags matching the `source_term_key` and having a value listed in the mapping will be changed.
      You can optionally specify a `target_term_key` if you want to change the term type as well (e.g., mapping species to a genus).
    - **Example Config:** Standardize different ways "Echolocation" might have been written for the `call_type` term.
      ```yaml
      transform:
        rules:
          - rule_type: map_value
            source_term_key: call_type # Look at 'call_type' tags
            # target_term_key is not specified, so the term stays 'call_type'
            value_mapping:
              echolocation: Echolocation
              Echolocation Call: Echolocation
              Echoloc.: Echolocation
              # Add mappings for other values like 'Social' if needed
      ```
    - **Example Config (Grouping):** Map specific Pipistrellus species tags to a single `genus: Pipistrellus` tag.
      ```yaml
      transform:
        rules:
          - rule_type: map_value
            source_term_key: species # Look at 'species' tags
            target_term_key: genus # Change the term to 'genus'
            value_mapping:
              "Pipistrellus pipistrellus": Pipistrellus
              "Pipistrellus pygmaeus": Pipistrellus
              "Pipistrellus nathusii": Pipistrellus
      ```

3.  **Derive a New Tag (`derive_tag`)**
    - **Use Case:** Automatically creating new information based on existing tags, like getting the genus from a species name.
    - **How it works:** You specify a `source_term_key` (e.g., `species`).
      You provide a `target_term_key` for the new tag to be created (e.g., `genus`).
      You also provide the name of a `derivation_function` (e.g., `"extract_genus"`) that knows how to perform the calculation (e.g., take "Myotis daubentonii" and return "Myotis").
      `batdetect2` has some built-in functions, or you can potentially define your own (see advanced documentation).
      You can also choose whether to keep the original source tag (`keep_source: true`).
    - **Example Config:** Create a `genus` tag from the existing `species` tag, keeping the species tag.
      ```yaml
      transform:
        rules:
          - rule_type: derive_tag
            source_term_key: species # Use the value from the 'species' tag
            target_term_key: genus # Create a tag with the 'genus' term
            derivation_function: extract_genus # Use the built-in function for this
            keep_source: true # Keep the original 'species' tag
      ```
    - **Another Example:** Convert species names to uppercase (modifying the value of the _same_ term).
      ```yaml
      transform:
        rules:
          - rule_type: derive_tag
            source_term_key: species # Use the value from the 'species' tag
            # target_term_key is not specified, so the term stays 'species'
            derivation_function: to_upper_case # Assume this function exists
            keep_source: false # Replace the original species tag
      ```

### Rule Order Matters!

Remember that rules are applied one after another.
If you have multiple rules, make sure they are ordered correctly to achieve the desired outcome.
For instance, you might want to standardize species names _before_ deriving the genus from them.

### Outcome

After applying all the transformation rules you've defined, the annotations will proceed to the next step (Step 4: Select Target Tags & Define Classes) with their tags potentially cleaned, standardized, or augmented based on your configuration.
If you don't define any rules, the tags simply pass through this step unchanged.
