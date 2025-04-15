# Step 2: Filtering Sound Events

## Purpose

When preparing your annotated audio data for training a `batdetect2` model, you often want to select only specific sound events.
For example, you might want to:

- Focus only on echolocation calls and ignore social calls or noise.
- Exclude annotations that were marked as low quality.
- Train only on specific species or groups of species.

This filtering module allows you to define rules based on the **tags** associated with each sound event annotation.
Only the events that pass _all_ your defined rules will be kept for further processing and training.

## How it Works: Rules

Filtering is controlled by a list of **rules**.
Each rule defines a condition based on the tags attached to a sound event.
An event must satisfy **all** the rules you define in your configuration to be included.
If an event fails even one rule, it is discarded.

## Defining Rules in Configuration

You define these rules within your main configuration file (usually a `.yaml` file) under a specific section (the exact name might depend on the main training config, but let's assume it's called `filtering`).

The configuration consists of a list named `rules`.
Each item in this list is a single filter rule.

Each **rule** has two parts:

1.  `match_type`: Specifies the _kind_ of check to perform.
2.  `tags`: A list of specific tags (each with a `key` and `value`) that the rule applies to.

```yaml
# Example structure in your configuration file
filtering:
  rules:
    - match_type: <TYPE_OF_CHECK_1>
      tags:
        - key: <tag_key_1a>
          value: <tag_value_1a>
        - key: <tag_key_1b>
          value: <tag_value_1b>
    - match_type: <TYPE_OF_CHECK_2>
      tags:
        - key: <tag_key_2a>
          value: <tag_value_2a>
    # ... add more rules as needed
```

## Understanding `match_type`

This determines _how_ the list of `tags` in the rule is used to check a sound event.
There are four types:

1.  **`any`**: (Keep if _at least one_ tag matches)

    - The sound event **passes** this rule if it has **at least one** of the tags listed in the `tags` section of the rule.
    - Think of it as an **OR** condition.
    - _Example Use Case:_ Keep events if they are tagged as `Species: Pip Pip` OR `Species: Pip Pyg`.

2.  **`all`**: (Keep only if _all_ tags match)

    - The sound event **passes** this rule only if it has **all** of the tags listed in the `tags` section.
      The event can have _other_ tags as well, but it must contain _all_ the ones specified here.
    - Think of it as an **AND** condition.
    - _Example Use Case:_ Keep events only if they are tagged with `Sound Type: Echolocation` AND `Quality: Good`.

3.  **`exclude`**: (Discard if _any_ tag matches)

    - The sound event **passes** this rule only if it does **not** have **any** of the tags listed in the `tags` section.
      If it matches even one tag in the list, the event is discarded.
    - _Example Use Case:_ Discard events if they are tagged `Quality: Poor` OR `Noise Source: Insect`.

4.  **`equal`**: (Keep only if tags match _exactly_)
    - The sound event **passes** this rule only if its set of tags is _exactly identical_ to the list of `tags` provided in the rule (no more, no less).
    - _Note:_ This is very strict and usually less useful than `all` or `any`.

## Combining Rules

Remember: A sound event must **pass every single rule** defined in the `rules` list to be kept.
The rules are checked one by one, and if an event fails any rule, it's immediately excluded from further consideration.

## Examples

**Example 1: Keep good quality echolocation calls**

```yaml
filtering:
  rules:
    # Rule 1: Must have the 'Echolocation' tag
    - match_type: any # Could also use 'all' if 'Sound Type' is the only tag expected
      tags:
        - key: Sound Type
          value: Echolocation
    # Rule 2: Must NOT have the 'Poor' quality tag
    - match_type: exclude
      tags:
        - key: Quality
          value: Poor
```

_Explanation:_ An event is kept only if it passes BOTH rules.
It must have the `Sound Type: Echolocation` tag AND it must NOT have the `Quality: Poor` tag.

**Example 2: Keep calls from Pipistrellus species recorded in a specific project, excluding uncertain IDs**

```yaml
filtering:
  rules:
    # Rule 1: Must be either Pip pip or Pip pyg
    - match_type: any
      tags:
        - key: Species
          value: Pipistrellus pipistrellus
        - key: Species
          value: Pipistrellus pygmaeus
    # Rule 2: Must belong to 'Project Alpha'
    - match_type: any # Using 'any' as it likely only has one project tag
      tags:
        - key: Project ID
          value: Project Alpha
    # Rule 3: Exclude if ID Certainty is 'Low' or 'Maybe'
    - match_type: exclude
      tags:
        - key: ID Certainty
          value: Low
        - key: ID Certainty
          value: Maybe
```

_Explanation:_ An event is kept only if it passes ALL three rules:

1.  It has a `Species` tag that is _either_ `Pipistrellus pipistrellus` OR `Pipistrellus pygmaeus`.
2.  It has the `Project ID: Project Alpha` tag.
3.  It does _not_ have an `ID Certainty: Low` tag AND it does _not_ have an `ID Certainty: Maybe` tag.

## Usage

You will typically specify the path to the configuration file containing these `filtering` rules when you set up your data processing or training pipeline in `batdetect2`.
The tool will then automatically load these rules and apply them to your annotated sound events.
