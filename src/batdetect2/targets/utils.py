from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batdetect2.targets.types import TargetProtocol


def check_target_compatibility(
    targets: "TargetProtocol",
    class_names: list[str],
) -> bool:
    """Check if a target definition can decode a model's outputs.

    Parameters
    ----------
    targets : TargetProtocol
        Target definition that would be used with the model outputs.
    class_names : list[str]
        Class names produced by the model checkpoint.

    Returns
    -------
    bool
        True when every model class name exists in the provided targets,
        False otherwise.
    """
    target_class_names = set(targets.class_names)
    model_class_names = set(class_names)

    return model_class_names.issubset(target_class_names)
