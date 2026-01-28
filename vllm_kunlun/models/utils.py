# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


class StageMissingLayer(nn.Module):
    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None:
        super().__init__()

        self.stage_name = stage_name

        # Don't register this as a child module in order to
        # avoid missing keys when loading weights
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        return getattr(self.__dict__["module"], name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self} should not be called")

    def extra_repr(self) -> str:
        return f"stage_name={self.stage_name!r}"


_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + ".")
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, (StageMissingLayer, PPMissingLayer)):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"
