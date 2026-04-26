# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Concrete loader for SPI default handler contract YAML templates (OMN-9755).

File I/O lives in omnibase_infra (concrete implementation layer).
YAML data files remain in omnibase_spi/contracts/defaults/ (declarative data, no I/O).
"""

from __future__ import annotations

from importlib.resources import files

import yaml

from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_spi.exceptions import TemplateNotFoundError, TemplateParseError

_VALID_TEMPLATES = frozenset(
    {
        "default_compute_handler.yaml",
        "default_effect_handler.yaml",
        "default_nondeterministic_compute_handler.yaml",
        "default_github_pr_poller.yaml",
    }
)


def load_default_handler_contract(template_name: str) -> ModelHandlerContract:
    """Load and validate a default handler contract template from omnibase_spi.

    Args:
        template_name: Bare filename of the YAML template
            (e.g. "default_effect_handler.yaml"). Absolute paths and
            directory traversal sequences are rejected.

    Returns:
        Validated ModelHandlerContract instance.

    Raises:
        TemplateNotFoundError: If the template name is invalid or not in the
            known template set.
        TemplateParseError: If the file contains invalid YAML.
    """
    from pathlib import Path

    name = Path(template_name)
    # Reject paths that aren't bare filenames (traversal, absolute, subdirs).
    if name.parts != (template_name,) or template_name not in _VALID_TEMPLATES:
        raise TemplateNotFoundError(
            f"Template not found: {template_name!r}",
            context={"template_name": template_name},
        )

    try:
        yaml_text = (
            files("omnibase_spi.contracts.defaults")
            .joinpath(template_name)
            .read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise TemplateNotFoundError(
            f"Template not found: {template_name!r}",
            context={"template_name": template_name},
        ) from exc

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise TemplateParseError(
            f"Invalid YAML in template: {template_name!r}",
            context={"template_name": template_name},
        ) from exc

    return ModelHandlerContract.model_validate(raw)
