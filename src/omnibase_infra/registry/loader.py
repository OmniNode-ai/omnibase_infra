# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry


def load_artifact_registry(path: Path) -> ModelArtifactRegistry:
    if not path.exists():
        raise FileNotFoundError(f"Artifact registry not found: {path}")
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Failed to parse artifact registry YAML at {path}: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid artifact registry: expected mapping, got {type(data).__name__} at {path}"
        )
    try:
        return ModelArtifactRegistry(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid artifact registry at {path}: {e}") from e
