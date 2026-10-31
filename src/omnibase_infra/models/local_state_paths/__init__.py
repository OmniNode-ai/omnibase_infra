# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared paths for process-local runtime state."""

from omnibase_infra.models.local_state_paths.model_local_state_paths import (
    ModelLocalStatePaths,
    load_local_state_paths,
)

__all__ = ["ModelLocalStatePaths", "load_local_state_paths"]
