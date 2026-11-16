# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request and result of the sibling-clone pin node (OMN-23947)."""

from __future__ import annotations

from omnibase_infra.models.sibling_pin.model_sibling_clone_pin_request import (
    ModelSiblingClonePinRequest,
)
from omnibase_infra.models.sibling_pin.model_sibling_clone_pin_result import (
    ModelSiblingClonePinResult,
)

__all__: list[str] = ["ModelSiblingClonePinRequest", "ModelSiblingClonePinResult"]
