# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validated Docker inspection data used by the host boot guard."""

from typing import NotRequired

from typing_extensions import TypedDict

from omnibase_infra.models.model_dev_boot_container_config import (
    ModelDevBootContainerConfig,
)
from omnibase_infra.models.model_dev_boot_state import ModelDevBootState


class ModelDevBootContainer(TypedDict):
    Id: str
    Image: str
    Config: ModelDevBootContainerConfig
    State: NotRequired[ModelDevBootState]
