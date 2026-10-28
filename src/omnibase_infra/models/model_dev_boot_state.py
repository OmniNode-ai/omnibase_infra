# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker runtime state used by the boot guard."""

from typing_extensions import TypedDict

from omnibase_infra.models.model_dev_boot_health_status import ModelDevBootHealthStatus


class ModelDevBootState(TypedDict, total=False):
    Status: str
    Health: ModelDevBootHealthStatus
