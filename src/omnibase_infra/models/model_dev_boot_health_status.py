# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker's boot health verdict."""

from typing_extensions import TypedDict

from omnibase_infra.models.model_dev_boot_health_probe import ModelDevBootHealthProbe


class ModelDevBootHealthStatus(TypedDict, total=False):
    Status: str
    FailingStreak: int
    Log: list[ModelDevBootHealthProbe]
