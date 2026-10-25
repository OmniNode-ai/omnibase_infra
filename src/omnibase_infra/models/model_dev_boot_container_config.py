# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker labels and healthcheck configuration."""

from typing import NotRequired

from typing_extensions import TypedDict

from omnibase_infra.models.model_dev_boot_healthcheck import ModelDevBootHealthcheck


class ModelDevBootContainerConfig(TypedDict):
    Labels: dict[str, str]
    Healthcheck: NotRequired[ModelDevBootHealthcheck]
