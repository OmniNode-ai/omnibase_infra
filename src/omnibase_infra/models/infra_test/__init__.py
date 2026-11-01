# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared ``onex-infra-test`` settings (OMN-21623)."""

from omnibase_infra.models.infra_test.model_infra_test_settings import (
    ModelInfraTestSettings,
    load_infra_test_settings,
)

__all__ = ["ModelInfraTestSettings", "load_infra_test_settings"]
