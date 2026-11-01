# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared endpoint defaults (OMN-21690)."""

from omnibase_infra.models.endpoint_defaults.model_endpoint_defaults import (
    ModelEndpointDefaults,
    load_endpoint_defaults,
)

__all__ = ["ModelEndpointDefaults", "load_endpoint_defaults"]
