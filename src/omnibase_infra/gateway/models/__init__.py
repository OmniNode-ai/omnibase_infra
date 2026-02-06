# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gateway Models Module.

This module exports Pydantic models for gateway configuration.

Exports:
    ModelGatewayConfig: Configuration model for gateway signing and validation
"""

from omnibase_infra.gateway.models.model_gateway_config import ModelGatewayConfig

__all__: list[str] = [
    "ModelGatewayConfig",
]
