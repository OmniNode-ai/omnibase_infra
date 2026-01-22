# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Models for Validation and Error Reporting.

This module exports handler-related models for structured validation
and error reporting in ONEX handlers.

.. versionadded:: 0.6.1
    Created as part of OMN-1091 structured validation and error reporting.

.. versionchanged:: 0.6.2
    Added ModelHandlerDescriptor and ModelContractDiscoveryResult for
    OMN-1097 filesystem handler discovery.

.. versionchanged:: 0.6.4
    Added ModelBootstrapHandlerDescriptor for OMN-1087 bootstrap handler
    validation with required handler_class field.

.. versionchanged:: 0.7.0
    Added ModelHandlerSourceConfig for OMN-1095 handler source mode
    configuration with production hardening features.

Note:
    ModelContractDiscoveryResult uses a forward reference to
    ModelHandlerValidationError to avoid circular imports. The forward
    reference is resolved via model_rebuild() in handler_contract_source.py
    after both classes are defined. This pattern is tested in
    tests/unit/runtime/test_handler_contract_source.py.
"""

from omnibase_infra.models.handlers.model_bootstrap_handler_descriptor import (
    ModelBootstrapHandlerDescriptor,
)
from omnibase_infra.models.handlers.model_contract_discovery_result import (
    ModelContractDiscoveryResult,
)
from omnibase_infra.models.handlers.model_handler_descriptor import (
    LiteralHandlerKind,
    ModelHandlerDescriptor,
)
from omnibase_infra.models.handlers.model_handler_identifier import (
    ModelHandlerIdentifier,
)
from omnibase_infra.models.handlers.model_handler_source_config import (
    ModelHandlerSourceConfig,
)

__all__ = [
    "LiteralHandlerKind",
    "ModelBootstrapHandlerDescriptor",
    "ModelContractDiscoveryResult",
    "ModelHandlerDescriptor",
    "ModelHandlerIdentifier",
    "ModelHandlerSourceConfig",
]
