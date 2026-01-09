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

Note:
    ModelContractDiscoveryResult uses a forward reference to
    ModelHandlerValidationError to avoid circular imports. The forward
    reference is resolved via model_rebuild() in handler_contract_source.py
    after both classes are defined. This pattern is tested in
    tests/unit/runtime/test_handler_contract_source.py.
"""

from omnibase_infra.models.handlers.model_contract_discovery_result import (
    ModelContractDiscoveryResult,
)
from omnibase_infra.models.handlers.model_handler_descriptor import (
    ModelHandlerDescriptor,
)
from omnibase_infra.models.handlers.model_handler_identifier import (
    ModelHandlerIdentifier,
)

__all__ = [
    "ModelContractDiscoveryResult",
    "ModelHandlerDescriptor",
    "ModelHandlerIdentifier",
]
