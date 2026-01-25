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
    ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError
    to avoid circular imports between models.handlers and models.errors packages.
    The forward reference is resolved via model_rebuild() calls in runtime modules
    that import ModelHandlerValidationError (e.g., handler_contract_source.py,
    handler_bootstrap_source.py, registry_contract_source.py). Each module calls
    model_rebuild() after importing both the model and the forward-referenced type.
    This pattern is required because:
    1. models.errors imports ModelHandlerIdentifier from models.handlers
    2. models.handlers cannot import from models.errors at module level (circular)
    3. model_rebuild() is idempotent, so multiple calls are harmless
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

# =============================================================================
# Forward Reference Resolution
# =============================================================================
# ModelContractDiscoveryResult uses TYPE_CHECKING to defer import of
# ModelHandlerValidationError to avoid circular imports:
#   - models.errors imports ModelHandlerIdentifier from models.handlers
#   - models.handlers cannot import ModelHandlerValidationError at module level
#
# The forward reference is resolved via model_rebuild() in runtime modules that
# import ModelHandlerValidationError (e.g., handler_contract_source.py,
# handler_bootstrap_source.py, registry_contract_source.py, handler_source_resolver.py).
# Each module calls model_rebuild() at module level after importing both the model
# and the forward-referenced type. This is safe because model_rebuild() is idempotent.
#
# Why NOT here at module level:
#   - Circular import: models.handlers.__init__ -> models.errors.__init__
#     -> model_handler_validation_error.py -> models.handlers (for identifier)
#   - Runtime modules load after model packages, avoiding this cycle
# =============================================================================
