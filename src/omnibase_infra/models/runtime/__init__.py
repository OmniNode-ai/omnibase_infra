# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Models for Handler Plugin Loading.

This module exports runtime-specific models used by the Handler Plugin Loader
and related runtime components.

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Handler Plugin Loader implementation.
"""

from omnibase_infra.models.runtime.model_handler_contract import ModelHandlerContract
from omnibase_infra.models.runtime.model_loaded_handler import ModelLoadedHandler

__all__ = [
    "ModelHandlerContract",
    "ModelLoadedHandler",
]
