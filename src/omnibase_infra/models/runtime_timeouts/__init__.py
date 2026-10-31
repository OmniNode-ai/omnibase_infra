# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared runtime timeouts (OMN-21719).

``RUNTIME_TIMEOUTS`` is the settings block loaded from ``runtime_timeouts.yaml``;
call sites read their waits from it and hold no timeout literal.
"""

from omnibase_infra.models.runtime_timeouts.model_runtime_timeouts import (
    ModelRuntimeTimeouts,
    load_runtime_timeouts,
)

RUNTIME_TIMEOUTS = load_runtime_timeouts()

__all__ = ["RUNTIME_TIMEOUTS", "ModelRuntimeTimeouts", "load_runtime_timeouts"]
