# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared operational timeouts (OMN-21802).

``TIMEOUTS`` is the settings block loaded from ``operational_timeouts.yaml``;
call sites read their waits from it and hold no timeout literal.
"""

from omnibase_infra.models.operational_timeouts.model_operational_timeouts import (
    ModelOperationalTimeouts,
    load_operational_timeouts,
)

TIMEOUTS = load_operational_timeouts()

__all__ = ["TIMEOUTS", "ModelOperationalTimeouts", "load_operational_timeouts"]
