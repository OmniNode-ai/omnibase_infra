# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared outbound-call timeouts (OMN-21623).

``OUTBOUND_TIMEOUTS`` is the settings block loaded from ``outbound_timeouts.yaml``;
call sites read their waits from it and hold no timeout literal.
"""

from omnibase_infra.models.outbound_timeouts.model_outbound_timeouts import (
    ModelOutboundTimeouts,
    load_outbound_timeouts,
)

OUTBOUND_TIMEOUTS = load_outbound_timeouts()

__all__ = ["OUTBOUND_TIMEOUTS", "ModelOutboundTimeouts", "load_outbound_timeouts"]
