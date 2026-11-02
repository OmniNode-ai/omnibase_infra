# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared outbound-call timeouts (OMN-21623).

Call sites load the settings block from ``outbound_timeouts.yaml`` on first
use. Importing this package performs no settings file I/O.
"""

from omnibase_infra.models.outbound_timeouts.model_outbound_timeouts import (
    ModelOutboundTimeouts,
    load_outbound_timeouts,
)

__all__ = ["ModelOutboundTimeouts", "load_outbound_timeouts"]
