# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Models for the release-identity compute node.

Provides the typed request (pre-collected gate inputs) and decision (exit code,
stream, message, reason code) models.

Ticket: OMN-14471
"""

from omnibase_infra.nodes.node_release_identity_compute.models.model_release_identity_decision import (
    ModelReleaseIdentityDecision,
)
from omnibase_infra.nodes.node_release_identity_compute.models.model_release_identity_request import (
    ModelReleaseIdentityRequest,
)

__all__: list[str] = [
    "ModelReleaseIdentityDecision",
    "ModelReleaseIdentityRequest",
]
