# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected ports for the dedicated action-authorization claim boundary."""

from __future__ import annotations

from typing import Protocol

from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    ModelActionAuthorizationClaimRequest,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_result import (
    ModelActionAuthorizationClaimResult,
)


class ProtocolActionAuthorizationClaimPort(Protocol):
    """Closed port that cannot infer action permission from caller input."""

    async def claim(
        self, request: ModelActionAuthorizationClaimRequest
    ) -> ModelActionAuthorizationClaimResult: ...


__all__ = ["ProtocolActionAuthorizationClaimPort"]
