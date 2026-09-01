# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Safe raw-wire ingress exposed by first-effect deployment composition."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from omnibase_infra.runtime.first_effect_ledger.model_verified_first_effect_grant_record import (
    ModelVerifiedFirstEffectGrantRecord,
)


class ProtocolSignedFirstEffectGrantIngress(Protocol):
    """Accept only a raw signed wire; no projection or recorder is exposed."""

    async def verify_and_record(
        self, wire: object, *, now: datetime
    ) -> ModelVerifiedFirstEffectGrantRecord: ...

    async def close(self) -> None: ...


__all__ = ["ProtocolSignedFirstEffectGrantIngress"]
