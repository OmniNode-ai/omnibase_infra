# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One authoritative surface that can vouch for a produced record (OMN-15861).

Split from ``ProtocolConfirmationStrategy`` (one protocol per file, enforced by
the ONEX architecture validator) but the pairing is the point: a readback source
reports FACTS and never decides policy, while a confirmation strategy decides
the durability VERDICT including what an indeterminate source means.

That division is what makes fail-closed enforceable in one place. If `observe`
were allowed to translate "I could not reach the broker" into ``False``, every
call site would have to re-derive whether a false meant absent or unknown -- and
the platform's existing UNKNOWN-fails-closed doctrine would be re-implemented,
inconsistently, per caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt


@runtime_checkable
class ProtocolReadbackSource(Protocol):
    """One authoritative surface that can be asked whether a record landed."""

    @property
    def transport(self) -> EnumInfraTransportType:
        """Transport this source can answer for.

        A strategy MUST refuse a receipt whose ``transport`` differs -- reading
        an in-memory history to confirm a Kafka produce is a false durable claim
        by construction.
        """
        ...

    async def observe(
        self,
        receipt: ModelPublishReceipt,
        *,
        deadline_seconds: float,
    ) -> bool:
        """Return whether the record at ``receipt`` is observable on this surface.

        Args:
            receipt: Coordinate to look for.
            deadline_seconds: Wall-clock budget. Implementations MUST return
                ``False`` on expiry rather than blocking indefinitely.

        Returns:
            ``True`` if the record was observed, ``False`` if the surface was
            reached and did not have it within the deadline.

        Raises:
            Exception: If the surface itself could not be consulted. Callers
                translate this to ``EnumConfirmationState.UNKNOWN`` -- a raised
                error is NOT evidence of absence.
        """
        ...


__all__: list[str] = ["ProtocolReadbackSource"]
