# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why the boot interleave will never attempt a contract (OMN-17372).

``subscribe_wired_contract_topics`` filters the wiring report down to an
``eligible`` list before it interleaves provision -> confirm-ready -> attach.
A contract dropped by one of those filters produces NO
:class:`~omnibase_infra.event_bus.model_contract_attach_result.ModelContractAttachResult`,
ever. That is correct behaviour for the interleave and catastrophic for any
consumer that assumed "every contract eventually reports": the contract-attach
readiness gate did assume exactly that and wedged ``/ready`` at 503 forever.

This enum is the vocabulary the interleave uses to SAY it skipped a contract,
so structural ineligibility is a reported fact rather than a silent absence.

Related Tickets:
    - OMN-17372: readiness must require the wired command topics — and must
      require only the contracts the interleave will actually attempt.
    - OMN-13237: the per-contract provision -> confirm-ready -> attach interleave.
    - OMN-15474: resolver-owned skips register zero dispatchers.
    - OMN-10864: plugin-managed contracts own their own subscription.
    - OMN-17562: an all-no-op dispatch must not consume and commit offsets.
"""

from __future__ import annotations

from enum import Enum


class EnumContractAttachExclusionReason(str, Enum):
    """Structural reasons a wired contract is never handed to the interleave.

    Values:
        NOT_WIRED: The contract's wiring outcome is not ``WIRED`` — most often
            ``SKIPPED`` for "No handler_routing declared in contract" or "No
            event_bus.subscribe_topics declared in contract".
        ABSENT_FROM_MANIFEST: The wiring report named a contract the manifest
            being subscribed does not contain.
        NO_DISPATCHERS_REGISTERED: The contract owns zero dispatchers, so it
            owns no consume callback (OMN-15474).
        RAW_EVENT_PROJECTION_WITHOUT_APPLIER: A raw audit/projection consumer
            with no result applier; subscribing it would consume offsets while
            dropping the intents it emits.
        PLUGIN_MANAGED: A domain plugin owns the Kafka subscription (OMN-10864).
        NO_LIVE_DISPATCHER: Every handler entry wires a no-op dispatch, so
            consuming would commit offsets over events no handler runs
            (OMN-17562).
    """

    NOT_WIRED = "not_wired"
    ABSENT_FROM_MANIFEST = "absent_from_manifest"
    NO_DISPATCHERS_REGISTERED = "no_dispatchers_registered"
    RAW_EVENT_PROJECTION_WITHOUT_APPLIER = "raw_event_projection_without_applier"
    PLUGIN_MANAGED = "plugin_managed"
    NO_LIVE_DISPATCHER = "no_live_dispatcher"


__all__: list[str] = ["EnumContractAttachExclusionReason"]
