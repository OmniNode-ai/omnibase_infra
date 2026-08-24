# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""HandlerFaultInjectFixture — deterministic double-publish-failure fixture.

Follow-on from OMN-14498's live probe of the result-applier fix, which proved
the "safe ACK via durable DLQ" half live but could not prove the offset-
withholding half (BoundaryApplyPublishError propagating so the consumer
offset is withheld). That probe found no existing platform mechanism could
prove it with acceptable blast radius: the DLQ leg re-publishes the ORIGINAL
inbound command (small), never the oversized outbound result, so a single
contract's DLQ write can never be made to fail by inflating its own result;
and IAM-denying the shared commands DLQ topic would affect every other
contract's failures platform-wide.

This node is the permanent, zero-blast-radius fixture: its own dedicated
command/result topic pair, and (deployment-side, see the node's
contract.yaml `metadata.fault_injection` block and the deployment runbook)
its own private `dead_letter_topic` override that is deliberately left
unprovisioned. Driving both failures then only requires one thing from this
handler: a deterministic, size-controlled result.

Canonical def-B dispatch entrypoint (OMN-14355 / OMN-14403 def-B; see
`omnibase_infra` CLAUDE.md rule 7a) — `handle(request) -> ModelY`, no
envelope import, no `ModelHandlerOutput` wrapping. Pure, deterministic,
zero I/O.

Ticket: OMN-16265
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_fault_inject_fixture_compute.models.model_fault_inject_fixture_command import (
    ModelFaultInjectFixtureCommand,
)
from omnibase_infra.nodes.node_fault_inject_fixture_compute.models.model_fault_inject_fixture_result import (
    ModelFaultInjectFixtureResult,
)

_PADDING_CHAR = "x"
"""Single-byte ASCII filler character. Deterministic and cheap to size-verify
(one padding char == one UTF-8 byte), so `len(padding) == inflate_result_bytes`
exactly — no encoding-width surprises for callers computing target sizes
against a live broker's message.max.bytes."""


class HandlerFaultInjectFixture:
    """Pure compute handler: returns a deterministic, size-controlled result.

    Classification: ``COMPUTE_HANDLER`` + ``COMPUTE`` — no I/O, no side
    effects. The runtime's own boundary-apply-publish path is what turns a
    large returned result into a primary-publish failure; this handler's
    only job is producing that result deterministically from the request.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        request: ModelFaultInjectFixtureCommand,
    ) -> ModelFaultInjectFixtureResult:
        """Produce a result whose serialized size is controlled by the request.

        ``padding`` is `inflate_result_bytes` copies of a single-byte ASCII
        character, so the result grows linearly and deterministically with
        the request — the same technique the OMN-14498 live probe used
        (small input, handler-amplified output) to push a result past the
        live broker's ``message.max.bytes`` / producer ``max_request_size``.
        """
        padding = _PADDING_CHAR * request.inflate_result_bytes
        return ModelFaultInjectFixtureResult(
            correlation_id=request.correlation_id,
            inflate_result_bytes=request.inflate_result_bytes,
            padding=padding,
            padding_byte_length=len(padding.encode("utf-8")),
        )


__all__: list[str] = ["HandlerFaultInjectFixture"]
