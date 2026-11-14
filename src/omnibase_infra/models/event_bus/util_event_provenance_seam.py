# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish-seam helper: stamp an envelope's JSON with the deployment provenance (OMN-23981)."""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.handlers.handler_event_provenance_stamp.handler import (
    REASON_ALREADY_STAMPED,
    REASON_STAMPED,
    HandlerEventProvenanceStamp,
    read_provenance_overlay,
)
from omnibase_infra.models.event_bus.model_event_provenance_stamp_request import (
    ModelEventProvenanceStampRequest,
)

logger = logging.getLogger(__name__)

_PROVENANCE_HANDLER = HandlerEventProvenanceStamp()


@lru_cache(maxsize=16)
def _warn_unstamped(reason: str) -> None:
    """Say ONCE per process and reason that events go out without provenance."""
    logger.warning(
        "event provenance not stamped: %s; this runtime's events read lane "
        "'unknown' (OMN-23981)",
        reason,
    )


def dump_envelope_json(envelope: object) -> object:
    """Return the JSON form of an envelope, stamped with this runtime's provenance.

    A ``ModelEventEnvelope`` that carries no provenance gets the overlay's
    when the overlay is whole; otherwise it goes out unchanged and reads
    ``unknown``, and the reason is logged once per process. Any other model
    is dumped as it is.
    """
    if isinstance(envelope, ModelEventEnvelope):
        result = _PROVENANCE_HANDLER.handle(
            ModelEventProvenanceStampRequest(
                envelope=envelope, overlay=read_provenance_overlay()
            )
        )
        if result.reason not in (REASON_STAMPED, REASON_ALREADY_STAMPED):
            _warn_unstamped(result.reason)
        return result.envelope.model_dump(mode="json")
    if isinstance(envelope, BaseModel):
        return envelope.model_dump(mode="json")
    raise TypeError(f"not a model: {type(envelope).__name__}")


__all__: list[str] = ["dump_envelope_json"]
