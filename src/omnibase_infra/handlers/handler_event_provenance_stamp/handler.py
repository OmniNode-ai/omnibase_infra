# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stamp an envelope with the provenance its deployment declares (OMN-23981).

Pure and deterministic, definition B: ``handle(request) -> result``. The
overlay is the only source of the stamp; nothing is defaulted. An overlay that
is absent, partial or invalid stamps nothing, so the envelope reads
``unknown``, never ``dev``. An envelope that already carries provenance
keeps it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from omnibase_core.enums.enum_provenance_install_source import (
    EnumProvenanceInstallSource,
)
from omnibase_core.enums.enum_provenance_lane import EnumProvenanceLane
from omnibase_core.enums.enum_provenance_lane_kind import EnumProvenanceLaneKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.events.model_event_provenance import ModelEventProvenance
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.event_bus.model_event_provenance_overlay import (
    ModelEventProvenanceOverlay,
)
from omnibase_infra.models.event_bus.model_event_provenance_stamp_request import (
    ModelEventProvenanceStampRequest,
)
from omnibase_infra.models.event_bus.model_event_provenance_stamp_result import (
    ModelEventProvenanceStampResult,
)

logger = logging.getLogger(__name__)

REASON_STAMPED = "stamped"
REASON_ALREADY_STAMPED = "already_stamped"
REASON_NO_OVERLAY = "no provenance overlay declared"


def _clean(value: str | None) -> str:
    return (value or "").strip()


def provenance_from_overlay(
    overlay: ModelEventProvenanceOverlay,
) -> tuple[ModelEventProvenance | None, str]:
    """Validate an overlay into the core provenance model, or say why not."""
    fields = {
        "lane": _clean(overlay.lane),
        "host": _clean(overlay.host),
        "install_source": _clean(overlay.install_source),
        "revision": _clean(overlay.revision),
    }
    missing = [name for name, value in fields.items() if not value]
    if len(missing) == len(fields):
        return None, REASON_NO_OVERLAY
    if missing:
        return None, f"overlay missing {', '.join(missing)}"
    kind = _clean(overlay.lane_kind).lower()
    try:
        lane = EnumProvenanceLane(fields["lane"].lower())
        if lane == EnumProvenanceLane.UNKNOWN:
            return None, "overlay lane unknown is the absence of a stamp"
        return (
            ModelEventProvenance(
                lane=lane,
                lane_kind=EnumProvenanceLaneKind(kind) if kind else None,
                host=fields["host"],
                install_source=EnumProvenanceInstallSource(
                    fields["install_source"].lower()
                ),
                revision=fields["revision"],
            ),
            REASON_STAMPED,
        )
    except (ValueError, ValidationError) as exc:
        return None, f"overlay refused: {exc}"


class HandlerEventProvenanceStamp:
    """Envelope plus overlay, to the envelope to publish."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelEventProvenanceStampRequest
    ) -> ModelEventProvenanceStampResult:
        """Stamp ``request.envelope`` when it carries no provenance and the overlay is whole."""
        envelope = request.envelope
        if envelope.provenance is not None:
            return ModelEventProvenanceStampResult(
                envelope=envelope, stamped=False, reason=REASON_ALREADY_STAMPED
            )
        provenance, reason = provenance_from_overlay(request.overlay)
        if provenance is None:
            return ModelEventProvenanceStampResult(
                envelope=envelope, stamped=False, reason=reason
            )
        return ModelEventProvenanceStampResult(
            envelope=envelope.model_copy(update={"provenance": provenance}),
            stamped=True,
            reason=REASON_STAMPED,
        )


_HANDLER = HandlerEventProvenanceStamp()


@lru_cache(maxsize=16)
def _warn_unstamped(reason: str) -> None:
    """Say ONCE per process and reason that events go out without provenance."""
    logger.warning(
        "event provenance not stamped: %s; this runtime's events read lane "
        "'unknown' (OMN-23981)",
        reason,
    )


def read_provenance_overlay(
    environ: Mapping[str, str] | None = None,
) -> ModelEventProvenanceOverlay:
    """Read the overlay the deployment declared; absent variables stay None."""
    return ModelEventProvenanceOverlay.from_environ(
        os.environ if environ is None else environ
    )


def dump_envelope_json(envelope: object) -> object:
    """Return the JSON form of an envelope, stamped with this runtime's provenance.

    A ``ModelEventEnvelope`` that carries no provenance gets the overlay's
    when the overlay is whole; otherwise it goes out unchanged and reads
    ``unknown``, and the reason is logged once per process. Any other model
    is dumped as it is.
    """
    if isinstance(envelope, ModelEventEnvelope):
        result = _HANDLER.handle(
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


__all__: list[str] = [
    "REASON_ALREADY_STAMPED",
    "REASON_NO_OVERLAY",
    "REASON_STAMPED",
    "HandlerEventProvenanceStamp",
    "dump_envelope_json",
    "provenance_from_overlay",
    "read_provenance_overlay",
]
