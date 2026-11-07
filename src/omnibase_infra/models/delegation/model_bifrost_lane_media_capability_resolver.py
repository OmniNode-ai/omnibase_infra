# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Endpoint shape check and capability resolver for lane overlays (OMN-22756).

A lane overlay backend declares at most one typed media capability, and a node
asks the overlay for a capability and gets the serving binding and its URL back;
it never names a host or port. The functions here hold the three rules the
binding and the overlay call:

* the endpoint URL is a complete http(s) endpoint whose path is one of the
  known shapes, and when the binding declares a capability the path is that
  capability's path;
* a lane has at most one serving backend per capability;
* a capability asked for resolves to its one serving backend, or raises a typed
  error naming the backend or the capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from omnibase_infra.enums.enum_bifrost_lane_media_capability import (
    EnumBifrostLaneMediaCapability,
)
from omnibase_infra.models.delegation.model_bifrost_lane_media_capability_error import (
    BifrostLaneMediaCapabilityAmbiguousError,
    BifrostLaneMediaCapabilityNotServingError,
    BifrostLaneMediaCapabilityShapeError,
    BifrostLaneMediaCapabilityUnknownError,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
        ModelBifrostLaneBackendBinding,
    )

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
#: OMN-17099: the embedding backend's real endpoint is OpenAI-embeddings-shaped,
#: not chat-completions-shaped (a live probe on 2026-09-27 found the chat path
#: answered 404 and the embeddings path returned real vectors). OMN-22756 adds the
#: speech, transcription and video shapes, one per typed media capability. These
#: are the only complete endpoint shapes Bifrost backends are known to serve.
COMPLETE_ENDPOINT_PATHS = frozenset(
    {_CHAT_COMPLETIONS_PATH, *(c.endpoint_path for c in EnumBifrostLaneMediaCapability)}
)
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def check_media_capability_endpoint_shape(
    backend_key: str,
    media_capability: EnumBifrostLaneMediaCapability | None,
    endpoint_path: str,
) -> None:
    """Refuse a binding whose capability and endpoint path disagree.

    Each typed media capability is served at exactly one endpoint path, so the
    declared capability is the whole truth about the shape. A binding with no
    typed capability is not checked here.

    Raises:
        BifrostLaneMediaCapabilityShapeError: naming the backend, the declared
            capability and the path it should end in.
    """
    if media_capability is None:
        return
    if endpoint_path != media_capability.endpoint_path:
        raise BifrostLaneMediaCapabilityShapeError(
            f"backend {backend_key!r} declares media capability "
            f"{media_capability.value!r}, which is served at "
            f"{media_capability.endpoint_path!r}, but its endpoint_url path is "
            f"{endpoint_path!r}",
            backend_key=backend_key,
            media_capability=media_capability.value,
            endpoint_path=endpoint_path,
        )


def validate_binding_endpoint(
    backend_key: str,
    endpoint_url: str,
    media_capability: EnumBifrostLaneMediaCapability | None,
) -> str:
    """Validate a binding's endpoint URL and return its scheme.

    Raises:
        ValueError: the URL is not a complete http(s) endpoint ending in one of
            the known shapes, or has a malformed port.
        BifrostLaneMediaCapabilityShapeError: the declared capability and the
            URL path disagree.
    """
    parsed = urlsplit(endpoint_url)
    try:
        _ = parsed.port  # raises ValueError on a malformed port
    except ValueError as exc:
        raise ValueError(
            f"endpoint_url for {backend_key!r} has an invalid port"
        ) from exc
    if (
        parsed.scheme not in _ALLOWED_SCHEMES
        or not parsed.hostname
        or parsed.path not in COMPLETE_ENDPOINT_PATHS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"endpoint_url for {backend_key!r} must be a complete "
            f"http(s) endpoint ending in one of {sorted(COMPLETE_ENDPOINT_PATHS)} "
            "with a host; userinfo, query, and fragment are forbidden, got "
            f"{endpoint_url!r}"
        )
    check_media_capability_endpoint_shape(backend_key, media_capability, parsed.path)
    return parsed.scheme


def check_one_serving_backend_per_capability(
    lane: str, backends: Sequence[ModelBifrostLaneBackendBinding]
) -> None:
    """Refuse a lane where two serving backends declare the same capability.

    Raises:
        BifrostLaneMediaCapabilityAmbiguousError: naming every such backend.
    """
    for capability in EnumBifrostLaneMediaCapability:
        serving = [
            b.backend_key
            for b in backends
            if b.serving and b.media_capability is capability
        ]
        if len(serving) > 1:
            raise BifrostLaneMediaCapabilityAmbiguousError(
                f"lane {lane!r}: backends {serving} all serve media capability "
                f"{capability.value!r}; exactly one serving backend may declare it",
                lane=lane,
                media_capability=capability.value,
                backend_keys=serving,
            )


def resolve_media_capability[B: ModelBifrostLaneBackendBinding](
    lane: str, backends: Sequence[B], capability: str
) -> B:
    """Return the one serving backend that declares ``capability``.

    Raises:
        BifrostLaneMediaCapabilityUnknownError: ``capability`` is not one of the
            typed capabilities.
        BifrostLaneMediaCapabilityNotServingError: no serving backend declares
            it; names any declared-but-dark backend.
    """
    try:
        wanted = EnumBifrostLaneMediaCapability(capability)
    except ValueError:
        raise BifrostLaneMediaCapabilityUnknownError(
            f"lane {lane!r}: {str(capability)!r} is not a media capability; one "
            f"of {[c.value for c in EnumBifrostLaneMediaCapability]}",
            lane=lane,
            media_capability=str(capability),
        ) from None
    declared = [b for b in backends if b.media_capability is wanted]
    serving = [b for b in declared if b.serving]
    if not serving:
        dark = [b.backend_key for b in declared]
        raise BifrostLaneMediaCapabilityNotServingError(
            f"lane {lane!r}: no serving backend declares media capability "
            f"{wanted.value!r}"
            + (f"; declared but not serving: {dark}" if dark else ""),
            lane=lane,
            media_capability=wanted.value,
            backend_keys=dark,
        )
    return serving[0]


__all__ = [
    "COMPLETE_ENDPOINT_PATHS",
    "check_media_capability_endpoint_shape",
    "check_one_serving_backend_per_capability",
    "resolve_media_capability",
    "validate_binding_endpoint",
]
