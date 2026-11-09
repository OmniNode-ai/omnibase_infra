# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Checks ``onex delegate`` applies to its value flags before building a request.

``--backend-id`` (OMN-19124) and ``--model`` (OMN-20844) are normalised here and
an empty value is refused. Moved unchanged out of ``omnibase_infra.cli.cli_delegate``
(OMN-23164), which is a baselined file that only shrinks.
"""

from __future__ import annotations

__all__ = ["validate_backend_pin", "validate_model_choice"]


def validate_backend_pin(backend_id: str | None) -> str | None:
    """Normalise ``--backend-id`` and refuse an empty one (OMN-19124).

    Whitespace-only is refused rather than normalised to ``None``: a pin the
    caller believes they set, silently dropped, walks the cheapest-first
    ladder and answers — the exact silence this flag exists to remove.

    The id itself is deliberately NOT validated against a list of known
    backends. The routing contract is the source of truth for what backends
    exist and the routing authority already raises loudly on an unresolvable
    pin (``_resolve_initial_backend``: "a resolution failure on a caller's
    EXPLICIT pin means the caller asked for a backend that doesn't exist, and
    hiding that behind a silent fallback would defeat the whole point of
    pinning"). A second, CLI-side copy of that vocabulary is a table that
    goes stale and starts refusing backends the contract declares.
    """
    if backend_id is None:
        return None
    pinned = backend_id.strip()
    if not pinned:
        raise ValueError(
            "--backend-id was given an empty value. Pass a backend id declared "
            "by the routing contract, or omit the flag to use the "
            "cheapest-first tier_order walk."
        )
    return pinned


def validate_model_choice(model: str | None) -> str | None:
    """Normalise ``--model`` and refuse an empty choice (OMN-20844)."""
    if model is None:
        return None
    chosen = model.strip()
    if not chosen:
        raise ValueError(
            "--model was given an empty value. Pass a model id for the "
            "customer's own provider key, or omit the flag to use the model "
            "stored with the key."
        )
    return chosen
