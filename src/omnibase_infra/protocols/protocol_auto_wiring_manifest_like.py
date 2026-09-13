# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol matching the shape of ``ModelAutoWiringManifest``.

Defined here rather than importing ``ModelAutoWiringManifest`` directly so that
modules consuming this shape can avoid circular imports at parse time.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class ProtocolAutoWiringManifestLike(Protocol):
    """Minimal protocol for auto-wiring discovery manifests.

    Any object exposing ``total_discovered``, ``total_errors``, and
    ``all_subscribe_topics()`` satisfies this protocol, including
    ``ModelAutoWiringManifest``.

    The two counts are declared READ-ONLY (OMN-18324). As bare annotated
    attributes they were settable variables, and a Protocol with a settable
    member is satisfied only by an implementation that is ALSO settable — so
    ``ModelAutoWiringManifest``, which computes both as properties, did not
    structurally satisfy the protocol named after it. Nothing noticed while
    every call site already held the protocol type; the first caller to pass
    the concrete model got an ``arg-type`` error naming a mismatch that had
    been there since the protocol was written. Read-only accepts both shapes
    and nothing here ever assigns to either.
    """

    @property
    def total_discovered(self) -> int: ...

    @property
    def total_errors(self) -> int: ...

    def all_subscribe_topics(self) -> Iterable[str]:
        pass


__all__: list[str] = ["ProtocolAutoWiringManifestLike"]
