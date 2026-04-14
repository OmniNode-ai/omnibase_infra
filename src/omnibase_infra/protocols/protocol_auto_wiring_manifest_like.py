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
    """

    total_discovered: int
    total_errors: int

    def all_subscribe_topics(self) -> Iterable[str]:
        pass


__all__: list[str] = ["ProtocolAutoWiringManifestLike"]
