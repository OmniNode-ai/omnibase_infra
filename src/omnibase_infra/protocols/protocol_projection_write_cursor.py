# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Minimal typed cursor used by the in-process projection write capability."""

from collections.abc import Mapping, Sequence
from typing import Protocol


class ProtocolProjectionWriteCursor(Protocol):
    """A bound SQL execution cursor; it owns neither commit nor publication."""

    def run_guarded_derived_update(
        self,
        statement: object,
        parameters: Mapping[str, object],
        *,
        returning: Sequence[str],
    ) -> list[dict[str, object]]:
        """Execute the guarded update and return its bound row columns."""
        ...
