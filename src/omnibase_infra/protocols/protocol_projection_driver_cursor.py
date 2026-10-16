# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed DB-API callable attributes used by the projection statement bridge."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol


class ProtocolProjectionDriverCursor(Protocol):
    """Describe the bound driver methods at the raw SQL boundary."""

    execute: Callable[[object, Mapping[str, object]], object]
    fetchall: Callable[[], Sequence[object]]
