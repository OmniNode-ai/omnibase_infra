# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The preflight seam the dev roll's failure announcement wraps (OMN-24048)."""

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol


class ProtocolDevPreflightRunner(Protocol):
    """What the dev roll's preflight step needs of the deploy executor."""

    def preflight(
        self, on_phase_update: Callable[[StrEnum, StrEnum], None]
    ) -> None: ...
