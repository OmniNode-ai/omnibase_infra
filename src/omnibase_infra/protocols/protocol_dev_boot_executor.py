# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host executor boundary for the dev boot effect."""

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol


class ProtocolDevBootExecutor(Protocol):
    def dev_boot_config_hashes(self, services: Sequence[str]) -> dict[str, str]: ...
    def verify_dev_boot_recovery(
        self, *, on_phase_update: Callable[[StrEnum, StrEnum], None]
    ) -> None: ...
