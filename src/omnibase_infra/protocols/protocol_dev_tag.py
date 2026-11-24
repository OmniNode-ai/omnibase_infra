# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Protocol


class ProtocolDevTag(Protocol):
    """One ``dev-<UTC stamp>-<sha12>`` tag on a repository's origin, as far as a rule reads it."""

    @property
    def sha(self) -> str: ...
