# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The defect classes the contract graph can prove statically (OMN-18013)."""

from __future__ import annotations

from typing import Literal

DefectClass = Literal[
    "ORPHANED_CONSUMER",
    "ORPHANED_PRODUCER",
    "DECLARED_BUT_UNWIRED",
    "DISCONNECTED_SUBGRAPH",
]

__all__ = ["DefectClass"]
