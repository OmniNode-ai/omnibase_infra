# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The phase a planned step belongs to; teardown and residue always run.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofStepPhase(StrEnum):
    """The phase a planned step belongs to; teardown and residue always run."""

    SETUP = "setup"
    PROVE = "prove"
    TEARDOWN = "teardown"
    RESIDUE = "residue"


__all__ = ["EnumLabProofStepPhase"]
