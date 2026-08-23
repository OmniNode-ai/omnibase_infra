# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Validator-catch severity enum for savings-correlation heuristics.

Ported unchanged from the legacy ServiceSavingsEstimator
(services/observability/savings_estimation/consumer.py, deleted alongside
the OMN-16293 correlation rewrite) — the heuristic severity classification is
not part of the architecture change, only its state source is.

Related Tickets:
    - OMN-7494: Heuristic savings from validator catches
    - OMN-16293: Postgres-projection-backed correlation rewrite
"""

from __future__ import annotations

from enum import StrEnum


class EnumCatchSeverity(StrEnum):
    """Severity levels for validator catches."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


__all__: list[str] = ["EnumCatchSeverity"]
