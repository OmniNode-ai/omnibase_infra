# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of the correlation-scoped quarantine-sink check (OMN-16773)."""

from __future__ import annotations

from enum import StrEnum


class EnumQuarantineCheckStatus(StrEnum):
    """Whether the quarantine leg ran, and what it saw.

    ``SKIPPED_NOT_CONFIGURED`` exists so a result can never imply a check
    that did not happen. "We looked and it was clean" and "we never
    looked" are different claims and the receipt must distinguish them.
    """

    # Scanned the tail of the quarantine topic; this run's correlation id
    # was absent.
    CLEAN = "clean"
    # This run's correlation id was present in the quarantine sink.
    FOUND = "found"
    # No bootstrap servers supplied — the leg was deliberately not run.
    # NOT a pass, NOT a failure; an absence of evidence, recorded as one.
    SKIPPED_NOT_CONFIGURED = "skipped_not_configured"
    # Configured, attempted, and failed. Fails the run.
    ERROR = "error"


__all__ = ["EnumQuarantineCheckStatus"]
