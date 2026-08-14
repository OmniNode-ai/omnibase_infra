# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composite per-runner readiness state (OMN-15255).

Distinct from ``EnumRunnerFleetHealthState``, which is a *precedence* chain --
the first matching bad state wins and the remaining signals are never
evaluated. Readiness is a *conjunction*: a runner is READY only when every
readiness signal passes, which is what friction F-04 asks for ("'online'
currently means registered, not ready to execute the governed workload").

Tri-state on purpose. ``UNKNOWN`` is not a polite ``READY``: a runner whose
signals could not be probed does not count as capacity, and it is also not
``NOT_READY`` -- a missing probe is not evidence of failure, and treating it
as one would quarantine the whole fleet the first time an SSH probe blips.
"""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerReadinessState(StrEnum):
    """Composite readiness verdict for one runner."""

    READY = "ready"
    """Every readiness signal PASSed. The runner may be routed governed work."""

    NOT_READY = "not_ready"
    """At least one readiness signal FAILed. Quarantined; may be bounce-eligible."""

    UNKNOWN = "unknown"
    """No signal FAILed but at least one could not be determined. Not routable,
    not quarantined -- absence of evidence is not evidence of failure."""


__all__ = ["EnumRunnerReadinessState"]
