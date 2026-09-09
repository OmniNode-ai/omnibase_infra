#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ONE rule that decides whether a lane refresh may destroy the lane [OMN-18061].

Both ``refresh_dev_lane.sh`` and ``refresh_stability_lane.sh`` end a failed
refresh with the same question: *may I recreate this lane's containers?* Until
now they answered it separately. The dev lane answered it correctly as of
OMN-16729 (``f6c182cc``) in a 45-line inline bash block; the stability lane --
the surface every live prod grant's ``stability-proven`` premise resolves from
-- still answered "yes, on any health-gate FAIL at all", which is the exact
trigger that destroyed the dev lane at 2026-09-08T18:48:41Z.

Two implementations of one rule is how the second lane keeps the first lane's
bug after the first lane is fixed. So the rule lives here, once, as a pure
function over the gate's own JSON, and both scripts call it. The lane that
wrote the dev-lane block named this extraction as its residual.

The distinction the rule turns on
--------------------------------

A rollback RECREATES containers. That repairs exactly one class of fault: the
lane is not serving. So it is gated on the dimensions that say whether the lane
is serving --

* ``health_ok``      -- the runtime's own ``/health`` verdict,
* ``manifest_ok``    -- the contract manifest is being served above its floor,
* ``cluster_healthy``-- the broker is up,
* ``core_services_running`` -- every core container exists and is ``running``,
* ``errors == []``   -- the gate could actually run its probes,

-- and on nothing else.

The provenance dimensions -- which revision label the running containers carry,
whether any image digest changed, whether the tracked clones moved forward --
say what the lane is RUNNING, never whether it is UP. No container recreate
repairs any of them. A refresh that fails on provenance alone has found a real
problem and must exit non-zero, but the lane it found the problem on is serving
and must be left exactly as it was.

Fail-closed on the probe errors is deliberate and is not symmetric with the
rest: an unseen lane is never treated as a healthy one.

Not a receipt reader
--------------------

This module reads the ``health_gate`` block a gate produced, not a receipt
file. Both callers already have that block on disk when they ask. Keeping the
input at the gate-report boundary is what lets the recorded receipts of the
2026-09-08 failures be replayed through the real decision object as fixtures,
with no live lane and no mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Refresh landed: the gate passed on every dimension and ancestry moved
#: forward. No rollback, exit 0.
RESULT_SUCCESS = "SUCCESS"

#: Refresh did NOT land the ref it intended, but the lane is serving on every
#: health dimension. The failing dimensions are build provenance only, which no
#: container recreate repairs. Report it, exit non-zero, touch nothing.
RESULT_FAILED_BUILD_PROVENANCE = "FAILED_BUILD_PROVENANCE"

#: The lane is not healthy. A recreate is warranted; the caller performs it and
#: then re-verifies, replacing this with its own post-rollback result.
RESULT_ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"

#: The refresh failed on a branch with nothing to roll back to (the cold-aware
#: path). Never destructive here -- the caller stops and reports.
RESULT_FAILED_NO_ROLLBACK_TARGET = "FAILED_NO_ROLLBACK_TARGET"

#: Recorded in the receipt whenever the rollback branch was deliberately not
#: taken, so a reader can tell a suppressed-by-design rollback from one that
#: never got the chance to run.
SUPPRESSED_REASON_PROVENANCE_ONLY = (
    "lane healthy on every health dimension; failing dimensions are build "
    "provenance only, which no container recreate repairs (OMN-16729/OMN-18061)"
)


def _flag(gate: Mapping[str, Any], key: str) -> bool:
    """Read a gate boolean, treating anything that is not ``True`` as False.

    A missing key is False, not an error: a gate that did not report a
    dimension has not proven it, and this decides whether to destroy a lane.
    """
    return gate.get(key) is True


def _names(gate: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = gate.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class LaneRollbackDecision:
    """What the refresh should do about a lane, and the reasons for it.

    Attributes:
        result: One of the ``RESULT_*`` constants.
        lane_is_healthy: Every health dimension held.
        rollback_warranted: The caller may recreate containers. True only when
            a health dimension failed AND a rollback target exists.
        unhealthy_dimensions: Named health dimensions that failed, e.g.
            ``core_services_running=false[runtime-effects=created]``.
        provenance_failures: Named provenance dimensions that failed.
        suppressed_reason: Why the rollback branch was not taken, or ``None``
            when the branch was not reached.
    """

    result: str
    lane_is_healthy: bool
    rollback_warranted: bool
    unhealthy_dimensions: tuple[str, ...]
    provenance_failures: tuple[str, ...]
    suppressed_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "result": self.result,
            "lane_is_healthy": self.lane_is_healthy,
            "rollback_warranted": self.rollback_warranted,
            "unhealthy_dimensions": list(self.unhealthy_dimensions),
            "provenance_failures": list(self.provenance_failures),
            "suppressed_reason": self.suppressed_reason,
        }


def health_dimension_failures(gate: Mapping[str, Any]) -> tuple[str, ...]:
    """Name every LANE-HEALTH dimension the gate reported as failing.

    These, and only these, justify a destructive recreate.
    """
    failures: list[str] = []
    if not _flag(gate, "health_ok"):
        failures.append("health_ok=false")
    if not _flag(gate, "manifest_ok"):
        failures.append("manifest_ok=false")
    if not _flag(gate, "cluster_healthy"):
        failures.append("cluster_healthy=false")
    if not _flag(gate, "core_services_running"):
        # A core service that is not RUNNING is a lane-health fact, not a
        # provenance one: this is the depends_on-stranded ``State=created``
        # container the 2026-09-08 rollback left behind, and a lane missing a
        # core container SHOULD be repaired by the recreate.
        not_running = ",".join(_names(gate, "core_services_not_running"))
        failures.append(f"core_services_running=false[{not_running}]")
    errors = gate.get("errors")
    error_count = len(errors) if isinstance(errors, list) else 1
    if error_count:
        # Probe errors mean the gate could not SEE the lane. Fail closed: an
        # unseen lane is never treated as a healthy one. Deliberately NOT
        # symmetric with the provenance side.
        failures.append(f"gate_errors={error_count}")
    return tuple(failures)


def provenance_dimension_failures(
    gate: Mapping[str, Any], *, ancestry_ok: bool
) -> tuple[str, ...]:
    """Name every BUILD-PROVENANCE dimension that failed.

    Reported and exited on, never recreated for.
    """
    failures: list[str] = []
    if not _flag(gate, "revision_readback_ok"):
        failures.append("revision_readback_ok=false")
    if _flag(gate, "require_digest_change") and not _flag(gate, "digest_changed"):
        failures.append("digest_changed=false")
    if not ancestry_ok:
        failures.append("merge_base_is_ancestor=false")
    return tuple(failures)


def decide_lane_rollback(
    gate: Mapping[str, Any],
    *,
    ancestry_ok: bool,
    branch: str,
) -> LaneRollbackDecision:
    """Decide what a failed lane refresh may do to the lane.

    Args:
        gate: The health-gate report, as the gate's own ``--json`` emits it.
        ancestry_ok: Whether the refresh proved forward progress. A provenance
            fact, carried separately because the gate does not compute it.
        branch: ``warm`` when a preflight rollback anchor exists for this run,
            anything else when there is nothing to roll back to.

    Returns:
        The decision, with every failing dimension named.
    """
    unhealthy = health_dimension_failures(gate)
    provenance = provenance_dimension_failures(gate, ancestry_ok=ancestry_ok)
    lane_is_healthy = not unhealthy
    overall = str(gate.get("overall") or "INFRA_ERROR")

    if overall == "PASS" and ancestry_ok:
        return LaneRollbackDecision(
            result=RESULT_SUCCESS,
            lane_is_healthy=lane_is_healthy,
            rollback_warranted=False,
            unhealthy_dimensions=unhealthy,
            provenance_failures=provenance,
            suppressed_reason=None,
        )

    if branch != "warm":
        # Cold-aware path: there is no preflight anchor, so there is nothing to
        # roll back TO. Never destructive.
        return LaneRollbackDecision(
            result=RESULT_FAILED_NO_ROLLBACK_TARGET,
            lane_is_healthy=lane_is_healthy,
            rollback_warranted=False,
            unhealthy_dimensions=unhealthy,
            provenance_failures=provenance,
            suppressed_reason="no preflight rollback anchor exists on this branch",
        )

    if lane_is_healthy:
        return LaneRollbackDecision(
            result=RESULT_FAILED_BUILD_PROVENANCE,
            lane_is_healthy=True,
            rollback_warranted=False,
            unhealthy_dimensions=(),
            provenance_failures=provenance,
            suppressed_reason=SUPPRESSED_REASON_PROVENANCE_ONLY,
        )

    return LaneRollbackDecision(
        result=RESULT_ROLLBACK_REQUIRED,
        lane_is_healthy=False,
        rollback_warranted=True,
        unhealthy_dimensions=unhealthy,
        provenance_failures=provenance,
        suppressed_reason=None,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI shim: read a gate report, print the decision as JSON.

    This is how both shell scripts reach the function. They pass the gate JSON
    they already wrote; nothing here reads a lane, a docker socket or a clock.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether a failed lane refresh may recreate the lane's "
            "containers. Reads a health-gate report; mutates nothing."
        )
    )
    parser.add_argument(
        "--gate-json",
        required=True,
        help="Path to the health-gate report the refresh just produced.",
    )
    parser.add_argument(
        "--ancestry-ok",
        required=True,
        choices=("true", "false"),
        help="Whether the refresh proved forward progress (a provenance fact).",
    )
    parser.add_argument(
        "--branch",
        required=True,
        help="'warm' when a preflight rollback anchor exists, else e.g. 'cold'.",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.gate_json, encoding="utf-8") as handle:
            gate = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        # A gate report this module cannot read is not evidence of a healthy
        # lane. Fail closed to the same shape an unreadable gate produces.
        gate = {"overall": "INFRA_ERROR", "errors": [f"unreadable gate report: {exc}"]}
    if not isinstance(gate, dict):
        gate = {"overall": "INFRA_ERROR", "errors": ["gate report is not an object"]}

    decision = decide_lane_rollback(
        gate,
        ancestry_ok=args.ancestry_ok == "true",
        branch=args.branch,
    )
    json.dump(decision.to_dict(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
