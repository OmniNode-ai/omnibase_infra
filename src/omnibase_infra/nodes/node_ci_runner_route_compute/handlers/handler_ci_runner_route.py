# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerCIRunnerRoute — chooses where one CI run's jobs execute.

Canonical definition-B handler: the dispatch entrypoint is the single
``handle(request) -> response`` method, taking ``ModelCIRunnerRouteRequest`` and
returning ``ModelCIRunnerRouteDecision``. No envelope import, no ``Plugin*``
base, no I/O -- every observation is probed at the boundary and handed in, so
the decision is a pure function that can be replayed from its own record.

THE DECISION IS AN ORDERED ELIMINATION, AND EVERY STEP CAN ONLY MOVE THE ANSWER
TOWARD HOSTED:

  S0  the calling workflow is on the hosted list      -> hosted
  S1  fork / untrusted pull-request event             -> hosted (public labels)
  S2  the seam names no fleet label                   -> the seam, VERBATIM
  S3  any probe error, of any class                   -> hosted
  S4  fleet saturated (idle floor or busy fraction)   -> hosted
  S5  fleet too small to trust at all                 -> hosted
  S6  lab reading missing / stale / loaded / starved  -> hosted
  S7  otherwise                                       -> the seam labels

Two guards then run on the ANSWER, in this order, and the order is
load-bearing:

  G1  never-widen: a fleet label may appear only when the ceiling carried one.
      A violation degrades to hosted rather than raising.
  G2  visibility: a PRIVATE repository is never placed on a hosted runner. It
      runs AFTER G1 precisely because G1 emits hosted, so running it earlier
      would leave the one path that ignores the elimination able to break the
      ruling.

WHY GUARDS ON THE ANSWER RATHER THAN INSIDE THE BRANCHES. There are ten return
points. A check placed inside one defends that one. An earlier build asserted
never-widen from inside the capacity branch against the ceiling itself, which
is a tautology -- it compared the ceiling with itself and never looked at what
was returned, and a deliberately injected widening bug passed it.

FAIL-CLOSED. Nothing here raises: a typed request cannot be malformed, and
every unusable observation is already modelled as ``ok=False`` with a named
class, which resolves to hosted. The boundary that performs the probes keeps
its own catch so a crash there still yields a usable placement.

Ticket: OMN-18412 (the node); OMN-18031 (the routing decision this re-homes)
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_decision import (
    EnumCIRunnerRouteDecision,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_force import (
    EnumCIRunnerRouteForce,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_reason import (
    EnumCIRunnerRouteReason,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_visibility import (
    EnumCIRunnerRouteVisibility,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_decision import (
    ModelCIRunnerRouteDecision,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_evidence import (
    ModelCIRunnerRouteEvidence,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_request import (
    ModelCIRunnerRouteRequest,
)

logger = logging.getLogger(__name__)

SELF_HOSTED_LABEL = "self-hosted"


def parse_labels(raw: str) -> tuple[str, ...] | None:
    """Parse a runner-variable JSON array. ``None`` on anything unusable.

    An unusable seam is a config fault, not a licence to pick: the caller turns
    ``None`` into a hosted placement with a named reason.
    """
    import json

    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(item, str) for item in parsed):
        return None
    return tuple(str(item) for item in parsed)


def is_fleet(labels: tuple[str, ...]) -> bool:
    return SELF_HOSTED_LABEL in labels


class HandlerCIRunnerRoute:
    """Pure placement decision for one CI run."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: compute handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute, no external I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelCIRunnerRouteRequest) -> ModelCIRunnerRouteDecision:
        """Choose the label set this run's jobs execute on."""
        decided_at = datetime.now(UTC).isoformat()
        raw = self._eliminate(request, decided_at)
        checked = self._guard_never_widens(raw, request, decided_at)
        return self._guard_visibility(checked, request, decided_at)

    # -- the elimination ---------------------------------------------------

    def _eliminate(
        self, request: ModelCIRunnerRouteRequest, decided_at: str
    ) -> ModelCIRunnerRouteDecision:
        policy = request.policy
        hosted = tuple(policy.hosted_labels)
        ceiling = parse_labels(request.seam_json)

        if ceiling is None:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.PROBE_ERROR,
                reason_detail="seam_unparseable",
            )

        # S0 -- reasons that outrank capacity entirely.
        if request.workflow_path in request.hosted_workflows:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.POLICY_ALLOWLIST,
            )

        # S1 -- fork isolation. INVIOLABLE, and deliberately ahead of every
        # capacity signal and of the operator override: untrusted code never
        # reaches self-hosted compute at any idle level, and no force value
        # may buy its way past this.
        is_fork = (
            request.github_event == "pull_request"
            and request.head_repo != request.repository
        )
        if is_fork or request.github_event == "pull_request_target":
            public = parse_labels(request.public_json) or hosted
            if is_fleet(public):
                # A misconfigured public variable can never widen a fork onto
                # the fleet; the declared hosted labels win.
                public = hosted
            return self._result(
                request,
                decided_at,
                runs_on=public,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.FORK_ISOLATION,
            )

        # S2 -- the seam is the CEILING. If it names no fleet label there is
        # nothing to decide: return it verbatim, not normalised, so a seam
        # naming another hosted image is honoured.
        if not is_fleet(ceiling):
            return self._result(
                request,
                decided_at,
                runs_on=ceiling,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.SEAM_CEILING_HOSTED,
            )

        # The operator override, after the two rules it may not cross.
        if request.force is EnumCIRunnerRouteForce.HOSTED:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.FORCED_HOSTED,
            )
        if request.force is EnumCIRunnerRouteForce.FLEET:
            return self._result(
                request,
                decided_at,
                runs_on=ceiling,
                decision=EnumCIRunnerRouteDecision.SELF_HOSTED,
                reason=EnumCIRunnerRouteReason.FORCED_FLEET,
            )

        # S3 -- anything that cannot PROVE capacity is hosted.
        fleet = request.fleet
        if not fleet.ok:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.PROBE_ERROR,
                reason_detail=fleet.error or "unknown",
            )
        if fleet.online is None or fleet.busy is None:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.PROBE_ERROR,
                reason_detail="unexpected_shape",
            )

        online, busy = fleet.online, fleet.busy
        idle = online - busy
        busy_fraction = (busy / online) if online else 1.0

        # S4 -- saturation, on either independent threshold. The idle floor is
        # HEADROOM, not a capacity match; the busy fraction sits well below the
        # observed peak because a threshold near it would flap run to run.
        if idle < policy.min_idle_runners or busy_fraction >= policy.max_busy_fraction:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.FLEET_SATURATED,
                idle=idle,
                busy_fraction=busy_fraction,
            )

        # S5 -- a fleet too small to trust at all, as a fraction of the
        # declared inventory so the floor tracks a resize instead of going
        # stale against it.
        degraded_floor = math.ceil(
            request.fleet_expected_count * policy.min_online_fraction
        )
        if online < degraded_floor:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.FLEET_DEGRADED,
                idle=idle,
                busy_fraction=busy_fraction,
            )

        # S6 -- the lab half, freshness-bounded. Stale is unknown, and unknown
        # is hosted; never "assume ample".
        lab = request.lab
        if not lab.ok:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.LAB_UNKNOWN,
                idle=idle,
                busy_fraction=busy_fraction,
                lab_error=lab.error or "unknown",
            )
        if (
            lab.age_seconds is None
            or lab.age_seconds > policy.lab_record_max_age_seconds
        ):
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.LAB_UNKNOWN,
                idle=idle,
                busy_fraction=busy_fraction,
                lab_error="stale",
            )
        if not lab.hosts:
            return self._result(
                request,
                decided_at,
                runs_on=hosted,
                decision=EnumCIRunnerRouteDecision.HOSTED,
                reason=EnumCIRunnerRouteReason.LAB_UNKNOWN,
                idle=idle,
                busy_fraction=busy_fraction,
                lab_error="no_hosts",
            )
        for host in lab.hosts:
            # Load ranks, memory ADMITS: a host at a tenth of its load ceiling
            # with 2.5 GiB free is the target that cost an earlier landing
            # hours of out-of-memory kills.
            if host.ratio > policy.max_lab_load_ratio:
                return self._result(
                    request,
                    decided_at,
                    runs_on=hosted,
                    decision=EnumCIRunnerRouteDecision.HOSTED,
                    reason=EnumCIRunnerRouteReason.LAB_SATURATED,
                    idle=idle,
                    busy_fraction=busy_fraction,
                    lab_age_seconds=lab.age_seconds,
                    lab_host=host.label,
                )
            if host.free_mem_mib < policy.min_lab_free_mem_mib:
                return self._result(
                    request,
                    decided_at,
                    runs_on=hosted,
                    decision=EnumCIRunnerRouteDecision.HOSTED,
                    reason=EnumCIRunnerRouteReason.LAB_SATURATED,
                    idle=idle,
                    busy_fraction=busy_fraction,
                    lab_age_seconds=lab.age_seconds,
                    lab_host=host.label,
                )

        # S7 -- capacity is available and the seam permits it.
        return self._result(
            request,
            decided_at,
            runs_on=ceiling,
            decision=EnumCIRunnerRouteDecision.SELF_HOSTED,
            reason=EnumCIRunnerRouteReason.CAPACITY_AVAILABLE,
            idle=idle,
            busy_fraction=busy_fraction,
            lab_age_seconds=lab.age_seconds,
        )

    # -- G1: never widen ---------------------------------------------------

    def _guard_never_widens(
        self,
        decision: ModelCIRunnerRouteDecision,
        request: ModelCIRunnerRouteRequest,
        decided_at: str,
    ) -> ModelCIRunnerRouteDecision:
        hosted = tuple(request.policy.hosted_labels)
        ceiling = parse_labels(request.seam_json) or ()
        returned = set(decision.runs_on)
        widened = ""
        if returned != set(hosted):
            if not returned <= set(ceiling):
                widened = (
                    f"routing widened beyond the seam ceiling: "
                    f"returned={list(decision.runs_on)} ceiling={list(ceiling)}"
                )
            elif SELF_HOSTED_LABEL in returned and SELF_HOSTED_LABEL not in ceiling:
                widened = f"routing invented a fleet label: ceiling={list(ceiling)}"
        if not widened:
            return decision
        return decision.model_copy(
            update={
                "runs_on": hosted,
                "decision": EnumCIRunnerRouteDecision.HOSTED,
                "reason": EnumCIRunnerRouteReason.NEVER_WIDEN_VIOLATION,
                "reason_detail": "",
                "decided_at": decided_at,
                "evidence": decision.evidence.model_copy(update={"violation": widened}),
            }
        )

    # -- G2: a private repository is never placed on a hosted runner --------

    def _guard_visibility(
        self,
        decision: ModelCIRunnerRouteDecision,
        request: ModelCIRunnerRouteRequest,
        decided_at: str,
    ) -> ModelCIRunnerRouteDecision:
        visibility = request.visibility
        if visibility is EnumCIRunnerRouteVisibility.PUBLIC:
            return decision
        if is_fleet(decision.runs_on):
            return decision

        policy = request.policy
        ceiling = parse_labels(request.seam_json) or ()
        reversible = decision.reason.value in policy.capacity_downgrade_reasons

        if reversible and is_fleet(ceiling):
            # The fleet being busy is not a reason to break the ruling; the job
            # waits for a runner instead.
            reason = (
                EnumCIRunnerRouteReason.PRIVATE_REPO_NO_HOSTED_DOWNGRADE
                if visibility is EnumCIRunnerRouteVisibility.PRIVATE
                else EnumCIRunnerRouteReason.VISIBILITY_UNKNOWN_NO_HOSTED_DOWNGRADE
            )
            return decision.model_copy(
                update={
                    "runs_on": ceiling,
                    "decision": EnumCIRunnerRouteDecision.SELF_HOSTED,
                    "reason": reason,
                    "reason_detail": "",
                    "decided_at": decided_at,
                    "evidence": decision.evidence.model_copy(
                        update={"downgrade_refused_from": decision.reason_wire}
                    ),
                }
            )

        if visibility is EnumCIRunnerRouteVisibility.UNKNOWN:
            # UNKNOWN NEVER REFUSES, deliberately and asymmetrically. Every
            # public repository's ceiling reads hosted today, so refusing here
            # would turn one transient metadata read failure into a fleet-wide
            # outage. The misconfigured-private case is covered statically by
            # the exported private-repo placement gate, which reads visibility
            # itself.
            return decision

        # Hosted is the only placement the policy allows, and this repository
        # may not be placed hosted. Refusing is the honest answer: the
        # alternatives are untrusted code on the fleet, or a job queued forever
        # against capacity the repository may not use, which reads as a mystery
        # rather than as a rule.
        return decision.model_copy(
            update={
                "runs_on": (),
                "decision": EnumCIRunnerRouteDecision.BLOCKED,
                "reason": EnumCIRunnerRouteReason.PRIVATE_REPO_HOSTED_FORBIDDEN,
                "reason_detail": decision.reason_wire,
                "decided_at": decided_at,
            }
        )

    # -- record construction ------------------------------------------------

    def _result(
        self,
        request: ModelCIRunnerRouteRequest,
        decided_at: str,
        *,
        runs_on: tuple[str, ...],
        decision: EnumCIRunnerRouteDecision,
        reason: EnumCIRunnerRouteReason,
        reason_detail: str = "",
        idle: int | None = None,
        busy_fraction: float | None = None,
        lab_age_seconds: int | None = None,
        lab_error: str = "",
        lab_host: str = "",
    ) -> ModelCIRunnerRouteDecision:
        return ModelCIRunnerRouteDecision(
            runs_on=runs_on,
            decision=decision,
            reason=reason,
            reason_detail=reason_detail,
            policy_version=request.policy.policy_version,
            decided_at=decided_at,
            evidence=ModelCIRunnerRouteEvidence(
                github_event=request.github_event,
                repository=request.repository,
                workflow_path=request.workflow_path,
                seam_json=request.seam_json,
                visibility=request.visibility,
                fleet_online=request.fleet.online,
                fleet_busy=request.fleet.busy,
                idle=idle,
                busy_fraction=(
                    round(busy_fraction, 4) if busy_fraction is not None else None
                ),
                lab_age_seconds=lab_age_seconds,
                lab_error=lab_error,
                lab_host=lab_host,
            ),
        )


__all__ = ["HandlerCIRunnerRoute", "is_fleet", "parse_labels"]
