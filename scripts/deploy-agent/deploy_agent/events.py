# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event schemas for deploy agent. Strongly typed, frozen, standalone."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from deploy_agent.tracking_ref import load_tracking_remote_ref_from_env

TOPIC_REBUILD_REQUESTED = "onex.cmd.deploy.rebuild-requested.v1"
TOPIC_REBUILD_COMPLETED = "onex.evt.deploy.rebuild-completed.v1"
TOPIC_REBUILD_REJECTED = "onex.evt.deploy.rebuild-rejected.v1"
# Dead-letter target for a command record the agent cannot decode or validate
# (OMN-16442). Shape follows the org convention onex.dlq.<producer>.<category>.<version>.
# A record that lands here is one the agent has committed past: it can never be
# decoded by redelivery, and withholding the offset stalls every command behind
# it -- see deploy_agent.consumer for the full argument.
TOPIC_DEPLOY_COMMAND_DLQ = "onex.dlq.omnibase-infra.deploy-command.v1"


class DeployInProgressError(RuntimeError):
    """Raised when a second deploy arrives while one is already running."""


class Scope(StrEnum):
    FULL = "full"
    RUNTIME = "runtime"
    CORE = "core"


class EnumRuntimeLane(StrEnum):
    """Runtime deployment lane.

    Each lane maps to its own compose overlay, compose project, and runtime
    health ports (see ``deploy_agent.executor.lane_config_for``). ``prod``
    deploys a stability-proven image digest rather than rebuilding from a ref.
    """

    DEV = "dev"
    STABILITY_TEST = "stability-test"
    PROD = "prod"


class EnumSelfUpdateBoundary(StrEnum):
    """The job boundary a self-update is allowed to fire at (OMN-16442).

    Self-update pulls the agent's own clone and replaces the process image, so
    it may only run where no job is in flight. Between deploy phases is not
    such a place: on 2026-09-08 command
    ``8d0c861a-f91e-4ca2-954e-a073759dd39d`` re-execed after the seed phase and
    the replacement process published that same command as ``failed`` after
    logging ``Recovered 1 crashed job(s)``.

    ``PRE_ACCEPT``
        In the consumer, after a command has passed the signature, payload,
        lane-fence, busy and dedup checks and BEFORE ``job_store.accept``
        marks it started. Nothing is in flight, and the command's offset is
        rewound rather than committed, so the replacement process re-reads it
        and processes it once.

    ``POST_TERMINAL``
        In the agent, after the single-flight lock is released and the job's
        terminal status has been published. Deferring to here is what lets a
        deploy that starts on version X complete on version X.

    ``IDLE_HEARTBEAT``
        In the agent's poll loop, on the branch where no command arrived, at a
        bounded cadence and only when the job store reports nothing accepted,
        in progress, or awaiting a terminal publish (OMN-18200).

        The two boundaries above are both JOB-DRIVEN, and that is a
        circularity when the change to be picked up is a change to this agent.
        On 2026-09-14 the only merge that would have published a rebuild
        command was ``omnibase_infra#3520``, the fix to the agent's own lab
        overlay build -- and until ``#3522`` made ``scripts/deploy-agent/**`` a
        lane-state path, that merge published nothing, so no job arrived, so
        neither boundary was ever reached. An agent that nobody sends a job to
        could not pick up its own fix at all.

        The idle branch is not a job boundary in the OMN-16442 sense; it is the
        absence of one, which is the same guarantee arrived at from the other
        side. The in-flight checks are asserted rather than assumed because
        "the poll returned nothing" and "nothing is in flight" are different
        facts: a terminal result can still be queued for publish.
    """

    PRE_ACCEPT = "pre_accept"
    POST_TERMINAL = "post_terminal"
    IDLE_HEARTBEAT = "idle_heartbeat"


class BuildSource(StrEnum):
    WORKSPACE = "workspace"
    RELEASE = "release"


class Phase(StrEnum):
    PREFLIGHT = "preflight"
    GIT = "git"
    COMPOSE_GEN = "compose_gen"
    SEED = "seed"
    CORE = "core"
    RUNTIME = "runtime"
    VERIFICATION = "verification"
    PUBLISH = "publish"


class PhaseStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


# The phases a deploy actually executes, in pipeline order. PUBLISH is
# deliberately absent: it is the act of emitting the terminal event, so an event
# can never carry a settled verdict for it (OMN-18057). Used by the terminal
# reconciliation to decide which phases a raised deploy never reached.
DEPLOY_PHASE_ORDER: tuple[Phase, ...] = (
    Phase.PREFLIGHT,
    Phase.GIT,
    Phase.COMPOSE_GEN,
    Phase.SEED,
    Phase.CORE,
    Phase.RUNTIME,
    Phase.VERIFICATION,
)


SCOPE_SERVICES: dict[Scope, list[str]] = {
    Scope.CORE: ["postgres", "redpanda", "valkey"],
    Scope.RUNTIME: [
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        # OMN-18387: was missing here while scripts/deploy-runtime.sh's
        # RUNTIME_SERVICES has always carried it. A re-publish through the
        # deploy agent built a fresh projection-api image and reported the
        # runtime phase SUCCESS end to end, but neither the up-target list
        # nor container-verification (both derived from this same dict, see
        # _requested_services_for_up / verify_containers_up in executor.py)
        # ever named the service, so the running container stayed on its
        # previous image. See tests/unit/test_runtime_services_parity_omn18387.py
        # for the anti-drift test against the bash array.
        "projection-api",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "context-audit-consumer",
        "intelligence-migration",
        "intelligence-api",
        "omninode-contract-resolver",
        "autoheal",
    ],
    Scope.FULL: [],  # resolved as union of core + runtime
}


# OMN-18108: the runtime services the DEV lane declares and no other lane does.
#
# These live only in ``docker/docker-compose.dev-lane.yml``. Membership in
# ``SCOPE_SERVICES[Scope.RUNTIME]`` above would be FATAL, not merely wrong: the
# service name does not exist in the prod, stability-test or judge merged
# compose, so every deploy to those lanes would abort on `no such service`.
# They are a DEV-lane addendum, resolved by ``services_for_scope`` only when the
# caller names that lane.
#
# WHY THIS IS DECLARED HERE AND NOT PARSED FROM THE SHELL SCRIPT
# --------------------------------------------------------------
# ``scripts/deploy-runtime.sh`` carries the same eight names in its
# ``DEV_LANE_ONLY_RUNTIME_SERVICES`` array, and three existing tests parse that
# hand-written literal out of the script by regex. Reshaping the array into a
# file both sides read would break those tests and edit the sanctioned deploy
# path for a refactor's sake. The two declarations are instead bound
# MECHANICALLY and bidirectionally by
# ``tests/unit/test_dev_lane_only_scope_omn18108.py``, which parses the array
# and asserts set equality both ways -- an edit to either side alone is a red
# test, which is the property "single source of truth" was wanted for.
#
# The defect this closes, measured on the .201 dev lane 2026-09-10T00:45Z: the
# runtime family carried the deploy agent's own build
# ``4598a4358bd9f59528875b8b320b6cde54383fb1`` while all eight of these carried
# ``3461e4b0aeae`` from the previous day, ~35 infra commits behind. Not an
# intermittent miss -- the agent's scope could not reach them at all, and
# ``restart: unless-stopped`` keeps a stale image running and healthy, so
# nothing reported it.
# OMN-18438: the dev lane's omninode_cloud migration one-shots.
#
# THE SAME DEFECT AS OMN-18108, ONE LAYER OVER. omnibase_infra#3636 put
# `cloud-migration-files` and `cloud-migration` into
# DEV_LANE_ONLY_MIGRATION_SERVICES in scripts/deploy-runtime.sh and wired them
# into that script's migration preflight. They still never ran, because THIS
# AGENT DOES NOT INVOKE THAT SCRIPT -- every mention of deploy-runtime.sh in
# this package is a comment. An array expanded only inside that script is
# unreachable from the path that actually deploys the dev lane.
#
# Measured on the .201 dev lane after the 14:43Z governed rebuild (agent
# command ee3d2cc6, ref 3586e65dc, which carried those arrays): zero
# cloud-migration containers had EVER been created, omninode_cloud held 0
# tables against 78 in omnibase_infra, and 450 lines of deploy journal held 0
# cloud-migration mentions against 2 migration controls.
#
# WHY THESE ARE NOT IN services_for_scope()
# -----------------------------------------
# That function answers "what does a runtime deploy restart", and its result
# reaches both _compose_up and the build set. These two are one-shots on
# upstream images -- postgres:16 and the tag-referenced migrate image -- so
# there is nothing to build, and starting them beside the runtime family would
# run them unordered and leave the up-readback waiting on containers that are
# supposed to exit. They belong in the migration preflight, the phase that
# already exists for run-to-completion boot-order work.
#
# ORDER IS THE ORDERING. cloud-migration-files copies the corpus, its MANIFEST
# and that image's own manifest evaluator into the shared volume;
# cloud-migration then applies it. Every command on this path carries
# --no-deps, which is exactly what switches compose's depends_on off, so this
# sequence is the only thing sequencing the copy before the apply.
#
# ONE tuple here where bash carries two arrays: deploy-runtime.sh separates
# services from one-shots because its lane-agnostic set mixes in a keepalive
# (migration-gate). Every member of the dev-lane set is a one-shot, so a
# second tuple would be a second thing to drift rather than a distinction --
# and tests/unit/test_dev_lane_cloud_migrations_omn18438.py asserts the two
# bash arrays are equal so that stays true.
#
# Bound to the bash declaration by
# tests/unit/test_dev_lane_cloud_migrations_omn18438.py
# ::test_python_declaration_equals_the_bash_array, which PARSES the array out
# of deploy-runtime.sh rather than restating it, and carries a positive control
# so an empty parse cannot read as agreement.
DEV_LANE_ONLY_MIGRATION_SERVICES: tuple[str, ...] = (
    "cloud-migration-files",
    "cloud-migration",
)


DEV_LANE_ONLY_RUNTIME_SERVICES: tuple[str, ...] = (
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "infra-routing-decisions-consumer",
    "onex-api",
    # OMN-18114: the TENANT-domain projection carrier, for a reason that is the
    # OPPOSITE of the writers' reason above and is stated separately so the two
    # do not merge. That service IS declared in docker-compose.infra.yml, so
    # every lane resolves the name -- which is exactly why it cannot be
    # lane-agnostic: a prod or judge `up -d --no-deps tenant-projection-writer`
    # would SUCCEED and start the carrier on a lane that never opted into it,
    # defeating the compose profile that keeps it inert there. Membership here
    # scopes it to the lane whose overlay puts it in the `runtime` profile.
    #
    # It is the only process that owns the eight omnimarket contracts declaring
    # `runtime_profiles: [tenant-projection]`, so an agent scope that could not
    # reach it would leave those eight running an image the rest of the lane had
    # moved past -- the exact defect measured above, on the one service where
    # nothing else would ever notice.
    "tenant-projection-writer",
)

# OMN-18108: the members of the array above that carry an ``image:`` and no
# ``build:`` -- tag-referenced, not lane-built.
#
# ``onex-api`` resolves ``${ONEX_API_IMAGE}`` from the operator env file on the
# host; the image is built out of a different repository. So a governed deploy
# can RECREATE it, which is what it actually needs (a container that is never
# recreated never reads a new environment, and this one carries the lane's
# broker credentials and eleven fail-closed variables), but it CANNOT advance
# the tag. Stating the boundary here, and asserting it in the test, rather than
# leaving an operator to discover that a "successful" deploy left the tag where
# it was. Advancing it needs an image build plus an env repoint, and no
# sanctioned script does either.
DEV_LANE_ONLY_TAG_REFERENCED_SERVICES: frozenset[str] = frozenset({"onex-api"})

# The subset ``docker compose build`` can be handed. Derived, never a second
# hand-written list.
DEV_LANE_ONLY_BUILDABLE_SERVICES: tuple[str, ...] = tuple(
    service
    for service in DEV_LANE_ONLY_RUNTIME_SERVICES
    if service not in DEV_LANE_ONLY_TAG_REFERENCED_SERVICES
)


# OMN-18134: the DEV lane's gateway compose project, and the services in it.
#
# THIS LIST IS NOT A SECOND DEV_LANE_ONLY_RUNTIME_SERVICES. The two are kept
# apart because their compose semantics are OPPOSITE, and merging them would be
# fatal in the same way membership in the base runtime list would be fatal for
# the eight above.
#
# The eight above live in ``docker/docker-compose.dev-lane.yml``, which IS one
# of the DEV lane's ``compose_files``: they are handed to
# ``docker compose -p omnibase-infra`` and that project resolves them. These
# live in ``docker/docker-compose.gateway.yml`` under the compose project
# ``omninode-gateway``, which appears in NO lane's ``compose_files``. Handing
# ``gateway-forwarder`` to the omnibase-infra project aborts the runtime phase
# on `no such service`.
#
# So membership here means exactly one thing: a DEV deploy is RESPONSIBLE for
# this service. Which command deploys it is a separate question, answered by
# ``DeployExecutor._deploy_gateway_lane`` -- it calls the sanctioned
# ``scripts/deploy-gateway.sh``, which owns the gateway project's build, digest
# pin, host-file sync, rollback record and systemd reload. Every call site that
# builds a compose ARGUMENT list subtracts these via
# ``without_gateway_services``.
#
# Only the forwarder is listed. ``gateway-dns-bastion`` is a sidecar of the
# same project that ``deploy-gateway.sh`` builds and that compose starts via
# the forwarder's ``depends_on``; it is not independently addressable as a
# deploy target, and listing it would let a command name it alone and get the
# whole gateway lane deployed anyway.
GATEWAY_COMPOSE_PROJECT = "omninode-gateway"
DEV_LANE_GATEWAY_SERVICES: tuple[str, ...] = ("gateway-forwarder",)


def without_gateway_services(services: Iterable[str]) -> list[str]:
    """Return ``services`` minus anything the gateway compose project owns.

    Used at every site that builds an argument list for the ``omnibase-infra``
    compose project, so a service that is legitimately in DEV *scope* can never
    become a compose *argument* for a project that does not declare it.
    """
    return [s for s in services if s not in DEV_LANE_GATEWAY_SERVICES]


def gateway_services_in(services: Iterable[str]) -> list[str]:
    """Return the subset of ``services`` the gateway compose project owns."""
    return [s for s in services if s in DEV_LANE_GATEWAY_SERVICES]


def services_for_scope(
    scope: Scope, *, lane: EnumRuntimeLane | None = None
) -> list[str]:
    """Return the services a deploy of ``scope`` targets on ``lane``.

    ``lane`` defaults to ``None``, which resolves the lane-agnostic base list
    exactly as before. A caller that does not name a lane therefore never
    silently acquires dev-lane services, and prod/stability-test scope is
    byte-unchanged whether the lane is passed or not (OMN-18108 AC3, OMN-18134
    AC2).

    This is the list of what a deploy is RESPONSIBLE for, not the list of what
    is handed to any one compose project -- see ``DEV_LANE_GATEWAY_SERVICES``.
    """
    if scope == Scope.FULL:
        base = SCOPE_SERVICES[Scope.CORE] + SCOPE_SERVICES[Scope.RUNTIME]
    else:
        base = list(SCOPE_SERVICES[scope])
    if lane == EnumRuntimeLane.DEV and scope in (Scope.RUNTIME, Scope.FULL):
        return (
            base
            + list(DEV_LANE_ONLY_RUNTIME_SERVICES)
            + list(DEV_LANE_GATEWAY_SERVICES)
        )
    return base


class ModelHealthCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    endpoint: str
    status: Literal["pass", "fail"]
    latency_ms: int = 0


class ModelContainerResidue(BaseModel):
    """One service left in a non-running state by a deploy phase (OMN-18057).

    The 2026-09-08 runtime-phase kill left ``runtime-effects``,
    ``runtime-worker`` and ``omninode-contract-resolver`` in ``Created`` with
    :8086 down, and the terminal event said nothing about any of them. Residue
    is recorded whether or not per-container recovery then succeeded, because
    "recovered after the ceiling blew" and "came up first time" are different
    facts about the lane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    state: str
    exit_code: int | None = None
    recovered: bool = False


class EnumRecreateOutcome(StrEnum):
    """How a supervised deps-phase compose recreate ended (OMN-18692).

    FIVE values, not two, because "the ceiling blew" was recorded as one fact
    on 2026-09-18 and it is at least three different facts, each of which
    demands a different next action:

    * ``completed`` -- the command returned on its own. The only ending that
      says nothing about the ceiling.
    * ``deferred_host_contention`` -- the recreate was never STARTED, because
      the host stayed above the committed saturation threshold. THE LANE WAS
      NOT TOUCHED; this is the one outcome that guarantees that.
    * ``ended_lane_settled`` -- past the ceiling with every expected service
      running and compose still not returned. Ending the command here cannot
      destroy the lane, because nothing was mid-removal.
    * ``killed_mid_recreate_stalled`` -- past the ceiling, mid-recreate, and
      the lane's container state had not changed for the declared stall
      window. Read as wedged rather than slow.
    * ``killed_hard_upper_bound`` -- past the ceiling, mid-recreate, still
      changing, and out of budget. The only ending that cancels a LIVE
      recreate, and the caller must converge the deps immediately after it.

    The 2026-09-18 kill was the last shape, recorded with none of this
    vocabulary, so the agent's own log could not distinguish it from a slow
    deploy and the next reader had to reconstruct it from the dockerd journal.
    """

    COMPLETED = "completed"
    DEFERRED_HOST_CONTENTION = "deferred_host_contention"
    ENDED_LANE_SETTLED = "ended_lane_settled"
    KILLED_MID_RECREATE_STALLED = "killed_mid_recreate_stalled"
    KILLED_HARD_UPPER_BOUND = "killed_hard_upper_bound"


class ModelRecreateSupervision(BaseModel):
    """What the deps-phase ceiling did, and what it was measured against.

    Carried on the terminal event beside ``container_residue`` so a WAIT and a
    DEFERRAL are durable facts rather than journal lines someone has to go and
    find on the host. ``anchored_elapsed_seconds`` and ``elapsed_seconds`` are
    both present deliberately: their difference is how long the command spent
    queueing before it touched a container, which is the quantity the flat
    ceiling was unknowingly charging against the recreate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    phase: str
    outcome: EnumRecreateOutcome
    ceiling_seconds: int
    hard_upper_bound_seconds: int
    elapsed_seconds: float
    anchored_elapsed_seconds: float
    waited_past_ceiling_seconds: float
    deferred_seconds: float
    anchored_at_first_container_change: bool
    mid_recreate_at_decision: bool
    returncode: int | None = None
    budget_description: str = ""

    def describe(self) -> str:
        """One-line, log-ready statement of the ending and the wait it took."""
        parts = [
            f"phase {self.phase} {self.outcome.value}",
            f"ceiling {self.budget_description or f'{self.ceiling_seconds}s'}",
            f"elapsed {self.elapsed_seconds:.0f}s "
            f"(anchored {self.anchored_elapsed_seconds:.0f}s)",
        ]
        if self.deferred_seconds:
            parts.append(f"deferred {self.deferred_seconds:.0f}s before starting")
        if self.waited_past_ceiling_seconds:
            parts.append(
                f"WAITED {self.waited_past_ceiling_seconds:.0f}s past the ceiling "
                f"rather than cancelling a live recreate"
            )
        return "; ".join(parts)


class EnumVerifyRecreateOutcome(StrEnum):
    """How a post-deploy verification recreate ended (OMN-18640 AC7).

    FOUR values rather than a boolean, because "we bounced the container" is
    not one fact. The 2026-09-18 wedge was invisible for 97 minutes precisely
    because the agent's record could not distinguish a remedy that worked from
    one that was never attempted.

    * ``recovered`` -- the service was recreated and the SAME health probe
      then passed. The lane repaired itself.
    * ``still_failing`` -- recreated, re-probed, still failing. The job fails
      exactly as it did before this remedy existed; a second recreate is NOT
      attempted, because a remedy that did not work the first time is a
      diagnosis, not something to repeat.
    * ``recreate_failed`` -- the ``docker compose up --force-recreate`` itself
      exited non-zero or was killed. The container may be in any state and the
      probe was not re-run; this is a different fact from a recreate that ran
      and did not help.
    * ``recreate_timed_out`` -- the recreate command exceeded its own ceiling.
      Named apart from ``recreate_failed`` because a compose command killed
      mid-recreate is the shape that removed a lane on 2026-09-18 (OMN-18692),
      and reading it as a plain non-zero exit is what cost that incident its
      diagnosis.
    """

    RECOVERED = "recovered"
    STILL_FAILING = "still_failing"
    RECREATE_FAILED = "recreate_failed"
    RECREATE_TIMED_OUT = "recreate_timed_out"


class ModelComposeInvocation(BaseModel):
    """One ``docker compose`` command this deploy issued, as argv (OMN-18640).

    The agent logs a phase, a ceiling and an outcome; it has never logged the
    COMMAND. On 2026-09-18/19 the only way to read the argv of a live deploy
    was to sample the host's process table while the child was running, which
    is how the core leg's unconditional ``--force-recreate`` was finally
    observed rather than inferred from source. A flag is the difference
    between converging a lane and replacing it, so the flags a deploy actually
    used belong in its durable record and not in a process table that empties
    when the command exits.

    ``argv`` is the list handed to the kernel, verbatim. It is safe to record:
    every compose call this agent issues passes secrets through the
    environment, never on the command line, precisely because a command line
    is readable by every process on the host.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    phase: Phase
    argv: tuple[str, ...]


class ModelDepsConvergenceFinding(BaseModel):
    """One core dependency's declaration, compared against what is running.

    OMN-18640. The deps leg converges rather than force-recreates, so it
    replaces a dependency only when compose's own config hash says the
    declaration changed. That is the correct behaviour and it is also
    invisible: on 2026-09-19 the job that replaced the lane's broker differed
    from its five neighbours by nothing an operator could read except a phase
    duration, 77 seconds against 5 to 10. This record says WHICH service is
    about to be replaced and WHY, before it happens.

    ``running_config_hash`` and ``rendered_config_hash`` are the authority.
    They are the same value compose itself compares: the label
    ``com.docker.compose.config-hash`` on the live container, and the output
    of ``docker compose config --hash <service>`` for the render this deploy
    is about to apply. Verified equal on all three core services of the .201
    dev lane on 2026-09-19.

    ``changed_fields`` is an ACCOUNT, never the authority. It names which of a
    fixed, declared set of fields differ -- image, healthcheck, mounts,
    environment keys -- and it can be EMPTY while ``differs`` is true, because
    the hash covers fields outside that set. A reader must not conclude from
    an empty list that nothing changed; that is what ``differs`` is for.
    Environment is compared by KEY ONLY and only key names are ever recorded,
    because the values are the lane's broker and database credentials.

    ``unreadable_reason`` is non-empty when either hash could not be read -- an
    absent container, a render that failed. An unreadable comparison is
    reported as unreadable and the deploy proceeds; this record observes, it
    never gates. Refusing here would strand a declared change, because nothing
    in the fleet emits a deps-only scope for a deliberate refresh to route to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    lane: EnumRuntimeLane
    compose_project: str
    running_config_hash: str = ""
    rendered_config_hash: str = ""
    differs: bool
    changed_fields: tuple[str, ...] = ()
    unreadable_reason: str = ""

    def describe(self) -> str:
        """One line naming the service and what is about to happen to it."""
        if self.unreadable_reason:
            return (
                f"{self.service}: convergence effect UNKNOWN "
                f"({self.unreadable_reason}); proceeding"
            )
        if not self.differs:
            return f"{self.service}: declaration unchanged, will be left running"
        fields = (
            ", ".join(self.changed_fields)
            if self.changed_fields
            else ("no field in the compared set; the hash covers more than that set")
        )
        return (
            f"{self.service}: declaration CHANGED ({fields}) -- convergence "
            f"will REPLACE this container; running "
            f"{self.running_config_hash[:12]} -> rendered "
            f"{self.rendered_config_hash[:12]}"
        )


class ModelVerifyRecreate(BaseModel):
    """One runtime container this deploy recreated because its health failed.

    Carried on the terminal event beside ``container_residue`` and
    ``recreate_supervision``, for the same reason both of those are: a
    mutation this agent decided to perform on a lane is a durable fact, not a
    log line the next reader has to go and find on the host.

    ``compose_project`` is recorded rather than derived by a reader, because
    the one property this remedy must never violate is WHICH project it
    touched -- the governed stability-test, judge and lakshman lanes are out
    of scope by construction, and the record is what makes that checkable
    after the fact rather than only assertable in a review.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    lane: EnumRuntimeLane
    compose_project: str
    endpoint: str
    outcome: EnumVerifyRecreateOutcome
    recreate_returncode: int | None = None
    readiness_wait_seconds: float = 0.0
    readiness_budget_seconds: int = 0
    detail: str = ""

    def describe(self) -> str:
        """One-line, log-ready statement of what was recreated and how it ended."""
        line = (
            f"{self.service} on {self.compose_project} recreated after "
            f"{self.endpoint} failed: {self.outcome.value}"
        )
        if self.outcome == EnumVerifyRecreateOutcome.RECOVERED:
            line += f" after {self.readiness_wait_seconds:.0f}s"
        if self.detail:
            line += f" ({self.detail})"
        return line


class ModelRebuildRequested(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    requested_by: str
    scope: Scope
    runtime_lane: EnumRuntimeLane
    build_source: BuildSource = BuildSource.RELEASE
    services: list[str] = Field(default_factory=list)
    # OMN-16442: a command that omits the ref deploys the branch this agent
    # DECLARES it tracks (DEPLOY_AGENT_TRACKING_REF), not a literal. The old
    # default was "origin/main"; on the .201 dev lane that asked the agent to
    # `git reset --hard` its deploy-source clone onto a release-synced branch
    # hundreds of commits behind the code the lane exists to run. There is no
    # default for the variable itself — an undeclared tracking ref raises
    # rather than guessing (rule 8).
    git_ref: str = Field(default_factory=load_tracking_remote_ref_from_env)
    # Carry both ref and digest; the digest is the authority. dev/stability-test
    # may build from a ref and leave the digest unresolved up front; prod must
    # pin the stability-proven digest (enforced below).
    image_ref: str | None = None
    image_digest: str | None = None

    @model_validator(mode="after")
    def validate_services_subset(self) -> ModelRebuildRequested:
        if self.services:
            # OMN-18108: lane-aware, so a dev command may name one of the
            # dev-lane-only services and a prod/stability command may not.
            allowed = services_for_scope(self.scope, lane=self.runtime_lane)
            invalid = [s for s in self.services if s not in allowed]
            if invalid:
                msg = f"Services {invalid} not in scope '{self.scope}'. Allowed: {allowed}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_prod_requires_digest(self) -> ModelRebuildRequested:
        if self.runtime_lane == EnumRuntimeLane.PROD and not self.image_digest:
            msg = (
                "prod runtime_lane requires image_digest: production deploys the "
                "exact stability-proven digest and never rebuilds from a ref"
            )
            raise ValueError(msg)
        return self


class EnumRejectionReason(StrEnum):
    """Why a command reached the rejection topic instead of running.

    Every value was already a bare string literal at a ``_publish_rejected``
    call site; naming them is what lets OMN-18143 AC6's requirement -- a
    terminal event that says "superseded" distinguishably from a timeout and
    from a rollback -- be a type rather than a convention about spelling.

    ``SUPERSEDED`` is the only one that is not a refusal of the command: the
    work it asked for IS being done, by the newer command named alongside it.
    """

    BUSY = "busy"
    DUPLICATE = "duplicate"
    IN_PROGRESS = "in_progress"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_SIGNATURE = "invalid_signature"
    LANE_NOT_ALLOWED = "lane_not_allowed"
    UNDECODABLE_PAYLOAD = "undecodable_payload"
    SUPERSEDED = "superseded"


class ModelRebuildRejected(BaseModel):
    """The terminal event for a command this agent will not run (OMN-18143).

    The wire shape is unchanged for every reason that predates this model: the
    two supersession fields are written ONLY when set, so a rejection for
    ``busy`` serialises byte-identically to the hand-built dict it replaces and
    a consumer that predates this change parses it unchanged. The same
    discipline ``ModelLabPassCheck.to_dict`` applies to its ``outcome`` key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    reason: EnumRejectionReason
    scope: Scope
    #: Set on, and only on, a ``SUPERSEDED`` rejection. Both or neither: a
    #: supersession that cannot name the commit that ran in its place is
    #: indistinguishable from a command that was silently dropped, which is
    #: the exact failure AC6 refuses.
    superseded_by_sha: str | None = None
    superseded_by_correlation_id: UUID | None = None

    @model_validator(mode="after")
    def _supersession_fields_match_the_reason(self) -> ModelRebuildRejected:
        named = self.superseded_by_sha is not None
        if named != (self.superseded_by_correlation_id is not None):
            msg = (
                "superseded_by_sha and superseded_by_correlation_id stand or "
                f"fall together; got sha={self.superseded_by_sha!r}, "
                f"correlation_id={self.superseded_by_correlation_id!r}"
            )
            raise ValueError(msg)
        if (self.reason is EnumRejectionReason.SUPERSEDED) != named:
            msg = (
                f"reason={self.reason.value!r} disagrees with the supersession "
                f"fields (sha={self.superseded_by_sha!r}). Only a superseded "
                "rejection may name a replacement, and every one must."
            )
            raise ValueError(msg)
        return self

    def to_wire(self) -> dict[str, object]:
        """The JSON body published to :data:`TOPIC_REBUILD_REJECTED`."""
        payload: dict[str, object] = {
            "correlation_id": str(self.correlation_id),
            "reason": self.reason.value,
            "scope": self.scope.value,
        }
        if self.superseded_by_sha is not None:
            payload["superseded_by_sha"] = self.superseded_by_sha
            payload["superseded_by_correlation_id"] = str(
                self.superseded_by_correlation_id
            )
        return payload


class ModelRebuildCompleted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    correlation_id: UUID
    requested_git_ref: str
    git_sha: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    scope: Scope
    runtime_lane: EnumRuntimeLane
    image_ref: str | None = None
    image_digest: str | None = None
    services_restarted: list[str] = Field(default_factory=list)
    # OMN-17135: repo -> the commit SHA RT-1 resolved and vendored for that
    # sibling in a workspace-mode build. ``requested_git_ref`` above pins ONE
    # repository (omnibase_infra), so on its own it said nothing about which
    # omnibase_core / omnibase_compat / omnimarket commit the image carries.
    # This is EVIDENCE beside the infra pin, never a key: the rule-24 lab-pass
    # receipt is keyed by the infra sha and stays that way. Empty for a
    # release-mode or prod digest deploy, which vendors no sibling trees.
    sibling_refs: dict[str, str] = Field(default_factory=dict)
    phase_results: dict[Phase, PhaseStatus]
    errors: list[str] = Field(default_factory=list)
    health_checks: list[ModelHealthCheck] = Field(default_factory=list)
    container_residue: list[ModelContainerResidue] = Field(default_factory=list)
    # OMN-18692: what the deps-phase ceiling DID -- whether it deferred before
    # touching the lane, waited past its ceiling rather than cancelling a live
    # recreate, or ended the command and why. Empty for a deploy whose deps
    # phase was never reached, which is a different fact from a deploy whose
    # deps phase ran without incident (that one carries a `completed` entry).
    recreate_supervision: list[ModelRecreateSupervision] = Field(default_factory=list)
    # OMN-18640 AC7: the runtime containers this deploy force-recreated because
    # their own health endpoint failed post-deploy, and whether that repaired
    # them. Empty is the normal reading and is a FACT, not an absence: it says
    # the verification phase found nothing to repair. Before this, a verify
    # failure left the container exactly as it was found and the event said
    # only that the job failed, which is why the same wedge recurred three
    # times across two nights with nothing to distinguish the occurrences.
    verify_recreate: list[ModelVerifyRecreate] = Field(default_factory=list)
    # OMN-18640: what the deps leg found before it acted -- per core service,
    # the running config hash, the rendered one, and whether convergence was
    # therefore about to replace that container. Empty for a deploy whose deps
    # leg was never reached. A non-empty list with every `differs` false is the
    # normal reading and is a FACT: the deps were left alone on purpose.
    deps_convergence: list[ModelDepsConvergenceFinding] = Field(default_factory=list)
    # OMN-18640: the argv of every compose command this deploy issued. Until
    # this field existed the flags a deploy used were observable only by
    # sampling the host process table while the child ran.
    compose_invocations: list[ModelComposeInvocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase_results_are_settled(self) -> ModelRebuildCompleted:
        """A terminal event may only carry settled phase verdicts (OMN-18057).

        Two shapes are refused here rather than merely discouraged upstream:

        * ``PhaseStatus.IN_PROGRESS`` -- the live defect. Command 23edaf62's
          terminal event carried ``runtime: in_progress`` alongside a
          ``completed_at`` and a duration, so the event asserted the deploy was
          over and simultaneously refused to say how it ended. An unreached
          phase is SKIPPED and a phase that raised is FAILED.
        * ``Phase.PUBLISH`` -- this event IS the publish. Its outcome is not
          knowable at the moment the payload is built, and reporting it as
          ``in_progress`` made ``status`` derive "failed" for every deploy the
          agent ever completed, successful ones included.
        """
        if Phase.PUBLISH in self.phase_results:
            raise ValueError(
                "phase_results must not carry Phase.PUBLISH: the completion "
                "event is the publish and cannot report its own outcome"
            )
        unsettled = sorted(
            phase.value
            for phase, status in self.phase_results.items()
            if status in (PhaseStatus.IN_PROGRESS, PhaseStatus.PENDING)
        )
        if unsettled:
            raise ValueError(
                f"phase_results carries unsettled verdicts for {unsettled}: a "
                "terminal event must report failed/skipped for a phase that "
                "raised or was never reached"
            )
        return self

    @computed_field
    @property
    def status(self) -> Literal["success", "failed"]:
        non_skipped = {
            k: v for k, v in self.phase_results.items() if v != PhaseStatus.SKIPPED
        }
        if all(v == PhaseStatus.SUCCESS for v in non_skipped.values()):
            return "success"
        return "failed"
