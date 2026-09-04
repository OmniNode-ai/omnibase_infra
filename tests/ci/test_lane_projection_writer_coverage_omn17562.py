# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A strict lane must declare a writer for every projection it cannot dispatch [OMN-17562].

The defect this gate exists for, read live off the ``.201`` **dev** lane
(``omninode-runtime``, root ``0.38.18``, 2026-09-03T01:08Z)::

    projection_write_path | DEGRADED | 15/37 declared projection(s) are subscribed
    here but dispatch NOTHING in this process (standalone-runner shape): offsets
    commit and no row is written unless a dedicated writer is deployed for each on
    this lane: node_projection_cost_by_repo, node_projection_instruction_eval,
    node_projection_receipt_gate, node_projection_skill_executions,
    projection_baselines, projection_intent_classification, projection_llm_cost,
    projection_overnight, +7 more

Byte-identical on ``omninode-stability-test-runtime``; ``2/13`` on both effects
processes. ``offsets commit and no row is written`` is the whole finding: these
are not idle subscriptions. The kernel joins the consumer group, takes every
message, advances the offset, and the callback returns before any handler runs.

Why a STATIC gate, and why it is this file rather than a health dimension
------------------------------------------------------------------------
``omnibase_infra.runtime.projection_dispatch_ledger`` refuses to make this
claim at runtime, in its own words::

    It deliberately does NOT claim "X has no writer anywhere". A kernel process
    cannot see a sibling Deployment in another pod or another compose service,
    and inventing that claim would report a permanent false outage on every lane
    where the standalone writer is correctly deployed. The corpus-level "every
    subscribing lane has a deployed writer" assertion is a static gate over the
    deployment manifests, not a runtime health dimension (OMN-17448 AC5).

This module is that gate. It is the missing half, not a second opinion.

The consequence, proven live rather than argued, is that **deploying a writer
does not move the runtime number**. On 2026-09-03T01:08Z the dev lane was
running ``omnimarket-projection-delegation-writer`` and
``omnimarket-projection-tenant-registry-writer`` (both ``Up 10 hours``, healthy)
and still reported ``15/37``; the stability lane, with zero writers of its own,
reported ``15/37`` with a byte-identical detail string. Both ledger call sites
fire at wiring time, before any sibling process could matter:

* ``handler_wiring.py:3556`` — the OMN-15905 standalone-runner branch.
* ``handler_wiring.py:3479`` — the OMN-17519 zero-route branch, which also
  catches the in-process sibling of every writer-owned contract.

So a green ``projection_write_path`` is NOT this gate's success condition and
must never be treated as one. Zeroing that dimension requires the contract to
leave the profile (OMN-17641 / OMN-17556, the ratified ``tenant-projection``
route) or the kernel to stop subscribing what it will never dispatch. This gate
answers the different, un-answered question: *given that the kernel writes
nothing for these, does the lane run a process that does?*

What is enforced
----------------
1. **Fail-closed lane discovery** — every compose overlay that binds
   ``ONEX_WIRING_STRICT_MODE`` is a strict lane and must be registered below. A
   fourth lab lane cannot be added outside this gate's view, which is exactly
   how the dev lane carried this defect green: its container probe is
   ``curl -sf /health`` (HTTP 200 on a body that says ``"status":"degraded"``)
   while stability runs ``onex-container-healthcheck --degraded-policy fail``.
2. **No orphan writer service** — a ``command: [python, -m, <module>]`` service
   in a lane overlay must name a runner module this registry knows, so a writer
   for a contract nobody tracks cannot appear.
3. **Coverage ratchet, shrink-only** — each lane's count of writer-owned
   projections with no writer service is pinned. It may fall and never rise.
4. **Proof-lane gap ratchet** — the stability lane is the surface the
   ``stability-proven`` premise of the prod-promotion gate is resolved from. It
   must not be weaker than the mutable dev lane by more than the pinned gap.
5. **Registry fidelity** — when ``omnimarket`` is importable, the registry is
   re-derived through the REAL runtime predicates and must match exactly, so it
   cannot drift from the wiring seam it describes. Skipped, loudly, in the CI
   job where omnimarket is absent (see ``_OMNIMARKET_ABSENT_REASON``).

Deliberately NOT enforced here: that a writer-owned contract has a writer on
*some* lane, or on onex-dev. ``runtime_profiles`` is declared in the omnimarket
contract and is identical on every lane, so the same 17 are non-writing on
onex-dev too; the k8s half is ``omninode_infra``'s manifest set and is governed
by OMN-17519 / OMN-17556, not by a file in this repo.

Related tickets:
    - OMN-17562: this gate.
    - OMN-17448: the ``projection_write_path`` dimension and its AC5 residual.
    - OMN-15905: why the kernel skip exists and the dedicated-writer pattern.
    - OMN-16874: ``_is_standalone_projection_runner``, the shape predicate.
    - OMN-17519: ``_projection_dispatch_owned_elsewhere``, the zero-route branch.
    - OMN-17531: bound ``ONEX_WIRING_STRICT_MODE`` on the lab lanes, exposing this.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import pytest
import yaml

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_auto_wiring_manifest_like import (
        ProtocolAutoWiringManifestLike,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"

# The binding that makes a lane strict. OMN-17531 bound it on the three lab
# lanes and deliberately left the base file's `${...:-0}` alone, so prod and
# judge do not carry it. Discovery keys off this rather than a hand-listed set
# of filenames: a new overlay that opts into strict wiring opts into this gate
# in the same edit, and cannot be added without one.
STRICT_MODE_BINDING = "ONEX_WIRING_STRICT_MODE"

# The base every lane merges. It DECLARES the binding as `${...:-0}` — the
# fail-open default OMN-17531 deliberately left alone so prod and judge keep the
# quarantining semantics — and is not itself a lane. Excluded by name as well as
# by the literal-value rule below, so neither check is solely load-bearing.
BASE_COMPOSE = "docker-compose.infra.yml"

# Overlays that are not lanes: fixtures, canaries, and single-purpose add-ons
# that never carry a runtime kernel. Listed by exact filename so a real lane
# cannot be waved through by a pattern that happens to match it.
NON_LANE_OVERLAYS: frozenset[str] = frozenset(
    {
        BASE_COMPOSE,
        "docker-compose.generated.yml",  # rendered artifact, not a source lane
        "docker-compose.e2e.yml",
        "docker-compose.gate-runner.yml",
        "docker-compose.gateway-attach-test-lane.yml",
        "docker-compose.runners.yml",
    }
)


class LaneSpec(NamedTuple):
    """A strict lane and the compose overlay that defines its services."""

    overlay: str
    # False for a lane this repo may not mutate. The gate still measures it —
    # an unmeasured lane is how this class hid — but its ratchet is owned by
    # the lane's owner, not by a PR in this repo.
    mutable_here: bool
    owner_note: str


GOVERNED_LANES: dict[str, LaneSpec] = {
    "dev": LaneSpec(
        overlay="docker-compose.dev-lane.yml",
        mutable_here=True,
        owner_note="fully mutable test platform (lane map)",
    ),
    "stability-test": LaneSpec(
        overlay="docker-compose.stability-test.yml",
        mutable_here=True,
        owner_note="proof lane; `stability-proven` is resolved from it (OMN-15243)",
    ),
    "lakshman": LaneSpec(
        overlay="docker-compose.lakshman.yml",
        mutable_here=False,
        owner_note=(
            "owner-controlled contractor lane (OMN-17150). Measured so it cannot "
            "silently diverge; its ratchet moves when its owner deploys, not here."
        ),
    ),
}


class WriterOwned(NamedTuple):
    """A projection the shared kernel subscribes and never dispatches."""

    profile: str
    runner_class: str
    runner_module: str


# The 16 writer-owned projections, derived through the real runtime predicates
# (`select_projection_contracts` -> `_projection_dispatch_owned_elsewhere` /
# `_is_standalone_projection_runner`) against the discovery manifest. The
# derivation reproduces both live lanes exactly: main 15/37, effects 2/13,
# workers 0/2. `test_registry_matches_the_runtime_predicate` re-runs it.
#
# 15 main + 2 effects = 17 is the ticket's headline; 16 of those 17 are here and
# the 17th is ORPHANED_PROJECTIONS below. `projection_registration`,
# `projection_savings`, `projection_delegation`, `projection_live_events`,
# `projection_tenant_credentials` and `projection_tenant_registry` are recorded
# by BOTH ledger branches (the runner entry takes every topic, its in-process
# sibling is left with none) — which is why a deployed writer cannot clear them.
WRITER_OWNED_PROJECTIONS: dict[str, WriterOwned] = {
    "node_projection_cost_by_repo": WriterOwned(
        "main",
        "CostByRepoProjectionRunner",
        "omnimarket.nodes.node_projection_cost_by_repo.handlers.handler_cost_by_repo",
    ),
    "node_projection_instruction_eval": WriterOwned(
        "main",
        "InstructionEvalProjectionRunner",
        "omnimarket.nodes.node_projection_instruction_eval.handlers.handler_instruction_eval",
    ),
    "node_projection_receipt_gate": WriterOwned(
        "main",
        "HandlerReceiptGateProjectionRunner",
        "omnimarket.nodes.node_projection_receipt_gate.handlers.handler_receipt_gate",
    ),
    "node_projection_skill_executions": WriterOwned(
        "main",
        "SkillExecutionsProjectionRunner",
        "omnimarket.nodes.node_projection_skill_executions.handlers.handler_skill_executions",
    ),
    "projection_baselines": WriterOwned(
        "main",
        "BaselinesProjectionRunner",
        "omnimarket.nodes.node_projection_baselines.handlers.handler_baselines",
    ),
    "projection_delegation": WriterOwned(
        "effects",
        "DelegationProjectionRunner",
        "omnimarket.nodes.node_projection_delegation.handlers.handler_delegation",
    ),
    "projection_intent_classification": WriterOwned(
        "main",
        "IntentClassificationProjectionRunner",
        "omnimarket.nodes.node_projection_intent_classification.handlers.handler_intent_classification",
    ),
    "projection_live_events": WriterOwned(
        "effects",
        "HandlerLiveEventsProjectionRunner",
        "omnimarket.nodes.node_projection_live_events.handlers.handler_live_events",
    ),
    "projection_llm_cost": WriterOwned(
        "main",
        "LlmCostProjectionRunner",
        "omnimarket.nodes.node_projection_llm_cost.handlers.handler_llm_cost",
    ),
    "projection_pattern_learning": WriterOwned(
        "main",
        "PatternLearningProjectionRunner",
        "omnimarket.nodes.node_projection_pattern_learning.handlers.handler_pattern_learning",
    ),
    "projection_registration": WriterOwned(
        "main",
        "RegistrationProjectionRunner",
        "omnimarket.nodes.node_projection_registration.handlers.handler_registration",
    ),
    "projection_routing_decision": WriterOwned(
        "main",
        "RoutingDecisionProjectionRunner",
        "omnimarket.nodes.node_projection_routing_decision.handlers.handler_routing_decision",
    ),
    "projection_savings": WriterOwned(
        "main",
        "SavingsProjectionRunner",
        "omnimarket.nodes.node_projection_savings.handlers.handler_savings",
    ),
    "projection_session_outcome": WriterOwned(
        "main",
        "SessionOutcomeProjectionRunner",
        "omnimarket.nodes.node_projection_session_outcome.handlers.handler_session_outcome",
    ),
    "projection_tenant_credentials": WriterOwned(
        "main",
        "HandlerTenantCredentialsProjectionRunner",
        "omnimarket.nodes.node_projection_tenant_credentials.handlers.handler_tenant_credentials_projection",
    ),
    "projection_tenant_registry": WriterOwned(
        "main",
        "HandlerTenantRegistryProjectionRunner",
        "omnimarket.nodes.node_projection_tenant_registry.handlers.handler_tenant_registry_projection",
    ),
}

# The 17th. Not writer-owned and NOT fixable by deploying a container: three
# `event_model`-typed handler entries compete for three subscribe topics, so
# `_topics_for_handler_entry` gives every entry zero routes and no entry owns a
# topic. There is no `*ProjectionRunner` class in the node at all. It persists
# nothing on ANY lane, onex-dev and prod included. `_projection_dispatch_owned_
# elsewhere`'s own docstring names it as the pre-existing orphan the OMN-15905
# pattern is distinguished from. Excluded from the coverage ratchets because a
# writer service would be a container with nothing to run.
# The 17th WAS `projection_overnight`, and it is gone: `omnimarket#2278`
# ("fix(OMN-17562): projection_overnight registers zero dispatch routes",
# merged 2026-09-03T07:31:06Z) switched its `handler_routing` from
# `operation_match` to `topic_match` and gave each of its three entries an
# owning `topic:`. Every entry now resolves a route, so the contract dispatches
# in-process and is no longer subscribed-and-never-dispatched on any lane. It
# needs no writer service and is not an orphan — it left this classification
# entirely rather than moving between its halves.
#
# Kept as an empty registry rather than deleted: a zero-route contract is a
# recurring class (`_projection_dispatch_owned_elsewhere`'s own docstring names
# it), and the next one must land in a named, ticket-citing home instead of
# being folded into the writer ratchets, where a writer service would be a
# container with nothing to run and the ratchet would become un-closeable.
ORPHANED_PROJECTIONS: dict[str, str] = {}

# The operator ruling of 2026-09-03 split the writer-owned set in two, and the
# split is the substance of this PR. Both halves are frozen here so neither can
# grow by accident.
#
# ADOPT — mirrored onto both mutable lab lanes as writer services by this
# change. Membership is not a judgement call: these are exactly the six with a
# checked-in writer Deployment under `omninode_infra` `k8s/onex-dev/runtime/` on
# origin/dev, i.e. a runner already proven to run as its own process.
# `omninode_infra#1147` ("revert(OMN-17519): remove rejected projection writer
# rollout", merged 2026-09-02T19:28:06Z, the exact inverse of #1146) removed the
# pattern-learning and routing-decision Deployments, stating that "the operator
# ruling identified that rollout direction as prohibited" — so those two are NOT
# adopted here. Their in-process sibling is the entry raising
# `ValueError: Projection handler requires topology bindings with configured
# DSNs: tenant_projection:ONEX_TENANT_DB_URL`, which is OMN-17557 / OMN-17556
# store-resolved-credential work, not writer-mirroring work.
ADOPTED_WRITER_PROJECTIONS: frozenset[str] = frozenset(
    {
        "projection_delegation",
        "projection_live_events",
        "projection_registration",
        "projection_savings",
        "projection_tenant_credentials",
        "projection_tenant_registry",
    }
)

# DROP — ruled out of the shared main/effects runtime profiles: no writer exists
# on any lane and nothing consumes their tables, so the ruling stops the kernel
# subscribing them rather than deploying four processes to write rows no reader
# reads. Delivered by the omnibase_infra#3156 subscription skip, NOT by editing
# the omnimarket contracts: none of the four declares `runtime_profiles` at all
# (so it defaults to `main`), an empty list is falsy and falls through to that
# same default, and naming a non-consumer-attached profile is rejected by
# `ValidatorRuntimeProfiles` against `CONSUMER_ATTACHED_RUNTIME_PROFILES`. A new
# registered profile name would need an omnibase_core release, which the same
# ruling forbids.
#
# Frozen at exactly four. A fifth name appearing here would mean a projection
# was dropped without a ruling — the failure mode this allowlist exists to make
# impossible, since "no writer" and "ruled to need none" are indistinguishable
# from the coverage count alone.
RULED_DROP_PROJECTIONS: dict[str, str] = {
    "node_projection_cost_by_repo": "OMN-17562 ruling (1): no writer on any lane, no consumer.",
    "node_projection_instruction_eval": "OMN-17562 ruling (1): no writer on any lane, no consumer.",
    "node_projection_skill_executions": "OMN-17562 ruling (1): no writer on any lane, no consumer.",
    "projection_llm_cost": "OMN-17562 ruling (1): no writer on any lane, no consumer.",
}

# Writer-owned projections with NO writer service on the lane, pinned at the
# value measured on 2026-09-03T01:08Z. Shrink-only: a PR that lands a writer
# lowers the number here in the same change; a PR that removes one fails.
#
# OMN-17562 moved both mutable lanes from 14/16 to 10: the six-name ADOPT set
# now has a writer service on each. The residual 10 is the four ruled-DROP names
# plus the six contracts that keep a LIVE in-process dispatcher beside their
# runner entry (`node_projection_receipt_gate`, `projection_baselines`,
# `projection_intent_classification`, `projection_session_outcome`,
# `projection_pattern_learning`, `projection_routing_decision`) — they write
# rows today and a writer container for them would be a second process
# competing for the same partitions. `lakshman` is unchanged: this repo measures
# that lane but does not deploy it.
LANE_UNCOVERED_RATCHET: dict[str, int] = {
    "dev": 10,
    "stability-test": 10,
    "lakshman": 16,
}

# Writers the mutable dev lane has that the PROOF lane does not. The proof lane
# being the weaker of the two is backwards — `stability-proven` is resolved from
# it — so the asymmetry is pinned and must shrink, never grow. OMN-17562 closed
# it: the two lanes now run the same six.
PROOF_LANE_WRITER_GAP: int = 0

_OMNIMARKET_ABSENT_REASON = (
    "omnimarket is not importable here. It is deliberately absent from this "
    "repo's canonical venv (the OMN-15620 purity gate rejects an undeclared "
    "`onex.nodes` provider), so the fidelity cross-check cannot run in that "
    "job. The four manifest-only gates above are unaffected and still run."
)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose's `!override` / `!reset` merge tags.

    Compose Spec tags are directives to compose's own merge algorithm, not data.
    Constructing the node's value and discarding the tag is exactly right for a
    gate that reads *which services and commands are declared*: the tag changes
    how a key merges across `-f` files, never whether the service exists.
    """


def _construct_tagged(loader: _ComposeLoader, _suffix: str, node: yaml.Node) -> Any:
    """Build a tagged node's value, discarding the tag itself."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    raise yaml.constructor.ConstructorError(
        None, None, f"unsupported tagged node {type(node).__name__}", node.start_mark
    )


_ComposeLoader.add_multi_constructor("!", _construct_tagged)  # type: ignore[no-untyped-call]


def _load_compose(path: Path) -> dict[str, Any]:
    """Parse one compose file into plain data."""
    with path.open(encoding="utf-8") as handle:
        # The S506 suppression below is safe: `_ComposeLoader` derives from
        # `yaml.SafeLoader`, so it cannot
        # construct arbitrary Python objects. The only added behaviour is a
        # multi-constructor that returns the scalar/sequence/mapping value of a
        # `!`-tagged node — compose's `!override` / `!reset` merge directives.
        # `yaml.safe_load` cannot be used here: it raises on those tags, which is
        # what made the stability-test overlay unreadable to a gate in the first
        # place.
        loaded = yaml.load(handle, Loader=_ComposeLoader)  # noqa: S506
    return loaded if isinstance(loaded, dict) else {}


def _services(path: Path) -> dict[str, dict[str, Any]]:
    raw = _load_compose(path).get("services", {})
    if not isinstance(raw, dict):
        return {}
    return {name: body for name, body in raw.items() if isinstance(body, dict)}


def _environment(body: dict[str, Any]) -> dict[str, str]:
    """A service's environment as a mapping, from either compose form.

    The lab overlays reach it through a YAML anchor merged with `!!merge <<:`;
    PyYAML resolves that to a plain dict before this sees it, so one lookup
    covers the anchored and the literal spellings alike.
    """
    raw = body.get("environment")
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        pairs: dict[str, str] = {}
        for item in raw:
            key, _, value = str(item).partition("=")
            pairs[key] = value
        return pairs
    return {}


def _binds_strict_mode(body: dict[str, Any]) -> bool:
    """Whether a service *opts into* strict wiring, as opposed to declaring it.

    The distinction is the whole point. ``docker-compose.infra.yml`` sets
    ``ONEX_WIRING_STRICT_MODE: ${ONEX_WIRING_STRICT_MODE:-0}`` — a default, not
    an opt-in, and one that resolves to the quarantining semantics. The lab
    overlays bind the literal ``"1"``, and OMN-17531 requires the literal
    precisely so an operator who exported the variable cannot silently return a
    lane to the weaker mode. So: a literal, non-interpolated, non-``0`` value.
    """
    value = _environment(body).get(STRICT_MODE_BINDING)
    if value is None or "${" in value:
        return False
    return value.strip() not in {"", "0"}


def _module_from_command(body: dict[str, Any]) -> str | None:
    """The module of a `command: [python, -m, <module>]` service, if any.

    This is the shape every standalone writer uses — compose and the onex-dev
    Deployments invoke the runner module's `if __name__ == "__main__"` block
    identically, so one matcher covers both and neither can drift into a shape
    the gate cannot see.
    """
    command = body.get("command")
    if not isinstance(command, list):
        return None
    parts = [str(item) for item in command]
    for index, part in enumerate(parts):
        if part == "-m" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _writer_modules_on_lane(lane: str) -> dict[str, str]:
    """Map ``runner module -> service name`` for one lane's overlay."""
    overlay = DOCKER_DIR / GOVERNED_LANES[lane].overlay
    found: dict[str, str] = {}
    for service, body in _services(overlay).items():
        module = _module_from_command(body)
        if module is not None:
            found[module] = service
    return found


def _covered_projections(lane: str) -> set[str]:
    """Writer-owned projections that have a writer service on ``lane``."""
    modules = set(_writer_modules_on_lane(lane))
    return {
        name
        for name, spec in WRITER_OWNED_PROJECTIONS.items()
        if spec.runner_module in modules
    }


def _discovered_strict_overlays() -> set[str]:
    """Every compose overlay in ``docker/`` that binds strict wiring."""
    return {
        path.name
        for path in sorted(DOCKER_DIR.glob("docker-compose.*.yml"))
        if path.name not in NON_LANE_OVERLAYS
        and any(_binds_strict_mode(body) for body in _services(path).values())
    }


def test_every_strict_lane_is_registered() -> None:
    """A compose overlay that opts into strict wiring must be governed here.

    Fail-closed discovery. The dev lane carried this defect green for as long as
    it did because it was strict-wired but shallow-probed and nothing compared
    it to the lane that would have caught it. A fourth lab lane must not be able
    to repeat that by simply not appearing in a hand-written list.
    """
    discovered = _discovered_strict_overlays()
    registered = {spec.overlay for spec in GOVERNED_LANES.values()}

    unregistered = sorted(discovered - registered)
    assert not unregistered, (
        f"Compose overlay(s) bind {STRICT_MODE_BINDING} but are not in "
        f"GOVERNED_LANES: {unregistered}. A strict lane runs the auto-wiring "
        "kernel and therefore subscribes the writer-owned projections this gate "
        "tracks. Add it to GOVERNED_LANES with its own LANE_UNCOVERED_RATCHET "
        "entry (measure it, do not guess), or add it to NON_LANE_OVERLAYS if it "
        "genuinely runs no kernel."
    )

    stale = sorted(registered - discovered)
    assert not stale, (
        f"GOVERNED_LANES names overlay(s) that no longer bind "
        f"{STRICT_MODE_BINDING}: {stale}. If the lane was retired, drop its "
        "GOVERNED_LANES and LANE_UNCOVERED_RATCHET entries. If strict mode was "
        "removed from a live lab lane, that is an OMN-17531 AC-4 regression — "
        "restore the binding rather than deleting the lane from this gate."
    )


def test_every_writer_service_maps_to_a_registered_projection() -> None:
    """A `python -m <module>` service must name a runner this registry knows.

    Blocks the inverse drift from the ratchets below: a writer container for a
    contract nobody tracks looks like coverage on `docker ps` and proves nothing.
    """
    known_modules = {spec.runner_module for spec in WRITER_OWNED_PROJECTIONS.values()}
    orphans: list[str] = []
    for lane in GOVERNED_LANES:
        for module, service in sorted(_writer_modules_on_lane(lane).items()):
            if module not in known_modules:
                orphans.append(f"{lane}/{service} -> {module}")

    assert not orphans, (
        f"Writer service(s) run a module this gate does not track: {orphans}. "
        "Either the module is a standalone projection runner and belongs in "
        "WRITER_OWNED_PROJECTIONS (re-derive it, do not hand-add), or the "
        "service is not a projection writer and should not use the "
        "`command: [python, -m, ...]` runner shape on a lane overlay."
    )


@pytest.mark.parametrize("lane", sorted(GOVERNED_LANES))
def test_lane_uncovered_writer_count_is_a_shrinking_ratchet(lane: str) -> None:
    """Each lane's count of writer-owned projections with no writer may only fall.

    This is the corpus-level assertion `projection_dispatch_ledger` defers to.
    It is NOT satisfied by the runtime's `projection_write_path` dimension going
    green, and going green there does not satisfy it: that dimension is
    process-local and, by explicit design, cannot see a sibling writer.
    """
    covered = _covered_projections(lane)
    uncovered = sorted(set(WRITER_OWNED_PROJECTIONS) - covered)
    pinned = LANE_UNCOVERED_RATCHET[lane]
    spec = GOVERNED_LANES[lane]

    assert len(uncovered) <= pinned, (
        f"Lane {lane!r} now has {len(uncovered)} writer-owned projection(s) with "
        f"no writer service, up from the pinned {pinned}. The kernel subscribes "
        "each of these and dispatches nothing, so their events are consumed, "
        "acked, and dropped. Missing: "
        f"{uncovered}. Lane note: {spec.owner_note}."
    )

    assert len(uncovered) == pinned, (
        f"Lane {lane!r} now has {len(uncovered)} uncovered writer-owned "
        f"projection(s), below the pinned {pinned}. This is the good direction — "
        f"lower LANE_UNCOVERED_RATCHET[{lane!r}] to {len(uncovered)} in this same "
        "change so the ratchet cannot slip back."
    )


def test_proof_lane_is_not_more_than_the_pinned_gap_weaker_than_dev() -> None:
    """stability-test must not trail the mutable dev lane by a growing margin.

    The stability lane is where `stability-proven` is resolved from for every
    live prod-promotion grant (OMN-15243). A proof lane that writes fewer
    projection tables than the throwaway dev lane inverts what the two lanes are
    for, and the asymmetry is pinned here rather than left to be rediscovered.
    """
    dev_only = sorted(
        _covered_projections("dev") - _covered_projections("stability-test")
    )

    assert len(dev_only) <= PROOF_LANE_WRITER_GAP, (
        f"The dev lane now runs {len(dev_only)} writer(s) the stability lane does "
        f"not, above the pinned {PROOF_LANE_WRITER_GAP}: {dev_only}. Add the "
        "writer to docker-compose.stability-test.yml in the same change that "
        "adds it to dev, or lower the dev lane instead — do not widen the gap."
    )

    assert len(dev_only) == PROOF_LANE_WRITER_GAP, (
        f"The dev/stability writer gap is now {len(dev_only)}, below the pinned "
        f"{PROOF_LANE_WRITER_GAP}. Lower PROOF_LANE_WRITER_GAP to "
        f"{len(dev_only)} in this same change."
    )


@pytest.mark.parametrize(
    "lane", sorted(name for name, spec in GOVERNED_LANES.items() if spec.mutable_here)
)
def test_every_adopted_projection_has_a_writer_service_on_every_mutable_lane(
    lane: str,
) -> None:
    """The ruled ADOPT set is deployed, not merely ratified.

    This is the closeable half of the coverage question. The ratchet above is a
    number that may fall; this is a NAMED set that must be complete on both
    lanes this repo deploys, so a writer cannot be quietly dropped from one lane
    while the ratchet stays satisfied by an unrelated addition.

    Deliberately not applied to `lakshman`: this repo measures that lane so it
    cannot silently diverge, but its owner deploys it.
    """
    missing = sorted(ADOPTED_WRITER_PROJECTIONS - _covered_projections(lane))

    assert not missing, (
        f"Lane {lane!r} declares no writer service for {missing}. Each has a "
        "checked-in onex-dev writer Deployment, so a runner provably exists to "
        "run; without the service here the kernel consumes their topics to LAG 0 "
        "and writes nothing. Add the service to "
        f"docker/{GOVERNED_LANES[lane].overlay}."
    )


def test_the_ruled_sets_are_exactly_the_ruled_names_and_disjoint() -> None:
    """ADOPT is six, DROP is four, and neither may grow without a ruling.

    Both are operator decisions, not derivations, so nothing else in this file
    can catch them growing. The count assertions are the whole guard: a seventh
    adopted name would mean a writer was deployed for a projection with no
    proven runner, and a fifth dropped name would mean a projection stopped
    being served with no ruling behind it.
    """
    assert len(ADOPTED_WRITER_PROJECTIONS) == 6, (
        f"ADOPTED_WRITER_PROJECTIONS is {sorted(ADOPTED_WRITER_PROJECTIONS)}. The "
        "ruled ADOPT set is the six with a checked-in onex-dev writer Deployment "
        "on omninode_infra origin/dev. Adding a name means deploying a writer "
        "whose runner has never run as its own process anywhere."
    )
    assert sorted(RULED_DROP_PROJECTIONS) == [
        "node_projection_cost_by_repo",
        "node_projection_instruction_eval",
        "node_projection_skill_executions",
        "projection_llm_cost",
    ], (
        f"RULED_DROP_PROJECTIONS is {sorted(RULED_DROP_PROJECTIONS)}, not the four "
        "names the OMN-17562 ruling dropped. A projection may only leave the "
        "served set by a ruling — 'has no writer' and 'was ruled to need none' "
        "are indistinguishable from the coverage count alone, which is exactly "
        "why this list is frozen rather than derived."
    )

    overlap = sorted(ADOPTED_WRITER_PROJECTIONS & set(RULED_DROP_PROJECTIONS))
    assert not overlap, (
        f"{overlap} are both adopted and dropped. A contract gets a writer or "
        "stops being subscribed; it cannot be ruled both ways."
    )

    unknown = sorted(
        (ADOPTED_WRITER_PROJECTIONS | set(RULED_DROP_PROJECTIONS))
        - set(WRITER_OWNED_PROJECTIONS)
    )
    assert not unknown, (
        f"{unknown} are ruled on but are not writer-owned per the registry "
        "above. A ruling that names a contract this gate does not track cannot "
        "be enforced by it — re-derive the registry rather than widening a set "
        "to match a name."
    )

    unticketed = sorted(
        name for name, reason in RULED_DROP_PROJECTIONS.items() if "OMN-" not in reason
    )
    assert not unticketed, (
        f"Dropped projection(s) {unticketed} cite no OMN ticket. A drop is a "
        "decision with an author, not a settled fact."
    )


def test_orphaned_projections_are_disjoint_and_cite_a_ticket() -> None:
    """The zero-route orphans are tracked, ticketed, and not counted as coverage.

    An orphan has no writer service to deploy, so silently folding it into the
    ratchets would make the ratchets un-closeable and hide a contract defect
    behind a deployment number.
    """
    overlap = sorted(set(ORPHANED_PROJECTIONS) & set(WRITER_OWNED_PROJECTIONS))
    assert not overlap, (
        f"{overlap} are listed as BOTH writer-owned and orphaned. A contract is "
        "one or the other: writer-owned means a runner class exists to run, "
        "orphaned means no entry owns a topic and none does."
    )

    unticketed = sorted(
        name for name, reason in ORPHANED_PROJECTIONS.items() if "OMN-" not in reason
    )
    assert not unticketed, (
        f"Orphaned projection(s) {unticketed} carry no OMN ticket. An orphan is "
        "an open contract defect, not a settled exclusion — it must be traceable "
        "to the ruling that will close it."
    )


@pytest.mark.skipif(
    importlib.util.find_spec("omnimarket") is None,
    reason=_OMNIMARKET_ABSENT_REASON,
)
def test_registry_matches_the_runtime_predicate() -> None:
    """Re-derive the registry through the REAL wiring predicates and compare.

    Keeps this file from becoming folklore. The registry is hand-committed data,
    so the only thing standing between it and drift is this test running the
    same functions the wiring seam runs: `select_projection_contracts` for
    scope, then the two branches that write the dispatch ledger
    (`_projection_dispatch_owned_elsewhere` at handler_wiring.py:3479 and the
    `_is_standalone_projection_runner` branch at :3556).
    """
    import importlib

    from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _is_standalone_projection_runner,
        _projection_dispatch_owned_elsewhere,
    )
    from omnibase_infra.runtime.auto_wiring.profile_ownership import (
        filter_manifest_for_runtime_profile,
    )
    from omnibase_infra.runtime.health.projection_liveness import (
        select_projection_contracts,
    )

    manifest = discover_contracts()
    derived_writer_owned: dict[str, str] = {}
    derived_nonwriting: set[str] = set()

    for profile in ("main", "effects", "workers"):
        owned = filter_manifest_for_runtime_profile(manifest, profile)
        by_name = {contract.name: contract for contract in owned.manifest.contracts}
        # The concrete manifest is frozen, so its attributes are read-only and
        # do not structurally satisfy the protocol's settable-variable members.
        # The runtime reaches this same function through a protocol-typed
        # parameter; the object is identical.
        for ref in select_projection_contracts(
            cast("ProtocolAutoWiringManifestLike", owned.manifest)
        ):
            contract = by_name[ref.name]
            routing = getattr(contract, "handler_routing", None)
            entries = list(routing.handlers) if routing is not None else []
            for entry in entries:
                handler = getattr(entry, "handler", None)
                class_name = getattr(handler, "name", None)

                # Branch one (handler_wiring.py:3479). Zero-route and
                # runner-ownership are ORTHOGONAL, so this does not `continue`:
                # `projection_tenant_credentials` declares the same
                # `HandlerTenantCredentialsProjectionRunner` on both of its two
                # entries, which leaves BOTH ambiguous and route-starved. Short-
                # circuiting here would classify a contract that has a perfectly
                # runnable writer as having none.
                if _projection_dispatch_owned_elsewhere(contract, entry):
                    derived_nonwriting.add(ref.name)

                module_path = getattr(entry, "handler_module", None) or getattr(
                    handler, "module", None
                )
                if not (class_name and module_path):
                    continue
                try:
                    handler_class = getattr(
                        importlib.import_module(module_path), class_name
                    )
                    # Branch two (handler_wiring.py:3556). The predicate reads an
                    # INSTANCE, so this constructs one exactly as the wiring seam
                    # does.
                    is_runner = _is_standalone_projection_runner(handler_class())
                except Exception:  # noqa: BLE001 - an unimportable sibling
                    # package, or a handler whose __init__ demands a DSN this
                    # host has not set, is a discovery-time condition the runtime
                    # already reports. It must not turn this gate red for an
                    # unrelated reason.
                    continue
                if is_runner:
                    derived_nonwriting.add(ref.name)
                    derived_writer_owned[ref.name] = module_path

    assert derived_writer_owned == {
        name: spec.runner_module for name, spec in WRITER_OWNED_PROJECTIONS.items()
    }, (
        "WRITER_OWNED_PROJECTIONS has drifted from the live wiring predicates. "
        "Re-derive it rather than editing by hand; a name added or removed here "
        "changes what every lane ratchet above means."
    )

    assert derived_nonwriting == set(WRITER_OWNED_PROJECTIONS) | set(
        ORPHANED_PROJECTIONS
    ), (
        "The set of projections the kernel subscribes and never dispatches has "
        "changed. Every such contract must be classified as writer-owned (a "
        "runner exists to deploy) or orphaned (none does) — an unclassified one "
        "is the silent consume-ack-drop this gate exists to prevent."
    )
