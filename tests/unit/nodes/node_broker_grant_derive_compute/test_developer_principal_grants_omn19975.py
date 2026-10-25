# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The grant derivation expresses prefixed group grants and the describe-only wildcard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.event_bus.util_consumer_group import compute_consumer_group_id
from omnibase_core.models.event_bus.model_bus_binding import ModelBusBinding
from omnibase_core.models.event_bus.model_bus_group_describe import (
    ModelBusGroupDescribe,
)
from omnibase_infra.handlers.handler_broker_grant_derive import (
    HandlerBrokerGrantDerive,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant import (
    ModelBrokerGrant,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_derivation import (
    ModelBrokerGrantDerivation,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_group_prefix import (
    ModelBrokerGroupPrefix,
)
from omnibase_infra.runtime.gateway_forwarder import (
    build_gateway_transports,
    load_gateway_forwarder_runtime_config,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[4]
BINDINGS = ROOT / "deploy/lab/developer-principal-bindings.yaml"
DERIVED = ROOT / "deploy/lab/developer-principal-derived-grants.json"
LANE_MIRROR_OVERLAY = ROOT / "docker/lane-overlays/local.lane-mirror.example.yaml"
LANE_MIRROR_CREDENTIALS = (
    ROOT / "docker/lane-overlays/local.lane-credentials.example.yaml"
)
LANE_MIRROR_BROKER = "lane.dev.kafka.broker"

# The grants `deploy/lab/developer-principal-grants.yaml` typed by hand until
# OMN-19975, spelled here as that file spelled them: (resource, name, pattern,
# operations). The derivation must reproduce exactly this set from the bindings.
TYPED_GRANTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "topic",
        "onex.cmd.omnimarket.delegate-skill.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.delegate-skill-completed.v1",
        "literal",
        ("read", "write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.delegate-skill-failed.v1",
        "literal",
        ("read", "write", "describe"),
    ),
    ("group", "local.omnibase_core.runtime_local_", "prefixed", ("read", "describe")),
    (
        "topic",
        "onex.cmd.omnimarket.work-ledger-append-from-{principal}.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.work-ledger-append-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.work-ledger-append-failed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "group",
        "local.omnimarket.work_ledger_append_client.consume.v1.",
        "prefixed",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omnimarket.pr-handoff-requested.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.pr-handoff-accepted.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.pr-handoff-handed-off.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.pr-handoff-failed.v1",
        "literal",
        ("read", "describe"),
    ),
    ("group", "pr-handoff-request.", "prefixed", ("read", "describe")),
    (
        "topic",
        "onex.cmd.omnimarket.github-repo-gateway-read-requested.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.github-repo-gateway-read-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "group",
        "local.omnimarket.scm_command_client.consume.v1.",
        "prefixed",
        ("read", "describe"),
    ),
    ("group", "*", "literal", ("describe",)),
)
# The grants added by declaring a binding after OMN-19975 (OMN-21045): the memory
# lifecycle proof publishes expire-memory and reads the two outcomes under its own
# group prefix. The pr-polish start command and its two result topics (OMN-21034): the
# group they are read under is the `local.omnibase_core.runtime_local_` prefix above.
# The morning ground-state run (OMN-20673) adds its start command, its two terminals,
# the delegation request and the reply under the morning_phase reply group.
# The morning friction sweep caller (OMN-21400) adds its start command, its two terminals read
# and the reply group prefix of its per-call request_response instance.
# The DoD closeout proof (OMN-21048) publishes the decide command and reads
# its decided event under its own group prefix.
# The laptop lane mirror (OMN-19983) writes, as the machine's issued principal, the
# delegation topic set its overlay selects onto the dev lane.
DECLARED_GRANTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "topic",
        "onex.cmd.omnimarket.work-ledger-query-requested.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.work-ledger-query-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "group",
        "local.omnimarket.work_ledger_query_client.consume.v1.",
        "prefixed",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omnimarket.pr-polish-start.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.pr-polish-phase-transition.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.pr-polish-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omnimemory.expire-memory.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimemory.memory-expired.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimemory.lifecycle-transition-failed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "group",
        "local.omnimemory.memory_lifecycle_probe.consume.v1.",
        "prefixed",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omnimarket.morning-ground-state-start.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.morning-ground-state-completed.v1",
        "literal",
        ("read", "write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.morning-ground-state-failed.v1",
        "literal",
        ("read", "write", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omnibase-infra.delegation-request.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-infra.delegation-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-infra.delegation-failed.v1",
        "literal",
        ("read", "describe"),
    ),
    ("group", "local.rr.morning_phase.", "prefixed", ("read", "describe")),
    (
        "topic",
        "onex.cmd.omnibase-internal.morning-friction-sweep.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-internal.morning-friction-sweep-completed.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-internal.morning-friction-sweep-failed.v1",
        "literal",
        ("read", "describe"),
    ),
    ("group", "local.rr.friction-run-", "prefixed", ("read", "describe")),
    (
        "topic",
        "onex.cmd.omnimarket.dod-closeout-sweep-decide-requested.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.dod-closeout-sweep-decided.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "group",
        "local.omnimarket.dod_closeout_probe.consume.v1.",
        "prefixed",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.projection-delegation-applied.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-infra.delegation-completed.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnibase-infra.delegation-failed.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.evt.omniintelligence.llm-call-completed.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omniintelligence.context-compaction-requested.v1",
        "literal",
        ("write", "describe"),
    ),
    (
        "topic",
        "onex.cmd.omniintelligence.context-compaction-package.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.context-compaction-prepared.v1",
        "literal",
        ("read", "describe"),
    ),
    (
        "topic",
        "onex.evt.omnimarket.context-compaction-fallback.v1",
        "literal",
        ("read", "describe"),
    ),
    ("group", "receipt-mode-client.", "prefixed", ("read", "describe")),
)
# The Claude Code hook edge (OMN-17427): a developer machine's emit drainer publishes
# its own hook events to the dev lane, so the lab projections (claude_hook_events among
# them) see every person's sessions and not only the operator's. These are the
# `onex.evt.omniclaude.*` observability topics the operator's own hook principal holds,
# produce only. No READ, and no `onex.cmd.*` hook topic: the command topics carry full
# prompts and content and stay a separate, consented grant.
HOOK_EDGE_TOPICS: tuple[str, ...] = (
    "onex.evt.omniclaude.agent-actions.v1",
    "onex.evt.omniclaude.agent-status.v1",
    "onex.evt.omniclaude.correlation-trace.v1",
    "onex.evt.omniclaude.delegation-shadow-comparison.v1",
    "onex.evt.omniclaude.dod-guard-fired.v1",
    "onex.evt.omniclaude.friction-observed.v1",
    "onex.evt.omniclaude.hook-event.v1",
    "onex.evt.omniclaude.hook-health-error.v1",
    "onex.evt.omniclaude.intent-commit-bound.v1",
    "onex.evt.omniclaude.llm-routing-decision.v1",
    "onex.evt.omniclaude.llm-routing-fallback.v1",
    "onex.evt.omniclaude.pattern-enforcement.v1",
    "onex.evt.omniclaude.phase-metrics.v1",
    "onex.evt.omniclaude.prompt-submitted.v1",
    "onex.evt.omniclaude.pr-validation-rollup.v1",
    "onex.evt.omniclaude.routing-decision.v1",
    "onex.evt.omniclaude.routing-feedback.v1",
    "onex.evt.omniclaude.session-ended.v1",
    "onex.evt.omniclaude.session-outcome.v1",
    "onex.evt.omniclaude.session-started.v1",
    "onex.evt.omniclaude.skill-completed.v1",
    "onex.evt.omniclaude.skill-invoked.v1",
    "onex.evt.omniclaude.skill-started.v1",
    "onex.evt.omniclaude.static-context-edit-detected.v1",
    "onex.evt.omniclaude.team-evidence-written.v1",
    "onex.evt.omniclaude.team-task-assigned.v1",
    "onex.evt.omniclaude.team-task-completed.v1",
    "onex.evt.omniclaude.team-task-progress.v1",
    "onex.evt.omniclaude.tool-executed.v1",
)
HOOK_EDGE_GRANTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = tuple(
    ("topic", topic, "literal", ("write", "describe")) for topic in HOOK_EDGE_TOPICS
)
# The compaction receipt-mode replies (OMN-21099): the Claude Code hook's `compaction`
# binding awaits these three topics (node_claude_code_hook_adapter_effect contract,
# `awaited_events`) through HandlerReceiptModePublishAwait, each under a fresh group
# `receipt-mode-client.<12 hex>`. READ and DESCRIBE only on the replies; the client
# publishes its request to the command topic below, WRITE and DESCRIBE.
COMPACTION_COMMAND_TOPIC = "onex.cmd.omniintelligence.context-compaction-requested.v1"
COMPACTION_REPLY_TOPICS: tuple[str, ...] = (
    "onex.cmd.omniintelligence.context-compaction-package.v1",
    "onex.evt.omnimarket.context-compaction-prepared.v1",
    "onex.evt.omnimarket.context-compaction-fallback.v1",
)
BROKER = "dev"


def typed_rows() -> set[tuple[str, str, str, str, str]]:
    return {
        (BROKER, resource.upper(), name, pattern.upper(), operation.upper())
        for resource, name, pattern, operations in (
            TYPED_GRANTS + DECLARED_GRANTS + HOOK_EDGE_GRANTS
        )
        for operation in operations
    }


def derive(bindings: ModelBrokerGrantBindings) -> set[tuple[str, str, str, str, str]]:
    return {
        (g.broker, g.resource_type, g.resource, g.pattern, g.operation)
        for g in HandlerBrokerGrantDerive().handle(bindings).grants
    }


def shipped_bindings() -> ModelBrokerGrantBindings:
    return ModelBrokerGrantBindings.model_validate(yaml.safe_load(BINDINGS.read_text()))


def test_bindings_derive_exactly_the_grants_that_were_typed_by_hand() -> None:
    assert derive(shipped_bindings()) == typed_rows()


def lane_mirror_rows() -> set[tuple[str, str, str, str, str]]:
    """The grants the shipped laptop lane-mirror overlay derives on the dev lane broker."""
    config = load_gateway_forwarder_runtime_config(
        LANE_MIRROR_OVERLAY,
        broker_ref_map_path=None,
        lane_credential_map_path=LANE_MIRROR_CREDENTIALS,
    )
    resolved, _ = build_gateway_transports(config)
    return {
        (BROKER, g.resource_type, g.resource, g.pattern, g.operation)
        for g in HandlerBrokerGrantDerive().handle(resolved).grants
        if g.broker == LANE_MIRROR_BROKER
    }


def test_the_issued_principal_holds_every_dev_grant_the_laptop_lane_mirror_derives() -> (
    None
):
    """The laptop forwarder writes the lab lane as the machine's issued principal.

    Falsifier: drop a delegation produce binding from the bindings file and the
    mirrored topic it covers is named missing here.
    """
    mirror = lane_mirror_rows()
    config = load_gateway_forwarder_runtime_config(
        LANE_MIRROR_OVERLAY,
        broker_ref_map_path=None,
        lane_credential_map_path=LANE_MIRROR_CREDENTIALS,
    )
    assert config.forwarder.lane_mirror is not None
    mirrored = set(config.forwarder.lane_mirror.topics)
    # Positive control: the overlay derives a write for every mirrored topic, so an
    # empty derivation cannot pass the subset check below.
    assert {r[2] for r in mirror if r[4] == "WRITE"} == mirrored
    assert sorted(mirror - derive(shipped_bindings())) == []


def test_the_laptop_lane_mirror_writes_only_the_mirrored_topics() -> None:
    """The lane-mirror grants are topic WRITE and DESCRIBE, never a read or a group."""
    assert {(r[1], r[3], r[4]) for r in lane_mirror_rows()} == {
        ("TOPIC", "LITERAL", "WRITE"),
        ("TOPIC", "LITERAL", "DESCRIBE"),
    }


def test_the_memory_lifecycle_grants_are_declared_and_nothing_wider() -> None:
    """The proof may publish expire-memory and read two outcomes, not the memory topics."""
    rows = derive(shipped_bindings())
    memory_topics = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if resource_type == "TOPIC" and ".omnimemory." in resource
    }
    assert memory_topics == {
        ("TOPIC", "onex.cmd.omnimemory.expire-memory.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnimemory.expire-memory.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimemory.memory-expired.v1", "READ"),
        ("TOPIC", "onex.evt.omnimemory.memory-expired.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimemory.lifecycle-transition-failed.v1", "READ"),
        ("TOPIC", "onex.evt.omnimemory.lifecycle-transition-failed.v1", "DESCRIBE"),
    }


def test_the_pr_polish_grants_start_it_and_read_both_phases_and_nothing_more() -> None:
    """A developer principal may start a polish and read its results, never write them."""
    rows = derive(shipped_bindings())
    polish = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if "pr-polish" in resource
    }
    assert polish == {
        ("TOPIC", "onex.cmd.omnimarket.pr-polish-start.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnimarket.pr-polish-start.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimarket.pr-polish-phase-transition.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.pr-polish-phase-transition.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimarket.pr-polish-completed.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.pr-polish-completed.v1", "DESCRIBE"),
    }
    # The result group is covered by the runtime_local_ prefix the principal holds, so
    # the polish adds no group row.
    assert not any(
        resource_type == "GROUP" and "polish" in resource
        for _, resource_type, resource, _, _ in rows
    )


def test_the_morning_grants_are_declared_and_nothing_wider() -> None:
    """The run may start the node, read and write its terminals, and delegate its phases."""
    rows = derive(shipped_bindings())
    morning = (
        {
            (resource_type, resource, operation)
            for _, resource_type, resource, _, operation in rows
            if "morning" in resource or "delegation" in resource
        }
        - {
            (resource_type, resource, operation)
            for _, resource_type, resource, _, operation in rows
            if "delegate-skill" in resource
            or "projection-delegation" in resource
            or resource in HOOK_EDGE_TOPICS
        }
        - {
            (resource_type, resource, operation)
            for _, resource_type, resource, _, operation in lane_mirror_rows()
            if operation == "WRITE"
        }
    )
    assert morning == {
        ("TOPIC", "onex.cmd.omnimarket.morning-ground-state-start.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnimarket.morning-ground-state-start.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-completed.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-completed.v1", "WRITE"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-completed.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-failed.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-failed.v1", "WRITE"),
        ("TOPIC", "onex.evt.omnimarket.morning-ground-state-failed.v1", "DESCRIBE"),
        ("TOPIC", "onex.cmd.omnibase-infra.delegation-request.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnibase-infra.delegation-request.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnibase-infra.delegation-completed.v1", "READ"),
        ("TOPIC", "onex.evt.omnibase-infra.delegation-completed.v1", "DESCRIBE"),
        ("TOPIC", "onex.evt.omnibase-infra.delegation-failed.v1", "READ"),
        ("TOPIC", "onex.evt.omnibase-infra.delegation-failed.v1", "DESCRIBE"),
        ("GROUP", "local.rr.morning_phase.", "READ"),
        ("GROUP", "local.rr.morning_phase.", "DESCRIBE"),
        ("TOPIC", "onex.cmd.omnibase-internal.morning-friction-sweep.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnibase-internal.morning-friction-sweep.v1", "DESCRIBE"),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-completed.v1",
            "READ",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-completed.v1",
            "DESCRIBE",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-failed.v1",
            "READ",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-failed.v1",
            "DESCRIBE",
        ),
    }


def test_the_friction_sweep_grants_start_it_read_both_terminals_and_write_neither() -> (
    None
):
    """The caller publishes the start command, reads under its own reply group prefix only."""
    rows = derive(shipped_bindings())
    friction = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if "friction-sweep" in resource or "friction-run-" in resource
    }
    assert friction == {
        ("TOPIC", "onex.cmd.omnibase-internal.morning-friction-sweep.v1", "WRITE"),
        ("TOPIC", "onex.cmd.omnibase-internal.morning-friction-sweep.v1", "DESCRIBE"),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-completed.v1",
            "READ",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-completed.v1",
            "DESCRIBE",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-failed.v1",
            "READ",
        ),
        (
            "TOPIC",
            "onex.evt.omnibase-internal.morning-friction-sweep-failed.v1",
            "DESCRIBE",
        ),
        ("GROUP", "local.rr.friction-run-", "READ"),
        ("GROUP", "local.rr.friction-run-", "DESCRIBE"),
    }


def test_the_dod_closeout_grants_are_declared_and_nothing_wider() -> None:
    """The proof may publish the decide command and read the decided event only."""
    rows = derive(shipped_bindings())
    closeout = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if resource_type == "TOPIC" and "dod-closeout" in resource
    }
    assert closeout == {
        (
            "TOPIC",
            "onex.cmd.omnimarket.dod-closeout-sweep-decide-requested.v1",
            "WRITE",
        ),
        (
            "TOPIC",
            "onex.cmd.omnimarket.dod-closeout-sweep-decide-requested.v1",
            "DESCRIBE",
        ),
        ("TOPIC", "onex.evt.omnimarket.dod-closeout-sweep-decided.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.dod-closeout-sweep-decided.v1", "DESCRIBE"),
    }


def test_committed_derived_grants_are_the_nodes_output() -> None:
    committed = ModelBrokerGrantDerivation.model_validate_json(DERIVED.read_text())
    # Equality with the bindings alone also accepts a stale pair of files. Check
    # the shipped permissions independently before comparing the serialization.
    assert {
        (g.broker, g.resource_type, g.resource, g.pattern, g.operation)
        for g in committed.grants
    } == typed_rows()
    assert len(committed.grants) == len(typed_rows())
    derivation = HandlerBrokerGrantDerive().handle(shipped_bindings())
    assert DERIVED.read_text() == derivation.model_dump_json(indent=2) + "\n", (
        "developer-principal-derived-grants.json is not the node's output; regenerate: "
        'uv run python -c "import yaml; from omnibase_infra.handlers.handler_broker_grant_derive '
        "import HandlerBrokerGrantDerive as H; from omnibase_infra.nodes."
        "node_broker_grant_derive_compute.models.model_broker_grant_bindings import "
        "ModelBrokerGrantBindings as B; print(H().handle(B.model_validate(yaml.safe_load("
        "open('deploy/lab/developer-principal-bindings.yaml')))).model_dump_json(indent=2))\" "
        "> deploy/lab/developer-principal-derived-grants.json"
    )


def consume(group: str, broker: str = BROKER) -> ModelBusBinding:
    return ModelBusBinding(
        broker=broker,
        physical_topic="fixture.topic",
        direction="consume",
        consumer_group=group,
    )


def test_a_consumed_group_under_a_prefix_gets_no_exact_grant() -> None:
    covered = ModelBrokerGrantBindings(
        principal="p",
        bindings=(consume("tail.call-1"),),
        group_prefixes=(ModelBrokerGroupPrefix(broker=BROKER, prefix="tail."),),
    )
    assert not any(row[1] == "GROUP" and row[3] == "LITERAL" for row in derive(covered))
    assert (BROKER, "GROUP", "tail.", "PREFIXED", "READ") in derive(covered)
    assert (BROKER, "GROUP", "tail.", "PREFIXED", "DESCRIBE") in derive(covered)


@pytest.mark.parametrize(
    ("prefix_broker", "prefix"), [("other", "tail."), (BROKER, "head.")]
)
def test_a_prefix_on_another_broker_or_another_name_covers_nothing(
    prefix_broker: str, prefix: str
) -> None:
    uncovered = ModelBrokerGrantBindings(
        principal="p",
        bindings=(consume("tail.call-1"),),
        group_prefixes=(ModelBrokerGroupPrefix(broker=prefix_broker, prefix=prefix),),
    )
    assert (BROKER, "GROUP", "tail.call-1", "LITERAL", "READ") in derive(uncovered)


def test_a_prefix_is_never_inferred_from_a_binding() -> None:
    plain = ModelBrokerGrantBindings(principal="p", bindings=(consume("tail.call-1"),))
    assert {row[3] for row in derive(plain)} == {"LITERAL"}


def test_the_wildcard_group_is_describe_only() -> None:
    inspect = ModelBrokerGrantBindings(
        principal="p",
        bindings=(),
        described_groups=(ModelBusGroupDescribe(broker=BROKER, group="*"),),
    )
    assert derive(inspect) == {(BROKER, "GROUP", "*", "LITERAL", "DESCRIBE")}
    with pytest.raises(ValidationError):
        HandlerBrokerGrantDerive().handle(
            ModelBrokerGrantBindings(principal="p", bindings=(consume("*"),))
        )


@pytest.mark.parametrize(
    ("resource_type", "operation", "pattern"),
    [
        ("TOPIC", "DESCRIBE", "LITERAL"),
        ("GROUP", "READ", "LITERAL"),
        ("GROUP", "DESCRIBE", "PREFIXED"),
    ],
)
def test_a_wildcard_grant_other_than_group_describe_is_not_a_grant(
    resource_type: str, operation: str, pattern: str
) -> None:
    with pytest.raises(ValidationError):
        ModelBrokerGrant.model_validate(
            {
                "broker": BROKER,
                "resource": "*",
                "resource_type": resource_type,
                "operation": operation,
                "pattern": pattern,
            }
        )


def test_the_contract_takes_the_prefix_carrying_input() -> None:
    contract = yaml.safe_load(
        (
            ROOT
            / "src/omnibase_infra/nodes/node_broker_grant_derive_compute/contract.yaml"
        ).read_text()
    )
    assert contract["input_model"]["name"] == "ModelBrokerGrantBindings"
    request = ModelBrokerGrantBindings.model_validate_json(
        shipped_bindings().model_dump_json()
    )
    assert request == shipped_bindings()


def test_the_delegation_terminal_readback_grants_read_one_topic_and_write_none() -> (
    None
):
    """The readback may read and describe the projection terminal, never write it."""
    rows = derive(shipped_bindings())
    readback = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if "projection-delegation" in resource
    }
    assert readback == {
        ("TOPIC", "onex.evt.omnimarket.projection-delegation-applied.v1", "READ"),
        ("TOPIC", "onex.evt.omnimarket.projection-delegation-applied.v1", "DESCRIBE"),
    }


def test_the_hook_edge_grants_produce_only_hook_events() -> None:
    """A developer machine may publish its hook events and read none of them back."""
    rows = derive(shipped_bindings())
    hook_edge = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if resource_type == "TOPIC" and ".omniclaude." in resource
    }
    assert hook_edge == {
        ("TOPIC", topic, operation)
        for topic in HOOK_EDGE_TOPICS
        for operation in ("WRITE", "DESCRIBE")
    }
    assert "onex.evt.omniclaude.hook-event.v1" in HOOK_EDGE_TOPICS


def test_the_non_mirror_omniintelligence_grants_are_exactly_compaction() -> None:
    """The mirror and compaction are independently checked exceptions."""
    rows = derive(shipped_bindings())
    mirror = {(r[1], r[2]) for r in lane_mirror_rows()}
    # The only omniintelligence topics outside the mirror are the compaction request and
    # the one omniintelligence-named compaction reply; the exclusion is exact, not a
    # blanket pass for anything compaction-shaped.
    non_mirror_omniintelligence = {
        resource
        for _, resource_type, resource, _, _ in rows
        if resource_type == "TOPIC"
        and "omniintelligence" in resource
        and (resource_type, resource) not in mirror
    }
    assert non_mirror_omniintelligence == {
        COMPACTION_COMMAND_TOPIC,
        *(t for t in COMPACTION_REPLY_TOPICS if "omniintelligence" in t),
    }


def test_the_compaction_replies_grant_read_on_three_topics_and_one_group_prefix() -> (
    None
):
    """The receipt-mode client reads its replies, never writes them, and writes its request."""
    rows = derive(shipped_bindings())
    compaction = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if ("context-compaction" in resource and resource != COMPACTION_COMMAND_TOPIC)
        or resource.startswith("receipt-mode")
    }
    compaction_command = {
        (resource_type, resource, operation)
        for _, resource_type, resource, _, operation in rows
        if resource == COMPACTION_COMMAND_TOPIC
    }
    assert compaction == {
        (resource_type, resource, operation)
        for resource_type, resource in (
            *(("TOPIC", topic) for topic in COMPACTION_REPLY_TOPICS),
            ("GROUP", "receipt-mode-client."),
        )
        for operation in ("READ", "DESCRIBE")
    }
    assert COMPACTION_COMMAND_TOPIC not in COMPACTION_REPLY_TOPICS
    assert compaction_command == {
        ("TOPIC", COMPACTION_COMMAND_TOPIC, "WRITE"),
        ("TOPIC", COMPACTION_COMMAND_TOPIC, "DESCRIBE"),
    }


@pytest.mark.parametrize("suffix", ["000000000000", "abcdef123456", "ffffffffffff"])
def test_compaction_replies_use_prefix_grants_for_concrete_runtime_groups(
    suffix: str,
) -> None:
    """Real per-call groups produce the same prefix rows as the representative binding."""
    shipped = shipped_bindings()
    concrete = shipped.model_copy(
        update={
            "bindings": tuple(
                binding.model_copy(
                    update={"consumer_group": f"receipt-mode-client.{suffix}"}
                )
                if binding.physical_topic in COMPACTION_REPLY_TOPICS
                else binding
                for binding in shipped.bindings
            )
        }
    )
    assert derive(concrete) == derive(shipped)
    assert not any(
        row[1] == "GROUP" and row[2] == f"receipt-mode-client.{suffix}"
        for row in derive(concrete)
    )


def test_compaction_replies_add_only_the_declared_dev_grants() -> None:
    """Removing the compaction declarations removes exactly ten ACL rows."""
    shipped = shipped_bindings()
    topics = {COMPACTION_COMMAND_TOPIC, *COMPACTION_REPLY_TOPICS}
    without = shipped.model_copy(
        update={
            "bindings": tuple(
                binding
                for binding in shipped.bindings
                if binding.physical_topic not in topics
            ),
            "group_prefixes": tuple(
                prefix
                for prefix in shipped.group_prefixes
                if prefix.prefix != "receipt-mode-client."
            ),
        }
    )
    expected = (
        {
            (BROKER, "TOPIC", topic, "LITERAL", operation)
            for topic in COMPACTION_REPLY_TOPICS
            for operation in ("READ", "DESCRIBE")
        }
        | {
            (BROKER, "TOPIC", COMPACTION_COMMAND_TOPIC, "LITERAL", operation)
            for operation in ("WRITE", "DESCRIBE")
        }
        | {
            (BROKER, "GROUP", "receipt-mode-client.", "PREFIXED", operation)
            for operation in ("READ", "DESCRIBE")
        }
    )
    assert derive(shipped) - derive(without) == expected
    assert derive(without) - derive(shipped) == set()
    assert len(expected) == 10
    assert not any(row[0] != BROKER for row in expected)


def test_compaction_replies_are_not_implicitly_granted_to_other_principals() -> None:
    """The compute handler derives only the bindings supplied for that principal."""
    unrelated = ModelBrokerGrantBindings(
        principal="unrelated-service", bindings=(consume("unrelated-group", "prod"),)
    )
    rows = derive(unrelated)
    assert rows == {
        ("prod", "TOPIC", "fixture.topic", "LITERAL", "READ"),
        ("prod", "TOPIC", "fixture.topic", "LITERAL", "DESCRIBE"),
        ("prod", "GROUP", "unrelated-group", "LITERAL", "READ"),
    }


def test_developer_principal_query_acl_grants_are_exact_and_client_scoped() -> None:
    """A query writes one request and reads one terminal under its own prefix."""
    bindings = shipped_bindings()
    prefix = "local.omnimarket.work_ledger_query_client.consume.v1."
    request = "onex.cmd.omnimarket.work-ledger-query-requested.v1"
    terminal = "onex.evt.omnimarket.work-ledger-query-completed.v1"
    query_bindings = tuple(
        binding
        for binding in bindings.bindings
        if binding.physical_topic in (request, terminal)
    )
    assert {
        (
            binding.broker,
            binding.physical_topic,
            binding.direction,
            binding.consumer_group,
        )
        for binding in query_bindings
    } == {
        (BROKER, request, "produce", None),
        (BROKER, terminal, "consume", prefix + "call"),
    }
    before = bindings.model_copy(
        update={
            "bindings": tuple(b for b in bindings.bindings if b not in query_bindings),
            "group_prefixes": tuple(
                p for p in bindings.group_prefixes if p.prefix != prefix
            ),
        }
    )
    expected = {
        (BROKER, "TOPIC", request, "LITERAL", "WRITE"),
        (BROKER, "TOPIC", request, "LITERAL", "DESCRIBE"),
        (BROKER, "TOPIC", terminal, "LITERAL", "READ"),
        (BROKER, "TOPIC", terminal, "LITERAL", "DESCRIBE"),
        (BROKER, "GROUP", prefix, "PREFIXED", "READ"),
        (BROKER, "GROUP", prefix, "PREFIXED", "DESCRIBE"),
    }
    after = derive(bindings)
    assert after - derive(before) == expected
    assert derive(before) - after == set()
    assert len(after) == len(derive(before)) + len(expected)
    committed = {
        (g["broker"], g["resource_type"], g["resource"], g["pattern"], g["operation"])
        for g in json.loads(DERIVED.read_text())["grants"]
    }
    assert committed == after


def test_developer_principal_query_acl_prefix_is_the_derived_client_group() -> None:
    """The query prefix is core's derived client group and covers no runtime group."""
    query_client = compute_consumer_group_id(
        env="local",
        service="omnimarket",
        node_name="work_ledger_query_client",
        version="v1",
    )
    append_client = compute_consumer_group_id(
        env="local",
        service="omnimarket",
        node_name="work_ledger_append_client",
        version="v1",
    )
    query_node = compute_consumer_group_id(
        env="local",
        service="omnimarket",
        node_name="work_ledger_query_effect",
        version="v1",
    )
    dev_prefixes = [
        p.prefix for p in shipped_bindings().group_prefixes if p.broker == BROKER
    ]
    assert f"{query_client}." in dev_prefixes
    assert f"{append_client}." in dev_prefixes
    # A PREFIXED group ACL covers a group whose id starts with the prefix.
    covering = [p for p in dev_prefixes if f"{query_client}.call".startswith(p)]
    assert covering == [f"{query_client}."]
    for group in (query_node, f"{query_node}.worker"):
        assert not any(group.startswith(prefix) for prefix in dev_prefixes), group
    assert not f"{append_client}.call".startswith(f"{query_client}.")


def test_h105_scm_pr_create_derived_file_is_exactly_the_nodes_output() -> None:
    source = ROOT / "deploy/lab/h105-scm-pr-create-bindings.yaml"
    bindings = ModelBrokerGrantBindings.model_validate(
        yaml.safe_load(source.read_text())
    )
    derived = source.with_name("h105-scm-pr-create-derived-grants.json")
    assert bindings.principal == "dev-jonah-omnibook"
    assert (
        derived.read_text()
        == HandlerBrokerGrantDerive().handle(bindings).model_dump_json(indent=2) + "\n"
    )
    assert derive(bindings) == {
        ("dev", resource_type, resource, pattern, operation)
        for resource_type, resource, pattern in (
            ("TOPIC", "onex.evt.omnimarket.github-repo-write-completed.v1", "LITERAL"),
            ("TOPIC", "onex.evt.omnimarket.github-repo-write-failed.v1", "LITERAL"),
            ("GROUP", "local.omnimarket.scm_command_client.consume.v1.", "PREFIXED"),
        )
        for operation in ("READ", "DESCRIBE")
    }
