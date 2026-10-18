# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The grant derivation expresses prefixed group grants and the describe-only wildcard."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

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
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_group_prefix import (
    ModelBrokerGroupPrefix,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[4]
BINDINGS = ROOT / "deploy/lab/developer-principal-bindings.yaml"
DERIVED = ROOT / "deploy/lab/developer-principal-derived-grants.json"

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
# The DoD closeout proof (OMN-21048) publishes the decide command and reads
# its decided event under its own group prefix.
DECLARED_GRANTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
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
)
BROKER = "dev"


def typed_rows() -> set[tuple[str, str, str, str, str]]:
    return {
        (BROKER, resource.upper(), name, pattern.upper(), operation.upper())
        for resource, name, pattern, operations in TYPED_GRANTS + DECLARED_GRANTS
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
