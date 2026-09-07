# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED-on-parent proof for the OMN-18013 topic/category/event-type closure gates.

Every test in this module FAILS on the parent commit (eee49719c) and PASSES on
this one. They are written against the DEFECT, not against the fix: each builds
the exact contract or envelope shape that was live on the parent and asserts the
new behaviour, so a revert of any single piece of OMN-18013 turns one of these
red.

The five gates, and the parent-commit fact each one contradicts:

1. ``contract-topic-category`` — the parent derived ONE category per handler
   entry from ``subscribe_topics[0]`` (a LIST POSITION) and stamped it on every
   route; ``_derive_message_category`` returned an unconditional ``"event"`` for
   any unrecognised kind segment. 32 live topics sat in the resulting gap.
2. ``handler-event-type-source`` — the parent's consume boundary let an untyped
   payload field ``data["event_type"]`` override the topic-derived event type.
3. ``contract-topic-graph`` — the parent's copy lived in omnimarket, had a
   ``--baseline`` flag, and 698 defects were frozen in it.
4. ``no-literal-event-type-in-tests`` — the parent had 12 sites in this repo
   feeding a TOPIC as an envelope ``event_type``; the bus carries the alias.
5. ``no-baseline-refreeze`` — the parent carried two baseline files this commit
   deletes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_message_category,
    _topics_for_handler_entry,
    derive_route_message_category,
)
from omnibase_infra.testing.publisher_contract_fixture import (
    PublisherContractCorpus,
    PublisherTopicError,
)
from omnibase_infra.validators import (
    contract_topic_category,
    handler_event_type_source,
    no_baseline_refreeze,
    no_literal_event_type_in_tests,
    subscriber_dispatcher_resolution,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "omnibase_infra"


def _contracts() -> list[object]:
    paths = sorted(
        p
        for p in SRC.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(paths)
    return list(getattr(discovered, "contracts", discovered))


def _contract_named(name: str) -> object:
    match = [c for c in _contracts() if c.name == name]
    assert len(match) == 1, f"expected exactly one contract named {name!r}"
    return match[0]


# --------------------------------------------------------------------------
# GATE 1 — category derived per topic, never from list position
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_sibling_event_topic_registers_as_event_not_position_zero_category() -> None:
    """A .evt. sibling of a .cmd. subscribe_topics[0] registers as EVENT.

    RED ON PARENT: ``derive_entry_message_category`` returned
    ``_derive_message_category(subscribe_topics[0])`` -> ``"command"`` and that
    single value was stamped on EVERY route, so this topic's route was
    ``command`` and ``_find_matching_dispatchers`` — which filters on the real
    category of the arriving topic — dropped every message on it.
    """
    contract = _contract_named("node_coding_agent_orchestrator")
    routing = contract.handler_routing
    assert routing is not None
    first_topic = contract.event_bus.subscribe_topics[0]
    assert first_topic.split(".")[1] == "cmd", (
        "fixture premise: subscribe_topics[0] must be a COMMAND topic for this to "
        "distinguish per-topic derivation from position-0 derivation"
    )

    seen: dict[str, EnumMessageCategory] = {}
    for entry in routing.handlers:
        for topic in _topics_for_handler_entry(contract, entry):
            seen[topic] = derive_route_message_category(contract, entry, topic)

    evt_topics = [t for t in seen if t.split(".")[1] == "evt"]
    assert evt_topics, "fixture premise: the contract must subscribe to a .evt. sibling"
    for topic in evt_topics:
        assert seen[topic] is EnumMessageCategory.EVENT, (
            f"{topic} registered as {seen[topic]}, not EVENT — the category came "
            "from somewhere other than the topic's own name"
        )
    for topic in (t for t in seen if t.split(".")[1] == "cmd"):
        assert seen[topic] is EnumMessageCategory.COMMAND


@pytest.mark.unit
def test_dlq_topic_has_no_default_category() -> None:
    """An undecodable topic yields None, never a silent ``"event"``.

    RED ON PARENT: ``_derive_message_category`` ended in an unconditional
    ``return "event"``, so ``onex.dlq.omnibase-infra.router.v1`` registered as
    EVENT while ``EnumMessageCategory.from_topic`` returned ``None`` at dispatch
    and the engine rejected the message as an invalid topic category. The two
    derivations disagreed on 32 live topics.
    """
    assert _derive_message_category("onex.dlq.omnibase-infra.router.v1") is None
    assert _derive_message_category("onex.snapshot.projection.live-events.v1") is None
    # ...and the two derivations now agree everywhere, by construction.
    for topic in (
        "onex.evt.platform.node-heartbeat.v1",
        "onex.cmd.platform.ledger-append.v1",
        "onex.intent.platform.runtime-tick.v1",
        "onex.dlq.omnibase-infra.commands.v1",
        "onex.dlq.omnibase-infra.router.v1",
    ):
        derived = _derive_message_category(topic)
        engine = EnumMessageCategory.from_topic(topic)
        assert derived == (None if engine is None else engine.value)


@pytest.mark.unit
def test_declared_category_contradicting_its_topic_is_refused() -> None:
    """``message_category:`` may not contradict the topic's own name.

    RED ON PARENT: an explicit ``entry.message_category`` won unconditionally,
    so ``node_ledger_projection_compute`` declaring ``"event"`` for
    ``onex.dlq.omnibase-infra.commands.v1`` (whose name derives ``command``)
    registered a route no message could reach, silently.
    """
    contract = _contract_named("node_ledger_projection_compute")
    entry = next(
        e
        for e in contract.handler_routing.handlers
        if (e.topic or "") == "onex.dlq.omnibase-infra.commands.v1"
    )
    assert (
        derive_route_message_category(
            contract, entry, "onex.dlq.omnibase-infra.commands.v1"
        )
        is EnumMessageCategory.COMMAND
    )

    class _Lie:
        topic = "onex.evt.platform.node-heartbeat.v1"
        message_category = "command"
        event_type = None
        event_model = None
        handler = None

    with pytest.raises(ModelOnexError, match="derives 'event'"):
        derive_route_message_category(
            contract, _Lie(), "onex.evt.platform.node-heartbeat.v1"
        )


@pytest.mark.unit
def test_contract_topic_category_gate_is_green_at_zero_with_no_baseline_flag() -> None:
    """The gate passes with ZERO findings and has no baseline to be frozen with.

    RED ON PARENT: the equivalent gate was ``mixed_category_routing``, it took a
    ``--baseline`` argument, and ``config/validation/mixed_category_routing_baseline.yaml``
    held 4 frozen entries.
    """
    findings, count = contract_topic_category.scan(SRC)
    assert findings == [], [f.detail for f in findings]
    assert count >= contract_topic_category.SCOPES["omnibase_infra"][1]
    source = Path(contract_topic_category.__file__).read_text(encoding="utf-8")
    assert '"--baseline"' not in source, "the gate regrew an argparse baseline flag"


# --------------------------------------------------------------------------
# GATE 2 — the handler's event type is read from the publisher's contract
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_payload_event_type_override_is_gone_from_the_consume_boundary() -> None:
    """The consume boundary no longer lets a payload field re-key a message.

    RED ON PARENT: ``handler_wiring`` read ``explicit_event_type =
    data.get("event_type")`` and, when truthy, copied it onto the envelope,
    overriding the topic-derived value. An uncontracted string in someone else's
    payload therefore chose which dispatcher ran.
    """
    source = (SRC / "runtime" / "auto_wiring" / "handler_wiring.py").read_text(
        encoding="utf-8"
    )
    assert "explicit_event_type = (" not in source
    assert 'data.get("event_type")' not in source


@pytest.mark.unit
def test_handler_matches_the_alias_the_bus_carries_not_a_typed_suffix() -> None:
    """The coding-agent orchestrator matches the contract-derived alias.

    RED ON PARENT: it branched on ``event_type.endswith("workspace-validated.v1")``
    — a suffix of a TOPIC. The auto-wired boundary stamps the ALIAS, so that
    branch was reachable only through the payload override removed above.
    """
    from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
        EVENT_TYPE_WORKSPACE_VALIDATED,
    )

    contract = _contract_named("node_coding_agent_orchestrator")
    topic = next(
        t
        for t in contract.event_bus.subscribe_topics
        if t.endswith("coding-agent-workspace-validated.v1")
    )
    assert derive_event_type_alias_for_topic(topic) == EVENT_TYPE_WORKSPACE_VALIDATED
    assert topic != EVENT_TYPE_WORKSPACE_VALIDATED


@pytest.mark.unit
def test_handler_event_type_source_gate_is_green_at_zero() -> None:
    """No entry alias and no handler literal is hand-typed in this repo.

    RED ON PARENT: ``node_coding_agent_orchestrator``'s handler carried the
    ``endswith("workspace-validated.v1")`` literal.
    """
    findings, count = handler_event_type_source.scan(SRC)
    assert findings == [], [f"{f.location}: {f.detail}" for f in findings]
    assert count >= handler_event_type_source.SCOPES["omnibase_infra"][1]


@pytest.mark.unit
def test_the_gate_actually_catches_a_hand_typed_literal(tmp_path: Path) -> None:
    """Positive control: the gate is not vacuously green.

    A zero-finding result means nothing unless the same code returns rows on an
    input known to contain one.
    """
    module = tmp_path / "handlers" / "handler_bad.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "def handle(event_type: str) -> bool:\n"
        '    return event_type == "onex.evt.platform.node-heartbeat.v1"\n',
        encoding="utf-8",
    )
    found = handler_event_type_source.source_literal_findings(tmp_path)
    assert len(found) == 1
    assert found[0].reason == handler_event_type_source.REASON_HAND_TYPED


# --------------------------------------------------------------------------
# GATE 3 — graph closure, re-homed, scoped, and baseline-free
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_contract_topic_graph_is_homed_here_scoped_and_has_no_baseline() -> None:
    """The validator lives in this repo, takes ``--scope``, and cannot be frozen.

    RED ON PARENT: no ``contract_topic_graph`` module existed anywhere in
    omnibase_infra (a grep for it across five repos returned zero files; the same
    grep in omnimarket returned five). The omnimarket copy took ``--baseline``
    and ``--write-baseline`` and carried 698 frozen defects.
    """
    from omnibase_infra.validators import contract_topic_graph

    source = Path(contract_topic_graph.__file__).read_text(encoding="utf-8")
    assert '"--baseline"' not in source
    assert '"--write-baseline"' not in source
    assert "class ModelBaseline" not in source
    assert '"--scope"' in source
    assert "omnibase_infra" in contract_topic_graph.SCOPES
    # omnimarket sits ABOVE omnibase_infra in the layering, so it must be a
    # checkout package here, never an installed dependency.
    assert "omnimarket" in contract_topic_graph.CHECKOUT_PACKAGES
    assert "omnimarket" not in contract_topic_graph.INSTALLED_PACKAGES


# --------------------------------------------------------------------------
# GATE 4 — golden chains build their input from the publisher's contract
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_fixture_event_type_is_the_alias_never_the_topic() -> None:
    """The fixture stamps what the bus carries.

    RED ON PARENT: no fixture existed and 12 sites in this repo passed the topic.
    """
    corpus = PublisherContractCorpus.from_repo_root(SRC)
    topic = "onex.evt.omnibase-infra.coding-agent-completed.v1"
    event_type = corpus.event_type_for(topic)
    assert event_type == derive_event_type_alias_for_topic(topic)
    assert event_type != topic
    envelope = corpus.publisher_envelope(topic, payload={"ok": True})
    assert envelope.event_type == event_type


@pytest.mark.unit
def test_fixture_refuses_a_topic_with_no_declared_publisher() -> None:
    """A chain cannot be built over a topic nothing produces."""
    corpus = PublisherContractCorpus.from_repo_root(SRC)
    with pytest.raises(PublisherTopicError, match="declares itself a publisher"):
        corpus.event_type_for("onex.evt.nobody.invented-for-this-test.v1")


@pytest.mark.unit
def test_no_literal_event_type_in_tests_is_green_at_zero() -> None:
    """No test in this repo feeds a topic as an envelope event_type.

    RED ON PARENT: 12 sites across 5 files did, including 5 in
    ``test_golden_chain_coding_agent.py``.
    """
    findings, module_count = no_literal_event_type_in_tests.scan(REPO_ROOT / "tests")
    assert findings == [], [f"{f.location}: {f.detail}" for f in findings]
    assert module_count >= no_literal_event_type_in_tests.DEFAULT_MIN_TEST_FILES


@pytest.mark.unit
def test_the_test_lint_actually_catches_a_topic_as_event_type(tmp_path: Path) -> None:
    """Positive control for the zero above."""
    module = tmp_path / "test_golden_chain_probe.py"
    body = (
        "def test_x() -> None:\n"
        "    envelope = ModelEventEnvelope(\n"
        '        payload={}, event_type="onex.cmd.platform.ledger-append.v1"\n'
        "    )\n"
        "    assert envelope\n"
    )
    module.write_text(body, encoding="utf-8")
    found = no_literal_event_type_in_tests.findings_for_module(module, body)
    assert len(found) == 1
    assert "is a TOPIC" in found[0].detail


@pytest.mark.unit
def test_the_test_lint_catches_a_topic_reached_through_a_local_variable(
    tmp_path: Path,
) -> None:
    """One indirection is the same defect, and it is how the lint was evaded.

    ``tests/unit/nodes/node_coding_agent/test_real_dispatch_multitopic_routing.py``
    bound the topic to a local and passed the local, so a constant-only reader
    reported the tree clean while the test drove the dispatch path with a
    spelling the bus never carries. Resolution is per SCOPE: a name rebound to a
    call in a DIFFERENT function must not mask this one.
    """
    module = tmp_path / "test_golden_chain_indirect.py"
    body = (
        "def test_other() -> None:\n"
        "    validated_topic = _publish_topics()\n"
        "    assert validated_topic\n"
        "\n"
        "def test_x() -> None:\n"
        '    validated_topic = "onex.evt.platform.ledger-appended.v1"\n'
        "    envelope = ModelEventEnvelope(\n"
        "        payload={}, event_type=validated_topic\n"
        "    )\n"
        "    assert envelope\n"
    )
    module.write_text(body, encoding="utf-8")
    found = no_literal_event_type_in_tests.findings_for_module(module, body)
    assert len(found) == 1, found
    assert "is a TOPIC" in found[0].detail


@pytest.mark.unit
def test_the_test_lint_does_not_flag_a_name_that_is_not_a_topic_literal(
    tmp_path: Path,
) -> None:
    """Negative control: a derived alias is exactly what the gate wants to see."""
    module = tmp_path / "test_golden_chain_alias.py"
    body = (
        "def test_x() -> None:\n"
        '    topic = "onex.evt.platform.ledger-appended.v1"\n'
        "    alias = derive_event_type_alias_for_topic(topic)\n"
        "    envelope = ModelEventEnvelope(payload={}, event_type=alias)\n"
        "    assert envelope\n"
    )
    module.write_text(body, encoding="utf-8")
    assert no_literal_event_type_in_tests.findings_for_module(module, body) == []


# --------------------------------------------------------------------------
# GATE 5 — the burned baselines stay deleted
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_both_burned_baseline_files_are_absent() -> None:
    """The two baselines this commit burned to zero are gone from the tree.

    RED ON PARENT: both files existed, with 22 and 4 frozen rows.
    """
    for rel in (
        "config/validation/subscriber_dispatcher_resolution_baseline.yaml",
        "config/validation/mixed_category_routing_baseline.yaml",
    ):
        assert not (REPO_ROOT / rel).exists(), f"{rel} is back"
    assert no_baseline_refreeze.findings(REPO_ROOT) == []


@pytest.mark.unit
def test_refreeze_guard_actually_fires(tmp_path: Path) -> None:
    """Positive control: recreating a burned baseline is refused."""
    target = tmp_path / "config" / "validation"
    target.mkdir(parents=True)
    (target / "subscriber_dispatcher_resolution_baseline.yaml").write_text(
        "known_unresolved_subscriptions: []\n", encoding="utf-8"
    )
    found = no_baseline_refreeze.findings(tmp_path)
    assert len(found) == 1
    assert "burned to zero and DELETED" in found[0].detail


@pytest.mark.unit
def test_subscriber_dispatcher_resolution_is_zero_and_baseline_free() -> None:
    """The OMN-16939 ratchet is at zero and has no baseline mechanism left."""
    findings, count = subscriber_dispatcher_resolution.scan(SRC)
    assert findings == [], [
        f"{f.contract} :: {f.topic} :: {f.reason}" for f in findings
    ]
    assert count >= subscriber_dispatcher_resolution.DEFAULT_MIN_EXPECTED_CONTRACTS
    source = Path(subscriber_dispatcher_resolution.__file__).read_text(encoding="utf-8")
    assert '"--baseline"' not in source
    assert "def load_baseline" not in source


# --------------------------------------------------------------------------
# Wiring — every gate is a pre-commit hook AND a CI step, in this same commit
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_every_gate_ships_as_both_a_precommit_hook_and_a_ci_step() -> None:
    """Enforcement, not detection: a gate that is not wired is advisory.

    RED ON PARENT: none of these four hook ids or CI step names existed.
    """
    hooks = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for hook_id, ci_step in (
        ("contract-topic-category", "Check contract topic category gate (OMN-18013)"),
        (
            "handler-event-type-source",
            "Check handler event_type source gate (OMN-18013)",
        ),
        (
            "no-literal-event-type-in-tests",
            "Check golden-chain contract input gate (OMN-18013)",
        ),
        ("no-baseline-refreeze", "Check no baseline refreeze (OMN-18013)"),
    ):
        assert f"- id: {hook_id}" in hooks, f"{hook_id} is not a pre-commit hook"
        assert ci_step in ci, f"{ci_step} is not a CI step"
    # The omnibase_infra enforcement surface is the `lint` job under the single
    # required `CI Summary` context, so a step added to `lint` IS required.
    lint_index = ci.index("\n  lint:")
    for ci_step in (
        "Check contract topic category gate (OMN-18013)",
        "Check handler event_type source gate (OMN-18013)",
        "Check golden-chain contract input gate (OMN-18013)",
        "Check no baseline refreeze (OMN-18013)",
    ):
        assert ci.index(ci_step) > lint_index, f"{ci_step} is outside the lint job"


@pytest.mark.unit
def test_the_retired_validator_and_its_module_are_gone() -> None:
    """``mixed_category_routing`` is superseded, not left lying around."""
    assert not (SRC / "validators" / "mixed_category_routing.py").exists()
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            r"(import|from)[^\n]*mixed_category_routing",
            "--",
            "src",
            "tests",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", (
        "live imports of the retired validator remain: " + result.stdout
    )
