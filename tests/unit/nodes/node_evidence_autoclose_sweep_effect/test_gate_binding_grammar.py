# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18414 — the one declared grammar for the ``Gate:`` key.

Before this ticket the admission guard and the evidence closer carried
DISJOINT vocabularies for the same key: the guard mandated four forms, the
closer accepted exactly one form that was not among them. Every ticket the
guard admitted was therefore a typed hold at the closer.

These tests are over the CONTRACT and the closer's resolver. The guard's own
side is pinned in omniclaude's ``tests/hooks/test_ticket_creation_guard.py``,
which asserts byte-equality against this contract.
"""

from __future__ import annotations

import inspect

import pytest

from omnibase_infra.gate_binding import (
    EnumGateBindingProbe,
    load_gate_binding_grammar,
    resolve_gate_binding,
)

pytestmark = pytest.mark.unit


# -- AC1: one declared contract, and every fixture it declares resolves ------


def test_every_accepted_fixture_resolves_to_the_form_it_declares() -> None:
    """A fixture the resolver cannot support turns this red, naming the form."""
    grammar = load_gate_binding_grammar()
    assert grammar.fixtures.accepted, "the contract declares no accepted fixtures"
    for fixture in grammar.fixtures.accepted:
        binding = resolve_gate_binding(f"Gate: {fixture.binding}")
        assert binding is not None, (
            f"the contract declares `{fixture.binding}` accepted as form "
            f"`{fixture.form}`, and the resolver returned nothing"
        )
        assert binding.form == fixture.form, (
            f"`{fixture.binding}` resolved to form `{binding.form}`, "
            f"but the contract declares it `{fixture.form}`"
        )


def test_every_rejected_fixture_is_refused() -> None:
    grammar = load_gate_binding_grammar()
    assert grammar.fixtures.rejected, "the contract declares no rejected fixtures"
    for fixture in grammar.fixtures.rejected:
        binding = resolve_gate_binding(f"Gate: {fixture.binding}")
        assert binding is None, (
            f"the contract declares `{fixture.binding}` rejected, and the "
            f"resolver read it as form `{binding.form if binding else ''}`"
        )


def test_the_resolver_spells_no_form_pattern_of_its_own() -> None:
    """Every form pattern the resolver matches is read from the contract."""
    from omnibase_infra.gate_binding import grammar as module

    source = inspect.getsource(module)
    for spelling in ("INV-", "AC-", "live-gate", "ya?ml", "C\\d"):
        assert spelling not in source, (
            f"`{spelling}` is spelled in the resolver module. The grammar is "
            "declared in the contract; a second spelling here is exactly the "
            "defect OMN-18414 closes."
        )


# -- AC2: all five forms, including the four the guard mandates -------------


@pytest.mark.parametrize(
    ("line", "form"),
    [
        ("Gate: OmniNode-ai/omnibase_infra chain-canary.yml", "workflow_run"),
        ("Gate: C7", "release_criterion"),
        ("Gate: INV-103", "invariant"),
        ("Gate: OMN-16729 AC-5", "parent_criterion"),
        ("Gate: live-gate defect: workspace-reconcile-status", "live_gate_defect"),
    ],
)
def test_the_four_guard_mandated_forms_and_the_workflow_form_all_resolve(
    line: str, form: str
) -> None:
    binding = resolve_gate_binding(line)
    assert binding is not None
    assert binding.form == form


def test_the_workflow_form_captures_its_repository_and_workflow_file() -> None:
    binding = resolve_gate_binding("Gate: OmniNode-ai/omnibase_infra chain-canary.yml")
    assert binding is not None
    assert binding.groups["repo"] == "OmniNode-ai/omnibase_infra"
    assert binding.groups["workflow"] == "chain-canary.yml"


# -- AC3: Linear rewrites the text the author wrote -------------------------


def test_a_linear_issue_mention_still_resolves_to_the_parent_criterion_form() -> None:
    """The live shape of OMN-18368's binding, markup included."""
    live = (
        'Gate: <issue id="59e3391c-7539-4ca5-9324-1c4094f02ff5" '
        'href="https://linear.app/omninode/issue/OMN-16990">OMN-16990</issue> AC-1'
    )
    binding = resolve_gate_binding(live)
    assert binding is not None
    assert binding.form == "parent_criterion"
    assert binding.groups["parent"] == "OMN-16990"
    assert binding.groups["ordinal"] == "1"


def test_a_bare_issue_mention_with_no_ordinal_is_still_refused() -> None:
    """Negative control: the markup rewrite must not manufacture a binding."""
    live = (
        'Gate: <issue id="59e3391c" '
        'href="https://linear.app/omninode/issue/OMN-16990">OMN-16990</issue>'
    )
    assert resolve_gate_binding(live) is None


# -- AC4: exactly one form is a proof pointer -------------------------------


def test_only_the_workflow_form_declares_a_probe() -> None:
    grammar = load_gate_binding_grammar()
    probing = {f.id for f in grammar.forms if f.probe is not EnumGateBindingProbe.NONE}
    assert probing == {"workflow_run"}, (
        "exactly one form may be a proof pointer; the contract declares "
        f"{sorted(probing)}"
    )


@pytest.mark.parametrize(
    "line",
    [
        "Gate: C7",
        "Gate: INV-103",
        "Gate: OMN-16729 AC-5",
        "Gate: live-gate defect: kb-doc-gate",
    ],
)
def test_a_traceability_form_carries_no_probe(line: str) -> None:
    binding = resolve_gate_binding(line)
    assert binding is not None
    assert binding.probe is EnumGateBindingProbe.NONE


# -- AC5: an unreadable binding is a typed hold, never a pass ---------------


def test_an_unparseable_binding_is_distinguishable_from_no_binding_at_all() -> None:
    """A ticket with no ``Gate:`` line declares nothing; a bad one is a defect.

    The two must not collapse into the same value, or the closer cannot tell
    "this ticket names no probe" from "this ticket names one I cannot read".
    """
    from omnibase_infra.gate_binding import gate_binding_line

    assert gate_binding_line("no binding here at all") == ""
    assert gate_binding_line("Gate: chain-canary.yml") == "chain-canary.yml"
    assert resolve_gate_binding("Gate: chain-canary.yml") is None


def test_the_last_binding_line_wins() -> None:
    """A description re-pointed mid-edit is judged on its current line."""
    from omnibase_infra.gate_binding import gate_binding_line

    body = "Gate: C7\n\nsome prose\n\nGate: INV-103\n"
    assert gate_binding_line(body) == "INV-103"


# -- the two line patterns, and the superset relation between them ----------


def test_the_reading_line_pattern_accepts_a_superset_of_the_authoring_one() -> None:
    """The asymmetry is deliberate; the containment is the invariant."""
    import re

    grammar = load_gate_binding_grammar()
    authoring = re.compile(grammar.line_pattern_authoring, re.MULTILINE)
    reading = re.compile(grammar.line_pattern_reading, re.MULTILINE)
    for fixture in grammar.fixtures.line_pattern_superset_fixtures:
        assert bool(authoring.search(fixture.line)) is fixture.authoring, (
            f"authoring pattern disagrees with the contract on {fixture.line!r}"
        )
        assert bool(reading.search(fixture.line)) is fixture.reading, (
            f"reading pattern disagrees with the contract on {fixture.line!r}"
        )
        if fixture.authoring:
            assert fixture.reading, (
                f"{fixture.line!r} is admitted at authoring time and unreadable "
                "at reading time, which is the failure this contract removes"
            )


# -- the closer's own decision, over the same contract -----------------------
#
# `_gate_binding_probe_target` is the pure half of the gate-probe conjunct: it
# decides what (if anything) has to be read live, with no I/O, so every branch
# below is exercised without a network.


def _target(description: str) -> tuple[str, str, str]:
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _gate_binding_probe_target,
    )

    return _gate_binding_probe_target(description)


def test_a_ticket_with_no_binding_line_names_no_probe_and_is_not_held() -> None:
    repo, workflow, hold = _target("a description that declares nothing")
    assert (repo, workflow, hold) == ("", "", "")


def test_the_workflow_form_is_the_thing_that_gets_probed() -> None:
    repo, workflow, hold = _target("Gate: OmniNode-ai/omnibase_infra chain-canary.yml")
    assert repo == "OmniNode-ai/omnibase_infra"
    assert workflow == "chain-canary.yml"
    assert hold == ""


@pytest.mark.parametrize(
    "description",
    [
        "Gate: C7",
        "Gate: INV-103",
        "Gate: OMN-16729 AC-5",
        "Gate: live-gate defect: workspace-reconcile-status",
        'Gate: <issue id="x" href="https://linear.app/omninode/issue/OMN-16990">'
        "OMN-16990</issue> AC-1",
    ],
)
def test_a_traceability_binding_is_neither_probed_nor_held(description: str) -> None:
    """The four forms the admission guard mandates all reach the next stage.

    Every one of these was `skipped_gate_probe_red` before OMN-18414, which is
    what held OMN-18403, OMN-18387, OMN-18368 and OMN-18365 on the
    2026-09-15T21:34Z tick.
    """
    assert _target(description) == ("", "", "")


def test_an_unreadable_binding_is_held_and_the_hold_names_the_accepted_forms() -> None:
    repo, workflow, hold = _target("Gate: chain-canary.yml")
    assert (repo, workflow) == ("", "")
    assert hold, "an unreadable binding must be held, never passed"
    assert "chain-canary.yml" in hold
    for example in ("C7", "INV-103", "AC-5", "live-gate defect"):
        assert example in hold, (
            f"the hold does not show `{example}` as an accepted form, so "
            "whoever reads it has to guess what would have been readable"
        )


def test_the_unreadable_hold_is_not_reached_by_a_readable_binding() -> None:
    """Positive control for the test above: the same call, a readable form."""
    assert _target("Gate: C7")[2] == ""
