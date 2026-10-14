# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The event-type validator must not repeatedly walk unrelated test scopes."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_infra.validators import no_literal_event_type_in_tests as validator


@pytest.mark.unit
@pytest.mark.parametrize("literal_site", [False, True])
def test_scan_cost_for_scopes_without_indirect_event_types(
    monkeypatch: pytest.MonkeyPatch, literal_site: bool
) -> None:
    # Model a large test class: its unrelated methods previously had their
    # bodies walked repeatedly while building module and function bindings.
    text = "class TestManyCases:\n" + "".join(
        f"    def test_case_{i}(self):\n"
        + "".join(f"        value_{j} = {j}\n" for j in range(30))
        for i in range(100)
    )
    if literal_site:
        text += (
            "    def test_envelope(self):\n"
            '        ModelEventEnvelope(event_type="onex.evt.producer.created.v1")\n'
        )
    walk = ast.walk
    node_count = sum(1 for _ in walk(ast.parse(text)))
    visits = 0

    def counted_walk(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal visits
        for child in walk(node):
            visits += 1
            yield child

    monkeypatch.setattr(ast, "walk", counted_walk)
    findings = validator.findings_for_module(Path("test_many_cases.py"), text)
    assert len(findings) == int(literal_site)
    # A deterministic work budget avoids a wall-clock threshold that depends
    # on hosted-runner load. Two complete traversals are ample for this case.
    assert visits <= 2 * node_count, (visits, node_count)


@pytest.mark.unit
def test_literal_sites_still_deduplicate_by_location() -> None:
    text = (
        'ModelEventEnvelope(event_type="onex.evt.producer.created.v1"); '
        'ModelEventEnvelope(event_type="onex.evt.producer.created.v1")\n'
    )
    findings = validator.findings_for_module(Path("test_envelope.py"), text)
    assert len(findings) == 1
    assert findings[0].location == "test_envelope.py:1"


@pytest.mark.unit
def test_golden_chain_escaped_dictionary_key_is_still_checked() -> None:
    text = 'data = {"event_\\x74ype": "onex.evt.producer.created.v1"}\n'
    findings = validator.findings_for_module(Path("test_golden_chain_probe.py"), text)
    assert len(findings) == 1


@pytest.mark.unit
def test_malformed_module_is_still_refused() -> None:
    findings = validator.findings_for_module(Path("test_invalid.py"), "def broken(:\n")
    assert len(findings) == 1
    assert "module does not parse" in findings[0].detail
