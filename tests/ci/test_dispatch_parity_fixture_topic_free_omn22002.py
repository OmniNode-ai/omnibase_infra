# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed dispatch-parity oracle carries no topic literal (OMN-22002).

Topics are declared in contract YAML. The oracle stores each as a ``${topic:<ref>}``
contract reference that the harness resolves at load time, so the hardcoded-topic hook
has nothing to refuse and a topic that leaves its contract fails the load.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ORACLE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dispatch_parity"
    / "selection-oracle-v2.json"
)
# Any topic-shaped string, quoted or embedded in a longer string or key.
_TOPIC_SHAPED = re.compile(r"onex\.[A-Za-z0-9\-]+(?:\.[A-Za-z0-9_\-]+)+")


def test_committed_oracle_text_has_no_topic_literal() -> None:
    literals = sorted(set(_TOPIC_SHAPED.findall(ORACLE.read_text(encoding="utf-8"))))
    assert literals == []


def test_committed_oracle_references_resolve_and_reencode_identically() -> None:
    from tests.fixtures.dispatch_parity import harness
    from tests.fixtures.dispatch_parity.topic_refs import decode_topics, encode_topics

    contracts = harness._corpus_from_discovery()[0]
    text = ORACLE.read_text(encoding="utf-8")
    assert "${topic:" in text
    resolved = decode_topics(text, contracts)
    assert "${topic:" not in resolved
    assert _TOPIC_SHAPED.search(resolved) is not None
    assert encode_topics(resolved, contracts) == text


def test_reference_to_an_undeclared_topic_fails_the_load() -> None:
    from tests.fixtures.dispatch_parity import harness
    from tests.fixtures.dispatch_parity.topic_refs import decode_topics

    contracts = harness._corpus_from_discovery()[0]
    with pytest.raises(ValueError, match="declared by no contract"):
        decode_topics("${topic:contract:rsd.CMD_NOT_DECLARED_V1}", contracts)


def test_a_topic_with_no_reference_is_refused_on_encode() -> None:
    from tests.fixtures.dispatch_parity import harness
    from tests.fixtures.dispatch_parity.topic_refs import encode_topics

    contracts = harness._corpus_from_discovery()[0]
    undeclared = ".".join(("onex", "cmd", "rsd", "undeclared", "v1"))
    with pytest.raises(ValueError, match="no contract, dlq or synthetic reference"):
        encode_topics(f'"{undeclared}"', contracts)
