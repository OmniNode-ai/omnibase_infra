# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Tests for the contract-sourced delegate-skill enum generation (OMN-13202).
#
# OMN-13202 migrates ``scripts/generate_topic_enums.py`` so the
# ``EnumOmnimarketTopic.EVT_DELEGATE_SKILL_*_V1`` members are derived from a
# contract-declarative topic manifest (the runtime ``topics.yaml`` that mirrors
# the omnimarket ``node_delegate_skill_orchestrator`` published events) instead
# of an AST scan of ``event_bus/topic_constants.py``. These tests pin that the
# migrated source produces a byte-identical ``EnumOmnimarketTopic`` WITHOUT the
# supplementary Python AST source, so the kept ``TOPIC_DELEGATE_SKILL_*``
# literals in ``topic_constants.py`` can be deleted.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
from omnibase_infra.tools.topic_enum_generator import TopicEnumGenerator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_topic_enums.py"
_CONTRACTS_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"
_RUNTIME_MANIFEST = _REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "topics.yaml"
_GENERATED_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "enums" / "generated"
_TOPIC_CONSTANTS = (
    _REPO_ROOT / "src" / "omnibase_infra" / "event_bus" / "topic_constants.py"
)

_DELEGATE_SKILL_TOPICS = {
    "onex.evt.omnimarket.delegate-skill-completed.v1",
    "onex.evt.omnimarket.delegate-skill-failed.v1",
}


def _load_script_module() -> object:
    """Import scripts/generate_topic_enums.py as a module for white-box checks."""
    spec = importlib.util.spec_from_file_location(
        "generate_topic_enums_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_runtime_manifest_declares_delegate_skill_topics() -> None:
    """The runtime topics.yaml is the contract-declarative source of the two
    delegate-skill terminal topics emitted by the runtime DispatchResultApplier."""
    assert _RUNTIME_MANIFEST.exists(), (
        f"runtime topic manifest missing: {_RUNTIME_MANIFEST}"
    )
    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(_RUNTIME_MANIFEST.parent)
    topics = {e.topic for e in entries}
    assert topics >= _DELEGATE_SKILL_TOPICS, (
        f"runtime manifest must declare delegate-skill topics; got {topics}"
    )


@pytest.mark.unit
def test_generator_no_longer_uses_topic_constants_ast() -> None:
    """The generator must not scan topic_constants.py as a supplementary AST
    source — the literals are deleted from that module by OMN-13202."""
    module = _load_script_module()
    default_sources = getattr(module, "_DEFAULT_SUPPLEMENTARY_SOURCES", ())
    joined = " ".join(str(p) for p in default_sources)
    assert "topic_constants.py" not in joined, (
        "generator still references topic_constants.py as a supplementary AST "
        f"source: {default_sources}"
    )


@pytest.mark.unit
def test_omnimarket_enum_byte_identical_without_topic_constants_ast() -> None:
    """Regenerating from contracts + runtime manifest (NO topic_constants.py AST)
    reproduces the committed EnumOmnimarketTopic byte-for-byte, including the two
    EVT_DELEGATE_SKILL_*_V1 members consumed by service_kernel bootstrap."""
    if not _CONTRACTS_ROOT.exists():
        pytest.skip("contracts root not found")

    extractor = ContractTopicExtractor()
    entries = extractor.extract_all(
        _CONTRACTS_ROOT,
        supplementary_sources=None,
        skill_manifests_roots=[_RUNTIME_MANIFEST.parent],
    )
    rendered = TopicEnumGenerator().render(entries, output_dir=_GENERATED_DIR)

    omnimarket_path = _GENERATED_DIR / "enum_omnimarket_topic.py"
    generated = rendered.get(omnimarket_path)
    assert generated is not None, "EnumOmnimarketTopic was not generated"

    committed = omnimarket_path.read_text(encoding="utf-8")
    assert generated == committed, (
        "EnumOmnimarketTopic drifted from the committed file when generated from "
        "contracts + runtime manifest. Run: "
        "uv run python scripts/generate_topic_enums.py --generate"
    )

    # The two delegate-skill members must be present (regression guard for the
    # service_kernel bootstrap dependency, OMN-11996).
    for topic in _DELEGATE_SKILL_TOPICS:
        assert topic in generated, f"missing delegate-skill member for {topic}"


@pytest.mark.unit
def test_topic_constants_has_no_topic_literals() -> None:
    """After OMN-13202, topic_constants.py contains zero TOPIC_* constants; only
    DLQ builders/format helpers remain."""
    text = _TOPIC_CONSTANTS.read_text(encoding="utf-8")
    for name in (
        "TOPIC_SESSION_COORDINATION_SIGNAL",
        "TOPIC_DELEGATE_SKILL_COMPLETED",
        "TOPIC_DELEGATE_SKILL_FAILED",
    ):
        assert f"{name}:" not in text and f'"{name}"' not in text, (
            f"{name} should have been removed from topic_constants.py"
        )
