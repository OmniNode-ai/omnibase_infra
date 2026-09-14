# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provisioning coverage for the three delegation v2 terminal topics (OMN-15622).

Scoped slice of OMN-15622, plan Task 1.2. The canonical registry rows and the
``TopicBase`` constants land in omnibase_core; this repo owns the broker-side
half — a ``ModelTopicSpec`` per topic so ``TopicProvisioner`` creates them, and a
``_LEGACY_ALLOWLIST`` entry per topic so ``check_contract_topic_parity.py`` does
not report them as python-only drift.

The allowlist entry is required because the parity scanner reads only this
repo's own ``nodes/``. The v2 terminal publisher contract is omnimarket-side
work that has not landed yet, so these three suffixes are declared here ahead of
their producer, the same disposition the thirteen v1 delegation entries already
carry.

The v2 family is three topics, not two: two failure classes sharing one
``delegation-failed.v2`` topic is a non-injective class -> topic map and is
refused in omnibase_core by ``assert_published_events_injective``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.topics import (
    ALL_PROVISIONED_SUFFIXES,
    SUFFIX_DELEGATION_COMPLETED,
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)
from omnibase_infra.topics.platform_topic_suffixes import (
    ALL_OMNIBASE_INFRA_TOPIC_SPECS,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCRIPT = REPO_ROOT / "scripts" / "check_contract_topic_parity.py"

V1_TERMINAL_SUFFIXES: tuple[str, ...] = (
    SUFFIX_DELEGATION_COMPLETED,
    SUFFIX_DELEGATION_FAILED,
)

V2_TERMINAL_SUFFIXES: tuple[str, ...] = (
    SUFFIX_DELEGATION_COMPLETED_V2,
    SUFFIX_DELEGATION_FAILED_ROUTED_V2,
    SUFFIX_DELEGATION_FAILED_UNROUTED_V2,
)

EXPECTED_V2_WIRE_NAMES: tuple[str, ...] = (
    "onex.evt.omnibase-infra.delegation-completed.v2",
    "onex.evt.omnibase-infra.delegation-failed-routed.v2",
    "onex.evt.omnibase-infra.delegation-failed-unrouted.v2",
)


def _spec_for(suffix: str) -> Any:
    for spec in ALL_OMNIBASE_INFRA_TOPIC_SPECS:
        if spec.suffix == suffix:
            return spec
    raise AssertionError(
        f"no ModelTopicSpec for {suffix!r} in ALL_OMNIBASE_INFRA_TOPIC_SPECS"
    )


def _legacy_allowlist() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "_omn15622_parity_probe", PARITY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allowlist: dict[str, str] = module._LEGACY_ALLOWLIST
    return allowlist


# ---------------------------------------------------------------------------
# Suffix constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected"),
    list(zip(V2_TERMINAL_SUFFIXES, EXPECTED_V2_WIRE_NAMES, strict=True)),
)
def test_v2_suffix_wire_name(suffix: str, expected: str) -> None:
    """Each v2 suffix is exactly the canonical wire name from the registry."""
    assert suffix == expected


def test_v2_suffixes_are_distinct() -> None:
    assert len(set(V2_TERMINAL_SUFFIXES)) == 3


# ---------------------------------------------------------------------------
# ModelTopicSpec provisioning entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_topic_is_provisioned(suffix: str) -> None:
    """TopicProvisioner creates the topic only if it is in the registry."""
    assert suffix in ALL_PROVISIONED_SUFFIXES


@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_spec_mirrors_the_v1_terminal_settings(suffix: str) -> None:
    """Partition count and retention mirror the v1 terminal topics exactly."""
    v1_spec = _spec_for(SUFFIX_DELEGATION_COMPLETED)
    v2_spec = _spec_for(suffix)
    assert v2_spec.partitions == v1_spec.partitions
    assert v2_spec.kafka_config == v1_spec.kafka_config


def test_v1_terminal_specs_are_the_reference_shape() -> None:
    """Positive control: the v1 pair this task mirrors really carries these settings."""
    for suffix in V1_TERMINAL_SUFFIXES:
        spec = _spec_for(suffix)
        assert spec.partitions == 3
        assert spec.kafka_config == {
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        }


# ---------------------------------------------------------------------------
# Parity-gate allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_suffix_is_allowlisted_in_the_parity_gate(suffix: str) -> None:
    """Without the entry the parity gate reports the suffix as python-only drift."""
    assert suffix in _legacy_allowlist()


@pytest.mark.parametrize("suffix", V2_TERMINAL_SUFFIXES)
def test_v2_allowlist_entry_carries_reason_owner_expiry(suffix: str) -> None:
    """The script's own header mandates the `reason | owner | expiry` shape."""
    note = _legacy_allowlist()[suffix]
    assert "| owner:" in note, note
    assert "| expiry:" in note, note
    reason = note.split("| owner:")[0].strip()
    assert reason, f"allowlist entry for {suffix} has an empty reason: {note!r}"


def test_v1_terminal_suffixes_are_allowlisted() -> None:
    """Positive control: the probe really reads the live allowlist."""
    allowlist = _legacy_allowlist()
    for suffix in V1_TERMINAL_SUFFIXES:
        assert suffix in allowlist


def test_a_topic_that_was_never_allowlisted_is_absent() -> None:
    """Negative control: the probe is not returning a dict that contains everything."""
    assert "onex.evt.omnibase-infra.omn15622-not-a-real-topic.v1" not in (
        _legacy_allowlist()
    )
