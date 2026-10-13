# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Conditional checks: a profile demands a check only of a PR that touches its surface.

Before this, the omnibase_infra runtime profile named ``changed_path_live`` and
``forwarder_refused_topic`` as mandatory for every PR, the lab pool produced
neither for a pin or lock bump, and every such receipt was
MISSING_MANDATORY_CHECK (omnibase_infra#4783, #4799, #4809, #4816 held on it).
These tests pin both directions against the committed registry: a diff that
touches a check's surface keeps the check mandatory, one that does not leaves it
out, and an unread or empty diff waives nothing.

Ticket: OMN-19566
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_kind import EnumLabProofKind
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
    profile_pin,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
SCRIPT = REPO_ROOT / "scripts" / "ci" / "validate_lab_proof_profiles.py"
INFRA = "OmniNode-ai/omnibase_infra"
MARKET = "OmniNode-ai/omnimarket"
LIVE = EnumLabProofCheck.CHANGED_PATH_LIVE
FORWARDER = EnumLabProofCheck.FORWARDER_REFUSED_TOPIC

# The shapes of the four held PRs, from their bodies: an omnimarket pin advance
# (#4809, #4816), a core lock bump (#4783), a Dockerfile pin (#4799).
OMNIMARKET_PIN_ADVANCE = [
    ".github/omnimarket-contract-pin.yaml",
    ".github/sibling-pins.yaml",
    ".github/workflows/ci.yml",
    "docker/catalog/database-topology/omninode_internal.yaml",
    "tests/ci/test_supplemental_declaration_expiry_omn18863.py",
]
CORE_LOCK_BUMP = ["pyproject.toml", "uv.lock"]
DOCKERFILE_PIN = ["docker/Dockerfile.runtime"]
NODE_CHANGE = [
    "src/omnibase_infra/nodes/node_runtime_error_triage_effect/handlers/handler_x.py",
    "tests/unit/nodes/test_x.py",
]
FORWARDER_TOPIC_CHANGE = [
    "src/omnibase_infra/nodes/node_bus_forwarder_effect/contract.yaml",
]
FORWARDER_ENTRYPOINT_CHANGE = ["src/omnibase_infra/runtime/gateway_forwarder.py"]
NON_NODE_SOURCE_CHANGE = ["src/omnibase_infra/runtime/service_kernel.py"]


@pytest.fixture(scope="module")
def registry() -> ModelLabProofProfileRegistry:
    return load_lab_proof_profile_registry(REGISTRY)


def _pin(
    registry: ModelLabProofProfileRegistry, repo: str, changed: list[str] | None
) -> tuple[set[EnumLabProofCheck], set[EnumLabProofCheck]]:
    pin = profile_pin(registry, repo, EnumLabProofKind.RUNTIME_IMAGE, changed)
    return set(pin.mandatory_checks), set(pin.inapplicable_checks)


# --- the direction that waives: a diff that does not touch the surface ------


@pytest.mark.parametrize(
    "changed", [OMNIMARKET_PIN_ADVANCE, CORE_LOCK_BUMP, DOCKERFILE_PIN]
)
def test_a_pin_or_lock_bump_is_not_held_to_either_surface_check(
    registry: ModelLabProofProfileRegistry, changed: list[str]
) -> None:
    mandatory, waived = _pin(registry, INFRA, changed)
    assert waived == {LIVE, FORWARDER}
    assert not mandatory & waived
    # every unconditional check still applies
    assert {
        EnumLabProofCheck.RUNTIME_IMAGE_IDENTITY,
        EnumLabProofCheck.MIGRATION_GATE_HEALTHY,
        EnumLabProofCheck.RUNTIME_MAIN_HEALTHY,
        EnumLabProofCheck.RUNTIME_EFFECTS_HEALTHY,
        EnumLabProofCheck.NO_WIRING_FAILURES,
        EnumLabProofCheck.FOCUSED_TESTS,
    } <= mandatory


def test_a_change_outside_any_node_does_not_demand_changed_path_live(
    registry: ModelLabProofProfileRegistry,
) -> None:
    _, waived = _pin(registry, INFRA, NON_NODE_SOURCE_CHANGE)
    assert LIVE in waived and FORWARDER in waived
    # nodes/__init__.py is not inside a node directory
    _, waived = _pin(registry, INFRA, ["src/omnibase_infra/nodes/__init__.py"])
    assert LIVE in waived


# --- the direction that keeps: a diff that touches the surface --------------


def test_a_node_change_is_held_to_changed_path_live_only(
    registry: ModelLabProofProfileRegistry,
) -> None:
    mandatory, waived = _pin(registry, INFRA, NODE_CHANGE)
    assert LIVE in mandatory
    assert waived == {FORWARDER}


def test_a_forwarder_topic_change_is_held_to_both(
    registry: ModelLabProofProfileRegistry,
) -> None:
    mandatory, waived = _pin(registry, INFRA, FORWARDER_TOPIC_CHANGE)
    assert {LIVE, FORWARDER} <= mandatory
    assert waived == set()


@pytest.mark.parametrize(
    "path",
    [
        "src/omnibase_infra/runtime/gateway_forwarder.py",
        "src/omnibase_infra/event_bus/kafka_transport.py",
        "docker/docker-compose.gateway.yml",
        "docker/gateway/onex-gateway-forwarder.service",
        "src/omnibase_infra/nodes/node_board_probe_effect/handlers/handler_forwarder_refused_topic.py",
    ],
)
def test_every_forwarder_surface_demands_forwarder_refused_topic(
    registry: ModelLabProofProfileRegistry, path: str
) -> None:
    mandatory, _ = _pin(registry, INFRA, [*CORE_LOCK_BUMP, path])
    assert FORWARDER in mandatory


def test_an_omnimarket_node_change_is_held_to_changed_path_live(
    registry: ModelLabProofProfileRegistry,
) -> None:
    mandatory, waived = _pin(
        registry, MARKET, ["src/omnimarket/nodes/node_x/handlers/handler_x.py"]
    )
    assert LIVE in mandatory and waived == set()
    mandatory, waived = _pin(registry, MARKET, ["uv.lock"])
    assert waived == {LIVE}


# --- no evidence about the diff waives nothing --------------------------------


@pytest.mark.parametrize("changed", [None, []])
def test_an_unread_or_empty_diff_waives_nothing(
    registry: ModelLabProofProfileRegistry, changed: list[str] | None
) -> None:
    mandatory, waived = _pin(registry, INFRA, changed)
    assert {LIVE, FORWARDER} <= mandatory
    assert waived == set()


# --- the validator CLI the workflow calls ------------------------------------


def _cli_pin(tmp_path: Path, changed: list[str] | None) -> dict[str, Any]:
    argv = [sys.executable, str(SCRIPT), "--pin", INFRA, "runtime_image"]
    if changed is not None:
        listing = tmp_path / "changed.txt"
        listing.write_text("".join(f"{p}\n" for p in changed), encoding="utf-8")
        argv += ["--changed-files", str(listing)]
    done = subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    doc = json.loads(done.stdout)
    assert isinstance(doc, dict)
    return doc


def test_the_cli_pin_reads_the_changed_files(tmp_path: Path) -> None:
    bump = _cli_pin(tmp_path, OMNIMARKET_PIN_ADVANCE)
    assert "changed_path_live" not in bump["mandatory_checks"]
    assert "forwarder_refused_topic" not in bump["mandatory_checks"]
    assert sorted(bump["inapplicable_checks"]) == [
        "changed_path_live",
        "forwarder_refused_topic",
    ]
    node = _cli_pin(tmp_path, NODE_CHANGE)
    assert "changed_path_live" in node["mandatory_checks"]
    unread = _cli_pin(tmp_path, None)
    assert {"changed_path_live", "forwarder_refused_topic"} <= set(
        unread["mandatory_checks"]
    )


def test_the_cli_refuses_a_changed_files_path_it_cannot_read(tmp_path: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pin",
            INFRA,
            "runtime_image",
            "--changed-files",
            str(tmp_path / "absent.txt"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1
    assert "no profile pin" in done.stderr


# --- row-shape rules -----------------------------------------------------------


def _raw() -> dict[str, Any]:
    loaded = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _infra_runtime(data: dict[str, Any]) -> dict[str, Any]:
    for profile in data["profiles"]:
        if profile["repo"] == INFRA:
            for variant in profile["variants"]:
                if variant["proof_kind"] == "runtime_image":
                    assert isinstance(variant, dict)
                    return variant
    raise AssertionError("no omnibase_infra runtime_image variant")


def _core_foundation(data: dict[str, Any]) -> dict[str, Any]:
    for profile in data["profiles"]:
        if profile["repo"] == "OmniNode-ai/omnibase_core":
            variant = profile["variants"][0]
            assert isinstance(variant, dict)
            return variant
    raise AssertionError("no omnibase_core row")


def test_the_committed_checks_are_conditional_and_never_deleted() -> None:
    variant = _infra_runtime(_raw())
    conditional = {entry["check"] for entry in variant["conditional_checks"]}
    assert conditional == {"changed_path_live", "forwarder_refused_topic"}
    assert all(entry["reason"].strip() for entry in variant["conditional_checks"])


def test_a_check_both_mandatory_and_conditional_is_refused() -> None:
    data = copy.deepcopy(_raw())
    variant = _infra_runtime(data)
    variant["mandatory_checks"].append("changed_path_live")
    with pytest.raises(ValidationError, match="both mandatory and conditional"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_a_conditional_check_listed_twice_is_refused() -> None:
    data = copy.deepcopy(_raw())
    variant = _infra_runtime(data)
    variant["conditional_checks"].append(
        copy.deepcopy(variant["conditional_checks"][0])
    )
    with pytest.raises(ValidationError, match="listed twice"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_a_conditional_check_on_a_node_row_is_refused() -> None:
    data = copy.deepcopy(_raw())
    _core_foundation(data)["conditional_checks"] = [
        {
            "check": "golden_chain_delegation",
            "when_changed": ["src/**"],
            "reason": "the node path would drop it",
        }
    ]
    with pytest.raises(ValidationError, match="only resolved for execution"):
        ModelLabProofProfileRegistry.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("when_changed", []), ("reason", ""), ("check", "not_a_check")],
)
def test_a_malformed_conditional_check_is_refused(field: str, value: object) -> None:
    data = copy.deepcopy(_raw())
    _infra_runtime(data)["conditional_checks"][0][field] = value
    with pytest.raises(ValidationError):
        ModelLabProofProfileRegistry.model_validate(data)
