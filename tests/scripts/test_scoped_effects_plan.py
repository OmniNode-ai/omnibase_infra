# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed rollout inputs and actual installed-content admission [OMN-17991]."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.runtime_build import scoped_image_probe as probe
from scripts.runtime_build.scoped_effects_plan import ModelEffectsDeployPlan

pytestmark = pytest.mark.unit


def plan_data(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "ticket_id": "OMN-17991",
        "reason": "Approved effects-only rollout after compatibility checks",
        "compose_project": "omnibase-infra",
        "service": "runtime-effects",
        "expected_container_id": "a" * 64,
        "expected_image_id": "sha256:" + "b" * 64,
        "candidate_image_id": "sha256:" + "c" * 64,
        "compose_files": [str(tmp_path / "compose.yaml")],
        "compose_working_dir": str(tmp_path),
        "expected_compose_sha256": "d" * 64,
        "source_pins": dict.fromkeys(probe.SIBLINGS, "e" * 40),
        "infra_source_sha": "f" * 40,
        "source_clones_root": str(tmp_path / "sources"),
        "hotpatch_ledger": str(tmp_path / "hotpatch.yaml"),
    }


def test_plan_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_data(tmp_path)))
    plan = ModelEffectsDeployPlan.load(path)
    assert plan.service == "runtime-effects"
    assert plan.compose_files == (tmp_path / "compose.yaml",)
    assert plan.health_timeout_seconds == 1800
    with pytest.raises(ValidationError):
        plan.reason = "changed after validation"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("service", "omninode-runtime"),
        ("compose_project", "omnibase-infra-prod"),
        ("schema_version", "2"),
        ("candidate_image_id", "runtime-effects:latest"),
        ("candidate_image_id", "sha256:" + "b" * 64),
        ("expected_container_id", "abc123"),
        ("reason", " " * 12),
        ("ticket_id", "OMN-0"),
        ("health_timeout_seconds", True),
        ("health_timeout_seconds", 1801),
        ("expected_compose_sha256", "unknown"),
        ("compose_working_dir", "relative"),
        ("compose_files", []),
        ("compose_files", ["/deployment/../compose.yaml"]),
        ("force", True),
    ],
)
def test_invalid_plan_refuses(tmp_path: Path, field: str, value: object) -> None:
    data = plan_data(tmp_path)
    data[field] = value
    with pytest.raises(ValidationError):
        ModelEffectsDeployPlan.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "pins",
    [
        {},
        {"omnimarket": "a" * 40},
        {**dict.fromkeys(probe.SIBLINGS, "a" * 40), "other": "b" * 40},
        {**dict.fromkeys(probe.SIBLINGS, "a" * 40), "omnimarket": "dev"},
    ],
)
def test_incomplete_or_mutable_source_pins_refuse(tmp_path: Path, pins: object) -> None:
    data = plan_data(tmp_path)
    data["source_pins"] = pins
    with pytest.raises(ValidationError):
        ModelEffectsDeployPlan.model_validate_json(json.dumps(data))


def distribution(root: Path, name: str = "demo") -> importlib.metadata.PathDistribution:
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    metadata = root / f"{name}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(
        "[onex.domain_plugins]\ndemo = demo:Plugin\n"
    )
    (metadata / "direct_url.json").write_text('{"url":"file:///build/source"}')
    files = [package / "__init__.py", *metadata.iterdir()]
    (metadata / "RECORD").write_text(
        "".join(f"{path.relative_to(root)},,\n" for path in files)
    )
    return importlib.metadata.PathDistribution(metadata)


def test_dependency_fingerprint_covers_payload_and_activation_metadata(
    tmp_path: Path,
) -> None:
    dist = distribution(tmp_path)
    baseline = probe.dependency_digest([dist])
    metadata = tmp_path / "demo-1.0.dist-info"
    (metadata / "direct_url.json").write_text('{"url":"file:///other/build"}')
    assert probe.dependency_digest([dist]) == baseline
    (metadata / "entry_points.txt").write_text(
        "[onex.domain_plugins]\nextra = demo:Other\n"
    )
    assert probe.dependency_digest([dist]) != baseline
    (metadata / "entry_points.txt").write_text(
        "[onex.domain_plugins]\ndemo = demo:Plugin\n"
    )
    (tmp_path / "demo" / "__init__.py").write_text("VALUE = 2\n")
    assert probe.dependency_digest([dist]) != baseline


def test_missing_or_editable_dependency_refuses(tmp_path: Path) -> None:
    dist = distribution(tmp_path)
    metadata = tmp_path / "demo-1.0.dist-info"
    (metadata / "direct_url.json").write_text('{"dir_info":{"editable":true}}')
    with pytest.raises(ValueError, match="editable"):
        probe.dependency_digest([dist])
    (metadata / "direct_url.json").write_text("{}")
    (tmp_path / "demo" / "__init__.py").unlink()
    with pytest.raises(ValueError, match="payload absent"):
        probe.dependency_digest([dist])


def test_market_payload_is_the_allowed_dependency_delta(tmp_path: Path) -> None:
    dist = distribution(tmp_path, "omnimarket")
    before = probe.dependency_digest([dist])
    (tmp_path / "omnimarket" / "__init__.py").write_text("VALUE = 2\n")
    assert probe.dependency_digest([dist]) == before


def test_workspace_proof_rechecks_actual_installed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dists = {name: distribution(tmp_path, name) for name in probe.SIBLINGS}
    monkeypatch.setattr(probe.importlib.metadata, "distribution", dists.__getitem__)
    pins = dict.fromkeys(probe.SIBLINGS, "a" * 40)
    manifest = {
        "build_source": "workspace",
        "per_repo_vcs_provenance": {
            "siblings": {
                name: {"vcs_ref": sha, "vcs_dirty": False} for name, sha in pins.items()
            }
        },
        "proofs": [
            {
                "repo": name,
                "status": "verified",
                "installed_package_digest": probe.tree_digest(tmp_path / name),
                "staged_package_digest": probe.tree_digest(tmp_path / name),
            }
            for name in probe.SIBLINGS
        ],
    }
    (tmp_path / "build-provenance.json").write_text(json.dumps(manifest))
    assert probe.source_pins(tmp_path) == pins
    (tmp_path / "omnimarket" / "__init__.py").write_text("STALE = True\n")
    with pytest.raises(ValueError, match="differs from source proof"):
        probe.source_pins(tmp_path)


def test_shared_contract_activation_change_refuses_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "entrypoint-runtime.sh").write_text("#!/bin/sh\nexec runtime\n")
    dists = {
        name: distribution(tmp_path, name)
        for name in (*probe.SIBLINGS, "omnibase_infra")
    }
    monkeypatch.setattr(probe.importlib.metadata, "distribution", dists.__getitem__)
    contract = tmp_path / "omnimarket" / "contract.yaml"
    content = "name: demo\ndescription: first\ndescriptor:\n  runtime_profiles: [effects]\nevent_bus:\n  subscribe_topics: [onex.evt.omnimarket.sample.v1]\n"
    contract.write_text(content)
    baseline, topics = probe.shared_evidence(tmp_path)
    assert topics == ["onex.evt.omnimarket.sample.v1"]
    contract.write_text(content.replace("description: first", "description: second"))
    assert probe.shared_evidence(tmp_path)[0] == baseline
    contract.write_text(content + "metadata:\n  related_tickets: [OMN-17991]\n")
    with_ticket = probe.shared_evidence(tmp_path)[0]
    contract.write_text(content + "metadata:\n  related_tickets: [OMN-17974]\n")
    assert probe.shared_evidence(tmp_path)[0] == with_ticket
    contract.write_text(content.replace("[effects]", "[default]"))
    after, same_topics = probe.shared_evidence(tmp_path)
    assert same_topics == topics
    assert after != baseline
    for requirement in (
        "db_io:\n  write_tables: [new_table]\n",
        "state_machine:\n  initial_state: ready\n",
        "configuration:\n  auto_create_schema: true\n",
    ):
        contract.write_text(content + requirement)
        assert probe.shared_evidence(tmp_path)[0] != baseline
    contract.write_text(content + "effective_on: 2026-09-06\n")
    dated = probe.shared_evidence(tmp_path)[0]
    contract.write_text(content + "effective_on: 2026-09-07\n")
    assert probe.shared_evidence(tmp_path)[0] != dated
