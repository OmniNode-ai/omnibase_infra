# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two lab proof node contracts bind to the handlers, models and bundle they name.

Ticket: OMN-19572
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
NODES = REPO_ROOT / "src/omnibase_infra/nodes"
EXPECTED = {
    "node_lab_proof_plan_compute": (
        "COMPUTE_GENERIC",
        {
            "lab_proof.plan": "HandlerLabProofPlan",
            "lab_proof.verdict": "HandlerLabProofVerdict",
        },
    ),
    "node_lab_proof_run_effect": (
        "EFFECT_GENERIC",
        {"lab_proof.run": "HandlerLabProofRun"},
    ),
}


def _contract(node: str) -> dict[str, Any]:
    loaded = yaml.safe_load(
        (NODES / node / "contract.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("node", sorted(EXPECTED))
def test_contract_routes_each_operation_to_an_importable_handler(node: str) -> None:
    node_type, routes = EXPECTED[node]
    contract = _contract(node)
    assert contract["node_type"] == node_type
    assert contract["name"] == node
    seen: dict[str, str] = {}
    for entry in contract["handler_routing"]["handlers"]:
        module = importlib.import_module(entry["handler"]["module"])
        handler_cls = getattr(module, entry["handler"]["name"])
        assert callable(handler_cls().handle)
        seen[entry["operation"]] = entry["handler"]["name"]
        model_module = importlib.import_module(entry["input_model"]["module"])
        assert hasattr(model_module, entry["input_model"]["name"])
    assert seen == routes
    for key in ("input_model", "output_model"):
        module = importlib.import_module(contract[key]["module"])
        assert hasattr(module, contract[key]["name"])


def test_plan_contract_bundle_block_is_a_complete_policy() -> None:
    config = _contract("node_lab_proof_plan_compute")["config"]["local_bundle"]
    policy = ModelLabProofBundlePolicy.model_validate(config)
    assert policy.compose_project == "omnibase-infra-local"
    for key in list(config):
        broken = {k: v for k, v in config.items() if k != key}
        with pytest.raises(ValueError, match=key):
            ModelLabProofBundlePolicy.model_validate(broken)


def test_the_override_dockerfile_the_contract_names_exists() -> None:
    config = _contract("node_lab_proof_plan_compute")["config"]["local_bundle"]
    dockerfile = REPO_ROOT / config["override_dockerfile"]
    text = dockerfile.read_text(encoding="utf-8")
    assert "--no-deps" in text and "--reinstall-package" in text
    assert 'file:///lab-proof/subject"' in text


def test_both_nodes_are_registered_entry_points() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for node in EXPECTED:
        assert f'{node} = "omnibase_infra.nodes.{node}"' in pyproject


def test_the_boundary_refuses_a_repository_with_no_node_executed_variant(
    tmp_path: Path,
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("src/x.py\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/lab_proof/run_lab_proof.py"),
            "--repo",
            "OmniNode-ai/omnimarket",
            "--pr",
            "1",
            "--head-sha",
            "a" * 40,
            "--base-sha",
            "b" * 40,
            "--changed-files",
            str(changed),
            "--host",
            "localhost",
            "--lane-root",
            str(tmp_path / "prove-x"),
            "--run-key",
            "run-refused",
            "--infra-sha",
            "c" * 40,
            "--model-endpoint",
            "http://127.0.0.1:8000/v1/chat/completions",
            "--positive-control-project",
            "none-such",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 5, result.stdout + result.stderr
    assert "0 node-executed variants" in result.stdout
    assert not (tmp_path / "prove-x").exists()
