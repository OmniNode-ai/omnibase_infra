# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the in-progress probe hygiene sweep reads no change-control clone.

The scheduled sweep checked out ``onex_change_control`` and handed its path to
the node as ``--occ-repo-dir``. A ticket's contract lives in the product
repository that carries its work, so the workflow now checks out each fleet
repository's ``contracts/`` and passes the roots as ``--contract-root``.

Falsifier: put the change-control checkout or the ``--occ-repo-dir`` argument
back in the workflow, the skill mapping or the node contract and a test here
names it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.handlers.handler_in_progress_probe_hygiene import (
    _contract_check_count,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_request import (
    ModelProbeHygieneRequest,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/in-progress-probe-hygiene.yml"
SKILL_MAPPING = ROOT / "src/omnibase_infra/cli/skill_mapping.yaml"
NODE_CONTRACT = (
    ROOT
    / "src/omnibase_infra/nodes/node_in_progress_probe_hygiene_effect/contract.yaml"
)
_OCC_TOKENS = re.compile(r"onex_change_control|\.occ-src|occ[-_]repo[-_]dir")


def _occ_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _OCC_TOKENS.search(line)]


def test_the_scan_names_a_change_control_reference() -> None:
    """Positive control: the pattern finds each shape the old workflow used."""
    assert _occ_lines("repository: OmniNode-ai/onex_change_control")
    assert _occ_lines("path: .occ-src")
    assert _occ_lines("--occ-repo-dir x")
    assert not _occ_lines("--contract-root x")


@pytest.mark.parametrize("path", [WORKFLOW, SKILL_MAPPING, NODE_CONTRACT])
def test_no_change_control_reference_in_the_sweep_surface(path: Path) -> None:
    sweep_lines = _occ_lines(path.read_text(encoding="utf-8"))
    if path == SKILL_MAPPING:
        # The mapping also holds the evidence closer's entry; only the hygiene
        # entry is in scope here.
        mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = next(
            e
            for e in mapping["skills"]
            if e["skill_name"] == "in_progress_probe_hygiene"
        )
        sweep_lines = _occ_lines(yaml.safe_dump(entry))
    assert sweep_lines == []


def test_the_workflow_passes_contract_roots() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '--contract-root "${contract_roots}"' in text
    assert "contracts/OMN-*.yaml" in text


def test_the_mapping_payload_fields_are_request_fields() -> None:
    mapping = yaml.safe_load(SKILL_MAPPING.read_text(encoding="utf-8"))
    entry = next(
        e for e in mapping["skills"] if e["skill_name"] == "in_progress_probe_hygiene"
    )
    fields = {arg["payload_field"] for arg in entry["args"]}
    assert "contract_roots" in fields
    assert fields <= set(ModelProbeHygieneRequest.model_fields)


def test_the_node_contract_declares_the_request_fields() -> None:
    contract = yaml.safe_load(NODE_CONTRACT.read_text(encoding="utf-8"))
    declared = set(contract["io_operations"][0]["input_fields"])
    assert "contract_roots" in declared
    assert declared <= set(ModelProbeHygieneRequest.model_fields) | {"correlation_id"}


def _workflow_contract_repos() -> set[str]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            repos = step.get("env", {}).get("CONTRACT_REPOS")
            if repos:
                return set(repos.split())
    raise AssertionError("no step sets CONTRACT_REPOS")


def test_the_contract_repos_are_named_and_the_private_ones_come_from_a_variable() -> (
    None
):
    text = WORKFLOW.read_text(encoding="utf-8")
    assert _workflow_contract_repos() >= {"omnibase_infra", "omnibase_core"}
    assert "vars.ONEX_CONTRACT_REPOS_PRIVATE" in text
    assert "${CONTRACT_REPOS} ${CONTRACT_REPOS_PRIVATE:-}" in text


def test_the_contract_repos_do_not_name_the_retired_repository() -> None:
    assert "onex_change_control" not in _workflow_contract_repos()


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/probe_hygiene_product_repo_contracts_omn17427.json"
)
def test_the_reader_counts_recorded_contracts_from_two_product_repositories(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    """Lay two real product-repository contracts out as the workflow does.

    The recording holds the bytes of one contract from each of two fleet
    repositories. The workflow checks each repository out as its own root with
    a ``contracts/`` directory; the reader must find each ticket in its own
    root and count the checks the YAML declares.
    """
    recording = recorded_response["response"]
    texts = "".join(row["contract"] for row in recording.values())
    assert (
        hashlib.sha256(texts.encode("utf-8")).hexdigest()
        == recorded_response["_provenance"]["source_sha256"]
    )
    roots = []
    for repo, row in recording.items():
        contracts = tmp_path / repo / "contracts"
        contracts.mkdir(parents=True)
        (contracts / f"{row['ticket']}.yaml").write_text(
            row["contract"], encoding="utf-8"
        )
        roots.append(str(tmp_path / repo))
    for row in recording.values():
        assert row["expected_checks"] > 0
        assert (
            _contract_check_count(tuple(roots), row["ticket"]) == row["expected_checks"]
        )
    assert _contract_check_count(tuple(roots), "OMN-1") == 0
