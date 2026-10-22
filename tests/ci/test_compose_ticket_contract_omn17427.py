# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Independent branch additions compose to the existing ticket-contract format."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_ticket_contract_compose import (
    HandlerTicketContractCompose,
)
from omnibase_infra.models.model_ticket_contract_compose_request import (
    ModelTicketContractComposeRequest,
)
from omnibase_infra.models.model_ticket_contract_compose_result import (
    ModelTicketContractComposeResult,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
CONTRACT = (
    REPO / "src/omnibase_infra/nodes/node_ticket_contract_compose_effect/contract.yaml"
)
TICKET = "OMN-17427"
ROOT = f"{TICKET}.yaml"
CHECK = "python -c 'assert 1 + 1 == 2'"


def criterion(scope: str) -> str:
    """Use distinct numeric labels accepted by the pinned ticket model."""
    return "AC1" + "".join(f"{ord(char):03d}" for char in scope) + "0001"


def contribution(scope: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "ticket_id": TICKET,
        "requirements": [
            {
                "id": f"req-{scope}-1",
                "statement": "An independently authored requirement.",
                "acceptance": [
                    {
                        "id": criterion(scope),
                        "statement": "The check succeeds.",
                    }
                ],
            }
        ],
        "dod_evidence": [
            {
                "id": f"dod-{scope}-1",
                "description": "An executable check bound to the branch's criterion.",
                "source": "manual",
                "binds_ac": [criterion(scope)],
                "checks": [{"check_type": "command", "check_value": CHECK}],
            }
        ],
    }


def root_contract() -> dict:
    root = contribution("existing")
    root["title"] = "Existing shared ticket"
    root["evidence_requirements"] = [
        {"kind": "tests", "description": "Existing proof", "command": CHECK}
    ]
    return root


def write_contracts(directory: Path, documents: dict[str, dict]) -> None:
    for name, document in documents.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def compose(directory: Path) -> ModelTicketContractComposeResult:
    return asyncio.run(
        HandlerTicketContractCompose().handle(
            ModelTicketContractComposeRequest(
                contracts_dir=str(directory), ticket_ref=TICKET
            )
        )
    )


def load_composed(directory: Path) -> dict:
    result = compose(directory)
    assert result.success, result.error_message
    return load_yaml_mapping_no_duplicates(
        result.contract_yaml, source="composed contract"
    )


def test_two_branch_contract_additions_merge_without_textual_conflict(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    left = tmp_path / "branch-alpha"
    right = tmp_path / "branch-beta"
    write_contracts(base, {ROOT: root_contract()})
    write_contracts(
        left,
        {ROOT: root_contract(), f"{TICKET}.d/alpha.yaml": contribution("alpha")},
    )
    write_contracts(
        right,
        {ROOT: root_contract(), f"{TICKET}.d/beta.yaml": contribution("beta")},
    )
    for branch, scope in ((left, "alpha"), (right, "beta")):
        assert [item["id"] for item in load_composed(branch)["dod_evidence"]] == [
            "dod-existing-1",
            f"dod-{scope}-1",
        ]

    # Merge each pathname with Git's actual textual merge algorithm. Neither
    # branch changes the shared root, and the additions have distinct paths.
    outputs = []
    paths = {
        p.relative_to(tree) for tree in (left, right) for p in tree.rglob("*.yaml")
    }
    for order, (ours, theirs) in enumerate(((left, right), (right, left))):
        merged = tmp_path / f"merged-{order}"
        for path in sorted(paths):
            target = merged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            versions = [tmp_path / f"version-{index}" for index in range(3)]
            for version, tree in zip(versions, (ours, base, theirs), strict=True):
                source = tree / path
                version.write_bytes(source.read_bytes() if source.exists() else b"")
            result = subprocess.run(
                ["git", "merge-file", "-p", *(str(p) for p in versions)],
                cwd=REPO,
                env=scrub_git_location_env(os.environ),
                capture_output=True,
                check=False,
                timeout=30,
            )
            assert result.returncode == 0, (
                result.stdout.decode() + result.stderr.decode()
            )
            target.write_bytes(result.stdout)
        assert (merged / ROOT).read_bytes() == (base / ROOT).read_bytes()
        output = load_composed(merged)
        assert [item["id"] for item in output["dod_evidence"]] == [
            "dod-existing-1",
            "dod-alpha-1",
            "dod-beta-1",
        ]
        assert output["title"] == root_contract()["title"]
        assert (
            output["evidence_requirements"] == root_contract()["evidence_requirements"]
        )
        for item in output["dod_evidence"]:
            assert item["checks"][0]["check_value"] == CHECK
            assert item["binds_ac"] == [criterion(item["id"][4:-2])]
        outputs.append(output)
    assert outputs[0] == outputs[1]


def test_legacy_contract_is_unchanged(tmp_path: Path) -> None:
    write_contracts(tmp_path, {ROOT: root_contract()})
    assert load_composed(tmp_path) == root_contract()


@pytest.mark.parametrize("invalid", [False, True])
def test_packaged_contract_routes_composition_and_persists_the_verdict(
    tmp_path: Path, invalid: bool
) -> None:
    contracts = tmp_path / "contracts"
    fragment = contribution("alpha")
    if invalid:
        fragment["dod_evidence"][0]["binds_ac"] = []
    write_contracts(
        contracts, {ROOT: root_contract(), f"{TICKET}.d/alpha.yaml": fragment}
    )
    request = ModelTicketContractComposeRequest(
        contracts_dir=str(contracts), ticket_ref=TICKET
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    state_root = tmp_path / "state"
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state_root,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    readback = json.loads((state_root / "workflow_result.json").read_text())
    result = ModelTicketContractComposeResult.model_validate(
        readback["terminal_payload"]
    )
    assert runtime.exit_code == readback["exit_code"] == (1 if invalid else 0)
    assert result.success is not invalid
    if invalid:
        assert not result.contract_yaml
        assert "unbound" in result.error_message
    else:
        composed = load_yaml_mapping_no_duplicates(
            result.contract_yaml, source="terminal"
        )
        assert [item["id"] for item in composed["dod_evidence"]] == [
            "dod-existing-1",
            "dod-alpha-1",
        ]
    assert yaml.safe_load((contracts / ROOT).read_text()) == root_contract()


def test_branch_names_remain_distinct_after_verifier_label_normalization(
    tmp_path: Path,
) -> None:
    write_contracts(
        tmp_path,
        {
            ROOT: root_contract(),
            f"{TICKET}.d/alpha-one.yaml": contribution("alpha-one"),
            f"{TICKET}.d/alphaone.yaml": contribution("alphaone"),
        },
    )
    composed = load_composed(tmp_path)
    labels = [item["binds_ac"][0] for item in composed["dod_evidence"]]
    assert labels == [
        criterion(scope) for scope in ("existing", "alpha-one", "alphaone")
    ]
    # This is the pinned verifier's label rule. In particular, plain hyphen
    # namespaces AC-alpha-one-1 and AC-alphaone-1 would both become ACALPHAONE1.
    canonical = [
        label.strip().upper().replace("-", "").replace("_", "") for label in labels
    ]
    assert len(set(canonical)) == len(labels)


def test_fragment_cannot_reuse_a_normalized_legacy_criterion(tmp_path: Path) -> None:
    root = root_contract()
    legacy = criterion("alpha").replace("AC", "ac-")
    root["requirements"][0]["acceptance"][0]["id"] = legacy
    root["dod_evidence"][0]["binds_ac"] = [legacy]
    write_contracts(
        tmp_path, {ROOT: root, f"{TICKET}.d/alpha.yaml": contribution("alpha")}
    )
    result = compose(tmp_path)
    assert not result.success
    assert "duplicate acceptance id" in result.error_message


def test_appending_to_the_shared_root_reproduces_the_textual_conflict(
    tmp_path: Path,
) -> None:
    documents = [root_contract(), root_contract(), root_contract()]
    for document, scope in zip(
        (documents[0], documents[2]), ("alpha", "beta"), strict=True
    ):
        addition = contribution(scope)
        document["requirements"] += addition["requirements"]
        document["dod_evidence"] += addition["dod_evidence"]
    paths = [tmp_path / name for name in ("ours.yaml", "base.yaml", "theirs.yaml")]
    for path, document in zip(paths, documents, strict=True):
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        ["git", "merge-file", "-p", *(str(path) for path in paths)],
        cwd=REPO,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode > 0
    assert b"<<<<<<<" in result.stdout


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("ticket", "ticket_id"),
        ("schema", "schema_version"),
        ("metadata", "unsupported fragment fields"),
        ("namespace", "acceptance id"),
        ("unbound", "unbound"),
        ("foreign_binding", "unknown criterion"),
        ("empty_checks", "checks"),
        ("bad_requirements", "requirements"),
        ("duplicate_ac", "duplicate acceptance id"),
        ("duplicate_evidence", "duplicate dod_evidence id"),
    ],
)
def test_invalid_fragment_is_refused(tmp_path: Path, change: str, message: str) -> None:
    fragment = contribution("alpha")
    if change == "ticket":
        fragment["ticket_id"] = "OMN-1"
    elif change == "schema":
        fragment["schema_version"] = "2.0.0"
    elif change == "metadata":
        fragment["title"] = "Overwrite the root"
    elif change == "namespace":
        fragment["requirements"][0]["acceptance"][0]["id"] = "AC.alpha.1"
    elif change == "unbound":
        fragment["dod_evidence"][0]["binds_ac"] = []
    elif change == "foreign_binding":
        fragment["dod_evidence"][0]["binds_ac"] = [criterion("existing")]
    elif change == "empty_checks":
        fragment["dod_evidence"][0]["checks"] = []
    elif change == "bad_requirements":
        fragment["requirements"] = "not a list"
    elif change == "duplicate_ac":
        fragment["requirements"][0]["acceptance"] *= 2
    elif change == "duplicate_evidence":
        fragment["dod_evidence"] *= 2
    write_contracts(
        tmp_path, {ROOT: root_contract(), f"{TICKET}.d/alpha.yaml": fragment}
    )
    result = compose(tmp_path)
    assert not result.success
    assert message in (result.error_message or "")
    assert not result.contract_yaml


def test_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    write_contracts(tmp_path, {ROOT: root_contract()})
    fragment = tmp_path / f"{TICKET}.d/alpha.yaml"
    fragment.parent.mkdir()
    fragment.write_text("ticket_id: OMN-17427\nticket_id: OMN-17427\n")
    result = compose(tmp_path)
    assert not result.success
    assert "duplicate key" in (result.error_message or "")
    assert not result.contract_yaml


def test_cross_file_duplicate_identity_is_refused(tmp_path: Path) -> None:
    root = root_contract()
    root["requirements"] += copy.deepcopy(contribution("alpha")["requirements"])
    write_contracts(
        tmp_path, {ROOT: root, f"{TICKET}.d/alpha.yaml": contribution("alpha")}
    )
    result = compose(tmp_path)
    assert not result.success
    assert "duplicate requirements id" in (result.error_message or "")


def test_symlink_fragment_is_refused(tmp_path: Path) -> None:
    write_contracts(
        tmp_path, {ROOT: root_contract(), "outside.yaml": contribution("alpha")}
    )
    fragments = tmp_path / f"{TICKET}.d"
    fragments.mkdir()
    (fragments / "alpha.yaml").symlink_to(tmp_path / "outside.yaml")
    result = compose(tmp_path)
    assert not result.success
    assert "symlink" in (result.error_message or "")


def test_non_yaml_fragment_is_not_silently_ignored(tmp_path: Path) -> None:
    write_contracts(
        tmp_path,
        {ROOT: root_contract(), f"{TICKET}.d/alpha.yml": contribution("alpha")},
    )
    result = compose(tmp_path)
    assert not result.success
    assert ".yaml" in (result.error_message or "")
