# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19508 AC4: the omnimarket release cut admits a PASS under either dev lane.

Task B6 of the second-deploy-slot plan (epic OMN-19500). The operator ruled on
2026-09-25T00:56:45Z that omnimarket changes may be proven on the second
deployed dev lane on .202 (receipt lane ``compose-dev-202``), and only
omnimarket: omnibase_infra stays proven on .201. So omnimarket's policy names
``compose-dev-or-instance-lanes`` (OMN-19543 made the instance set the routing
table's, which also carries dev-200), the loader refuses that value for a repo
no instance proves, and the premise reads ``compose-dev`` first, then each
instance lane.
Every green assertion has a refusing sibling.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_POLICY_PATH = _REPO_ROOT / "config" / "release_train_policy.yaml"


def _load(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load(_MODULE, "release_train_under_test_omn19508")
lanes = rt.lab_pass_receipt.EnumLabLane
PASS = rt.lab_pass_receipt.EnumLabPassResult.PASS
FAIL = rt.lab_pass_receipt.EnumLabPassResult.FAIL

pytestmark = pytest.mark.unit

SHA = "5be72e12f1e9cb8168271a2dd31f8f6c5656e896"


def _surface(
    receipts: dict[Any, Any],
) -> tuple[Any, Any]:
    """Fakes for the two GitHub reads, keyed by lane."""
    by_name = {
        rt.lab_pass_receipt.artifact_name(lane, SHA): (index, lane, result)
        for index, (lane, result) in enumerate(receipts.items(), start=1)
    }

    def list_artifacts(repo: str, name: str) -> list[dict[str, Any]]:
        assert repo == "OmniNode-ai/omnimarket"
        if name not in by_name:
            return []
        index, _lane, _result = by_name[name]
        return [{"id": index, "created_at": "2026-09-25T02:00:00Z"}]

    def download_receipt(repo: str, artifact_id: int) -> Any:
        for index, lane, result in by_name.values():
            if index == artifact_id:
                return SimpleNamespace(sha=SHA, lane=lane, result=result, checks=[])
        raise AssertionError(artifact_id)

    return list_artifacts, download_receipt


def _classify(receipts: dict[Any, Any], evidence: Any) -> tuple[Any, str]:
    list_artifacts, download_receipt = _surface(receipts)
    return rt.classify_lab_receipt(
        "omnimarket",
        SHA,
        list_artifacts=list_artifacts,
        download_receipt=download_receipt,
        lanes=rt.lab_evidence_lanes(evidence, "omnimarket"),
    )


EITHER = rt.EnumLabEvidence.COMPOSE_DEV_OR_INSTANCE


def test_release_train_lab_evidence_compose_dev_202_pass_alone_admits() -> None:
    reason, detail = _classify({lanes.COMPOSE_DEV_202: PASS}, EITHER)
    assert reason is None
    assert "compose-dev-202 lane" in detail


def test_release_train_lab_evidence_compose_dev_202_compose_dev_pass_still_admits() -> (
    None
):
    reason, _detail = _classify({lanes.COMPOSE_DEV: PASS}, EITHER)
    assert reason is None


def test_release_train_lab_evidence_compose_dev_202_neither_refuses_naming_both() -> (
    None
):
    reason, detail = _classify({}, EITHER)
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT
    assert "lab-pass-receipt-compose-dev-" in detail
    assert "compose-dev-202" in detail


def test_release_train_lab_evidence_compose_dev_202_fail_on_both_refuses() -> None:
    reason, _detail = _classify(
        {lanes.COMPOSE_DEV: FAIL, lanes.COMPOSE_DEV_202: FAIL}, EITHER
    )
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_FAIL


def test_release_train_lab_evidence_compose_dev_202_is_not_read_under_compose_dev() -> (
    None
):
    """Flipped sibling: a repo whose evidence is compose-dev alone is refused
    by a compose-dev-202 PASS."""
    reason, _detail = _classify(
        {lanes.COMPOSE_DEV_202: PASS}, rt.EnumLabEvidence.COMPOSE_DEV
    )
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT


def test_release_train_lab_evidence_compose_dev_202_committed_policy() -> None:
    policy = rt.load_policy(_POLICY_PATH)
    assert policy["omnimarket"].lab_evidence is EITHER
    assert policy["omnibase_infra"].lab_evidence is rt.EnumLabEvidence.COMPOSE_DEV


def test_release_train_lab_evidence_compose_dev_202_refused_for_omnibase_infra(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(_POLICY_PATH.read_text(encoding="utf-8"))
    raw["repos"]["omnibase_infra"]["lab_evidence"] = EITHER.value
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(rt.ReleaseTrainConfigError, match=r"no instance in"):
        rt.load_policy(path)
