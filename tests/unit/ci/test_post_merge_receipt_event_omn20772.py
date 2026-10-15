# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20772: a post-merge lab-pass receipt publishes one typed event.

The event carries the lane, the deployment generation, the identity the pass was
taken against, the receipt key, the verdict, each check's outcome with its error
class, the binding validity and the failing dimensions. The script that builds it
is stdlib only, so these tests are where the typed model and the script's
document are held to each other.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.models.lab_pass import (
    EnumPostMergeLabPassErrorClass,
    ModelPostMergeLabPassReceiptEvent,
)
from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

SHA = "a" * 40
LANE = receipts.EnumLabLane.COMPOSE_DEV.value
STARTED = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
FINISHED = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)
CONTAINER_ID = "c" * 64
COMMAND_ID = "6f1c2c1e-9a3b-4c55-8e0a-0b6d1d2f3a4b"
IDENTITY_PAYLOAD: dict[str, Any] = {
    "generation": {
        "container": "omninode-runtime",
        "container_id": CONTAINER_ID,
        "image": "omninode-runtime:dev",
        "revision": SHA,
    },
    "pins": {"omnibase-core": "omnibase-core==0.43.2"},
    "image_ids": {"omninode-runtime": "sha256:" + "d" * 64},
    "overlay_digest": "sha256:" + "e" * 64,
    "migration_fingerprint": "sha256:" + "f" * 64,
}


def _receipt(
    *checks: receipts.ModelLabPassCheck,
    finished_at: datetime = FINISHED,
) -> receipts.ModelLabPassReceipt:
    rows = checks or (
        receipts.ModelLabPassCheck(name="main_ready", ok=True, evidence="200"),
        receipts.ModelLabPassCheck(name="effects_ready", ok=True, evidence="200"),
    )
    return receipts.build_receipt(
        sha=SHA,
        lane=receipts.EnumLabLane.COMPOSE_DEV,
        started_at=STARTED,
        finished_at=finished_at,
        checks=rows,
        agent_command_id=COMMAND_ID,
    )


def _identity() -> receipts.ModelPostMergeIdentity:
    return receipts.parse_post_merge_identity(IDENTITY_PAYLOAD)


def _event(
    receipt: receipts.ModelLabPassReceipt,
    identity: receipts.ModelPostMergeIdentity | None = None,
    reason: str = "",
) -> dict[str, Any]:
    return receipts.build_post_merge_event(receipt, identity, reason)


def test_post_merge_receipt_event_pass_sets_every_field() -> None:
    document = _event(_receipt(), _identity())

    event = ModelPostMergeLabPassReceiptEvent.model_validate(document)

    assert event.verdict == "PASS"
    assert event.binding_valid is True
    assert event.failing_dimensions == ()
    assert event.receipt_key == receipts.artifact_name(
        receipts.EnumLabLane.COMPOSE_DEV, SHA
    )
    assert event.lane == LANE
    assert event.generation.container_ref == CONTAINER_ID
    assert event.identity.migration_fingerprint == "sha256:" + "f" * 64
    assert event.identity.pins == {"omnibase-core": "omnibase-core==0.43.2"}
    assert event.identity.image_ids == {"omninode-runtime": "sha256:" + "d" * 64}
    assert [c.name for c in event.checks] == ["main_ready", "effects_ready"]
    assert all(
        c.error_class is EnumPostMergeLabPassErrorClass.NONE for c in event.checks
    )
    # Positive control: nothing in the document is left unset.
    for key, value in document.items():
        assert value not in (None, "", [], {}) or key in {
            "binding_reason",
            "failing_dimensions",
        }, key


def test_post_merge_receipt_event_fail_names_failing_dimensions() -> None:
    receipt = _receipt(
        receipts.ModelLabPassCheck(name="main_ready", ok=True, evidence="200"),
        receipts.ModelLabPassCheck(
            name="health_dimensions", ok=False, evidence="GET -> 503 degraded"
        ),
    )

    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        _event(receipt, _identity())
    )

    assert event.verdict == "FAIL"
    assert event.receipt_result == "FAIL"
    assert event.binding_valid is True
    assert event.failing_dimensions == ("health_dimensions",)
    failing = {c.name: c for c in event.checks}["health_dimensions"]
    assert failing.outcome == "fail"
    assert failing.error_class is EnumPostMergeLabPassErrorClass.ASSERTION


def test_post_merge_receipt_event_unreadable_provenance_is_fail_binding_invalid() -> (
    None
):
    receipt = _receipt()
    assert receipt.result is receipts.EnumLabPassResult.PASS

    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        _event(receipt, None, "identity file /x/identity.json is missing")
    )

    assert event.verdict == "FAIL"
    assert event.receipt_result == "PASS"
    assert event.binding_valid is False
    assert "missing" in event.binding_reason
    assert event.failing_dimensions == ("binding",)
    assert event.generation.container_ref == ""
    assert event.identity.pins == {}


def test_post_merge_receipt_event_binding_class_failure_is_binding_invalid() -> None:
    receipt = _receipt(
        receipts.ModelLabPassCheck(name="main_ready", ok=True, evidence="200"),
        receipts.ModelLabPassCheck(
            name="probe_generation_bound", ok=False, evidence="other generation"
        ),
    )

    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        _event(receipt, _identity())
    )

    assert event.verdict == "FAIL"
    assert event.binding_valid is False
    assert event.failing_dimensions == ("binding", "probe_generation_bound")


def test_post_merge_receipt_event_model_refuses_a_pass_over_unbound_provenance() -> (
    None
):
    document = _event(_receipt(), None, "unreadable")
    document["verdict"] = "PASS"

    with pytest.raises(ValidationError):
        ModelPostMergeLabPassReceiptEvent.model_validate(document)


def test_post_merge_receipt_event_model_refuses_a_hand_edited_dimension_list() -> None:
    document = _event(_receipt(), _identity())
    document["failing_dimensions"] = ["main_ready"]

    with pytest.raises(ValidationError):
        ModelPostMergeLabPassReceiptEvent.model_validate(document)


def test_post_merge_receipt_event_identity_parser_refuses_a_partial_record() -> None:
    partial = {k: v for k, v in IDENTITY_PAYLOAD.items() if k != "overlay_digest"}

    with pytest.raises(ValueError, match="overlay_digest"):
        receipts.parse_post_merge_identity(partial)


def test_post_merge_receipt_event_cli_writes_one_event_per_emit(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(IDENTITY_PAYLOAD), encoding="utf-8")
    out = tmp_path / "receipt.json"
    event_out = tmp_path / "post-merge-event.json"

    code = receipts.main(
        [
            "emit",
            "--sha",
            SHA,
            "--lane",
            LANE,
            "--started-at",
            STARTED.isoformat(),
            "--finished-at",
            FINISHED.isoformat(),
            "--check",
            "main_ready:ok:GET -> 200",
            "--agent-command-id",
            COMMAND_ID,
            "--identity-json",
            str(identity),
            "--post-merge-event-out",
            str(event_out),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    event = ModelPostMergeLabPassReceiptEvent.model_validate_json(
        event_out.read_text(encoding="utf-8")
    )
    assert event.verdict == "PASS"
    assert event.receipt_key == receipts.artifact_name(
        receipts.EnumLabLane.COMPOSE_DEV, SHA
    )


@pytest.mark.parametrize("identity_state", ["missing", "malformed"])
def test_post_merge_receipt_event_cli_unreadable_identity_still_publishes_fail(
    tmp_path: Path, identity_state: str
) -> None:
    identity = tmp_path / "identity.json"
    if identity_state == "malformed":
        identity.write_text("{not json", encoding="utf-8")
    event_out = tmp_path / "post-merge-event.json"

    code = receipts.main(
        [
            "emit",
            "--sha",
            SHA,
            "--lane",
            LANE,
            "--started-at",
            STARTED.isoformat(),
            "--finished-at",
            FINISHED.isoformat(),
            "--check",
            "main_ready:ok:GET -> 200",
            "--agent-command-id",
            COMMAND_ID,
            "--identity-json",
            str(identity),
            "--post-merge-event-out",
            str(event_out),
            "--out",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert code == 0, "the receipt itself passed; only the event is FAIL"
    event = ModelPostMergeLabPassReceiptEvent.model_validate_json(
        event_out.read_text(encoding="utf-8")
    )
    assert event.verdict == "FAIL"
    assert event.binding_valid is False


def test_receipt_transport_error_class_for_a_refused_connection() -> None:
    # Port 9 on loopback refuses a connection: a transport exception, no answer.
    check = receipts.check_ready("main_ready", "http://127.0.0.1:9/ready", 1.0)

    assert check.ok is False
    assert check.error_class == "transport"
    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        _event(_receipt(check), _identity())
    )
    [outcome] = event.checks
    assert outcome.error_class is EnumPostMergeLabPassErrorClass.TRANSPORT
    assert outcome.error_class is not EnumPostMergeLabPassErrorClass.ASSERTION


def test_receipt_transport_error_class_not_set_for_a_wrong_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(receipts, "_http_get", lambda url, timeout: (503, "degraded"))

    check = receipts.check_ready("main_ready", "http://lane/ready", 1.0)

    assert check.ok is False
    assert check.error_class == ""
    event = ModelPostMergeLabPassReceiptEvent.model_validate(
        _event(_receipt(check), _identity())
    )
    assert event.checks[0].error_class is EnumPostMergeLabPassErrorClass.ASSERTION


def test_receipt_transport_error_class_survives_the_checks_json_hop() -> None:
    check = receipts.ModelLabPassCheck(
        name="main_ready", ok=False, evidence="URLError", error_class="transport"
    )

    assert receipts.ModelLabPassCheck.from_dict(check.to_dict()) == check
    passing = receipts.ModelLabPassCheck(name="x", ok=True, evidence="200")
    assert "error_class" not in passing.to_dict()
    with pytest.raises(ValueError, match="error_class"):
        receipts.ModelLabPassCheck(
            name="x", ok=True, evidence="200", error_class="transport"
        )


def test_receipt_event_idempotent_same_receipt_same_id() -> None:
    first = _event(_receipt(), _identity())
    second = _event(_receipt(), _identity())

    assert first["event_id"] == second["event_id"]
    assert UUID(first["event_id"])


def test_receipt_event_idempotent_distinguishes_a_different_receipt() -> None:
    first = _event(_receipt(), _identity())
    rerun = _event(
        _receipt(finished_at=datetime(2026, 10, 15, 9, 40, tzinfo=UTC)), _identity()
    )

    assert first["receipt_key"] == rerun["receipt_key"]
    assert first["event_id"] != rerun["event_id"]


def test_receipt_event_idempotent_identity_does_not_move_the_id() -> None:
    bound = _event(_receipt(), _identity())
    unbound = _event(_receipt(), None, "unreadable")

    assert bound["event_id"] == unbound["event_id"]


REPO_ROOT = Path(__file__).resolve().parents[3]


def _fake_generation(container: str) -> receipts.ModelLaneGeneration:
    return receipts.ModelLaneGeneration(
        container=container,
        container_id=CONTAINER_ID,
        image="sha256:" + "d" * 64,
        revision=SHA,
    )


def test_post_merge_receipt_event_lane_identity_is_readable_by_the_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(receipts, "read_lane_generation", _fake_generation)
    monkeypatch.setattr(
        receipts, "read_applied_migrations", lambda ledger: ("001_a", "002_b")
    )
    overlay = tmp_path / "overlay.yml"
    overlay.write_text("services: {}\n", encoding="utf-8")

    record = receipts.collect_post_merge_identity(
        container="omninode-runtime",
        image_containers=["omninode-runtime", "omninode-effects"],
        pyproject=REPO_ROOT / "pyproject.toml",
        overlay_files=[overlay],
        ledger=receipts.ModelMigrationLedger(container="pg", database="omnibase"),
    )

    identity = receipts.parse_post_merge_identity(record)
    assert "omnibase-core" in dict(identity.pins)
    assert set(dict(identity.image_ids)) == {"omninode-runtime", "omninode-effects"}
    assert identity.overlay_digest.startswith("sha256:")
    assert identity.migration_fingerprint.startswith("sha256:")


def test_post_merge_receipt_event_lane_identity_fingerprint_follows_the_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(receipts, "read_lane_generation", _fake_generation)
    overlay = tmp_path / "overlay.yml"
    overlay.write_text("services: {}\n", encoding="utf-8")
    fingerprints = []
    for applied in (("001_a",), ("001_a", "002_b")):
        monkeypatch.setattr(
            receipts, "read_applied_migrations", lambda ledger, a=applied: a
        )
        fingerprints.append(
            receipts.collect_post_merge_identity(
                container="c",
                image_containers=["c"],
                pyproject=REPO_ROOT / "pyproject.toml",
                overlay_files=[overlay],
                ledger=receipts.ModelMigrationLedger(container="pg", database="d"),
            )["migration_fingerprint"]
        )

    assert fingerprints[0] != fingerprints[1]


def test_post_merge_receipt_event_lane_identity_cli_writes_nothing_when_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unreadable(container: str) -> receipts.ModelLaneGeneration:
        msg = f"docker inspect {container} failed"
        raise ValueError(msg)

    monkeypatch.setattr(receipts, "read_lane_generation", unreadable)
    out = tmp_path / "identity.json"
    overlay = tmp_path / "overlay.yml"
    overlay.write_text("services: {}\n", encoding="utf-8")

    code = receipts.main(
        [
            "lane-identity",
            "--container",
            "omninode-runtime",
            "--pyproject",
            str(REPO_ROOT / "pyproject.toml"),
            "--overlay-file",
            str(overlay),
            "--migration-container",
            "pg",
            "--migration-database",
            "omnibase",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert not out.exists()
    identity, reason = receipts.load_post_merge_identity(out)
    assert identity is None
    assert "unreadable" in reason


def _lab_pass_steps() -> list[dict[str, Any]]:
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/runtime-rebuild-trigger.yml").read_text(
            encoding="utf-8"
        )
    )
    for job in workflow["jobs"].values():
        if any(
            "lab_pass_receipt.py emit" in step.get("run", "")
            and "--event-out" in step.get("run", "")
            for step in job.get("steps", [])
        ):
            return list(job["steps"])
    raise AssertionError("no job emits the lab-pass receipt with its bus event")


def _step_running(steps: list[dict[str, Any]], needle: str) -> int:
    for index, step in enumerate(steps):
        if needle in step.get("run", ""):
            return index
    raise AssertionError(f"no step runs {needle!r}")


def test_post_merge_receipt_event_workflow_reads_identity_emits_and_publishes() -> None:
    steps = _lab_pass_steps()
    identity = _step_running(steps, "lab_pass_receipt.py lane-identity")
    emit = _step_running(steps, "lab_pass_receipt.py emit")
    publish = _step_running(steps, "publish_lab_fact_event.py")

    assert identity < emit < publish
    assert "--identity-json" in steps[emit]["run"]
    assert "--post-merge-event-out" in steps[emit]["run"]
    assert "post-merge-event.json" in steps[publish]["run"]
    assert steps[publish]["if"] == "always()"
    assert "KAFKA_SASL_PASSWORD" not in steps[publish]["run"]
    assert "KAFKA_SASL_PASSWORD" in steps[publish]["env"]


def test_post_merge_receipt_event_publisher_takes_every_event_and_never_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "omn20772_publish_lab_fact_event",
        REPO_ROOT / "scripts" / "ci" / "publish_lab_fact_event.py",
    )
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    first = tmp_path / "event.json"
    second = tmp_path / "post-merge-event.json"
    first.write_text(json.dumps({"topic": "t.one"}), encoding="utf-8")
    second.write_text(json.dumps({"topic": "t.two"}), encoding="utf-8")

    code = publisher.main(["--event", str(first), "--event", str(second)])

    assert code == 0
    warnings = capsys.readouterr().err
    # With no overlay each document is reported on its own, so a skipped first
    # document does not hide the second.
    assert warnings.count("no CI bus overlay") == 2
