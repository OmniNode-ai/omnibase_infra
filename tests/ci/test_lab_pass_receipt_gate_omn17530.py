# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17530 — the lab-pass receipt contract and the fail-closed staging gate.

``omni_home`` ``CLAUDE.md`` Operating Rule 24(b): delivery to staging fails
closed unless a passing lab receipt exists for the delivered sha. These tests
pin the four branches that decide whether that sentence is true —

  present PASS -> continue; FAIL -> fail; missing -> fail; malformed -> fail

— plus the receipt-model invariants that stop a receipt lying about itself.

The gate's failure branches matter more than its success branch: a gate that
fails open on an unreadable surface is indistinguishable from no gate, and
"nothing was red" is exactly how the defects rule 24 exists for reached
``onex-dev``.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    RECEIPT_VERSION,
    EnumLabLane,
    EnumLabPassResult,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    artifact_name,
    build_receipt,
    check_health_dimensions,
    evaluate_gate,
    parse_check_argument,
    parse_receipt,
)

pytestmark = pytest.mark.unit

SHA = "98b9fb764dc19ce32dc4c4b382266b659f78cb9e"
OTHER_SHA = "7f6c2b8dbd89708f9e1521a74e25682beb7eeb0f"
REPO = "OmniNode-ai/omnibase_infra"

STARTED = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 8, 12, 20, 0, tzinfo=UTC)


def _checks(*, ok: bool = True) -> list[ModelLabPassCheck]:
    return [
        ModelLabPassCheck(name="ready_main", ok=True, evidence="GET /ready -> 200"),
        ModelLabPassCheck(
            name="health_dimensions",
            ok=ok,
            evidence="GET /health -> 200, 7 dimensions, all healthy"
            if ok
            else "GET /health -> 200, 7 dimensions, unhealthy: ['consumer_coverage']",
        ),
    ]


def _receipt(
    *,
    sha: str = SHA,
    lane: EnumLabLane = EnumLabLane.COMPOSE_DEV,
    ok: bool = True,
) -> ModelLabPassReceipt:
    return build_receipt(
        sha=sha,
        lane=lane,
        started_at=STARTED,
        finished_at=FINISHED,
        checks=_checks(ok=ok),
        agent_command_id="bdf92958-8ee6-4be4-9df0-04d3e15d9c59",
    )


# ---------------------------------------------------------------------------
# The receipt contract
# ---------------------------------------------------------------------------
class TestReceiptModel:
    def test_verdict_is_derived_from_the_checks_not_supplied(self) -> None:
        assert _receipt(ok=True).result is EnumLabPassResult.PASS
        assert _receipt(ok=False).result is EnumLabPassResult.FAIL

    def test_pass_carrying_a_failed_check_is_refused(self) -> None:
        """The 'green while doing nothing' shape, refused at the model."""
        with pytest.raises(ValueError, match="must be supported by every check"):
            ModelLabPassReceipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                result=EnumLabPassResult.PASS,
                checks=tuple(_checks(ok=False)),
                agent_command_id=None,
            )

    def test_fail_with_every_check_passing_is_refused(self) -> None:
        """Refused in both directions: a FAIL cannot hide an unrecorded check."""
        with pytest.raises(ValueError, match="contradicts"):
            ModelLabPassReceipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                result=EnumLabPassResult.FAIL,
                checks=tuple(_checks(ok=True)),
                agent_command_id=None,
            )

    def test_empty_checks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checks is empty"):
            ModelLabPassReceipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                result=EnumLabPassResult.FAIL,
                checks=(),
                agent_command_id=None,
            )

    @pytest.mark.parametrize("bad", ["98b9fb7", SHA.upper(), "", "not-a-sha"])
    def test_abbreviated_or_malformed_sha_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="40-character lowercase"):
            build_receipt(
                sha=bad,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=_checks(),
                agent_command_id=None,
            )

    def test_agent_command_id_has_no_default(self) -> None:
        """Rule 8: an emitter states it or states null; it is never guessed."""
        with pytest.raises((ValueError, TypeError)):
            ModelLabPassReceipt(  # type: ignore[call-arg]
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                result=EnumLabPassResult.PASS,
                checks=tuple(_checks()),
            )

    def test_duplicate_check_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate check names"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=[
                    ModelLabPassCheck(name="ready_main", ok=True, evidence="a"),
                    ModelLabPassCheck(name="ready_main", ok=True, evidence="b"),
                ],
                agent_command_id=None,
            )

    def test_check_evidence_is_required_on_a_passing_check(self) -> None:
        """An `ok: true` with no evidence is a check that was never run."""
        with pytest.raises((ValueError, TypeError)):
            ModelLabPassCheck(name="ready_main", ok=True, evidence="")

    def test_finished_before_started_is_refused(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=FINISHED,
                finished_at=STARTED,
                checks=_checks(),
                agent_command_id=None,
            )

    def test_artifact_name_carries_the_exact_sha_and_lane(self) -> None:
        assert (
            artifact_name(EnumLabLane.ONEX_LAB, SHA)
            == f"lab-pass-receipt-onex-lab-{SHA}"
        )
        assert artifact_name(EnumLabLane.COMPOSE_DEV, SHA) != artifact_name(
            EnumLabLane.COMPOSE_DEV, OTHER_SHA
        )

    def test_a_governed_lane_cannot_name_itself_in_a_receipt(self) -> None:
        """prod / stability-test / judge are not lab lanes and never will be."""
        for governed in ("prod", "stability-test", "judge", "lakshman"):
            with pytest.raises(ValueError, match="is not a valid EnumLabLane"):
                EnumLabLane(governed)

    def test_the_module_imports_nothing_outside_the_stdlib(self) -> None:
        """The regression that took a real delivery down, pinned.

        Run 34235502322 (2026-09-08T14:14:36Z) died at import with
        ``ModuleNotFoundError: No module named 'pydantic'`` AFTER all four of
        its checks had passed, and took the whole dev-candidate delivery with
        it. Both call sites run on a bare runner with no project environment,
        and the boot gate checks this repository out into a subdirectory so it
        cannot even reference the shared setup action. A third-party import here
        is therefore not a style question -- it breaks the delivery path.
        """
        import ast
        import sys
        from pathlib import Path as _Path

        module = _Path("scripts/ci/lab_pass_receipt.py")
        tree = ast.parse(module.read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        non_stdlib = sorted(roots - set(sys.stdlib_module_names))
        assert not non_stdlib, (
            f"scripts/ci/lab_pass_receipt.py imports {non_stdlib}, which the "
            "bare runners that emit and read receipts do not have"
        )

    def test_round_trips_through_json(self) -> None:
        original = _receipt()
        assert parse_receipt(original.to_json()) == original


class TestSerialisation:
    """Hand-written parsing has to refuse everything a model would have."""

    def test_an_unknown_receipt_field_is_refused(self) -> None:
        payload = json.loads(_receipt().to_json())
        payload["parity_exclusions"] = ["msk"]
        with pytest.raises(ValueError, match="unknown receipt field"):
            ModelLabPassReceipt.from_json(json.dumps(payload))

    def test_a_missing_receipt_field_is_refused(self) -> None:
        payload = json.loads(_receipt().to_json())
        del payload["agent_command_id"]
        with pytest.raises(ValueError, match="missing required field"):
            ModelLabPassReceipt.from_json(json.dumps(payload))

    def test_an_unknown_lane_is_refused(self) -> None:
        payload = json.loads(_receipt().to_json())
        payload["lane"] = "prod"
        with pytest.raises(ValueError, match="is not a valid EnumLabLane"):
            ModelLabPassReceipt.from_json(json.dumps(payload))

    def test_an_unknown_check_field_is_refused(self) -> None:
        payload = json.loads(_receipt().to_json())
        payload["checks"][0]["severity"] = "high"
        with pytest.raises(ValueError, match="unknown check field"):
            ModelLabPassReceipt.from_json(json.dumps(payload))


class TestCheckArgumentParsing:
    def test_evidence_may_contain_colons(self) -> None:
        parsed = parse_check_argument("ready_main:ok:GET http://x:8085/ready -> 200")
        assert parsed.name == "ready_main"
        assert parsed.ok is True
        assert parsed.evidence == "GET http://x:8085/ready -> 200"

    @pytest.mark.parametrize("bad", ["ready_main:ok", "ready_main", ""])
    def test_a_check_without_evidence_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not 'name:ok\\|fail:evidence'"):
            parse_check_argument(bad)

    def test_an_unknown_verdict_word_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 'ok' or 'fail'"):
            parse_check_argument("ready_main:probably:whatever")


class TestHealthDimensionProbe:
    """An absent dimension set is not a healthy dimension set."""

    def test_missing_dimensions_object_fails(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (200, json.dumps({"status": "ok"})),
        )
        check = check_health_dimensions("http://lane/health", 1.0)
        assert check.ok is False
        assert "not a healthy dimension set" in check.evidence

    def test_unhealthy_dimension_is_named(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (
                200,
                json.dumps(
                    {
                        "dimensions": {
                            "broker": {"status": "healthy"},
                            "consumer_coverage": {"status": "degraded"},
                        }
                    }
                ),
            ),
        )
        check = check_health_dimensions("http://lane/health", 1.0)
        assert check.ok is False
        assert "consumer_coverage" in check.evidence

    def test_all_healthy_passes(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (
                200,
                json.dumps({"dimensions": {"broker": "healthy", "db": "ok"}}),
            ),
        )
        assert check_health_dimensions("http://lane/health", 1.0).ok is True

    def test_transport_failure_fails_closed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (0, "URLError: connection refused"),
        )
        check = check_health_dimensions("http://lane/health", 1.0)
        assert check.ok is False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def _zip_of(body: str, member: str = "receipt.json") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, body)
    return buffer.getvalue()


class _Surface:
    """A fake GitHub artifact surface, keyed by artifact name exactly as the
    real REST API is."""

    def __init__(self, bodies: dict[str, str], *, list_raises: bool = False) -> None:
        self.bodies = bodies
        self.list_raises = list_raises
        self.ids = {name: 1000 + i for i, name in enumerate(sorted(bodies))}

    def __call__(self, path: str) -> bytes:
        if self.list_raises:
            msg = "`gh api` exited 1: HTTP 503"
            raise RuntimeError(msg)
        if "/actions/artifacts?name=" in path:
            name = path.split("name=")[1].split("&")[0]
            if name not in self.bodies:
                return json.dumps({"artifacts": []}).encode()
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "id": self.ids[name],
                            "name": name,
                            "expired": False,
                            "created_at": "2026-09-08T12:20:00Z",
                        }
                    ]
                }
            ).encode()
        artifact_id = int(path.split("/artifacts/")[1].split("/")[0])
        name = next(n for n, i in self.ids.items() if i == artifact_id)
        return _zip_of(self.bodies[name])


def _run_gate(surface: _Surface, monkeypatch: Any, sha: str = SHA) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    out = io.StringIO()
    code = evaluate_gate(REPO, sha, list(EnumLabLane), out)
    return code, out.getvalue()


class TestGate:
    def test_present_pass_continues(self, monkeypatch: Any) -> None:
        receipt = _receipt()
        surface = _Surface({artifact_name(receipt.lane, SHA): receipt.to_json()})
        code, output = _run_gate(surface, monkeypatch)
        assert code == 0
        # AC: the gate PRINTS sha, lane, result and checks.
        assert SHA in output
        assert "compose-dev" in output
        assert "PASS" in output
        assert "ready_main" in output
        assert "health_dimensions" in output

    def test_fail_receipt_fails_and_names_the_sha(self, monkeypatch: Any) -> None:
        receipt = _receipt(ok=False)
        surface = _Surface({artifact_name(receipt.lane, SHA): receipt.to_json()})
        code, output = _run_gate(surface, monkeypatch)
        assert code == 1
        assert f"lab-pass gate FAILED for {SHA}" in output
        # The failing check is still rendered: the gate says WHY, not just no.
        assert "[FAIL] health_dimensions" in output

    def test_missing_receipt_fails_and_is_not_a_skip(self, monkeypatch: Any) -> None:
        code, output = _run_gate(_Surface({}), monkeypatch)
        assert code == 1
        assert f"lab-pass gate FAILED for {SHA}" in output
        assert "no receipt artifact named" in output
        assert "This is not a skip" in output

    def test_malformed_receipt_fails(self, monkeypatch: Any) -> None:
        surface = _Surface(
            {artifact_name(EnumLabLane.COMPOSE_DEV, SHA): '{"sha": "nope"}'}
        )
        code, output = _run_gate(surface, monkeypatch)
        assert code == 1
        assert "malformed" in output
        assert SHA in output

    def test_unknown_receipt_version_is_refused(self, monkeypatch: Any) -> None:
        payload = json.loads(_receipt().to_json())
        payload["receipt_version"] = "lab_pass_receipt.v2"
        surface = _Surface(
            {artifact_name(EnumLabLane.COMPOSE_DEV, SHA): json.dumps(payload)}
        )
        code, _ = _run_gate(surface, monkeypatch)
        assert code == 1

    def test_a_receipt_for_a_different_sha_cannot_satisfy_the_gate(
        self, monkeypatch: Any
    ) -> None:
        """The exactness requirement, driven rather than asserted.

        A receipt filed under this sha's NAME whose PAYLOAD names another
        commit is a name/payload disagreement, and the gate refuses it. There
        is no descendant window: the plan of record grants none.
        """
        wrong = _receipt(sha=OTHER_SHA)
        surface = _Surface(
            {artifact_name(EnumLabLane.COMPOSE_DEV, SHA): wrong.to_json()}
        )
        code, output = _run_gate(surface, monkeypatch)
        assert code == 1
        assert "the name and the payload disagree" in output

    def test_a_surface_read_error_fails_closed(self, monkeypatch: Any) -> None:
        """Rule 16: an errored sweep must not read as a clean bill of health."""
        code, output = _run_gate(_Surface({}, list_raises=True), monkeypatch)
        assert code == 1
        assert "unreadable" in output

    def test_either_lab_lane_satisfies_the_gate(self, monkeypatch: Any) -> None:
        """Rule 24(b) asks for 'a passing lab receipt', not a specific lane's."""
        receipt = _receipt(lane=EnumLabLane.ONEX_LAB)
        surface = _Surface(
            {artifact_name(EnumLabLane.ONEX_LAB, SHA): receipt.to_json()}
        )
        code, output = _run_gate(surface, monkeypatch)
        assert code == 0
        assert "onex-lab" in output

    def test_an_abbreviated_sha_is_refused_before_any_lookup(
        self, monkeypatch: Any
    ) -> None:
        code, output = _run_gate(_Surface({}), monkeypatch, sha="98b9fb7")
        assert code == 1
        assert "Refusing to resolve an abbreviated ref" in output

    def test_there_is_no_override_flag(self) -> None:
        """AC: no --force, no skip input. Asserted against the parser itself, so
        adding one is a test failure rather than a review catch."""
        from scripts.ci.lab_pass_receipt import build_parser

        # Read the parser's own option strings, not its rendered help: the help
        # text quotes "--force" while explaining that there isn't one, and a
        # test that greps prose would fail on its own documentation.
        subparsers = next(
            action
            for action in build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        options = {
            option
            for parser in subparsers.choices.values()
            for action in parser._actions
            for option in action.option_strings
        }
        for banned in ("--force", "--skip", "--allow-missing", "--warn-only"):
            assert banned not in options, (
                f"{banned} is an override the gate must not have"
            )


class TestGateWiring:
    """The gate is only a gate if the delivery workflow actually runs it."""

    def test_the_delivery_workflow_runs_the_gate_before_dispatch(self) -> None:
        from pathlib import Path

        workflow = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")
        text = workflow.read_text(encoding="utf-8")
        assert "lab_pass_receipt.py gate" in text, (
            "deliver-dev-candidate-to-staging.yml no longer runs the lab-pass "
            "gate; rule 24(b) would be doctrine again."
        )
        assert "lab-pass-gate" in text

    def test_the_dispatch_job_depends_on_the_gate(self) -> None:
        from pathlib import Path

        import yaml

        workflow = yaml.safe_load(
            Path(".github/workflows/deliver-dev-candidate-to-staging.yml").read_text(
                encoding="utf-8"
            )
        )
        needs = workflow["jobs"]["dispatch-to-staging"]["needs"]
        assert "lab-pass-gate" in needs, (
            "dispatch-to-staging must not be reachable without the lab-pass gate"
        )
        # No `if:` override — GitHub's default success() is what makes a skipped
        # or cancelled gate leave the dispatch unfired.
        assert "if" not in workflow["jobs"]["dispatch-to-staging"]

    def test_both_lab_emitters_publish_a_receipt_artifact(self) -> None:
        from pathlib import Path

        deliver = Path(
            ".github/workflows/deliver-dev-candidate-to-staging.yml"
        ).read_text(encoding="utf-8")
        rebuild = Path(".github/workflows/runtime-rebuild-trigger.yml").read_text(
            encoding="utf-8"
        )
        # onex-lab, from the boot gate; compose-dev, from the .201 convergence job.
        assert "lab_pass_receipt.py emit" in deliver
        assert "--lane onex-lab" in deliver
        assert "lab_pass_receipt.py emit" in rebuild
        assert "--lane compose-dev" in rebuild
