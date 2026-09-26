# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC3 of OMN-18708 -- the receipt carries the lane's own node inventory.

The acceptance criterion, and its falsifier, in the ticket's words:

    The lab-pass receipt carries the same triples, and its check's evidence is
    the lane's own introspection manifest rather than a restatement of the
    build. -- falsifier: asserts the new check appears with non-empty evidence
    on a passing fixture and that its triples equal the manifest's, and asserts
    a receipt carrying the check with empty evidence is refused.

"Rather than a restatement of the build" is the load-bearing half, and it is
what the ``test_the_evidence_names_the_lane_endpoint_it_read`` case pins: an
emitter that echoed its own input would prove only that it can echo.

Also pinned here, because both are properties a future change could quietly
break: that a receipt written before this change still parses byte-identically
(no ``RECEIPT_VERSION`` bump, no new required field), and that the gate's
refusal behaviour is unchanged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    COMPOSE_DEV_HTTP_CHECKS,
    INTROSPECTION_MANIFEST_PATH,
    NODE_INVENTORY_CHECK,
    RECEIPT_VERSION,
    EnumLabLane,
    EnumLabPassResult,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    ModelNodeInventoryTriple,
    build_receipt,
    check_node_inventory,
    load_node_inventory_json,
    main,
    parse_node_inventory,
)

pytestmark = pytest.mark.unit

HASH_A = "a" * 64
HASH_B = "b" * 64
SHA = "0" * 40
WINDOW = {
    "started_at": datetime(2026, 9, 18, 20, 0, tzinfo=UTC),
    "finished_at": datetime(2026, 9, 18, 20, 5, tzinfo=UTC),
}


def _manifest_body(contracts: list[dict[str, Any]]) -> str:
    return json.dumps({"runtime_profile": "main", "contracts": contracts})


def _patch_http(monkeypatch: pytest.MonkeyPatch, status: int, body: str) -> list[str]:
    """Record the URLs read, so a test can assert WHAT was read, not only what came back."""
    seen: list[str] = []

    def _get(url: str, _timeout: float) -> tuple[int, str]:
        seen.append(url)
        return status, body

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._http_get", _get)
    return seen


def _passing_check() -> ModelLabPassCheck:
    return ModelLabPassCheck(name="ready_main", ok=True, evidence="GET /ready -> 200")


# ---------------------------------------------------------------------------
# the check, and where its evidence comes from
# ---------------------------------------------------------------------------
class TestTheCheckReadsTheLanesOwnManifest:
    def test_the_check_passes_with_non_empty_evidence_on_a_passing_fixture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http(
            monkeypatch,
            200,
            _manifest_body(
                [
                    {
                        "name": "alpha",
                        "node_version": "1.0.0",
                        "contract_content_hash": HASH_A,
                    }
                ]
            ),
        )

        probe = check_node_inventory(
            "http://lane:8085" + INTROSPECTION_MANIFEST_PATH, 5
        )

        assert probe.check.name == NODE_INVENTORY_CHECK
        assert probe.check.ok
        assert probe.check.evidence

    def test_the_evidence_names_the_lane_endpoint_it_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a restatement of the build: the evidence cites the lane's URL."""
        url = "http://lane:8085" + INTROSPECTION_MANIFEST_PATH
        seen = _patch_http(
            monkeypatch,
            200,
            _manifest_body(
                [
                    {
                        "name": "alpha",
                        "node_version": "1.0.0",
                        "contract_content_hash": HASH_A,
                    }
                ]
            ),
        )

        probe = check_node_inventory(url, 5)

        assert seen == [url]
        assert url in probe.check.evidence
        assert INTROSPECTION_MANIFEST_PATH in probe.check.evidence

    def test_the_triples_equal_the_manifests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contracts = [
            {"name": "beta", "node_version": "2.1.0", "contract_content_hash": HASH_B},
            {"name": "alpha", "node_version": "1.0.0", "contract_content_hash": HASH_A},
        ]
        _patch_http(monkeypatch, 200, _manifest_body(contracts))

        probe = check_node_inventory("http://lane:8085/m", 5)

        assert probe.triples == parse_node_inventory(
            [
                {
                    "name": c["name"],
                    "node_version": c["node_version"],
                    "contract_content_hash": c["contract_content_hash"],
                }
                for c in contracts
            ]
        )
        assert [t.name for t in probe.triples] == ["alpha", "beta"]

    @pytest.mark.parametrize(
        ("status", "body", "reason"),
        [
            (503, '{"error":"manifest not yet built"}', "503"),
            (200, "not json", "not JSON"),
            (200, "[]", "not an object"),
            (200, '{"contracts":[]}', "absent or empty"),
            (200, '{"runtime_profile":"main"}', "absent or empty"),
        ],
    )
    def test_an_unusable_manifest_fails_the_check_and_yields_no_triples(
        self, monkeypatch: pytest.MonkeyPatch, status: int, body: str, reason: str
    ) -> None:
        _patch_http(monkeypatch, status, body)

        probe = check_node_inventory("http://lane:8085/m", 5)

        assert not probe.check.ok
        assert reason in probe.check.evidence
        assert probe.triples == ()

    def test_a_contract_with_no_content_hash_fails_rather_than_being_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silently shorter inventory is a PASS narrower than it looks."""
        _patch_http(
            monkeypatch,
            200,
            _manifest_body(
                [
                    {
                        "name": "alpha",
                        "node_version": "1.0.0",
                        "contract_content_hash": HASH_A,
                    },
                    {
                        "name": "beta",
                        "node_version": "1.0.0",
                        "contract_content_hash": None,
                    },
                ]
            ),
        )

        probe = check_node_inventory("http://lane:8085/m", 5)

        assert not probe.check.ok
        assert "beta" in probe.check.evidence
        assert probe.triples == ()

    def test_the_check_is_declared_in_the_compose_dev_check_set(self) -> None:
        assert NODE_INVENTORY_CHECK in COMPOSE_DEV_HTTP_CHECKS


# ---------------------------------------------------------------------------
# the receipt field
# ---------------------------------------------------------------------------
class TestTheReceiptCarriesTheTriples:
    def test_a_receipt_round_trips_its_inventory(self) -> None:
        receipt = build_receipt(
            sha=SHA,
            lane=EnumLabLane.COMPOSE_DEV,
            checks=[_passing_check()],
            agent_command_id=None,
            node_inventory=[
                ModelNodeInventoryTriple("alpha", "1.0.0", HASH_A),
            ],
            **WINDOW,
        )

        assert ModelLabPassReceipt.from_json(receipt.to_json()) == receipt

    def test_a_check_with_empty_evidence_is_refused(self) -> None:
        """The AC3 falsifier, which the check model already enforces for all checks."""
        with pytest.raises(ValueError, match="evidence is required"):
            ModelLabPassCheck(name=NODE_INVENTORY_CHECK, ok=True, evidence="")

    def test_a_passing_inventory_check_with_no_triples_is_refused(self) -> None:
        """The check asserts the lane named its nodes; the record must contain them."""
        with pytest.raises(ValueError, match="carries no node_inventory"):
            ModelLabPassReceipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                result=EnumLabPassResult.PASS,
                checks=(
                    ModelLabPassCheck(
                        name=NODE_INVENTORY_CHECK, ok=True, evidence="GET ... -> 200"
                    ),
                ),
                agent_command_id=None,
                **WINDOW,
            )

    def test_a_failing_inventory_check_with_no_triples_is_allowed(self) -> None:
        """A failed read has nothing to carry, and must still be recorded."""
        receipt = ModelLabPassReceipt(
            sha=SHA,
            lane=EnumLabLane.COMPOSE_DEV,
            result=EnumLabPassResult.FAIL,
            checks=(
                ModelLabPassCheck(
                    name=NODE_INVENTORY_CHECK, ok=False, evidence="GET ... -> 503"
                ),
            ),
            agent_command_id=None,
            **WINDOW,
        )

        assert receipt.node_inventory == ()

    def test_duplicate_names_in_the_inventory_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate node name"):
            ModelLabPassReceipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                result=EnumLabPassResult.PASS,
                checks=(_passing_check(),),
                agent_command_id=None,
                node_inventory=(
                    ModelNodeInventoryTriple("alpha", "1.0.0", HASH_A),
                    ModelNodeInventoryTriple("alpha", "2.0.0", HASH_B),
                ),
                **WINDOW,
            )

    @pytest.mark.parametrize("bad", ["", "deadbeef", "A" * 64, ("x" * 63)])
    def test_a_triple_refuses_a_hash_that_is_not_the_canonical_form(
        self, bad: str
    ) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            ModelNodeInventoryTriple("alpha", "1.0.0", bad)


# ---------------------------------------------------------------------------
# backwards compatibility -- the reason RECEIPT_VERSION did not move
# ---------------------------------------------------------------------------
class TestReceiptsWrittenBeforeThisChangeStillParse:
    def test_the_receipt_version_did_not_move(self) -> None:
        """A bump would have made the gate refuse every receipt already in flight."""
        assert RECEIPT_VERSION == "lab_pass_receipt.v1"

    def test_a_pre_change_receipt_parses_unchanged(self) -> None:
        legacy = json.dumps(
            {
                "receipt_version": RECEIPT_VERSION,
                "sha": SHA,
                "lane": "compose-dev",
                "started_at": "2026-09-18T20:00:00+00:00",
                "finished_at": "2026-09-18T20:05:00+00:00",
                "result": "PASS",
                "checks": [
                    {"name": "ready_main", "ok": True, "evidence": "GET -> 200"}
                ],
                "agent_command_id": None,
            }
        )

        assert ModelLabPassReceipt.from_json(legacy).node_inventory == ()

    def test_a_receipt_with_no_inventory_serialises_without_the_key(self) -> None:
        receipt = build_receipt(
            sha=SHA,
            lane=EnumLabLane.COMPOSE_DEV,
            checks=[_passing_check()],
            agent_command_id=None,
            **WINDOW,
        )

        assert "node_inventory" not in json.loads(receipt.to_json())


# ---------------------------------------------------------------------------
# the CLI plumbing that carries the triples from probe-lane to emit
# ---------------------------------------------------------------------------
class TestCliPlumbing:
    def test_probe_lane_writes_the_triples_and_emit_carries_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http(
            monkeypatch,
            200,
            _manifest_body(
                [
                    {
                        "name": "alpha",
                        "node_version": "1.0.0",
                        "contract_content_hash": HASH_A,
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt.wait_for_lane_ready",
            lambda *a, **k: type(
                "O",
                (),
                # granted_seconds added by OMN-18886: probe_compose_dev now
                # DERIVES the health-observation budget from the unused
                # settle remainder, so a stub missing it is an
                # incomplete double of the real ModelSettleOutcome. 0.0
                # granted keeps these cases single-sample, which is what
                # they were written to exercise.
                {
                    "phrase": "settled",
                    "timed_out": False,
                    "waited_seconds": 0.0,
                    "granted_seconds": 0.0,
                },
            )(),
        )
        inventory_out = tmp_path / "inventory.json"
        checks_out = tmp_path / "checks.json"
        receipt_out = tmp_path / "receipt.json"

        assert (
            main(
                [
                    "probe-lane",
                    "--lane",
                    "compose-dev",
                    "--main-url",
                    "http://lane:8085",
                    "--effects-url",
                    "http://lane:8086",
                    "--projection-url",
                    "http://lane:3002",
                    "--manifest-url",
                    "http://lane:8085",
                    "--node-inventory-out",
                    str(inventory_out),
                ]
            )
            == 0
        )
        assert json.loads(inventory_out.read_text())[0]["name"] == "alpha"

        checks_out.write_text(json.dumps([_passing_check().to_dict()]))
        assert (
            main(
                [
                    "emit",
                    "--sha",
                    SHA,
                    "--lane",
                    "compose-dev",
                    "--started-at",
                    "2026-09-18T20:00:00+00:00",
                    "--finished-at",
                    "2026-09-18T20:05:00+00:00",
                    "--checks-json",
                    str(checks_out),
                    "--node-inventory-json",
                    str(inventory_out),
                    "--agent-command-id",
                    "",
                    "--out",
                    str(receipt_out),
                ]
            )
            == 0
        )
        written = ModelLabPassReceipt.from_json(receipt_out.read_text())
        assert [t.name for t in written.node_inventory] == ["alpha"]

    def test_probe_lane_without_manifest_url_makes_no_inventory_claim(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Omitting the URL emits no check, rather than an empty inventory."""
        _patch_http(monkeypatch, 200, "{}")
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt.wait_for_lane_ready",
            lambda *a, **k: type(
                "O",
                (),
                # granted_seconds added by OMN-18886: probe_compose_dev now
                # DERIVES the health-observation budget from the unused
                # settle remainder, so a stub missing it is an
                # incomplete double of the real ModelSettleOutcome. 0.0
                # granted keeps these cases single-sample, which is what
                # they were written to exercise.
                {
                    "phrase": "settled",
                    "timed_out": False,
                    "waited_seconds": 0.0,
                    "granted_seconds": 0.0,
                },
            )(),
        )

        assert (
            main(
                [
                    "probe-lane",
                    "--lane",
                    "compose-dev",
                    "--main-url",
                    "http://lane:8085",
                    "--effects-url",
                    "http://lane:8086",
                    "--projection-url",
                    "http://lane:3002",
                ]
            )
            == 0
        )
        names = {c["name"] for c in json.loads(capsys.readouterr().out)}
        assert NODE_INVENTORY_CHECK not in names

    def test_a_missing_inventory_file_raises_rather_than_emitting_an_empty_one(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(OSError, match="No such file"):
            load_node_inventory_json(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# OMN-19802: the manifest is read AFTER the settle wait, never before it
# ---------------------------------------------------------------------------
class TestTheManifestIsReadAfterTheSettleWait:
    """``probe-lane`` must not read the manifest of a lane still coming up.

    Measured on omnimarket compose-dev receipt artifact 10909257483 (sha
    befa9cc0ab, 2026-09-26T15:24:50Z): ``node_inventory`` FAILED with
    ``Errno 111 Connection refused`` on port 8085, annotated ``[lane ready
    after 331s of a 900s settle budget]``, while ``ready_main`` on the same
    port read 200. The manifest had been read BEFORE the settle wait and then
    labelled as if it were read after. Here the lane refuses every connection
    until the wait runs, which is exactly what a freshly recreated lane does.
    """

    def test_probe_lane_reads_the_manifest_after_the_settle_wait(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        events: list[str] = []
        lane = {"up": False}
        manifest = _manifest_body(
            [
                {
                    "name": "alpha",
                    "node_version": "1.0.0",
                    "contract_content_hash": HASH_A,
                }
            ]
        )

        def _get(url: str, _timeout: float) -> tuple[int, str]:
            events.append(url)
            if not lane["up"]:
                return 0, "URLError: <urlopen error [Errno 111] Connection refused>"
            if url.endswith(INTROSPECTION_MANIFEST_PATH):
                return 200, manifest
            return 200, "{}"

        def _wait(*_a: object, **_k: object) -> object:
            events.append("<settle wait>")
            lane["up"] = True
            return type(
                "O",
                (),
                {
                    "phrase": "lane ready after 331s of a 900s settle budget",
                    "timed_out": False,
                    "waited_seconds": 331.0,
                    "granted_seconds": 0.0,
                },
            )()

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._http_get", _get)
        monkeypatch.setattr("scripts.ci.lab_pass_receipt.wait_for_lane_ready", _wait)
        inventory_out = tmp_path / "inventory.json"

        assert (
            main(
                [
                    "probe-lane",
                    "--lane",
                    "compose-dev",
                    "--main-url",
                    "http://lane:8085",
                    "--effects-url",
                    "http://lane:8086",
                    "--projection-url",
                    "http://lane:3002",
                    "--manifest-url",
                    "http://lane:8085",
                    "--node-inventory-out",
                    str(inventory_out),
                ]
            )
            == 0
        )

        manifest_reads = [
            i for i, e in enumerate(events) if e.endswith(INTROSPECTION_MANIFEST_PATH)
        ]
        wait_at = events.index("<settle wait>")
        # ONE read, and after the wait: the check and the triples still come
        # from the same response.
        assert len(manifest_reads) == 1, events
        assert manifest_reads[0] > wait_at, events

        checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)}
        assert checks[NODE_INVENTORY_CHECK]["ok"] is True, checks[NODE_INVENTORY_CHECK]
        assert json.loads(inventory_out.read_text())[0]["name"] == "alpha"
