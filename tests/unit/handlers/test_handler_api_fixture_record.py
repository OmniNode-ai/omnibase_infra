# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recorder failure controls and contract-routed HTTP capture on both bus profiles."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import os
import runpy
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
import yaml

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.handlers.handler_api_fixture_record import (
    HandlerApiFixtureRecord,
    build_signed_webhook,
    mask_value,
    validate_recorded_set,
)
from omnibase_infra.models.model_api_fixture_record_request import (
    ModelApiFixtureRecordRequest,
)
from omnibase_infra.models.model_api_fixture_record_result import (
    ModelApiFixtureRecordResult,
)

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO / "src/omnibase_infra/nodes/node_api_fixture_record_effect/contract.yaml"
)
POLICY = yaml.safe_load(CONTRACT.read_text())["config"]["recording_policy"]
RESOURCES = ("tenants", "accounts", "keycloak_users")


@pytest.fixture
def recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An actual HTTP listener, with isolated resource counts and signed hooks."""
    credential = secrets.token_hex(32)
    monkeypatch.setenv("FIXTURE_HTTP_AUTH", credential)
    monkeypatch.setenv("FIXTURE_SIGNING_SECRET", secrets.token_hex(32))
    state = {
        "counts": dict(zip(RESOURCES, (2, 3, 4), strict=True)),
        "baseline": dict(zip(RESOURCES, (2, 3, 4), strict=True)),
        "calls": [],
        "leak": "",
        "fail_route": False,
        "fail_cleanup": "",
        "unknown_key": False,
        "fail_final_count": "",
        "fail_seed": False,
        "tenant_id": str(uuid4()),
        "request_tenant_id": "",
        "scope": "fixture-" + uuid4().hex,
    }

    class Server(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def do_DELETE(self):
            self.respond()

        def respond(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            state["calls"].append((self.command, path))
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            status = 200
            body = {"ok": True}
            authenticated = hmac.compare_digest(
                self.headers.get("x-api-key", ""), credential
            )
            if not authenticated:
                status = 401
            elif path.startswith("/counts/"):
                if parse_qs(parsed.query).get("scope") != [state["scope"]]:
                    status = 403
                elif (
                    path.rsplit("/", 1)[1] == state["fail_final_count"]
                    and len([p for _, p in state["calls"] if p.startswith("/counts/")])
                    > 3
                ):
                    status = 500
                else:
                    body = {"count": state["counts"][path.rsplit("/", 1)[1]]}
            elif path == "/seed":
                for resource in RESOURCES:
                    state["counts"][resource] += 1
                if state["fail_seed"]:
                    status = 500
                body = {"tenant_id": state["tenant_id"]}
            elif path.startswith("/cleanup/"):
                resource = path.rsplit("/", 1)[1]
                if resource == state["fail_cleanup"]:
                    status = 500
                elif resource != state["leak"]:
                    state["counts"][resource] = state["baseline"][resource]
            elif path == "/signed/github":
                actual = self.headers.get("x-hub-signature-256", "")
                expected = (
                    "sha256="
                    + hmac.new(
                        os.environ["FIXTURE_SIGNING_SECRET"].encode(),
                        raw,
                        hashlib.sha256,
                    ).hexdigest()
                )
                if not hmac.compare_digest(actual, expected):
                    status = 401
                if self.headers.get("x-github-event") != "ping":
                    status = 400
            elif path == "/signed/stripe":
                signature = dict(
                    part.split("=", 1)
                    for part in self.headers.get("stripe-signature", "").split(",")
                )
                signed = signature["t"].encode() + b"." + raw
                expected = hmac.new(
                    os.environ["FIXTURE_SIGNING_SECRET"].encode(),
                    signed,
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(signature.get("v1", ""), expected):
                    status = 401
            else:
                if raw:
                    state["request_tenant_id"] = json.loads(raw).get("tenant_id", "")
                if state["fail_route"]:
                    status = 503
                body = {
                    "ok": True,
                    "email": "fixture@example.invalid",
                    "client_secret": credential,
                    "id": str(uuid4()),
                    "created_at": 1234567890,
                    "nested": {"message": "private fixture text"},
                }
                if state["unknown_key"]:
                    body["unreviewed-dynamic-key"] = True
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    fixture = {
        "routes": [
            {"method": "POST" if i < 31 else "GET", "path": f"/probe/{i}"}
            for i in range(63)
        ]
    }
    cleanups = [
        {
            "name": "cleanup-" + resource,
            "method": "DELETE",
            "path": "/cleanup/" + resource,
            "query": {"scope": "{scope}"},
            "credential_headers": {"x-api-key": "FIXTURE_HTTP_AUTH"},
        }
        for resource in RESOURCES
    ]
    routes = []
    for index, route in enumerate(fixture["routes"]):
        step = {
            "name": f"route-{index}",
            "method": route["method"],
            "path": route["path"],
            "credential_headers": {"x-api-key": "FIXTURE_HTTP_AUTH"},
            "cleanup": [c["name"] for c in cleanups]
            if route["method"] == "POST"
            else [],
        }
        case = {"path": route["path"], "step": step}
        if index in (0, 1):
            kind = "github" if index == 0 else "stripe"
            step["path"] = "/signed/" + kind
            case.update(webhook=kind, signing_secret_env="FIXTURE_SIGNING_SECRET")
        routes.append(case)
    overlay = {
        "target_kind": "throwaway_slot",
        "sanctioned": True,
        "targets": {"api": f"http://127.0.0.1:{server.server_port}"},
        "scope_prefix": state["scope"],
        "private_output_directory": str(tmp_path),
        "routes": routes,
        "counts": [
            {
                "resource": resource,
                "step": {
                    "name": "count-" + resource,
                    "method": "GET",
                    "path": "/counts/" + resource,
                    "query": {"scope": "{scope}"},
                    "credential_headers": {"x-api-key": "FIXTURE_HTTP_AUTH"},
                },
                "pointer": "/count",
            }
            for resource in RESOURCES
        ],
        "seeds": [
            {
                "name": "seed-resources",
                "method": "POST",
                "path": "/seed",
                "body": {"scope": "{scope}"},
                "credential_headers": {"x-api-key": "FIXTURE_HTTP_AUTH"},
                "cleanup": [c["name"] for c in cleanups],
            }
        ],
        "cleanup": cleanups,
        "approved_body_keys": [
            "ok",
            "email",
            "client_secret",
            "id",
            "created_at",
            "nested",
            "message",
            "zen",
            "hook_id",
            "object",
            "type",
            "created",
            "data",
        ],
    }
    fixture_path, overlay_path, output, credentials_path = (
        tmp_path / name
        for name in ("routes.json", "overlay.json", "recorded.json", "creds.json")
    )
    fixture_path.write_text(json.dumps(fixture))
    overlay_path.write_text(json.dumps(overlay))
    credentials_path.write_text(
        json.dumps(
            {
                name: os.environ[name]
                for name in ("FIXTURE_HTTP_AUTH", "FIXTURE_SIGNING_SECRET")
            }
        )
    )
    credentials_path.chmod(0o600)
    request = ModelApiFixtureRecordRequest(
        fixture_path=str(fixture_path),
        policy_path=str(CONTRACT),
        overlay_path=str(overlay_path),
        recorded_path=str(output),
        credentials_path=str(credentials_path),
    )
    try:
        yield {
            "request": request,
            "state": state,
            "overlay": overlay,
            "fixture": fixture,
            "output": output,
            "overlay_path": overlay_path,
        }
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def run_capture(recording) -> ModelApiFixtureRecordResult:
    recording["overlay_path"].write_text(json.dumps(recording["overlay"]))
    return asyncio.run(HandlerApiFixtureRecord().handle(recording["request"]))


def test_contract_bus_profile_records_and_cleans(recording, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(recording["request"].model_dump_json())
    backend = os.environ.get("ONEX_RECORD_TEST_BUS", "inmemory")
    assert backend in {"inmemory", "kafka"}
    overrides = {"event_bus": backend}
    if backend == "kafka":
        overrides["kafka_bootstrap"] = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=tmp_path / "runtime",
        input_path=request_path,
        backend_overrides=overrides,
        timeout=45,
    )
    runtime.run()
    readback = json.loads((tmp_path / "runtime/workflow_result.json").read_text())
    result = ModelApiFixtureRecordResult.model_validate(readback["terminal_payload"])
    assert runtime.exit_code == 0, result.error_code
    assert result.success and result.pair_count == 63
    assert result.before == result.after == recording["state"]["baseline"]
    assert result.cleanup_completed
    assert (
        len([p for _, p in recording["state"]["calls"] if p.startswith("/counts/")])
        == 6
    )
    assert {p for _, p in recording["state"]["calls"] if p.startswith("/cleanup/")} == {
        "/cleanup/" + resource for resource in RESOURCES
    }
    output = recording["output"]
    assert output.stat().st_mode & 0o777 == 0o600
    document = json.loads(output.read_text())
    assert len(document["pairs"]) == 63
    assert (
        document["pairs"][0]["request"]["headers"]["x-api-key"] == "<MASK:credential>"
    )
    validate_recorded_set(recording["fixture"], document, POLICY)
    leaked = any(
        value in output.read_text()
        for value in (
            os.environ["FIXTURE_HTTP_AUTH"],
            os.environ["FIXTURE_SIGNING_SECRET"],
            "fixture@example.invalid",
            recording["state"]["scope"],
        )
    )
    assert leaked is False, "private value reached recorded output"
    assert "pairs" not in readback["terminal_payload"]
    validate = recording["request"].model_copy(update={"mode": "validate"})
    validated = asyncio.run(HandlerApiFixtureRecord().handle(validate))
    assert validated.success


@pytest.mark.parametrize("resource", RESOURCES)
def test_ac3_detects_a_leak_in_each_resource(recording, resource: str) -> None:
    recording["state"]["leak"] = resource
    result = run_capture(recording)
    assert not result.success
    assert result.error_code == "resource_count_delta"
    assert result.before[resource] + 1 == result.after[resource]
    assert result.cleanup_completed
    assert not recording["output"].exists()


@pytest.mark.parametrize("failure", ["fail_route", "unknown_key", "fail_seed"])
def test_cleanup_runs_after_request_or_masking_failure(recording, failure: str) -> None:
    recording["state"][failure] = True
    result = run_capture(recording)
    assert not result.success and result.error_code == "capture_failed"
    assert result.before == result.after
    assert (
        len([p for _, p in recording["state"]["calls"] if p.startswith("/cleanup/")])
        == 3
    )
    assert not recording["output"].exists()


def test_cleanup_failure_still_cleans_other_resources_and_reads_counts(
    recording,
) -> None:
    recording["state"]["fail_cleanup"] = "keycloak_users"
    result = run_capture(recording)
    assert not result.success and result.error_code == "cleanup_failed"
    assert not result.cleanup_completed
    assert result.after["tenants"] == result.before["tenants"]
    assert result.after["accounts"] == result.before["accounts"]
    assert (
        len([p for _, p in recording["state"]["calls"] if p.startswith("/counts/")])
        == 6
    )
    assert not recording["output"].exists()


def test_count_failure_still_attempts_each_final_count(recording) -> None:
    recording["state"]["fail_final_count"] = "tenants"
    result = run_capture(recording)
    assert not result.success and result.error_code == "cleanup_failed"
    assert (
        len([p for _, p in recording["state"]["calls"] if p.startswith("/counts/")])
        == 6
    )
    assert not recording["output"].exists()


def test_cancellation_still_finishes_cleanup_and_counting(recording) -> None:
    async def cancel_after_seed() -> None:
        task = asyncio.create_task(
            HandlerApiFixtureRecord().handle(recording["request"])
        )
        for _ in range(1000):
            if ("POST", "/seed") in recording["state"]["calls"]:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("seed positive control was not reached")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_after_seed())
    assert recording["state"]["counts"] == recording["state"]["baseline"]
    assert (
        len([p for _, p in recording["state"]["calls"] if p.startswith("/counts/")])
        == 6
    )
    assert not recording["output"].exists()


@pytest.mark.parametrize("destination", ["existing", "symlink", "public-tree"])
def test_bad_private_destination_fails_before_network_io(
    recording, destination: str, tmp_path: Path
) -> None:
    if destination == "existing":
        recording["output"].write_text("preserve this file")
    elif destination == "symlink":
        target = tmp_path / "target.json"
        target.write_text("preserve this file")
        recording["output"].symlink_to(target)
    else:
        recording["overlay"]["private_output_directory"] = str(REPO)
        recording["request"] = recording["request"].model_copy(
            update={"recorded_path": str(REPO / "private-output-must-not-exist.json")}
        )
    result = run_capture(recording)
    assert not result.success
    assert not recording["state"]["calls"]
    if destination != "public-tree":
        assert recording["output"].read_text() == "preserve this file"


def test_request_query_is_masked_with_named_rules(recording) -> None:
    recording["overlay"]["routes"][2]["step"]["query"] = {
        "email": "fixture@example.invalid"
    }
    result = run_capture(recording)
    assert result.success
    pair = json.loads(recording["output"].read_text())["pairs"][2]
    assert pair["request"]["query"] == {"email": ["<MASK:identity>"]}
    assert pair["masks"]["/request/query/email/0"] == "identity"


def test_generated_seed_identity_is_bound_to_later_requests(recording) -> None:
    recording["overlay"]["seeds"][0]["bindings"] = {"tenant_id": "/tenant_id"}
    recording["overlay"]["routes"][-1]["step"]["body"] = {"tenant_id": "{tenant_id}"}
    recording["overlay"]["approved_body_keys"].append("tenant_id")
    assert run_capture(recording).success
    assert recording["state"]["request_tenant_id"] == recording["state"]["tenant_id"]
    pair = json.loads(recording["output"].read_text())["pairs"][-1]
    assert pair["request"]["body"] == {"tenant_id": "<MASK:identifier>"}


@pytest.mark.parametrize(
    "change",
    [
        "missing-route",
        "duplicate-route",
        "missing-cleanup",
        "missing-count",
        "credential-value",
        "unapproved-target",
        "unscoped-count",
        "global-on-dev",
        "rebound-scope",
        "cleanup-binding",
    ],
)
def test_bad_plan_fails_before_network_io(recording, change: str) -> None:
    overlay = recording["overlay"]
    if change == "missing-route":
        overlay["routes"].pop()
    elif change == "duplicate-route":
        overlay["routes"].append(copy.deepcopy(overlay["routes"][0]))
    elif change == "missing-cleanup":
        overlay["routes"][0]["step"]["cleanup"] = []
    elif change == "missing-count":
        overlay["counts"].pop()
    elif change == "credential-value":
        overlay["routes"][0]["step"]["credential_headers"]["x-api-key"] = (
            "literal-not-an-environment-reference"
        )
    elif change == "unapproved-target":
        overlay["sanctioned"] = False
    elif change == "unscoped-count":
        overlay["counts"][0]["step"]["query"] = {}
        overlay["counts"][0]["step"]["name"] = "count-{scope}"
    elif change == "global-on-dev":
        overlay["target_kind"] = "dev"
        overlay["targets"]["api"] = "https://recording.invalid"
        overlay["routes"][0]["step"]["scope"] = "isolated_slot"
    elif change == "rebound-scope":
        overlay["seeds"][0]["bindings"] = {"scope": "/tenant_id"}
    elif change == "cleanup-binding":
        overlay["cleanup"][0]["bindings"] = {"tenant_id": "/tenant_id"}
    result = run_capture(recording)
    assert not result.success
    assert result.error_code == "recording_precondition_failed"
    assert not recording["state"]["calls"]
    assert not recording["output"].exists()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("empty", "empty_recorded_set"),
        ("missing", "recorded_identity_mismatch"),
        ("duplicate", "recorded_identity_mismatch"),
        ("extra", "recorded_identity_mismatch"),
        ("unknown-rule", "unnamed_mask"),
        ("unannotated-mask", "unnamed_mask"),
        ("no-cleanup", "mutation_cleanup_missing"),
        ("undeclared-cleanup", "cleanup_not_declared"),
        ("raw-text", "unmasked_private_text"),
    ],
)
def test_ac1_ac2_reject_corrupt_recorded_sets(
    recording, change: str, code: str
) -> None:
    result = run_capture(recording)
    assert result.success
    document = json.loads(recording["output"].read_text())
    if change == "empty":
        document["pairs"] = []
    elif change == "missing":
        document["pairs"].pop()
    elif change == "duplicate":
        document["pairs"].append(copy.deepcopy(document["pairs"][0]))
    elif change == "extra":
        document["pairs"][0]["path"] = "/unexpected"
    elif change == "unknown-rule":
        pair = document["pairs"][2]
        pair["response"]["body"]["email"] = "<MASK:unregistered>"
        pair["masks"]["/response/body/email"] = "unregistered"
    elif change == "unannotated-mask":
        document["pairs"][0]["masks"] = {}
    elif change == "no-cleanup":
        document["pairs"][0]["cleanup"] = []
    elif change == "undeclared-cleanup":
        document["pairs"][0]["cleanup"] = ["cleanup-undeclared"]
    elif change == "raw-text":
        pair = document["pairs"][2]
        pair["response"]["body"]["email"] = "unmasked@example.invalid"
        del pair["masks"]["/response/body/email"]
    with pytest.raises(ValueError, match=code):
        validate_recorded_set(recording["fixture"], document, POLICY)


def test_masking_nested_lists_and_json_pointer_escaping() -> None:
    masks = {}
    value = {"items": [{"some/key~": "private", "count": 4, "created_at": 3}]}
    result = mask_value(
        value,
        pointer="/body",
        approved_keys=frozenset({"items", "some/key~", "count", "created_at"}),
        masks=masks,
    )
    assert result == {
        "items": [
            {
                "some/key~": "<MASK:private_text>",
                "count": 4,
                "created_at": "<MASK:volatile>",
            }
        ]
    }
    assert masks == {
        "/body/items/0/some~1key~0": "private_text",
        "/body/items/0/created_at": "volatile",
    }


@pytest.mark.parametrize("kind", ["github", "stripe"])
def test_planted_webhook_signatures_cover_exact_raw_bytes(kind: str) -> None:
    secret = secrets.token_bytes(32)
    body, headers = build_signed_webhook(
        kind, secret=secret, delivery=uuid4(), timestamp=1234567890
    )
    if kind == "github":
        signature = headers["x-hub-signature-256"][7:]
        signed = body
        assert headers["x-github-event"] == "ping"
    else:
        parts = dict(
            part.split("=", 1) for part in headers["stripe-signature"].split(",")
        )
        signature = parts["v1"]
        signed = parts["t"].encode() + b"." + body
        assert json.loads(body)["type"] == "fixture.recording"
    assert hmac.compare_digest(
        signature, hmac.new(secret, signed, hashlib.sha256).hexdigest()
    )
    assert not hmac.compare_digest(
        signature, hmac.new(secret, signed + b" ", hashlib.sha256).hexdigest()
    )
    assert not hmac.compare_digest(
        signature, hmac.new(secrets.token_bytes(32), signed, hashlib.sha256).hexdigest()
    )


def test_node_contract_declares_real_handler_and_profiles() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["runtime_profiles"] == ["main"]
    assert (
        contract["handler_routing"]["handlers"][0]["handler"]["name"]
        == "HandlerApiFixtureRecord"
    )
    assert contract["event_bus"]["subscribe_topics"]
    assert contract["terminal_event"] in contract["event_bus"]["publish_topics"]


def test_independent_acceptance_probes_and_their_failure_controls(
    recording, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_capture(recording).success
    monkeypatch.setenv("ONEX_RECORD_ROUTE_FIXTURE", recording["request"].fixture_path)
    monkeypatch.setenv("ONEX_RECORD_RECORDED_SET", str(recording["output"]))
    probes = runpy.run_path(
        str(REPO / "tests/acceptance/onex_api_recording_acceptance.py")
    )
    probes["test_ac1_recorded_identities_equal_inventory"]()
    probes["test_ac2_named_masks_and_mutation_cleanup"]()
    probes["test_ac3_count_receipt_has_no_resource_delta"]()
    document = json.loads(recording["output"].read_text())
    document["pairs"][0]["masks"] = {}
    recording["output"].write_text(json.dumps(document))
    with pytest.raises(AssertionError, match="annotations"):
        probes["test_ac2_named_masks_and_mutation_cleanup"]()
    document["lifecycle"]["after"]["keycloak_users"] += 1
    recording["output"].write_text(json.dumps(document))
    with pytest.raises(AssertionError):
        probes["test_ac3_count_receipt_has_no_resource_delta"]()
    recording["output"].write_text('{"pairs": []}')
    with pytest.raises(AssertionError, match="multiplicities"):
        probes["test_ac1_recorded_identities_equal_inventory"]()
    with pytest.raises(AssertionError, match="empty recorded set"):
        probes["test_ac2_named_masks_and_mutation_cleanup"]()
