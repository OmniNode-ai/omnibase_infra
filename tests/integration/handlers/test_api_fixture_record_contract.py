# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch the recorder contract across HTTP and inspect its persisted readback."""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.handlers.handler_api_fixture_record import validate_recorded_set
from omnibase_infra.models.model_api_fixture_record_request import (
    ModelApiFixtureRecordRequest,
)
from omnibase_infra.models.model_api_fixture_record_result import (
    ModelApiFixtureRecordResult,
)

_CONTRACT = Path(__file__).resolve().parents[3] / (
    "src/omnibase_infra/nodes/node_api_fixture_record_effect/contract.yaml"
)
_RESOURCES = ("tenants", "accounts", "keycloak_users")


@pytest.mark.parametrize("failure", ["", "route", "leak"])
def test_contract_records_http_pairs_and_reads_back_cleanup(
    httpserver: HTTPServer, tmp_path: Path, failure: str
) -> None:
    credential = secrets.token_hex(32)
    scope = "omn22158-integration-" + uuid4().hex
    counts = dict.fromkeys(_RESOURCES, 0)
    calls: list[tuple[str, str]] = []

    def respond(request: Request) -> Response:
        calls.append((request.method, request.path))
        assert request.headers.get("x-api-key") == credential
        assert request.args.get("scope") == scope
        if request.path.startswith("/counts/"):
            body = {"count": counts[request.path.rsplit("/", 1)[1]]}
        elif request.path == "/seed":
            assert request.json == {"scope": scope}
            counts.update(dict.fromkeys(_RESOURCES, 1))
            body = {"ok": True}
        elif request.path.startswith("/cleanup/"):
            resource = request.path.rsplit("/", 1)[1]
            if failure != "leak" or resource != "accounts":
                counts[resource] = 0
            body = {"ok": True}
        else:
            body = {"id": scope, "message": credential}
        status = 503 if failure == "route" and request.path == "/route/7" else 200
        return Response(
            json.dumps(body), status=status, content_type="application/json"
        )

    # Each request uses a real socket; response failures still require cleanup.
    httpserver.expect_request(re.compile("/.*")).respond_with_handler(respond)
    headers = {"x-api-key": "FIXTURE_HTTP_AUTH"}
    query = {"scope": "{scope}"}
    cleanup = [
        {
            "name": "cleanup-" + resource,
            "method": "DELETE",
            "path": "/cleanup/" + resource,
            "query": query,
            "credential_headers": headers,
        }
        for resource in _RESOURCES
    ]
    routes = [
        {"method": "POST" if index == 0 else "GET", "path": f"/route/{index}"}
        for index in range(63)
    ]
    overlay = {
        "target_kind": "throwaway_slot",
        "sanctioned": True,
        "targets": {"api": httpserver.url_for("/")},
        "scope_prefix": scope,
        "private_output_directory": str(tmp_path),
        "timeout_seconds": 0.75,
        "routes": [
            {
                "path": route["path"],
                "step": {
                    "name": f"route-{index}",
                    "method": route["method"],
                    "path": route["path"],
                    "query": query,
                    "credential_headers": headers,
                    "cleanup": [step["name"] for step in cleanup] if index == 0 else [],
                },
            }
            for index, route in enumerate(routes)
        ],
        "counts": [
            {
                "resource": resource,
                "pointer": "/count",
                "step": {
                    "name": "count-" + resource,
                    "method": "GET",
                    "path": "/counts/" + resource,
                    "query": query,
                    "credential_headers": headers,
                },
            }
            for resource in _RESOURCES
        ],
        "seeds": [
            {
                "name": "seed",
                "method": "POST",
                "path": "/seed",
                "body": {"scope": "{scope}"},
                "query": query,
                "credential_headers": headers,
                "cleanup": [step["name"] for step in cleanup],
            }
        ],
        "cleanup": cleanup,
        "approved_body_keys": ["id", "message"],
    }
    fixture_path = tmp_path / "inventory.json"
    fixture_path.write_text(json.dumps({"routes": routes}))
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay))
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps({"FIXTURE_HTTP_AUTH": credential}))
    credentials_path.chmod(0o600)
    output = tmp_path / "recorded.json"
    request = ModelApiFixtureRecordRequest(
        fixture_path=str(fixture_path),
        policy_path=str(_CONTRACT),
        overlay_path=str(overlay_path),
        credentials_path=str(credentials_path),
        recorded_path=str(output),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    runtime = RuntimeLocal(
        workflow_path=_CONTRACT,
        state_root=tmp_path / "runtime",
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=45,
    )
    runtime.run()
    readback = json.loads((tmp_path / "runtime/workflow_result.json").read_text())
    result = ModelApiFixtureRecordResult.model_validate(readback["terminal_payload"])
    httpserver.check_assertions()
    assert runtime.exit_code == 0
    assert result.cleanup_completed
    assert result.before == dict.fromkeys(_RESOURCES, 0)
    assert result.after == counts
    assert [path for method, path in calls if method == "DELETE"] == [
        "/cleanup/" + resource for resource in reversed(_RESOURCES)
    ]
    assert len([path for _, path in calls if path.startswith("/counts/")]) == 6
    assert "pairs" not in readback["terminal_payload"]
    if failure:
        assert not result.success
        assert result.error_code == (
            "capture_failed" if failure == "route" else "resource_count_delta"
        )
        assert not output.exists()
        return
    assert result.success and result.pair_count == 63
    assert result.before == result.after
    assert output.stat().st_mode & 0o777 == 0o600
    document = json.loads(output.read_text())
    policy = yaml.safe_load(_CONTRACT.read_text())["config"]["recording_policy"]
    validate_recorded_set({"routes": routes}, document, policy)
    assert len(document["pairs"]) == 63
    assert document["pairs"][0]["response"]["body"] == {
        "id": "<MASK:identifier>",
        "message": "<MASK:private_text>",
    }
    assert all(
        pair["cleanup"] for pair in document["pairs"] if pair["method"] == "POST"
    )
    assert credential not in output.read_text()
    assert scope not in output.read_text()
