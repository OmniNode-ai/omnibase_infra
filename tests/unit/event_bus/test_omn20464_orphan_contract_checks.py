# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20464: the orphan line names its consumer group, and the contract's checks can fail.

``contracts/OMN-20464.yaml`` carries two local-done-gate commands that run inside
the dev runtime container. A command that prints and exits 0 whether or not the
group is present cannot fail, so an un-rolled runtime would pass it. These tests
run the contract's own command text, with the container hop removed, against
the model as it is and against the model as it was before the change, and hold
the exit status to non-zero without the group and zero with it.
"""

from __future__ import annotations

import http.server
import json
import shlex
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.models.event_bus.model_orphaned_dispatch import (
    ModelOrphanedDispatch,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "contracts" / "OMN-20464.yaml"
MODEL_SOURCE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "models"
    / "event_bus"
    / "model_orphaned_dispatch.py"
)
MODEL_MODULE = "omnibase_infra.models.event_bus.model_orphaned_dispatch"
GROUP_LINE_SOURCE = '            f"group={self.consumer_group} "\n'
HEALTH_URL = "http://localhost:8085/health"
SUBPROCESS_TIMEOUT_SECONDS = 60

#: Runs the contract's python code with the model module loaded from a given
#: source file instead of the installed one.
_PRE_IMAGE_RUNNER = """
import sys, types
module_name, source_path, code = sys.argv[1], sys.argv[2], sys.argv[3]
for parent in ("omnibase_infra", "omnibase_infra.models", "omnibase_infra.models.event_bus"):
    sys.modules[parent] = types.ModuleType(parent)
module = types.ModuleType(module_name)
sys.modules[module_name] = module
exec(compile(open(source_path).read(), source_path, "exec"), module.__dict__)
exec(code)
"""


def _check_value(item_id: str) -> str:
    contract = yaml.safe_load(CONTRACT.read_text())
    (item,) = [i for i in contract["dod_evidence"] if i["id"] == item_id]
    (check,) = item["checks"]
    return str(check["check_value"])


def _python_code_of(command: str) -> str:
    """The ``python -c`` program of a ``docker exec <container> python -c '<code>'`` command."""
    tokens = shlex.split(command)
    assert tokens[:2] == ["docker", "exec"], tokens[:3]
    assert tokens[3:5] == ["python", "-c"], tokens[3:5]
    assert len(tokens) == 6, "the program must be the last argument"
    return tokens[5]


def _orphan(**overrides: Any) -> ModelOrphanedDispatch:
    values: dict[str, Any] = {
        "topic": "t",
        "consumer_group": "g",
        "subscription_key": "s",
        "correlation_id": uuid.uuid4(),
        "record_coordinate": (0, 1),
        "started_at": 0.0,
        "abandoned_at": 0.0,
    }
    values.update(overrides)
    return ModelOrphanedDispatch(**values)


# --- the model ---------------------------------------------------------------


def test_the_line_carries_the_full_group_token_between_offset_and_subscription() -> (
    None
):
    line = _orphan(consumer_group="dev.omnibase_infra.node_x.consume.v1").describe(1.0)
    assert line == (
        "t partition=0 offset=1 group=dev.omnibase_infra.node_x.consume.v1 "
        "subscription=s age=1s"
    )


def test_the_group_is_a_required_string_so_the_line_never_reads_group_none() -> None:
    values = _orphan().model_dump()
    del values["consumer_group"]
    with pytest.raises(ValidationError):
        ModelOrphanedDispatch(**values)
    with pytest.raises(ValidationError):
        _orphan(consumer_group=None)
    assert "group=None" not in _orphan().describe(1.0)


# --- the installed-and-read command (local done gate) ------------------------


def test_the_installed_check_exits_zero_on_the_model_with_the_group() -> None:
    code = _python_code_of(_check_value("dod-omn20464-ac1-installed-and-read"))
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "orphan group ok" in done.stdout


def test_the_installed_check_exits_nonzero_on_the_model_without_the_group(
    tmp_path: Path,
) -> None:
    """The pre-change model (the group line removed) must fail the contract command."""
    source = MODEL_SOURCE.read_text()
    assert source.count(GROUP_LINE_SOURCE) == 1, (
        "the model no longer has the group line"
    )
    pre_image = tmp_path / "model_orphaned_dispatch_pre_change.py"
    pre_image.write_text(source.replace(GROUP_LINE_SOURCE, ""))
    code = _python_code_of(_check_value("dod-omn20464-ac1-installed-and-read"))
    done = subprocess.run(
        [sys.executable, "-c", _PRE_IMAGE_RUNNER, MODEL_MODULE, str(pre_image), code],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert done.returncode != 0, (
        "the installed-and-read check passed on a model with no group in the line"
    )
    assert "AssertionError" in done.stderr
    assert "orphan group ok" not in done.stdout


# --- the runtime read (local done gate) --------------------------------------


@contextmanager
def _health_server(body: dict[str, Any]) -> Iterator[str]:
    payload = json.dumps(body).encode()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_runtime_read(orphans: list[str]) -> subprocess.CompletedProcess[str]:
    command = _check_value("dod-omn20464-ac1-runtime-health-orphan-line")
    code = _python_code_of(command)
    assert HEALTH_URL in code
    body = {
        "status": "degraded",
        "details": {
            "dispatch_deadline": {"orphans": orphans},
            "note": "no orphan here",
        },
    }
    with _health_server(body) as url:
        return subprocess.run(
            [sys.executable, "-c", code.replace(HEALTH_URL, url)],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )


def test_the_runtime_read_exits_nonzero_on_an_orphan_line_without_the_group() -> None:
    done = _run_runtime_read(
        [
            "onex.intent.platform.runtime-tick.v1 partition=0 offset=7 "
            "subscription=sub-1 age=1602s"
        ]
    )
    assert done.returncode == 1, done.stderr
    assert "orphan lines 1 without group 1" in done.stdout


def test_the_runtime_read_exits_zero_on_an_orphan_line_with_the_group() -> None:
    done = _run_runtime_read(
        [
            "onex.intent.platform.runtime-tick.v1 partition=0 offset=7 "
            "group=dev.omnibase_infra.node_x.consume.v1 subscription=sub-1 age=1602s"
        ]
    )
    assert done.returncode == 0, done.stderr
    assert "orphan lines 1 without group 0" in done.stdout


def test_the_runtime_read_sees_zero_lines_on_a_runtime_with_no_orphan() -> None:
    """Zero orphans passes by design; the contract description says so."""
    done = _run_runtime_read([])
    assert done.returncode == 0, done.stderr
    assert "orphan lines 0 without group 0" in done.stdout


# --- the contract describes each check at the scope it runs ------------------


def test_every_check_has_a_runnable_command_and_a_scope_matching_its_description() -> (
    None
):
    """A command that needs the runtime container is local-done-gate only, and says so."""
    contract = yaml.safe_load(CONTRACT.read_text())
    items = contract["dod_evidence"]
    assert items
    for item in items:
        checks = item["checks"]
        assert checks, item["id"]
        for check in checks:
            assert check["check_type"] in {"command", "test_passes"}, item["id"]
            assert str(check["check_value"]).strip(), item["id"]
        needs_container = any(
            str(check["check_value"]).startswith("docker exec") for check in checks
        )
        description = " ".join(str(item["description"]).split())
        if needs_container:
            assert item.get("execution_scope") == "local_done_gate", item["id"]
            assert description.startswith("Local done gate only"), item["id"]
            assert "hosted runner can run" not in description, item["id"]
        else:
            assert item.get("execution_scope", "hosted") == "hosted", item["id"]
            assert "Local done gate" not in description, item["id"]
    assert any(
        str(c["check_value"]).startswith("docker exec")
        and "/health" in c["check_value"]
        for item in items
        for c in item["checks"]
    ), "no check reads the running runtime's /health orphan line"
