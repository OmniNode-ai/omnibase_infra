# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed dev roll keeps the failed container's full log on its incident (OMN-24013).

The guard reads the failed container's complete log, keeps the last
``log_byte_cap`` bytes, records the byte counts and sha256 on the incident, and
hands the incident to the injected events object with the issuer of the roll's command.
Events that raise never stop recovery, and a guard prepared without a command publishes
nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from deploy_agent import executor as executor_module

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from omnibase_infra.models.model_dev_boot_report_config import ModelDevBootReportConfig
from omnibase_infra.models.model_dev_boot_request import ModelDevBootRequest

pytestmark = pytest.mark.integration

EXECUTOR_FILE = (
    Path(__file__).resolve().parents[3]
    / "scripts/deploy-agent/deploy_agent/executor.py"
)

PROJECT = "omnibase-infra"
OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40
PROBE_OUTPUT = "FAIL [http_error] health endpoint returned HTTP 503"
ROLL_FAILURE = (
    "Runtime roll stopped at omninode-runtime: container omninode-runtime is unhealthy"
)


class _Lane(Enum):
    DEV = "dev"


@contextlib.contextmanager
def _probe_file_args(lane: object, service: str) -> Iterator[list[str]]:
    yield ["-f", "compose.yml"]


def _container(service: str, revision: str) -> dict[str, Any]:
    return {
        "Id": f"id-{revision[:1]}-{service}",
        "Image": f"sha256:{revision[:1]}-{service}",
        "Config": {
            "Labels": {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.service": service,
                "com.docker.compose.config-hash": f"hash-{service}",
                "org.opencontainers.image.revision": revision,
            },
            "Healthcheck": {
                "StartPeriod": 1_800_000_000_000,
                "Retries": 5,
                "Interval": 30_000_000_000,
                "Timeout": 10_000_000_000,
            },
        },
        "State": {"Status": "running", "Health": {"Status": "healthy"}},
    }


def _family(revision: str) -> list[dict[str, Any]]:
    return [_container(s, revision) for s in guard_mod.DEV_BOOT_SERVICES]


def _unhealthy_runtime_family() -> list[dict[str, Any]]:
    family = _family(NEW_REVISION)
    runtime = next(
        c
        for c in family
        if c["Config"]["Labels"]["com.docker.compose.service"] == "omninode-runtime"
    )
    runtime["RestartCount"] = 0
    runtime["State"] = {
        "Status": "running",
        "StartedAt": "2026-11-10T04:07:46Z",
        "ExitCode": 0,
        "OOMKilled": False,
        "Health": {
            "Status": "unhealthy",
            "FailingStreak": 5,
            "Log": [
                {
                    "Start": f"2026-11-10T04:{minute}Z",
                    "End": f"2026-11-10T04:{minute}Z",
                    "ExitCode": 1,
                    "Output": f"{PROBE_OUTPUT}\n",
                }
                for minute in ("08:07", "08:37", "09:07", "09:37", "10:07")
            ],
        },
    }
    return family


@dataclass
class _Host:
    containers: list[dict[str, Any]] = field(
        default_factory=lambda: _family(OLD_REVISION)
    )
    written: dict[Path, Any] = field(default_factory=dict)
    logs_fail: bool = False
    log_text: str = ""

    def run(
        self, argv: list[str], timeout: int, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        verb = argv[1]
        if verb == "ps":
            ids = "\n".join(c["Id"] for c in self.containers)
            return subprocess.CompletedProcess(argv, 0, ids, "")
        if verb == "inspect":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.containers), "")
        if verb == "logs":
            return subprocess.CompletedProcess(
                argv, 1 if self.logs_fail else 0, self.log_text, ""
            )
        if "config" in argv:
            services = {s: {"image": f"img-{s}"} for s in guard_mod.DEV_BOOT_SERVICES}
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"services": services}), ""
            )
        if "up" in argv:
            self.containers = _family(OLD_REVISION)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected host command: {argv}")

    def execution(self) -> ModuleType:
        module = ModuleType("fake_deploy_execution")
        module.__dict__.update(
            EnumRuntimeLane=_Lane,
            subprocess=subprocess,
            _dev_boot_clock=lambda: 0.0,
            _dev_boot_timestamp=lambda: datetime(2026, 11, 10, tzinfo=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            _read_private_text=lambda path: path.read_text(encoding="utf-8"),
            lane_config_for=lambda lane: SimpleNamespace(
                compose_project=PROJECT, disabled_services=()
            ),
            lane_folded_profiles=lambda lane: frozenset(),
            _run=self.run,
            _compose_file_args=lambda lane: ["-f", "compose.yml"],
            candidate_probe_file_args=_probe_file_args,
            _compose_env=dict,
            IMAGE_PREFLIGHT_SCRIPT="pass",
            CONFIG_HASH_TIMEOUT_SECONDS=30,
            VERIFY_RECREATE_TIMEOUT_SECONDS=30,
            _tag_image=lambda image, tag: None,
            _atomic_write_private_json=self.written.__setitem__,
            __file__=str(EXECUTOR_FILE),
            load_dev_boot_guard=executor_module.load_dev_boot_guard,
        )
        return module


COMMAND: dict[str, Any] = {
    "correlation_id": "6f1c0e0a-0000-4000-8000-000000000135",
    "requested_by": "gha/omnimarket/pr-5942",
    "git_ref": NEW_REVISION,
    "scope": "full",
    "requested_at": "2026-11-10T04:06:00Z",
    "open_pr": "",
    "rollback": None,
}


@dataclass
class _Events:
    """Records what the guard announces, in order."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    raises: bool = False

    def _note(self, kind: str, value: Any) -> None:
        self.calls.append((kind, value))
        if self.raises:
            raise RuntimeError("events down")

    def manifest(self, command: Any) -> None:
        self._note("manifest", dict(command))

    def started(self, job_id: str, started_at: datetime, cause: Any) -> None:
        self._note("started", (job_id, started_at, str(cause)))

    def incident(self, guard_incident: Any, *, issuer: str) -> None:
        self._note("incident", (dict(guard_incident), issuer))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


def _guard(
    host: _Host,
    tmp_path: Path,
    *,
    log_byte_cap: int | None = None,
    events: Any = None,
    command: dict[str, Any] | None = None,
) -> guard_mod.HandlerDevBootGuard:
    guard = guard_mod.HandlerDevBootGuard(
        tmp_path,
        config_hashes=lambda services: {s: f"hash-{s}" for s in services},
        verify_recovery=lambda update: None,
        execution=host.execution(),
        reporting=ModelDevBootReportConfig(
            registry_root=tmp_path, plugin_dir=None, ledger_via_bus=False
        ),
        log_byte_cap=log_byte_cap,
    )
    guard.events = events
    if command is None:
        guard.handle(ModelDevBootRequest(operation="prepare"))
    else:
        guard.prepare(command)
    return guard


def _fail_roll(
    host: _Host,
    tmp_path: Path,
    *,
    log_byte_cap: int | None = None,
    events: Any = None,
    command: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    guard = _guard(
        host, tmp_path, log_byte_cap=log_byte_cap, events=events, command=command
    )
    failed: list[str] = []
    with pytest.raises(guard_mod.DevBootRecoveryError, match=r"HTTP 503"):
        with guard.protect("job-0135046e", NEW_REVISION, failed.append):
            host.containers = _unhealthy_runtime_family()
            raise RuntimeError(ROLL_FAILURE)
    incidents = [v for k, v in host.written.items() if k.name.startswith("incident-")]
    assert incidents, "the guard must persist an incident record"
    return incidents[-1], failed


def _capture(incident: dict[str, Any]) -> dict[str, Any]:
    captures = incident["log_captures"]
    assert len(captures) == 1
    record: dict[str, Any] = captures[0]
    return record


def test_full_log_capture_records_bytes_sha_and_artifact(tmp_path: Path) -> None:
    log_text = "line\n" * 500
    host = _Host(log_text=log_text)
    events = _Events()
    incident, _ = _fail_roll(
        host, tmp_path, log_byte_cap=1024, events=events, command=COMMAND
    )

    record = _capture(incident)
    assert record["service"] == "omninode-runtime"
    encoded = log_text.encode("utf-8")
    assert record["bytes_total"] == len(encoded)
    assert record["bytes_kept"] == 1024
    assert record["complete"] is False
    kept_bytes = encoded[-1024:]
    assert record["sha256"] == hashlib.sha256(kept_bytes).hexdigest()
    kept_text = kept_bytes.decode("utf-8", errors="replace")
    artifact = Path(record["artifact_path"])
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == kept_text
    sunk, issuer = next(v for k, v in events.calls if k == "incident")
    assert issuer == COMMAND["requested_by"]
    assert sunk["job"] == incident["job"]
    assert sunk["log_captures"][0]["log"] == kept_text
    assert "log" not in record, (
        "the incident file must not hold a second copy of the log"
    )


def test_small_log_is_complete(tmp_path: Path) -> None:
    log_text = "a\n" * 10
    host = _Host(log_text=log_text)
    incident, _ = _fail_roll(host, tmp_path, log_byte_cap=1024)

    record = _capture(incident)
    assert record["complete"] is True
    assert record["bytes_kept"] == record["bytes_total"]


def test_a_roll_prepared_without_a_command_keeps_no_log(tmp_path: Path) -> None:
    host = _Host(log_text="x\n" * 10)
    incident, _ = _fail_roll(host, tmp_path)

    assert "log_captures" not in incident


def test_events_that_raise_still_recover_and_write(tmp_path: Path) -> None:
    host = _Host(log_text="x\n" * 10)
    incident, _ = _fail_roll(
        host,
        tmp_path,
        log_byte_cap=1024,
        events=_Events(raises=True),
        command=COMMAND,
    )

    assert incident["recovery_verified"] is True


def test_unwritable_artifact_keeps_record_with_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def raise_write_text(self: Path, *args: Any, **kwargs: Any) -> None:
        raise OSError("no write")

    monkeypatch.setattr(Path, "write_text", raise_write_text)
    host = _Host(log_text="x\n" * 10)
    incident, _ = _fail_roll(host, tmp_path, log_byte_cap=1024)

    record = _capture(incident)
    assert record["error"]
    assert record["complete"] is True
    assert record["bytes_kept"] == record["bytes_total"]


def test_runtime_started_at_reports_the_runtime_start(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)
    host.containers = _unhealthy_runtime_family()

    assert guard_mod.runtime_started_at(guard) == "2026-11-10T04:07:46Z"


def test_runtime_started_at_is_none_when_no_containers(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)
    host.containers = []

    assert guard_mod.runtime_started_at(guard) is None


def test_a_log_holding_a_credential_is_redacted_and_never_reads_complete(
    tmp_path: Path,
) -> None:
    token = "sk-" + "a" * 24
    log_text = (
        "runtime boot sequence started\n"
        f"attach upstream with {token}\n"
        "runtime boot sequence halted on health probe failure\n"
    )
    host = _Host(log_text=log_text)
    incident, _ = _fail_roll(host, tmp_path, log_byte_cap=1024)

    record = _capture(incident)
    kept = Path(record["artifact_path"]).read_text(encoding="utf-8")
    assert "runtime boot sequence started" in kept
    assert "runtime boot sequence halted on health probe failure" in kept
    assert token not in kept
    assert record["redacted"] is True
    assert record["complete"] is False
    assert record["bytes_kept"] > 39


def test_a_prose_log_without_credential_shapes_is_kept_whole(tmp_path: Path) -> None:
    """The words alone must not cost the log: this is the text the whole-text
    sanitizer turned into ``[REDACTED - potentially sensitive data]``."""
    log_text = (
        "the auth token expired before the probe ran\n"
        "the password was rejected by the database\n"
    )
    host = _Host(log_text=log_text)
    incident, _ = _fail_roll(host, tmp_path, log_byte_cap=1024)

    record = _capture(incident)
    assert record["redacted"] is False
    assert record["complete"] is True
    assert Path(record["artifact_path"]).read_text(encoding="utf-8") == log_text


def test_an_unwritable_artifact_still_redacts_the_kept_log(
    tmp_path: Path, monkeypatch: Any
) -> None:
    token = "sk-" + "a" * 24
    log_text = f"boot line one\nattach upstream with {token}\nboot line three\n"

    def raise_write_text(self: Path, *args: Any, **kwargs: Any) -> None:
        raise OSError("no write")

    monkeypatch.setattr(Path, "write_text", raise_write_text)
    host = _Host(log_text=log_text)
    incident, _ = _fail_roll(host, tmp_path, log_byte_cap=1024)

    record = _capture(incident)
    assert record["error"]
    assert token not in json.dumps(incident)


# The roll events (manifest, start, incident) announced by the guard.
INSIDE_THE_ROLL = "2999-01-01T00:00:00Z"
BEFORE_THE_ROLL = "2000-01-01T00:00:00Z"


def _runtime(containers: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        c
        for c in containers
        if c["Config"]["Labels"]["com.docker.compose.service"] == "omninode-runtime"
    )


def _failing_family(started_at: str) -> list[dict[str, Any]]:
    family = _unhealthy_runtime_family()
    _runtime(family)["State"]["StartedAt"] = started_at
    return family


def _roll(
    host: _Host,
    tmp_path: Path,
    events: _Events,
    *,
    family: list[dict[str, Any]],
    fail: bool,
) -> None:
    guard = _guard(host, tmp_path, events=events, command=COMMAND)
    host.containers = family
    if fail:
        with pytest.raises(guard_mod.DevBootRecoveryError):
            with guard.protect("job-1", NEW_REVISION, lambda _: None):
                raise RuntimeError(ROLL_FAILURE)
        return
    with guard.protect("job-1", NEW_REVISION, lambda _: None):
        pass


def test_a_failed_roll_announces_manifest_then_start_then_incident(
    tmp_path: Path,
) -> None:
    events = _Events()
    _roll(_Host(), tmp_path, events, family=_failing_family(INSIDE_THE_ROLL), fail=True)

    assert events.kinds() == ["manifest", "started", "incident"]
    assert events.calls[0][1]["requested_by"] == COMMAND["requested_by"]
    job_id, _started_at, cause = events.calls[1][1]
    assert job_id == "job-1"
    assert cause.endswith("roll_recreate")
    assert events.calls[2][1][1] == COMMAND["requested_by"]


def test_a_start_older_than_the_roll_is_the_previous_container(
    tmp_path: Path,
) -> None:
    events = _Events()
    _roll(_Host(), tmp_path, events, family=_failing_family(BEFORE_THE_ROLL), fail=True)

    assert events.kinds() == ["manifest", "incident"]


def test_a_passing_roll_announces_manifest_and_one_start(tmp_path: Path) -> None:
    host = _Host()
    family = _family(NEW_REVISION)
    _runtime(family)["State"]["StartedAt"] = INSIDE_THE_ROLL
    events = _Events()
    _roll(host, tmp_path, events, family=family, fail=False)

    assert events.kinds() == ["manifest", "started"]


def test_a_roll_prepared_without_a_command_announces_nothing(tmp_path: Path) -> None:
    events = _Events()
    host = _Host()
    guard = _guard(host, tmp_path, events=events)
    host.containers = _failing_family(INSIDE_THE_ROLL)
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("job-1", NEW_REVISION, lambda _: None):
            raise RuntimeError(ROLL_FAILURE)

    assert events.calls == []


def test_preparing_with_a_command_turns_on_the_real_events_and_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events_module = guard_mod._roll_events_module(_Host().execution())
    sent: list[str] = []
    monkeypatch.setattr(
        events_module,
        "agent_send",
        lambda: lambda topic, key, payload: sent.append(topic) or True,
    )
    guard = _guard(_Host(), tmp_path, command=COMMAND)

    assert guard.log_byte_cap == events_module.log_byte_cap(_Host().execution())
    assert sent == ["onex.evt.omnimarket.dev-roll-manifest.v1"]


def test_the_byte_cap_is_the_one_the_node_contract_declares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events_module = guard_mod._roll_events_module(_Host().execution())
    monkeypatch.setattr(events_module, "agent_send", lambda: lambda *_: True)
    guard = _guard(_Host(), tmp_path, command=COMMAND)

    assert guard.log_byte_cap == events_module.log_byte_cap(_Host().execution()) > 0
