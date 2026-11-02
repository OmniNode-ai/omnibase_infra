# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the boot bound, join limit, join priority and memory limit are contract-owned.

* ``runtime_policy.contract.yaml`` declares main's boot bound, join limit, join
  priority and memory limit, and the renderer writes them into the policy env.
* No compose file carries a literal for them: main's memory limit and the join
  limit are fail-fast references, and no ``1536M`` stays on main.
* The kernel reads the rendered values, and past the bound it names the required
  topic that holds no partition, without lowering the readiness gate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers import boot_readiness_checks
from omnibase_infra.handlers.boot_readiness_checks import (
    parse_priority_contracts,
    parse_ready_bound_seconds,
    required_topics_without_partitions,
    start_ready_bound_watch,
    watch_ready_bound,
)
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.runtime.service_kernel import resolve_topic_readiness_config

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = ROOT / "docker" / "runtime-policy.env"
COMPOSE_DIR = ROOT / "docker"
INFRA_COMPOSE = COMPOSE_DIR / "docker-compose.infra.yml"

pytestmark = pytest.mark.unit

sys.path.insert(0, str(ROOT / "scripts"))
import render_runtime_policy_env as renderer


def _contract() -> ModelRuntimePolicyContract:
    return ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )


def _compose(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8").replace("!override", ""))
    assert isinstance(loaded, dict)
    return loaded


def test_dev_main_declares_the_boot_policy_and_memory_limit() -> None:
    boot = _contract().profiles["dev"].processes["main"].boot

    assert boot is not None
    assert boot.ready_bound_seconds >= 1
    assert boot.max_concurrent_contract_attach >= 1
    assert "node_delegation_orchestrator" in boot.priority_contracts
    assert "projection_claude_hook_events" in boot.priority_contracts


def test_dev_main_memory_limit_fits_the_four_kernel_figure() -> None:
    """2,367 MiB is the one-runtime plan's four-kernel figure; the limit must exceed it."""
    boot = _contract().profiles["dev"].processes["main"].boot
    assert boot is not None
    unit = 1024 if boot.memory_limit.endswith("G") else 1

    assert int(boot.memory_limit[:-1]) * unit > 2367


def test_the_renderer_writes_main_policy_and_the_committed_env_matches() -> None:
    contract = _contract()
    rendered = renderer.render_env(contract)
    boot = contract.profiles["dev"].processes["main"].boot
    assert boot is not None

    assert rendered["DEV_RUNTIME_MAIN_READY_BOUND_SECONDS"] == str(
        boot.ready_bound_seconds
    )
    assert rendered["DEV_RUNTIME_MAIN_MAX_CONCURRENT_CONTRACT_ATTACH"] == str(
        boot.max_concurrent_contract_attach
    )
    assert rendered["DEV_RUNTIME_MAIN_BOOT_PRIORITY_CONTRACTS"] == ",".join(
        boot.priority_contracts
    )
    assert rendered["DEV_RUNTIME_MAIN_MEMORY_LIMIT"] == boot.memory_limit
    assert POLICY_ENV_PATH.read_text(encoding="utf-8") == renderer.format_dotenv(
        rendered
    )


def test_a_process_that_declares_no_boot_policy_renders_none() -> None:
    rendered = renderer.render_env(_contract())

    assert not [
        key
        for key in rendered
        if key.startswith("DEV_RUNTIME_EFFECTS_")
        and key.endswith(("_READY_BOUND_SECONDS", "_MEMORY_LIMIT"))
    ]


def test_main_has_no_memory_literal_in_any_compose_file() -> None:
    main = _compose(INFRA_COMPOSE)["services"]["omninode-runtime"]
    memory = main["deploy"]["resources"]["limits"]["memory"]

    assert memory.startswith("${DEV_RUNTIME_MAIN_MEMORY_LIMIT:?")
    assert "1536M" not in str(main)


def test_no_compose_file_pins_the_join_limit_with_a_literal() -> None:
    offenders: list[str] = []
    for path in sorted(COMPOSE_DIR.glob("docker-compose*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"\s*-?\s*ONEX_MAX_CONCURRENT_CONTRACT_ATTACH[:=]\s*(\S+)", line
            )
            if match and not match.group(1).strip("\"'").startswith("${"):
                offenders.append(f"{path.name}: {line.strip()}")

    assert not offenders, offenders
    main_env = _compose(INFRA_COMPOSE)["services"]["omninode-runtime"]["environment"]
    assert main_env["ONEX_MAX_CONCURRENT_CONTRACT_ATTACH"].startswith(
        "${DEV_RUNTIME_MAIN_MAX_CONCURRENT_CONTRACT_ATTACH:?"
    )
    assert main_env["ONEX_BOOT_READY_BOUND_SECONDS"].startswith(
        "${DEV_RUNTIME_MAIN_READY_BOUND_SECONDS:?"
    )
    assert main_env["ONEX_BOOT_PRIORITY_CONTRACTS"].startswith(
        "${DEV_RUNTIME_MAIN_BOOT_PRIORITY_CONTRACTS:?"
    )


def test_the_kernel_reads_the_rendered_join_limit_and_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_MAX_CONCURRENT_CONTRACT_ATTACH", "7")
    monkeypatch.setenv("ONEX_BOOT_PRIORITY_CONTRACTS", "node_a, node_b,node_a,")

    config = resolve_topic_readiness_config()

    assert config.max_concurrent_contract_attach == 7
    assert config.priority_contracts == ("node_a", "node_b")


def test_the_kernel_declares_no_priority_without_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONEX_BOOT_PRIORITY_CONTRACTS", raising=False)

    assert resolve_topic_readiness_config().priority_contracts == ()
    assert parse_priority_contracts({}) == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600", 600.0),
        ("", None),
        ("abc", None),
        ("0", None),
        ("-5", None),
        ("inf", None),
    ],
)
def test_the_boot_bound_is_read_from_the_rendered_env(
    raw: str, expected: float | None
) -> None:
    assert parse_ready_bound_seconds({"ONEX_BOOT_READY_BOUND_SECONDS": raw}) == expected


def test_no_bound_in_the_env_declares_no_bound() -> None:
    assert parse_ready_bound_seconds({}) is None


UNJOINABLE = "test-unjoinable"
JOINED = "test-joined"


def _readiness(*, unjoinable_assignment: list[int]) -> dict[str, object]:
    """A /ready body with one joined and one planted required topic."""
    event_bus_ok = bool(unjoinable_assignment)
    return {
        "ready": event_bus_ok,
        "is_running": True,
        "event_bus_readiness": {
            "is_ready": event_bus_ok,
            "required_topics": [JOINED, UNJOINABLE],
            "assignments": {JOINED: [0], UNJOINABLE: unjoinable_assignment},
        },
    }


class _FakeRuntime:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body
        self.calls = 0

    async def readiness_check(self) -> dict[str, object]:
        self.calls += 1
        return self.body


def test_required_topics_without_partitions_names_empty_and_absent_topics() -> None:
    body = _readiness(unjoinable_assignment=[])["event_bus_readiness"]
    assert isinstance(body, dict)

    assert required_topics_without_partitions(body) == (UNJOINABLE,)
    body["assignments"] = {JOINED: [0]}
    assert required_topics_without_partitions(body) == (UNJOINABLE,)


def test_the_required_set_is_not_narrowed_to_make_the_check_pass() -> None:
    """A required topic with no partition is named; none is dropped from the set."""
    body = _readiness(unjoinable_assignment=[])["event_bus_readiness"]
    assert isinstance(body, dict)
    required = body["required_topics"]
    assert isinstance(required, list)

    assert set(required) == {JOINED, UNJOINABLE}
    assert UNJOINABLE in required_topics_without_partitions(body)


@pytest.mark.asyncio
async def test_a_planted_unjoinable_required_topic_exceeds_the_bound_and_is_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _FakeRuntime(_readiness(unjoinable_assignment=[]))

    with caplog.at_level(logging.ERROR):
        within = await watch_ready_bound(
            runtime.readiness_check,
            bound_seconds=0.05,
            started_monotonic=time.monotonic(),
            poll_seconds=0.01,
        )

    assert within is False
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Boot bound exceeded" in m and UNJOINABLE in m for m in errors)
    assert runtime.calls >= 2


@pytest.mark.asyncio
async def test_a_runtime_that_is_ready_inside_the_bound_reports_within_bound() -> None:
    runtime = _FakeRuntime(_readiness(unjoinable_assignment=[0]))

    within = await watch_ready_bound(
        runtime.readiness_check,
        bound_seconds=60.0,
        started_monotonic=time.monotonic(),
        poll_seconds=0.01,
    )

    assert within is True
    assert runtime.calls == 1


@pytest.mark.asyncio
async def test_the_watch_starts_only_when_the_policy_declares_a_bound() -> None:
    runtime = _FakeRuntime(_readiness(unjoinable_assignment=[0]))
    stops: list[Callable[[], Awaitable[None]]] = []

    start_ready_bound_watch(runtime.readiness_check, {}, stops)
    assert stops == []

    start_ready_bound_watch(
        runtime.readiness_check, {"ONEX_BOOT_READY_BOUND_SECONDS": "600"}, stops
    )
    assert len(stops) == 1
    await stops[0]()


@pytest.mark.asyncio
async def test_readiness_probe_exception_is_logged_and_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def probe() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("dependency still starting")
        return _readiness(unjoinable_assignment=[0])

    with caplog.at_level(logging.INFO):
        within = await watch_ready_bound(
            probe,
            bound_seconds=60,
            started_monotonic=time.monotonic(),
            poll_seconds=0.001,
        )

    assert within is True
    assert calls == 2
    assert "Boot readiness probe failed; retrying" in caplog.text
    assert "within_bound=True" in caplog.text


@pytest.mark.asyncio
async def test_persistent_probe_failure_still_reports_the_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def probe() -> dict[str, object]:
        raise RuntimeError("dependency unavailable")

    with caplog.at_level(logging.ERROR):
        within = await watch_ready_bound(
            probe,
            bound_seconds=1,
            started_monotonic=time.monotonic() - 2,
        )

    assert within is False
    assert "Boot bound exceeded" in caplog.text
    assert "readiness_error=dependency unavailable" in caplog.text


@pytest.mark.asyncio
async def test_late_readiness_logs_once_for_the_bound_and_keeps_the_missed_verdict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def probe() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _readiness(unjoinable_assignment=[0] if calls == 3 else [])

    with caplog.at_level(logging.INFO):
        within = await asyncio.wait_for(
            watch_ready_bound(
                probe,
                bound_seconds=1,
                started_monotonic=time.monotonic() - 2,
                poll_seconds=0.001,
                observe_late_ready=True,
            ),
            timeout=1,
        )

    assert within is False
    assert calls == 3
    assert caplog.text.count("Boot bound exceeded") == 1
    assert "within_bound=False" in caplog.text
    assert UNJOINABLE in caplog.text


@pytest.mark.asyncio
async def test_watch_task_failure_is_observed_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def failed_watch(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("watch failed")

    monkeypatch.setattr(boot_readiness_checks, "watch_ready_bound", failed_watch)
    runtime = _FakeRuntime(_readiness(unjoinable_assignment=[]))
    stops: list[Callable[[], Awaitable[None]]] = []
    with caplog.at_level(logging.ERROR):
        start_ready_bound_watch(
            runtime.readiness_check, {"ONEX_BOOT_READY_BOUND_SECONDS": "600"}, stops
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert "Boot readiness watch failed" in caplog.text
        assert "watch failed" in caplog.text
        await stops[0]()


@pytest.mark.asyncio
async def test_stopping_the_watch_cancels_a_pending_readiness_probe() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def probe() -> dict[str, object]:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return {}

    stops: list[Callable[[], Awaitable[None]]] = []
    start_ready_bound_watch(probe, {"ONEX_BOOT_READY_BOUND_SECONDS": "600"}, stops)
    await asyncio.wait_for(entered.wait(), timeout=1)
    await stops[0]()
    assert cancelled.is_set()
