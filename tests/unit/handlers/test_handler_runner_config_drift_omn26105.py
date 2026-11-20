# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26105: the image roll recreates an idle runner whose mounts or declared env differ from the synced Compose.

Before this change a runner already online on the declared image was skipped, so a Compose change to binds or env
(the h053 NVMe job directories of OMN-26093) never reached a running container. The comparison must read mounts and a
declared set of non-secret env keys, never the registration token: the rendered ``RUNNER_TOKEN`` changes on every
render, so comparing the whole environment would recreate every online runner on every call (that killed a job on
2026-09-28).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runner_image_roll import (
    HandlerRunnerImageRoll,
    RunnerRollError,
)
from omnibase_infra.models.model_runner_config_drift_policy import (
    ModelRunnerConfigDriftPolicy,
)
from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)
from omnibase_infra.models.model_runner_image_roll_result import (
    ModelRunnerImageRollResult,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FLEET = ROOT / "config/runner_fleet.yaml"
HOME = "/home/runner"
WORK = f"{HOME}/actions-runner/_work"
NVME_WORK = "/mnt/h053-nvme/runners/hp053-ci-runner/1/work"


Mount = dict[str, str | bool]
Env = dict[str, str]

RENDERED_VOLUMES: list[Mount] = [
    {
        "type": "bind",
        "source": "/var/run/docker.sock",
        "target": "/var/run/docker.sock",
    },
    {
        "type": "bind",
        "source": "/usr/share/zoneinfo",
        "target": "/usr/share/zoneinfo",
        "read_only": True,
    },
    {"type": "bind", "source": NVME_WORK, "target": WORK},
    {"type": "volume", "source": "creds", "target": f"{HOME}/.runner-creds"},
]
RUNNING_MOUNTS: list[Mount] = [
    {
        "Type": "bind",
        "Source": "/var/run/docker.sock",
        "Destination": "/var/run/docker.sock",
        "RW": True,
    },
    {
        "Type": "bind",
        "Source": "/usr/share/zoneinfo",
        "Destination": "/usr/share/zoneinfo",
        "RW": False,
    },
    {"Type": "bind", "Source": NVME_WORK, "Destination": WORK, "RW": True},
    {
        "Type": "volume",
        # Compose prefixes the project to the declared volume name.
        "Name": "pool_creds",
        "Source": "/var/lib/docker/volumes/x/_data",
        "Destination": f"{HOME}/.runner-creds",
        "RW": True,
    },
]
RENDERED_ENV: Env = {
    "RUNNER_NAME": "hp053-ci-runner-1",
    "RUNNER_LABELS": "self-hosted,omnibase-ci",
    "RUNNER_TOKEN": "token-rendered-now",
    "TMPDIR": f"{HOME}/scratch",
}
RUNNING_ENV: Env = {**RENDERED_ENV, "RUNNER_TOKEN": "token-minted-last-week"}


def _drift(
    *,
    volumes: list[Mount] = RENDERED_VOLUMES,
    rendered_env: Env = RENDERED_ENV,
    mounts: list[Mount] = RUNNING_MOUNTS,
    tmpfs: dict[str, str] | None = None,
    running_env: Env = RUNNING_ENV,
) -> tuple[str, ...]:
    rendered = {"volumes": volumes, "tmpfs": [], "environment": rendered_env}
    running = {"mounts": mounts, "tmpfs": tmpfs or {}, "environment": running_env}
    return ModelRunnerConfigDriftPolicy.drift(json.dumps(rendered), json.dumps(running))


def _without(mounts: list[Mount], key: str, target: str) -> list[Mount]:
    return [mount for mount in mounts if mount[key] != target]


def _with(volumes: list[Mount], target: str, **changes: str | bool) -> list[Mount]:
    return [{**v, **changes} if v["target"] == target else v for v in volumes]


def test_runner_config_drift_identical_runner_has_none_and_token_is_ignored() -> None:
    # The 2026-09-28 failure: the token differs on every render, so a whole-env compare recreated every runner.
    assert RENDERED_ENV["RUNNER_TOKEN"] != RUNNING_ENV["RUNNER_TOKEN"]
    assert _drift() == ()


def test_runner_config_drift_tmpfs_workspace_against_nvme_bind_is_drift() -> None:
    assert _drift(
        mounts=_without(RUNNING_MOUNTS, "Destination", WORK),
        tmpfs={WORK: "rw,size=9g"},
    ) == (f"mount:{WORK}",)


def test_runner_config_drift_changed_bind_source_is_drift() -> None:
    assert _drift(volumes=_with(RENDERED_VOLUMES, WORK, source="/mnt/other/work")) == (
        f"mount:{WORK}",
    )


def test_runner_config_drift_read_only_flip_is_drift() -> None:
    zoneinfo = "/usr/share/zoneinfo"
    assert _drift(volumes=_with(RENDERED_VOLUMES, zoneinfo, read_only=False)) == (
        f"mount:{zoneinfo}",
    )


def test_runner_config_drift_mount_removed_from_compose_is_drift() -> None:
    zoneinfo = "/usr/share/zoneinfo"
    assert _drift(volumes=_without(RENDERED_VOLUMES, "target", zoneinfo)) == (
        f"mount:{zoneinfo}",
    )


def test_runner_config_drift_image_anonymous_volume_is_not_drift() -> None:
    anonymous: Mount = {
        "Type": "volume",
        "Name": "a" * 64,
        "Source": "/var/lib/docker/volumes/a/_data",
        "Destination": "/image/declared/volume",
        "RW": True,
    }
    assert _drift(mounts=[*RUNNING_MOUNTS, anonymous]) == ()


def test_runner_config_drift_named_volume_prefix_added_by_compose_is_not_drift() -> (
    None
):
    # Only the target identifies a named volume; the running name carries the Compose project prefix.
    assert _drift() == ()


def test_runner_config_drift_declared_env_value_change_is_drift() -> None:
    assert _drift(rendered_env={**RENDERED_ENV, "RUNNER_LABELS": "self-hosted"}) == (
        "env:RUNNER_LABELS",
    )


def test_runner_config_drift_secret_and_undeclared_env_never_count() -> None:
    assert (
        _drift(
            rendered_env={
                **RENDERED_ENV,
                "LOCAL_LLM_SHARED_SECRET": "new",
                "SOME_UNDECLARED_KEY": "new",
            },
            running_env={
                **RUNNING_ENV,
                "LOCAL_LLM_SHARED_SECRET": "old",
                "SOME_UNDECLARED_KEY": "old",
            },
        )
        == ()
    )


def test_runner_config_drift_declared_env_keys_never_include_a_secret() -> None:
    keys = ModelRunnerConfigDriftPolicy().env_keys
    assert keys == tuple(sorted(keys))
    assert "RUNNER_NAME" in keys
    assert not [
        key
        for key in keys
        if "TOKEN" in key or "SECRET" in key or "PASSWORD" in key or "KEY" in key
    ]


def test_runner_config_drift_unparseable_readback_is_an_error_not_no_drift() -> None:
    with pytest.raises(ValueError):
        ModelRunnerConfigDriftPolicy.drift("not json", "{}")


# ---------------------------------------------------------------- the host read


def _request(**fields: object) -> ModelRunnerImageRollRequest:
    return ModelRunnerImageRollRequest(
        host="h053",
        prefix="hp053-ci-runner",
        repo_root=ROOT,
        fleet_config=FLEET,
        token_path=Path("/nonexistent"),
        **fields,
    )


@pytest.mark.asyncio
async def test_runner_config_drift_host_read_keeps_secrets_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    scripts: list[str] = []

    async def remote(
        _host: str, script: str, *, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        scripts.append(script)
        assert stdin is None
        return 0, (
            json.dumps({"volumes": [], "tmpfs": [], "environment": {}})
            + "\n"
            + json.dumps({"mounts": [], "tmpfs": {}, "environment": {}})
        )

    monkeypatch.setattr(handler, "_remote", remote)
    drift = await handler._config_drift(
        _request(), "hp053-ci-runner-1", "container-id", ("RUNNER_TOKEN",)
    )
    assert drift == ()
    (script,) = scripts
    # Read-only on the host: a render and an inspect, nothing that writes or recreates.
    assert "docker inspect" in script and "config --format json" in script
    for write in (" up ", " stop ", " rm ", " restart ", " kill "):
        assert write not in script
    # The host filters the environment to the declared keys before anything crosses SSH.
    for key in ModelRunnerConfigDriftPolicy().env_keys:
        assert key in script
    assert "LOCAL_LLM_SHARED_SECRET" not in script


@pytest.mark.asyncio
async def test_runner_config_drift_host_read_failure_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()

    async def remote(*_args: object, **_kwargs: object) -> tuple[int, str]:
        return 1, ""

    monkeypatch.setattr(handler, "_remote", remote)
    with pytest.raises(RunnerRollError):
        await handler._config_drift(
            _request(), "hp053-ci-runner-1", "container-id", ("RUNNER_TOKEN",)
        )


# ---------------------------------------------------------------- the roll loop


class _Roll:
    """A pool on the declared image: ``drifted`` names differ from the synced Compose; ``busy`` names are not idle."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handler: HandlerRunnerImageRoll,
        *,
        drifted: frozenset[str],
        busy: frozenset[str] = frozenset(),
        still_drifted_after_recreate: bool = False,
        unreadable: frozenset[str] = frozenset(),
    ) -> None:
        self.drifted = set(drifted)
        self.busy = set(busy)
        self.unreadable = set(unreadable)
        self.sticky = still_drifted_after_recreate
        self.events: list[tuple[str, str]] = []
        self.recreated_ids: dict[str, str] = {}
        self.drift_reads: list[tuple[str, str]] = []

        async def prepare(*_a: object) -> str:
            return "sha256:declared"

        async def idle(_pool: object, name: str, _org: str) -> tuple[str, str] | None:
            self.events.append(("idle-check", name))
            if name in self.busy:
                return None
            return (self.recreated_ids.get(name, "before"), "sha256:declared")

        async def container(_pool: object, name: str) -> tuple[str, str] | None:
            return (self.recreated_ids.get(name, "before"), "sha256:declared")

        async def config_drift(
            _pool: object,
            name: str,
            container_id: str,
            _vars: tuple[str, ...],
        ) -> tuple[str, ...]:
            self.events.append(("drift-read", name))
            assert name not in self.busy, (
                "a busy runner must never be probed or touched"
            )
            if name in self.unreadable:
                raise RunnerRollError("Could not read the running configuration.")
            self.drift_reads.append((name, container_id))
            if container_id == "before" and name in self.drifted:
                return ("mount:/work",)
            if name in self.drifted and self.sticky:
                return ("mount:/work",)
            return ()

        async def recreate(
            _pool: object,
            name: str,
            previous_id: str,
            token: bytes,
            tag: str,
            _vars: tuple[str, ...],
        ) -> bool:
            assert previous_id == "before"
            assert name not in self.busy
            self.events.append(("recreate", name))
            self.recreated_ids[name] = "after"
            return True

        async def online(
            _pool: object, name: str, previous_id: str, image: str, _org: str
        ) -> bool:
            self.events.append(("readback", name))
            return True

        async def retry() -> None:
            self.events.append(("retry", ""))

        monkeypatch.setattr(handler, "_prepare_image", prepare)
        monkeypatch.setattr(handler, "_idle_container", idle)
        monkeypatch.setattr(handler, "_container", container)
        monkeypatch.setattr(handler, "_config_drift", config_drift, raising=False)
        monkeypatch.setattr(handler, "_recreate", recreate)
        monkeypatch.setattr(handler, "_wait_online", online)
        monkeypatch.setattr(handler, "_retry_pause", retry)

    def names(self, kind: str) -> list[str]:
        return [name for event, name in self.events if event == kind]


async def _roll(
    handler: HandlerRunnerImageRoll, limit: int = 0, count: int = 3
) -> ModelRunnerImageRollResult:
    return await handler._roll(
        _request(limit=limit),
        targets=tuple(f"hp053-ci-runner-{i}" for i in range(1, count + 1)),
        image_tag="test-image",
        token=b"fixture-token",
        org="Fixture",
        token_vars=("RUNNER_TOKEN",),
    )


@pytest.mark.asyncio
async def test_runner_config_drift_idle_drifted_runner_on_declared_image_is_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    roll = _Roll(monkeypatch, handler, drifted=frozenset({"hp053-ci-runner-2"}))
    result = await _roll(handler)
    assert result.success is True
    assert result.recreated == ("hp053-ci-runner-2",)
    assert roll.names("recreate") == ["hp053-ci-runner-2"]
    assert roll.names("readback") == ["hp053-ci-runner-2"]
    # The replacement is read back against the Compose too, on its own container id.
    assert ("hp053-ci-runner-2", "after") in roll.drift_reads


@pytest.mark.asyncio
async def test_runner_config_drift_undrifted_runner_on_declared_image_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    roll = _Roll(monkeypatch, handler, drifted=frozenset())
    result = await _roll(handler)
    assert result.success is True
    assert result.recreated == ()
    assert roll.names("recreate") == []


@pytest.mark.asyncio
async def test_runner_config_drift_busy_runner_is_never_probed_or_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    roll = _Roll(
        monkeypatch,
        handler,
        drifted=frozenset({"hp053-ci-runner-1", "hp053-ci-runner-2"}),
        busy=frozenset({"hp053-ci-runner-1"}),
    )
    result = await _roll(handler, count=2)
    assert roll.names("recreate") == ["hp053-ci-runner-2"]
    assert "hp053-ci-runner-1" not in roll.names("drift-read")
    assert result.success is False
    assert "hp053-ci-runner-1" in (result.error_message or "")


@pytest.mark.asyncio
async def test_runner_config_drift_recreates_one_runner_at_a_time_and_honours_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    names = frozenset({"hp053-ci-runner-1", "hp053-ci-runner-2", "hp053-ci-runner-3"})
    roll = _Roll(monkeypatch, handler, drifted=names)
    result = await _roll(handler, limit=2)
    assert result.recreated == ("hp053-ci-runner-1", "hp053-ci-runner-2")
    recreate_events = [e for e in roll.events if e[0] in {"recreate", "readback"}]
    assert recreate_events == [
        ("recreate", "hp053-ci-runner-1"),
        ("readback", "hp053-ci-runner-1"),
        ("recreate", "hp053-ci-runner-2"),
        ("readback", "hp053-ci-runner-2"),
    ]


@pytest.mark.asyncio
async def test_runner_config_drift_still_drifted_after_recreate_halts_the_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A comparison that can never converge must stop at the canary, not walk the whole pool.
    handler = HandlerRunnerImageRoll()
    roll = _Roll(
        monkeypatch,
        handler,
        drifted=frozenset({"hp053-ci-runner-1", "hp053-ci-runner-2"}),
        still_drifted_after_recreate=True,
    )
    result = await _roll(handler)
    assert result.success is False
    assert "HALT" in (result.error_message or "")
    assert roll.names("recreate") == ["hp053-ci-runner-1"]


@pytest.mark.asyncio
async def test_runner_config_drift_unreadable_configuration_recreates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = HandlerRunnerImageRoll()
    roll = _Roll(
        monkeypatch,
        handler,
        drifted=frozenset({"hp053-ci-runner-1"}),
        unreadable=frozenset({"hp053-ci-runner-1"}),
    )
    result = await _roll(handler)
    assert result.success is False
    assert roll.names("recreate") == []
