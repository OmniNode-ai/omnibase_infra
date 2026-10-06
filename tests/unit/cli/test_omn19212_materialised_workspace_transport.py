# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A bound workspace resolves its transport through the materialised copy (OMN-19212).

The workspace tier of ``resolve_embedded_runtime_config`` (OMN-19193) read
``<root>/config/onex/runtime/runtime_config.yaml`` from the shared working
tree, which lags ``origin/main`` whenever a peer has work staged; a missing file
refused every default ``onex delegate``. The tier now reads the copy the
materialiser wrote from ``origin/main`` first, names its sha in the provenance,
labels a stale copy, falls back to the working tree, and refuses only when
neither exists -- naming both paths and the explicit offline override.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import pytest

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    MATERIALIZED_CONTRACTS_RELATIVE_PATH,
    MATERIALIZED_SIDECAR_NAME,
    SOURCE_PATH_IN_REPO,
    STALE_AFTER,
)
from omnibase_infra.runtime.service_kernel import (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH,
    resolve_embedded_runtime_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_config_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "config-owner"))


DECLARED_DEV_BROKER = "declared-dev.example:19092"
_SHA = "0123456789abcdef0123456789abcdef01234567"

_LANES = f"""
lanes:
  dev:
    broker: "{DECLARED_DEV_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  stability-test:
    broker: "stability.example:39092"
    security_protocol: PLAINTEXT
"""

_DEV = 'event_bus:\n  type: "kafka"\n  profile: "local"\n  lane: "dev"\n'
_STABILITY = (
    'event_bus:\n  type: "kafka"\n  profile: "local"\n  lane: "stability-test"\n'
)


def _write_working_tree_config(root: Path, text: str) -> Path:
    path = (
        Path(os.environ["ONEX_WORKSPACE_CONFIG_ROOT"])
        / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
        / "runtime"
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "runtime_config.yaml").write_text(text, encoding="utf-8")
    return path / "runtime_config.yaml"


def _write_materialised_copy(
    root: Path,
    text: str,
    *,
    age: timedelta = timedelta(minutes=1),
    sha: str = _SHA,
) -> Path:
    runtime = root / MATERIALIZED_CONTRACTS_RELATIVE_PATH / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "runtime_config.yaml").write_text(text, encoding="utf-8")
    (runtime / MATERIALIZED_SIDECAR_NAME).write_text(
        json.dumps(
            {
                "source_repository": str(
                    (Path(os.environ["ONEX_WORKSPACE_CONFIG_ROOT"])).resolve()
                ),
                "source_ref": "origin/main",
                "source_path": SOURCE_PATH_IN_REPO,
                "sha": sha,
                "materialized_at": (datetime.now(UTC) - age).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return runtime / "runtime_config.yaml"


def _declare_lanes(root: Path) -> None:
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_LANES, encoding="utf-8")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


class TestTheResolverReadsTheMaterialisedCopy:
    """AC3: materialised copy first, named by sha; stale labelled; tree fallback."""

    def test_a_copy_alone_answers_when_the_tree_has_no_config(
        self, tmp_path: Path
    ) -> None:
        copy_path = _write_materialised_copy(tmp_path, _DEV)
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path)
        assert config.event_bus.type is EnumEventBusType.KAFKA
        assert config.event_bus.lane == "dev"
        assert _SHA in source
        assert "origin/main" in source
        assert str(copy_path) in source
        assert "STALE" not in source

    def test_the_copy_outranks_a_working_tree_that_disagrees(
        self, tmp_path: Path
    ) -> None:
        _write_materialised_copy(tmp_path, _DEV)
        _write_working_tree_config(tmp_path, _STABILITY)
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path)
        assert config.event_bus.lane == "dev"
        assert _SHA in source

    def test_a_stale_copy_is_labelled_and_still_answers(self, tmp_path: Path) -> None:
        _write_materialised_copy(tmp_path, _DEV, age=STALE_AFTER + timedelta(hours=2))
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path)
        assert config.event_bus.lane == "dev"
        assert "STALE" in source
        assert _SHA in source

    def test_with_no_copy_the_working_tree_answers_as_before(
        self, tmp_path: Path
    ) -> None:
        tree_path = _write_working_tree_config(tmp_path, _DEV)
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path)
        assert config.event_bus.lane == "dev"
        assert "workspace tier-1" in source
        assert str(tree_path) in source
        assert "origin/main" not in source

    def test_an_unattributable_copy_falls_back_to_the_working_tree(
        self, tmp_path: Path
    ) -> None:
        copy_path = _write_materialised_copy(tmp_path, _DEV)
        (copy_path.parent / MATERIALIZED_SIDECAR_NAME).write_text(
            "{not json", encoding="utf-8"
        )
        _write_working_tree_config(tmp_path, _STABILITY)
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path)
        assert config.event_bus.lane == "stability-test"
        assert str(copy_path) not in source

    def test_the_bootstrap_pointer_still_outranks_a_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pointed = tmp_path / "pointed"
        (pointed / "runtime").mkdir(parents=True)
        (pointed / "runtime" / "runtime_config.yaml").write_text(
            'event_bus:\n  type: "inmemory"\n  profile: "local"\n', encoding="utf-8"
        )
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(pointed))
        _write_materialised_copy(tmp_path / "ws", _DEV)
        config, source = resolve_embedded_runtime_config(workspace_root=tmp_path / "ws")
        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert str(pointed) in source


class TestTheRefusal:
    """AC4: refuse only when neither exists, naming both paths and the override."""

    def test_neither_a_copy_nor_a_tree_config_names_both_paths(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ProtocolConfigurationError) as exc:
            resolve_embedded_runtime_config(workspace_root=tmp_path)
        message = str(exc.value)
        materialised = (
            tmp_path
            / MATERIALIZED_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        tree = (
            Path(os.environ["ONEX_WORKSPACE_CONFIG_ROOT"])
            / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        assert str(materialised) in message
        assert str(tree) in message
        assert "origin/main" in message
        assert "--bus inmemory" in message


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout


def _lagging_workspace(root: Path) -> str:
    """origin/main carries the tier-1 config; the shared tree's HEAD does not."""
    root.mkdir(parents=True, exist_ok=True)
    owner = Path(os.environ["ONEX_WORKSPACE_CONFIG_ROOT"])
    owner.mkdir(parents=True, exist_ok=True)
    _git(owner, "init", "-q", "-b", "main")
    (owner / ".git" / "info" / "exclude").write_text(".onex_state/\n", encoding="utf-8")
    _declare_lanes(root)
    _write_working_tree_config(root, _DEV)
    _git(owner, "add", "-A")
    _git(owner, "commit", "-q", "-m", "config lands")
    sha = _git(owner, "rev-parse", "HEAD").strip()
    _git(owner, "update-ref", "refs/remotes/origin/main", sha)
    _git(owner, "rm", "-q", SOURCE_PATH_IN_REPO)
    _git(owner, "commit", "-q", "-m", "tree lags")
    assert not (owner / SOURCE_PATH_IN_REPO).exists()
    return sha


class TestRunDelegateOnALaggingTree:
    """The lab scenario: a default run on a tree that lacks the file resolves."""

    @staticmethod
    def _capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
        home = tmp_path / "fake-home"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
        StoreLaneCredential(onex_home=home / ".onex").save(
            lane="dev",
            sasl_username="dev-cli-under-test",
            sasl_password="not-a-real-secret",
        )
        return captured

    def _run(self, tmp_path: Path, root: Path, *, bus: str | None = None) -> int:
        return run_delegate(
            prompt="document the router",
            task_type="document",
            max_tokens=None,
            bus=bus,
            # Pinned in-process: the subject is the transport and its address,
            # not the live-consumer gate, which would need a broker.
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            omni_home=root,
        )

    def test_the_default_run_materialises_then_lands_on_the_configured_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        root = tmp_path / "ws"
        sha = _lagging_workspace(root)
        assert not (root / SOURCE_PATH_IN_REPO).exists()
        assert self._run(tmp_path, root) == 0
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }
        sidecar = (
            root
            / MATERIALIZED_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / MATERIALIZED_SIDECAR_NAME
        )
        assert json.loads(sidecar.read_text(encoding="utf-8"))["sha"] == sha

    def test_the_run_log_names_the_sha_that_chose_the_transport(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._capture(tmp_path, monkeypatch)
        root = tmp_path / "ws"
        sha = _lagging_workspace(root)
        with caplog.at_level("INFO"):
            assert self._run(tmp_path, root) == 0
        assert sha in caplog.text

    def test_a_tree_with_neither_is_refused_and_dispatches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        root = tmp_path / "ws"
        _declare_lanes(root)
        with pytest.raises(click.ClickException) as exc:
            self._run(tmp_path, root)
        assert "--bus inmemory" in str(exc.value.message)
        assert str(root / MATERIALIZED_CONTRACTS_RELATIVE_PATH) in str(
            exc.value.message
        )
        assert captured == {}

    def test_an_explicit_inmemory_bus_materialises_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        root = tmp_path / "ws"
        _lagging_workspace(root)
        assert self._run(tmp_path, root, bus="inmemory") == 0
        assert captured["backend_overrides"] == {"event_bus": "inmemory"}
        assert not (root / MATERIALIZED_CONTRACTS_RELATIVE_PATH).exists()
