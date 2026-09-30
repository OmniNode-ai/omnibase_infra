# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A developer's lane binding is a tier of the one transport authority (OMN-19973).

``developer.lane_binding`` in ``~/.onex/config.yaml`` makes a default
``onex delegate`` dispatch to that lab lane, so an underpowered laptop uses the
lab without a shell alias. It ranks below the ``ONEX_CONTRACTS_DIR`` bootstrap
pointer and above the workspace tier-1 config, and an explicit ``--bus`` still
outranks everything. stderr and the run files name which authority chose the
transport, so a profile-bound run is never read as one whose flags chose it.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import resolve_default_bus, run_delegate
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.store_developer_profile import StoreDeveloperProfile
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime.service_kernel import (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH,
    resolve_embedded_runtime_config,
)

pytestmark = pytest.mark.unit

DEV_BROKER = "declared-dev.example:19092"
STABILITY_BROKER = "stability.example:39092"

_LANES = f"""
lanes:
  dev:
    broker: "{DEV_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  stability-test:
    broker: "{STABILITY_BROKER}"
    security_protocol: PLAINTEXT
"""


def _workspace(root: Path, *, tier1_lane: str | None = None) -> Path:
    """A workspace root with a lane declaration and, optionally, a tier-1 config."""
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_LANES, encoding="utf-8")
    if tier1_lane is not None:
        config = (
            root
            / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            yaml.safe_dump(
                {"event_bus": {"type": "kafka", "profile": "local", "lane": tier1_lane}}
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


class TestTheBindingTier:
    """Where the binding ranks inside ``resolve_embedded_runtime_config``."""

    def test_a_binding_answers_with_kafka_on_that_lane(self) -> None:
        config, source = resolve_embedded_runtime_config(developer_lane_binding="dev")
        assert config.event_bus.type is EnumEventBusType.KAFKA
        assert config.event_bus.lane == "dev"
        assert "developer lane binding 'dev'" in source

    def test_a_binding_outranks_a_bound_workspace_with_no_tier1_config(
        self, tmp_path: Path
    ) -> None:
        # Without the binding this workspace is refused (OMN-19193); with it,
        # the developer's own choice answers.
        config, _ = resolve_embedded_runtime_config(
            workspace_root=_workspace(tmp_path), developer_lane_binding="dev"
        )
        assert config.event_bus.lane == "dev"

    def test_a_binding_outranks_the_workspace_tier1_lane(self, tmp_path: Path) -> None:
        config, source = resolve_embedded_runtime_config(
            workspace_root=_workspace(tmp_path, tier1_lane="dev"),
            developer_lane_binding="stability-test",
        )
        assert config.event_bus.lane == "stability-test"
        assert "developer lane binding" in source

    def test_the_bootstrap_pointer_outranks_the_binding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pointed = tmp_path / "pointed"
        (pointed / "runtime").mkdir(parents=True)
        (pointed / "runtime" / "runtime_config.yaml").write_text(
            'event_bus:\n  type: "inmemory"\n  profile: "local"\n', encoding="utf-8"
        )
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(pointed))
        config, source = resolve_embedded_runtime_config(developer_lane_binding="dev")
        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert str(pointed) in source

    def test_resolve_default_bus_carries_the_bound_lane(self) -> None:
        resolved = resolve_default_bus(developer_lane_binding="dev")
        assert resolved.bus == "kafka"
        assert resolved.lane == "dev"
        assert "developer lane binding 'dev'" in resolved.reason


class TestRunDelegateWithABinding:
    """End to end through ``run_delegate`` with no ``--bus``."""

    @staticmethod
    def _capture(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, binding: object
    ) -> dict[str, object]:
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
        monkeypatch.setattr(
            cli_delegate, "StoreDeveloperProfile", StoreDeveloperProfile
        )
        onex_home = home / ".onex"
        StoreLaneCredential(onex_home=onex_home).save(
            lane="dev",
            sasl_username="dev-cli-under-test",
            sasl_password="not-a-real-secret",
        )
        document = yaml.safe_load((onex_home / "config.yaml").read_text())
        document["developer"] = {"lane_binding": binding}
        (onex_home / "config.yaml").write_text(yaml.safe_dump(document))
        return captured

    @staticmethod
    def _run(tmp_path: Path, root: Path | None, **overrides: object) -> int:
        kwargs: dict[str, object] = {
            "prompt": "document the router",
            "task_type": "document",
            "max_tokens": None,
            # Pinned in-process: the subject is the transport and its address,
            # not the live-consumer gate, which would need a broker.
            "locus": EnumDelegateLocus.IN_PROCESS,
            "state_root": tmp_path / "state",
            "timeout": 60,
            "verbose": False,
            "emit_socket": tmp_path / "no-daemon.sock",
            "omni_home": root,
        }
        kwargs.update(overrides)
        return run_delegate(**kwargs)  # type: ignore[arg-type]

    def test_the_default_run_dispatches_to_the_bound_lane(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch, binding="dev")
        assert self._run(tmp_path, _workspace(tmp_path / "ws")) == 0
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DEV_BROKER,
        }
        transport_lines = [
            line
            for line in capsys.readouterr().err.splitlines()
            if line.startswith("transport: ")
        ]
        assert len(transport_lines) == 1
        assert transport_lines[0].startswith("transport: bus=kafka lane=dev (")
        assert "developer lane binding 'dev'" in transport_lines[0]

    def test_an_explicit_inmemory_bus_outranks_the_binding(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch, binding="dev")
        assert self._run(tmp_path, _workspace(tmp_path / "ws"), bus="inmemory") == 0
        assert captured["backend_overrides"] == {"event_bus": "inmemory"}
        assert (
            "transport: bus=inmemory lane=none (explicit --bus inmemory)"
            in capsys.readouterr().err
        )

    def test_an_unreadable_binding_refuses_and_dispatches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch, binding="   ")
        with pytest.raises(click.ClickException) as exc:
            self._run(tmp_path, _workspace(tmp_path / "ws"))
        assert "developer.lane_binding" in str(exc.value.message)
        assert captured == {}


class TestTheRunFilesNameTheAuthority:
    def test_transport_authority_is_a_run_file_field(self) -> None:
        addressing = ModelDelegateRunAddressing(
            locus=EnumDelegateLocus.DEPLOYED_LANE,
            bus="kafka",
            lane="dev",
            transport_authority="developer lane binding 'dev'",
        )
        assert addressing.as_run_file_fields()["transport_authority"] == (
            "developer lane binding 'dev'"
        )
