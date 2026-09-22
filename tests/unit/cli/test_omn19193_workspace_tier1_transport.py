# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A registry workspace's embedded runtime resolves its own tier-1 config (OMN-19193).

THE DEFECT. OMN-14376 was closed by making ``onex delegate`` follow the machine's
configured bus, and on the operator's workspace that configuration WAS
``ONEX_EVENT_BUS_TYPE=kafka`` in a shell profile. OMN-17304 then ruled -- rightly
-- that an env var holds no tier in transport resolution, and that an
unconfigured install answers with the shipped tier-0 default (in-memory bus,
local SQLite evidence). Tier-1 (self-hosted) overlays were to compose on top.
None was ever declared for the registry workspace, so from 2026-09-01 every
default ``onex delegate`` there resolved tier-0: evidence landed only in the
local SQLite file and never in the shared ``delegation_events`` projection,
with a stderr warning as the only trace. Observed on run
``844cac18-1d4a-4e74-9377-4049a94b8904``.

THE FIX, at the configuration authority and nowhere else:

* the runtime config gains ``event_bus.lane`` -- which declared lane a
  local-profile runtime's shared bus is -- because a kafka transport with no
  lane is refused (OMN-16871) and a config could not otherwise say which;
* ``resolve_embedded_runtime_config`` gains a WORKSPACE tier between the
  ``ONEX_CONTRACTS_DIR`` bootstrap pointer and tier-0: the workspace's own
  ``config/onex/runtime/runtime_config.yaml`` under the workspace root the CLI
  already binds to find the lane declaration. The file belongs to the
  workspace; this package ships the convention and no lab values (OMN-19184).
  A bound root that declares none is refused, never answered with tier-0;
* ``run_delegate`` uses the configured lane only when the transport itself
  came from the configuration. An explicit ``--bus`` is tier 1 and keeps its
  own addressing; an explicit ``--bus inmemory`` never inherits a lane.

A configured kafka lane with no live orchestrator is refused by the existing
fail-closed locus gate -- loudly, naming the topic -- rather than degrading to
the in-memory bus. That gate is pinned in ``test_delegate_locus.py`` and is
not re-tested here.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import resolve_default_bus, run_delegate
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.delegate_locus import DelegateLocusRefusedError
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.enum_event_bus_profile import EnumEventBusProfile
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig
from omnibase_infra.runtime.service_kernel import (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH,
    load_runtime_config,
    resolve_embedded_runtime_config,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: A synthetic dev-lane address, deliberately unlike any real listener.
DECLARED_DEV_BROKER = "declared-dev.example:19092"

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

_TIER1 = """
description: "workspace tier-1 runtime config (test)"
event_bus:
  type: "kafka"
  profile: "local"
  lane: "dev"
"""


def _workspace(root: Path, *, tier1: str | None = _TIER1) -> Path:
    """A workspace root with a lane declaration and, optionally, a tier-1 config."""
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_LANES, encoding="utf-8")
    if tier1 is not None:
        config = (
            root
            / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(tier1, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    # The operator's shell: an ignored transport export and an ambient broker.
    monkeypatch.setenv("ONEX_EVENT_BUS_TYPE", "kafka")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "192.0.2.1:9092")
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


class TestTheWorkspaceTier:
    """AC3: the workspace tier is read through the one resolution authority."""

    def test_a_bound_workspace_with_a_tier1_config_resolves_it(
        self, tmp_path: Path
    ) -> None:
        root = _workspace(tmp_path)
        config, source = resolve_embedded_runtime_config(workspace_root=root)
        assert config.event_bus.type is EnumEventBusType.KAFKA
        assert config.event_bus.lane == "dev"
        assert "workspace tier-1" in source
        assert str(root / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH) in source

    def test_a_bound_workspace_without_one_is_refused_not_tier0(
        self, tmp_path: Path
    ) -> None:
        root = _workspace(tmp_path, tier1=None)
        with pytest.raises(ProtocolConfigurationError) as exc:
            resolve_embedded_runtime_config(workspace_root=root)
        message = str(exc.value)
        assert (
            str(root / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH / "runtime") in message
        )
        assert "--bus inmemory" in message

    def test_the_bootstrap_pointer_still_outranks_the_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pointed = tmp_path / "pointed"
        (pointed / "runtime").mkdir(parents=True)
        (pointed / "runtime" / "runtime_config.yaml").write_text(
            'event_bus:\n  type: "inmemory"\n  profile: "local"\n', encoding="utf-8"
        )
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(pointed))
        config, source = resolve_embedded_runtime_config(
            workspace_root=_workspace(tmp_path / "ws")
        )
        assert config.event_bus.type is EnumEventBusType.INMEMORY
        assert str(pointed) in source

    def test_resolve_default_bus_carries_the_configured_lane(
        self, tmp_path: Path
    ) -> None:
        resolved = resolve_default_bus(workspace_root=_workspace(tmp_path))
        assert resolved.bus == "kafka"
        assert resolved.lane == "dev"
        assert "workspace tier-1" in resolved.reason

    def test_no_workspace_root_is_unchanged_tier0(self) -> None:
        resolved = resolve_default_bus()
        assert resolved.bus == "inmemory"
        assert resolved.lane is None


class TestTheLaneField:
    """The field is expressible, and only where it means something."""

    def test_a_lane_on_the_in_memory_bus_is_refused(self) -> None:
        with pytest.raises(ValidationError) as exc:
            ModelEventBusConfig(type="inmemory", profile="local", lane="dev")
        assert "lane" in str(exc.value)

    def test_an_empty_lane_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelEventBusConfig(type="kafka", profile="local", lane="")

    def test_the_product_ships_no_lane_value(self) -> None:
        # Lab configuration is never hardcoded in the product every customer
        # runs (OMN-19184). The package ships the resolution tier and the
        # optional field; the lane a workspace uses is the workspace's own.
        assert not (_REPO_ROOT / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH).exists()
        declared = [
            path
            for path in _REPO_ROOT.rglob("runtime_config.yaml")
            if "tests" not in path.relative_to(_REPO_ROOT).parts
            and ".venv" not in path.relative_to(_REPO_ROOT).parts
            and load_runtime_config(path.parent.parent).event_bus.lane is not None
        ]
        assert declared == []


class TestRunDelegateDefaultPath:
    """AC3/AC4 end to end through ``run_delegate`` with no ``--bus``."""

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

    def _run(self, tmp_path: Path, root: Path | None, **overrides: object) -> int:
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

    def test_the_default_run_lands_on_the_configured_lane_not_inmemory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        assert self._run(tmp_path, _workspace(tmp_path / "ws")) == 0
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }

    def test_a_bound_root_with_no_tier1_config_refuses_and_dispatches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        with pytest.raises(click.ClickException) as exc:
            self._run(tmp_path, _workspace(tmp_path / "ws", tier1=None))
        assert "--bus inmemory" in str(exc.value.message)
        assert captured == {}

    def test_positive_control_no_workspace_root_is_tier0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        assert self._run(tmp_path, None) == 0
        assert captured["backend_overrides"] == {"event_bus": "inmemory"}

    def test_a_lane_with_no_orchestrator_is_refused_naming_it_and_the_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)

        def _no_consumer(**_: object) -> object:
            raise DelegateLocusRefusedError(
                "no live consumer group is bound to 'onex.cmd.test.v1'"
            )

        monkeypatch.setattr(cli_delegate, "resolve_delegate_locus", _no_consumer)
        with pytest.raises(click.ClickException) as exc:
            self._run(
                tmp_path, _workspace(tmp_path / "ws"), locus=EnumDelegateLocus.AUTO
            )
        message = str(exc.value.message)
        assert "no live consumer group" in message
        assert "lane 'dev'" in message
        assert DECLARED_DEV_BROKER in message
        assert "--bus inmemory" in message
        assert captured == {}

    def test_an_explicit_inmemory_bus_does_not_inherit_the_configured_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        assert self._run(tmp_path, _workspace(tmp_path / "ws"), bus="inmemory") == 0
        assert captured["backend_overrides"] == {"event_bus": "inmemory"}

    def test_an_explicit_lane_outranks_the_configured_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        assert (
            self._run(
                tmp_path,
                _workspace(tmp_path / "ws"),
                bus="kafka",
                lane="stability-test",
            )
            == 0
        )
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": "stability.example:39092",
        }


class TestTheRefusalNamesTheBoundVariable:
    """AC5: the no-root refusal names the variable --omni-home actually binds."""

    def test_the_message_names_omnibase_path_not_omni_home(self) -> None:
        from omnibase_infra.cli.delegate_lane import (
            DelegateLaneSelectionError,
            resolve_lane_declaration_path,
        )

        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_declaration_path(None)
        assert "$OMNIBASE_PATH" in str(exc.value)
        assert "$OMNI_HOME" not in str(exc.value)
