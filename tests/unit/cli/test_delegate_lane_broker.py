# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16871: the delegate broker ADDRESS comes from a lane, not the shell.

`~/.omnibase/.env` on the launching Mac sets `KAFKA_BOOTSTRAP_SERVERS` to the
`.201` STABILITY-TEST lane. Until this change `onex delegate --bus kafka` with
no explicit broker let `EventBusKafka` read that variable, so every ad hoc
delegation from a developer shell published onto a governed proof lane -- the
lane the compose-path prod-promotion gate resolves its `stability-proven`
premise from. The lane recorded it: four
`local.omnibase_core.runtime_local_terminal_run_*` consumer groups belonging
to CLI processes on that Mac were sitting on the stability broker when the
finding was re-verified on 2026-09-16.

These tests pin the fix as a REFUSAL, not a better default. A default is a
value someone can be wrong about silently; a refusal that names the missing
selection cannot be. The two assertions that matter most are negative ones:
with the ambient variable exported to the stability address, a kafka
delegation that named no lane FAILS, and a kafka delegation that named
`--lane` uses the DECLARED address rather than the exported one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    build_backend_overrides,
    delegate_command,
    run_delegate,
)
from omnibase_infra.cli.delegate_lane import (
    LANE_DECLARATION_RELATIVE_PATH,
    DelegateLaneSelectionError,
    declared_lane_ids,
    resolve_lane_declaration_path,
    resolve_lane_selection,
    resolve_lane_target,
)
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bound_lane_client_transport,
)

pytestmark = pytest.mark.unit

#: The ambient value this ticket exists about. Spelled here so a test can
#: prove it is NOT what the CLI publishes to.
AMBIENT_STABILITY_BROKER = "192.168.86.201:39092"  # onex-allow-internal-ip OMN-16871 reason="test fixture quoting the ambient env value the CLI must no longer resolve; not a configurable endpoint"

#: A synthetic dev-lane address, deliberately unlike the ambient one.
DECLARED_DEV_BROKER = "declared-dev.example:19092"

#: Captured before any fixture runs. ``tests/unit/cli/conftest.py`` deletes
#: OMNI_HOME per test to keep the drift guard hermetic; the positive control
#: below still needs to know whether this host has the workspace.
_OMNI_HOME_AT_IMPORT = os.environ.get("OMNI_HOME", "").strip()

_DECLARATION = f"""
lanes:
  dev:
    broker: "{DECLARED_DEV_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  stability-test:
    broker: "stability.example:39092"
    security_protocol: PLAINTEXT
  prod:
    broker: inmemory
"""


def _workspace(root: Path, body: str = _DECLARATION) -> Path:
    """A workspace root carrying a lane declaration where the CLI looks."""
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(body, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _ambient_env_points_at_the_governed_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the launching Mac, so a silent fallback would be visible.

    Every test in this module runs with the defect's precondition present. If
    the env path ever comes back, the tests that assert a refusal will resolve
    an address instead and fail.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", AMBIENT_STABILITY_BROKER)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


class TestLaneDeclarationLookup:
    """Locating the one declaration, and refusing when it is not there."""

    def test_declaration_is_found_under_the_workspace_root(
        self, tmp_path: Path
    ) -> None:
        root = _workspace(tmp_path)
        assert resolve_lane_declaration_path(root) == (
            root / "omnimarket" / "config" / "ci_bus_lanes.yaml"
        )

    def test_no_workspace_root_is_refused_naming_the_flag(self) -> None:
        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_declaration_path(None)
        assert "--omnibase-path" in str(exc.value)

    def test_absent_declaration_is_refused_naming_the_path(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_declaration_path(tmp_path)
        assert str(tmp_path / LANE_DECLARATION_RELATIVE_PATH) in str(exc.value)

    def test_declared_lane_ids_are_listed(self, tmp_path: Path) -> None:
        declaration = resolve_lane_declaration_path(_workspace(tmp_path))
        assert declared_lane_ids(declaration) == ("dev", "prod", "stability-test")

    def test_a_declaration_with_no_lanes_is_refused(self, tmp_path: Path) -> None:
        declaration = resolve_lane_declaration_path(_workspace(tmp_path, "lanes: {}\n"))
        with pytest.raises(DelegateLaneSelectionError, match="declares no lanes"):
            declared_lane_ids(declaration)


class TestLaneSelection:
    """A selected lane resolves to the address the declaration binds to it."""

    def test_declared_lane_resolves_to_its_broker_and_transport(
        self, tmp_path: Path
    ) -> None:
        selection = resolve_lane_selection(lane="dev", omni_home=_workspace(tmp_path))
        assert selection.bootstrap_servers == DECLARED_DEV_BROKER
        assert selection.security_protocol == "SASL_PLAINTEXT"
        assert selection.sasl_mechanism == "SCRAM-SHA-256"
        assert selection.declared_in.name == "ci_bus_lanes.yaml"

    def test_the_resolved_address_is_never_the_ambient_one(
        self, tmp_path: Path
    ) -> None:
        """The whole ticket, as one assertion.

        The ambient variable is exported (autouse fixture) and names the
        governed lane; the declaration names something else; the declaration
        wins.
        """
        assert os.environ["KAFKA_BOOTSTRAP_SERVERS"] == AMBIENT_STABILITY_BROKER
        selection = resolve_lane_selection(lane="dev", omni_home=_workspace(tmp_path))
        assert selection.bootstrap_servers != AMBIENT_STABILITY_BROKER

    def test_undeclared_lane_is_refused_and_lists_what_is_declared(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_selection(lane="judge", omni_home=_workspace(tmp_path))
        message = str(exc.value)
        assert "judge" in message
        assert "dev" in message

    def test_an_inmemory_lane_is_refused_rather_than_published_to(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DelegateLaneSelectionError, match="in-memory"):
            resolve_lane_selection(lane="prod", omni_home=_workspace(tmp_path))


class TestSelectionPolicy:
    """Which combinations of flags are a complete broker selection."""

    def test_kafka_with_no_selection_is_refused_naming_the_missing_flag(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_target(
                bus="kafka",
                lane=None,
                kafka_bootstrap=None,
                omni_home=_workspace(tmp_path),
            )
        message = str(exc.value)
        assert "--lane" in message
        # The declared lanes are listed, so the refusal is actionable rather
        # than merely correct.
        assert "dev" in message and "stability-test" in message

    def test_the_refusal_does_not_offer_the_environment_as_a_path(
        self, tmp_path: Path
    ) -> None:
        """A refusal that names the env var as the remedy reopens the defect."""
        with pytest.raises(DelegateLaneSelectionError) as exc:
            resolve_lane_target(
                bus="kafka",
                lane=None,
                kafka_bootstrap=None,
                omni_home=_workspace(tmp_path),
            )
        message = str(exc.value)
        assert "KAFKA_BOOTSTRAP_SERVERS" in message, (
            "the refusal should name the removed path so the operator "
            "understands why their export stopped working"
        )
        assert "NOT read from KAFKA_BOOTSTRAP_SERVERS" in message

    def test_both_selections_together_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DelegateLaneSelectionError, match="both"):
            resolve_lane_target(
                bus="kafka",
                lane="dev",
                kafka_bootstrap="broker.example:9092",
                omni_home=_workspace(tmp_path),
            )

    def test_explicit_bootstrap_alone_consults_no_declaration(
        self, tmp_path: Path
    ) -> None:
        # No workspace is written at all: the explicit address must not need
        # one, because it carries its own provenance.
        assert (
            resolve_lane_target(
                bus="kafka",
                lane=None,
                kafka_bootstrap="redpanda:9092",
                omni_home=tmp_path,
            )
            is None
        )

    def test_a_lane_on_an_in_process_bus_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DelegateLaneSelectionError, match="only valid with"):
            resolve_lane_target(
                bus="inmemory",
                lane="dev",
                kafka_bootstrap=None,
                omni_home=_workspace(tmp_path),
            )

    def test_an_in_process_bus_needs_no_selection(self, tmp_path: Path) -> None:
        assert (
            resolve_lane_target(
                bus="inmemory",
                lane=None,
                kafka_bootstrap=None,
                omni_home=tmp_path,
            )
            is None
        )


class TestOverrideMapHasNoEnvFallback:
    """The structural half: a kafka override map without an address is refused."""

    def test_kafka_without_an_address_is_refused(self) -> None:
        with pytest.raises(ValueError) as exc:
            build_backend_overrides(bus="kafka", kafka_bootstrap=None)
        assert "--lane" in str(exc.value)

    def test_kafka_with_an_address_threads_it(self) -> None:
        assert build_backend_overrides(
            bus="kafka", kafka_bootstrap=DECLARED_DEV_BROKER
        ) == {"event_bus": "kafka", "kafka_bootstrap": DECLARED_DEV_BROKER}


class TestRunDelegateAddressing:
    """End to end through ``run_delegate``, with the ambient value exported."""

    @staticmethod
    def _capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        captured: dict[str, object] = {}

        def _fake_run_receipt_mode(**kwargs: object) -> int:
            captured.update(kwargs)
            # OMN-18432: what was bound WHILE the dispatch ran, captured here
            # because the binding is scoped and is already gone by the time
            # run_delegate returns. That scoping is the property being pinned.
            captured["lane_transport"] = bound_lane_client_transport()
            return 0

        monkeypatch.setattr(
            cli_delegate,
            "_resolve_packaged_contract",
            lambda _name: tmp_path / "contract.yaml",
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)

        # OMN-18432: the declared dev lane is SASL_PLAINTEXT, and a machine
        # holding no identity for it is now REFUSED rather than allowed to
        # connect anonymously. This fixture is that machine holding one --
        # the refusal itself is pinned in
        # tests/unit/cli/test_omn18432_delegate_lane_credentials.py, so
        # asserting it again here would only re-test the refusal instead of
        # the addressing this class is about.
        home = tmp_path / "fake-home"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
        StoreLaneCredential(onex_home=home / ".onex").save(
            lane="dev",
            sasl_username="dev-cli-under-test",
            sasl_password="not-a-real-value",
        )
        return captured

    def test_selected_lane_is_what_the_runtime_is_handed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        exit_code = run_delegate(
            prompt="document the router",
            task_type="document",
            max_tokens=None,
            bus="kafka",
            # The subject is addressing, so the locus is pinned in-process --
            # otherwise this would also require a live consumer group.
            locus=EnumDelegateLocus.IN_PROCESS,
            lane="dev",
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            omni_home=_workspace(tmp_path / "workspace"),
        )
        assert exit_code == 0
        assert captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }
        # OMN-18432: the address still travels in backend_overrides, because
        # that is the only key core accepts. The rest of the declared
        # transport travels beside it, bound for this one broker.
        lane_transport = captured["lane_transport"]
        assert lane_transport is not None
        assert lane_transport.bootstrap_servers == DECLARED_DEV_BROKER
        assert lane_transport.security_protocol == "SASL_PLAINTEXT"
        assert lane_transport.sasl_mechanism == "SCRAM-SHA-256"
        assert lane_transport.sasl_username == "dev-cli-under-test"

    def test_no_lane_refuses_instead_of_using_the_ambient_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture(tmp_path, monkeypatch)
        with pytest.raises(Exception) as exc:
            run_delegate(
                prompt="document the router",
                task_type="document",
                max_tokens=None,
                bus="kafka",
                locus=EnumDelegateLocus.IN_PROCESS,
                state_root=tmp_path / "state",
                timeout=60,
                verbose=False,
                emit_socket=tmp_path / "no-daemon.sock",
                omni_home=_workspace(tmp_path / "workspace"),
            )
        assert "--lane" in str(exc.value)
        # Nothing was dispatched: the refusal happens before any receipt run.
        assert captured == {}


class TestCommandSurface:
    """What the operator sees: the flag, and the help text."""

    @staticmethod
    def _help() -> str:
        result = CliRunner().invoke(delegate_command, ["--help"])
        assert result.exit_code == 0, result.output
        return " ".join(result.output.split())

    def test_help_documents_the_lane_flag(self) -> None:
        assert "--lane" in self._help()

    def test_help_no_longer_documents_the_env_var_as_the_resolution_path(
        self,
    ) -> None:
        help_text = self._help()
        assert "Omit to resolve from KAFKA_BOOTSTRAP_SERVERS" not in help_text

    def test_a_kafka_run_with_no_lane_exits_non_zero(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            delegate_command,
            [
                "document the router",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--locus",
                "in-process",
                "--omnibase-path",
                str(_workspace(tmp_path)),
                "--state-root",
                str(tmp_path / "state"),
            ],
        )
        assert result.exit_code != 0, result.output
        assert "--lane" in result.output


class TestShippedDeclarationPositiveControl:
    """A zero from the synthetic fixtures above is not a statement about prod.

    The synthetic declaration proves the mechanism. This proves the mechanism
    is pointed at the file that actually carries the lab lanes, on a host that
    has the workspace. It SKIPS where the clone is absent (CI runners), and
    the skip reason says so rather than reading as a pass.
    """

    @staticmethod
    def _shipped() -> Path:
        # Read at IMPORT time, not here: this directory's conftest deletes
        # OMNI_HOME from every CLI test's environment to keep the drift guard
        # hermetic, so a fixture-time read would make this control skip on
        # every host including the one that has the clone -- a control that
        # can never run is not a control.
        omni_home = _OMNI_HOME_AT_IMPORT
        if not omni_home:
            pytest.skip(
                "no $OMNI_HOME workspace on this host, so the shipped lane "
                "declaration cannot be read -- UNKNOWN, not a pass"
            )
        declaration = Path(omni_home) / LANE_DECLARATION_RELATIVE_PATH
        if not declaration.is_file():
            pytest.skip(
                f"{declaration} is absent on this host (no omnimarket clone) "
                "-- UNKNOWN, not a pass"
            )
        return declaration

    def test_the_shipped_declaration_binds_dev_away_from_the_governed_lane(
        self,
    ) -> None:
        declaration = self._shipped()
        omni_home = declaration.parents[2]
        dev = resolve_lane_selection(lane="dev", omni_home=omni_home)
        stability = resolve_lane_selection(lane="stability-test", omni_home=omni_home)
        assert dev.bootstrap_servers != stability.bootstrap_servers
        assert dev.bootstrap_servers != AMBIENT_STABILITY_BROKER
        assert stability.bootstrap_servers == AMBIENT_STABILITY_BROKER, (
            "the ambient env value is the STABILITY lane address -- if this "
            "stops holding, the premise of OMN-16871 has changed"
        )
