# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` stamps its request with the tenant of the install (OMN-17427).

The request model has declared ``tenant_id`` since OMN-14058 and the producer
carries it onto the verdict, but this CLI never wrote it: 845 of 856 delegate
completions reached the writer with no tenant and were dead-lettered. The CLI
is the only place that can stamp it, because a deployed lane holds no local
store.

What these tests pin:

* the stamp is the install's minted identity, or a declared tenant overlay, and
  nothing else -- there is no default;
* a request bound for a deployed lane from an install that never ran
  ``onex local init`` is refused, naming the command, with nothing written and
  nothing sent;
* the one request that carries no stamp is the in-process first delegation, the
  documented OMN-19966 path on which the local port mints the identity itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import click
import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    INSTALL_IDENTITY_MODULE,
    TENANT_OVERLAY_ENV,
    DelegateTenantRefusedError,
    _write_payload,
    resolve_delegate_tenant,
    run_delegate,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from tests.helpers.cli_registry_stand_in import STAND_IN_INSTALL_TENANT

# Captured at import, before the autouse fixture replaces the module attribute.
_REAL_READ_INSTALL_IDENTITY = cli_delegate.read_install_identity

pytestmark = pytest.mark.unit

_KAFKA_BOOTSTRAP = "localhost:19092"
_MINTED = "0b2a6f1e-7c1d-4a39-8d55-3e8a4c9f1b27"
_OVERLAY = "lab-house"


def _uninitialised_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: None)


class TestResolution:
    def test_the_install_identity_is_the_stamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
        assert resolve_delegate_tenant(in_process=False, environ={}) == _MINTED

    def test_a_declared_overlay_wins_over_the_install_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
        stamp = resolve_delegate_tenant(
            in_process=False, environ={TENANT_OVERLAY_ENV: f"  {_OVERLAY} "}
        )
        assert stamp == _OVERLAY

    def test_a_blank_overlay_is_not_a_declaration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
        assert (
            resolve_delegate_tenant(in_process=False, environ={TENANT_OVERLAY_ENV: " "})
            == _MINTED
        )

    def test_an_uninitialised_install_is_refused_for_a_deployed_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _uninitialised_install(monkeypatch)
        with pytest.raises(DelegateTenantRefusedError) as refusal:
            resolve_delegate_tenant(in_process=False, environ={})
        assert "onex local init" in str(refusal.value)

    def test_an_uninitialised_in_process_request_carries_no_stamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OMN-19966: the local port mints the identity on the first delegation.
        _uninitialised_install(monkeypatch)
        assert resolve_delegate_tenant(in_process=True, environ={}) is None


class TestInstallIdentityReader:
    """The reader resolves omnimarket's module at run time, never at import."""

    @staticmethod
    def _install_module(
        monkeypatch: pytest.MonkeyPatch, reader: object, error: type[Exception]
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            INSTALL_IDENTITY_MODULE,
            SimpleNamespace(
                read_local_tenant_identity=reader, LocalTenantIdentityError=error
            ),
        )

    def test_it_returns_the_minted_uuid_as_a_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_module(
            monkeypatch,
            lambda: SimpleNamespace(tenant_uuid=UUID(_MINTED)),
            RuntimeError,
        )
        assert _REAL_READ_INSTALL_IDENTITY() == _MINTED

    def test_an_install_that_never_minted_reads_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_module(monkeypatch, lambda: None, RuntimeError)
        assert _REAL_READ_INSTALL_IDENTITY() is None

    def test_a_corrupt_identity_is_a_refusal_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _CorruptError(RuntimeError):
            pass

        def _read() -> object:
            raise _CorruptError("recorded value is not a UUID")

        self._install_module(monkeypatch, _read, _CorruptError)
        with pytest.raises(DelegateTenantRefusedError, match="not a UUID"):
            _REAL_READ_INSTALL_IDENTITY()

    def test_an_unresolvable_module_is_a_refusal_naming_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, INSTALL_IDENTITY_MODULE, None)
        with pytest.raises(DelegateTenantRefusedError, match="not resolvable"):
            _REAL_READ_INSTALL_IDENTITY()


class TestPayload:
    def test_the_stamp_is_written_as_tenant_id(self, tmp_path: Path) -> None:
        path = _write_payload(
            prompt="Reply with exactly: OK",
            task_type="summarization",
            source="claude-code",
            state_root=tmp_path,
            run_id=uuid4(),
            correlation_id=uuid4(),
            max_tokens=None,
            tenant_id=_MINTED,
        )
        assert json.loads(path.read_text(encoding="utf-8"))["tenant_id"] == _MINTED

    def test_no_stamp_leaves_the_field_out_entirely(self, tmp_path: Path) -> None:
        path = _write_payload(
            prompt="Reply with exactly: OK",
            task_type="summarization",
            source="claude-code",
            state_root=tmp_path,
            run_id=uuid4(),
            correlation_id=uuid4(),
            max_tokens=None,
        )
        assert "tenant_id" not in json.loads(path.read_text(encoding="utf-8"))


class TestRunDelegate:
    """The whole command: what reaches the receipt layer, and what never does."""

    @staticmethod
    def _run(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        bus: str,
        locus: EnumDelegateLocus,
        omni_home: Path | None = None,
    ) -> dict[str, object]:
        """Run the command with the receipt layer faked; return what it was handed."""
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
        run_delegate(
            prompt="document the router",
            task_type="document",
            max_tokens=None,
            bus=bus,
            locus=locus,
            kafka_bootstrap=_KAFKA_BOOTSTRAP if bus == "kafka" else None,
            state_root=tmp_path / "state",
            timeout=60,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            omni_home=omni_home,
        )
        return captured

    @staticmethod
    def _sent_payload(captured: dict[str, object]) -> dict[str, object]:
        input_path = captured["input_path"]
        assert isinstance(input_path, Path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    def test_a_request_from_an_initialised_install_carries_its_tenant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._run(
            tmp_path, monkeypatch, bus="kafka", locus=EnumDelegateLocus.IN_PROCESS
        )
        assert self._sent_payload(captured)["tenant_id"] == STAND_IN_INSTALL_TENANT

    def test_a_declared_overlay_is_the_tenant_the_request_carries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TENANT_OVERLAY_ENV, _OVERLAY)
        captured = self._run(
            tmp_path, monkeypatch, bus="kafka", locus=EnumDelegateLocus.IN_PROCESS
        )
        assert self._sent_payload(captured)["tenant_id"] == _OVERLAY

    def test_an_uninitialised_install_sends_nothing_to_a_deployed_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _uninitialised_install(monkeypatch)
        with pytest.raises(click.ClickException, match="onex local init"):
            self._run(
                tmp_path,
                monkeypatch,
                bus="kafka",
                locus=EnumDelegateLocus.DEPLOYED_LANE,
            )
        # Refused before the request was even written, let alone published.
        assert not (tmp_path / "state" / "tmp").exists()

    def test_the_default_locus_on_a_shared_bus_is_a_deployed_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``--bus kafka`` with no ``--locus`` resolves to the deployed lane, so
        # the refusal applies without the operator having to say so.
        _uninitialised_install(monkeypatch)
        with pytest.raises(click.ClickException, match="onex local init"):
            self._run(tmp_path, monkeypatch, bus="kafka", locus=EnumDelegateLocus.AUTO)

    def test_the_in_process_first_delegation_stays_unstamped_for_the_port_to_mint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _uninitialised_install(monkeypatch)
        captured = self._run(
            tmp_path, monkeypatch, bus="inmemory", locus=EnumDelegateLocus.IN_PROCESS
        )
        assert "tenant_id" not in self._sent_payload(captured)


class TestLabHostTenant:
    @pytest.mark.parametrize(
        ("host", "install"),
        [
            ("h201", "a630ba90-7baa-4aec-b990-c34ec71ea2fb"),
            ("h202", "89941c22-11fc-4ff5-bd21-8eb7e9ba4567"),
        ],
    )
    def test_lab_host_without_overlay_stamps_declared_house_tenant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, install: str
    ) -> None:
        house = "820272f9-4aaf-5add-a2df-0af942852ab2"
        table = tmp_path / "lab_run_hosts.yaml"
        table.write_text(
            f"hosts:\n  - name: {host}\n    target: lab.invalid\n    tenant_id: {house}\n"
        )
        monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: install)
        stamp = resolve_delegate_tenant(
            in_process=False,
            environ={
                "ONEX_LAB_RUN_HOSTS": str(table),
                "ONEX_LANE_HOST": host,
            },
        )
        assert stamp == house

    @pytest.mark.parametrize(
        "rows",
        [
            "[]",
            "[{name: h201, target: lab.invalid}]",
            "[{name: h201, target: lab.invalid, tenant_id: bad-uuid}]",
            "[{name: h202, target: lab.invalid, tenant_id: 820272f9-4aaf-5add-a2df-0af942852ab2}]",
            "[{name: h201, target: lab.invalid}, {name: h201, target: lab.invalid}]",
        ],
    )
    def test_unresolved_lab_declaration_refuses_instead_of_reading_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: str
    ) -> None:
        table = tmp_path / "hosts.yaml"
        table.write_text(f"hosts: {rows}")

        def never_read_install() -> str:
            pytest.fail(
                "an invalid lab declaration must never read the install identity"
            )

        monkeypatch.setattr(cli_delegate, "read_install_identity", never_read_install)
        with pytest.raises(DelegateTenantRefusedError, match="lab tenant declaration"):
            resolve_delegate_tenant(
                in_process=True,
                environ={
                    "ONEX_LAB_RUN_HOSTS": str(table),
                    "ONEX_LANE_HOST": "h201",
                },
            )

    def test_missing_table_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(DelegateTenantRefusedError, match="lab tenant declaration"):
            resolve_delegate_tenant(
                in_process=False,
                environ={
                    "ONEX_LAB_RUN_HOSTS": str(tmp_path / "absent"),
                    "ONEX_LANE_HOST": "h201",
                },
            )

    def test_native_job_reads_row_through_symlinked_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = tmp_path / "registry" / "omni_home"
        registry.mkdir(parents=True)
        alias = tmp_path / "Code" / "omni_home"
        alias.parent.mkdir()
        alias.symlink_to(registry, target_is_directory=True)
        table = (
            registry.parent
            / "omnibase_internal/src/omnibase_internal/lab_run_hosts.yaml"
        )
        table.parent.mkdir(parents=True)
        house = "820272f9-4aaf-5add-a2df-0af942852ab2"
        table.write_text(
            f"hosts:\n  - name: h201\n    target: lab.invalid\n    tenant_id: {house}\n"
        )
        monkeypatch.setattr(cli_delegate.socket, "gethostname", lambda: "omninode-pc")
        monkeypatch.setattr(
            cli_delegate, "_lab_target_is_local", lambda target: target == "lab.invalid"
        )
        assert (
            resolve_delegate_tenant(in_process=False, environ={"OMNI_HOME": str(alias)})
            == house
        )

    def test_customer_registry_without_lab_table_keeps_install_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_delegate, "read_install_identity", lambda: _MINTED)
        assert (
            resolve_delegate_tenant(
                in_process=False, environ={"OMNI_HOME": str(tmp_path)}
            )
            == _MINTED
        )

    def test_explicit_customer_overlay_keeps_precedence(self, tmp_path: Path) -> None:
        assert (
            resolve_delegate_tenant(
                in_process=False,
                environ={
                    TENANT_OVERLAY_ENV: _MINTED,
                    "ONEX_LANE_HOST": "h201",
                    "ONEX_LAB_RUN_HOSTS": str(tmp_path / "absent"),
                },
            )
            == _MINTED
        )

    def test_lab_stamp_reaches_request_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        house = "820272f9-4aaf-5add-a2df-0af942852ab2"
        table = tmp_path / "hosts.yaml"
        table.write_text(
            f"hosts:\n  - name: h202\n    target: lab.invalid\n    tenant_id: {house}\n"
        )
        monkeypatch.setenv("ONEX_LAB_RUN_HOSTS", str(table))
        monkeypatch.setenv("ONEX_LANE_HOST", "h202")
        captured = TestRunDelegate._run(
            tmp_path, monkeypatch, bus="kafka", locus=EnumDelegateLocus.IN_PROCESS
        )
        assert TestRunDelegate._sent_payload(captured)["tenant_id"] == house

    def test_wrapper_workspace_binding_stamps_lab_tenant_without_omni_home_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = tmp_path / "omni_home"
        registry.mkdir()
        table = tmp_path / "omnibase_internal/src/omnibase_internal/lab_run_hosts.yaml"
        table.parent.mkdir(parents=True)
        house = "820272f9-4aaf-5add-a2df-0af942852ab2"
        table.write_text(
            f"hosts:\n  - name: h202\n    target: lab.invalid\n    tenant_id: {house}\n"
        )
        monkeypatch.setattr(cli_delegate.socket, "gethostname", lambda: "h202")
        captured = TestRunDelegate._run(
            tmp_path,
            monkeypatch,
            bus="kafka",
            locus=EnumDelegateLocus.IN_PROCESS,
            omni_home=registry,
        )
        assert TestRunDelegate._sent_payload(captured)["tenant_id"] == house
