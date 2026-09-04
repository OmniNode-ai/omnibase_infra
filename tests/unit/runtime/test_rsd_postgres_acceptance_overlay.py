# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
import re
import textwrap
from pathlib import Path

import pytest

from omnibase_infra.testing.rsd_postgres_acceptance_plugin import load_overlay


def test_overlay_is_typed_and_topology_free() -> None:
    path = (
        Path(__file__).parents[3]
        / "docker/lane-overlays/dev.rsd-postgres-acceptance.yaml"
    )
    overlay = load_overlay(path)
    assert overlay.lane == "dev" and overlay.locale == "lab"
    overlay_text = path.read_text()
    forbidden = re.compile(
        r"(?i)(?:\b(?:host|port|dsn|password|passwd|username|user|url|endpoint|ip|address|env)\s*:|"
        r"\b(?:host|port|dsn|password|passwd|username|user|url|endpoint|ip|address|env)\s*=|"
        r"https?://|(?:\d{1,3}\.){3}\d{1,3}|(?:postgres(?:ql)?|mysql)://)"
    )
    assert forbidden.search(overlay_text) is None


def test_overlay_allows_opaque_uuid_capability_reference(tmp_path: Path) -> None:
    path = tmp_path / "uuid-overlay.yaml"
    path.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\n"
        "lane: dev\n"
        "locale: lab\n"
        "rsd_distribution_ref: omninode-rsd/0.1.0\n"
        "postgres_capability_ref: capability://rsd/postgres/"
        "123e4567-e89b-12d3-a456-426614174000\n"
    )
    overlay = load_overlay(path)
    assert overlay.postgres_capability_ref.endswith(
        "123e4567-e89b-12d3-a456-426614174000"
    )


def test_explicit_plugin_injection_harness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the plugin through pytest's explicit ``-p`` loading path."""
    overlay = tmp_path / "valid-overlay.yaml"
    overlay.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\n"
        "lane: dev\n"
        "locale: lab\n"
        "rsd_distribution_ref: omninode-rsd/0.1.0\n"
        "postgres_capability_ref: capability://rsd/postgres/acceptance\n"
    )
    mismatch_overlay = tmp_path / "mismatch-overlay.yaml"
    mismatch_overlay.write_text(overlay.read_text().replace("acceptance", "mismatch"))
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            """
            from contextlib import contextmanager
            from uuid import uuid4

            EXPECTED_REF = "capability://rsd/postgres/acceptance"
            active_connections = 0

            class FakeConnection:
                def __init__(self):
                    self.transaction_id = uuid4()
                    self.transaction_state = "IDLE"
                    self.exclusive = True
                    self.closed = False

            def _resolver(capability_ref):
                if capability_ref != EXPECTED_REF:
                    raise LookupError("unsupported capability reference")
                return _factory

            def _factory():
                @contextmanager
                def _connection():
                    global active_connections
                    if active_connections:
                        raise AssertionError("connection CMs must be exclusive")
                    active_connections += 1
                    connection = FakeConnection()
                    try:
                        yield connection
                    finally:
                        assert connection.transaction_state == "IDLE"
                        connection.closed = True
                        active_connections -= 1

                return _connection()

            def pytest_configure(config):
                config.rsd_postgres_acceptance_capability_resolver = _resolver
            """
        ).lstrip()
    )
    harness = tmp_path / "test_plugin_harness.py"
    harness.write_text(
        textwrap.dedent(
            """
            import pytest

            def test_fresh_exclusive_transaction_idle_connections(
                postgres_lifecycle_connection_factory,
            ):
                first_cm = postgres_lifecycle_connection_factory()
                with first_cm as first:
                    assert first.exclusive
                    assert first.transaction_state == "IDLE"
                    with pytest.raises(AssertionError, match="exclusive"):
                        with postgres_lifecycle_connection_factory():
                            pass
                second_cm = postgres_lifecycle_connection_factory()
                assert first_cm is not second_cm
                with second_cm as second:
                    assert second.exclusive
                    assert second.transaction_state == "IDLE"
                    assert first.transaction_id != second.transaction_id
                    assert first.closed
                assert second.closed
            """
        ).lstrip()
    )
    plugin = "omnibase_infra.testing.rsd_postgres_acceptance_plugin"
    option = f"--rsd-postgres-acceptance-overlay={overlay}"
    assert pytest.main(["-q", "-p", plugin, option, str(harness)]) == pytest.ExitCode.OK
    missing_result = pytest.main(["-q", "-p", plugin, str(harness)])
    assert missing_result == pytest.ExitCode.TESTS_FAILED
    assert (
        "RSD PostgreSQL acceptance overlay path must be explicit"
        in capsys.readouterr().out
    )
    mismatch_option = f"--rsd-postgres-acceptance-overlay={mismatch_overlay}"
    mismatch_result = pytest.main(["-q", "-p", plugin, mismatch_option, str(harness)])
    assert mismatch_result == pytest.ExitCode.TESTS_FAILED


def test_overlay_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\nlane: dev\nlocale: lab\nrsd_distribution_ref: omninode-rsd/0.1.0\npostgres_capability_ref: capability://rsd/postgres/acceptance\nhost: bad\n"
    )
    with pytest.raises(Exception):
        load_overlay(path)
