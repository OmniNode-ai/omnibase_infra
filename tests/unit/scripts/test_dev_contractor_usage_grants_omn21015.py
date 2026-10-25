# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contractor grant entry point cannot select another lane or database."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "dev_contractor_grants", ROOT / "scripts/apply_application_database_acl.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "identity",
    [
        'omnibase-infra-prod postgres {"5432/tcp":[{"HostPort":"5436"}]}',
        'omnibase-infra-lakshman postgres {"5432/tcp":[{"HostPort":"5436"}]}',
        'omnibase-infra postgres {"5432/tcp":[{"HostPort":"5437"}]}',
        'omnibase-infra postgres {"5432/tcp":null}',
        'omnibase-infra other {"5432/tcp":[{"HostPort":"5436"}]}',
    ],
)
def test_refuses_other_lanes(identity: str) -> None:
    with patch.object(runner, "run", return_value=identity):
        with pytest.raises(RuntimeError, match="dev compose postgres"):
            runner.validate_target()


def test_refuses_target_override_before_docker() -> None:
    with patch.object(runner, "run") as command:
        with pytest.raises(SystemExit):
            runner._dev_contractor_main(["--database", "other"])
        command.assert_not_called()


def test_accepts_local_dev_and_reads_as_contractor() -> None:
    identity = 'omnibase-infra postgres {"5432/tcp":[{"HostPort":"5436"}]}'
    with patch.object(runner, "run", return_value=identity) as command:
        runner.validate_target()
        assert command.call_args.args[0][:3] == [
            "docker",
            "--host",
            "unix:///var/run/docker.sock",
        ]
    assert runner.DATABASE == "omnidash_analytics"
    assert "SET SESSION AUTHORIZATION lakshman_ro" in runner.readback_sql()
    assert "BEGIN READ ONLY" in runner.readback_sql()


def test_dry_run_never_applies() -> None:
    with (
        patch.object(runner, "validate_target"),
        patch.object(runner, "psql") as psql,
        patch("sys.argv", ["apply_dev_contractor_usage_grants.py"]),
    ):
        runner._dev_contractor_main([])
    psql.assert_not_called()


def test_check_reads_back_without_applying() -> None:
    with (
        patch.object(runner, "validate_target") as target,
        patch.object(runner, "psql", return_value="") as psql,
    ):
        runner._dev_contractor_main(["--check"])
    target.assert_called_once_with()
    assert [call.args[0] for call in psql.call_args_list] == [runner.readback_sql()]


def test_execute_applies_the_tracked_sql_file_then_reads_back() -> None:
    tracked = (ROOT / "deploy/lab/dev-contractor-usage-grants.sql").read_text()
    assert runner.GRANTS.read_text() == tracked
    with (
        patch.object(runner, "validate_target") as target,
        patch.object(runner, "psql", return_value="") as psql,
    ):
        runner._dev_contractor_main(["--execute"])
    target.assert_called_once_with()
    applied, readback = (call.args[0] for call in psql.call_args_list)
    assert applied == "SET onex.dev_contractor_grants = 'dev';\n" + tracked
    assert readback == runner.readback_sql()


def test_psql_runs_inside_the_dev_container_against_analytics() -> None:
    with patch.object(runner, "run", return_value="") as command:
        runner.psql("SELECT 1;")
    argv = command.call_args.args[0]
    assert argv[:6] == [*runner.DOCKER, "exec", "-i", "omnibase-infra-postgres"]
    assert argv[argv.index("-d") + 1] == "omnidash_analytics"
    assert argv[argv.index("-v") + 1] == "ON_ERROR_STOP=1"
    assert command.call_args.kwargs["body"] == "SELECT 1;"
