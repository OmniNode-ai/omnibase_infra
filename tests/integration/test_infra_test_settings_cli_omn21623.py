# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the CLI, YAML loader and command timeouts across external boundaries."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.infra_test import env as env_module
from omnibase_infra.cli.infra_test.cli import cli
from omnibase_infra.models.infra_test import load_infra_test_settings

settings_module = importlib.import_module(
    "omnibase_infra.models.infra_test.model_infra_test_settings"
)
suite_module = importlib.import_module("omnibase_infra.cli.infra_test.run_suite")

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("command", "timeout_fields", "connect_count", "sleep_fields"),
    [
        (
            ["env", "up"],
            ["compose_command", "docker_query", "compose_command"],
            0,
            [],
        ),
        (["env", "down"], ["compose_command"], 0, []),
        (["env", "down", "--no-volumes"], ["compose_command"], 0, []),
        (["introspect"], ["rpk_command"], 0, []),
        (["verify", "topics"], ["rpk_command"], 0, []),
        (["verify", "snapshots"], ["rpk_command", "rpk_consume"], 0, []),
        (["verify", "registry"], [], 1, []),
        (
            ["verify", "idempotency"],
            ["rpk_command"] * 3,
            1,
            ["idempotency_settle_seconds"],
        ),
        (
            ["run", "--suite", "smoke"],
            ["docker_query", "rpk_command"] + ["cli_subcommand"] * 3,
            1,
            [],
        ),
        (
            ["run", "--suite", "idempotency"],
            ["docker_query"] + ["rpk_command"] * 3,
            1,
            ["idempotency_settle_seconds"],
        ),
        (
            ["run", "--suite", "failure"],
            [
                "docker_query",
                "rpk_command",
                "docker_query",
                "rpk_command",
                "compose_start",
            ],
            3,
            ["runtime_restart_wait_seconds"],
        ),
    ],
)
def test_cli_commands_use_loaded_settings(
    command: list[str],
    timeout_fields: list[str],
    connect_count: int,
    sleep_fields: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Give every numeric setting a distinct value, different from the shipped
    # defaults. A call using a literal or the wrong field must fail this check.
    document = yaml.safe_load(settings_module._SETTINGS_PATH.read_text())
    block = document["settings"]
    for index, key in enumerate(block):
        if isinstance(block[key], (int, float)):
            block[key] = index + 41
    block["broker_address"] = "settings.example.invalid:19092"
    block["runtime_url"] = "http://runtime.example.invalid:8080"
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(settings_module, "_SETTINGS_PATH", settings_path)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_RUNTIME_URL", raising=False)
    monkeypatch.delenv("ONEX_INTROSPECTION_TOPIC", raising=False)
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://test:test@db.example.invalid/test"
    )
    load_infra_test_settings.cache_clear()

    def external_command(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["rpk", "topic", "list"]:
            stdout = "[]"
        elif args[0] == "docker":
            stdout = "runtime\thealthy" if "--format" in args else "runtime"
        else:
            stdout = "{}"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (1,)
    connection.__enter__.return_value = connection
    cursor.fetchall.return_value = []
    try:
        with (
            patch("subprocess.run", side_effect=external_command) as run,
            patch("psycopg2.connect", return_value=connection) as connect,
            patch("time.sleep") as sleep,
            patch.object(
                suite_module,
                "_wait_for_registration",
                wraps=suite_module._wait_for_registration,
            ) as registration_wait,
        ):
            result = CliRunner().invoke(
                cli,
                ["--project-name", "landing-l19772-lab", *command],
            )
        assert result.exit_code == 0, (result.output, result.exception)
        assert [call.kwargs["timeout"] for call in run.call_args_list] == [
            block[f"{field}_timeout_seconds"] for field in timeout_fields
        ]
        if command[:2] == ["env", "down"]:
            expected_args = ["down"] if "--no-volumes" in command else ["down", "-v"]
            assert run.call_args.args[0][6:] == expected_args
        assert connect.call_count == connect_count
        for call in connect.call_args_list:
            assert (
                call.kwargs["connect_timeout"]
                == block["postgres_connect_timeout_seconds"]
            )
        assert [call.args[0] for call in sleep.call_args_list] == [
            block[field] for field in sleep_fields
        ]
        expected_waits = []
        if command == ["run", "--suite", "smoke"]:
            expected_waits = [block["registration_wait_seconds"]]
        elif command == ["run", "--suite", "failure"]:
            expected_waits = [
                block["registration_wait_seconds"],
                block["registration_backfill_wait_seconds"],
            ]
        assert [
            call.kwargs["timeout"] for call in registration_wait.call_args_list
        ] == expected_waits
        if command == ["introspect"]:
            args = run.call_args.args[0]
            assert args[args.index("--brokers") + 1] == block["broker_address"]
            payload = json.loads(run.call_args.kwargs["input"])
            assert payload["endpoints"]["health"].startswith(block["runtime_url"] + "/")
    finally:
        load_infra_test_settings.cache_clear()


@pytest.mark.parametrize("contents", [None, "settings: [invalid", "settings: {}"])
def test_import_and_help_do_not_load_settings(
    contents: str | None, tmp_path: Path
) -> None:
    """A fresh import and help remain available when configuration is unavailable."""
    settings_path = tmp_path / "settings.yaml"
    if contents is not None:
        settings_path.write_text(contents, encoding="utf-8")
    with patch.object(settings_module, "_SETTINGS_PATH", settings_path):
        load_infra_test_settings.cache_clear()
        try:
            # Re-execute decorators as a fresh process would, while preserving the
            # original command objects used by the other tests.
            import runpy

            namespace = runpy.run_path(str(env_module.__file__))
            result = CliRunner().invoke(namespace["env"], ["up", "--help"])
            assert result.exit_code == 0, result.output
            assert "--timeout" in result.output
            assert load_infra_test_settings.cache_info().misses == 0
        finally:
            load_infra_test_settings.cache_clear()
