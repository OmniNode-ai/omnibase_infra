# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for onex env CLI commands (render-settings, sync-settings)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.cli.cli_env import (
    CheckResult,
    CheckStatus,
    check_config_no_hooks_block,
    check_config_settings_exists,
    check_config_settings_paths,
    check_env_no_duplicates,
    check_topology_plugin_symlink,
    fix_env_no_duplicates,
    fix_no_hooks_block,
    render_settings_json,
    run_checks_for_machine,
    write_settings_local,
    write_settings_remote,
)
from omnibase_infra.models.environment.model_machine_registry import (
    EnumMachineRole,
    ModelMachineEntry,
)

# Canonical output schema. render_settings_json must produce exactly this shape.
EXPECTED_ENV_KEYS = {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "OMNI_HOME",
    "ONEX_STATE_DIR",
    "OMNICLAUDE_MODE",
    "OMNI_INFRA_HOST",
    "MAX_THINKING_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "DISABLE_TELEMETRY",
    "DISABLE_ERROR_REPORTING",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "VALKEY_HOST",
    "VALKEY_PORT",
    "KAFKA_BOOTSTRAP_SERVERS",
}
EXPECTED_TOP_KEYS = {
    "$schema",
    "env",
    "includeCoAuthoredBy",
    "model",
    "statusLine",
    "enabledPlugins",
    "voiceEnabled",
    "skipDangerousModePermissionPrompt",
    "autoUpdates",
}


def test_render_settings_for_dev_machine():
    machine = ModelMachineEntry(
        machine_id="test-dev",
        hostname="test.local",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home="/Users/test/Code/omni_home",
        ssh_user="test",
    )
    result = render_settings_json(machine)
    # Shape check
    assert set(result.keys()) == EXPECTED_TOP_KEYS
    assert set(result["env"].keys()) == EXPECTED_ENV_KEYS
    # Path check
    assert result["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"
    assert result["env"]["ONEX_STATE_DIR"] == "/Users/test/Code/omni_home/.onex_state"
    assert result["statusLine"]["command"].startswith("/Users/test/Code/omni_home/")
    # No empty hooks block
    assert "hooks" not in result
    # Common values (not machine-specific)
    assert result["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
    assert result["env"]["OMNI_INFRA_HOST"] == "192.168.86.201"
    assert result["model"] == "opus[1m]"
    # Unset values are omitted, not null
    for v in result["env"].values():
        assert v is not None


def test_render_settings_for_infra_machine():
    machine = ModelMachineEntry(
        machine_id="infra",
        hostname="srv",
        ip="2.2.2.2",
        role=EnumMachineRole.INFRA,
        omni_home="/data/omninode/omni_home",
        ssh_user="jonah",
        home_dir="/home/jonah",
    )
    result = render_settings_json(machine)
    assert result["env"]["OMNI_HOME"] == "/data/omninode/omni_home"
    assert result["env"]["ONEX_STATE_DIR"] == "/data/omninode/omni_home/.onex_state"
    assert result["statusLine"]["command"].startswith("/data/omninode/")


def test_rendered_json_is_valid():
    """Rendered output must be valid JSON when serialized."""
    machine = ModelMachineEntry(
        machine_id="t",
        hostname="h",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home="/x",
        ssh_user="u",
    )
    result = render_settings_json(machine)
    serialized = json.dumps(result)
    reparsed = json.loads(serialized)
    assert reparsed == result


# ---------------------------------------------------------------------------
# sync-settings: local write tests
# ---------------------------------------------------------------------------


def _make_machine(omni_home: str = "/Users/test/Code/omni_home") -> ModelMachineEntry:
    return ModelMachineEntry(
        machine_id="test-dev",
        hostname="test.local",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home=omni_home,
        ssh_user="test",
    )


def test_sync_writes_local_settings(tmp_path: Path) -> None:
    """Local sync writes to claude_settings_path, backs up existing."""
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"

    # Write pre-existing settings
    old_content = {"old": "settings"}
    settings_file.write_text(json.dumps(old_content))

    machine = _make_machine()
    # Override claude_settings_path to point at tmp_path
    target = str(settings_file)

    write_settings_local(machine, target_path=target, dry_run=False)

    # Backup should exist
    bak = Path(target + ".bak")
    assert bak.exists()
    assert json.loads(bak.read_text()) == old_content

    # New file should be valid JSON with correct OMNI_HOME
    written = json.loads(settings_file.read_text())
    assert written["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"
    assert written["env"]["ONEX_STATE_DIR"] == "/Users/test/Code/omni_home/.onex_state"

    # .tmp should NOT exist (was renamed away)
    assert not Path(target + ".tmp").exists()


def test_sync_creates_parent_dirs(tmp_path: Path) -> None:
    """Local sync creates parent directories if they don't exist."""
    settings_file = tmp_path / "deep" / "nested" / "settings.json"
    machine = _make_machine()

    write_settings_local(machine, target_path=str(settings_file), dry_run=False)

    assert settings_file.exists()
    written = json.loads(settings_file.read_text())
    assert written["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"


def test_sync_no_backup_when_no_existing_file(tmp_path: Path) -> None:
    """No .bak created when there is no existing settings file."""
    settings_file = tmp_path / "settings.json"
    machine = _make_machine()

    write_settings_local(machine, target_path=str(settings_file), dry_run=False)

    assert settings_file.exists()
    assert not Path(str(settings_file) + ".bak").exists()


def test_sync_validates_json_after_write(tmp_path: Path) -> None:
    """After writing, the file must parse as valid JSON with correct OMNI_HOME."""
    settings_file = tmp_path / "settings.json"
    machine = _make_machine()

    write_settings_local(machine, target_path=str(settings_file), dry_run=False)

    content = settings_file.read_text()
    parsed = json.loads(content)
    assert parsed["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"
    assert "$schema" in parsed


def test_dry_run_shows_diff_without_writing(tmp_path: Path) -> None:
    """--dry-run prints what would change but does not modify files."""
    settings_file = tmp_path / "settings.json"
    old_content = {"old": "data"}
    settings_file.write_text(json.dumps(old_content))

    machine = _make_machine()
    diff = write_settings_local(
        machine, target_path=str(settings_file), dry_run=True
    )

    # File should be unchanged
    assert json.loads(settings_file.read_text()) == old_content
    # No backup created
    assert not Path(str(settings_file) + ".bak").exists()
    # No .tmp left
    assert not Path(str(settings_file) + ".tmp").exists()
    # Diff should be returned as a string
    assert diff is not None
    assert isinstance(diff, str)
    assert len(diff) > 0


def test_dry_run_new_file(tmp_path: Path) -> None:
    """--dry-run on a new file returns the full rendered content."""
    settings_file = tmp_path / "settings.json"
    machine = _make_machine()

    diff = write_settings_local(
        machine, target_path=str(settings_file), dry_run=True
    )

    # File should NOT exist
    assert not settings_file.exists()
    assert diff is not None
    assert "OMNI_HOME" in diff


def test_sync_failure_cleans_up_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On failure after .tmp write, .tmp is removed and .bak stays."""
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"
    settings_file.write_text(json.dumps({"old": "data"}))

    machine = _make_machine()
    target = str(settings_file)

    # Monkey-patch os.replace to simulate failure during atomic rename
    import os

    original_replace = os.replace

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_settings_local(machine, target_path=target, dry_run=False)

    # .bak should exist (backup was made before the failure)
    bak = Path(target + ".bak")
    assert bak.exists()
    assert json.loads(bak.read_text()) == {"old": "data"}

    # .tmp should be cleaned up
    assert not Path(target + ".tmp").exists()


# ---------------------------------------------------------------------------
# sync-settings: remote write tests
# ---------------------------------------------------------------------------


def test_write_settings_remote_calls_ssh(tmp_path: Path) -> None:
    """Remote write invokes SSH commands in the correct order."""
    machine = ModelMachineEntry(
        machine_id="infra-server",
        hostname="omninode-infra",
        ip="192.168.86.201",
        role=EnumMachineRole.INFRA,
        omni_home="/data/omninode/omni_home",
        ssh_user="jonah",
        home_dir="/home/jonah",
    )

    ssh_calls: list[str] = []

    def mock_run(cmd: list[str], **kwargs: object) -> object:
        """Capture SSH commands."""
        ssh_calls.append(" ".join(cmd))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    with patch("subprocess.run", side_effect=mock_run):
        write_settings_remote(machine, dry_run=False)

    # Should have called SSH at least for: backup, write tmp, validate, rename
    assert len(ssh_calls) >= 3
    # All calls should target the right host
    for call in ssh_calls:
        assert "jonah@192.168.86.201" in call or "192.168.86.201" in call


def test_write_settings_remote_dry_run_no_ssh() -> None:
    """Remote --dry-run should NOT invoke any SSH commands."""
    machine = ModelMachineEntry(
        machine_id="infra-server",
        hostname="omninode-infra",
        ip="192.168.86.201",
        role=EnumMachineRole.INFRA,
        omni_home="/data/omninode/omni_home",
        ssh_user="jonah",
        home_dir="/home/jonah",
    )

    with patch("subprocess.run") as mock_run:
        result = write_settings_remote(machine, dry_run=True)
        mock_run.assert_not_called()

    assert result is not None
    assert "OMNI_HOME" in result


# ---------------------------------------------------------------------------
# env check: conformance verification tests
# ---------------------------------------------------------------------------


def _make_check_machine(
    omni_home: str = "/Users/test/Code/omni_home",
) -> ModelMachineEntry:
    return ModelMachineEntry(
        machine_id="test-dev",
        hostname="test.local",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home=omni_home,
        ssh_user="test",
    )


class TestCheckResult:
    """CheckResult model basics."""

    def test_check_result_pass(self) -> None:
        r = CheckResult(
            check_id="config.settings_exists",
            status=CheckStatus.PASS,
            detail="File exists",
            fixable=False,
        )
        assert r.status == CheckStatus.PASS
        assert r.fixable is False

    def test_check_result_fail(self) -> None:
        r = CheckResult(
            check_id="config.settings_paths",
            status=CheckStatus.FAIL,
            detail="OMNI_HOME mismatch",
            fixable=True,
        )
        assert r.status == CheckStatus.FAIL
        assert r.fixable is True

    def test_check_result_serializes(self) -> None:
        r = CheckResult(
            check_id="env.no_duplicates",
            status=CheckStatus.WARN,
            detail="1 duplicate key",
            fixable=True,
        )
        d = r.to_dict()
        assert d["status"] == "WARN"
        assert d["fixable"] is True


class TestCheckConfigSettingsExists:
    """config.settings_exists check."""

    def test_pass_when_file_exists(self, tmp_path: Path) -> None:
        settings = tmp_path / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text("{}")
        machine = _make_check_machine()
        result = check_config_settings_exists(machine, settings_path=str(settings))
        assert result.status == CheckStatus.PASS

    def test_fail_when_file_missing(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        result = check_config_settings_exists(
            machine, settings_path=str(tmp_path / "nonexistent.json")
        )
        assert result.status == CheckStatus.FAIL
        assert result.fixable is False


class TestCheckConfigSettingsPaths:
    """config.settings_paths check."""

    def test_pass_when_paths_match(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = render_settings_json(machine)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        result = check_config_settings_paths(
            machine, settings_path=str(settings_file)
        )
        assert result.status == CheckStatus.PASS

    def test_fail_when_omni_home_wrong(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = render_settings_json(machine)
        settings["env"]["OMNI_HOME"] = "/wrong/path"
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        result = check_config_settings_paths(
            machine, settings_path=str(settings_file)
        )
        assert result.status == CheckStatus.FAIL
        assert result.fixable is True
        assert "OMNI_HOME" in result.detail

    def test_fail_when_statusline_wrong(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = render_settings_json(machine)
        settings["statusLine"]["command"] = "/wrong/statusline.sh"
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        result = check_config_settings_paths(
            machine, settings_path=str(settings_file)
        )
        assert result.status == CheckStatus.FAIL
        assert "statusLine" in result.detail

    def test_skip_when_file_missing(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        result = check_config_settings_paths(
            machine, settings_path=str(tmp_path / "nope.json")
        )
        assert result.status == CheckStatus.SKIP


class TestCheckConfigNoHooksBlock:
    """config.no_hooks_block check."""

    def test_pass_when_no_hooks(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = {"env": {}, "model": "opus"}
        f = tmp_path / "settings.json"
        f.write_text(json.dumps(settings))
        result = check_config_no_hooks_block(machine, settings_path=str(f))
        assert result.status == CheckStatus.PASS

    def test_fail_when_empty_hooks(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = {"env": {}, "hooks": {}}
        f = tmp_path / "settings.json"
        f.write_text(json.dumps(settings))
        result = check_config_no_hooks_block(machine, settings_path=str(f))
        assert result.status == CheckStatus.FAIL
        assert result.fixable is True

    def test_pass_when_hooks_has_content(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = {"env": {}, "hooks": {"PreToolUse": [{"command": "test"}]}}
        f = tmp_path / "settings.json"
        f.write_text(json.dumps(settings))
        result = check_config_no_hooks_block(machine, settings_path=str(f))
        assert result.status == CheckStatus.PASS

    def test_fix_removes_empty_hooks(self, tmp_path: Path) -> None:
        settings = {"env": {}, "hooks": {}}
        f = tmp_path / "settings.json"
        f.write_text(json.dumps(settings))
        fix_no_hooks_block(str(f))
        result = json.loads(f.read_text())
        assert "hooks" not in result
        assert Path(str(f) + ".bak").exists()


class TestCheckEnvNoDuplicates:
    """env.no_duplicates check."""

    def test_pass_when_no_duplicates(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=1\nBAR=2\n")
        machine = _make_check_machine()
        result = check_env_no_duplicates(machine, env_path=str(env_file))
        assert result.status == CheckStatus.PASS

    def test_fail_when_duplicates_exist(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=1\nBAR=2\nFOO=3\n")
        machine = _make_check_machine()
        result = check_env_no_duplicates(machine, env_path=str(env_file))
        assert result.status == CheckStatus.FAIL
        assert result.fixable is True
        assert "FOO" in result.detail

    def test_pass_when_file_missing(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        result = check_env_no_duplicates(
            machine, env_path=str(tmp_path / "nope.env")
        )
        assert result.status == CheckStatus.PASS
        assert "not found" in result.detail.lower()

    def test_fix_deduplicates_last_wins(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=1\nBAR=2\nFOO=3\nBAZ=4\nBAR=5\n")
        fix_env_no_duplicates(str(env_file))
        # Backup created
        assert Path(str(env_file) + ".bak").exists()
        # Read result
        lines = env_file.read_text().strip().split("\n")
        keys = [l.split("=", 1)[0] for l in lines if "=" in l]
        # No duplicate keys
        assert len(keys) == len(set(keys))
        # Last-wins: FOO=3, BAR=5
        env_dict = dict(l.split("=", 1) for l in lines if "=" in l)
        assert env_dict["FOO"] == "3"
        assert env_dict["BAR"] == "5"
        assert env_dict["BAZ"] == "4"

    def test_fix_preserves_comments_and_blanks(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nFOO=1\n\nFOO=2\n# tail\n")
        fix_env_no_duplicates(str(env_file))
        content = env_file.read_text()
        assert "# comment" in content
        assert "# tail" in content


class TestCheckTopologyPluginSymlink:
    """topology.plugin_symlink check."""

    def test_pass_when_symlink(self, tmp_path: Path) -> None:
        machine = _make_check_machine(omni_home=str(tmp_path / "omni_home"))
        canonical = tmp_path / "omni_home" / "omniclaude" / "plugins" / "onex"
        canonical.mkdir(parents=True)
        cache_parent = tmp_path / "cache"
        cache_parent.mkdir()
        link = cache_parent / "onex"
        link.symlink_to(canonical)
        result = check_topology_plugin_symlink(
            machine, plugin_cache_path=str(link)
        )
        assert result.status == CheckStatus.PASS

    def test_fail_when_regular_dir(self, tmp_path: Path) -> None:
        machine = _make_check_machine(omni_home=str(tmp_path / "omni_home"))
        cache_dir = tmp_path / "cache" / "onex"
        cache_dir.mkdir(parents=True)
        result = check_topology_plugin_symlink(
            machine, plugin_cache_path=str(cache_dir)
        )
        assert result.status == CheckStatus.FAIL
        assert result.fixable is True

    def test_pass_when_no_cache(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        result = check_topology_plugin_symlink(
            machine, plugin_cache_path=str(tmp_path / "nonexistent")
        )
        assert result.status == CheckStatus.PASS


class TestRunChecksForMachine:
    """Integration: run_checks_for_machine returns structured output."""

    def test_output_structure(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = render_settings_json(machine)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=1\n")

        results = run_checks_for_machine(
            machine,
            settings_path=str(settings_file),
            env_path=str(env_file),
            plugin_cache_path=str(tmp_path / "nonexistent"),
            skip_secrets=True,
        )
        # Should have results for all non-secrets checks
        assert "config.settings_exists" in results
        assert "config.settings_paths" in results
        assert "config.no_hooks_block" in results
        assert "env.no_duplicates" in results
        assert "topology.plugin_symlink" in results
        # Each result has required keys
        for check_id, result in results.items():
            assert "status" in result
            assert "detail" in result
            assert "fixable" in result

    def test_all_pass_on_correct_setup(self, tmp_path: Path) -> None:
        machine = _make_check_machine()
        settings = render_settings_json(machine)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=1\n")

        results = run_checks_for_machine(
            machine,
            settings_path=str(settings_file),
            env_path=str(env_file),
            plugin_cache_path=str(tmp_path / "nonexistent"),
            skip_secrets=True,
        )
        for check_id, result in results.items():
            assert result["status"] in ("PASS", "SKIP"), (
                f"{check_id}: {result}"
            )
