# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Environment management CLI commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import click

from omnibase_infra.models.environment.model_machine_registry import (
    ModelMachineEntry,
    ModelMachineRegistry,
)

# ---------------------------------------------------------------------------
# Fleet-wide constants (same for all machines)
# ---------------------------------------------------------------------------
_SCHEMA_URL = "https://json.schemastore.org/claude-code-settings.json"
_OMNI_INFRA_HOST = "192.168.86.201"
_KAFKA_BOOTSTRAP_SERVERS = "192.168.86.201:19092"
_POSTGRES_HOST = "192.168.86.201"
_POSTGRES_PORT = "5436"
_VALKEY_HOST = "192.168.86.201"
_VALKEY_PORT = "16379"
_MODEL = "opus[1m]"


def render_settings_json(machine: ModelMachineEntry) -> dict[str, Any]:
    """Produce the canonical settings.json shape for a machine.

    Machine-specific values (paths) come from the registry entry.
    Common values (infra host, Kafka, Postgres, etc.) are fleet-wide constants.
    """
    return {
        "$schema": _SCHEMA_URL,
        "env": {
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
            "OMNI_HOME": machine.omni_home,
            "ONEX_STATE_DIR": machine.onex_state_dir,
            "OMNICLAUDE_MODE": "full",
            "OMNI_INFRA_HOST": _OMNI_INFRA_HOST,
            "MAX_THINKING_TOKENS": "31999",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "POSTGRES_HOST": _POSTGRES_HOST,
            "POSTGRES_PORT": _POSTGRES_PORT,
            "VALKEY_HOST": _VALKEY_HOST,
            "VALKEY_PORT": _VALKEY_PORT,
            "KAFKA_BOOTSTRAP_SERVERS": _KAFKA_BOOTSTRAP_SERVERS,
        },
        "includeCoAuthoredBy": False,
        "model": _MODEL,
        "statusLine": {
            "type": "command",
            "command": machine.statusline_path,
            "padding": 0,
        },
        "enabledPlugins": {
            "code-review@claude-plugins-official": True,
            "onex@omninode-tools": True,
        },
        "voiceEnabled": True,
        "skipDangerousModePermissionPrompt": True,
        "autoUpdates": True,
    }


# ---------------------------------------------------------------------------
# CLI command group
# ---------------------------------------------------------------------------


@click.group("env")
def env_group() -> None:  # stub-ok: click group
    """Environment management commands."""


@env_group.command("render-settings")
@click.option("--machine-id", required=True, help="Machine ID from the registry")
@click.option(
    "--registry",
    default="config/machines.yaml",
    help="Path to machines.yaml registry file",
)
def render_settings(machine_id: str, registry: str) -> None:
    """Render the canonical settings.json for a machine."""
    reg = ModelMachineRegistry.from_yaml(Path(registry))
    machine = reg.get_machine(machine_id)
    settings = render_settings_json(machine)
    click.echo(json.dumps(settings, indent=2))


# ---------------------------------------------------------------------------
# sync-settings: atomic write helpers
# ---------------------------------------------------------------------------


def _diff_settings(old_content: str | None, new_content: str) -> str:
    """Produce a human-readable diff between old and new settings."""
    if old_content is None:
        return f"[new file]\n{new_content}"
    if old_content == new_content:
        return "[no changes]"
    # Simple line-by-line diff
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    import difflib

    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile="current", tofile="rendered"
    )
    return "".join(diff)


def write_settings_local(
    machine: ModelMachineEntry,
    *,
    target_path: str | None = None,
    dry_run: bool = False,
) -> str | None:
    """Write rendered settings.json to a local path with atomic rename.

    Strategy:
      1. Render settings JSON
      2. If existing file: copy to settings.json.bak
      3. Write to settings.json.tmp
      4. Validate settings.json.tmp parses as valid JSON
      5. Atomic rename settings.json.tmp -> settings.json
      On failure: leave .bak intact, remove .tmp

    Args:
        machine: Machine entry to render settings for.
        target_path: Override path (defaults to machine.claude_settings_path).
        dry_run: If True, return diff string without writing.

    Returns:
        Diff string if dry_run, None otherwise.
    """
    target = Path(target_path or machine.claude_settings_path)
    settings = render_settings_json(machine)
    new_content = json.dumps(settings, indent=2) + "\n"

    # Read existing content for diff
    old_content: str | None = None
    if target.exists():
        old_content = target.read_text()

    if dry_run:
        return _diff_settings(old_content, new_content)

    # Ensure parent directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(str(target) + ".tmp")
    bak_path = Path(str(target) + ".bak")

    try:
        # Step 1: Backup existing file
        if target.exists():
            shutil.copy2(str(target), str(bak_path))

        # Step 2: Write to .tmp
        tmp_path.write_text(new_content)

        # Step 3: Validate .tmp is valid JSON
        json.loads(tmp_path.read_text())

        # Step 4: Atomic rename .tmp -> target
        os.replace(str(tmp_path), str(target))

    except Exception:
        # Cleanup: remove .tmp if it exists, leave .bak intact
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return None


def write_settings_remote(
    machine: ModelMachineEntry,
    *,
    dry_run: bool = False,
) -> str | None:
    """Write rendered settings.json to a remote machine via SSH.

    Strategy:
      1. Render settings JSON
      2. SSH: copy existing to settings.json.bak
      3. SSH: write to settings.json.tmp via heredoc
      4. SSH: validate with python3 -c on remote
      5. SSH: atomic mv settings.json.tmp settings.json
      On failure: leave .bak in place, remove .tmp, report error

    Args:
        machine: Machine entry to render settings for.
        dry_run: If True, return rendered content without SSH.

    Returns:
        Rendered content string if dry_run, None otherwise.
    """
    settings = render_settings_json(machine)
    new_content = json.dumps(settings, indent=2) + "\n"

    if dry_run:
        return f"[would write to {machine.ssh_user}@{machine.ip}:{machine.claude_settings_path}]\n{new_content}"

    remote = f"{machine.ssh_user}@{machine.ip}"
    target = machine.claude_settings_path
    tmp_target = f"{target}.tmp"
    bak_target = f"{target}.bak"
    ssh_base = ["ssh", "-o", "ConnectTimeout=10", "-p", str(machine.ssh_port), remote]

    def _ssh(cmd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*ssh_base, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        # Step 1: Ensure parent directory exists
        _ssh(f"mkdir -p $(dirname {target})")

        # Step 2: Backup existing file (ignore error if file doesn't exist)
        _ssh(f"[ -f {target} ] && cp -p {target} {bak_target} || true")

        # Step 3: Write to .tmp via cat heredoc
        # Use base64 to avoid heredoc escaping issues
        import base64

        encoded = base64.b64encode(new_content.encode()).decode()
        result = _ssh(f"echo '{encoded}' | base64 -d > {tmp_target}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write .tmp on remote: {result.stderr}")

        # Step 4: Validate JSON on remote
        result = _ssh(f"python3 -c \"import json; json.load(open('{tmp_target}'))\"")
        if result.returncode != 0:
            raise RuntimeError(f"JSON validation failed on remote: {result.stderr}")

        # Step 5: Atomic rename
        result = _ssh(f"mv {tmp_target} {target}")
        if result.returncode != 0:
            raise RuntimeError(f"Atomic rename failed on remote: {result.stderr}")

    except Exception:
        # Cleanup: remove .tmp on remote, leave .bak intact
        try:
            _ssh(f"rm -f {tmp_target}")
        except Exception:
            pass
        raise

    return None


# ---------------------------------------------------------------------------
# sync-settings CLI command
# ---------------------------------------------------------------------------


@env_group.command("sync-settings")
@click.option("--machine-id", default=None, help="Sync to specific machine")
@click.option("--all", "sync_all", is_flag=True, help="Sync to all machines")
@click.option(
    "--registry",
    default="config/machines.yaml",
    help="Path to machines.yaml registry file",
)
@click.option("--dry-run", is_flag=True, help="Show diff without writing")
def sync_settings(
    machine_id: str | None,
    sync_all: bool,
    registry: str,
    dry_run: bool,
) -> None:
    """Sync rendered settings.json to one or all machines.

    Performs atomic writes: write .tmp, validate, rename to final.
    Backs up existing settings.json to .bak before overwrite.
    Locality resolved at runtime via resolve_local_machine().
    """
    reg = ModelMachineRegistry.from_yaml(Path(registry))

    if not machine_id and not sync_all:
        raise click.UsageError("Specify --machine-id or --all")

    local = reg.resolve_local_machine()
    targets = reg.machines if sync_all else [reg.get_machine(machine_id)]  # type: ignore[arg-type]

    for machine in targets:
        is_local = local is not None and machine.machine_id == local.machine_id
        label = f"{machine.machine_id} ({'local' if is_local else 'remote'})"

        if dry_run:
            click.echo(f"--- {label} ---")

        try:
            if is_local:
                result = write_settings_local(machine, dry_run=dry_run)
            else:
                result = write_settings_remote(machine, dry_run=dry_run)

            if dry_run and result:
                click.echo(result)
            elif not dry_run:
                click.echo(f"[ok] {label}: settings.json written")

        except Exception as exc:
            click.echo(f"[FAIL] {label}: {exc}", err=True)
            if not sync_all:
                raise SystemExit(1)


# ---------------------------------------------------------------------------
# env check: conformance verification
# ---------------------------------------------------------------------------


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: CheckStatus
    detail: str
    fixable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "detail": self.detail,
            "fixable": self.fixable,
        }


# --- Individual check functions ---


def check_config_settings_exists(
    machine: ModelMachineEntry,
    *,
    settings_path: str | None = None,
) -> CheckResult:
    """Check that settings.json exists at expected path."""
    path = settings_path or machine.claude_settings_path
    if Path(path).exists():
        return CheckResult(
            check_id="config.settings_exists",
            status=CheckStatus.PASS,
            detail=f"File exists: {path}",
            fixable=False,
        )
    return CheckResult(
        check_id="config.settings_exists",
        status=CheckStatus.FAIL,
        detail=f"File missing: {path}",
        fixable=False,
    )


def check_config_settings_paths(
    machine: ModelMachineEntry,
    *,
    settings_path: str | None = None,
) -> CheckResult:
    """Check OMNI_HOME, ONEX_STATE_DIR, and statusLine path match registry."""
    path = Path(settings_path or machine.claude_settings_path)
    if not path.exists():
        return CheckResult(
            check_id="config.settings_paths",
            status=CheckStatus.SKIP,
            detail="Settings file not found, cannot check paths",
            fixable=True,
        )

    try:
        current = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return CheckResult(
            check_id="config.settings_paths",
            status=CheckStatus.FAIL,
            detail=f"Cannot parse settings: {exc}",
            fixable=True,
        )

    expected = render_settings_json(machine)
    mismatches: list[str] = []

    # Check env paths
    current_env = current.get("env", {})
    for key in ("OMNI_HOME", "ONEX_STATE_DIR"):
        actual = current_env.get(key)
        want = expected["env"][key]
        if actual != want:
            mismatches.append(f"{key}: {actual!r} != {want!r}")

    # Check statusLine command
    current_sl = current.get("statusLine", {}).get("command")
    expected_sl = expected["statusLine"]["command"]
    if current_sl != expected_sl:
        mismatches.append(f"statusLine.command: {current_sl!r} != {expected_sl!r}")

    if mismatches:
        return CheckResult(
            check_id="config.settings_paths",
            status=CheckStatus.FAIL,
            detail="; ".join(mismatches),
            fixable=True,
        )
    return CheckResult(
        check_id="config.settings_paths",
        status=CheckStatus.PASS,
        detail="All paths match registry",
        fixable=True,
    )


def check_config_no_hooks_block(
    machine: ModelMachineEntry,
    *,
    settings_path: str | None = None,
) -> CheckResult:
    """Check that settings.json has no empty hooks block."""
    path = Path(settings_path or machine.claude_settings_path)
    if not path.exists():
        return CheckResult(
            check_id="config.no_hooks_block",
            status=CheckStatus.SKIP,
            detail="Settings file not found",
            fixable=True,
        )

    try:
        current = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return CheckResult(
            check_id="config.no_hooks_block",
            status=CheckStatus.SKIP,
            detail="Cannot parse settings",
            fixable=True,
        )

    hooks = current.get("hooks")
    if hooks is not None and not hooks:
        return CheckResult(
            check_id="config.no_hooks_block",
            status=CheckStatus.FAIL,
            detail="Empty hooks block found in settings.json",
            fixable=True,
        )
    return CheckResult(
        check_id="config.no_hooks_block",
        status=CheckStatus.PASS,
        detail="No empty hooks block",
        fixable=True,
    )


def check_env_no_duplicates(
    machine: ModelMachineEntry,
    *,
    env_path: str | None = None,
) -> CheckResult:
    """Check for duplicate keys in .env file."""
    path = Path(env_path or f"{machine.resolved_home_dir}/.omnibase/.env")
    if not path.exists():
        return CheckResult(
            check_id="env.no_duplicates",
            status=CheckStatus.PASS,
            detail=f".env not found at {path} (nothing to check)",
            fixable=True,
        )

    lines = path.read_text().splitlines()
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0]
        if key in seen:
            if key not in duplicates:
                duplicates.append(key)
        seen[key] = seen.get(key, 0) + 1

    if duplicates:
        return CheckResult(
            check_id="env.no_duplicates",
            status=CheckStatus.FAIL,
            detail=f"Duplicate keys: {', '.join(duplicates)}",
            fixable=True,
        )
    return CheckResult(
        check_id="env.no_duplicates",
        status=CheckStatus.PASS,
        detail="No duplicate keys",
        fixable=True,
    )


def check_secrets_infisical_identity(
    machine: ModelMachineEntry,
) -> CheckResult:
    """Check Infisical client ID authenticates (report-only, not auto-fixable)."""
    # This check requires network access; we report SKIP if not testable
    infisical_host = f"http://{_OMNI_INFRA_HOST}:8880"
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{infisical_host}/api/v1/auth/universal-auth/login",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "clientId": os.environ.get("INFISICAL_CLIENT_ID", ""),
                    "clientSecret": os.environ.get("INFISICAL_CLIENT_SECRET", ""),
                }
            ).encode(),
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            if "accessToken" in body:
                return CheckResult(
                    check_id="secrets.infisical_identity",
                    status=CheckStatus.PASS,
                    detail="Infisical identity authenticated",
                    fixable=False,
                )
    except Exception as exc:
        return CheckResult(
            check_id="secrets.infisical_identity",
            status=CheckStatus.FAIL,
            detail=f"Infisical auth failed: {exc}",
            fixable=False,
        )

    return CheckResult(
        check_id="secrets.infisical_identity",
        status=CheckStatus.FAIL,
        detail="Infisical auth returned no accessToken",
        fixable=False,
    )


def check_topology_plugin_symlink(
    machine: ModelMachineEntry,
    *,
    plugin_cache_path: str | None = None,
) -> CheckResult:
    """Check plugin cache is a symlink to canonical clone, not stale copy."""
    if plugin_cache_path is None:
        plugin_cache_path = (
            f"{machine.resolved_home_dir}/.claude/plugins/onex@omninode-tools"
        )

    cache = Path(plugin_cache_path)
    if not cache.exists():
        return CheckResult(
            check_id="topology.plugin_symlink",
            status=CheckStatus.PASS,
            detail=f"Plugin cache not present at {plugin_cache_path} (ok)",
            fixable=True,
        )

    if cache.is_symlink():
        return CheckResult(
            check_id="topology.plugin_symlink",
            status=CheckStatus.PASS,
            detail=f"Plugin cache is symlink -> {os.readlink(str(cache))}",
            fixable=True,
        )

    return CheckResult(
        check_id="topology.plugin_symlink",
        status=CheckStatus.FAIL,
        detail=f"Plugin cache at {plugin_cache_path} is a regular directory, not symlink",
        fixable=True,
    )


# --- Fix functions ---


def fix_no_hooks_block(settings_path: str) -> None:
    """Remove empty hooks block from settings.json, creating .bak first."""
    path = Path(settings_path)
    bak = Path(settings_path + ".bak")
    shutil.copy2(str(path), str(bak))
    data = json.loads(path.read_text())
    if "hooks" in data and not data["hooks"]:
        del data["hooks"]
    path.write_text(json.dumps(data, indent=2) + "\n")


def fix_env_no_duplicates(env_path: str) -> None:
    """Deduplicate .env keeping last occurrence of each key. Creates .bak."""
    path = Path(env_path)
    bak = Path(env_path + ".bak")
    shutil.copy2(str(path), str(bak))

    lines = path.read_text().splitlines()
    # First pass: find last occurrence index of each key
    last_index: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0]
        last_index[key] = i

    # Second pass: keep comments/blanks and only the last occurrence of each key
    seen_keys: set[str] = set()
    output_lines: list[str] = []
    # Build a set of indices to keep (the last occurrence of each key)
    keep_indices = set(last_index.values())
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue
        if "=" not in stripped:
            output_lines.append(line)
            continue
        if i in keep_indices:
            output_lines.append(line)

    path.write_text("\n".join(output_lines) + "\n")


def fix_topology_plugin_symlink(
    machine: ModelMachineEntry, plugin_cache_path: str
) -> None:
    """Replace regular directory with symlink to canonical plugin path. Creates .bak."""
    cache = Path(plugin_cache_path)
    bak = Path(plugin_cache_path + ".bak")
    if cache.exists() and not cache.is_symlink():
        shutil.move(str(cache), str(bak))
    canonical = machine.plugin_path
    cache.symlink_to(canonical)


# --- Orchestrator ---


def run_checks_for_machine(
    machine: ModelMachineEntry,
    *,
    settings_path: str | None = None,
    env_path: str | None = None,
    plugin_cache_path: str | None = None,
    skip_secrets: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all conformance checks for a machine. Returns {check_id: {status, detail, fixable}}."""
    results: dict[str, dict[str, Any]] = {}

    checks = [
        check_config_settings_exists(machine, settings_path=settings_path),
        check_config_settings_paths(machine, settings_path=settings_path),
        check_config_no_hooks_block(machine, settings_path=settings_path),
        check_env_no_duplicates(machine, env_path=env_path),
        check_topology_plugin_symlink(machine, plugin_cache_path=plugin_cache_path),
    ]

    if not skip_secrets:
        checks.append(check_secrets_infisical_identity(machine))

    for result in checks:
        results[result.check_id] = result.to_dict()

    return results


def _apply_fixes_for_machine(
    machine: ModelMachineEntry,
    check_results: dict[str, dict[str, Any]],
    *,
    settings_path: str | None = None,
    env_path: str | None = None,
    plugin_cache_path: str | None = None,
) -> list[str]:
    """Apply auto-fixes for failed fixable checks. Returns list of fixed check_ids."""
    fixed: list[str] = []
    sp = settings_path or machine.claude_settings_path
    ep = env_path or f"{machine.resolved_home_dir}/.omnibase/.env"
    pp = (
        plugin_cache_path
        or f"{machine.resolved_home_dir}/.claude/plugins/onex@omninode-tools"
    )

    for check_id, result in check_results.items():
        if result["status"] != "FAIL" or not result["fixable"]:
            continue

        if check_id == "config.settings_paths":
            write_settings_local(machine, target_path=sp, dry_run=False)
            fixed.append(check_id)
        elif check_id == "config.no_hooks_block":
            if Path(sp).exists():
                fix_no_hooks_block(sp)
                fixed.append(check_id)
        elif check_id == "env.no_duplicates":
            if Path(ep).exists():
                fix_env_no_duplicates(ep)
                fixed.append(check_id)
        elif check_id == "topology.plugin_symlink":
            fix_topology_plugin_symlink(machine, pp)
            fixed.append(check_id)

    return fixed


# ---------------------------------------------------------------------------
# env check CLI command
# ---------------------------------------------------------------------------


@env_group.command("check")
@click.option(
    "--registry",
    default="config/machines.yaml",
    help="Path to machines.yaml registry file",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Auto-fix fixable issues (local only unless --fix-remote)",
)
@click.option("--fix-remote", is_flag=True, help="Allow auto-fix on remote machines")
@click.option("--machine-id", default=None, help="Check specific machine only")
@click.option("--skip-secrets", is_flag=True, help="Skip Infisical identity check")
def check_env(
    registry: str,
    fix: bool,
    fix_remote: bool,
    machine_id: str | None,
    skip_secrets: bool,
) -> None:
    """Run conformance checks across machines.

    Output structured as {machine_id: {check_id: {status, detail, fixable}}}.
    --fix only applies to auto-fixable checks, local only by default.
    --fix-remote required for remote machines (explicit opt-in).
    Every fix creates .bak backup.
    """
    reg = ModelMachineRegistry.from_yaml(Path(registry))
    local = reg.resolve_local_machine()
    targets = [reg.get_machine(machine_id)] if machine_id else reg.machines

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for machine in targets:
        is_local = local is not None and machine.machine_id == local.machine_id
        results = run_checks_for_machine(machine, skip_secrets=skip_secrets)
        all_results[machine.machine_id] = results

        # Apply fixes if requested and allowed
        if fix and (is_local or fix_remote):
            fixed = _apply_fixes_for_machine(machine, results)
            for check_id in fixed:
                results[check_id]["status"] = "FIXED"
                results[check_id]["detail"] += " [auto-fixed]"
        elif fix and not is_local:
            for check_id, result in results.items():
                if result["status"] == "FAIL" and result["fixable"]:
                    result["detail"] += " [use --fix-remote to auto-fix]"

    click.echo(json.dumps(all_results, indent=2))


# ---------------------------------------------------------------------------
# env bootstrap: one-command machine setup
# ---------------------------------------------------------------------------


class BootstrapStepStatus(StrEnum):
    VERIFIED = "verified"
    CREATED = "created"
    SKIPPED = "skipped"
    WARNED = "warned"
    FAILED = "failed"


@dataclass(frozen=True)
class BootstrapStepResult:
    step: str
    status: BootstrapStepStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"step": self.step, "status": str(self.status), "detail": self.detail}


def _step_ssh_connectivity(
    machine: ModelMachineEntry,
    *,
    ssh_runner: Any | None = None,
) -> BootstrapStepResult:
    """Step 1: Verify SSH connectivity to machine."""
    step = "ssh_connectivity"
    runner = ssh_runner or _default_ssh_runner
    try:
        rc, stdout, stderr = runner(machine, "echo ok")
        if rc == 0 and "ok" in stdout:
            return BootstrapStepResult(
                step, BootstrapStepStatus.VERIFIED, "SSH connection ok"
            )
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"SSH returned rc={rc}: {stderr}"
        )
    except Exception as exc:
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"SSH failed: {exc}"
        )


def _default_ssh_runner(
    machine: ModelMachineEntry,
    cmd: str,
) -> tuple[int, str, str]:
    """Run a command on a remote machine via SSH. Returns (rc, stdout, stderr)."""
    ssh_base = [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        "-p",
        str(machine.ssh_port),
        f"{machine.ssh_user}@{machine.ip}",
    ]
    result = subprocess.run(
        [*ssh_base, cmd],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _step_omni_home_exists(
    machine: ModelMachineEntry,
    *,
    is_local: bool = False,
    ssh_runner: Any | None = None,
) -> BootstrapStepResult:
    """Step 2: Verify omni_home directory exists on the machine."""
    step = "omni_home_exists"
    if is_local:
        if Path(machine.omni_home).is_dir():
            return BootstrapStepResult(
                step, BootstrapStepStatus.VERIFIED, f"{machine.omni_home} exists"
            )
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"{machine.omni_home} not found"
        )

    runner = ssh_runner or _default_ssh_runner
    try:
        rc, stdout, _stderr = runner(
            machine, f"test -d {machine.omni_home} && echo exists"
        )
        if rc == 0 and "exists" in stdout:
            return BootstrapStepResult(
                step, BootstrapStepStatus.VERIFIED, f"{machine.omni_home} exists"
            )
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"{machine.omni_home} not found on remote"
        )
    except Exception as exc:
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"Check failed: {exc}"
        )


def _step_write_env(
    machine: ModelMachineEntry,
    *,
    is_local: bool = False,
    env_template_path: str | None = None,
    env_target_path: str | None = None,
    ssh_runner: Any | None = None,
) -> BootstrapStepResult:
    """Step 3: Write .env from template if missing."""
    step = "write_env"
    target = env_target_path or f"{machine.resolved_home_dir}/.omnibase/.env"

    if is_local:
        target_p = Path(target)
        if target_p.exists():
            return BootstrapStepResult(
                step, BootstrapStepStatus.VERIFIED, f".env already exists at {target}"
            )

        # Find template
        template = Path(env_template_path) if env_template_path else None
        if template is None:
            # Try relative to this package
            candidates = [
                Path(__file__).resolve().parents[3]
                / ".env.example",  # src/../../../.env.example
                Path(machine.omni_home) / "omnibase_infra" / ".env.example",
            ]
            for c in candidates:
                if c.exists():
                    template = c
                    break

        if template is None or not template.exists():
            return BootstrapStepResult(
                step, BootstrapStepStatus.WARNED, "No .env.example template found"
            )

        target_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(template), str(target_p))
        return BootstrapStepResult(
            step, BootstrapStepStatus.CREATED, f".env written from {template}"
        )
    else:
        runner = ssh_runner or _default_ssh_runner
        try:
            rc, stdout, _ = runner(machine, f"test -f {target} && echo exists")
            if rc == 0 and "exists" in stdout:
                return BootstrapStepResult(
                    step,
                    BootstrapStepStatus.VERIFIED,
                    f".env already exists at {target}",
                )
            return BootstrapStepResult(
                step,
                BootstrapStepStatus.WARNED,
                "Remote .env missing; copy template manually",
            )
        except Exception as exc:
            return BootstrapStepResult(
                step, BootstrapStepStatus.WARNED, f"Could not check remote .env: {exc}"
            )


def _step_render_sync_settings(
    machine: ModelMachineEntry,
    *,
    is_local: bool = False,
    target_path: str | None = None,
    ssh_runner: Any | None = None,
) -> BootstrapStepResult:
    """Step 4: Render + sync settings.json."""
    step = "render_sync_settings"
    settings = render_settings_json(machine)
    new_content = json.dumps(settings, indent=2) + "\n"

    if is_local:
        target = Path(target_path or machine.claude_settings_path)
        if target.exists():
            existing = target.read_text()
            if existing == new_content:
                return BootstrapStepResult(
                    step, BootstrapStepStatus.VERIFIED, "settings.json already matches"
                )

        try:
            write_settings_local(machine, target_path=target_path, dry_run=False)
            return BootstrapStepResult(
                step, BootstrapStepStatus.CREATED, "settings.json written"
            )
        except Exception as exc:
            return BootstrapStepResult(
                step, BootstrapStepStatus.FAILED, f"Failed to write settings: {exc}"
            )
    else:
        try:
            write_settings_remote(machine, dry_run=False)
            return BootstrapStepResult(
                step, BootstrapStepStatus.CREATED, "settings.json written to remote"
            )
        except Exception as exc:
            return BootstrapStepResult(
                step,
                BootstrapStepStatus.FAILED,
                f"Failed to write remote settings: {exc}",
            )


def _step_symlink_plugin(
    machine: ModelMachineEntry,
    *,
    is_local: bool = False,
    plugin_cache_path: str | None = None,
    ssh_runner: Any | None = None,
) -> BootstrapStepResult:
    """Step 5: Symlink plugin cache to canonical clone."""
    step = "symlink_plugin"
    cache_path = (
        plugin_cache_path
        or f"{machine.resolved_home_dir}/.claude/plugins/onex@omninode-tools"
    )
    canonical = machine.plugin_path

    if is_local:
        cache = Path(cache_path)
        if cache.is_symlink():
            link_target = str(cache.readlink())
            if link_target == canonical:
                return BootstrapStepResult(
                    step,
                    BootstrapStepStatus.VERIFIED,
                    f"Symlink correct -> {canonical}",
                )
            # Wrong target, re-link
            cache.unlink()
            cache.symlink_to(canonical)
            return BootstrapStepResult(
                step, BootstrapStepStatus.CREATED, f"Symlink re-pointed to {canonical}"
            )
        if cache.exists():
            bak = Path(cache_path + ".bak")
            shutil.move(str(cache), str(bak))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.symlink_to(canonical)
        return BootstrapStepResult(
            step, BootstrapStepStatus.CREATED, f"Symlink created -> {canonical}"
        )
    else:
        runner = ssh_runner or _default_ssh_runner
        try:
            rc, stdout, _ = runner(machine, f"readlink {cache_path} 2>/dev/null")
            if rc == 0 and stdout.strip() == canonical:
                return BootstrapStepResult(
                    step,
                    BootstrapStepStatus.VERIFIED,
                    f"Symlink correct -> {canonical}",
                )
            # Create symlink remotely
            runner(
                machine,
                f"mkdir -p $(dirname {cache_path}) && ln -sfn {canonical} {cache_path}",
            )
            return BootstrapStepResult(
                step,
                BootstrapStepStatus.CREATED,
                f"Remote symlink created -> {canonical}",
            )
        except Exception as exc:
            return BootstrapStepResult(
                step, BootstrapStepStatus.FAILED, f"Symlink failed: {exc}"
            )


def _step_provision_infisical(
    machine: ModelMachineEntry,
    *,
    infra_root: str | None = None,
) -> BootstrapStepResult:
    """Step 6: Provision Infisical identity (delegate to provision-infisical.py)."""
    step = "provision_infisical"
    if not machine.infisical_participant:
        return BootstrapStepResult(
            step, BootstrapStepStatus.SKIPPED, "Machine is not an Infisical participant"
        )

    script = Path(infra_root or ".") / "scripts" / "provision-infisical.py"
    if not script.exists():
        return BootstrapStepResult(
            step, BootstrapStepStatus.WARNED, f"Script not found: {script}"
        )

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            return BootstrapStepResult(
                step, BootstrapStepStatus.CREATED, "Infisical identity provisioned"
            )
        return BootstrapStepResult(
            step,
            BootstrapStepStatus.WARNED,
            f"Provision returned rc={result.returncode}: {result.stderr[:200]}",
        )
    except Exception as exc:
        return BootstrapStepResult(
            step, BootstrapStepStatus.WARNED, f"Provision failed: {exc}"
        )


def _step_seed_infisical(
    machine: ModelMachineEntry,
    *,
    infra_root: str | None = None,
) -> BootstrapStepResult:
    """Step 7: Seed Infisical secrets (delegate to seed-infisical.py --execute)."""
    step = "seed_infisical"
    if not machine.infisical_participant:
        return BootstrapStepResult(
            step, BootstrapStepStatus.SKIPPED, "Machine is not an Infisical participant"
        )

    script = Path(infra_root or ".") / "scripts" / "seed-infisical.py"
    if not script.exists():
        return BootstrapStepResult(
            step, BootstrapStepStatus.WARNED, f"Script not found: {script}"
        )

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script), "--execute"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            return BootstrapStepResult(
                step, BootstrapStepStatus.CREATED, "Infisical secrets seeded"
            )
        return BootstrapStepResult(
            step,
            BootstrapStepStatus.WARNED,
            f"Seed returned rc={result.returncode}: {result.stderr[:200]}",
        )
    except Exception as exc:
        return BootstrapStepResult(
            step, BootstrapStepStatus.WARNED, f"Seed failed: {exc}"
        )


def _step_run_env_check(
    machine: ModelMachineEntry,
    *,
    settings_path: str | None = None,
    env_path: str | None = None,
    plugin_cache_path: str | None = None,
) -> BootstrapStepResult:
    """Step 8: Run env check to verify."""
    step = "env_check"
    results = run_checks_for_machine(
        machine,
        settings_path=settings_path,
        env_path=env_path,
        plugin_cache_path=plugin_cache_path,
        skip_secrets=True,
    )

    failures = [cid for cid, r in results.items() if r["status"] == "FAIL"]
    warns = [cid for cid, r in results.items() if r["status"] == "WARN"]

    if failures:
        return BootstrapStepResult(
            step, BootstrapStepStatus.FAILED, f"Checks failed: {', '.join(failures)}"
        )
    if warns:
        return BootstrapStepResult(
            step, BootstrapStepStatus.WARNED, f"Checks warned: {', '.join(warns)}"
        )
    return BootstrapStepResult(step, BootstrapStepStatus.VERIFIED, "All checks passed")


def run_bootstrap(
    machine: ModelMachineEntry,
    *,
    is_local: bool = False,
    continue_on_error: bool = False,
    ssh_runner: Any | None = None,
    env_template_path: str | None = None,
    env_target_path: str | None = None,
    settings_target_path: str | None = None,
    plugin_cache_path: str | None = None,
    infra_root: str | None = None,
) -> list[BootstrapStepResult]:
    """Run all 8 bootstrap steps for a machine.

    Returns list of step results. Stops on FAILED unless continue_on_error.
    """
    results: list[BootstrapStepResult] = []

    def _run(result: BootstrapStepResult) -> bool:
        """Append result, return True if pipeline should continue."""
        results.append(result)
        if result.status == BootstrapStepStatus.FAILED and not continue_on_error:
            return False
        return True

    # Step 1: SSH connectivity (skip for local)
    if is_local:
        results.append(
            BootstrapStepResult(
                "ssh_connectivity", BootstrapStepStatus.SKIPPED, "Local machine"
            )
        )
    else:
        r = _step_ssh_connectivity(machine, ssh_runner=ssh_runner)
        if not _run(r):
            return results

    # Step 2: omni_home exists
    r = _step_omni_home_exists(machine, is_local=is_local, ssh_runner=ssh_runner)
    if not _run(r):
        return results

    # Step 3: Write .env
    r = _step_write_env(
        machine,
        is_local=is_local,
        env_template_path=env_template_path,
        env_target_path=env_target_path,
        ssh_runner=ssh_runner,
    )
    if not _run(r):
        return results

    # Step 4: Render + sync settings
    r = _step_render_sync_settings(
        machine,
        is_local=is_local,
        target_path=settings_target_path,
        ssh_runner=ssh_runner,
    )
    if not _run(r):
        return results

    # Step 5: Symlink plugin cache
    r = _step_symlink_plugin(
        machine,
        is_local=is_local,
        plugin_cache_path=plugin_cache_path,
        ssh_runner=ssh_runner,
    )
    if not _run(r):
        return results

    # Step 6: Provision Infisical
    r = _step_provision_infisical(machine, infra_root=infra_root)
    if not _run(r):
        return results

    # Step 7: Seed Infisical
    r = _step_seed_infisical(machine, infra_root=infra_root)
    if not _run(r):
        return results

    # Step 8: Run env check
    r = _step_run_env_check(
        machine,
        settings_path=settings_target_path,
        env_path=env_target_path,
        plugin_cache_path=plugin_cache_path,
    )
    _run(r)

    return results


def format_bootstrap_summary(results: list[BootstrapStepResult]) -> str:
    """Format a human-readable summary of bootstrap results."""
    lines = []
    for r in results:
        icon = {
            BootstrapStepStatus.VERIFIED: "[ok]",
            BootstrapStepStatus.CREATED: "[++]",
            BootstrapStepStatus.SKIPPED: "[--]",
            BootstrapStepStatus.WARNED: "[!!]",
            BootstrapStepStatus.FAILED: "[FAIL]",
        }.get(r.status, "[??]")
        lines.append(f"  {icon} {r.step}: {r.detail}")

    # Summary counts
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary_parts = [f"{count} {status}" for status, count in sorted(counts.items())]
    lines.append(f"\nSummary: {', '.join(summary_parts)}")
    return "\n".join(lines)


@env_group.command("bootstrap")
@click.option("--machine-id", required=True, help="Machine ID from the registry")
@click.option(
    "--registry",
    default="config/machines.yaml",
    help="Path to machines.yaml registry file",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue after non-SSH failures",
)
def bootstrap(machine_id: str, registry: str, continue_on_error: bool) -> None:
    """One-command machine bootstrap.

    Runs 8 steps: SSH, omni_home, .env, settings, plugin symlink,
    Infisical provision, Infisical seed, env check.

    Each step reports verified/created/skipped/warned/failed.
    Re-running reports verified for already-correct steps (idempotent).
    """
    reg = ModelMachineRegistry.from_yaml(Path(registry))
    machine = reg.get_machine(machine_id)
    local = reg.resolve_local_machine()
    is_local = local is not None and machine.machine_id == local.machine_id

    click.echo(f"Bootstrapping {machine_id} ({'local' if is_local else 'remote'})...")

    # Resolve infra_root from registry file location
    registry_path = Path(registry).resolve()
    infra_root = (
        str(registry_path.parent.parent)
        if registry_path.parent.name == "config"
        else "."
    )

    results = run_bootstrap(
        machine,
        is_local=is_local,
        continue_on_error=continue_on_error,
        infra_root=infra_root,
    )

    click.echo(format_bootstrap_summary(results))

    # Exit with error if any step failed
    if any(r.status == BootstrapStepStatus.FAILED for r in results):
        raise SystemExit(1)
