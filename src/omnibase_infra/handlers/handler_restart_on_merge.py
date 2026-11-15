# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An agent restart on merge never touches a prod agent and never leaves an agent stopped (OMN-24626).

WHAT A MERGE RESTARTS
---------------------
Each agent self-updates its own code clone (``executor.self_update``). When the pulled
revision changes the agent's OWN tracked unit, the agent replaces its installed unit and
restarts that unit, and no other (``executor._sync_own_unit_after_pull``). So a merge
that edits tracked units restarts every agent whose unit it edits, one agent at a time,
each by its own process. The agent units are the entries of
``deploy/unit-drift-manifest.yaml`` under ``scripts/deploy-agent/deploy/``:

* ``deploy-agent-dev.service`` -- h201 dev agent, fence ``dev`` plus the stability lane
  it follows (``DEPLOY_AGENT_FOLLOW_STABILITY=1``). There is no separate stability agent.
* the second dev agent unit, on the other lab host, fence ``dev``.
* ``deploy-agent.service`` -- the prod agent. The tracked base unit declares no fence;
  its drop-in ``deploy-agent.service.d/override.conf`` declares ``prod``. Both tracked
  files are read here and parsed by the agent's own fence parser, not guessed from a
  unit name.

THE RULES
---------
1. The restart set is every declared agent unit whose fence does not name ``prod``.
2. An agent is never stopped before its replacement settings are verified. A restart
   re-runs the unit's launcher over the same host env store, so a setting counts as
   present when the tracked unit declares it or the running process carries it; with
   any of ``REQUIRED_SETTINGS`` in neither place the unit is left as installed and the
   agent is not restarted (the new code refuses to start without them). The same holds
   for a re-exec: a process that lacks the settings keeps the revision it loaded. The
   tracked units cannot name the canonical clone or the overlay source clone: the
   public-repo hygiene gate refuses their private directory names on added lines, so
   those two stay in the host env store and are verified at runtime.
3. When the restart fails the agent is restarted on the revision it was running (the
   previous unit bytes and the code clone at the loaded commit). When that fails too the
   failure is raised loudly, and no other agent is touched: an agent only restarts its
   own unit.

The deploy agent loads this module from its own checkout, like the other host effects,
and is handed its own module as ``execution`` where it needs the agent's unit-drift
functions, which are not copied here.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import signal
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import yaml

from omnibase_infra.models.model_restart_on_merge import ModelRestartOnMerge

logger = logging.getLogger(__name__)

#: The deploy-source settings a started agent refuses to run without (OMN-24562).
REQUIRED_SETTINGS = (
    "DEPLOY_AGENT_REPO_DIR",
    "DEPLOY_AGENT_CANONICAL_CLONE",
    "DEPLOY_AGENT_LAB_OVERLAY_SOURCE",
)
AGENT_UNIT_MARKER = "DEPLOY_AGENT_TRACKING_REF"
FENCE_SETTING = "DEPLOY_AGENT_ALLOWED_LANES"
PROD_LANE = "prod"
MANIFEST_RELPATH = "deploy/unit-drift-manifest.yaml"
_RESTART_OK = (0, -signal.SIGTERM)


class RestartRefusedError(RuntimeError):
    """The unit must not be synced or restarted; it stays as installed."""


class AgentLeftStoppedError(RuntimeError):
    """The restart failed and so did the restart on the previous revision."""


def declared_environment(unit_text: str) -> dict[str, str]:
    """``Environment=`` assignments of a unit or drop-in; the last one wins."""
    values: dict[str, str] = {}
    for line in unit_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Environment="):
            continue
        for token in shlex.split(stripped.removeprefix("Environment=")):
            name, separator, value = token.partition("=")
            if separator:
                values[name] = value
    return values


def _unit_text(entry: Any, repo_root: Path) -> str:
    path = repo_root / entry.tracked
    return path.read_text("utf-8") if path.is_file() else ""


def agent_units(entries: Sequence[Any], repo_root: Path) -> list[Any]:
    """The declared agent instances: the manifest's services whose unit tracks a ref."""
    return [
        entry
        for entry in entries
        if entry.name.endswith(".service")
        and AGENT_UNIT_MARKER in declared_environment(_unit_text(entry, repo_root))
    ]


def declared_fence(entry: Any, repo_root: Path) -> frozenset[str]:
    """The lane names the tracked unit and its drop-ins declare in the agent's own fence.

    Parsed as the agent parses it (``lane_policy.parse_allowed_lanes``): comma separated
    lane names. An empty or absent fence is refused there at agent start.
    """
    tracked = repo_root / entry.tracked
    texts = [_unit_text(entry, repo_root)]
    dropins = tracked.with_name(tracked.name + ".d")
    texts.extend(path.read_text("utf-8") for path in sorted(dropins.glob("*.conf")))
    declared: dict[str, str] = {}
    for text in texts:
        declared.update(declared_environment(text))
    raw = declared.get(FENCE_SETTING, "")
    return frozenset(token.strip() for token in raw.split(",") if token.strip())


def missing_settings(entry: Any, repo_root: Path, env: Mapping[str, str]) -> list[str]:
    """Required settings neither the tracked unit nor the running process carries."""
    declared = declared_environment(_unit_text(entry, repo_root))
    return [
        name
        for name in REQUIRED_SETTINGS
        if not declared.get(name, "").strip() and not env.get(name, "").strip()
    ]


def restart_set(
    entries: Sequence[Any], repo_root: Path, env: Mapping[str, str]
) -> list[str]:
    """Agent units a merge may restart: none whose fence names prod, none lacking settings."""
    return [
        entry.name
        for entry in agent_units(entries, repo_root)
        if PROD_LANE not in declared_fence(entry, repo_root)
        and not missing_settings(entry, repo_root, env)
    ]


def assert_restartable(
    unit_name: str,
    entries: Sequence[Any],
    repo_root: Path,
    env: Mapping[str, str],
) -> None:
    """Refuse (before anything is replaced or stopped) a unit outside the restart set."""
    entry = next(
        (e for e in agent_units(entries, repo_root) if e.name == unit_name), None
    )
    if entry is None:
        return
    if PROD_LANE in declared_fence(entry, repo_root):
        raise RestartRefusedError(
            f"{unit_name} is fenced to prod: a merge never restarts it"
        )
    missing = missing_settings(entry, repo_root, env)
    if missing:
        raise RestartRefusedError(
            f"{unit_name} would restart without {', '.join(missing)}: the new code "
            "refuses to start without them, so the unit is left as installed"
        )


def missing_runtime_settings(env: Mapping[str, str]) -> list[str]:
    """The required settings the RUNNING process does not carry."""
    return [name for name in REQUIRED_SETTINGS if not env.get(name, "").strip()]


def reexec_blocked_by_settings(env: Mapping[str, str]) -> bool:
    """Whether a re-exec would start code that refuses to run for want of settings.

    A re-exec keeps this process's environment, so a process that lacks the settings
    (a unit installed before OMN-24562, the prod unit a merge never syncs) must keep
    running the revision it loaded: dying in the new code is a stopped agent.
    """
    missing = missing_runtime_settings(env)
    if missing:
        logger.error(
            "OMN-24626 not re-execing into the pulled revision: this process lacks %s",
            ", ".join(missing),
        )
    return bool(missing)


def sync_own_unit_after_pull(
    execution: ModuleType, agent_dir: str, timeout: int, env: Mapping[str, str]
) -> str | None:
    """Repair only our Linux unit and return its name when restart is needed.

    OMN-20037: pulling the PATH fix left the installed unit stale for two days.
    A re-exec cannot pick up systemd's changed unit environment; a restart can.
    OMN-24626: a unit outside the restart set is left exactly as installed.
    """
    drift = execution.unit_drift
    home = Path(env["HOME"])

    def run(command: list[str]) -> int:
        return int(execution._run(command, timeout).returncode)

    if sys.platform != "linux":
        return None
    try:
        unit_name = drift.own_unit_name_from_cgroup(
            Path("/proc/self/cgroup").read_text()
        )
    except OSError:
        return None
    if unit_name is None:
        return None
    repo_root = Path(agent_dir).resolve().parents[1]
    entries = drift.load_manifest(repo_root / MANIFEST_RELPATH)
    try:
        assert_restartable(unit_name, entries, repo_root, env)
    except RestartRefusedError as exc:
        logger.warning("self_update: unit not synced: %s", exc)
        return None
    changed = drift.sync_own_unit(
        entries,
        unit_name=unit_name,
        repo_root=repo_root,
        hostname=socket.gethostname(),
        home=home,
        runner=run,
        stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    return unit_name if changed else None


def restore_previous_unit(
    execution: ModuleType,
    unit_name: str,
    entries: Sequence[Any],
    repo_root: Path,
    home: Path,
    run: Callable[[list[str]], int],
) -> None:
    """Put the newest backup of the installed unit back and reload systemd."""
    entry = next(e for e in entries if e.name == unit_name)
    result = execution.unit_drift.check_units(
        [entry], repo_root=repo_root, hostname="", home=home
    )[0]
    installed: Path = result.installed_path
    backups = sorted(installed.parent.glob(f"{installed.name}.bak-unit-drift-*"))
    if backups:
        shutil.copy2(backups[-1], installed)
    run(["systemctl", "--user", "daemon-reload"])


def restart_synced_unit(
    unit_name: str,
    *,
    restart_command: Sequence[str],
    run: Callable[[list[str]], int],
    rollback: Sequence[Callable[[], None]],
) -> None:
    """Restart ``unit_name``; on failure restart it on the revision it was running.

    ``run`` returns an exit code. ``rollback`` steps put the code clone and the
    installed unit back on what the process loaded. A failure of the second restart
    raises :class:`AgentLeftStoppedError`; nothing else is stopped in either case.
    """
    command = [*restart_command, unit_name]
    if run(command) in _RESTART_OK:
        return
    logger.error(
        "OMN-24626 restart of %s failed; restarting it on the revision it was running",
        unit_name,
    )
    for step in rollback:
        try:
            step()
        except Exception:
            logger.exception("OMN-24626 rollback step failed for %s", unit_name)
    if run(command) in _RESTART_OK:
        logger.error("OMN-24626 %s restarted on its previous revision", unit_name)
        return
    raise AgentLeftStoppedError(
        f"restart of {unit_name} failed and so did its restart on the previous "
        "revision: operator action needed; no other agent was touched"
    )


def restart_own_unit(
    execution: ModuleType,
    unit_name: str,
    *,
    agent_dir: str,
    loaded_sha: str,
    timeout: int,
    env: Mapping[str, str],
) -> None:
    """Restart our own synced unit; see :func:`restart_synced_unit` for a failure."""
    repo_root = Path(agent_dir).resolve().parents[1]
    home = Path(env["HOME"])

    def run(command: list[str]) -> int:
        return int(execution._run(command, timeout).returncode)

    def restore_code() -> None:
        argv = ["git", "-C", agent_dir, "checkout", "--quiet", "--detach", loaded_sha]
        if run(argv):
            raise RuntimeError(f"could not put {agent_dir} back on {loaded_sha[:12]}")

    def restore_unit() -> None:
        entries = execution.unit_drift.load_manifest(repo_root / MANIFEST_RELPATH)
        restore_previous_unit(execution, unit_name, entries, repo_root, home, run)

    restart_synced_unit(
        unit_name,
        restart_command=execution.SELF_RESTART_COMMAND.split(),
        run=run,
        rollback=(restore_code, restore_unit),
    )


class HandlerRestartOnMerge:
    """Decide whether a merge may sync and restart one agent unit."""

    def handle(self, request: ModelRestartOnMerge) -> ModelRestartOnMerge:
        repo_root = Path(request.repo_root)
        manifest = yaml.safe_load((repo_root / MANIFEST_RELPATH).read_text("utf-8"))
        entries = [SimpleNamespace(**unit) for unit in manifest["units"]]
        carried = dict.fromkeys(request.carried_settings, "set")
        try:
            assert_restartable(request.unit_name, entries, repo_root, carried)
        except RestartRefusedError as exc:
            return request.model_copy(update={"reason": str(exc)})
        return request.model_copy(update={"restartable": True})
