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
   re-runs the unit's launcher, which sources the unit's env files (``EnvironmentFile=``
   and ``DEPLOY_AGENT_ENV_FILE``), so a setting counts as present when the tracked unit
   declares it or one of those FILES assigns it, read when the check runs and never
   from this process's start-time environment (OMN-24677). With a required setting in
   neither place the unit is left as installed, the agent is not restarted, and one
   STATUS row names the gap. A re-exec goes through the same launcher, so it counts the
   process environment or the files, and a refused re-exec writes the same kind of
   row. The overlay source is required only for a fence naming the lab dev lane.
   ``check_host`` runs the same check over EVERY agent installed on a host, the
   prod-fenced unit and each launchd instance included: the code running at a merge
   restarts or re-execs all of them, so the check a merge waits on cannot skip one.
3. The restart runs in a transient unit outside the agent's cgroup, which survives the
   agent's own stop. It restarts the unit, waits, and verifies the unit is active with
   no automatic restart since (a ``Type=simple`` start returns 0 on fork, so the exit
   code alone says nothing). When the start died it puts the code clone back on the
   loaded commit and the previous unit bytes, restarts and verifies again; when that
   fails too it exits non-zero and :class:`AgentLeftStoppedError` is raised to the
   caller. No other agent is touched: an agent only restarts its own unit. The script's
   first command stops the caller, so no exit code reaches it: after a rollback the
   script itself leaves a marker naming the revision it rolled back from, and
   ``self_update`` skips that revision (one STATUS row) until the remote moves on.

The deploy agent loads this module from its own checkout, like the other host effects,
and is handed its own module as ``execution`` where it needs the agent's unit-drift
functions, which are not copied here.
"""

from __future__ import annotations

import logging
import plistlib
import re
import shlex
import signal
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import yaml

from omnibase_infra.models.model_agent_unit import ModelAgentUnit
from omnibase_infra.models.model_restart_on_merge import ModelRestartOnMerge
from omnibase_infra.models.model_unit_settings_gap import ModelUnitSettingsGap

logger = logging.getLogger(__name__)

#: The deploy-source settings a started agent refuses to run without (OMN-24562).
REQUIRED_SETTINGS = (
    "DEPLOY_AGENT_REPO_DIR",
    "DEPLOY_AGENT_CANONICAL_CLONE",
    "DEPLOY_AGENT_LAB_OVERLAY_SOURCE",
)
#: Required only by an agent whose fence names the lab dev lane (OMN-24677).
LAB_ONLY_SETTINGS = frozenset({"DEPLOY_AGENT_LAB_OVERLAY_SOURCE"})
LAB_LANE = "dev"
AGENT_UNIT_MARKER = "DEPLOY_AGENT_TRACKING_REF"
FENCE_SETTING = "DEPLOY_AGENT_ALLOWED_LANES"
ENV_FILE_SETTING = "DEPLOY_AGENT_ENV_FILE"
PROD_LANE = "prod"
MANIFEST_RELPATH = "deploy/unit-drift-manifest.yaml"
SETTLE_SECONDS = 45
REFUSAL_MARKER = "restart-settings-refusal.txt"
REEXEC_REFUSAL_MARKER = "reexec-settings-refusal.txt"
#: Left by the transient unit after a rollback: the revision the agent died on.
ROLLBACK_MARKER = "restart-rolled-back-revision.txt"
ROLLBACK_STATUS_MARKER = "restart-rolled-back-status.txt"
LAUNCHD_LABEL_PREFIX = "ai.omninode.deploy-agent-"
_RESTART_OK = (0, -signal.SIGTERM)
#: Transient-unit exit codes: the previous revision runs again / nothing runs.
ROLLED_BACK = 2
LEFT_STOPPED = 3


class RestartRefusedError(RuntimeError):
    """The unit must not be synced or restarted; it stays as installed."""


class AgentLeftStoppedError(RuntimeError):
    """The restart failed and so did the restart on the previous revision."""


CommandRunner = Callable[[list[str]], int]
"""Runs one argv and returns its exit code."""


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


def _unit_texts(entry: ModelAgentUnit, repo_root: Path) -> list[str]:
    tracked = repo_root / entry.tracked
    texts = [tracked.read_text("utf-8") if tracked.is_file() else ""]
    dropins = tracked.with_name(tracked.name + ".d")
    texts.extend(path.read_text("utf-8") for path in sorted(dropins.glob("*.conf")))
    return texts


def _declared(entry: ModelAgentUnit, repo_root: Path) -> dict[str, str]:
    declared: dict[str, str] = {}
    for text in _unit_texts(entry, repo_root):
        declared.update(declared_environment(text))
    return declared


def as_agent_units(entries: Sequence[object]) -> list[ModelAgentUnit]:
    """The agent's own manifest entries, read as the typed entries this module takes."""
    return [
        entry
        if isinstance(entry, ModelAgentUnit)
        else ModelAgentUnit.model_validate(entry, from_attributes=True)
        for entry in entries
    ]


def agent_units(
    entries: Sequence[ModelAgentUnit], repo_root: Path
) -> list[ModelAgentUnit]:
    """The declared agent instances: the manifest's services whose unit tracks a ref."""
    return [
        entry
        for entry in as_agent_units(entries)
        if entry.name.endswith(".service")
        and AGENT_UNIT_MARKER in _declared(entry, repo_root)
    ]


def declared_fence(entry: ModelAgentUnit, repo_root: Path) -> frozenset[str]:
    """The lane names the tracked unit and its drop-ins declare in the agent's own fence.

    Parsed as the agent parses it (``lane_policy.parse_allowed_lanes``): comma separated
    lane names. An empty or absent fence is refused there at agent start.
    """
    raw = _declared(entry, repo_root).get(FENCE_SETTING, "")
    return frozenset(token.strip() for token in raw.split(",") if token.strip())


def required_settings(fence: frozenset[str]) -> tuple[str, ...]:
    """The settings an agent with ``fence`` refuses to start without."""
    if LAB_LANE in fence:
        return REQUIRED_SETTINGS
    return tuple(name for name in REQUIRED_SETTINGS if name not in LAB_ONLY_SETTINGS)


def _expand(path: str, home: Path) -> Path:
    return Path(path.lstrip("-").replace("%h", str(home)))


def unit_env_files(
    entry: ModelAgentUnit, repo_root: Path, home: Path
) -> tuple[Path, ...]:
    """The env files the unit's launcher sources: ``EnvironmentFile=`` and the store."""
    files: list[Path] = []
    for text in _unit_texts(entry, repo_root):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("EnvironmentFile="):
                files.append(_expand(stripped.removeprefix("EnvironmentFile="), home))
    store = _declared(entry, repo_root).get(ENV_FILE_SETTING, "")
    if store:
        files.append(_expand(store, home))
    return tuple(files)


_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?(?P<name>[A-Za-z_]\w*)=(?P<value>.*)$")


def env_file_names(path: Path) -> frozenset[str]:
    """The names an env file assigns a non-empty value; values are never returned."""
    try:
        text = path.read_text("utf-8")
    except OSError:
        return frozenset()
    names: set[str] = set()
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line)
        if match and match["value"].strip().strip("'\"").strip():
            names.add(match["name"])
    return frozenset(names)


def settings_gap(
    entry: ModelAgentUnit, repo_root: Path, home: Path
) -> ModelUnitSettingsGap:
    """Required settings neither the tracked unit nor one of its env files assigns."""
    declared = _declared(entry, repo_root)
    files = unit_env_files(entry, repo_root, home)
    in_files: set[str] = set()
    for path in files:
        in_files |= env_file_names(path)
    fence = declared_fence(entry, repo_root)
    required = required_settings(fence)
    missing = tuple(
        name
        for name in required
        if not declared.get(name, "").strip() and name not in in_files
    )
    return ModelUnitSettingsGap(
        unit=entry.name,
        env_files=tuple(str(p) for p in files),
        missing=missing,
        fence=tuple(sorted(fence)),
        found=tuple(name for name in required if name not in missing),
    )


def missing_settings(entry: ModelAgentUnit, repo_root: Path, home: Path) -> list[str]:
    """Required settings neither the tracked unit nor its env files carry."""
    return list(settings_gap(entry, repo_root, home).missing)


def restart_set(
    entries: Sequence[ModelAgentUnit], repo_root: Path, home: Path
) -> list[str]:
    """Agent units a merge may restart: none whose fence names prod, none lacking settings."""
    return [
        entry.name
        for entry in agent_units(entries, repo_root)
        if PROD_LANE not in declared_fence(entry, repo_root)
        and not missing_settings(entry, repo_root, home)
    ]


def assert_restartable(
    target: ModelRestartOnMerge, entries: Sequence[ModelAgentUnit]
) -> None:
    """Refuse (before anything is replaced or stopped) a unit outside the restart set."""
    repo_root = Path(target.repo_root)
    entry = next(
        (e for e in agent_units(entries, repo_root) if e.name == target.unit), None
    )
    if entry is None:
        return
    if PROD_LANE in declared_fence(entry, repo_root):
        raise RestartRefusedError(
            f"{target.unit} is fenced to prod: a merge never restarts it"
        )
    gap = settings_gap(entry, repo_root, Path(target.home))
    if gap.missing:
        raise RestartRefusedError(
            f"{target.unit} would restart without {', '.join(gap.missing)} "
            f"(env files read: {', '.join(gap.env_files) or 'none'}): the new code "
            "refuses to start without them, so the unit is left as installed"
        )


def check_host(
    entries: Sequence[ModelAgentUnit], repo_root: Path, home: Path
) -> tuple[ModelUnitSettingsGap, ...]:
    """Every agent unit installed under ``home`` whose env files lack a setting.

    The prod-fenced unit is read too: it is never restarted by THIS module, but the
    code running at a merge restarts every installed agent, and the prod unit takes
    its deploy-source settings from an env file a merge cannot repair.
    """
    installed = home / ".config" / "systemd" / "user"
    return tuple(
        gap
        for entry in agent_units(entries, repo_root)
        if (installed / entry.name).exists()
        and (gap := settings_gap(entry, repo_root, home)).missing
    )


def launchd_readings(
    home: Path,
) -> tuple[tuple[str, ...], tuple[ModelUnitSettingsGap, ...]]:
    """Every installed launchd agent instance, and the ones whose store lacks a setting.

    A launchd instance re-execs through its launcher at a merge with no unit sync and
    no guard in the code running then, so each one is named, and read the way the
    launcher resolves its settings: the plist's environment, then the store it sources.
    """
    labels: list[str] = []
    gaps: list[ModelUnitSettingsGap] = []
    agents = home / "Library" / "LaunchAgents"
    for plist in sorted(agents.glob(f"{LAUNCHD_LABEL_PREFIX}*.plist")):
        try:
            declared = plistlib.loads(plist.read_bytes())
        except (OSError, plistlib.InvalidFileException, ValueError):
            declared = {}
        label = str(declared.get("Label") or plist.stem)
        labels.append(label)
        env = {
            str(k): str(v)
            for k, v in (declared.get("EnvironmentVariables") or {}).items()
        }
        missing = missing_runtime_settings(env)
        if missing:
            gaps.append(
                ModelUnitSettingsGap(
                    unit=label,
                    env_files=(env.get(ENV_FILE_SETTING, ""),),
                    missing=tuple(missing),
                    fence=tuple(sorted(_fence_of(env))),
                )
            )
    return tuple(labels), tuple(gaps)


def _fence_of(env: Mapping[str, str]) -> frozenset[str]:
    return frozenset(
        token.strip() for token in env.get(FENCE_SETTING, "").split(",") if token
    )


def missing_runtime_settings(env: Mapping[str, str]) -> list[str]:
    """Required settings neither the process nor the store the launcher re-sources has."""
    fence = _fence_of(env)
    store = env.get(ENV_FILE_SETTING, "")
    in_store = env_file_names(Path(store)) if store else frozenset()
    return [
        name
        for name in required_settings(fence)
        if not env.get(name, "").strip() and name not in in_store
    ]


def _marker(env: Mapping[str, str], name: str) -> Path | None:
    state_dir = env.get("DEPLOY_AGENT_STATE_DIR", "")
    return Path(state_dir) / name if state_dir else None


def clear_refusal(env: Mapping[str, str], name: str = REFUSAL_MARKER) -> None:
    """A refusal that no longer holds is forgotten, so its return writes a new row."""
    marker = _marker(env, name)
    if marker is not None:
        marker.unlink(missing_ok=True)


def reexec_blocked_by_settings(execution: ModuleType, env: Mapping[str, str]) -> bool:
    """Whether a re-exec would start code that refuses to run for want of settings.

    A re-exec keeps this process's environment and the launcher re-sources the store,
    so a setting in neither keeps the process on the revision it loaded: dying in the
    new code is a stopped agent. A refusal writes one STATUS row, as the unit sync's
    refusal does (this path runs on every platform, the sync only on Linux), and a
    pass clears the row's marker.
    """
    missing = missing_runtime_settings(env)
    if not missing:
        clear_refusal(env, REEXEC_REFUSAL_MARKER)
        return False
    reason = (
        "not re-execing into the pulled revision: neither this process nor its env "
        f"store names {', '.join(missing)}"
    )
    logger.error("OMN-24626 %s", reason)
    write_refusal_status(execution, reason, env, REEXEC_REFUSAL_MARKER)
    return True


def skip_refused_revision(
    execution: ModuleType, remote_sha: str, env: Mapping[str, str]
) -> bool:
    """Whether ``remote_sha`` is the revision the last restart rolled back from.

    The rolled-back agent runs the previous revision again, and would pull the same
    head at every boundary and be restarted into it again. One STATUS row names the
    refused revision; a remote that moved on clears the marker.
    """
    marker = _marker(env, ROLLBACK_MARKER)
    if marker is None or not marker.is_file():
        return False
    if marker.read_text("utf-8").strip() != remote_sha:
        marker.unlink(missing_ok=True)
        clear_refusal(env, ROLLBACK_STATUS_MARKER)
        return False
    write_refusal_status(
        execution,
        f"revision {remote_sha} refused: the agent restarted on it died and was "
        "rolled back; it is skipped until the remote moves on",
        env,
        ROLLBACK_STATUS_MARKER,
    )
    return True


def write_refusal_status(
    execution: ModuleType,
    reason: str,
    env: Mapping[str, str],
    marker_name: str = REFUSAL_MARKER,
) -> None:
    """One STATUS row per distinct refusal, so a frozen agent is named on the ledger."""
    marker = _marker(env, marker_name)
    if marker is not None and marker.is_file() and marker.read_text("utf-8") == reason:
        return
    config = execution.dev_boot_report_config()
    if config.plugin_dir is None or not config.ledger_via_bus:
        logger.warning("OMN-24677 restart refusal STATUS pending: %s", reason)
        return
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    host = env.get("ONEX_LEDGER_HOST_NAME", "") or socket.gethostname()
    row = (
        f"{stamp} | STATUS | lane=deploy-agent-{host} | actor=script:deploy-agent "
        f"| model=none | ticket=OMN-24677 | restart refused: {reason.replace('|', '/')}"
    )
    interpreter = config.registry_root / ".onex-dispatch-venv" / "bin" / "python"
    argv = [
        str(interpreter),
        str(config.plugin_dir / "scripts" / "onex_ledger.py"),
        "--append",
        row,
    ]
    if execution._run(argv, 60).returncode == 0 and marker is not None:
        marker.write_text(reason, "utf-8")


def read_own_cgroup() -> str:
    """This process's cgroup file: the unit systemd runs it in."""
    return Path("/proc/self/cgroup").read_text("utf-8")


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

    def run_unit_command(command: list[str]) -> int:
        return int(execution._run(command, timeout).returncode)

    if sys.platform != "linux":
        return None
    try:
        unit_name = drift.own_unit_name_from_cgroup(read_own_cgroup())
    except OSError:
        return None
    if unit_name is None:
        return None
    repo_root = Path(agent_dir).resolve().parents[1]
    entries = drift.load_manifest(repo_root / MANIFEST_RELPATH)
    target = ModelRestartOnMerge(
        unit=unit_name, repo_root=str(repo_root), home=str(home)
    )
    try:
        assert_restartable(target, as_agent_units(entries))
    except RestartRefusedError as exc:
        logger.warning("self_update: unit not synced: %s", exc)
        write_refusal_status(execution, str(exc), env)
        return None
    clear_refusal(env)
    changed = drift.sync_own_unit(
        entries,
        unit_name=unit_name,
        repo_root=repo_root,
        hostname=socket.gethostname(),
        home=home,
        runner=run_unit_command,
        stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    return unit_name if changed else None


def installed_unit_path(target: ModelRestartOnMerge) -> Path:
    """Where an agent unit is installed: the user unit directory under ``home``."""
    return Path(target.home) / ".config" / "systemd" / "user" / target.unit


def transient_restart_script(
    unit: str,
    *,
    installed: Path,
    agent_dir: str,
    loaded_sha: str,
    settle: int,
    refused_marker: Path | None = None,
) -> str:
    """The script a transient unit runs: restart, verify, roll back, verify again.

    Exit 0: the new revision is up. ``ROLLED_BACK``: it died and the previous revision
    is up; the script writes the revision it rolled back from to ``refused_marker``
    first, because its caller was stopped by the first command and cannot read the
    exit. ``LEFT_STOPPED``: neither is up.
    """
    q = shlex.quote
    up = (
        f"sleep {settle}; systemctl --user is-active --quiet {q(unit)} && "
        f'[ "$(systemctl --user show -p NRestarts --value {q(unit)})" = 0 ]'
    )
    leave_marker = (
        f"mkdir -p {q(str(refused_marker.parent))} && "
        f'printf %s "$refused" > {q(str(refused_marker))}; '
        if refused_marker is not None
        else ""
    )
    return "\n".join(
        (
            f"systemctl --user restart {q(unit)}",
            f"if {up}; then exit 0; fi",
            f"refused=$(git -C {q(agent_dir)} rev-parse HEAD)",
            f"git -C {q(agent_dir)} checkout --quiet --detach {q(loaded_sha)}",
            f"backup=$(ls -1 {q(str(installed))}.bak-unit-drift-* 2>/dev/null | tail -n 1)",
            f'[ -n "$backup" ] && cp -p "$backup" {q(str(installed))}',
            "systemctl --user daemon-reload",
            f"systemctl --user restart {q(unit)}",
            f"if {up}; then {leave_marker}exit {ROLLED_BACK}; fi",
            f"exit {LEFT_STOPPED}",
        )
    )


def restart_synced_unit(
    target: ModelRestartOnMerge,
    *,
    transient_command: Sequence[str],
    script: str,
    run_unit_command: CommandRunner,
) -> None:
    """Run the restart script in a transient unit; raise when it left the agent stopped.

    A ``-SIGTERM`` is this process being stopped by its own restart: the transient
    unit goes on verifying and rolling back without it. Any exit other than the
    script's own means the transient unit never ran and nothing was stopped.
    """
    code = run_unit_command([*transient_command, "bash", "-c", script])
    if code in _RESTART_OK:
        return
    if code == ROLLED_BACK:
        logger.error(
            "OMN-24626 the restarted %s died; it runs its previous revision again",
            target.unit,
        )
        return
    if code == LEFT_STOPPED:
        raise AgentLeftStoppedError(
            f"restart of {target.unit} failed and so did its restart on the previous "
            "revision: operator action needed; no other agent was touched"
        )
    raise RuntimeError(
        f"the transient restart unit for {target.unit} did not run (exit {code}): "
        "the agent was not stopped and keeps its revision"
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
    """Restart our own synced unit from a transient unit; see :func:`restart_synced_unit`."""
    target = ModelRestartOnMerge(
        unit=unit_name,
        repo_root=str(Path(agent_dir).resolve().parents[1]),
        home=env["HOME"],
    )
    argv = execution.SELF_RESTART_COMMAND.split()
    restart_synced_unit(
        target,
        transient_command=argv[: argv.index("systemctl")],
        script=transient_restart_script(
            unit_name,
            installed=installed_unit_path(target),
            agent_dir=agent_dir,
            loaded_sha=loaded_sha,
            settle=SETTLE_SECONDS,
            refused_marker=_marker(env, ROLLBACK_MARKER),
        ),
        run_unit_command=lambda command: int(
            execution._run(command, timeout + 2 * SETTLE_SECONDS).returncode
        ),
    )


class HandlerRestartOnMerge:
    """Decide whether a merge may sync and restart one agent unit, or a whole host."""

    def handle(self, request: ModelRestartOnMerge) -> ModelRestartOnMerge:
        repo_root = Path(request.repo_root)
        manifest = yaml.safe_load((repo_root / MANIFEST_RELPATH).read_text("utf-8"))
        entries = [
            ModelAgentUnit(name=unit["name"], tracked=unit["tracked"])
            for unit in manifest["units"]
        ]
        if not request.home:
            return request.model_copy(
                update={"reason": "home is required: the env files resolve under it"}
            )
        home = Path(request.home)
        if request.operation == "check_host":
            launchd, launchd_gaps = launchd_readings(home)
            gaps = (*check_host(entries, repo_root, home), *launchd_gaps)
            reason = "; ".join(f"{g.unit}: {', '.join(g.missing)}" for g in gaps)
            return request.model_copy(
                update={
                    "gaps": gaps,
                    "restartable": not gaps,
                    "reason": reason,
                    "launchd_instances": launchd,
                }
            )
        target = ModelRestartOnMerge(
            unit=request.unit, repo_root=str(repo_root), home=str(home)
        )
        try:
            assert_restartable(target, entries)
        except RestartRefusedError as exc:
            return request.model_copy(update={"reason": str(exc)})
        return request.model_copy(update={"restartable": True})
