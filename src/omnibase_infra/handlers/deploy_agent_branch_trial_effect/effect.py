# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-only agent branch trials, supervised outside the candidate process.

The launcher loads this parent from dev. The signed bus adapter queues a typed
request; this parent owns its deadline, the candidate process group and recovery.
A failed import, re-exec or unresponsive health socket cannot remove the parent.
No compose, broker, production unit or remote-host command belongs to this route.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import http.client
import ipaddress
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from operator import or_
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from deploy_agent.agent import DeployAgent
    from deploy_agent.events import ModelRebuildCompleted, ModelRebuildRequested
    from deploy_agent.job_state import JobStore

    from omnibase_infra.models.model_deploy_agent_branch_trial_host import (
        ModelDeployAgentBranchTrialHost,
    )
    from omnibase_infra.models.model_deploy_agent_branch_trial_policy import (
        ModelDeployAgentBranchTrialPolicy,
    )
    from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
        ModelDeployAgentBranchTrialRequest,
    )
    from omnibase_infra.models.model_deploy_agent_branch_trial_result import (
        ModelDeployAgentBranchTrialResult,
    )


logger = logging.getLogger(__name__)
ACTIVE_STATUSES = frozenset({"queued", "running"})
_HOST: ContextVar[ModelDeployAgentBranchTrialHost] = ContextVar(
    "deploy_agent_branch_trial_host"
)


def configure_host(environment: dict[str, str]) -> None:
    from omnibase_infra.handlers.deploy_agent_branch_trial_effect.composition import (
        load_policy,
    )
    from omnibase_infra.models.model_deploy_agent_branch_trial_host import (
        ModelDeployAgentBranchTrialHost,
    )

    _HOST.set(
        ModelDeployAgentBranchTrialHost(
            policy=load_policy(Path(environment["DEPLOY_AGENT_TRIAL_CONTRACT"])),
            environment=environment,
            allowed_lanes=environment["DEPLOY_AGENT_ALLOWED_LANES"],
            tracking_ref=environment["DEPLOY_AGENT_TRACKING_REF"],
            repo_root=Path(environment["DEPLOY_AGENT_DIR"]).resolve().parents[1],
            state_dir=Path(environment["DEPLOY_AGENT_STATE_DIR"]),
            python=Path(environment["DEPLOY_AGENT_PYTHON"]),
            health_port=int(environment["DEPLOY_AGENT_PORT"]),
            home=Path(environment["HOME"]),
            registry_root=Path(environment["UNIT_DRIFT_REGISTRY_ROOT"])
            if environment.get("UNIT_DRIFT_REGISTRY_ROOT")
            else None,
        )
    )


def _policy() -> ModelDeployAgentBranchTrialPolicy:
    return _HOST.get().policy


def _guard() -> None:
    if _HOST.get().allowed_lanes != "dev":
        raise ValueError("agent branch trials require a fence of exactly dev")
    if _HOST.get().tracking_ref != "dev":
        raise ValueError("agent branch trials return only to origin/dev")


def _root(state_dir: Path) -> Path:
    return state_dir / "agent-branch-trial"


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".trial-", text=True)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        directory = os.open(path.parent, or_(os.O_RDONLY, os.O_DIRECTORY))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def read_receipt(state_dir: Path) -> dict[str, object]:
    try:
        payload = json.loads((_root(state_dir) / "receipt.json").read_text())
        if not isinstance(payload, dict):
            raise ValueError("invalid trial receipt")
        return payload
    except FileNotFoundError:
        return {"status": "none"}


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, or_(fcntl.LOCK_EX, fcntl.LOCK_NB))
        yield


def _git(repo: Path, *args: str, timeout: float | None = None) -> str:
    # A node may be invoked by a Git hook. Its process state must not redirect
    # worktree operations into the hook's repository.
    env = {k: v for k, v in _HOST.get().environment.items() if not k.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(
            0.1, timeout if timeout is not None else _policy().git_timeout_seconds
        ),
    )
    if result.returncode:
        raise RuntimeError(f"git {args[0]} exited {result.returncode}")
    return result.stdout.strip()


def _spawn(
    repo: Path, *, trial: bool, args: Sequence[str] = ()
) -> subprocess.Popen[bytes]:
    agent_dir = repo / "scripts/deploy-agent"
    env = dict(_HOST.get().environment)
    env["DEPLOY_AGENT_DIR"] = str(agent_dir)
    env["DEPLOY_AGENT_SUPERVISED"] = "1"
    paths = [str(agent_dir)]
    if trial:
        paths.append(str(repo / "src"))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    argv = [str(_HOST.get().python), "-m", "deploy_agent", *args]
    if trial:
        # Present even on old heads. The parent, rather than the candidate,
        # owns this bounded pause and always starts dev without this flag.
        argv.append("--skip-self-update")
    return subprocess.Popen(argv, cwd=agent_dir, env=env, start_new_session=True)


def _stop(child: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    # Also retire descendants when the group leader already exited. An
    # installer or broken candidate must not leave a background writer alive.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=_policy().stop_timeout_seconds)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=_policy().stop_timeout_seconds)


def _health() -> dict[str, object]:
    port = int(_HOST.get().health_port)
    if not 1 <= port <= 65535:
        raise ValueError("invalid agent health port")
    # A direct loopback connection follows neither redirects nor proxy env.
    policy = _policy()
    host = policy.health_host
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("agent trial health host must be loopback")
    connection = http.client.HTTPConnection(
        host, port, timeout=policy.health_timeout_seconds
    )
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"agent health returned HTTP {response.status}")
        body = response.read(65537)
        if len(body) > 65536:
            raise ValueError("agent health exceeded its readback bound")
        payload = json.loads(body)
    finally:
        connection.close()
    if not isinstance(payload, dict):
        raise ValueError("agent health was not an object")
    return payload


def _wait_loaded(child: subprocess.Popen[bytes], head: str, deadline: float) -> None:
    while time.time() < deadline:
        if child.poll() is not None:
            raise RuntimeError(
                f"candidate exit={child.returncode} before loaded-code readback"
            )
        try:
            if _health().get("loaded_code_sha") == head:
                return
        except (OSError, ValueError, http.client.HTTPException):
            pass
        time.sleep(0.1)
    raise RuntimeError("loaded-code readback timed out")


# The one declared host action is a node installer, never caller-supplied argv.
_IMAGE_NODE = "node_deploy_agent_image_prune_effect"
_IMAGE_RELATIVE = "src/omnibase_infra/nodes/" + _IMAGE_NODE
_IMAGE_SERVICE = "onex-deploy-agent-image-prune.service"
_IMAGE_TIMER = "onex-deploy-agent-image-prune.timer"
_IMAGE_FILES = (
    ".config/systemd/user/" + _IMAGE_SERVICE,
    ".config/systemd/user/" + _IMAGE_TIMER,
    ".config/onex/deploy-agent-image-prune.env",
)


def _action(
    argv: list[str],
    *,
    cwd: Path,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    if time.time() >= deadline:
        raise TimeoutError("branch host action exceeded its deadline")
    env = dict(_HOST.get().environment)
    env["DEPLOY_AGENT_ALLOWED_LANES"] = "dev"
    env["PYTHONPATH"] = str(cwd / "src")
    child = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=max(0.1, deadline - time.time()))
        return subprocess.CompletedProcess(argv, child.returncode, stdout, stderr)
    finally:
        _stop(child)


def _systemctl(
    root: Path, deadline: float, *args: str
) -> subprocess.CompletedProcess[str]:
    return _action(["systemctl", "--user", *args], cwd=root, deadline=deadline)


def _require_ok(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode:
        # Node output may contain environment information. Retain the bounded
        # operation's exit code, never raw stderr in the health API.
        raise RuntimeError(f"host action exited {result.returncode}")
    return result.stdout


def _restore_units(root: Path, deadline: float) -> None:
    snapshot_file = root / "unit-snapshot.json"
    if not snapshot_file.exists():
        return
    snapshot = json.loads(snapshot_file.read_text())
    stopped = _systemctl(root, deadline, "stop", _IMAGE_TIMER, _IMAGE_SERVICE)
    if stopped.returncode not in (0, 5):
        _require_ok(stopped)
    disabled = _systemctl(root, deadline, "disable", _IMAGE_TIMER)
    if disabled.returncode and (_HOST.get().home / _IMAGE_FILES[1]).exists():
        _require_ok(disabled)
    home = _HOST.get().home
    for relative in _IMAGE_FILES:
        path = home / relative
        prior = snapshot["files"][relative]
        if prior is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(prior["content"], validate=True))
            path.chmod(prior["mode"])
    _require_ok(_systemctl(root, deadline, "daemon-reload"))
    if snapshot["timer_enabled"]:
        _require_ok(_systemctl(root, deadline, "enable", _IMAGE_TIMER))
    if snapshot["timer_active"]:
        _require_ok(_systemctl(root, deadline, "start", _IMAGE_TIMER))
    snapshot_file.unlink()


def _install_image_prune(tree: Path, root: Path, deadline: float) -> tuple[str, str]:
    manifest = tree / _IMAGE_RELATIVE / "install.json"
    contract = tree / _IMAGE_RELATIVE / "contract.yaml"
    expected = {
        "operation": "deploy_agent.images.install",
        "repository": ".",
        "policy_path": _IMAGE_RELATIVE + "/contract.yaml",
    }
    # Reject symlink and path redirects before touching a unit. Only the exact
    # installer input shipped by this node is admitted.
    if (
        manifest.resolve() != manifest
        or contract.resolve() != contract
        or json.loads(manifest.read_text()) != expected
    ):
        raise ValueError(
            "image-prune install manifest does not match the declared host operation"
        )
    registry_root = _HOST.get().registry_root
    if registry_root is None:
        raise ValueError("image-prune installer requires the host registry root")
    dispatch = registry_root / ".onex-dispatch-venv/bin/python"
    if not dispatch.is_file() or not os.access(dispatch, os.X_OK):
        raise ValueError("image-prune installer requires the host dispatch interpreter")
    home = _HOST.get().home
    files: dict[str, object] = {}
    for relative in _IMAGE_FILES:
        path = home / relative
        if path.is_symlink():
            raise ValueError("image-prune unit destination is a symlink")
        files[relative] = (
            {
                "content": base64.b64encode(path.read_bytes()).decode(),
                "mode": path.stat().st_mode & 0o777,
            }
            if path.exists()
            else None
        )
    enabled = _systemctl(root, deadline, "is-enabled", _IMAGE_TIMER)
    active = _systemctl(root, deadline, "is-active", _IMAGE_TIMER)
    service = _systemctl(root, deadline, "is-active", _IMAGE_SERVICE)
    for reading in (enabled, active, service):
        if reading.returncode not in (0, 1, 3, 4):
            raise RuntimeError("image-prune unit state is unreadable")
    if service.returncode == 0:
        raise ValueError("image-prune service is already running; retry when idle")
    _write(
        root / "unit-snapshot.json",
        {
            "files": files,
            "timer_enabled": enabled.returncode == 0,
            "timer_active": active.returncode == 0,
        },
    )
    # The timer expects this exact console path. It is a source-bound node
    # adapter using the host's existing dispatch dependencies, not a venv sync
    # or an edit to the dispatch environment. Keep it inside the leased tree.
    entry = tree / ".venv/bin/onex"
    if (tree / ".venv").exists() or (tree / ".venv").is_symlink():
        raise ValueError("image-prune source must not carry an existing venv")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "#!" + str(dispatch) + "\n"
        "# SPDX-License-Identifier: MIT\n"
        "import sys\n"
        "sys.path.insert(0, " + repr(str(tree / "src")) + ")\n"
        "from omnibase_infra.cli.cli_node import run_node_by_name\n"
        "if sys.argv[1:2] != ['node']: raise SystemExit(2)\n"
        "args = sys.argv[2:]\n"
        "if '--contract' not in args: args += ['--contract', "
        + repr(str(contract))
        + "]\n"
        "run_node_by_name.main(args=args, prog_name='onex node')\n"
    )
    entry.chmod(0o700)
    argv = [
        str(entry),
        "node",
        _IMAGE_NODE,
        "--contract",
        str(contract),
        "--input",
        str(manifest),
        "--backend",
        "event_bus=inmemory",
        "--output",
        "receipt",
        "--state-root",
        str(root / "installer"),
        "--timeout",
        str(max(1, int(deadline - time.time()))),
    ]
    output = json.loads(_require_ok(_action(argv, cwd=tree, deadline=deadline)))
    installed = output.get("result")
    if (
        output.get("status") != "success"
        or not isinstance(installed, dict)
        or installed.get("success") is not True
    ):
        raise RuntimeError("image-prune node did not return an installation receipt")
    revision = installed.get("installed_revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("image-prune node omitted installed revision")
    _require_ok(_systemctl(root, deadline, "start", _IMAGE_SERVICE))
    observation = _require_ok(
        _systemctl(
            root,
            deadline,
            "show",
            _IMAGE_SERVICE,
            "--property=Result",
            "--property=ExecMainStatus",
        )
    )
    if set(observation.splitlines()) != {"Result=success", "ExecMainStatus=0"}:
        raise RuntimeError("image-prune unit tick did not read back success")
    return revision, observation.strip()


def supervise_trial(
    request: ModelDeployAgentBranchTrialRequest,
    *,
    repo_root: Path,
    state_dir: Path,
    deadline: float | None = None,
    on_restore: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> ModelDeployAgentBranchTrialResult:
    """Run a pinned head and restore dev even when startup or an action fails."""
    from omnibase_infra.models.model_deploy_agent_branch_trial_result import (
        ModelDeployAgentBranchTrialResult,
    )

    _guard()
    if request.target != "dev":
        raise ValueError("agent branch trials admit only dev")
    deadline = (
        deadline if deadline is not None else time.time() + request.duration_seconds
    )
    root = _root(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    tree = root / str(request.correlation_id) / "tree"
    tree.parent.mkdir(parents=True, exist_ok=True)
    result = ModelDeployAgentBranchTrialResult(
        correlation_id=request.correlation_id,
        requested_head=request.head,
        status="failed",
    )
    candidate = None
    fallback = None
    return_head = _git(repo_root, "rev-parse", "origin/dev^{commit}")
    try:
        if _git(repo_root, "rev-parse", "HEAD") != return_head:
            raise ValueError("baseline must already be origin/dev before a trial")
        if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("baseline has tracked changes; refusing a branch trial")
        _git(repo_root, "check-ref-format", "refs/heads/" + request.branch)
        _git(
            repo_root,
            "fetch",
            "--no-tags",
            "origin",
            "refs/heads/" + request.branch,
            timeout=min(30, deadline - time.time()),
        )
        if _git(repo_root, "rev-parse", "FETCH_HEAD^{commit}") != request.head:
            raise ValueError("pushed branch head differs from the requested head")
        _git(
            repo_root,
            "worktree",
            "add",
            "--detach",
            str(tree),
            request.head,
            timeout=min(30, deadline - time.time()),
        )
        candidate = _spawn(tree, trial=True)
        _wait_loaded(candidate, request.head, min(deadline, time.time() + 30))
        result = result.model_copy(
            update={
                "status": "running",
                "loaded_code_sha": request.head,
                "observed_at": datetime.now(UTC),
            }
        )
        _write(root / "receipt.json", result.model_dump(mode="json"))
        if request.installer:
            revision, observation = _install_image_prune(tree, root, deadline)
            result = result.model_copy(
                update={
                    "installed_revision": revision,
                    "installer_observed": observation,
                }
            )
            _write(root / "receipt.json", result.model_dump(mode="json"))
        while time.time() < deadline:
            if candidate.poll() is not None:
                raise RuntimeError(f"candidate exit={candidate.returncode}")
            if _health().get("loaded_code_sha") != request.head:
                raise RuntimeError("candidate loaded-code readback changed")
            time.sleep(min(0.2, max(0, deadline - time.time())))
        result = result.model_copy(update={"status": "expired"})
    except (SystemExit, KeyboardInterrupt):
        result = result.model_copy(
            update={
                "status": "failed",
                "error": "supervisor interrupted; returning to origin/dev",
            }
        )
        raise
    except Exception as exc:
        logger.exception("Dev agent branch trial failed")
        result = result.model_copy(update={"status": "failed", "error": str(exc)})
    finally:
        if candidate is not None:
            _stop(candidate)
        try:
            _restore_units(root, time.time() + 30)
        except Exception as exc:
            logger.exception("Dev branch unit restoration failed")
            result = result.model_copy(
                update={
                    "status": "failed",
                    "error": result.error + f"; unit restoration: {exc}",
                }
            )
        try:
            # The normal tracking ref may advance while the candidate runs.
            return_head = _git(repo_root, "rev-parse", "origin/dev^{commit}")
            _git(repo_root, "merge", "--ff-only", return_head)
            fallback = _spawn(repo_root, trial=False)
            _wait_loaded(fallback, return_head, time.time() + 30)
            result = result.model_copy(update={"restored_loaded_code_sha": return_head})
            if on_restore is not None:
                on_restore(fallback)
                fallback = None
        except Exception as exc:
            logger.exception("Dev agent restoration failed")
            result = result.model_copy(
                update={
                    "status": "failed",
                    "error": result.error + f"; dev restoration: {exc}",
                }
            )
        finally:
            _write(root / "receipt.json", result.model_dump(mode="json"))
            (root / "request.json").unlink(missing_ok=True)
            if fallback is not None:
                _stop(fallback)
            if tree.exists() and not (root / "unit-snapshot.json").exists():
                _git(repo_root, "worktree", "remove", "--force", str(tree))
    return result


def _supervise(args: Sequence[str]) -> int:
    """Service-lifetime backend: only the dev launcher starts this parent."""
    from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
        ModelDeployAgentBranchTrialRequest,
    )

    _guard()
    repo = (
        Path(str(_HOST.get().repo_root / "scripts/deploy-agent")).resolve().parents[1]
    )
    state = Path(str(_HOST.get().state_dir))
    root = _root(state)
    child: subprocess.Popen[bytes] | None = None

    def stop(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with _lock(root / "supervisor.lock"):
        # A parent restart never resumes a lease. systemd kills the whole
        # cgroup; start dev before accepting another request.
        if read_receipt(state).get("status") in ACTIVE_STATUSES:
            record = read_receipt(state)
            record.update(
                status="failed", error="supervisor restarted; returning to origin/dev"
            )
            _write(root / "receipt.json", record)
        (root / "request.json").unlink(missing_ok=True)
        try:
            _restore_units(root, time.time() + 30)
        except Exception:
            # A unit-manager failure must not keep the agent off dev. Retain
            # the snapshot and refuse further trials until recovery can run.
            logger.exception("Retry of previous branch unit restoration failed")
        try:
            child = _spawn(repo, trial=False, args=args)
            while True:
                _write(
                    root / "supervisor.json",
                    {
                        "pid": os.getpid(),
                        "observed_at": time.time(),
                        "target": "dev",
                        "return_ref": "origin/dev",
                        "valid_until": time.time()
                        + _policy().heartbeat_timeout_seconds,
                    },
                )
                pending = root / "request.json"
                if pending.exists():
                    data = json.loads(pending.read_text())
                    health = _health()
                    if time.time() >= data["deadline"]:
                        record = read_receipt(state)
                        record.update(
                            status="failed",
                            error="trial expired before agent became idle",
                        )
                        _write(root / "receipt.json", record)
                        pending.unlink(missing_ok=True)
                        continue
                    if (
                        health.get("state") == "idle"
                        and not health.get("active_job")
                        and not health.get("pending_publish_count")
                    ):
                        request = ModelDeployAgentBranchTrialRequest.model_validate(
                            data["request"]
                        )
                        _stop(child)
                        child = None

                        def keep_restored(process: subprocess.Popen[bytes]) -> None:
                            nonlocal child
                            child = process

                        supervise_trial(
                            request,
                            repo_root=repo,
                            state_dir=state,
                            deadline=data["deadline"],
                            on_restore=keep_restored,
                        )
                        if child is None:
                            child = _spawn(repo, trial=False, args=args)
                if child.poll() is not None:
                    return child.returncode or 0
                time.sleep(0.2)
        finally:
            if child is not None:
                _stop(child)
            (root / "supervisor.json").unlink(missing_ok=True)


def supervise_from_environment(environment: dict[str, str], args: Sequence[str]) -> int:
    configure_host(environment)
    return _supervise(args)


def queue_request(
    request: ModelDeployAgentBranchTrialRequest, state_root: Path
) -> ModelDeployAgentBranchTrialResult:
    from omnibase_infra.models.model_deploy_agent_branch_trial_result import (
        ModelDeployAgentBranchTrialResult,
    )

    if request.target != "dev":
        raise ValueError("agent branch trials admit only dev")
    state = state_root
    root = _root(state)
    heartbeat = json.loads((root / "supervisor.json").read_text())
    if heartbeat.get("target") != "dev" or heartbeat.get("return_ref") != "origin/dev":
        raise ValueError("trusted branch supervisor must be dev-only")
    if time.time() > heartbeat["valid_until"]:
        raise ValueError("trusted dev branch supervisor is stale")
    os.kill(int(heartbeat["pid"]), 0)
    with _lock(root / "request.lock"):
        if read_receipt(state).get("status") in ACTIVE_STATUSES:
            raise ValueError("a dev agent branch trial is already active")
        if (root / "unit-snapshot.json").exists():
            raise ValueError("previous dev trial unit restoration is pending")
        result = ModelDeployAgentBranchTrialResult(
            correlation_id=request.correlation_id,
            requested_head=request.head,
            status="queued",
        )
        # The receipt precedes the queue: a parent can never finish a trial
        # and then have its readback overwritten by the registrar.
        _write(root / "receipt.json", result.model_dump(mode="json"))
        _write(
            root / "request.json",
            {
                "request": request.model_dump(mode="json"),
                "deadline": time.time() + request.duration_seconds,
            },
        )
    return result


def validate_rebuild_command(command: ModelRebuildRequested) -> ModelRebuildRequested:
    import re

    from deploy_agent.events import EnumRuntimeLane, Scope, services_for_scope

    if command.scope is Scope.AGENT_TRIAL:
        if command.runtime_lane is not EnumRuntimeLane.DEV:
            raise ValueError("agent-trial requires runtime_lane dev")
        if command.agent_branch is None or command.agent_trial_seconds is None:
            raise ValueError("agent-trial requires a branch and bounded duration")
        if re.fullmatch(r"[0-9a-f]{40}", command.git_ref) is None:
            raise ValueError("agent-trial requires the pushed full head SHA")
        if (
            command.services
            or command.image_ref
            or command.image_digest
            or command.rollback
            or command.open_pr
        ):
            raise ValueError("agent-trial cannot carry a runtime deployment")
    elif (
        command.agent_branch is not None
        or command.agent_trial_seconds is not None
        or command.agent_installer is not None
    ):
        raise ValueError("agent trial fields require scope agent-trial")

    if command.requested_at is not None and command.requested_at.utcoffset() is None:
        msg = (
            f"requested_at={command.requested_at.isoformat()} carries no UTC "
            "offset; a publish time compared across hosts must be aware"
        )
        raise ValueError(msg)

    if command.open_pr and (
        command.runtime_lane != EnumRuntimeLane.DEV
        or command.rollback is None
        or not command.open_pr.endswith(f"@{command.git_ref}")
    ):
        msg = (
            "open_pr needs runtime_lane dev, a rollback declaration, git_ref = its head"
        )
        raise ValueError(msg)

    if command.services:
        # OMN-18108: lane-aware, so a dev command may name one of the
        # dev-lane-only services and a prod/stability command may not.
        allowed = services_for_scope(command.scope, lane=command.runtime_lane)
        invalid = [s for s in command.services if s not in allowed]
        if invalid:
            msg = (
                f"Services {invalid} not in scope '{command.scope}'. Allowed: {allowed}"
            )
            raise ValueError(msg)

    if command.runtime_lane == EnumRuntimeLane.PROD and not command.image_digest:
        msg = (
            "prod runtime_lane requires image_digest: production deploys the "
            "exact stability-proven digest and never rebuilds from a ref"
        )
        raise ValueError(msg)
    return command


def queue_agent_trial(
    host: DeployAgent,
    cmd: ModelRebuildRequested,
    state_dir: Path,
    queue: Callable[
        [Path, ModelDeployAgentBranchTrialRequest], ModelDeployAgentBranchTrialResult
    ],
) -> None:
    """Adapt the signed bus command to the canonical host effect.

    Success here acknowledges queueing only. /agent-branch-trial serves
    the parent's actual loaded-code and restoration readbacks.
    """
    from deploy_agent.events import EnumRuntimeLane

    host._state = "deploying"
    try:
        from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
            ModelDeployAgentBranchTrialRequest,
        )

        if (
            cmd.runtime_lane is not EnumRuntimeLane.DEV
            or cmd.agent_branch is None
            or cmd.agent_trial_seconds is None
        ):
            raise ValueError("agent trial requires a bounded dev request")
        request = ModelDeployAgentBranchTrialRequest(
            correlation_id=cmd.correlation_id,
            target="dev",
            branch=cmd.agent_branch,
            head=cmd.git_ref,
            duration_seconds=cmd.agent_trial_seconds,
            installer=cmd.agent_installer,
        )

        queue(state_dir, request)
        host.job_store.complete(cmd.correlation_id, status="success")
    except Exception as exc:
        logger.exception("Agent branch trial request refused")
        host.job_store.complete(cmd.correlation_id, status="failed", errors=[str(exc)])
    finally:
        host._state = "idle"


def branch_installs_readback(state_dir: Path) -> dict[str, object]:
    """The declared installs and the last readback, for ``/branch-installs``."""
    root = state_dir / "branch-installs"
    try:
        declared = json.loads((root / "records.json").read_text(encoding="utf-8"))[
            "installs"
        ]
    except FileNotFoundError:
        declared = []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"installs": None, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = None
    readbacks = status.get("readbacks") if isinstance(status, dict) else None
    return {
        "read_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "declared": declared,
        "last_check": status,
        "missing": [
            str(r.get("install_key"))
            for r in (readbacks if isinstance(readbacks, list) else [])
            if isinstance(r, dict) and r.get("state") in ("missing", "failed")
        ],
    }


def last_completed_readback(store: JobStore) -> dict[str, object] | None:
    from deploy_agent.job_state import JobState

    last_result: dict[str, object] | None = None
    completed_jobs: list[tuple[datetime, JobState]] = []
    for path in store.state_dir.glob("*.json"):
        try:
            job = JobState.model_validate_json(path.read_text())
            completed_at = job.completed_at
            if job.status in ("success", "failed") and completed_at is not None:
                completed_jobs.append((completed_at, job))
        except (OSError, ValueError, TypeError):
            continue
    if completed_jobs:
        completed_jobs.sort(key=lambda pair: pair[0], reverse=True)
        latest = completed_jobs[0][1]
        last_result = {
            "correlation_id": str(latest.correlation_id),
            "status": latest.status,
            "completed_at": latest.completed_at.isoformat()
            if latest.completed_at
            else None,
            "phase_results": {str(k): str(v) for k, v in latest.phase_results.items()},
            # OMN-18636 AC5. A reader that only looks at /health must be able to
            # tell a finished job from one whose post-terminal work is still
            # running on this agent's job thread.
            "settling": latest.settling_stage is not None,
            "settling_stage": (
                latest.settling_stage.value if latest.settling_stage else None
            ),
        }

    return last_result


def broker_timestamp(msg: object) -> datetime | None:
    """The record's broker timestamp, the lineage fence's fallback publish time.

    OMN-19270. On this topic it is set by the hop that FORWARDS the command,
    just before this agent reads it, so it is never earlier than the running
    build's start: a command that carries no ``requested_at`` is therefore never
    superseded on time, which is the direction that builds.
    """
    raw = getattr(msg, "timestamp", None)
    if not isinstance(raw, int) or raw < 0:
        return None
    return datetime.fromtimestamp(raw / 1000, tz=UTC)


def command_body(command: ModelRebuildRequested) -> dict[str, object]:
    from deploy_agent.events import Scope

    body = command.model_dump(mode="json")
    if command.scope is not Scope.AGENT_TRIAL:
        for field in ("agent_branch", "agent_trial_seconds", "agent_installer"):
            body.pop(field)
    return body


def extend_trigger_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-branch", help="Pushed branch for --scope agent-trial")
    parser.add_argument(
        "--agent-trial-seconds", type=int, help="Trial lifetime, 1..3600 seconds"
    )
    parser.add_argument(
        "--agent-installer",
        choices=["image-prune"],
        help="Run the pinned head's image-prune install.json on this host",
    )


def resolve_git_ref(raw: str | None) -> str:
    from deploy_agent.tracking_ref import (
        ENV_TRACKING_REF,
        load_tracking_remote_ref_from_env,
    )
    from deploy_agent.trigger import TriggerRefusedError

    if raw:
        return raw
    try:
        return load_tracking_remote_ref_from_env()
    except RuntimeError as exc:
        raise TriggerRefusedError(
            f"no --git-ref given and {ENV_TRACKING_REF} is not set.\n"
            f"       Pass --git-ref origin/<branch>, or export\n"
            f"       {ENV_TRACKING_REF}=<branch> (e.g. dev) to supply the "
            f"default.\n       {exc}"
        ) from exc


def validate_completed(command: ModelRebuildCompleted) -> None:
    from deploy_agent.events import Phase, PhaseStatus

    if Phase.PUBLISH in command.phase_results:
        raise ValueError(
            "phase_results must not carry Phase.PUBLISH: the completion "
            "event is the publish and cannot report its own outcome"
        )
    unsettled = sorted(
        phase.value
        for phase, status in command.phase_results.items()
        if status in (PhaseStatus.IN_PROGRESS, PhaseStatus.PENDING)
    )
    if unsettled:
        raise ValueError(
            f"phase_results carries unsettled verdicts for {unsettled}: a "
            "terminal event must report failed/skipped for a phase that "
            "raised or was never reached"
        )
