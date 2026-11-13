# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the dev lane's LLM endpoint env from the llm_endpoints overlay (OMN-24025).

For every contract slot flagged ``endpoint_from_overlay`` this module renders
the env variable that carries the slot's address from the overlay, and, when
the slot carries a lane overlay binding, a rewritten copy of the lane bifrost
overlay directory in which the bound backend points at the same overlay
address.

Loaded by path by the standalone agent (``executor.load_dev_boot_guard``), so
it imports the standard library, pyyaml and the repo's own models only.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import NamedTuple
from urllib.parse import urlparse

import yaml

from omnibase_infra.models.contracts.model_llm_endpoint_contract import (
    ModelLlmEndpointContract,
)
from omnibase_infra.models.contracts.model_llm_endpoint_entry import (
    ModelLlmEndpointEntry,
)
from omnibase_infra.models.contracts.model_llm_endpoint_lane_overlay_binding import (
    ModelLlmEndpointLaneOverlayBinding,
)
from omnibase_infra.models.contracts.model_llm_endpoint_overlay import (
    ModelLlmEndpointOverlay,
)


class LlmEndpointEnvError(RuntimeError):
    """The llm_endpoints overlay cannot be rendered as written."""


class RenderedLlmEndpointEnv(NamedTuple):
    """The rendered env and the lane overlay directory, when a binding applied."""

    env: dict[str, str]
    lane_overlay_dir: str | None


def _repo_root(execution: ModuleType) -> Path:
    module_file = execution.__file__
    if not module_file:
        raise LlmEndpointEnvError(
            "the llm endpoint env source needs the deploy agent source location"
        )
    return Path(module_file).resolve().parents[3]


def declaration(execution: ModuleType) -> Mapping[str, object]:
    """The node contract's ``config.llm_endpoint_env_render`` block.

    ``execution`` is the deploy agent's executor module, which locates the
    agent's checkout, as for the estimator build's settings.
    """
    contract = (
        _repo_root(execution)
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    config = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"]
    declared: Mapping[str, object] = config["llm_endpoint_env_render"]
    return declared


def _load_contract(
    execution: ModuleType, declared: Mapping[str, object]
) -> ModelLlmEndpointContract:
    root = _repo_root(execution)
    path = root / str(declared["contract_path"])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelLlmEndpointContract.model_validate(raw)


def _flagged_slots(contract: ModelLlmEndpointContract) -> list[ModelLlmEndpointEntry]:
    return [e for e in contract.endpoints if e.endpoint_from_overlay]


def _read_overlay(
    environ: Mapping[str, str], flagged: list[ModelLlmEndpointEntry]
) -> ModelLlmEndpointOverlay:
    try:
        overlay = ModelLlmEndpointOverlay.read(environ)
    except ValueError as exc:
        raise LlmEndpointEnvError(str(exc)) from exc
    if overlay is None:
        slot_ids = ", ".join(e.slot_id for e in flagged)
        raise LlmEndpointEnvError(
            f"no llm_endpoints overlay found; flagged slots: {slot_ids}"
        )
    return overlay


def _resolve_base_url(
    overlay: ModelLlmEndpointOverlay, slot: ModelLlmEndpointEntry
) -> str:
    if slot.slot_id not in overlay.endpoint_urls:
        raise LlmEndpointEnvError(
            f"slot {slot.slot_id!r} is not in the llm_endpoints overlay endpoint_urls"
        )
    return overlay.endpoint_urls[slot.slot_id]


def _write_if_changed(path: Path, data: bytes) -> None:
    """Write ``data`` unless the file already holds it, so mtimes stay put."""
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _rebound_document(
    relative: Path,
    text: str,
    binding: ModelLlmEndpointLaneOverlayBinding,
    base_url: str,
) -> tuple[bytes, str] | None:
    """The lane overlay's bytes with the bound backend moved to ``base_url``, and its new URL.

    ``None`` when the document does not hold the bound backend.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict) or not isinstance(document.get("backends"), list):
        return None
    for backend in document["backends"]:
        if (
            not isinstance(backend, dict)
            or backend.get("backend_id") != binding.backend
        ):
            continue
        old_url = backend.get("endpoint_url")
        parsed = urlparse(old_url) if isinstance(old_url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            raise LlmEndpointEnvError(
                f"{relative}: backend {binding.backend!r} endpoint_url {old_url!r} "
                "must be an http(s) URL with a host"
            )
        new_url = f"{base_url}{parsed.path}" + (
            f"?{parsed.query}" if parsed.query else ""
        )
        backend["endpoint_url"] = new_url
        return yaml.safe_dump(document, sort_keys=False).encode("utf-8"), new_url
    return None


def _rewrite_lane_overlay(
    source: Path,
    target: Path,
    binding: ModelLlmEndpointLaneOverlayBinding,
    base_url: str,
    lane_overlay_glob: str,
) -> str | None:
    """Copy ``source`` to ``target`` with the bound backend moved; the agreed URL or None."""
    urls: set[str] = set()
    for source_file in sorted(source.rglob("*")):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        data = source_file.read_bytes()
        if fnmatch.fnmatch(source_file.name, lane_overlay_glob):
            rebound = _rebound_document(
                relative, data.decode("utf-8"), binding, base_url
            )
            if rebound is not None:
                data, url = rebound
                urls.add(url)
        _write_if_changed(target / relative, data)
    if len(urls) > 1:
        raise LlmEndpointEnvError(
            f"backend {binding.backend!r} has differing endpoint URLs across "
            f"lane overlay files: {sorted(urls)}"
        )
    return next(iter(urls), None)


def _remove_stale(target: Path, source: Path) -> None:
    """Delete what ``target`` holds that ``source`` does not, empty directories too."""
    if not target.is_dir():
        return
    for stale in sorted(target.rglob("*"), reverse=True):
        if stale.is_file() and not (source / stale.relative_to(target)).is_file():
            stale.unlink()
        elif stale.is_dir() and not any(stale.iterdir()):
            stale.rmdir()


def render(
    execution: ModuleType,
    environ: Mapping[str, str],
    *,
    lane_overlay_dir: str | None,
) -> RenderedLlmEndpointEnv:
    """Render the dev lane's LLM endpoint env from the overlay."""
    declared = declaration(execution)
    contract = _load_contract(execution, declared)
    flagged = _flagged_slots(contract)
    if not flagged:
        return RenderedLlmEndpointEnv(env={}, lane_overlay_dir=None)

    overlay = _read_overlay(environ, flagged)
    env: dict[str, str] = {}
    rendered_dir: str | None = None

    for slot in flagged:
        base_url = _resolve_base_url(overlay, slot)
        if slot.url_env_var:
            env[slot.url_env_var] = base_url

        if slot.lane_overlay_binding and lane_overlay_dir is not None:
            state_dir_env = str(declared["state_dir_env"])
            state_dir_raw = environ.get(state_dir_env, "").strip()
            if not state_dir_raw:
                raise LlmEndpointEnvError(
                    f"{state_dir_env} must name the state directory"
                )
            state_dir = Path(state_dir_raw)
            target = state_dir / str(declared["rendered_dir_name"])
            src = Path(lane_overlay_dir)
            if not src.is_dir():
                raise LlmEndpointEnvError(f"{lane_overlay_dir} is not a directory")
            lane_overlay_glob = str(declared["lane_overlay_glob"])
            agreed = _rewrite_lane_overlay(
                src, target, slot.lane_overlay_binding, base_url, lane_overlay_glob
            )
            _remove_stale(target, src)
            if agreed is not None:
                env[slot.lane_overlay_binding.env_var] = agreed
            rendered_dir = str(target)

    return RenderedLlmEndpointEnv(env=env, lane_overlay_dir=rendered_dir)


def render_dev_lane(
    execution: ModuleType, environ: Mapping[str, str]
) -> tuple[dict[str, str], str | None]:
    """The dev lane's LLM endpoint env rendered from the llm_endpoints overlay, or nothing.

    The address of every contract slot flagged ``endpoint_from_overlay`` is the overlay's,
    never the operator store's: the store keeps whatever a person last typed, and compose
    reads this render over it. The lane bifrost overlay directory comes back rewritten when
    the slot is bound to one of its backends. Fenced like the lane env fragment: a deploy of
    any other lane takes nothing from here, and a dev deploy whose overlay cannot be read is
    refused, never run on a stale address.
    """
    raw = environ.get(execution.ENV_ALLOWED_LANES, "").strip()
    if not (
        raw
        and execution.is_lab_dev_fence(execution.parse_allowed_lanes(raw))
        and execution._ActiveDeployLane.lane in (None, execution.EnumRuntimeLane.DEV)
    ):
        return {}, None
    try:
        rendered = render(
            execution,
            environ,
            lane_overlay_dir=environ.get(execution.LANE_OVERLAY_DIR_ENV),
        )
    except LlmEndpointEnvError as exc:
        raise RuntimeError(
            "the dev lane's LLM endpoint env cannot be rendered from the "
            f"llm_endpoints overlay: {exc}"
        ) from exc
    return dict(rendered.env), rendered.lane_overlay_dir


def refresh_exports(
    execution: ModuleType,
    environ: Mapping[str, str],
    search_root: Path,
    state_dir: Path,
) -> str:
    """Shell exports of the dev lane's LLM endpoint env rendered from the overlay, for the warm refresh.

    ``scripts/runtime_build/refresh_dev_lane.sh`` sources the operator env file, which keeps
    whatever a person last typed for the embedding address (OMN-24081). The warm refresh takes
    the render the deploy agent takes (``render_dev_lane``) so it never puts the removed
    address back. The refresh is dev only, so this skips the agent's lane fence. The overlay
    home clone is the one under ``search_root`` that carries lane fragments, unless ``environ``
    already names one; the rewritten lane overlay directory lands under ``state_dir`` unless
    ``environ`` names a state directory. An overlay that cannot be rendered is a RuntimeError.
    """
    from deploy_agent.instance_lanes import (  # the agent source tree holds this
        discover_lane_overlay_clone,
        lane_env_shell_exports,
    )

    env = dict(environ)
    clone = discover_lane_overlay_clone(search_root)
    if clone is not None:
        env.setdefault("DEPLOY_AGENT_OVERLAY_HOME_CLONE", str(clone))
    env.setdefault(str(declaration(execution)["state_dir_env"]), str(state_dir))
    try:
        rendered = render(
            execution,
            env,
            lane_overlay_dir=execution.resolve_lane_overlay_dir(env),
        )
    except LlmEndpointEnvError as exc:
        raise RuntimeError(
            "the dev lane's LLM endpoint env cannot be rendered from the "
            f"llm_endpoints overlay: {exc}"
        ) from exc
    exports = dict(rendered.env)
    if rendered.lane_overlay_dir is not None:
        exports[execution.LANE_OVERLAY_DIR_ENV] = rendered.lane_overlay_dir
    return lane_env_shell_exports(exports)


def validate_env_contract(
    execution: ModuleType, operator_store: Path, explicit_env_file: str | None
) -> None:
    """Fail a runtime deploy when a configured LLM endpoint drifts from the contract.

    Moved here from ``DeployExecutor.validate_llm_endpoint_env_contract`` (OMN-24025); the
    method passes the executor module, the settings store its unit names (or the operator's
    default one) and ``OMNIBASE_ENV_FILE``, and every executor name below is read at call
    time. Compose reads the dev render over the store, so the process check runs on the
    environment compose reads, and a store file's own value for a name the dev render sets
    is superseded and not checked.
    """
    repo_dir = execution.REPO_DIR
    script = f"{repo_dir}/scripts/check_llm_endpoint_env_contract.py"
    venv_python = f"{repo_dir}/.venv/bin/python"
    base_cmd = (
        [venv_python, script]
        if Path(venv_python).is_file()
        else ["uv", "run", "--project", repo_dir, "python", script]
    )

    # OMN-19507: the store this agent was launched against, when its unit declares one, is
    # the file its lane's compose reads; another lab host's lane never reads the operator's
    # workstation file, whose drift must not refuse that lane's deploy.
    candidates = [operator_store, Path(repo_dir) / ".env"]
    if explicit_env_file:
        candidates.insert(0, Path(explicit_env_file))
    env_files: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in env_files:
            env_files.append(candidate)
    if explicit_env_file and Path(explicit_env_file) not in env_files:
        raise RuntimeError(
            "LLM endpoint env contract check failed: OMNIBASE_ENV_FILE does not "
            f"exist: {explicit_env_file}"
        )

    compose_env = execution._compose_env()
    slot_env, _ = execution._llm_endpoint_env_overlay(
        {k: v for k, v in compose_env.items() if k != execution.LANE_OVERLAY_DIR_ENV}
    )
    skip = [arg for name in sorted(slot_env) for arg in ("--skip-env-var", name)]
    commands = [base_cmd]
    commands.extend([*base_cmd, "--env-file", str(path), *skip] for path in env_files)
    for cmd in commands:
        result = execution._run(
            cmd,
            timeout=execution.PHASE_TIMEOUTS[execution.Phase.PREFLIGHT],
            env=None if "--env-file" in cmd else compose_env,
        )
        if result.returncode != 0:
            source = "process environment"
            if "--env-file" in cmd:
                source = cmd[cmd.index("--env-file") + 1]
            raise RuntimeError(
                "LLM endpoint env contract check failed for "
                f"{source}: {result.stderr.strip() or result.stdout.strip()}"
            )


__all__ = [
    "LlmEndpointEnvError",
    "RenderedLlmEndpointEnv",
    "declaration",
    "render",
    "refresh_exports",
    "render_dev_lane",
    "validate_env_contract",
]
