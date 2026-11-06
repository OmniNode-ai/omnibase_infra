# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay home's ``runtime_lane_env`` overlay as a second source of a lane's env (OMN-22634).

The host deploy agent builds a lane's compose environment from the operator store and, for the dev
lane, the lane's deployment fragment. The overlay home (the private repository of deployment overlays)
holds ``overlays/runtime_lane_env/overlay.yaml``: a schema version and, per runtime lane id, the env that
lane's deploys take. The agent reads it from the clone its unit names in ``DEPLOY_AGENT_OVERLAY_HOME_CLONE``
at the clone's ``origin/main``, never its working tree, and puts each lane's env through the fragment
parser it passes in, so both sources share one allowlist and one set of line rules.

The variable has no default: a unit that names no clone reads no overlay-home source, and a unit that
names a clone the agent cannot read refuses, because a clone that cannot be read is not the same as an
overlay that holds nothing. A key the two sources set to different values is refused by name, never
resolved by picking one.

Loaded by path by the standalone agent (``executor.load_dev_boot_guard``), so it imports the standard
library, pyyaml and omnibase_core only.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

logger = logging.getLogger(__name__)

#: The fragment parser's shape: ``parse(text, lane_id=..., source=...)`` returns the lane's env.
FragmentParser = Callable[..., dict[str, str]]


class OverlayHomeLaneEnvError(RuntimeError):
    """The overlay home's runtime_lane_env overlay cannot be applied as written."""


def declaration(execution: ModuleType) -> Mapping[str, str | int]:
    """The node contract's ``config.overlay_home_lane_env``: clone variable, ref, overlay path, read timeout.

    ``execution`` is the deploy agent's executor module, which locates the agent's checkout, as for
    the estimator build's settings.
    """
    module_file = execution.__file__
    if not module_file:
        raise OverlayHomeLaneEnvError(
            "the overlay home source needs the deploy agent source location"
        )
    contract = (
        Path(module_file).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    config = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"]
    declared: Mapping[str, str | int] = config["overlay_home_lane_env"]
    return declared


def overlay_home_clone(
    environ: Mapping[str, str], declared: Mapping[str, str | int]
) -> Path | None:
    """The overlay home's clone the unit names, or ``None`` when it names none."""
    clone_env = str(declared["clone_env"])
    raw = environ.get(clone_env, "").strip()
    if not raw:
        return None
    clone = Path(raw)
    if not clone.is_absolute():
        raise OverlayHomeLaneEnvError(
            f"{clone_env} must name an absolute path, got a relative one"
        )
    return clone


def _document_problems(raw: object) -> list[str]:
    """Why ``raw`` is not a runtime_lane_env document, as field-named reasons."""
    if not isinstance(raw, dict):
        return ["<document>: must be a mapping"]
    problems = [
        f"{key}: not a field"
        for key in sorted(set(raw) - {"schema_version", "lanes"}, key=str)
    ]
    if not isinstance(raw.get("schema_version"), str):
        problems.append("schema_version: must be a string")
    lanes = raw.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        return [*problems, "lanes: must be a non-empty mapping"]
    for lane, entry in lanes.items():
        if not isinstance(lane, str) or not isinstance(entry, dict):
            problems.append(f"lanes.{lane}: must be a mapping under a string lane id")
            continue
        problems += [
            f"lanes.{lane}.{key}: not a field"
            for key in sorted(set(entry) - {"description", "env"}, key=str)
        ]
        env = entry.get("env")
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            problems.append(
                f"lanes.{lane}.env: must be a mapping of strings to strings"
            )
        if not isinstance(entry.get("description", ""), str):
            problems.append(f"lanes.{lane}.description: must be a string")
    return problems


def parse_overlay_home_lane_env(
    text: str, *, lane_id: str, source: str, parse_fragment: FragmentParser
) -> dict[str, str]:
    """``lane_id``'s env from the overlay's text, or nothing for a lane it omits.

    The document is validated whole, so a stray key anywhere refuses; the lane's own env
    then goes through ``parse_fragment``.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OverlayHomeLaneEnvError(f"{source}: not YAML: {exc}") from exc
    problems = _document_problems(raw)
    if problems:
        raise OverlayHomeLaneEnvError(
            f"{source}: invalid overlay ({'; '.join(problems)})"
        )
    lane = raw["lanes"].get(lane_id)
    # The version is checked for a lane the overlay omits too: an unreadable document is not an empty one.
    fragment = {
        "schema_version": raw["schema_version"],
        "lane_id": lane_id,
        "description": (lane or {}).get("description", ""),
        "env": (lane or {}).get("env", {}),
    }
    env = parse_fragment(json.dumps(fragment), lane_id=lane_id, source=source)
    return env if lane is not None else {}


def read_overlay_home_lane_env(
    clone: Path,
    lane_id: str,
    *,
    declared: Mapping[str, str | int],
    parse_fragment: FragmentParser,
) -> dict[str, str]:
    """``lane_id``'s env from the overlay home ``clone`` at the declared ref; empty when it holds none.

    An absent overlay file at the ref, or a lane the overlay omits, changes nothing. A clone
    that is missing, is not a git clone, or holds no ``ref`` refuses, because the unit named it.
    """
    ref = str(declared["ref"])
    relpath = str(declared["overlay_path"])
    if not clone.is_dir():
        raise OverlayHomeLaneEnvError(
            f"{declared['clone_env']} names {clone}, which is not a directory"
        )

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=int(declared["read_timeout_seconds"]),
            env=scrub_git_location_env(),
        )

    if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        raise OverlayHomeLaneEnvError(
            f"{clone} holds no {ref}; the overlay home's runtime_lane_env overlay cannot be "
            "read, and a clone that cannot be read is not the same as an overlay that holds nothing"
        )
    if git("cat-file", "-e", f"{ref}:{relpath}").returncode != 0:
        return {}
    shown = git("show", f"{ref}:{relpath}")
    if shown.returncode != 0:
        raise OverlayHomeLaneEnvError(
            f"git show {ref}:{relpath} in {clone} exited "
            f"{shown.returncode}: {shown.stderr.strip()[:240]}"
        )
    return parse_overlay_home_lane_env(
        shown.stdout,
        lane_id=lane_id,
        source=f"{clone}@{ref}:{relpath}",
        parse_fragment=parse_fragment,
    )


def merge_lane_env_sources(
    fragment: Mapping[str, str], overlay_home: Mapping[str, str]
) -> dict[str, str]:
    """The two sources' env as one, refusing a key they set to different values."""
    conflicts = sorted(
        key
        for key in fragment.keys() & overlay_home.keys()
        if fragment[key] != overlay_home[key]
    )
    if conflicts:
        raise OverlayHomeLaneEnvError(
            f"{', '.join(conflicts)} set to different values by the deployment fragment and "
            "the overlay home's runtime_lane_env overlay; set each key in one source, "
            "or the same value in both"
        )
    return {**fragment, **overlay_home}


class _LoggedLaneEnv:
    """The env last logged, so a change is logged once, not per compose command."""

    last: dict[str, str] | None = None


def lane_env_with_overlay_home(
    execution: ModuleType,
    fragment_env: Mapping[str, str],
    environ: Mapping[str, str],
    *,
    lane_id: str,
    fragment_clone: Path,
    parse_fragment: FragmentParser,
) -> dict[str, str]:
    """``fragment_env`` plus the overlay home's env for ``lane_id``, when the unit names a clone.

    Logs the merged env once per change: names and values of non-secret lane settings, the
    same line the agent logged for the fragment alone.
    """
    declared = declaration(execution)
    clone = overlay_home_clone(environ, declared)
    overlay = dict(fragment_env)
    if clone is not None:
        overlay = merge_lane_env_sources(
            fragment_env,
            read_overlay_home_lane_env(
                clone, lane_id, declared=declared, parse_fragment=parse_fragment
            ),
        )
    if overlay != _LoggedLaneEnv.last:
        logger.info(
            "lane env fragment for %s from %s%s: %s",
            lane_id,
            fragment_clone,
            f" and {clone}" if clone is not None else "",
            ", ".join(f"{k}={v}" for k, v in sorted(overlay.items())) or "none",
        )
        _LoggedLaneEnv.last = dict(overlay)
    return overlay
