# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each deploy-agent instance's dev-lane composition, declared as data (OMN-19543).

WHY THIS EXISTS
---------------
OMN-19522 gave the second dev instance (``dev-202``) its own composition as a
Python literal in ``executor.DEV_INSTANCE_LANE_CONFIGS``. The operator's
2026-09-25 ruling puts a deploy slot on every lab host (omni_home ledger RULING
2026-09-25T10:22:22Z, "well for parllelism anyway"), so a literal per host is a
Python edit per host. This module reads the composition from the same table
that already names the instances, ``config/deploy_lane_routing.yaml``: an
instance other than the default carries a ``lane:`` block, and the executor
builds its ``ModelLaneConfig`` from that block. A new instance is a table row
and an overlay file, never a code change.

WHAT A ``lane:`` BLOCK SAYS
---------------------------
Everything that identified dev-202 in the literal, and nothing else:

* ``compose_overlay`` -- the instance's overlay, relative to the deploy-source
  clone, layered on the .201 dev lane's pair (infra plus dev-lane).
* ``compose_project`` -- the project it builds and runs under (both, as the
  literal did: compose names a built image ``<project>-<service>``).
* ``postgres_container`` and ``runtime_container`` -- its renamed containers.
* ``health_ports`` -- ``main`` and ``effects``, the host ports its overlay
  publishes the two runtime services on. The health targets keep the SERVICE
  names, as on the .201 dev lane (the verify recreate looks them up by label).
* ``disabled_phases`` -- the .201-only phases it never runs. Disabling
  ``gateway-deploy`` also disables the gateway project's services, which have
  no lane on a host without that phase.
* ``disabled_services`` -- the services its overlay disables by profile.
* ``receipt_lane`` and ``proves`` -- the lab-pass receipt name it emits and the
  repositories whose merges that receipt may prove. The executor does not read
  them; the receipt readers in ``scripts/ci`` do, from this same table.

THE DEFAULT INSTANCE HAS NO BLOCK. ``dev-201`` is the historical dev lane and
keeps the composition the executor has always had; the table may not redeclare
it, so there is one source for it.

PARSING IS STRICT
-----------------
An unknown key, an unknown phase, a missing field or a non-default instance
without a block refuses. The agent reads its own clone's table at start, so a
malformed table stops the agent there, as a malformed routing table already
does. ``routing.parse_routing_table`` ignores the ``lane:`` key, so an agent
reading a newer table at a command's ref still routes.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from deploy_agent.routing import ROUTING_TABLE_RELPATH, RoutingTableError

#: The key an instance's composition sits under in the routing table.
LANE_KEY: Final = "lane"


class ModelHealthPorts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    main: int
    effects: int


class ModelInstanceLane(BaseModel):
    """One instance's ``lane:`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_overlay: str
    compose_project: str
    postgres_container: str
    runtime_container: str
    health_ports: ModelHealthPorts
    disabled_phases: tuple[str, ...] = ()
    disabled_services: tuple[str, ...] = ()
    receipt_lane: str
    proves: tuple[str, ...]


def parse_instance_lanes(text: str) -> dict[str, ModelInstanceLane]:
    """Every non-default instance's block, keyed by instance name.

    Refuses when a non-default instance has no block, when the default
    instance has one, or when a block does not validate.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RoutingTableError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("instances"), dict):
        raise RoutingTableError("routing table declares no instances")
    default = str(raw.get("default_instance") or "").strip()
    lanes: dict[str, ModelInstanceLane] = {}
    for name, spec in raw["instances"].items():
        block = spec.get(LANE_KEY) if isinstance(spec, dict) else None
        if str(name) == default:
            if block is not None:
                raise RoutingTableError(
                    f"default instance {name!r} declares a {LANE_KEY}: block; it "
                    "keeps the dev lane's historical composition, which has one "
                    "source, the executor"
                )
            continue
        if not isinstance(block, dict):
            raise RoutingTableError(
                f"instance {name!r} declares no {LANE_KEY}: block, so it has no "
                "dev-lane composition to deploy with"
            )
        try:
            lanes[str(name)] = ModelInstanceLane.model_validate(block)
        except ValidationError as exc:
            raise RoutingTableError(
                f"instance {name!r} {LANE_KEY}: block is invalid: {exc}"
            ) from exc
    receipt_lanes = [lane.receipt_lane for lane in lanes.values()]
    if len(set(receipt_lanes)) != len(receipt_lanes):
        raise RoutingTableError(
            f"two instances declare the same receipt_lane: {sorted(receipt_lanes)}"
        )
    return lanes


def load_instance_lanes(repo_root: str | Path) -> dict[str, ModelInstanceLane]:
    """The blocks in ``repo_root``'s committed table."""
    path = Path(repo_root) / ROUTING_TABLE_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingTableError(f"routing table unreadable at {path}: {exc}") from exc
    return parse_instance_lanes(text)


# ---------------------------------------------------------------------------
# A lane's env fragment from the private deployment overlay (OMN-20867)
# ---------------------------------------------------------------------------
#
# Per-lane runtime settings that must differ between lanes (today: whether the
# lane's landing orchestrator and CI-red triage act on GitHub) are deployment
# configuration, so they live in the private deployment overlay repository, not
# here (ledger RULING 2026-10-10T21:57:38Z). The deploy agent's compose env and
# the dev lane's warm refresh both overlay the operator env store with the
# fragment for their own lane, read from the host's overlay clone at
# ``origin/dev`` -- the same clone and ref the lab-overlay apply archives.
#
# This module names the keys a fragment may set and nothing else: no lane id,
# host or value of any lane is written here. A fragment is a JSON document at
# ``<LANE_ENV_FRAGMENT_DIR>/<lane_id><LANE_ENV_FRAGMENT_SUFFIX>``; the lane id is
# the instance's ``verify.receipt_lane`` in the routing table, the name the
# overlay already uses for the lane.
#
# No clone, or no fragment at the ref, changes nothing. A fragment that sets a
# key outside its schema's allowlist, an empty value, or anything malformed is
# refused, naming the problem: a value nobody can see the source of is the
# failure this exists to prevent.

#: The overlay clone's ref the fragment is read at, never its working tree.
LANE_ENV_FRAGMENT_REF: Final = "origin/dev"
#: Where fragments live in the overlay repository.
LANE_ENV_FRAGMENT_DIR: Final = "deploy-overlays/runtime-lane-env/local"
LANE_ENV_FRAGMENT_SUFFIX: Final = ".runtime.env.json"
#: Each schema version and the only keys a fragment of that version may set.
#: ``runtime_lane_env.v1``: omnimarket's landing orchestrator and CI-red triage
#: act switches, read through contract ``${env.VAR:default}`` references whose
#: inline default is the shipped behaviour.
LANE_ENV_FRAGMENT_SCHEMA_KEYS: Final[dict[str, frozenset[str]]] = {
    "runtime_lane_env.v1": frozenset(
        {
            "ONEX_PR_LANDING_GITHUB_MODE",
            "ONEX_PR_LANDING_ARM_ACTION_MODE",
            "ONEX_CI_RED_TRIAGE_ACT_RUNNER",
            "ONEX_CI_RED_TRIAGE_ACT_PR_OWN",
            "ONEX_CI_RED_TRIAGE_ACT_SHARED_CAUSE",
            "ONEX_CI_RED_TRIAGE_ACT_DEV_HEAD",
        }
    ),
}
GIT_READ_TIMEOUT_SECONDS: Final = 30


class LaneEnvFragmentError(RuntimeError):
    """A lane env fragment exists but cannot be applied as written."""


class ModelLaneEnvFragment(BaseModel):
    """One lane's env fragment document."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: str
    lane_id: str
    description: str = ""
    env: dict[str, str]


def instance_runtime_lanes_from_text(text: str) -> dict[str, str]:
    """Each instance's runtime lane id: its ``verify.receipt_lane``."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RoutingTableError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("instances"), dict):
        raise RoutingTableError("routing table declares no instances")
    lanes: dict[str, str] = {}
    for name, spec in raw["instances"].items():
        verify = spec.get("verify") if isinstance(spec, dict) else None
        lane = verify.get("receipt_lane") if isinstance(verify, dict) else None
        if not isinstance(lane, str) or not lane.strip():
            raise RoutingTableError(
                f"instance {name!r} declares no verify.receipt_lane, so it has no "
                "runtime lane id to read its env fragment under"
            )
        lanes[str(name)] = lane.strip()
    if len(set(lanes.values())) != len(lanes):
        raise RoutingTableError(
            f"two instances declare the same verify.receipt_lane: {sorted(lanes.values())}"
        )
    return lanes


def load_instance_runtime_lanes(repo_root: str | Path) -> dict[str, str]:
    """The runtime lane ids in ``repo_root``'s committed table."""
    path = Path(repo_root) / ROUTING_TABLE_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingTableError(f"routing table unreadable at {path}: {exc}") from exc
    return instance_runtime_lanes_from_text(text)


def parse_lane_env_fragment(text: str, *, lane_id: str, source: str) -> dict[str, str]:
    """The env a fragment sets, or :class:`LaneEnvFragmentError` naming why not."""
    try:
        fragment = ModelLaneEnvFragment.model_validate(json.loads(text))
    except json.JSONDecodeError as exc:
        raise LaneEnvFragmentError(f"{source}: not JSON: {exc}") from exc
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<document>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise LaneEnvFragmentError(f"{source}: invalid fragment ({reasons})") from exc
    allowed = LANE_ENV_FRAGMENT_SCHEMA_KEYS.get(fragment.schema_version)
    if allowed is None:
        raise LaneEnvFragmentError(
            f"{source}: unknown schema_version {fragment.schema_version!r} "
            f"(known: {', '.join(sorted(LANE_ENV_FRAGMENT_SCHEMA_KEYS))})"
        )
    if fragment.lane_id != lane_id:
        raise LaneEnvFragmentError(
            f"{source}: lane_id {fragment.lane_id!r} is not the lane it is read "
            f"for, {lane_id!r}"
        )
    refused = sorted(set(fragment.env) - allowed)
    if refused:
        raise LaneEnvFragmentError(
            f"{source}: {fragment.schema_version} may not set {', '.join(refused)}; "
            f"it may set only {', '.join(sorted(allowed))}"
        )
    for key, value in fragment.env.items():
        if not value.strip():
            raise LaneEnvFragmentError(
                f"{source}: {key} is empty; leave a key out to keep its default, "
                "an empty value is refused where it is read"
            )
        if "\n" in value or "\r" in value or "\0" in value:
            raise LaneEnvFragmentError(f"{source}: {key} spans more than one line")
    return dict(fragment.env)


def read_lane_env_fragment(
    clone: Path, lane_id: str, *, ref: str = LANE_ENV_FRAGMENT_REF
) -> dict[str, str]:
    """``lane_id``'s fragment from ``clone`` at ``ref``; empty when there is none.

    Read with ``git`` from the ref, never the working tree: the overlay clone is
    a canonical clone that lanes only fetch, so its working tree can be months
    behind the ref every other reader of it uses.
    """
    if not clone.is_dir():
        return {}

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            timeout=GIT_READ_TIMEOUT_SECONDS,
            check=False,
        )

    if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        raise LaneEnvFragmentError(
            f"{clone} holds no {ref}; the lane env fragment cannot be read, and "
            "an overlay clone that cannot be read is not the same as no fragment"
        )
    relpath = f"{LANE_ENV_FRAGMENT_DIR}/{lane_id}{LANE_ENV_FRAGMENT_SUFFIX}"
    if git("cat-file", "-e", f"{ref}:{relpath}").returncode != 0:
        return {}
    shown = git("show", f"{ref}:{relpath}")
    if shown.returncode != 0:
        raise LaneEnvFragmentError(
            f"git show {ref}:{relpath} in {clone} exited {shown.returncode}: "
            f"{shown.stderr.strip()[:240]}"
        )
    return parse_lane_env_fragment(
        shown.stdout, lane_id=lane_id, source=f"{clone}@{ref}:{relpath}"
    )


def lane_env_shell_exports(env: Mapping[str, str]) -> str:
    """``export KEY=<quoted value>`` lines a shell can ``source``."""
    return "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in sorted(env.items())
    )


def discover_lane_overlay_clone(search_root: Path) -> Path | None:
    """The one clone directly under ``search_root`` that carries lane fragments.

    For a caller that has no agent to name the overlay clone (the warm refresh):
    the clone is found by what it holds at :data:`LANE_ENV_FRAGMENT_REF` -- the
    fragment directory -- never by a name written here. None carries it: no
    clone. More than one: refused, since either could be the lane's.
    """
    if not search_root.is_dir():
        return None
    found = sorted(
        child
        for child in search_root.iterdir()
        if (child / ".git").exists()
        and subprocess.run(
            [
                "git",
                "-C",
                str(child),
                "cat-file",
                "-e",
                f"{LANE_ENV_FRAGMENT_REF}:{LANE_ENV_FRAGMENT_DIR}",
            ],
            capture_output=True,
            text=True,
            timeout=GIT_READ_TIMEOUT_SECONDS,
            check=False,
        ).returncode
        == 0
    )
    if len(found) > 1:
        raise LaneEnvFragmentError(
            f"more than one clone under {search_root} carries "
            f"{LANE_ENV_FRAGMENT_DIR} at {LANE_ENV_FRAGMENT_REF}: "
            f"{', '.join(str(c) for c in found)}"
        )
    return found[0] if found else None


def default_instance_lane_env_exports(repo_root: str | Path, search_root: Path) -> str:
    """Shell exports of the default instance's fragment, for the warm refresh.

    ``scripts/runtime_build/refresh_dev_lane.sh`` refreshes the default
    instance's lane only, so it reads that instance's fragment the way the agent
    does, and a refresh never undoes what the agent applied. The overlay clone
    is the one under ``search_root`` that carries fragments.
    """
    path = Path(repo_root) / ROUTING_TABLE_RELPATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = str(raw.get("default_instance") or "").strip()
    lanes = load_instance_runtime_lanes(repo_root)
    if default not in lanes:
        raise RoutingTableError(f"default instance {default!r} has no runtime lane")
    clone = discover_lane_overlay_clone(search_root)
    if clone is None:
        return ""
    return lane_env_shell_exports(read_lane_env_fragment(clone, lanes[default]))
