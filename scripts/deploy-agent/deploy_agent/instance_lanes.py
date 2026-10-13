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

import io
import json
import shlex
import subprocess
import tarfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, NoReturn

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from deploy_agent.events import (
    EnumLaneMirrorOutcome,
    ModelLaneMirrorForwarder,
    ModelLaneMirrorReceipt,
)
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
#: inline default is the shipped behaviour; and the lane class the runtime is an
#: instance of with the class contracts it takes, read by the auto-wiring
#: ownership filter.
LANE_ENV_FRAGMENT_SCHEMA_KEYS: Final[dict[str, frozenset[str]]] = {
    "runtime_lane_env.v1": frozenset(
        {
            "ONEX_PR_LANDING_GITHUB_MODE",
            "ONEX_PR_LANDING_ARM_ACTION_MODE",
            "ONEX_CI_RED_TRIAGE_ACT_RUNNER",
            "ONEX_CI_RED_TRIAGE_ACT_PR_OWN",
            "ONEX_CI_RED_TRIAGE_ACT_SHARED_CAUSE",
            "ONEX_CI_RED_TRIAGE_ACT_DEV_HEAD",
            "ONEX_RUNTIME_LANE_CLASS",
            "ONEX_RUNTIME_LANE_CLASS_CONTRACTS",
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


# ---------------------------------------------------------------------------
# The lane-mirror forwarders from the private deployment overlay (OMN-20867)
# ---------------------------------------------------------------------------
#
# The runtime's bus takes one bootstrap address for its life, so topics that
# live on another broker reach a lane through a mirror: ``onex-gateway-forwarder``
# (node_bus_forwarder_effect, mirror-only) consumes a source lane and republishes
# a named topic set onto the lane's broker. The overlay that says which mirrors a
# host runs -- the compose file, one ``forwarder.yaml`` per mirror and one
# ``*.vars`` per host -- is deployment configuration and lives in the private
# overlay repository under :data:`LANE_MIRROR_DIR`, never here.
#
# The deploy agent is the one process allowed to join a declared lane's
# network, so it is the one that brings the mirrors up: after the runtime is
# verified, on the lane's runtime image, every deploy. Which host vars applies is
# not named anywhere here either: it is the one vars file whose
# :data:`LANE_MIRROR_NETWORK_KEY` is a network the lane's runtime container is
# attached to, read live. That is also the mirror's safety property (it joins
# this lane's network and no other), so the selection and the property are one
# fact.
#
# No overlay clone, no overlay directory at the ref, no vars file naming this
# lane, or a vars file enabling no profile: a typed no-op. An enabled profile
# that does not come up healthy on the runtime image: ``FAILED``, per forwarder.

LANE_MIRROR_DIR: Final = "deploy-overlays/lane-mirror"
LANE_MIRROR_COMPOSE_FILE: Final = "docker-compose.lane-mirror.yml"
LANE_MIRROR_VARS_SUFFIX: Final = ".vars"
LANE_MIRROR_NETWORK_KEY: Final = "LANE_MIRROR_LANE_NETWORK"
LANE_MIRROR_PROFILES_KEY: Final = "COMPOSE_PROFILES"
LANE_MIRROR_PROJECT_KEY: Final = "COMPOSE_PROJECT_NAME"
LANE_MIRROR_IMAGE_KEY: Final = "LANE_MIRROR_IMAGE"
#: Operator-held: the absolute path of the credential map the forwarders mount.
#: Never defaulted (a guessed path would mount some other file).
LANE_MIRROR_CREDENTIAL_MAP_KEY: Final = "LANE_MIRROR_CREDENTIAL_MAP_FILE"
LANE_MIRROR_UP_TIMEOUT_SECONDS: Final = 300
#: The overlay's healthcheck is start_period 30s, interval 15s, retries 4: a
#: forwarder is healthy or unhealthy within about 90 s.
LANE_MIRROR_READY_TIMEOUT_SECONDS: Final = 150.0
LANE_MIRROR_POLL_SECONDS: Final = 5.0
DOCKER_READ_TIMEOUT_SECONDS: Final = 30

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ModelLaneMirrorRequest(BaseModel):
    """Everything the lane-mirror step needs, resolved by the caller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The host's deployment overlay clone (the one the env fragment is read from).
    clone: Path
    #: The lane's main runtime container: its image and its networks.
    runtime_container: str
    #: Where overlay snapshots are materialised, one directory per overlay commit.
    materialize_root: Path
    #: The compose env of the deploy, which the forwarders' interpolation reads.
    compose_env: dict[str, str]
    ref: str = LANE_ENV_FRAGMENT_REF
    ready_timeout_seconds: float = LANE_MIRROR_READY_TIMEOUT_SECONDS
    poll_seconds: float = LANE_MIRROR_POLL_SECONDS


class _LaneMirrorStopError(Exception):
    """Ends the step early with a receipt."""

    def __init__(self, receipt: ModelLaneMirrorReceipt) -> None:
        super().__init__(receipt.detail)
        self.receipt = receipt


def _default_run(
    argv: list[str], timeout: float, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )


def parse_lane_mirror_vars(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines of a host vars file; comments and blanks skipped."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def lane_mirror_profile_services(compose_text: str) -> dict[str, list[tuple[str, str]]]:
    """profile -> ``(service, container_name)`` of every service it enables."""
    raw = yaml.safe_load(compose_text)
    services = raw.get("services") if isinstance(raw, dict) else None
    if not isinstance(services, dict):
        raise ValueError(f"{LANE_MIRROR_COMPOSE_FILE} declares no services")
    by_profile: dict[str, list[tuple[str, str]]] = {}
    for name, spec in services.items():
        if not isinstance(spec, dict):
            continue
        container = str(spec.get("container_name") or name)
        for profile in spec.get("profiles") or ():
            by_profile.setdefault(str(profile), []).append((str(name), container))
    return by_profile


def _git_text(run: Runner, clone: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(clone), *args], timeout=GIT_READ_TIMEOUT_SECONDS)


def _materialize_overlay(clone: Path, sha: str, root: Path) -> Path:
    """The overlay directory at ``sha`` on disk, written once per commit.

    A directory per commit, never rewritten: a running forwarder bind-mounts
    files from it, so an overlay change lands in a new directory and compose
    recreates the forwarder onto it.
    """
    dest = root / sha
    if dest.is_dir():
        return dest
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "archive",
            "--format=tar",
            f"{sha}:{LANE_MIRROR_DIR}",
        ],
        capture_output=True,
        check=False,
        timeout=GIT_READ_TIMEOUT_SECONDS,
    )
    if archive.returncode != 0:
        raise OSError(
            f"git archive {sha}:{LANE_MIRROR_DIR} exited {archive.returncode}: "
            f"{archive.stderr.decode(errors='replace').strip()[:240]}"
        )
    staging = root / f".{sha}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(staging, filter="data")
    staging.rename(dest)
    return dest


def _forwarder_state(
    run: Runner, profile: str, service: str, container: str, runtime_image: str
) -> ModelLaneMirrorForwarder:
    result = run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}"
            "{{else}}none{{end}}|{{.Image}}",
            container,
        ],
        timeout=DOCKER_READ_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return ModelLaneMirrorForwarder(
            profile=profile,
            service=service,
            container=container,
            state="absent",
            health="absent",
            ready=False,
        )
    state, _, rest = result.stdout.strip().partition("|")
    health, _, image_id = rest.partition("|")
    return ModelLaneMirrorForwarder(
        profile=profile,
        service=service,
        container=container,
        state=state,
        health=health,
        image_id=image_id,
        ready=state == "running" and health == "healthy" and image_id == runtime_image,
    )


def _settled_unready(forwarder: ModelLaneMirrorForwarder) -> bool:
    """A forwarder that will not become ready by waiting longer."""
    return forwarder.state in ("absent", "exited", "dead") or forwarder.health in (
        "unhealthy",
        "none",
    )


def handle_lane_mirror(
    request: ModelLaneMirrorRequest,
    *,
    run: Runner = _default_run,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ModelLaneMirrorReceipt:
    """Bring this lane's enabled mirror forwarders up on its runtime image.

    Never raises: the receipt is the channel. A raise would cost the deploy's
    terminal publish, and the compose lane's verdict is already settled.
    """
    try:
        return _handle_lane_mirror(request, run=run, sleep=sleep, clock=clock)
    except _LaneMirrorStopError as stop:
        return stop.receipt
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        tarfile.TarError,
        yaml.YAMLError,
    ) as exc:
        # Every failure the step's reads, parses and subprocesses can raise.
        # pydantic's ValidationError and json's decode error are ValueErrors.
        return ModelLaneMirrorReceipt(
            outcome=EnumLaneMirrorOutcome.FAILED,
            detail=f"lane-mirror step raised {type(exc).__name__}: {exc}",
        )


def _handle_lane_mirror(
    request: ModelLaneMirrorRequest,
    *,
    run: Runner,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> ModelLaneMirrorReceipt:
    clone, ref = request.clone, request.ref

    def stop(outcome: EnumLaneMirrorOutcome, detail: str, **facts: object) -> NoReturn:
        raise _LaneMirrorStopError(
            ModelLaneMirrorReceipt.model_validate(
                {"outcome": outcome, "detail": detail, **facts}
            )
        )

    if not clone.is_dir():
        stop(
            EnumLaneMirrorOutcome.NO_OVERLAY, f"no deployment overlay clone at {clone}"
        )
    resolved = _git_text(
        run, clone, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"
    )
    if resolved.returncode != 0:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"{clone} holds no {ref}; an overlay clone that cannot be read is not "
            "the same as no overlay",
        )
    sha = resolved.stdout.strip()
    if (
        _git_text(run, clone, "cat-file", "-e", f"{sha}:{LANE_MIRROR_DIR}").returncode
        != 0
    ):
        stop(
            EnumLaneMirrorOutcome.NO_OVERLAY,
            f"{clone}@{ref} carries no {LANE_MIRROR_DIR}",
            overlay_sha=sha,
        )

    listing = _git_text(
        run, clone, "ls-tree", "--name-only", f"{sha}:{LANE_MIRROR_DIR}"
    )
    if listing.returncode != 0:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"git ls-tree {LANE_MIRROR_DIR} exited {listing.returncode}",
            overlay_sha=sha,
        )
    names = listing.stdout.split()
    if LANE_MIRROR_COMPOSE_FILE not in names:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"{LANE_MIRROR_DIR} at {sha[:12]} has no {LANE_MIRROR_COMPOSE_FILE}",
            overlay_sha=sha,
        )

    networks_read = run(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .NetworkSettings.Networks}}",
            request.runtime_container,
        ],
        timeout=DOCKER_READ_TIMEOUT_SECONDS,
    )
    if networks_read.returncode != 0:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"the lane's runtime container {request.runtime_container} could not "
            f"be inspected: {networks_read.stderr.strip()[:240]}",
            overlay_sha=sha,
        )
    networks = set(json.loads(networks_read.stdout or "{}") or {})

    matches: list[tuple[str, dict[str, str]]] = []
    for name in sorted(n for n in names if n.endswith(LANE_MIRROR_VARS_SUFFIX)):
        shown = _git_text(run, clone, "show", f"{sha}:{LANE_MIRROR_DIR}/{name}")
        if shown.returncode != 0:
            stop(
                EnumLaneMirrorOutcome.FAILED,
                f"git show {LANE_MIRROR_DIR}/{name} exited {shown.returncode}",
                overlay_sha=sha,
            )
        values = parse_lane_mirror_vars(shown.stdout)
        if values.get(LANE_MIRROR_NETWORK_KEY) in networks:
            matches.append((name, values))
    if not matches:
        stop(
            EnumLaneMirrorOutcome.NOT_THIS_LANE,
            f"no {LANE_MIRROR_VARS_SUFFIX} file in {LANE_MIRROR_DIR} names a "
            f"network of {request.runtime_container} ({', '.join(sorted(networks))})",
            overlay_sha=sha,
        )
    if len(matches) > 1:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            "more than one vars file names this lane's network: "
            + ", ".join(name for name, _ in matches),
            overlay_sha=sha,
        )
    vars_file, values = matches[0]
    lane_network = values[LANE_MIRROR_NETWORK_KEY]
    profiles = tuple(
        p.strip()
        for p in values.get(LANE_MIRROR_PROFILES_KEY, "").split(",")
        if p.strip()
    )
    facts: dict[str, object] = {
        "overlay_sha": sha,
        "vars_file": vars_file,
        "lane_network": lane_network,
        "profiles": profiles,
    }
    if not profiles:
        stop(
            EnumLaneMirrorOutcome.NO_PROFILES,
            f"{vars_file} enables no profile",
            **facts,
        )
    project = values.get(LANE_MIRROR_PROJECT_KEY, "").strip()
    if not project:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"{vars_file} sets no {LANE_MIRROR_PROJECT_KEY}",
            **facts,
        )

    compose_text = _git_text(
        run, clone, "show", f"{sha}:{LANE_MIRROR_DIR}/{LANE_MIRROR_COMPOSE_FILE}"
    )
    by_profile = lane_mirror_profile_services(compose_text.stdout)
    enabled: list[tuple[str, str, str]] = []
    for profile in profiles:
        services = by_profile.get(profile)
        if not services:
            stop(
                EnumLaneMirrorOutcome.FAILED,
                f"{vars_file} enables profile {profile!r}, which no service in "
                f"{LANE_MIRROR_COMPOSE_FILE} carries",
                **facts,
            )
        enabled.extend((profile, service, container) for service, container in services)

    image = run(
        ["docker", "inspect", "--format", "{{.Image}}", request.runtime_container],
        timeout=DOCKER_READ_TIMEOUT_SECONDS,
    )
    runtime_image = image.stdout.strip() if image.returncode == 0 else ""
    if not runtime_image:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"the image of {request.runtime_container} could not be read",
            **facts,
        )
    facts["runtime_image"] = runtime_image

    credential_map = request.compose_env.get(LANE_MIRROR_CREDENTIAL_MAP_KEY, "").strip()
    if not credential_map:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"{vars_file} enables {', '.join(profiles)} but "
            f"{LANE_MIRROR_CREDENTIAL_MAP_KEY} is not set in the operator env store",
            **facts,
        )
    if not Path(credential_map).is_file():
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"{LANE_MIRROR_CREDENTIAL_MAP_KEY} names {credential_map}, which is not a file",
            **facts,
        )

    overlay_dir = _materialize_overlay(clone, sha, request.materialize_root)
    env = dict(request.compose_env)
    env.update(values)
    env[LANE_MIRROR_IMAGE_KEY] = runtime_image
    argv = ["docker", "compose", "-p", project]
    argv += ["-f", str(overlay_dir / LANE_MIRROR_COMPOSE_FILE)]
    for profile in profiles:
        argv += ["--profile", profile]
    argv += ["up", "-d", *sorted({service for _, service, _ in enabled})]
    facts["compose_argv"] = tuple(argv)
    up = run(argv, timeout=LANE_MIRROR_UP_TIMEOUT_SECONDS, env=env)

    deadline = clock() + request.ready_timeout_seconds
    while True:
        forwarders = tuple(
            _forwarder_state(run, profile, service, container, runtime_image)
            for profile, service, container in enabled
        )
        if all(f.ready for f in forwarders):
            break
        if up.returncode != 0 or clock() >= deadline:
            break
        if any(_settled_unready(f) for f in forwarders if not f.ready):
            break
        sleep(request.poll_seconds)
    facts["forwarders"] = forwarders

    if up.returncode != 0:
        tail = (up.stderr or up.stdout).strip()[-400:]
        stop(
            EnumLaneMirrorOutcome.FAILED,
            f"docker compose up exited {up.returncode}: {tail}",
            **facts,
        )
    unready = [f for f in forwarders if not f.ready]
    if unready:
        stop(
            EnumLaneMirrorOutcome.FAILED,
            "not ready: "
            + "; ".join(
                f"{f.container} state={f.state} health={f.health}"
                + (
                    ""
                    if f.image_id in ("", runtime_image)
                    else " image=not-the-runtime-image"
                )
                for f in unready
            ),
            **facts,
        )
    return ModelLaneMirrorReceipt.model_validate(
        {
            "outcome": EnumLaneMirrorOutcome.READY,
            "detail": "ready: " + ", ".join(f.container for f in forwarders),
            **facts,
        }
    )
