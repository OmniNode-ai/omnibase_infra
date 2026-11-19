# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render one revision of the overlays repository into the dev agent's state dir (OMN-24216).

The dev deploy agent renders one revision of the overlays repository into its own state
directory so compose can mount it. A deploy never runs on a revision that is not
rendered; nothing is written until the revision is verified against the clone.

It also writes the dev runtime's product clone pin file (OMN-25901) when a dev runtime job
starts, from the node contract's ``product_clone_pin`` block, because this is the module
the agent already runs at that point.

Loaded by path by the standalone agent (``executor.load_dev_boot_guard``), so it imports
the standard library, pyyaml and omnibase_core only.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
logger = logging.getLogger(__name__)


class OverlayRevisionError(RuntimeError):
    """The overlays revision cannot be rendered as written."""


def _repo_root(execution: ModuleType) -> Path:
    module_file = execution.__file__
    if not module_file:
        raise OverlayRevisionError(
            "the overlay revision source needs the deploy agent source location"
        )
    return Path(module_file).resolve().parents[3]


def declaration(execution: ModuleType) -> Mapping[str, object]:
    """The node contract's ``config.overlay_revision_render`` block.

    ``execution`` is a module of the deploy agent, which locates the agent's checkout,
    as for the llm endpoint env.
    """
    contract = (
        _repo_root(execution)
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    config = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"]
    declared: Mapping[str, object] = config["overlay_revision_render"]
    return declared


def _git(clone: Path, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=scrub_git_location_env(),
    )


def product_clone_pin_declaration(execution: ModuleType) -> Mapping[str, object]:
    """The node contract's ``config.product_clone_pin`` block (OMN-25901)."""
    contract = (
        _repo_root(execution)
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    config = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"]
    declared: Mapping[str, object] = config["product_clone_pin"]
    return declared


def product_clone_pin_path(registry_root: Path, declared: Mapping[str, object]) -> Path:
    """Host path of the pin file: the directory both dev compose files mount."""
    parts = [str(part) for part in cast("Sequence[object]", declared["pin_dir_parts"])]
    return registry_root.joinpath(*parts, str(declared["pin_file_name"]))


def render_product_clone_pins(
    registry_root: Path, declared: Mapping[str, object], *, recorded_at: str = ""
) -> dict[str, object]:
    """Build the pin file omnimarket's ``ModelProductClonePinSet`` reads (OMN-25901).

    One pin per declared clone, its ``pinned_sha`` the clone's own ``git rev-parse
    HEAD``. A clone whose revision cannot be read gets no pin and stays refused
    ``PRODUCT_CLONE_NOT_FRESH``; a revision is never guessed. The verifier measures
    ``HEAD..<pin>``, so a clone the canonical-clone sync fast-forwarded afterwards
    still contains its pin, while a clone reset to an unrelated commit does not.
    """
    stamp = recorded_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    timeout = int(str(declared["read_timeout_seconds"]))
    root = str(declared["container_root"])
    pins: list[dict[str, str]] = []
    for name in cast("Sequence[object]", declared["repositories"]):
        try:
            head = _git(registry_root / str(name), "rev-parse", "HEAD", timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            logger.exception("OMN-25901: no product clone pin for %s", name)
            continue
        sha = head.stdout.strip()
        if head.returncode != 0 or not _REVISION_RE.match(sha):
            logger.error(
                "OMN-25901: no product clone pin for %s: git rev-parse HEAD exited %d",
                name,
                head.returncode,
            )
            continue
        pins.append(
            {
                "repo_root": f"{root}/{name}",
                "pinned_sha": sha,
                "upstream_ref": "HEAD at deploy job start",
                "recorded_at": stamp,
            }
        )
    return {"version": int(str(declared["schema_version"])), "pins": pins}


def write_product_clone_pins(execution: ModuleType) -> Path:
    """Write the dev runtime's pin file from the clones as they are now (OMN-25901).

    The registry root is the parent of ``execution.REPO_DIR``, the clone the compose
    files live in, which is what ``../..`` resolves to from ``docker/``. The file is
    written beside its target and renamed into place, readable by the runtime user.
    """
    declared = product_clone_pin_declaration(execution)
    registry_root = Path(str(vars(execution)["REPO_DIR"])).parent
    path = product_clone_pin_path(registry_root, declared)
    payload = render_product_clone_pins(registry_root, declared)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    staging.chmod(0o644)
    staging.replace(path)
    logger.info("OMN-25901: wrote the product clone pin file %s", path)
    return path


def apply(
    execution: ModuleType,
    state_dir: Path,
    environ: MutableMapping[str, str],
    revision: str,
) -> None:
    """Render ``revision`` into the state dir, or clear the store root when empty.

    ``execution`` is a module of the deploy agent, which locates the node contract.

    An empty revision clears the store root only when it points at a folder this
    module rendered under the state dir; a value set any other way is left alone.
    A non-empty revision must name a commit the clone holds; nothing is written
    until the revision is verified and rendered.
    """
    try:
        if "REPO_DIR" in vars(execution):
            write_product_clone_pins(execution)
        else:
            logger.warning(
                "OMN-25901: the execution module names no REPO_DIR; no product clone pin written"
            )
    except OSError:
        # The clones stay refused PRODUCT_CLONE_NOT_FRESH; the deploy is not stopped.
        logger.exception("OMN-25901: the product clone pin file was not written")

    declared = declaration(execution)
    store_root_env = str(declared["store_root_env"])
    rendered_dir_name = str(declared["rendered_dir_name"])
    overlay_dir = str(declared["overlay_dir"])
    clone_env = str(declared["clone_env"])
    timeout = int(str(declared["read_timeout_seconds"]))

    if revision == "":
        value = environ.get(store_root_env, "")
        rendered_root = state_dir / rendered_dir_name
        if value and Path(value).is_relative_to(rendered_root):
            del environ[store_root_env]
        return

    if not _REVISION_RE.match(revision):
        raise OverlayRevisionError(
            f"revision {revision!r} is not a 40-character lowercase hex SHA"
        )

    clone_raw = environ.get(clone_env, "").strip()
    if not clone_raw:
        raise OverlayRevisionError(f"{clone_env} must name the overlays clone")
    clone = Path(clone_raw)
    if not clone.is_dir():
        raise OverlayRevisionError(
            f"{clone_env} names {clone}, which is not a directory"
        )

    check = _git(clone, "cat-file", "-e", f"{revision}^{{commit}}", timeout=timeout)
    if check.returncode != 0:
        raise OverlayRevisionError(
            f"{clone} holds no {revision}; refresh the overlays clone so it holds "
            "the revision"
        )

    final_dir = state_dir / rendered_dir_name / revision
    if not final_dir.is_dir():
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / rendered_dir_name).mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{rendered_dir_name}-{revision[:8]}.", dir=state_dir
            )
        )
        try:
            archive = subprocess.run(
                [
                    "git",
                    "-C",
                    str(clone),
                    "archive",
                    "--format=tar",
                    revision,
                    overlay_dir,
                ],
                capture_output=True,
                check=False,
                timeout=timeout,
                env=scrub_git_location_env(),
            )
            if archive.returncode != 0:
                raise OverlayRevisionError(
                    f"git archive {revision} {overlay_dir} in {clone} exited "
                    f"{archive.returncode}: {archive.stderr.decode('utf-8', 'replace').strip()[:240]}"
                )
            with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tf:
                tf.extractall(path=tmp_dir, filter="data")
            tmp_dir.chmod(0o755)  # the runtime container reads it as another user
            tmp_dir.replace(final_dir)
        except BaseException:
            if tmp_dir.is_dir():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    environ[store_root_env] = str(final_dir / overlay_dir)
