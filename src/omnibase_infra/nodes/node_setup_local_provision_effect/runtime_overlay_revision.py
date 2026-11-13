# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render one revision of the overlays repository into the dev agent's state dir (OMN-24216).

The dev deploy agent renders one revision of the overlays repository into its own state
directory so compose can mount it. The revision is recorded in the state directory and
rendered on demand; a deploy never runs on a revision that is not rendered.

Loaded by path by the standalone agent (``executor.load_dev_boot_guard``), so it imports
the standard library, pyyaml and omnibase_core only.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class OverlayRevisionError(RuntimeError):
    """The overlays revision cannot be recorded or rendered as written."""


def _repo_root(execution: ModuleType) -> Path:
    module_file = execution.__file__
    if not module_file:
        raise OverlayRevisionError(
            "the overlay revision source needs the deploy agent source location"
        )
    return Path(module_file).resolve().parents[3]


def declaration(execution: ModuleType) -> Mapping[str, object]:
    """The node contract's ``config.overlay_revision_render`` block.

    ``execution`` is the deploy agent's executor module, which locates the agent's
    checkout, as for the estimator build's settings.
    """
    contract = (
        _repo_root(execution)
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    config = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"]
    declared: Mapping[str, object] = config["overlay_revision_render"]
    return declared


def _git(
    clone: Path, *args: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=scrub_git_location_env(),
    )


def record(
    execution: ModuleType, environ: Mapping[str, str], revision: str
) -> None:
    """Record ``revision`` in the state directory, or clear it when empty.

    The revision must name a commit the clone holds; nothing is written until the
    revision is verified and rendered.
    """
    declared = declaration(execution)
    state_dir_env = str(declared["state_dir_env"])
    state_dir_raw = environ.get(state_dir_env, "").strip()
    if not state_dir_raw:
        if revision == "":
            return  # no state directory: nothing recorded, nothing to clear
        raise OverlayRevisionError(f"{state_dir_env} must name the state directory")
    state_dir = Path(state_dir_raw)
    active_file = state_dir / str(declared["active_file_name"])

    if revision == "":
        if active_file.is_file():
            active_file.unlink()
        return

    if not _REVISION_RE.match(revision):
        raise OverlayRevisionError(
            f"revision {revision!r} is not a 40-character lowercase hex SHA"
        )

    clone_env = str(declared["clone_env"])
    clone_raw = environ.get(clone_env, "").strip()
    if not clone_raw:
        raise OverlayRevisionError(f"{clone_env} must name the overlays clone")
    clone = Path(clone_raw)
    if not clone.is_dir():
        raise OverlayRevisionError(
            f"{clone_env} names {clone}, which is not a directory"
        )

    timeout = int(str(declared["read_timeout_seconds"]))
    overlay_dir = str(declared["overlay_dir"])

    # Verify the revision exists in the clone before writing anything.
    check = _git(clone, "cat-file", "-e", f"{revision}^{{commit}}", timeout=timeout)
    if check.returncode != 0:
        raise OverlayRevisionError(
            f"{clone} holds no {revision}; refresh the overlays clone so it holds "
            "the revision"
        )

    # Render: extract the overlay directory at the revision into the state dir.
    rendered_dir_name = str(declared["rendered_dir_name"])
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
            tmp_dir.replace(final_dir)
        except BaseException:
            if tmp_dir.is_dir():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    # Write the active revision atomically.
    tmp_active = state_dir / f".{active_file.name}.tmp"
    tmp_active.write_text(revision, encoding="utf-8")
    tmp_active.replace(active_file)


def render_dev_lane(
    execution: ModuleType, environ: Mapping[str, str]
) -> dict[str, str]:
    """The dev lane's overlay store root, or nothing when no revision is active.

    Other lanes get nothing. An active revision whose rendered folder is missing
    refuses: a deploy never runs on a revision that is not rendered.
    """
    raw = environ.get(execution.ENV_ALLOWED_LANES, "").strip()
    if not (
        raw
        and execution.is_lab_dev_fence(execution.parse_allowed_lanes(raw))
        and execution._ActiveDeployLane.lane in (None, execution.EnumRuntimeLane.DEV)
    ):
        return {}

    declared = declaration(execution)
    state_dir_env = str(declared["state_dir_env"])
    state_dir_raw = environ.get(state_dir_env, "").strip()
    if not state_dir_raw:
        return {}
    state_dir = Path(state_dir_raw)
    active_file = state_dir / str(declared["active_file_name"])
    if not active_file.is_file():
        return {}
    revision = active_file.read_text(encoding="utf-8").strip()
    if not revision:
        return {}

    rendered_dir_name = str(declared["rendered_dir_name"])
    overlay_dir = str(declared["overlay_dir"])
    folder = state_dir / rendered_dir_name / revision / overlay_dir
    if not folder.is_dir():
        raise RuntimeError(
            f"{state_dir / rendered_dir_name / revision} is not rendered; "
            "a deploy never runs on a revision that is not rendered"
        )
    store_root_env = str(declared["store_root_env"])
    return {store_root_env: str(folder)}
