# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-2 (OMN-14438): the cut-lab-ref one-command lab deploy wrapper.

Covers the wrapper's own logic without a real Docker deploy: the dry-run plan
(DEPLOY_REF / BUILD_SOURCE / compose project the RT-1 mechanism needs), the
--hotpatch plan, the prod-lane refusal, and the --execute path cutting a
lab/<lane>/<utc>-<shortsha> tag before delegating to a stub deploy-runtime.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUT_LAB_REF = REPO_ROOT / "scripts" / "runtime_build" / "cut-lab-ref.sh"
STAGE_PINNED_PROOF_ROOT = (
    REPO_ROOT / "scripts" / "runtime_build" / "stage_pinned_proof_root.py"
)

MANIFEST = REPO_ROOT / "scripts" / "runtime_build" / "sibling_clone_manifest.sh"


def _manifest_array(name: str) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; printf "%s\\n" "${' + name + '[@]}"',
            "_",
            str(MANIFEST),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return tuple(result.stdout.split())


# OMN-19072: the repos a lab tag must cover are every clone the pin preflight
# reads (the manifest) plus the declared tag-only extras. Resolved from the
# manifest rather than spelled here, so this test cannot keep a list of its own.
LAB_REF_REPOS = (
    *_manifest_array("SIBLING_CLONE_MANIFEST"),
    *_manifest_array("SIBLING_EXTRA_TRACKED_REPOS"),
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ) | {"GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


def _make_omni_home(tmp_path: Path) -> Path:
    omni_home = tmp_path / "omni_home"
    for repo in LAB_REF_REPOS:
        path = omni_home / repo
        path.mkdir(parents=True)
        _git(path, "init", "-q", "-b", "dev")
        _git(path, "config", "user.email", "t@t.t")
        _git(path, "config", "user.name", "t")
        (path / "f.txt").write_text("x\n", encoding="utf-8")
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", "init")
    return omni_home


def _run(
    omni_home: Path, *args: str, deploy_runtime: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "OMNI_HOME": str(omni_home)}
    if deploy_runtime is not None:
        env["DEPLOY_RUNTIME"] = str(deploy_runtime)
    return subprocess.run(
        ["bash", str(CUT_LAB_REF), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_dry_run_plan_dev_lane(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    result = _run(omni_home, "--ref", "dev", "--lane", "dev")
    assert result.returncode == 0, result.stderr
    plan = result.stderr
    assert "DEPLOY_REF=dev" in plan
    assert "BUILD_SOURCE=workspace" in plan
    assert "OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra" in plan
    assert "DEPLOY_HOTPATCH=0" in plan
    # OMN-14562: the lab fast lane always forces a same-version overwrite --
    # deploy-runtime.sh's version-directory collision guard otherwise fires
    # before the RT-1 clean-ref checkout ever runs.
    assert "--execute --force" in plan
    assert "dry-run" in plan  # no build/deploy performed


@pytest.mark.unit
def test_dry_run_plan_stability_lane(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    result = _run(omni_home, "--lane", "stability-test")
    assert result.returncode == 0, result.stderr
    assert (
        "OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra-stability-test" in result.stderr
    )


@pytest.mark.unit
def test_dry_run_plan_dogfood_lane_is_isolated(tmp_path: Path) -> None:
    """Dogfood resolves to its own compose project, never the shared dev lane.

    A dogfood build only runs from a proof root staged from a pinned snapshot
    (OMN-19086), so the plan is taken against one.
    """
    omni_home = _make_omni_home(tmp_path)
    spi = omni_home / "omnibase_spi"
    spi.mkdir()
    _git(spi, "init", "-q", "-b", "dev")
    _git(spi, "config", "user.email", "t@t.t")
    _git(spi, "config", "user.name", "t")
    (spi / "f.txt").write_text("x\n", encoding="utf-8")
    _git(spi, "add", "-A")
    _git(spi, "commit", "-q", "-m", "init")
    proof_root = tmp_path / "proof-root"
    staged = subprocess.run(
        [
            sys.executable,
            str(STAGE_PINNED_PROOF_ROOT),
            "stage",
            "--dest",
            str(proof_root),
            "--source-root",
            str(omni_home),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )
    assert staged.returncode == 0, staged.stderr
    result = _run(proof_root, "--lane", "dogfood", "--hotpatch")
    assert result.returncode == 0, result.stderr
    assert "OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra-dogfood" in result.stderr
    assert "--profile dogfood" in result.stderr


@pytest.mark.unit
def test_hotpatch_plan_labels_and_omits_ref(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    result = _run(omni_home, "--hotpatch")
    assert result.returncode == 0, result.stderr
    plan = result.stderr
    assert "DEPLOY_HOTPATCH=1" in plan
    # In hotpatch mode the dirty tree is deployed AS-IS; DEPLOY_REF is not set.
    assert "DEPLOY_REF=" not in plan
    assert "hotpatch" in plan.lower()


@pytest.mark.unit
def test_prod_lane_is_refused(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    result = _run(omni_home, "--lane", "prod")
    assert result.returncode == 2
    assert "Train 2" in result.stderr


@pytest.mark.unit
def test_missing_omni_home_fails(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(CUT_LAB_REF), "--ref", "dev"],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "OMNI_HOME"},
    )
    assert result.returncode == 1
    assert "OMNI_HOME must be set" in result.stderr


@pytest.mark.unit
def test_execute_cuts_lab_tag_and_delegates(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    marker = tmp_path / "deploy_ran.marker"
    stub = tmp_path / "stub-deploy-runtime.sh"
    stub.write_text(
        f'#!/usr/bin/env bash\necho "stub deploy: $*" >&2\ntouch {marker}\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    result = _run(
        omni_home,
        "--ref",
        "dev",
        "--lane",
        "dev",
        "--cut-tag",
        "--execute",
        deploy_runtime=stub,
    )
    assert result.returncode == 0, result.stderr
    # The stub deploy-runtime was actually invoked.
    assert marker.exists()
    # OMN-14562: --force must reach deploy-runtime.sh so a same-version lab
    # redeploy overwrites rather than tripping the version-directory guard.
    assert "--force" in result.stderr

    # OMN-19072 AC-2: the tag set is the manifest's clones plus the change-control
    # repo. Before the fix cut-lab-ref.sh kept its own list and left
    # omnibase_spi, a repo the pin preflight reads, untagged.
    assert "omnibase_spi" in LAB_REF_REPOS
    assert "onex_change_control" in LAB_REF_REPOS
    assert len(LAB_REF_REPOS) == 6
    # A lab tag lab/dev/<utc>-<shortsha> was cut in every sibling clone at its dev SHA.
    tag_re = re.compile(r"^lab/dev/\d{8}T\d{6}Z-[0-9a-f]{12}$")
    for repo in LAB_REF_REPOS:
        clone = omni_home / repo
        tags = _git(clone, "tag", "--list", "lab/dev/*").splitlines()
        assert len(tags) == 1, f"{repo}: expected one lab tag, got {tags}"
        assert tag_re.match(tags[0]), f"{repo}: bad lab tag {tags[0]!r}"
        # The tag points at the repo's dev HEAD.
        assert _git(clone, "rev-parse", f"{tags[0]}^{{commit}}") == _git(
            clone, "rev-parse", "dev"
        )


@pytest.mark.unit
def test_ref_missing_in_a_sibling_falls_back_instead_of_aborting(
    tmp_path: Path,
) -> None:
    """OMN-19072 review: a --ref that resolves in omnibase_infra but not in a
    sibling -- a lab tag cut before omnibase_spi joined the tag set, or a raw
    infra sha -- must not abort under set -e with the tag half-cut and the
    deploy never run. The sibling is tagged at its own fallback ref, the same
    fallback stage_workspace.sh checks siblings out at, and says so."""
    omni_home = _make_omni_home(tmp_path)
    old_tag = "lab/dev/20260901T000000Z-000000000000"
    for repo in LAB_REF_REPOS:
        if repo != "omnibase_spi":
            _git(omni_home / repo, "tag", old_tag, "dev")
    stub = tmp_path / "stub-deploy-runtime.sh"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(CUT_LAB_REF),
            "--ref",
            old_tag,
            "--lane",
            "dev",
            "--cut-tag",
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **scrub_git_location_env(os.environ),
            "OMNI_HOME": str(omni_home),
            "DEPLOY_RUNTIME": str(stub),
            "DEPLOY_SIBLING_FALLBACK_REF": "dev",
        },
    )
    assert result.returncode == 0, result.stderr
    spi = omni_home / "omnibase_spi"
    new_tags = [
        t for t in _git(spi, "tag", "--list", "lab/dev/*").splitlines() if t != old_tag
    ]
    assert len(new_tags) == 1, new_tags
    assert _git(spi, "rev-parse", f"{new_tags[0]}^{{commit}}") == _git(
        spi, "rev-parse", "dev"
    )
    assert "omnibase_spi" in result.stderr and "falling back" in result.stderr


@pytest.mark.unit
def test_ref_missing_in_the_infra_clone_still_aborts(tmp_path: Path) -> None:
    """The fallback is for siblings only: a --ref the build-context repo cannot
    resolve names nothing to build, and must still refuse."""
    omni_home = _make_omni_home(tmp_path)
    stub = tmp_path / "stub-deploy-runtime.sh"
    marker = tmp_path / "deploy_ran.marker"
    stub.write_text(f"#!/usr/bin/env bash\ntouch {marker}\n", encoding="utf-8")
    stub.chmod(0o755)
    result = _run(
        omni_home,
        "--ref",
        "no-such-ref-anywhere",
        "--cut-tag",
        "--execute",
        deploy_runtime=stub,
    )
    assert result.returncode != 0
    assert not marker.exists()
