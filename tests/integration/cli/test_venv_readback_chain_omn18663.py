# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end: the repair chain proves itself against a REAL venv (OMN-18663).

The unit tests around this change stand a shim in for the target interpreter.
That is the right shape for the classification logic, and it is exactly the
wrong shape for the question this defect turned on: whether the readback agrees
with what ``importlib.metadata`` reports out of a real site-packages after a
repair claims to have written it.

So this exercises the whole chain with the real pieces --
``check-omnimarket-venv-drift.sh`` -> the lock -> the co-install step ->
``venv_readback.py`` -> a real interpreter reading a real ``direct_url.json`` --
and varies only whether the install actually lands.

Hermetic and offline. The canonical remote is a local bare git repo, and the
"installed omnimarket" is a real ``.dist-info`` written into a real venv's
site-packages rather than a wheel pulled from github. A real ``uv pip install``
from the network is what the co-install step does in production and is
deliberately NOT what is under test here: the defect was never in uv, it was in
believing uv's exit status.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRIFT_SCRIPT = _REPO_ROOT / "scripts" / "check-omnimarket-venv-drift.sh"

_EXIT_DRIFT = 1


def _scrubbed_git_env() -> dict[str, str]:
    """A git environment that cannot reach out of ``tmp_path`` (OMN-14891).

    git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into every hook
    environment, and those OVERRIDE both ``cwd=`` and ``git -C``: a fixture that
    shells out to git while running under a pre-commit or pre-push hook would
    mutate the REAL invoking worktree. The keys are named literally as well as
    scrubbed, because the guard verifies a module-local scrubber by reading the
    keys it drops and a delegated call is invisible to that check.
    """
    env = scrub_git_location_env(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    return env


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_scrubbed_git_env(),
    ).stdout.strip()


def _make_omni_home(root: Path, *, version: str) -> tuple[Path, str]:
    """An $OMNI_HOME whose omnimarket clone tracks a local bare 'origin'."""
    work = root / "upstream"
    work.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", "-b", "dev"],
        cwd=work,
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    (work / "pyproject.toml").write_text(
        f'[project]\nname = "omnimarket"\nversion = "{version}"\n'
        'dependencies = ["omnibase-compat>=0.5.7,<0.6.0"]\n',
        encoding="utf-8",
    )
    _git(work, "add", "pyproject.toml")
    _git(work, "commit", "--quiet", "-m", "init")

    bare = root / "bare.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(work), str(bare)],
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )

    omni_home = root / "omni_home"
    omni_home.mkdir()
    subprocess.run(
        ["git", "clone", "--quiet", str(bare), str(omni_home / "omnimarket")],
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )
    return omni_home, _git(omni_home / "omnimarket", "rev-parse", "HEAD")


def _seed_packaging(site_packages: Path) -> None:
    """Give the synthetic venv the ``packaging`` that every real one carries.

    ``venv --without-pip`` leaves site-packages EMPTY, and
    ``scripts/venv_readback.py`` runs INSIDE the target interpreter and imports
    ``packaging`` to evaluate the packaged floors each installed distribution
    declares (OMN-18752). With it absent the floors row reads UNREADABLE, the
    whole readback is INDETERMINATE, and the chain refuses a repair that did in
    fact land.

    That refusal is the readback being CORRECT about a venv no real co-install
    produces: ``uv pip install`` puts ``packaging`` in every venv this chain
    actually targets, so an empty one is unrepresentative of the thing under
    test. The fixture is what is wrong, not the fail-closed readback -- do not
    relax the assertion to make this pass.

    Copied out of the running interpreter rather than installed from an index,
    which keeps the fixture offline and hermetic like the rest of this file.
    """
    spec = importlib.util.find_spec("packaging")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "the interpreter running these tests has no 'packaging'; it cannot "
            "seed the synthetic venv with the distribution a real co-install "
            "venv always carries"
        )
    package_dir = Path(spec.origin).parent
    shutil.copytree(package_dir, site_packages / package_dir.name)

    # The metadata too, so the seeded venv reads back the way a real install
    # does rather than as an importable package no distribution claims.
    dist_info = next(
        iter(sorted(package_dir.parent.glob("packaging-*.dist-info"))), None
    )
    if dist_info is not None:
        shutil.copytree(dist_info, site_packages / dist_info.name)


def _make_real_venv(root: Path) -> tuple[Path, Path]:
    """A real venv. Returns (its python, its site-packages)."""
    venv = root / "target-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
    )
    python_bin = venv / "bin" / "python"
    site_packages = next((venv / "lib").glob("python3.*/site-packages"))
    _seed_packaging(site_packages)
    return python_bin, site_packages


def _write_installed_omnimarket(
    site_packages: Path, *, version: str, commit: str
) -> None:
    """Write the metadata a git co-install of omnimarket would leave behind.

    Real ``.dist-info``, read by the real ``importlib.metadata`` in the target
    interpreter -- which is the whole point of doing this here rather than with
    a shim. ``direct_url.json`` with ``vcs_info.commit_id`` is what a
    ``uv pip install git+...@<sha>`` writes and what every probe in this path
    reads (a PyPI wheel has none, which is OMN-14064).
    """
    for stale in site_packages.glob("omnimarket-*.dist-info"):
        subprocess.run(["rm", "-rf", str(stale)], check=True)
    dist_info = site_packages / f"omnimarket-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: omnimarket\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    (dist_info / "INSTALLER").write_text("uv\n", encoding="utf-8")
    (dist_info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/OmniNode-ai/omnimarket.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": commit,
                    "requested_revision": "dev",
                },
            }
        ),
        encoding="utf-8",
    )


def _installer_that_lands(
    root: Path, site_packages: Path, *, version: str, commit: str
) -> Path:
    """A co-install step that really rewrites the venv's omnimarket metadata."""
    script = root / "landing-install.sh"
    payload = root / "land.py"
    payload.write_text(
        "import json, subprocess, sys\n"
        f"site = {str(site_packages)!r}\n"
        f"version = {version!r}\n"
        f"commit = {commit!r}\n"
        "import pathlib\n"
        "site_path = pathlib.Path(site)\n"
        "for stale in site_path.glob('omnimarket-*.dist-info'):\n"
        "    subprocess.run(['rm', '-rf', str(stale)], check=True)\n"
        "d = site_path / f'omnimarket-{version}.dist-info'\n"
        "d.mkdir(parents=True)\n"
        "(d / 'METADATA').write_text("
        "f'Metadata-Version: 2.1\\nName: omnimarket\\nVersion: {version}\\n')\n"
        "(d / 'RECORD').write_text('')\n"
        "(d / 'INSTALLER').write_text('uv\\n')\n"
        "(d / 'direct_url.json').write_text(json.dumps("
        "{'url': 'https://github.com/OmniNode-ai/omnimarket.git', "
        "'vcs_info': {'vcs': 'git', 'commit_id': commit}}))\n",
        encoding="utf-8",
    )
    script.write_text(
        f"#!/usr/bin/env bash\necho '== install: applying =='\nexec {sys.executable} {payload}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _installer_that_does_nothing(root: Path) -> Path:
    script = root / "noop-install.sh"
    script.write_text(
        "#!/usr/bin/env bash\necho '== install: exiting 0 =='\nexit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _repair(
    omni_home: Path, python_bin: Path, installer: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["OMNI_HOME"] = str(omni_home)
    env["ONEX_DRIFT_INSTALL_SCRIPT"] = str(installer)
    env.pop("ONEX_VENV_RECONCILE_LOCK", None)
    return subprocess.run(
        ["bash", str(_DRIFT_SCRIPT), "--repair", str(python_bin)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_repair_that_lands_is_proven_against_the_real_venv(tmp_path: Path) -> None:
    """The whole chain, with the install actually rewriting site-packages."""
    omni_home, head = _make_omni_home(tmp_path, version="0.4.120")
    python_bin, site_packages = _make_real_venv(tmp_path)
    _write_installed_omnimarket(site_packages, version="0.4.119", commit="c" * 40)

    result = _repair(
        omni_home,
        python_bin,
        _installer_that_lands(tmp_path, site_packages, version="0.4.120", commit=head),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROVEN" in result.stdout
    # The claim is about the real interpreter, so read it back the same way.
    installed = json.loads(
        (
            next(site_packages.glob("omnimarket-*.dist-info")) / "direct_url.json"
        ).read_text(encoding="utf-8")
    )
    assert installed["vcs_info"]["commit_id"] == head


def test_repair_that_does_not_land_is_refused_against_the_real_venv(
    tmp_path: Path,
) -> None:
    """THE DEFECT, end to end: exit 0 from the install is not evidence."""
    omni_home, head = _make_omni_home(tmp_path, version="0.4.120")
    python_bin, site_packages = _make_real_venv(tmp_path)
    stale = "c" * 40
    _write_installed_omnimarket(site_packages, version="0.4.119", commit=stale)

    result = _repair(omni_home, python_bin, _installer_that_does_nothing(tmp_path))

    assert result.returncode == _EXIT_DRIFT, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "DRIFTED" in combined
    assert stale in combined
    assert head in combined


def test_an_absent_omnimarket_is_refused_against_the_real_venv(
    tmp_path: Path,
) -> None:
    """A venv the install emptied reads as drift, never as a pass."""
    omni_home, _head = _make_omni_home(tmp_path, version="0.4.120")
    python_bin, site_packages = _make_real_venv(tmp_path)
    _write_installed_omnimarket(site_packages, version="0.4.119", commit="c" * 40)
    for stale in site_packages.glob("omnimarket-*.dist-info"):
        subprocess.run(["rm", "-rf", str(stale)], check=True)

    result = _repair(omni_home, python_bin, _installer_that_does_nothing(tmp_path))

    assert result.returncode != 0
    assert "not installed" in (result.stdout + result.stderr)
