# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``scripts/reconcile-host.sh`` (OMN-17307).

Hermetic and offline. Every test builds a throwaway ``$OMNI_HOME`` tree with
real local git repositories (a bare "origin" plus a clone, so ``git fetch``
works over the filesystem with no network), a hand-built ``site-packages``, and
STUBBED delegates. No ``uv`` runs, no real venv is synced, and no real clone is
touched.

The delegates are stubbed on purpose rather than exercised: the property under
test is not whether the clone reconciler or the venv reconciler works. It is
that **this** script's verdict does not depend on their exit status. Every stub
below exits 0. The tests that must fail are the ones where a stub exits 0 and
changes nothing -- which is precisely the shape all four motivating incidents
took, and precisely what an exit-code-driven reconciler cannot see.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-host.sh"
_VERIFIER = _REPO_ROOT / "scripts" / "reconcile_verify_movement.py"
_MANIFEST = _REPO_ROOT / "scripts" / "runtime_build" / "sibling_clone_manifest.sh"

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_INDETERMINATE = 3

# Index-aligned with the shipped manifest; asserted below so this file cannot
# quietly drift from the single source of truth the script sources.
GOVERNED = (
    "omnibase_infra",
    "omnibase_core",
    "omnibase_spi",
    "omnibase_compat",
    "omnimarket",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class Workspace:
    root: Path
    infra: Path
    scripts: Path
    site_packages: Path
    alert_witness: Path
    delegate_witness: Path

    @property
    def floor(self) -> Path:
        return self.root / ".onex-workspace-floor.json"

    @property
    def receipt(self) -> Path:
        return self.root / ".onex-workspace-reconcile.json"


def _make_clone(root: Path, name: str) -> tuple[Path, Path]:
    """A bare origin plus a working clone of it, both on ``dev``."""
    origin = root / "_origins" / f"{name}.git"
    origin.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", "--bare", "-b", "dev", str(origin)], check=True
    )

    seed = root / "_seed" / name
    seed.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", "-b", "dev", str(seed)], check=True)
    _git(seed, "config", "user.email", "t@example.invalid")
    _git(seed, "config", "user.name", "t")
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "--quiet", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "--quiet", "origin", "dev")

    clone = root / name
    subprocess.run(
        ["git", "clone", "--quiet", "-b", "dev", str(origin), str(clone)], check=True
    )
    _git(clone, "config", "user.email", "t@example.invalid")
    _git(clone, "config", "user.name", "t")
    return clone, seed


def _advance_origin(seed: Path, message: str) -> str:
    (seed / f"{message}.txt").write_text(message, encoding="utf-8")
    _git(seed, "add", "-A")
    _git(seed, "commit", "--quiet", "-m", message)
    _git(seed, "push", "--quiet", "origin", "dev")
    return _git(seed, "rev-parse", "HEAD")


def _write_dist(
    site_packages: Path, name: str, version: str, commit: str | None = None
) -> None:
    d = site_packages / f"{name}-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    if commit:
        (d / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "https://github.com/OmniNode-ai/omnimarket.git",
                    "vcs_info": {"vcs": "git", "commit_id": commit},
                }
            ),
            encoding="utf-8",
        )


def _stub(path: Path, witness: Path, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$0 $*" >> {witness}
            {body}
            exit 0
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "omni_home"
    infra = root / "omnibase_infra"
    scripts = infra / "scripts"
    scripts.mkdir(parents=True)

    shutil.copy2(_SCRIPT, scripts / "reconcile-host.sh")
    (scripts / "reconcile-host.sh").chmod(0o755)
    shutil.copy2(_VERIFIER, scripts / "reconcile_verify_movement.py")
    (scripts / "runtime_build").mkdir()
    shutil.copy2(_MANIFEST, scripts / "runtime_build" / "sibling_clone_manifest.sh")

    site_packages = infra / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)

    return Workspace(
        root=root,
        infra=infra,
        scripts=scripts,
        site_packages=site_packages,
        alert_witness=tmp_path / "alerts.log",
        delegate_witness=tmp_path / "delegates.log",
    )


def _lock(ws: Workspace, **versions: str) -> None:
    blocks = "\n".join(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in versions.items()
    )
    (ws.infra / "uv.lock").write_text(f"version = 1\n\n{blocks}", encoding="utf-8")


def _run(
    ws: Workspace, *args: str, omni_home: str | None = "auto"
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("OMNI_HOME", None)
    if omni_home == "auto":
        env["OMNI_HOME"] = str(ws.root)
    elif omni_home is not None:
        env["OMNI_HOME"] = omni_home
    env["ONEX_RECONCILE_ALERT_CMD"] = f"{_writer_stub(ws)} {ws.alert_witness}"
    return subprocess.run(
        ["bash", str(ws.scripts / "reconcile-host.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def _writer_stub(ws: Workspace) -> Path:
    """A tiny ``sh`` script standing in for the Slack post.

    The alert transport is best-effort by design, so proving "it alerted" needs
    a seam. This is that seam and nothing more: it appends its argv to a file.
    """
    stub = ws.root.parent / "alertcmd.sh"
    if not stub.exists():
        stub.write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$2" >> "$1"\n', encoding="utf-8"
        )
        stub.chmod(0o755)
    return stub


# --------------------------------------------------------------------------- #
# The manifest is the single source of truth for scope
# --------------------------------------------------------------------------- #
def test_governed_repo_set_matches_the_shipped_manifest() -> None:
    """This test file must not carry its own idea of which repos are in scope.

    OMN-15137 is the precedent: a hand-maintained second copy of the repo list
    drifted from the first. Asserting the constant against the manifest keeps
    this file from becoming a third.
    """
    text = _MANIFEST.read_text(encoding="utf-8")
    block = text.split("SIBLING_CLONE_MANIFEST=(", 1)[1].split(")", 1)[0]
    declared = tuple(line.strip().strip('"') for line in block.strip().splitlines())
    assert declared == GOVERNED


# --------------------------------------------------------------------------- #
# Configuration: fail fast, never guess (rule 8)
# --------------------------------------------------------------------------- #
def test_unset_omni_home_is_indeterminate_not_a_guessed_default(ws: Workspace) -> None:
    proc = _run(ws, omni_home=None)
    assert proc.returncode == EXIT_INDETERMINATE
    assert "OMNI_HOME is not set" in proc.stderr
    assert not ws.floor.exists()


# --------------------------------------------------------------------------- #
# AC1 -- the core.bare trap, asserted at the orchestrator
# --------------------------------------------------------------------------- #
def test_core_bare_clone_fails_even_though_the_delegate_exited_zero(
    ws: Workspace,
) -> None:
    """The `.201` 2026-08-31 shape, reproduced end to end.

    ``git fetch`` on this clone exits 0 and ``git checkout`` exits 128, so the
    delegate below (which exits 0 and does nothing, exactly like a fetch-only
    loop) must NOT be able to make this run pass.
    """
    clone, _seed = _make_clone(ws.root, "omnibase_core")
    _git(clone, "config", "core.bare", "true")
    _lock(ws, **{"omnibase-core": "0.46.9"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")

    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "clone:omnibase_core" in proc.stderr
    assert "core.bare" in proc.stderr
    assert not ws.floor.exists(), "a failed run must not stamp the floor"


def test_clone_behind_origin_fails_when_the_delegate_is_a_no_op(ws: Workspace) -> None:
    """AC1's general case: the delegate exits 0 and the clone never advances."""
    clone, seed = _make_clone(ws.root, "omnibase_core")
    ahead = _advance_origin(seed, "moved")
    assert _git(clone, "rev-parse", "HEAD") != ahead

    _lock(ws, **{"omnibase-core": "0.46.9"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "clone:omnibase_core: DID_NOT_MOVE" in proc.stderr
    assert not ws.floor.exists()


def test_clone_that_actually_advanced_passes(ws: Workspace) -> None:
    """The positive control: a delegate that really moves the clone passes.

    Without this, every assertion above would also be satisfied by a script that
    simply always fails.
    """
    clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "moved")

    _lock(ws, **{"omnibase-core": "0.46.9"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
        body=f'git -C "{clone}" fetch --quiet origin dev && '
        f'git -C "{clone}" reset --hard --quiet origin/dev',
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK, proc.stderr
    assert "clone:omnibase_core: MOVED" in proc.stderr
    assert ws.floor.exists()


# --------------------------------------------------------------------------- #
# AC2 -- the venv surface
# --------------------------------------------------------------------------- #
def test_venv_below_lock_target_fails_when_the_delegate_exited_zero(
    ws: Workspace,
) -> None:
    """The OMN-16262 shape: the repair ran, exited 0, and the pin is still wrong."""
    _make_clone(ws.root, "omnibase_core")
    _lock(ws, **{"omnibase-core": "0.46.9", "omnibase-compat": "0.5.6"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")
    _write_dist(ws.site_packages, "omnibase_compat", "0.5.5")  # the downgrade

    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
        body=f'git -C "{ws.root}/omnibase_core" reset --hard --quiet origin/dev',
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "venv:omnibase-compat: DID_NOT_MOVE" in proc.stderr
    assert "0.5.5" in proc.stderr and "0.5.6" in proc.stderr
    assert not ws.floor.exists()


def test_omnimarket_commit_is_verified_against_the_clone_head(ws: Workspace) -> None:
    """omnimarket is not in the lock, so its target is the canonical clone HEAD.

    That is the same comparison the OMN-14060 drift guard makes, and pinning it
    here means the reconciler cannot report success on a venv the guard will
    then refuse.
    """
    market, _ = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _lock(ws)
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit="0" * 40)

    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "venv:omnimarket: DID_NOT_MOVE" in proc.stderr
    assert head[:12] in proc.stderr


# --------------------------------------------------------------------------- #
# AC3 -- unreadable is a failure, never an assumption
# --------------------------------------------------------------------------- #
def test_pypi_installed_omnimarket_with_no_direct_url_is_indeterminate(
    ws: Workspace,
) -> None:
    """No ``direct_url.json`` means "cannot tell", which must fail closed.

    This is the exact state the OMN-17190 foreign interpreter was in.
    """
    _make_clone(ws.root, "omnimarket")
    _lock(ws)
    _write_dist(ws.site_packages, "omnimarket", "0.4.10")  # no commit

    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "venv:omnimarket: INDETERMINATE" in proc.stderr
    assert not ws.floor.exists()


# --------------------------------------------------------------------------- #
# Uncovered surface is a failure, not a skip
# --------------------------------------------------------------------------- #
def test_missing_clone_delegate_is_uncovered_not_skipped(ws: Workspace) -> None:
    _make_clone(ws.root, "omnibase_core")
    _lock(ws)
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "clone-surface: UNCOVERED" in proc.stderr
    assert "reconciled by nobody" in proc.stderr


def test_missing_venv_delegate_is_uncovered_not_skipped(ws: Workspace) -> None:
    _make_clone(ws.root, "omnibase_core")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
        body=f'git -C "{ws.root}/omnibase_core" reset --hard --quiet origin/dev',
    )

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert "venv-surface: UNCOVERED" in proc.stderr


# --------------------------------------------------------------------------- #
# AC4 -- alert on failure, and never a success line
# --------------------------------------------------------------------------- #
def test_failure_alerts_and_leaves_the_previous_floor_untouched(ws: Workspace) -> None:
    _clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "moved")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    previous = '{"schema": "onex.workspace.floor.v1", "proven": "earlier"}\n'
    ws.floor.write_text(previous, encoding="utf-8")

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED
    assert ws.alert_witness.exists(), "a failing surface must alert"
    alert_text = ws.alert_witness.read_text(encoding="utf-8")
    assert "omnibase_core" in alert_text
    assert ws.floor.read_text(encoding="utf-8") == previous, (
        "a failed reconcile must leave the last PROVEN floor in place, not "
        "overwrite it and not delete it"
    )
    assert "VERDICT: FAILED" in proc.stderr
    assert "VERDICT: IN_SYNC" not in proc.stderr


def test_success_does_not_alert(ws: Workspace) -> None:
    _make_clone(ws.root, "omnibase_core")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
        body=f'git -C "{ws.root}/omnibase_core" reset --hard --quiet origin/dev',
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK, proc.stderr
    assert not ws.alert_witness.exists()


# --------------------------------------------------------------------------- #
# AC5 -- idempotence
# --------------------------------------------------------------------------- #
def test_two_consecutive_runs_both_pass_and_the_second_mutates_nothing(
    ws: Workspace,
) -> None:
    market, _ = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _lock(ws, **{"omnibase-core": "0.46.9"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    _make_clone(ws.root, "omnibase_core")
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    first = _run(ws)
    assert first.returncode == EXIT_OK, first.stderr
    assert "ALREADY_AT_TARGET" in first.stderr
    floor_first = json.loads(ws.floor.read_text(encoding="utf-8"))

    second = _run(ws)
    assert second.returncode == EXIT_OK, second.stderr
    floor_second = json.loads(ws.floor.read_text(encoding="utf-8"))

    assert floor_first["distributions"] == floor_second["distributions"]
    assert floor_first["omnimarket_commit"] == floor_second["omnimarket_commit"]


# --------------------------------------------------------------------------- #
# --check is read-only
# --------------------------------------------------------------------------- #
def test_check_mode_runs_no_delegate_and_stamps_no_floor(ws: Workspace) -> None:
    _make_clone(ws.root, "omnibase_core")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws, "--check")

    assert proc.returncode == EXIT_OK, proc.stderr
    assert not ws.delegate_witness.exists(), "--check must mutate nothing"
    assert not ws.floor.exists(), "--check must not stamp a floor it did not prove"


def test_check_mode_still_reports_drift(ws: Workspace) -> None:
    _clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "moved")
    _lock(ws)

    proc = _run(ws, "--check")

    assert proc.returncode == EXIT_FAILED
    assert "clone:omnibase_core: DID_NOT_MOVE" in proc.stderr


# --------------------------------------------------------------------------- #
# Receipt
# --------------------------------------------------------------------------- #
def test_receipt_is_written_on_both_outcomes(ws: Workspace) -> None:
    _clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "moved")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)

    proc = _run(ws)
    assert proc.returncode == EXIT_FAILED

    receipt = json.loads(ws.receipt.read_text(encoding="utf-8"))
    assert receipt["schema"] == "onex.workspace.reconcile.v1"
    assert receipt["failures"] >= 1
    surfaces = {s["surface"]: s["verdict"] for s in receipt["surfaces"]}
    assert surfaces["clone:omnibase_core"] == "DID_NOT_MOVE"


def test_script_is_executable_in_the_repo() -> None:
    assert os.access(_SCRIPT, os.X_OK), "the scheduler invokes this directly"
