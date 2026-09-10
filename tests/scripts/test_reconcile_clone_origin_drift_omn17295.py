# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The venv reconciler must not call a workspace IN_SYNC while it is behind dev (OMN-17295).

THE DEFECT

``reconcile-workspace-venvs.sh`` compares exactly one pair: the installed
``omnimarket`` commit against the local clone's ``HEAD``. That is the correct
target for the *provider layer* -- the OMN-14060 drift guard compares against
the same thing, and pinning to ``origin/dev`` instead would leave the venv
*ahead* of the clone and still refused (OMN-16366). Nothing about that changes
here.

What is wrong is the VERDICT it prints on top of that one comparison::

    [reconcile] verdict: IN_SYNC (omnimarket 87afb9c33215)

A clone eleven commits behind ``origin/dev`` produces that line, unchanged.
Every word of it is true about the layer the script owns and false about the
workspace the reader is asking about -- and the SessionStart hook advertises
this exact command as the way to "settle it now" when no reconcile tick has
run. That is the OMN-17295 defect class stated in the ticket's own words: a
probe that silently measures something narrower than it names does not fail
loudly, it produces a confident wrong answer.

AC5 is "local venv -> origin/dev drift is detected on this Mac, not just
venv -> clone", and it says to coordinate with OMN-17291 rather than duplicate
its reconciler. So what this file pins down is DETECTION, and the absence of a
second repair path:

* the clone -> ``origin/<branch>`` leg is observed and reported, per clone,
  with the behind-count;
* a clone behind the tracked branch makes the verdict DRIFT, not IN_SYNC;
* the observation NEVER fetches. ``reconcile-host.sh`` fetches, under an
  ownership plan, precisely because a fetch writes objects, refs and reflogs
  into the clone -- and this script's ``--check`` mode deliberately exempts
  itself from that ownership plan on the grounds that it writes nothing
  (OMN-17366). Adding a fetch here would silently void that exemption and put
  root-owned objects back inside operator-owned clones on the ``.201`` cron
  path. So the target is read as last fetched, and the report says so.
* the clone is never advanced here. There is exactly one clone reconciler
  (``scripts/runtime_build/reconcile_deploy_clones.sh``, OMN-17291) and one
  venv reconciler (this script), composed by ``reconcile-host.sh``; a
  fast-forward added here would be the third implementation that composition
  exists to prevent.
* the composed layer still installs. ``omnimarket`` + its ``--no-deps``
  companions living in ``omnibase_infra``'s venv is BY DESIGN, and a stale
  clone must not turn into a reason to skip or strip it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

_EXIT_OK = 0
_EXIT_DRIFT = 1

_SHORT = 12


# --------------------------------------------------------------------------- #
# Fixture construction -- fully offline, no network, no real remote
# --------------------------------------------------------------------------- #
def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "f.txt").write_text(text, encoding="utf-8")
    _git("add", "f.txt", cwd=repo)
    _git("commit", "--quiet", "-m", text, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _make_uv_shim(bin_dir: Path) -> None:
    """``uv`` on PATH: logs argv, and answers every ``--check`` probe with 0.

    The lock layer is deliberately conformant in every test here, so the only
    thing that can move the verdict is the leg under test.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$UV_SHIM_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)


def _make_install_shim(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _make_fake_venv(project: Path, installed_commit: str | None) -> None:
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    python = venv / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "cat >/dev/null 2>&1 || true\n"
        f"printf '%s\\n' '{installed_commit or ''}'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)


class _Workspace:
    """An ``$OMNI_HOME`` whose ``omnimarket`` clone has a real (local) origin.

    The existing OMN-17190 fixture builds a clone with no remote at all, which
    is the "nothing to be stale against" case. This one adds the remote, so the
    clone -> ``origin/<branch>`` leg has something to observe.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True)

        # A bare repo standing in for GitHub. Never fetched over a network.
        self.remote = root / "remotes" / "omnimarket.git"
        self.remote.mkdir(parents=True)
        _git("init", "--quiet", "--bare", "-b", "dev", cwd=self.remote)

        self.omnimarket = root / "omnimarket"
        self.omnimarket.mkdir()
        _git("init", "--quiet", "-b", "dev", cwd=self.omnimarket)
        _git("config", "user.email", "test@example.com", cwd=self.omnimarket)
        _git("config", "user.name", "Test", cwd=self.omnimarket)
        _git("remote", "add", "origin", str(self.remote), cwd=self.omnimarket)
        self.base = _commit(self.omnimarket, "one")
        _git("push", "--quiet", "-u", "origin", "dev", cwd=self.omnimarket)

        self.infra = root / "omnibase_infra"
        (self.infra / "scripts").mkdir(parents=True)
        (self.infra / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
        self.install_script = self.infra / "scripts" / "install-node-skill-package.sh"
        _make_install_shim(self.install_script)

        self.omniclaude = root / "omniclaude"
        self.omniclaude.mkdir()
        (self.omniclaude / "uv.lock").write_text("claude-lock-v1\n", encoding="utf-8")
        _make_fake_venv(self.omniclaude, None)

        # The venv is at the clone HEAD: the venv -> clone leg is IN_SYNC, so a
        # DRIFT verdict in these tests can only come from the new leg.
        _make_fake_venv(self.infra, self.base)

        self.bin_dir = root / "shimbin"
        self.uv_log = root / "uv.log"
        self.install_log = root / "install.log"
        _make_uv_shim(self.bin_dir)

    # -- manipulation ------------------------------------------------------- #
    def advance_remote(self, text: str) -> str:
        """Move ``origin/dev`` ahead of the clone, updating the tracking ref.

        The push updates ``refs/remotes/origin/dev`` in the clone as a side
        effect, which is what makes this observable WITHOUT a fetch -- exactly
        the state a real workspace is in after ``reconcile-host.sh`` fetched.
        """
        head_before = _git("rev-parse", "HEAD", cwd=self.omnimarket)
        tip = _commit(self.omnimarket, text)
        _git("push", "--quiet", "origin", "dev", cwd=self.omnimarket)
        _git("reset", "--hard", "--quiet", head_before, cwd=self.omnimarket)
        return tip

    def advance_remote_without_tracking_ref(self, text: str) -> str:
        """Move the bare repo's ``dev`` and leave the tracking ref untouched.

        A separate working copy pushes, so nothing in the clone under test
        learns about it. If the reconciler ever starts fetching, it will see
        this commit -- and that is the assertion.
        """
        scratch = self.root / "scratch"
        _git("clone", "--quiet", str(self.remote), str(scratch), cwd=self.root)
        _git("config", "user.email", "test@example.com", cwd=scratch)
        _git("config", "user.name", "Test", cwd=scratch)
        tip = _commit(scratch, text)
        _git("push", "--quiet", "origin", "dev", cwd=scratch)
        return tip

    def set_installed_commit(self, commit: str | None) -> None:
        _make_fake_venv(self.infra, commit)

    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.omnimarket)

    # -- invocation --------------------------------------------------------- #
    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "OMNI_HOME": str(self.root),
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "UV_SHIM_LOG": str(self.uv_log),
            "INSTALL_SHIM_LOG": str(self.install_log),
            "ONEX_RECONCILE_INSTALL_SCRIPT": str(self.install_script),
            "CLAUDE_PLUGIN_DATA": str(self.root / "no-such-plugin-data"),
        }

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(_SCRIPT), *args],
            capture_output=True,
            text=True,
            env=self.env(),
            check=False,
        )

    def install_refs(self) -> list[str]:
        if not self.install_log.exists():
            return []
        return [
            line
            for line in self.install_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


# --------------------------------------------------------------------------- #
# The detection leg AC5 asks for
# --------------------------------------------------------------------------- #
def test_check_reports_drift_when_the_clone_is_behind_origin_dev(
    ws: _Workspace,
) -> None:
    """The whole ticket in one assertion.

    The venv matches the clone exactly, so the pre-OMN-17295 script prints
    ``verdict: IN_SYNC``. The clone is one commit behind ``origin/dev``, so the
    workspace is not in sync with what is merged, and the verdict must say so.
    """
    tip = ws.advance_remote("two")
    assert tip != ws.head()

    result = ws.run("--check")
    out = result.stdout + result.stderr

    assert result.returncode == _EXIT_DRIFT, (
        "a clone behind origin/dev must not verdict IN_SYNC; got "
        f"exit {result.returncode}\n{out}"
    )
    assert "verdict: DRIFT" in out, out


def test_check_names_the_clone_the_two_shas_and_the_behind_count(
    ws: _Workspace,
) -> None:
    """A verdict that does not say WHICH clone and HOW FAR is a dead end.

    The reader has to be able to act on the line without running a second
    command, which is the same standard every other refusal in this script is
    held to.
    """
    tip = ws.advance_remote("two")
    head = ws.head()

    out = ws.run("--check").stdout + ws.run("--check").stderr

    assert "omnimarket" in out, out
    assert head[:_SHORT] in out, f"clone HEAD {head[:_SHORT]} missing from:\n{out}"
    assert tip[:_SHORT] in out, f"origin tip {tip[:_SHORT]} missing from:\n{out}"
    assert "behind" in out.lower(), out
    assert "1" in out, out


def test_check_is_in_sync_when_the_clone_is_at_origin_dev(ws: _Workspace) -> None:
    """The green path stays green: at the tip, with a conformant venv, IN_SYNC."""
    result = ws.run("--check")
    out = result.stdout + result.stderr

    assert result.returncode == _EXIT_OK, out
    assert "verdict: IN_SYNC" in out, out


def test_check_does_not_fetch_the_tracking_ref(ws: _Workspace) -> None:
    """Read the target as last fetched; never fetch it here.

    ``--check`` exempts itself from the OMN-17366 ownership plan because it
    writes nothing. A fetch writes -- objects, refs and reflogs -- so a fetch
    added here would void that exemption silently, and on the ``.201`` root
    cron it would deposit root-owned objects into an operator-owned clone.
    """
    unseen = ws.advance_remote_without_tracking_ref("remote-only")

    out = ws.run("--check").stdout + ws.run("--check").stderr

    assert unseen[:_SHORT] not in out, (
        "the reconciler fetched: it reported a commit that only exists on the "
        f"remote ({unseen[:_SHORT]})\n{out}"
    )


def test_clone_with_no_origin_remote_is_not_reported_as_drift(
    tmp_path: Path,
) -> None:
    """A clone with no remote has nothing to be stale against.

    This is the shape of every hermetic fixture in the OMN-17190 suite, and
    manufacturing a failure out of it would make the new leg fire on workspaces
    where the question is not even askable.
    """
    ws = _Workspace(tmp_path / "omni_home")
    _git("remote", "remove", "origin", cwd=ws.omnimarket)
    _git("update-ref", "-d", "refs/remotes/origin/dev", cwd=ws.omnimarket)

    result = ws.run("--check")
    out = result.stdout + result.stderr

    assert result.returncode == _EXIT_OK, out
    assert "verdict: IN_SYNC" in out, out


def test_tracking_ref_that_was_never_fetched_is_drift_not_silence(
    ws: _Workspace,
) -> None:
    """An unknown target renders as unproven, never as fresh.

    A clone that HAS an origin but no ``origin/<branch>`` ref has never been
    fetched. Reporting IN_SYNC there is the same substitution as reporting a
    stale cached verdict as a current one.
    """
    _git("update-ref", "-d", "refs/remotes/origin/dev", cwd=ws.omnimarket)

    result = ws.run("--check")
    out = result.stdout + result.stderr

    assert result.returncode == _EXIT_DRIFT, out
    assert "omnimarket" in out, out


# --------------------------------------------------------------------------- #
# What must NOT change: the composed layer, and the single clone reconciler
# --------------------------------------------------------------------------- #
def test_repair_still_installs_the_composed_layer_when_the_clone_is_stale(
    ws: _Workspace,
) -> None:
    """A stale clone is not a reason to skip or strip the provider layer.

    ``omnimarket`` + its ``--no-deps`` companions in ``omnibase_infra``'s venv
    is by design; without it every ``onex skill`` / ``onex delegate`` dispatch
    dies on the OMN-14060 guard. The pin is still the LOCAL clone HEAD, because
    that is what the guard compares against (OMN-16366).
    """
    ws.advance_remote("two")
    head = ws.head()
    ws.set_installed_commit(None)  # force the provider layer to be (re)installed

    ws.run()

    assert ws.install_refs() == [head], (
        "the provider co-install must still run, pinned to the local clone "
        f"HEAD {head}; got {ws.install_refs()}"
    )


def test_repair_never_advances_the_clone(ws: _Workspace) -> None:
    """Detection here, repair in the one clone reconciler (OMN-17291).

    ``reconcile-host.sh`` composes the two and owns the ordering. A
    fast-forward added to this script would be the third implementation of a
    clone reconciler in one repo, which is what that composition exists to
    prevent -- and AC5 says to coordinate with OMN-17291, not duplicate it.
    """
    ws.advance_remote("two")
    before = ws.head()

    ws.run()

    assert ws.head() == before, (
        "the venv reconciler moved the canonical clone; advancing clones "
        "belongs to scripts/runtime_build/reconcile_deploy_clones.sh"
    )
