# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``scripts/onex`` -- the sanctioned ONEX CLI invocation (OMN-17190).

Fully hermetic and offline. Nothing here touches the real workspace: every test
builds a throwaway ``$OMNI_HOME``-shaped tree containing a copy of the wrapper,
a fake CLI-venv entrypoint, a fake reconciler, and a fake ``onex`` on ``PATH``.
No ``uv`` runs and no venv is ever synced.

## What is being pinned, and why it is not a style preference

The OMN-17190 self-heal (guard detects drift -> runs the reconciler -> re-checks
-> proceeds) is correct code that demonstrably works when it is the code that
runs. The verification failure was that through the documented invocation --

    alias onex='uv run --project $OMNI_HOME/omnibase_infra onex'

-- it frequently was NOT the code that ran. ``uv run`` does not pin the command
to the project environment: it prepends that environment's ``bin/`` to ``PATH``
and then resolves the command name normally, so whenever
``<project>/.venv/bin/onex`` is not resolvable it silently executes the first
``onex`` on the inherited ``PATH``. On the verification host that was a
``uv tool`` environment carrying omnibase_infra 0.38.11 (a
``check_omnimarket_drift()`` with no ``reconcile`` parameter -- no self-heal
exists in it) and a PyPI omnimarket with no ``direct_url.json`` (so the guard
reports "NOT INSTALLED from git" unconditionally). That single mechanism
accounts for the whole reported signature: the verbatim pre-OMN-17190 refusal,
no evidence a reconcile was attempted, and a failure against a venv that was
confirmed IN_SYNC moments earlier.

So the invariant under test is INTERPRETER IDENTITY, not drift:

    the sanctioned invocation runs the CLI venv's own entrypoint by absolute
    path, or it refuses -- it never runs some other ``onex``.

Every test below is an assertion about that. ``test_never_execs_the_path_onex_*``
are the direct regression tests for the failure above: they fail if the wrapper
is edited to resolve ``onex`` through ``PATH`` the way ``uv run`` does.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_SOURCE = _REPO_ROOT / "scripts" / "onex"

_EXIT_REFUSED = 2
_SENTINEL_OK = 41  # a status no shell/bash failure mode produces by accident


class _Workspace:
    """A throwaway ``$OMNI_HOME`` tree holding a copy of the wrapper.

    The wrapper derives ``OMNI_HOME`` from its own location when the variable
    is unset, so it must be *copied* into the fake tree rather than invoked
    from the repo -- otherwise a test that omits ``OMNI_HOME`` would resolve
    the real workspace and mutate it.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.infra = root / "omnibase_infra"
        self.scripts = self.infra / "scripts"
        self.scripts.mkdir(parents=True)
        self.venv_bin = self.infra / ".venv" / "bin"
        self.venv_bin.mkdir(parents=True)

        self.wrapper = self.scripts / "onex"
        shutil.copy2(_WRAPPER_SOURCE, self.wrapper)
        self.wrapper.chmod(0o755)

        self.entrypoint = self.venv_bin / "onex"
        self.reconciler = self.scripts / "reconcile-workspace-venvs.sh"

        self.shim_bin = root / "shimbin"
        self.shim_bin.mkdir()
        self.path_onex = self.shim_bin / "onex"
        self.witness = root / "witness.log"

        self.satisfy_floor()

    # -- collaborators ----------------------------------------------------- #
    def satisfy_floor(self) -> None:
        """Provision a floor this workspace already meets (OMN-17309).

        Every test in THIS file is about interpreter identity: which ``onex``
        runs. The OMN-17309 floor gate is a second, orthogonal question -- is
        the workspace one that has been proven -- and it refuses
        evidence-producing subcommands when the answer is no. Several tests here
        drive ``onex node ...``, which is evidence-producing, so without a
        satisfied floor they would measure the floor gate instead of the thing
        they were written to pin.

        The floor is therefore part of the baseline fixture, not part of any
        assertion. The floor gate's own behaviour is pinned in
        ``test_onex_wrapper_floor_omn17309.py``, including the case this
        deliberately excludes: an absent floor.
        """
        site_packages = self.infra / ".venv" / "lib" / "python3.12" / "site-packages"
        dist_info = site_packages / "omnibase_infra-0.38.16.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(
            "Name: omnibase-infra\nVersion: 0.38.16\n", encoding="utf-8"
        )
        (self.root / ".onex-workspace-floor.json").write_text(
            json.dumps(
                {
                    "schema": "onex.workspace.floor.v1",
                    "generated_at": "2026-08-31T00:00:00Z",
                    "host": "test",
                    "omni_home": str(self.root),
                    "distributions": {"omnibase_infra": "0.38.16"},
                    "omnimarket_commit": "",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def install_entrypoint(self) -> None:
        """The real CLI entrypoint: records its argv and exits with a sentinel."""
        self.entrypoint.write_text(
            "#!/usr/bin/env bash\n"
            'printf "ENTRYPOINT %s\\n" "$*" >> "$WITNESS"\n'
            f"exit {_SENTINEL_OK}\n",
            encoding="utf-8",
        )
        self.entrypoint.chmod(0o755)

    def install_path_onex(self) -> None:
        """The stale ``onex`` on PATH -- the thing that must never run.

        Stands in for ``~/.local/bin/onex`` on the verification host: a
        different interpreter that answers the drift check from its own,
        unrelated site-packages.
        """
        self.path_onex.write_text(
            "#!/usr/bin/env bash\n"
            'printf "PATH_ONEX %s\\n" "$*" >> "$WITNESS"\n'
            "echo 'Error: omnimarket is NOT INSTALLED from git in this interpreter' >&2\n"
            "exit 1\n",
            encoding="utf-8",
        )
        self.path_onex.chmod(0o755)

    def install_reconciler(
        self, *, creates_entrypoint: bool, exit_code: int = 0
    ) -> None:
        body = [
            "#!/usr/bin/env bash",
            'printf "RECONCILE %s\\n" "$*" >> "$WITNESS"',
        ]
        if creates_entrypoint:
            body += [
                f'cat > "{self.entrypoint}" <<EOS',
                "#!/usr/bin/env bash",
                'printf "ENTRYPOINT %s\\n" "$*" >> "$WITNESS"',
                f"exit {_SENTINEL_OK}",
                "EOS",
                f'chmod +x "{self.entrypoint}"',
            ]
        body.append(f"exit {exit_code}")
        self.reconciler.write_text("\n".join(body) + "\n", encoding="utf-8")
        self.reconciler.chmod(0o755)

    # -- invocation -------------------------------------------------------- #
    def run(
        self,
        *args: str,
        set_omni_home: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "WITNESS": str(self.witness),
            # A PATH that contains ONLY the shim dir plus the system basics, so
            # the test can never reach a real `onex` on the developer's box.
            "PATH": f"{self.shim_bin}:/usr/bin:/bin",
            **(extra_env or {}),
        }
        if set_omni_home:
            env["OMNI_HOME"] = str(self.root)
        else:
            env.pop("OMNI_HOME", None)
        return subprocess.run(
            [str(self.wrapper), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def witness_lines(self) -> list[str]:
        if not self.witness.exists():
            return []
        return [
            ln for ln in self.witness.read_text(encoding="utf-8").splitlines() if ln
        ]


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path)


# --------------------------------------------------------------------------- #
# The regression: never resolve `onex` through PATH
# --------------------------------------------------------------------------- #
def test_never_execs_the_path_onex_when_entrypoint_exists(
    workspace: _Workspace,
) -> None:
    """A stale ``onex`` sitting first on PATH must be ignored entirely.

    This is the exact shape of the OMN-17190 verification failure: on the host,
    PATH's ``onex`` was a uv-tool build three minor versions behind whose guard
    has no ``reconcile`` parameter, so it emitted the pre-OMN-17190 refusal and
    never attempted a self-heal.
    """
    workspace.install_entrypoint()
    workspace.install_path_onex()

    result = workspace.run("delegate", "reply with the single word: ok")

    assert result.returncode == _SENTINEL_OK, result.stderr
    assert any(ln.startswith("ENTRYPOINT") for ln in workspace.witness_lines())
    assert not any(ln.startswith("PATH_ONEX") for ln in workspace.witness_lines()), (
        "the wrapper executed the PATH `onex` -- this is the uv-run fallback "
        "the wrapper exists to make impossible"
    )


def test_never_execs_the_path_onex_when_entrypoint_is_missing(
    workspace: _Workspace,
) -> None:
    """A missing entrypoint is a refusal, never a fallback.

    ``uv run`` treats "not in the project env" as "look somewhere else". That
    is the defect. Here it must be "repair, or say why not".
    """
    workspace.install_path_onex()
    workspace.install_reconciler(creates_entrypoint=False)

    result = workspace.run("node", "node_aislop_sweep")

    assert result.returncode == _EXIT_REFUSED
    assert not any(ln.startswith("PATH_ONEX") for ln in workspace.witness_lines())


def test_refusal_names_the_path_onex_it_declined_to_run(workspace: _Workspace) -> None:
    """The refusal has to name the thing it did not do, or the operator will
    reasonably assume no other ``onex`` was involved -- which is precisely the
    wrong conclusion on a host that has one."""
    workspace.install_path_onex()
    workspace.install_reconciler(creates_entrypoint=False)

    result = workspace.run("skill", "merge_sweep")

    assert result.returncode == _EXIT_REFUSED
    assert "REFUSED" in result.stderr
    assert str(workspace.entrypoint) in result.stderr
    assert str(workspace.reconciler) in result.stderr
    assert str(workspace.path_onex) in result.stderr


# --------------------------------------------------------------------------- #
# Self-heal: same policy owner as the in-CLI drift guard
# --------------------------------------------------------------------------- #
def test_missing_entrypoint_reconciles_once_then_execs(workspace: _Workspace) -> None:
    workspace.install_path_onex()
    workspace.install_reconciler(creates_entrypoint=True)

    result = workspace.run("delegate", "x")

    assert result.returncode == _SENTINEL_OK, result.stderr
    lines = workspace.witness_lines()
    assert [ln.split()[0] for ln in lines] == ["RECONCILE", "ENTRYPOINT"]
    assert str(workspace.root) in lines[0], (
        "the reconciler must be handed the resolved root"
    )


def test_reconcile_runs_exactly_once_and_does_not_loop(workspace: _Workspace) -> None:
    """A reconcile that ran and did not fix it will not be fixed by running it
    again; a retry loop on the CLI hot path turns a clear refusal into a hang.
    Same rule the drift guard already applies."""
    workspace.install_path_onex()
    workspace.install_reconciler(creates_entrypoint=False)

    result = workspace.run("delegate", "x")

    assert result.returncode == _EXIT_REFUSED
    assert [ln for ln in workspace.witness_lines() if ln.startswith("RECONCILE")] != []
    assert (
        len([ln for ln in workspace.witness_lines() if ln.startswith("RECONCILE")]) == 1
    )


def test_healthy_entrypoint_never_invokes_the_reconciler(workspace: _Workspace) -> None:
    """The wrapper is on the hot path of every dispatch. Reconciling on a
    working install would add a uv round-trip to every single invocation."""
    workspace.install_entrypoint()
    workspace.install_reconciler(creates_entrypoint=False)

    result = workspace.run("node", "x")

    assert result.returncode == _SENTINEL_OK
    assert not any(ln.startswith("RECONCILE") for ln in workspace.witness_lines())


def test_no_reconcile_override_refuses_immediately(workspace: _Workspace) -> None:
    workspace.install_path_onex()
    workspace.install_reconciler(creates_entrypoint=True)

    result = workspace.run(
        "delegate", "x", extra_env={"ONEX_WRAPPER_NO_RECONCILE": "1"}
    )

    assert result.returncode == _EXIT_REFUSED
    assert workspace.witness_lines() == []


# --------------------------------------------------------------------------- #
# Shadow visibility
# --------------------------------------------------------------------------- #
def test_shadowing_path_onex_is_named_on_stderr_even_on_success(
    workspace: _Workspace,
) -> None:
    """A bare ``onex`` typed outside this wrapper -- or run from any script or
    hook -- still resolves through PATH. Silence about that is how the stale
    build stayed invisible for a whole verification session."""
    workspace.install_entrypoint()
    workspace.install_path_onex()

    result = workspace.run("node", "x")

    assert result.returncode == _SENTINEL_OK
    assert "WARNING" in result.stderr
    assert str(workspace.path_onex) in result.stderr
    assert str(workspace.entrypoint) in result.stderr


def test_no_warning_when_nothing_shadows_the_entrypoint(workspace: _Workspace) -> None:
    workspace.install_entrypoint()

    result = workspace.run("node", "x")

    assert result.returncode == _SENTINEL_OK
    assert "WARNING" not in result.stderr


# --------------------------------------------------------------------------- #
# Invocation contract
# --------------------------------------------------------------------------- #
def test_argv_is_passed_through_verbatim(workspace: _Workspace) -> None:
    """Including arguments that look like flags the wrapper itself might own --
    the wrapper parses nothing and must stay transparent."""
    workspace.install_entrypoint()

    result = workspace.run("delegate", "--verbose", "--omni-home", "/elsewhere", "a b")

    assert result.returncode == _SENTINEL_OK
    assert (
        "ENTRYPOINT delegate --verbose --omni-home /elsewhere a b"
        in workspace.witness_lines()[0]
    )


def test_omni_home_is_derived_from_the_script_location_when_unset(
    workspace: _Workspace,
) -> None:
    """Deriving is exact, not a guessed default: this file lives at
    ``<omni_home>/omnibase_infra/scripts/onex``, so its own path *is* the
    answer. A wrapper that hard-failed here would be unusable from cron, from
    launchd, and from any hook that does not export the variable."""
    workspace.install_entrypoint()

    result = workspace.run("node", "x", set_omni_home=False)

    assert result.returncode == _SENTINEL_OK, result.stderr
    assert any(ln.startswith("ENTRYPOINT") for ln in workspace.witness_lines())


def test_explicit_omni_home_wins_over_the_derived_root(tmp_path: Path) -> None:
    """Two checkouts, one wrapper: the exported root selects which venv runs."""
    wrapper_home = _Workspace(tmp_path / "wrapper_home")
    target_home = _Workspace(tmp_path / "target_home")
    target_home.install_entrypoint()
    target_home.witness = wrapper_home.witness

    env = {
        **os.environ,
        "WITNESS": str(wrapper_home.witness),
        "PATH": "/usr/bin:/bin",
        "OMNI_HOME": str(target_home.root),
    }
    result = subprocess.run(
        [str(wrapper_home.wrapper), "node", "x"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == _SENTINEL_OK, result.stderr
    assert any(ln.startswith("ENTRYPOINT") for ln in wrapper_home.witness_lines())


def test_wrapper_is_executable_in_the_repo() -> None:
    """It is invoked as ``$OMNI_HOME/omnibase_infra/scripts/onex`` from a shell
    alias; a non-executable file makes that alias fail on a fresh clone."""
    assert os.access(_WRAPPER_SOURCE, os.X_OK)
