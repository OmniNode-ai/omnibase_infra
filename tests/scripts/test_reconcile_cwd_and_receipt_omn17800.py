# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A privilege-dropped child may not inherit the caller's CWD (OMN-17800).

THE INCIDENT

`.201`'s hourly root reconcile cron reported ``"failures": 3`` on every tick for
as long as it had existed. All five clone surfaces reconciled; all three venv
surfaces did not::

    {"surface": "venv:omnibase-infra", "verdict": "DID_NOT_MOVE",
     "detail": "observed 0.38.16 but target is 0.38.18 ..."}
    {"surface": "venv:omnibase-core",  "verdict": "DID_NOT_MOVE", ...}
    {"surface": "venv:omnimarket",     "verdict": "DID_NOT_MOVE", ...}

and the log said why, 67 times over::

    == step 1: install omnimarket + omni-internal leaf deps (--no-deps) ==
    error: failed to open file `/root/uv.toml`: Permission denied (os error 13)
    [reconcile] FAILED: provider co-install did not complete; omnimarket is not installed.

Nothing about that message is about uv.toml. It is the shape of a privilege drop
that moves the user but not the ground under it:

1. cron runs the reconciler as root, so the shell's CWD is ``/root`` (0700).
2. ``as_owner`` drops to the surface owner via
   ``runuser -u <owner> -- env HOME=...``. ``runuser`` without ``-l`` changes
   UID, GID and HOME -- and **not** the inherited working directory. The child
   is now the operator, standing in root's home, which the operator may not
   even stat.
3. ``install-node-skill-package.sh`` calls ``uv pip install``, and uv discovers
   configuration by walking UP from the working directory. The first thing it
   reaches is ``/root/uv.toml``, and it cannot open it.

The co-install is the FIRST step of the venv repair and forces the lock pass
that follows it, so one inaccessible directory took out all three venv surfaces
at once -- while the identical script on the Mac reported ``"failures": 0``,
because there the reconciler runs as the surface owner, ``RUN_AS`` is empty, and
the ``/root`` precondition cannot occur.

WHY THIS WAS NOT ALREADY GUARDED

Every other ``as_owner``-wrapped package operation in the venv reconciler is
already written ``(cd "$project" && as_owner ...)``. The provider co-install was
the single site without it. OMN-17383 had already fixed a defect on that exact
line -- the child could not find ``uv`` because it inherited the cron PATH --
and the working directory is the same defect one variable over: the child
inherits an environment the parent never chose for it.

So the test below asserts the general property, not the ``/root`` symptom: the
co-install's working directory is one the reconciler PICKED, and configuration
sitting beside the caller is not visible from it.

THE SECOND DEFECT, FOUND WHILE PROVING THE FIRST

``reconcile-host.sh`` builds the receipt's ``surfaces`` array with
``sep=",\\n"``. Bash does not interpret ``\\n`` in a plain double-quoted
assignment -- only ``$'...'`` does -- and the value is then passed as a printf
``%s`` ARGUMENT, where printf does not interpret escapes either. So the receipt
carries the literal three characters ``,\\n`` between elements and is not valid
JSON::

    json.decoder.JSONDecodeError: Expecting value: line 8 column 159 (char 333)

The existing receipt test did not catch it because its workspace produces
exactly ONE surface, and the separator is only emitted from the second element
onward. That is the whole lesson: a separator is untested until something is
separated. The test here reports two surfaces.

This one was live on BOTH hosts, including the "healthy" Mac receipt -- an
invalid artifact that a passing run had been writing all along, in a script
whose own header says "Invalid evidence is worse than no evidence, because it
outlives the invocation".
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_FAILED,
    Workspace,
    _advance_origin,
    _lock,
    _make_clone,
    _run,
    _stub,
    build_workspace,
)
from tests.scripts.test_reconcile_workspace_venvs import _make_uv_shim, _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"
_GATE = _REPO_ROOT / "scripts" / "check_reconciler_privilege.py"


# --------------------------------------------------------------------------- #
# Defect 1 -- the co-install's working directory
# --------------------------------------------------------------------------- #
def _install_shim_recording_cwd(path: Path, probe_log: Path) -> None:
    """Stand-in for the co-install that reports the ground it is standing on.

    It records two things, and the second is the one that matters. ``CHILD_PWD``
    proves which directory the child got. ``CHILD_UV_TOML`` reproduces uv's
    actual config discovery -- walk UP from the working directory until a
    ``uv.toml`` appears -- which is the mechanism that failed on `.201`.
    Asserting on the directory alone would pin an implementation detail;
    asserting on what uv would FIND from there pins the behaviour.
    """
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "CHILD_PWD=%s\\n" "$PWD" >> "{probe_log}"\n'
        'd="$PWD"\n'
        "found=NONE\n"
        'while [[ -n "$d" && "$d" != "/" ]]; do\n'
        '  if [[ -e "$d/uv.toml" ]]; then found="$d/uv.toml"; break; fi\n'
        '  d="$(dirname "$d")"\n'
        "done\n"
        f'printf "CHILD_UV_TOML=%s\\n" "$found" >> "{probe_log}"\n'
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        'printf "install %s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$ORDER_LOG"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _recorded(probe_log: Path, key: str) -> str:
    lines = probe_log.read_text(encoding="utf-8").splitlines()
    return next(line.split("=", 1)[1] for line in lines if line.startswith(f"{key}="))


def _run_with_caller_cwd(tmp_path: Path) -> tuple[_Workspace, Path, Path]:
    """Run the venv reconciler from a caller directory that is NOT the workspace.

    The caller's directory holds a ``uv.toml``. On `.201` that file was
    ``/root/uv.toml`` and it was unreadable; here it is readable, which makes the
    test STRONGER rather than weaker -- an unreadable file could only prove the
    child failed, while a readable one proves the child never had any business
    looking there in the first place.
    """
    workspace = _Workspace(tmp_path / "omni_home")
    # Provider drift is what makes the reconciler run the co-install at all.
    workspace.set_installed_commit("0" * 40)

    caller_cwd = tmp_path / "caller_cwd"
    caller_cwd.mkdir()
    (caller_cwd / "uv.toml").write_text(
        "# stands in for /root/uv.toml on the .201 cron\n", encoding="utf-8"
    )

    probe_log = tmp_path / "child.log"
    _install_shim_recording_cwd(workspace.install_script, probe_log)
    _make_uv_shim(workspace.bin_dir)

    result = subprocess.run(
        ["bash", str(_VENV_SCRIPT)],
        capture_output=True,
        text=True,
        env=workspace.env(),
        cwd=str(caller_cwd),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert probe_log.exists(), "the co-install never ran"
    return workspace, caller_cwd, probe_log


def test_the_co_install_does_not_run_in_the_callers_directory(tmp_path: Path) -> None:
    """The `.201` condition: the caller's CWD is not the child's business.

    Before this fix the child's ``$PWD`` was whatever the caller happened to be
    standing in -- under cron, ``/root``.
    """
    _workspace, caller_cwd, probe_log = _run_with_caller_cwd(tmp_path)

    child_pwd = Path(_recorded(probe_log, "CHILD_PWD")).resolve()

    assert child_pwd != caller_cwd.resolve(), (
        "the provider co-install inherited the caller's working directory. "
        "Under the .201 root cron that directory is /root (0700), and the "
        "privilege drop hands it to a user who cannot enter it -- so uv's "
        "config search dies on 'failed to open file /root/uv.toml: Permission "
        "denied' and all three venv surfaces report DID_NOT_MOVE (OMN-17800)."
    )


def test_the_co_install_cannot_see_configuration_beside_the_caller(
    tmp_path: Path,
) -> None:
    """The mechanism, not the symptom: what uv would DISCOVER from the child's CWD.

    A fix that merely moved the child somewhere else would still be wrong if
    that somewhere else sat under the caller. This asserts the property that
    actually failed on the host.
    """
    _workspace, _caller_cwd, probe_log = _run_with_caller_cwd(tmp_path)

    discovered = _recorded(probe_log, "CHILD_UV_TOML")

    assert discovered == "NONE", (
        f"walking up from the co-install's working directory reaches "
        f"{discovered}, which belongs to the CALLER, not the workspace. That is "
        "exactly how the .201 cron's uv reached /root/uv.toml (OMN-17800)."
    )


def test_the_co_install_runs_inside_the_workspace_it_is_reconciling(
    tmp_path: Path,
) -> None:
    """Positive form: the directory is chosen, and it is the one being reconciled.

    ``$OMNI_HOME`` rather than the infra project: the co-install is a
    ``uv pip install`` and would newly discover ``omnibase_infra/pyproject.toml``'s
    ``[tool.uv] override-dependencies`` from inside the project, which is the
    layer beneath that this install exists NOT to re-resolve (its own ``--no-deps``
    doctrine). The workspace root carries no uv configuration on any host, so
    the fix changes where the child stands without changing what it installs.
    """
    workspace, _caller_cwd, probe_log = _run_with_caller_cwd(tmp_path)

    child_pwd = Path(_recorded(probe_log, "CHILD_PWD")).resolve()

    assert child_pwd == workspace.root.resolve(), (
        f"expected the co-install to run in $OMNI_HOME ({workspace.root}), "
        f"got {child_pwd}"
    )


# --------------------------------------------------------------------------- #
# Defect 1 -- the gate (CLAUDE.md rule 5: a check that is not a gate is advisory)
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(_GATE), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_rejects_a_privilege_dropped_package_op_with_no_working_directory(
    tmp_path: Path,
) -> None:
    """Mutation proof on the real shipped file.

    Deleting the working directory the fix introduces must fail the build.
    Otherwise the next person to "tidy" that invocation -- as happened twice
    already on this same line, OMN-17383 then OMN-17800 -- reintroduces the
    outage with nothing to stop them.
    """
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    source = _VENV_SCRIPT.read_text(encoding="utf-8")
    mutated = source.replace(
        'if ! (cd "$OMNI_HOME" && as_owner env OMNIMARKET_REF="$head"',
        'if ! as_owner env OMNIMARKET_REF="$head"',
    )
    assert mutated != source, "the mutation target moved; update this test"
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        mutated, encoding="utf-8"
    )

    result = _gate(scratch)

    assert result.returncode == 1
    assert "without setting a working directory" in result.stderr


def test_the_gate_is_wired_into_precommit_and_ci() -> None:
    """The new rule rides the hook and CI step OMN-17335 already wired."""
    precommit = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check-reconciler-privilege" in precommit

    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/check_reconciler_privilege.py" in ci


# --------------------------------------------------------------------------- #
# Defect 2 -- the receipt must be machine-readable
# --------------------------------------------------------------------------- #
def _two_surface_run(ws: Workspace) -> subprocess.CompletedProcess[str]:
    """A run that reports MORE THAN ONE surface, so a separator is emitted.

    The pre-existing receipt test builds one clone and therefore one surface,
    where the separator variable is never used. Two clones is the smallest
    workspace that exercises it.
    """
    _clone, core_seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(core_seed, "core-moved")
    _make_clone(ws.root, "omnibase_spi")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)
    return _run(ws)


def test_the_receipt_is_parseable_json_with_more_than_one_surface(
    tmp_path: Path,
) -> None:
    """The receipt is evidence, and evidence nothing can read is not evidence.

    Before this fix ``json.loads`` raised ``Expecting value: line 8 column 159``
    on the live `.201` receipt AND on the Mac's, because the separator between
    array elements was the literal three characters ``,\\n``.
    """
    ws = build_workspace(tmp_path)
    proc = _two_surface_run(ws)
    assert proc.returncode == EXIT_FAILED, proc.stdout + proc.stderr

    raw = ws.receipt.read_text(encoding="utf-8")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - the failure message
        pytest.fail(
            f"the reconcile receipt is not valid JSON ({exc}). "
            f"reconcile-host.sh writes it; every consumer has to parse it. "
            f"Raw:\n{raw}"
        )

    assert len(receipt["surfaces"]) > 1, (
        "this test is only meaningful with a separator to get wrong; "
        "the workspace produced one surface"
    )


def test_the_receipt_carries_no_literal_backslash_escape(tmp_path: Path) -> None:
    """Name the exact byte sequence, so a future rewrite cannot reintroduce it.

    ``sep=",\\n"`` in bash is a comma, a backslash and an ``n`` -- not a newline.
    ``$',\\n'`` is the form that interprets it. A JSON-parseability assertion
    alone would pass on a receipt that had merely stopped separating at all.
    """
    ws = build_workspace(tmp_path)
    _two_surface_run(ws)

    raw = ws.receipt.read_text(encoding="utf-8")
    assert "\\n" not in raw, (
        "the receipt contains a literal backslash-n. bash does not interpret "
        "escapes in a plain double-quoted assignment, nor printf in a %s "
        "ARGUMENT -- use $',\\n' (OMN-17800)."
    )
