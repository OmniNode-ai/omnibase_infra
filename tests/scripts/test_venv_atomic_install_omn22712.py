# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The atomic in-place install, handlers/venv_atomic_install (OMN-22712).

The reconcile's additive pass used to clone the whole live dispatch venv, install
into the clone, swap it in and delete the old tree: about 57,000 creates and
57,000 removes a pass, about 45 passes a day on a lab Mac, which is what grows
fseventsd without bound. The pass now stages ONLY the packages the plan moves in
a small side directory, reads the venv back with them overlaid, and exchanges
each package directory into place with one syscall and a journal.

These tests use real ``uv`` and real venvs. The packages are wheels built here
byte by byte, so nothing needs a network or an index.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = (
    _REPO_ROOT
    / "src/omnibase_infra/handlers/venv_atomic_install/atomic_venv_install.py"
)
_TOOL = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
)

sys.path.insert(0, str(_REPO_ROOT / "scripts"))

_SPEC = importlib.util.spec_from_file_location("atomic_venv_install", _LIB)
assert _SPEC is not None and _SPEC.loader is not None
plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan)

_MODULES = 120


def _wheel(
    out: Path, name: str, version: str, marker: str, *, modules: int = 0
) -> Path:
    """A pure-python wheel with a console script, built without any tooling."""
    dist = f"{name}-{version}.dist-info"
    files: dict[str, str] = {
        f"{name}/__init__.py": f"MARKER = {marker!r}\n\ndef main():\n    print(MARKER)\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist}/entry_points.txt": f"[console_scripts]\n{name}-cli = {name}:main\n",
    }
    for i in range(modules):
        files[f"{name}/m{i}.py"] = f"VALUE = {i}\n"
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())
        record.append(f"{path},sha256={digest.rstrip(b'=').decode()},{len(text)}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    wheel = out / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel


def _uv(*args: str) -> None:
    subprocess.run(["uv", *args], check=True, capture_output=True, text=True)


class _Venv:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.wheels = root / "wheels"
        self.wheels.mkdir(parents=True)
        self.venv = root / "live"
        _uv("venv", "--relocatable", "--python", sys.executable, str(self.venv))
        self.python = str(self.venv / "bin" / "python")
        self.stage = root / "stage"
        # An old generation of the moved package, plus bystanders that must not
        # be touched: a second package and filler that stands for the other
        # ~57,000 files of the real venv.
        self.old = _wheel(self.wheels, "moved_pkg", "1.0", "old", modules=_MODULES)
        self.new = _wheel(self.wheels, "moved_pkg", "1.1", "new", modules=_MODULES)
        self.bystander = _wheel(
            self.wheels, "bystander_pkg", "2.0", "bystander", modules=_MODULES
        )
        _uv(
            "pip",
            "install",
            "--python",
            self.python,
            "--no-deps",
            str(self.old),
            str(self.bystander),
        )
        site = self.site()
        for i in range(400):
            filler = site / "filler" / f"f{i}.py"
            filler.parent.mkdir(exist_ok=True)
            filler.write_text(f"X = {i}\n", encoding="utf-8")

    def site(self) -> Path:
        out = subprocess.run(
            [
                self.python,
                "-c",
                "import sysconfig;print(sysconfig.get_paths()['purelib'])",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(out.stdout.strip())

    def run(self, command: str, *args: str) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run(
            [
                sys.executable,
                str(_TOOL),
                command,
                "--python",
                self.python,
                "--stage-dir",
                str(self.stage),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def stage_new(self) -> subprocess.CompletedProcess[str]:
        return self.run("stage", "--no-deps", "--", str(self.new))

    def snapshot(self) -> dict[str, str]:
        """Every path under the venv with a content digest (None for dirs)."""
        out: dict[str, str] = {}
        for path in sorted(self.venv.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(self.venv))
            if path.is_file() and not path.is_symlink():
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                out[rel] = "-"
        return out

    def marker(self, *, overlay: Path | None = None) -> str:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if overlay is not None:
            env["PYTHONPATH"] = str(overlay)
        out = subprocess.run(
            [self.python, "-c", "import moved_pkg;print(moved_pkg.MARKER)"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        return out.stdout.strip() or out.stderr.strip()

    def script(self) -> str:
        out = subprocess.run(
            [str(self.venv / "bin" / "moved_pkg-cli")],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or out.stderr.strip()


@pytest.fixture
def env(tmp_path: Path) -> _Venv:
    return _Venv(tmp_path)


def test_staging_writes_only_the_moved_package_and_never_the_live_venv(
    env: _Venv,
) -> None:
    """The stage holds the moved package alone; the live venv is not touched."""
    before = env.snapshot()
    live_files = sum(1 for p in env.venv.rglob("*") if p.is_file())

    result = env.stage_new()

    assert result.returncode == plan.EXIT_OK, result.stdout + result.stderr
    assert env.snapshot() == before, "staging wrote into the live venv"
    staged = [p for p in env.stage.rglob("*") if p.is_file()]
    assert staged, "nothing was staged"
    assert len(staged) < live_files, (
        f"the stage ({len(staged)} files) is not smaller than the venv ({live_files})"
    )
    names = {p.name for p in (env.stage / "site").iterdir()}
    assert "moved_pkg" in names and "moved_pkg-1.1.dist-info" in names, names
    assert not any(n.startswith("bystander_pkg") or n == "filler" for n in names), (
        f"the stage carries a package that did not move: {names}"
    )


def test_the_staged_package_is_readable_as_an_overlay_before_any_exchange(
    env: _Venv,
) -> None:
    """Every check can run against the venv plus the stage, before the commit."""
    assert env.stage_new().returncode == plan.EXIT_OK
    assert env.marker() == "old"
    assert env.marker(overlay=env.stage / "site") == "new"
    wrapped = subprocess.run(
        [str(env.stage / "python"), "-c", "import moved_pkg;print(moved_pkg.MARKER)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert wrapped.stdout.strip() == "new", "the wrapper interpreter misses the stage"


def test_the_wrapper_interpreter_writes_no_bytecode_into_the_stage(env: _Venv) -> None:
    """An import through the wrapper leaves no .pyc in the stage (file events, stale paths)."""
    assert env.stage_new().returncode == plan.EXIT_OK
    subprocess.run(
        [str(env.stage / "python"), "-c", "import moved_pkg"],
        capture_output=True,
        text=True,
        check=True,
    )
    pycs = list(env.stage.rglob("*.pyc"))
    assert not pycs, f"the wrapper wrote bytecode into the stage: {pycs}"


def test_commit_puts_the_new_generation_in_place_and_rollback_restores_it(
    env: _Venv,
) -> None:
    before = env.snapshot()
    assert env.stage_new().returncode == plan.EXIT_OK

    commit = env.run("commit")

    assert commit.returncode == plan.EXIT_OK, commit.stdout + commit.stderr
    assert env.marker() == "new"
    assert env.script() == "new", "the console script still runs the old package"
    site_entries = {p.name for p in env.site().iterdir()}
    assert "moved_pkg-1.1.dist-info" in site_entries
    assert "moved_pkg-1.0.dist-info" not in site_entries, "the old metadata survived"
    assert any(p.name.startswith("bystander_pkg") for p in env.site().iterdir()), (
        "a package that did not move was disturbed"
    )

    rolled = env.run("rollback")

    assert rolled.returncode == plan.EXIT_OK, rolled.stdout + rolled.stderr
    assert env.marker() == "old"
    assert env.script() == "old"
    assert env.snapshot() == before, "rollback did not restore the venv exactly"


def test_a_failure_part_way_through_the_commit_restores_every_path(
    env: _Venv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected break: the third put raises. The venv comes back byte for byte."""
    before = env.snapshot()
    assert env.stage_new().returncode == plan.EXIT_OK
    calls = {"n": 0}
    real_put = plan._put

    def broken_put(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("injected break")
        real_put(*args, **kwargs)

    monkeypatch.setattr(plan, "_put", broken_put)

    status = plan.commit_stage(env.python, env.stage)

    assert status == plan.EXIT_COMMIT_FAILED
    assert calls["n"] == 3, "the break did not land mid-commit"
    assert env.snapshot() == before, "a failed commit left the venv changed"
    assert env.marker() == "old"


def test_no_import_fails_while_the_commit_exchanges_the_package_in(env: _Venv) -> None:
    """OMN-19432's property survives: the package is never half there.

    A reinstall in place removes the package and writes it back file by file, so
    an import in that window fails. The commit exchanges the package directory
    in one syscall. A poller imports the last module of the package, in a fresh
    interpreter each time, for as long as the commit runs.
    """
    assert env.stage_new().returncode == plan.EXIT_OK
    failures: list[str] = []
    successes = {"n": 0}
    stop = threading.Event()
    probe = f"import moved_pkg.m{_MODULES - 1}, moved_pkg"
    env_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def poll() -> None:
        while not stop.is_set():
            out = subprocess.run(
                [env.python, "-c", probe],
                capture_output=True,
                text=True,
                check=False,
                env=env_vars,
            )
            if out.returncode == 0:
                successes["n"] += 1
            else:
                failures.append(out.stderr.strip().splitlines()[-1])

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        while successes["n"] < 3:  # the poller is provably running first
            pass
        commit = env.run("commit")
    finally:
        stop.set()
        poller.join()

    assert commit.returncode == plan.EXIT_OK, commit.stdout + commit.stderr
    assert successes["n"] >= 3
    assert not failures, f"an import failed during the exchange: {failures[:3]}"


def test_a_downgrade_is_still_refused_before_anything_is_staged(env: _Venv) -> None:
    """The planner's refusal is unchanged in staged mode."""
    _uv("pip", "install", "--python", env.python, "--no-deps", str(env.new))
    before = env.snapshot()

    result = env.run("stage", "--no-deps", "--", str(env.old))

    assert result.returncode != plan.EXIT_OK, result.stdout + result.stderr
    assert not env.stage.exists() or not any(env.stage.rglob("*.dist-info"))
    assert env.snapshot() == before


def test_staging_a_requirement_the_venv_already_satisfies_stages_nothing(
    env: _Venv,
) -> None:
    """Positive control: no moved package, no stage, no commit work."""
    result = env.run("stage", "--no-deps", "--", str(env.old))

    assert result.returncode == plan.EXIT_OK, result.stdout + result.stderr
    assert "no change" in result.stdout, result.stdout
    commit = env.run("commit")
    assert commit.returncode == plan.EXIT_OK
    assert "nothing to commit" in commit.stdout or "nothing staged" in commit.stdout


def test_the_staged_script_names_no_absolute_python(env: _Venv) -> None:
    """A script staged by `uv pip install --target` is made self-locating."""
    assert env.stage_new().returncode == plan.EXIT_OK
    text = (env.stage / "bin" / "moved_pkg-cli").read_text(encoding="utf-8")
    assert textwrap.dedent(text).splitlines()[1].endswith(
        """/'python' "$0" "$@\""""
    ) or ("'python'" in text.splitlines()[1]), text.splitlines()[:3]
    assert str(env.venv) not in text.splitlines()[1]


def test_the_poller_does_see_a_package_that_is_missing(env: _Venv) -> None:
    """Positive control: the probe the no-window test uses can fail.

    The package directory is moved aside for a moment, as an in-place reinstall
    does while it rewrites the package; the same import must then fail.
    """
    site = env.site()
    probe = f"import moved_pkg.m{_MODULES - 1}, moved_pkg"
    env_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    (site / "moved_pkg").rename(site / "moved_pkg.aside")
    try:
        gone = subprocess.run(
            [env.python, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            env=env_vars,
        )
    finally:
        (site / "moved_pkg.aside").rename(site / "moved_pkg")
    assert gone.returncode != 0, "the probe passed with the package missing"
    here = subprocess.run(
        [env.python, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=env_vars,
    )
    assert here.returncode == 0, here.stderr


def test_the_readback_sees_the_staged_package_only_through_the_wrapper(
    env: _Venv,
) -> None:
    """The readback drops PYTHONPATH on purpose; the wrapper interpreter is its way in."""
    import venv_readback

    assert env.stage_new().returncode == plan.EXIT_OK
    plain = venv_readback.probe_installed(env.python, ["moved_pkg"])
    overlaid = venv_readback.probe_installed(str(env.stage / "python"), ["moved_pkg"])
    assert plain["moved_pkg"]["version"] == "1.0", plain
    assert overlaid["moved_pkg"]["version"] == "1.1", overlaid


def test_the_typed_handler_runs_the_same_steps(tmp_path: Path) -> None:
    """HandlerVenvAtomicInstall is the typed form of the runtime's commands."""
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.handlers.venv_atomic_install.handler_venv_atomic_install import (
        HandlerVenvAtomicInstall,
    )
    from omnibase_infra.models.model_venv_atomic_install_request import (
        ModelVenvAtomicInstallRequest,
    )

    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    live, staging = tmp_path / "live", tmp_path / "staging"
    live.mkdir()
    staging.mkdir()
    (live / "which").write_text("old", encoding="utf-8")
    (staging / "which").write_text("new", encoding="utf-8")

    swapped = handler.handle(
        ModelVenvAtomicInstallRequest(command="exchange", paths=(staging, live))
    )
    assert swapped.status == "ok", swapped
    assert (live / "which").read_text(encoding="utf-8") == "new"

    missing = handler.handle(
        ModelVenvAtomicInstallRequest(
            command="exchange", paths=(staging, tmp_path / "absent")
        )
    )
    assert missing.status == "failed" and missing.exit_code != 0, missing

    no_stage = handler.handle(ModelVenvAtomicInstallRequest(command="commit"))
    assert no_stage.status == "failed", "a commit with no stage dir must not pass"

    done = handler.handle(
        ModelVenvAtomicInstallRequest(command="finalize", stage_dir=staging)
    )
    assert done.status == "ok" and not staging.exists()


_SHAPE_BASELINE = _REPO_ROOT / ".onex_ratchets/canonical_file_shape_baseline.txt"
_NODE_RUNTIME = "nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
_BASELINED_SCRIPTS = (
    "scripts/install-node-skill-package.sh",
    "scripts/reconcile-workspace-venvs.sh",
)
_NODE_SIDE_PREFIXES = (
    "src/omnibase_infra/nodes/node_venv_atomic_install/",
    "src/omnibase_infra/handlers/venv_atomic_install/",
    "src/omnibase_infra/models/model_venv_atomic_install_",
)


def test_the_baselined_scripts_call_the_node_and_the_baseline_gained_nothing() -> None:
    """AC4: the two baselined scripts keep only their call to the node (OMN-22712).

    Before the change the scripts carried the exchange themselves and named no
    node, so this reads red there. The baseline is shrink-only: no entry for the
    node, its handler or its models may appear in it.
    """
    baseline_lines = [
        line.strip()
        for line in _SHAPE_BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for script in _BASELINED_SCRIPTS:
        assert script in baseline_lines, f"{script} left the baseline"
    raised = [line for line in baseline_lines if line.startswith(_NODE_SIDE_PREFIXES)]
    assert not raised, f"baseline entries raised for the node: {raised}"

    reconcile = (_REPO_ROOT / _BASELINED_SCRIPTS[1]).read_text(encoding="utf-8")
    install = (_REPO_ROOT / _BASELINED_SCRIPTS[0]).read_text(encoding="utf-8")
    assert _NODE_RUNTIME in install, "the install script does not call the node"
    assert "ONEX_ATOMIC_INSTALL" in reconcile, "the reconcile does not take the path"
