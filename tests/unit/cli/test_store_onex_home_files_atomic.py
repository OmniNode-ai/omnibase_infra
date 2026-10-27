# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20827: the two ``~/.onex`` files are replaced atomically.

A direct ``write_text`` truncates the target before it writes it, so a crash
in the middle leaves a short or empty file. An empty ``credentials.json`` is
refused by the next ``load_secrets`` as not valid JSON, and that refusal is
exactly what stops a rerun of ``lane-login`` from repairing the machine. Each
file is now written to a temporary file in the same directory, created at
0600, flushed, and renamed over the target, and the directory entry is
flushed after the rename.

A crash is injected by making one step of the write raise: the flush before
the rename, or the rename itself. Between the two files, a crash after the
config write and before the secrets write leaves a config naming a secret the
0600 file does not hold yet (first login) or still holding the previous one
(rotation). Both are states a rerun of ``lane-login`` repairs.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli import store_onex_home_files
from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles

pytestmark = pytest.mark.unit

_LANE = "local"
_PASSWORD_REF = StoreLaneCredential.password_ref(_LANE)


class _InjectedCrashError(RuntimeError):
    """Stands in for the process dying at the patched step."""


@pytest.fixture
def onex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``lane-login`` at a real directory instead of the operator's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home / ".onex"


def _lane_login(username: str, password: str) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            _LANE,
            "--sasl-username",
            username,
            "--sasl-password-stdin",
        ],
        input=f"{password}\n",
    )
    assert result.exit_code == 0, result.output


def _leftovers(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.name not in {"config.yaml", "credentials.json"}
    )


def _crash_at(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """Make one OS step of the store's write raise, as a crash there would."""

    def crash(*args: object, **kwargs: object) -> None:
        raise _InjectedCrashError(step)

    monkeypatch.setattr(store_onex_home_files.os, step, crash)


# --- AC1: a temporary file in the same directory, renamed over the target ----


@pytest.mark.parametrize("target", ["config", "secrets"])
def test_atomic_replace_writes_a_temporary_file_and_renames_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    files = StoreOnexHomeFiles(tmp_path)
    renames: list[tuple[Path, Path]] = []
    flushed_dirs: list[int] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_replace(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        source = Path(src)
        assert stat.S_IMODE(source.stat().st_mode) == 0o600
        renames.append((source, Path(dst)))
        real_replace(src, dst)

    def recording_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            flushed_dirs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(store_onex_home_files.os, "replace", recording_replace)
    monkeypatch.setattr(store_onex_home_files.os, "fsync", recording_fsync)

    if target == "config":
        files.write_config({"lanes": {}})
        expected = files.config_path
    else:
        files.write_secrets({_PASSWORD_REF: "value"})
        expected = files.credentials_path

    ((source, destination),) = renames
    assert destination == expected
    assert source.parent == expected.parent
    assert source != expected
    assert flushed_dirs, "the directory entry was not flushed after the rename"
    assert stat.S_IMODE(expected.stat().st_mode) == 0o600


@pytest.mark.parametrize("target", ["config", "secrets"])
def test_atomic_replace_never_writes_the_target_path_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Opening the target itself for writing is the truncate-then-write this removes."""
    files = StoreOnexHomeFiles(tmp_path)
    files.write_config({"lanes": {}})
    files.write_secrets({_PASSWORD_REF: "old"})
    targets = {str(files.config_path), str(files.credentials_path)}
    real_open = os.open

    def guarded_open(path: str | os.PathLike[str], flags: int, *args: int) -> int:
        if str(path) in targets and flags & (os.O_WRONLY | os.O_RDWR):
            raise AssertionError(f"direct write to {path}")
        return real_open(path, flags, *args)

    monkeypatch.setattr(store_onex_home_files.os, "open", guarded_open)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, *a, **k: pytest.fail(f"write_text on {self}"),
    )

    if target == "config":
        files.write_config({"lanes": {"dev": {}}})
    else:
        files.write_secrets({_PASSWORD_REF: "new"})


# --- AC2: a crash leaves the previous file byte-identical, and a rerun repairs -


@pytest.mark.parametrize("step", ["fsync", "replace"])
@pytest.mark.parametrize("target", ["config", "secrets"])
def test_a_crash_inside_one_write_leaves_the_previous_file_byte_identical(
    onex_home: Path, monkeypatch: pytest.MonkeyPatch, step: str, target: str
) -> None:
    _lane_login("local-developer", "first-secret")
    files = StoreOnexHomeFiles(onex_home)
    path = files.config_path if target == "config" else files.credentials_path
    before = path.read_bytes()

    write: Callable[[], None] = (
        (lambda: files.write_config({"lanes": {}}))
        if target == "config"
        else (lambda: files.write_secrets({_PASSWORD_REF: "second-secret"}))
    )
    with monkeypatch.context() as patch:
        _crash_at(patch, step)
        with pytest.raises(_InjectedCrashError):
            write()

    assert path.read_bytes() == before
    assert _leftovers(onex_home) == []


# --- AC3: the temporary file is 0600 from creation, and never left behind -----


@pytest.mark.parametrize("target", ["config", "secrets"])
def test_the_temporary_file_is_never_wider_than_0600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    files = StoreOnexHomeFiles(tmp_path)
    modes: list[int] = []
    real_write = os.write
    real_umask = os.umask(0)  # the widest umask a caller could run under

    def recording_write(fd: int, data: bytes) -> int:
        modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return real_write(fd, data)

    try:
        monkeypatch.setattr(store_onex_home_files.os, "write", recording_write)
        if target == "config":
            files.write_config({"lanes": {}})
        else:
            files.write_secrets({_PASSWORD_REF: "value"})
    finally:
        os.umask(real_umask)

    assert modes, "the store did not write through a file descriptor"
    assert all(mode == 0o600 for mode in modes), [f"{m:04o}" for m in modes]
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("step", ["write", "fsync", "replace"])
def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    files = StoreOnexHomeFiles(tmp_path)
    _crash_at(monkeypatch, step)

    with pytest.raises(_InjectedCrashError):
        files.write_secrets({_PASSWORD_REF: "value"})

    assert _leftovers(tmp_path) == []
    assert not files.credentials_path.exists()
