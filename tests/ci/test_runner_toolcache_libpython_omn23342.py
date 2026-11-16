# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image keeps libpython3.12 on the loader path (OMN-23342).

The Actions tool-cache CPython builds (``_work/_tool/Python/<ver>/x64``) link a
shared ``libpython3.X.so.1.0`` and carry no RPATH, so the interpreter starts
only where the system loader (or ``setup-python``'s ``LD_LIBRARY_PATH``) can
find the library. Until 2026-10-21 the image's deadsnakes ``python3.12`` package
put ``libpython3.12.so.1.0`` on the loader path. OMN-17427 replaced that package
with a uv-managed CPython under ``/opt/uv-python``, which left the library only
in that private directory. Measured on 2026-11-09, ``ldd`` of the tool-cache
``python3.12`` on hp053-ci-runner-1 to -5 read ``libpython3.12.so.1.0 => not
found``, and any step that ran it, or a venv built from it, without the
variable died with ``error while loading shared libraries``.

The Dockerfile now writes an ``ld.so.conf.d`` entry for the uv-managed CPython's
lib directory and runs ``ldconfig`` in the same layer. These tests read the
Dockerfile layer that installs the interpreter and run its registration
commands against a fixture, and each fails on the Dockerfile as it stood before
this change. The image identity lock moves with it, so the fleet drift check
sees a new image.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
LOCK_FILE = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"

BASH = shutil.which("bash")


def _python_layer() -> str:
    """The Dockerfile RUN instruction that installs the uv-managed CPython."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(
        r"^RUN export UV_PYTHON_INSTALL_DIR=/opt/uv-python .*?(?=^\S)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "the uv-managed CPython install layer moved"
    return match.group(0)


def test_python_layer_registers_the_uv_cpython_lib_dir_with_the_loader() -> None:
    layer = _python_layer()
    assert "/etc/ld.so.conf.d/" in layer, (
        "the uv-managed CPython's lib directory must be written to ld.so.conf.d, "
        "or the tool-cache python3.12 cannot find libpython3.12.so.1.0"
    )
    assert re.search(r"&&\s+ldconfig\b", layer), (
        "ldconfig must run in the same layer so the cache holds the library"
    )
    assert layer.index("/etc/ld.so.conf.d/") < layer.index("ldconfig"), (
        "the conf entry must exist before ldconfig reads it"
    )


def test_registration_writes_the_directory_that_holds_libpython(
    tmp_path: Path,
) -> None:
    """Run the layer's own registration line against a fixture install tree."""
    assert BASH is not None
    line = next(
        raw.strip().removeprefix("&& ").removesuffix(" \\")
        for raw in _python_layer().splitlines()
        if "/etc/ld.so.conf.d/" in raw and raw.strip().startswith("&&")
    )
    install = tmp_path / "cpython-3.12.10-linux-x86_64-gnu"
    (install / "bin").mkdir(parents=True)
    (install / "lib").mkdir()
    (install / "lib" / "libpython3.12.so.1.0").write_bytes(b"\x7fELF")
    conf_dir = tmp_path / "ld.so.conf.d"
    conf_dir.mkdir()
    line = line.replace("/etc/ld.so.conf.d", str(conf_dir))

    result = subprocess.run(
        [BASH, "-c", f'py_bin="{install / "bin"}"; {line}'],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    confs = sorted(conf_dir.glob("*.conf"))
    assert len(confs) == 1, "exactly one loader entry is written"
    registered = confs[0].read_text(encoding="utf-8").strip()
    assert registered == str(install / "lib")
    assert (Path(registered) / "libpython3.12.so.1.0").exists(), (
        "the registered directory must hold the shared libpython"
    )


def test_image_version_moved_past_thirteen() -> None:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    assert data["image_version"] >= 14, (
        "the runner image content changed; bump image_version so the fleet "
        "drift check sees a new image"
    )
