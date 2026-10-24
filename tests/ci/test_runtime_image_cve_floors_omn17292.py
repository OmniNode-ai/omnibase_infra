# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the runtime image clears the filelock and undici HIGH CVEs.

The Trivy scan in build-workspace-candidate-runtime.yml failed on dev head
6018e9df5 (run 40115303998) with two HIGH findings, both with a fixed release:

- filelock 3.32.3 (CVE-2026-112940, fixed 4.2.1) in ``/app/.venv``. filelock is
  not in the ``--no-dev`` lock set; the torch install pulls it from the PyTorch
  CPU index, whose newest mirrored filelock is 3.32.3.
- undici 6.28.1 (CVE-2026-109869, fixed 6.29.1) bundled under npm's own
  ``node_modules``, which the npm bundle patch loop overwrote at 6.28.1.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "docker" / "Dockerfile.runtime"

_FILELOCK_FIXED = (4, 2, 1)
_UNDICI_FIXED = (6, 29, 1)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _builder_stage() -> str:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    return text.split("AS runtime", 1)[0]


def test_filelock_floor_installed_after_torch_from_pytorch_index() -> None:
    builder = _builder_stage()
    torch_at = builder.find(
        "pip install torch --index-url https://download.pytorch.org"
    )
    assert torch_at != -1, "the torch install from the PyTorch CPU index moved"
    floors = [
        (match.start(), _version(match.group(1)))
        for match in re.finditer(r'pip install "filelock>=([0-9.]+)"', builder)
    ]
    assert floors, "no filelock upgrade in the builder stage"
    assert any(at > torch_at and floor >= _FILELOCK_FIXED for at, floor in floors), (
        f"filelock must be upgraded to >={_FILELOCK_FIXED} after the torch install; "
        f"found {floors}"
    )


def test_npm_bundled_undici_patched_to_fixed_release() -> None:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    specs = re.findall(r"\bundici@([0-9.]+)\b", text)
    assert specs, "the npm bundle patch loop no longer patches undici"
    for spec in specs:
        version = _version(spec)
        assert version[0] == _UNDICI_FIXED[0], (
            f"undici@{spec} leaves node-gyp's ^6 range"
        )
        assert version >= _UNDICI_FIXED, f"undici@{spec} is below {_UNDICI_FIXED}"
