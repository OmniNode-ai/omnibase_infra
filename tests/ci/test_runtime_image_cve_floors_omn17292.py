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

_VERSION = r"([0-9]+(?:\.[0-9]+)*)"
_STAGE_RE = re.compile(r"^FROM\s+\S+\s+AS\s+(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_TORCH_RE = re.compile(
    r"pip install torch --index-url https://download\.pytorch\.org/", re.MULTILINE
)
_FILELOCK_RE = re.compile(r"""pip install\s+["']?filelock\b([^"'\s]*)["']?""")
_FILELOCK_FLOOR_RE = re.compile(rf"^>={_VERSION}$")
_PATCH_LOOP_RE = re.compile(r"^\s*for spec in ([^;\n]+);\s*do\b", re.MULTILINE)
_UNDICI_SPEC_RE = re.compile(rf"^undici@{_VERSION}$")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _floor(spec: str) -> tuple[int, ...] | None:
    match = _FILELOCK_FLOOR_RE.match(spec)
    return _version(match.group(1)) if match else None


def _stages() -> dict[str, str]:
    """Map each named build stage to its instructions, comment lines removed."""
    lines = _DOCKERFILE.read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    starts = list(_STAGE_RE.finditer(text))
    names = [match.group(1) for match in starts]
    assert len(names) == len(set(names)), f"duplicate stage names: {names}"
    ends = [match.start() for match in starts[1:]] + [len(text)]
    return {
        match.group(1): text[match.start() : end]
        for match, end in zip(starts, ends, strict=True)
    }


def test_stages_parsed_from_from_lines() -> None:
    stages = _stages()
    assert {"builder", "runtime"} <= stages.keys(), sorted(stages)
    venv_copy = "COPY --from=builder --chown=omniinfra:omniinfra /app/.venv /app/.venv"
    assert venv_copy in stages["runtime"], (
        "the runtime stage no longer ships the builder's /app/.venv"
    )


def test_filelock_floor_installed_after_torch_from_pytorch_index() -> None:
    stages = _stages()
    builder = stages["builder"]
    torch_installs = [match.end() for match in _TORCH_RE.finditer(builder)]
    assert torch_installs, "the torch install from the PyTorch CPU index moved"
    installs = [
        (match.start(), match.group(1)) for match in _FILELOCK_RE.finditer(builder)
    ]
    assert installs, "no filelock upgrade in the builder stage"
    last_torch = max(torch_installs)
    fixed = [
        at
        for at, spec in installs
        if at > last_torch and (_floor(spec) or ()) >= _FILELOCK_FIXED
    ]
    assert fixed, (
        f"filelock must be upgraded to >={_FILELOCK_FIXED} after the torch install; "
        f"found {installs}"
    )
    later = [spec for at, spec in installs if at > min(fixed)]
    assert all((_floor(spec) or ()) >= _FILELOCK_FIXED for spec in later), (
        f"a later builder install re-pins filelock below the floor: {later}"
    )
    for name, body in stages.items():
        if name != "builder":
            assert not _FILELOCK_RE.search(body), f"stage {name} reinstalls filelock"


def test_npm_bundled_undici_patched_to_fixed_release() -> None:
    loops = [
        match.group(1).split()
        for match in _PATCH_LOOP_RE.finditer(_stages()["runtime"])
        if any(spec.startswith("undici@") for spec in match.group(1).split())
    ]
    assert len(loops) == 1, "the npm bundle patch loop no longer patches undici"
    undici = [spec for spec in loops[0] if spec.startswith("undici@")]
    assert len(undici) == 1, f"the patch loop names undici more than once: {undici}"
    match = _UNDICI_SPEC_RE.match(undici[0])
    assert match, f"{undici[0]} is not an exact undici release"
    version = _version(match.group(1))
    assert version[0] == _UNDICI_FIXED[0], f"{undici[0]} leaves node-gyp's ^6 range"
    assert version >= _UNDICI_FIXED, f"{undici[0]} is below {_UNDICI_FIXED}"
