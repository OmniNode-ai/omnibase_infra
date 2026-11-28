# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contracts revision guard against this checkout's real runtime contracts (OMN-25856).

The stamp is written by the Dockerfile's own command (``find | sort | xargs
sha256sum``, run in ``contracts/runtime``), not by a test helper, and the
guard is entered through the entrypoint's ``run_preflight`` path. Known good:
the tree the stamp was taken from boots. Known bad: the same tree with
``runtime_config.yaml`` changed to another revision's content is refused.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_contracts_revision_guard as guard

pytestmark = [pytest.mark.integration]

_REAL_RUNTIME = Path(__file__).resolve().parents[3] / "contracts" / "runtime"
_STAMP_CMD = "find . -type f | LC_ALL=C sort | xargs -r sha256sum"


def _image_stamp(runtime: Path, stamp: Path) -> None:
    out = subprocess.run(
        ["bash", "-c", _STAMP_CMD],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    stamp.write_text(out)


def _env(runtime: Path, stamp: Path) -> dict[str, str]:
    return {
        "ONEX_RUNTIME_CONTRACTS_DIR": str(runtime),
        "ONEX_CONTRACTS_STAMP_PATH": str(stamp),
    }


def test_real_contracts_match_their_own_dockerfile_stamp(tmp_path: Path) -> None:
    assert (_REAL_RUNTIME / "runtime_config.yaml").is_file()
    stamp = tmp_path / "contracts_runtime.sha256"
    _image_stamp(_REAL_RUNTIME, stamp)

    assert guard.refuse_mismatched_contracts(_env(_REAL_RUNTIME, stamp)) == 0


def test_real_contracts_from_another_revision_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stamp = tmp_path / "contracts_runtime.sha256"
    _image_stamp(_REAL_RUNTIME, stamp)
    mounted = tmp_path / "host_runtime"
    shutil.copytree(_REAL_RUNTIME, mounted)
    config = mounted / "runtime_config.yaml"
    config.write_text(config.read_text() + "\n# another revision\n")

    assert guard.refuse_mismatched_contracts(_env(mounted, stamp)) == 1
    err = capsys.readouterr().err
    assert "CONTRACTS_REVISION_MISMATCH" in err
    assert "runtime_config.yaml" in err
