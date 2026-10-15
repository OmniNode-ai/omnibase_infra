# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The release train's version bump leaves a lockfile ``uv`` accepts (OMN-18595).

omnimarket#3888 (0.4.314) was opened by the train with ``pyproject.toml`` at
0.4.314 and ``uv.lock`` still at 0.4.313, and ``uv sync --locked`` refused that
checkout in every check that installs the project.

The recording is ``uv lock --check`` run over the release head's own files, and
over the same ``pyproject.toml`` beside the ``uv.lock`` that ``uv`` itself wrote
for 0.4.314 on omnimarket dev. The replay builds the release head's root
package block, bumps it with the train's code, and requires the bytes ``uv``
wrote.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]


def _release_train() -> Any:
    name = "release_train_bump_lockfile_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / "scripts" / "ci" / "release_train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.live_contact("tests/ci/fixtures/release_train_bump_lockfile_omn18595.json")
def test_uv_refused_the_release_head_and_accepted_the_lock_it_wrote_itself(
    recorded_response: dict[str, object],
) -> None:
    refused = cast("dict[str, object]", recorded_response["uv_lock_check_release_head"])
    accepted = cast(
        "dict[str, object]", recorded_response["uv_lock_check_with_uv_written_lock"]
    )
    assert refused["exit_code"] == 1
    assert "needs to be updated" in str(refused["stderr"])
    assert accepted["exit_code"] == 0


@pytest.mark.live_contact("tests/ci/fixtures/release_train_bump_lockfile_omn18595.json")
def test_the_trains_bump_of_the_release_heads_lock_equals_the_lock_uv_wrote(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    module = _release_train()
    head_block = str(recorded_response["release_head_root_block"])
    uv_block = str(recorded_response["uv_written_root_block"])
    assert head_block != uv_block, "the recording must differ before the bump"

    (tmp_path / "pyproject.toml").write_text(
        str(recorded_response["release_head_pyproject_project_table"]),
        encoding="utf-8",
    )
    neighbour = '[[package]]\nname = "aiokafka"\nversion = "0.4.313"\n'
    (tmp_path / "uv.lock").write_text(
        f"version = 1\n\n{neighbour}\n{head_block}", encoding="utf-8"
    )
    written = module.bump_declared_version(tmp_path, "0.4.314")
    assert written == ["pyproject.toml", "uv.lock"]
    assert (tmp_path / "uv.lock").read_text(
        encoding="utf-8"
    ) == f"version = 1\n\n{neighbour}\n{uv_block}"
