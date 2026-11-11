# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool driver reads its judge command from an overlay root, never from the frozen pool overlay (OMN-23715).

The command names ``node_lab_proof_judge_compute``; its home is the overlays repository
(``overlays/prepr_runtime_pool/overlay.yaml``), found through ``ONEX_SKILL_OVERLAY_ROOTS``. The overlays
repository's ``tests/test_prepr_runtime_pool_judge_overlay.py`` holds the other side of the key seam.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.models.prepr_pool import (
    ModelPreprPoolJudgeOverlay,
    ModelPreprPoolOverlay,
)
from tests.helpers.prepr_pool_overlay import POOL_CONFIG_ROOT

pytestmark = pytest.mark.unit


def _overlay(root: Path, command: str = "judge", **extra: object) -> Path:
    path = root / "prepr_runtime_pool" / "overlay.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "lab_proof_judge_command": command, **extra}
        ),
        encoding="utf-8",
    )
    return root


def _environ(*roots: Path) -> dict[str, str]:
    return {"ONEX_SKILL_OVERLAY_ROOTS": ":".join(str(r) for r in roots)}


def test_the_first_root_holding_the_overlay_wins(tmp_path: Path) -> None:
    first = _overlay(tmp_path / "a", "from-a")
    second = _overlay(tmp_path / "b", "from-b")
    command = ModelPreprPoolJudgeOverlay.command(_environ(first, second), tmp_path)
    assert command == "from-a"


def test_a_root_without_the_overlay_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    held = _overlay(tmp_path / "held", "from-held")
    command = ModelPreprPoolJudgeOverlay.command(
        _environ(tmp_path / "empty", held), tmp_path
    )
    assert command == "from-held"


def test_failing_control_no_overlay_anywhere_is_an_empty_command(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "workspace_config"
    assert ModelPreprPoolJudgeOverlay.command({}, config_root) == ""


def test_the_overlays_clone_beside_the_config_root_is_read_when_no_root_is_named(
    tmp_path: Path,
) -> None:
    _overlay(tmp_path / "overlays" / "overlays", "from-beside")
    command = ModelPreprPoolJudgeOverlay.command({}, tmp_path / "workspace_config")
    assert command == "from-beside"


def test_a_named_root_comes_before_the_clone_beside_the_config_root(
    tmp_path: Path,
) -> None:
    _overlay(tmp_path / "overlays" / "overlays", "from-beside")
    named = _overlay(tmp_path / "named", "from-named")
    command = ModelPreprPoolJudgeOverlay.command(
        _environ(named), tmp_path / "workspace_config"
    )
    assert command == "from-named"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": ""},
        {"command": "judge", "unknown_key": 1},
    ],
    ids=["empty-command", "unknown-key"],
)
def test_failing_control_an_overlay_that_does_not_validate_is_refused(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    root = _overlay(tmp_path, **kwargs)
    with pytest.raises(ValidationError):
        ModelPreprPoolJudgeOverlay.command(_environ(root), tmp_path)


def test_failing_control_a_schema_version_other_than_one_is_refused(
    tmp_path: Path,
) -> None:
    root = _overlay(tmp_path)
    path = root / "prepr_runtime_pool" / "overlay.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "schema_version: 1", "schema_version: 2"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        ModelPreprPoolJudgeOverlay.command(_environ(root), tmp_path)


def _config_root(tmp_path: Path, extra: str = "") -> Path:
    root = tmp_path / "config_root"
    shutil.copytree(POOL_CONFIG_ROOT, root)
    overlay = root / "config/onex/overlays/prepr_runtime_pool/overlay.yaml"
    overlay.write_text(overlay.read_text(encoding="utf-8") + extra, encoding="utf-8")
    ModelPreprPoolOverlay.read.cache_clear()
    return root


def test_the_pool_overlay_carries_the_command_read_from_the_overlay_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(_overlay(tmp_path / "o", "cmd")))
    assert (
        ModelPreprPoolOverlay.read(_config_root(tmp_path)).lab_proof_judge_command
        == "cmd"
    )


def test_failing_control_the_frozen_pool_overlay_carrying_the_key_is_refused(
    tmp_path: Path,
) -> None:
    root = _config_root(tmp_path, "lab_proof_judge_command: judge\n")
    with pytest.raises(ValueError, match="belongs in the overlays repository"):
        ModelPreprPoolOverlay.read(root)


def test_seam_with_the_overlays_repository_the_keys_the_driver_reads_are_these() -> (
    None
):
    """The overlays repository pins the same two keys in its own list."""
    assert set(ModelPreprPoolJudgeOverlay.model_fields) == {
        "schema_version",
        "lab_proof_judge_command",
    }
