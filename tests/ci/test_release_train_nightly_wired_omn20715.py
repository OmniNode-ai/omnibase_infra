# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The release train's decision-surface tests read as wired by the audit (OMN-20715).

omnibase_infra#4708 moved omnimarket's policy row to ``mode: cut`` and added
``tests/scripts/test_release_train_nightly.py``. The audit that judges whether a
ruling's mechanism is wired found no workflow job running the file and no
pre-commit hook naming it, because the change-aware selector does not pick
``tests/scripts/`` for a ``config/`` or ``scripts/`` change.

The recording is that audit's own ``wiring_for`` verdict on the file, taken over
a clone at the base revision and over the tree of this change. The tree tests
below check the two files the verdict is read from still name the path, so the
recording cannot outlive the wiring it describes.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_TEST_PATH = "tests/scripts/test_release_train_nightly.py"


@pytest.mark.live_contact(
    "tests/ci/fixtures/release_train_nightly_wiring_omn20715.json"
)
def test_the_audit_reads_the_file_unwired_at_the_base_and_wired_at_the_head(
    recorded_response: dict[str, object],
) -> None:
    wiring = cast("dict[str, dict[str, object]]", recorded_response["wiring"])
    base = wiring["base_0ba1a61d3"]
    head = wiring["head"]
    assert base["wired"] is False
    assert "no pre-commit hook references it" in str(base["detail"])
    assert head["wired"] is True
    assert (
        head["detail"] == "required workflow context and pre-commit hook reference it"
    )


def test_the_ci_workflow_and_the_hook_config_still_name_the_file() -> None:
    for relative in (".github/workflows/ci.yml", ".pre-commit-config.yaml"):
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert _TEST_PATH in text, f"{relative} no longer names {_TEST_PATH}"
