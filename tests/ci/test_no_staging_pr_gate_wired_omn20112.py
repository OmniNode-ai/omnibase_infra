# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The no-staging pull-request gate test reads as wired by the audit (OMN-20112).

``tests/ci/test_no_staging_pr_gate_omn20112.py`` existed with no workflow job
running it and no pre-commit hook naming it. The audit that judges whether a
ruling's mechanism is wired found it unwired.

The recording is that audit's own ``wiring_for`` verdict on the file, taken over
a clone at the base revision and over the tree of this change. The tree test
below checks the two files the verdict is read from still name the path, so the
recording cannot outlive the wiring it describes.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_TEST_PATH = "tests/ci/test_no_staging_pr_gate_omn20112.py"


@pytest.mark.live_contact("tests/ci/fixtures/no_staging_pr_gate_wiring_omn20112.json")
def test_the_audit_reads_the_file_unwired_at_the_base_and_wired_at_the_head(
    recorded_response: dict[str, object],
) -> None:
    wiring = cast("dict[str, dict[str, object]]", recorded_response["wiring"])
    base = wiring["base_eed16cd5e"]
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
