# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""deploy-runtime.sh must enforce the prod promotion-lineage guard (OMN-12626).

Before OMN-12626, deploy-runtime.sh only `log_warn`ed on a dirty tree and had
no promoted-lineage check, so it could build the prod image from a dirty or
dev-only source clone. These tests assert the script:

  - defines a guard_prod_promotion_lineage function,
  - delegates to scripts/check_prod_promotion_lineage.py (single source of
    truth for the clean-tree + ancestor-of-origin/main rules),
  - hard-fails (exit 1) on guard failure rather than warning,
  - exposes a --prod flag and honors ONEX_DEPLOY_LANE=prod,
  - calls the guard from main().
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.mark.unit
def test_defines_prod_lineage_guard_function() -> None:
    text = _script_text()
    assert re.search(r"^guard_prod_promotion_lineage\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define guard_prod_promotion_lineage()"
    )


@pytest.mark.unit
def test_delegates_to_promotion_lineage_script() -> None:
    text = _script_text()
    assert "scripts/check_prod_promotion_lineage.py" in text, (
        "guard must delegate to scripts/check_prod_promotion_lineage.py"
    )


@pytest.mark.unit
def test_guard_hard_fails_not_warns() -> None:
    text = _script_text()
    # The guard function must exit 1 on failure (not merely log_warn).
    func_match = re.search(
        r"guard_prod_promotion_lineage\s*\(\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(1)
    assert "exit 1" in body, "prod lineage guard must hard-fail (exit 1) on failure"


@pytest.mark.unit
def test_exposes_prod_flag() -> None:
    text = _script_text()
    assert re.search(r"--prod\)", text), "deploy-runtime.sh must expose a --prod flag"


@pytest.mark.unit
def test_honors_onex_deploy_lane_env() -> None:
    text = _script_text()
    assert "ONEX_DEPLOY_LANE" in text, (
        "deploy-runtime.sh must honor ONEX_DEPLOY_LANE=prod"
    )


@pytest.mark.unit
def test_main_invokes_guard() -> None:
    text = _script_text()
    # The guard must be wired into the deployment flow, not just defined.
    occurrences = text.count("guard_prod_promotion_lineage")
    assert occurrences >= 2, (
        "guard_prod_promotion_lineage must be defined AND called "
        f"(found {occurrences} reference(s))"
    )
