# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: the stability follow declares, and admits, only the live pin's own PR.

Runs the real pin file round trip: a pin is written to a state directory with the
handler's own ``write_pin`` and read back from disk by ``follow`` and ``admit_open_pr``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_lane_code_ref_pin import (
    admit_open_pr,
    declared_open_pr,
    follow,
    read_active_pin,
    write_pin,
)
from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin

pytestmark = pytest.mark.integration

TAG = "dev-20261115134817-ebb006955dd0"
HEAD = "6013569daf5e0328a733858f9f7c49d639f90a87"


def _pin(state_dir: Path, *, expires_in: timedelta) -> ModelDevLaneCodeRefPin:
    now = datetime.now(UTC)
    pin = ModelDevLaneCodeRefPin(
        branch="omn-24562-stability-integ-1740",
        head=HEAD,
        pr=6011,
        ticket="OMN-24562",
        registered_by="stab-integ2-1750",
        registered_at=now - timedelta(hours=1),
        expires_at=now + expires_in,
    )
    write_pin(state_dir, pin)
    return pin


def test_live_pin_is_followed_with_its_own_open_pr(tmp_path: Path) -> None:
    pin = _pin(tmp_path, expires_in=timedelta(hours=2))
    fields = follow("omnimarket", TAG, tmp_path)
    assert fields["git_ref"] == HEAD
    assert fields["open_pr"] == declared_open_pr(pin) == f"omnibase_infra#6011@{HEAD}"
    assert fields["sibling_tags"] == {"omnimarket": TAG}
    # Known good: the pin's own declaration is admitted on the stability lane.
    admit_open_pr(True, declared_open_pr(pin), tmp_path)


def test_foreign_open_pr_is_refused_on_stability(tmp_path: Path) -> None:
    _pin(tmp_path, expires_in=timedelta(hours=2))
    # Known bad: another pull request, and another head of the pinned one.
    with pytest.raises(ValueError, match="builds only the open pull request"):
        admit_open_pr(True, f"omnibase_infra#1@{HEAD}", tmp_path)
    with pytest.raises(ValueError, match="builds only the open pull request"):
        admit_open_pr(True, f"omnibase_infra#6011@{'b' * 40}", tmp_path)


def test_no_live_pin_admits_no_declaration_and_follows_the_tag_only(
    tmp_path: Path,
) -> None:
    pin = _pin(tmp_path, expires_in=timedelta(seconds=-1))
    assert read_active_pin(tmp_path, datetime.now(UTC)) is None
    assert follow("omnimarket", TAG, tmp_path) == {"sibling_tags": {"omnimarket": TAG}}
    with pytest.raises(ValueError, match="no pin is live"):
        admit_open_pr(True, declared_open_pr(pin), tmp_path)
    # The dev lane keeps its own rule and is not read here.
    admit_open_pr(False, declared_open_pr(pin), tmp_path)
