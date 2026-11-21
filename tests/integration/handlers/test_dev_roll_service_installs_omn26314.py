# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev roll applies one service's registered installs through the real store and handler (OMN-26314).

The roll step starts a replacement runtime without the compose wait, applies that service's
registered installs into the new container, then issues the unchanged wait. This runs the
real record store, the real branch-install handler and the real entry against a fake Docker:
only the named service's files go in, nothing but that container restarts, a second apply
finds the files in place and restarts nothing, the shared records and status file are never
written, and a prod or stability compose project is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallRefusedError,
    DevLaneBranchInstallStore,
)
from omnibase_infra.models.model_dev_lane_branch_install_receipt import (
    ModelDevLaneBranchInstallReceipt,
)
from omnibase_infra.nodes.node_dev_roll_events_effect import (
    runtime_dev_service_installs as entry,
)
from tests.unit.handlers.test_handler_dev_lane_branch_install_omn17427 import (
    BRANCH_COPY,
    DEV_COPY,
    NOW,
    PROJECT,
    SERVICE,
    WRITER,
    _declare,
    _Lane,
    _sha,
)

pytestmark = pytest.mark.integration

OTHER = "omninode-runtime-effects"


def _apply(
    store_dir: Path, lane: _Lane, project: str = PROJECT
) -> ModelDevLaneBranchInstallReceipt:
    return entry.apply_service_installs(
        store_dir,
        project,
        SERVICE,
        run=lane,
        clock=lambda: NOW,
        sleep=lambda _s: None,
    )


def test_only_the_rolled_services_install_goes_in_and_a_second_apply_is_quiet(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    store = DevLaneBranchInstallStore(state)
    _declare(tmp_path, store)
    lane = _Lane({WRITER: DEV_COPY})
    status_before = store.status_path.exists()
    records_before = store.records_path.read_bytes()

    first = _apply(state, lane)

    assert first.ok
    assert lane.files[WRITER] == BRANCH_COPY
    assert _sha(lane.files[WRITER]) == _sha(BRANCH_COPY)
    assert len(lane.restarts) == 1
    second = _apply(state, lane)
    assert second.ok
    assert len(lane.copies) == 1, "a file already in place is not copied again"
    assert len(lane.restarts) == 1, "and its container is not restarted again"
    assert store.records_path.read_bytes() == records_before
    assert store.status_path.exists() == status_before, (
        "the shared status is not written"
    )


def test_an_install_for_another_service_is_not_applied(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = DevLaneBranchInstallStore(state)
    _declare(tmp_path, store)
    lane = _Lane({WRITER: DEV_COPY})

    receipt = entry.apply_service_installs(
        state, PROJECT, OTHER, run=lane, clock=lambda: NOW, sleep=lambda _s: None
    )

    assert receipt.ok
    assert lane.copies == []
    assert lane.restarts == []
    assert lane.files[WRITER] == DEV_COPY


@pytest.mark.parametrize("project", ["omnibase-infra-prod", "omnibase-infra-stability"])
def test_a_project_that_is_not_the_dev_lane_is_refused(
    tmp_path: Path, project: str
) -> None:
    state = tmp_path / "state"
    store = DevLaneBranchInstallStore(state)
    _declare(tmp_path, store)
    lane = _Lane({WRITER: DEV_COPY})

    with pytest.raises(DevLaneBranchInstallRefusedError):
        _apply(state, lane, project)

    assert lane.copies == []
