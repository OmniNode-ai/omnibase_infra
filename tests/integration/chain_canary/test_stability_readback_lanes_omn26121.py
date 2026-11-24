# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-26121 — an authorized non-dev lane's readback blocks, overlay to env.

Drives both readback loaders from one overlay file on disk through to the env
the workflow exports, the path the chain canary takes for a lane it grades.
The lane id is synthetic: lab lane ids live in the overlay, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    LEDGER_READBACK_DSN_ENV_NAME_VAR,
    PROJECTION_READBACK_DSN_ENV_NAME_VAR,
    ledger_readback_env,
    load_lane_ledger_readback,
    load_lane_projection_readback,
    projection_readback_env,
)

_OVERLAY = (
    "lanes:\n"
    "  lane-a:\n"
    "    projection_readback:\n"
    "      dsn_env: SOME_PROJECTION_NAME\n"
    "      authorized_by: OMN-26121\n"
    "    ledger_readback:\n"
    "      dsn_env: SOME_LEDGER_NAME\n"
    "      authorized_by: OMN-26121\n"
    "  lane-b:\n"
    "    projection_readback:\n"
    "      dsn_env: SOME_PROJECTION_NAME\n"
    "    ledger_readback:\n"
    "      dsn_env: SOME_LEDGER_NAME\n"
)


@pytest.mark.integration
def test_an_authorized_lane_exports_both_declared_names(tmp_path: Path) -> None:
    overlay = tmp_path / "ci_bus_lanes.yaml"
    overlay.write_text(_OVERLAY, encoding="utf-8")

    projection = load_lane_projection_readback(overlay, "lane-a")
    ledger = load_lane_ledger_readback(overlay, "lane-a")

    assert projection_readback_env(projection) == {
        PROJECTION_READBACK_DSN_ENV_NAME_VAR: "SOME_PROJECTION_NAME"
    }
    assert ledger_readback_env(ledger) == {
        LEDGER_READBACK_DSN_ENV_NAME_VAR: "SOME_LEDGER_NAME"
    }


@pytest.mark.integration
def test_a_lane_without_authorization_is_refused_on_both_legs(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "ci_bus_lanes.yaml"
    overlay.write_text(_OVERLAY, encoding="utf-8")

    with pytest.raises(ValueError, match="may not declare"):
        load_lane_projection_readback(overlay, "lane-b")
    with pytest.raises(ValueError, match="may not declare"):
        load_lane_ledger_readback(overlay, "lane-b")
