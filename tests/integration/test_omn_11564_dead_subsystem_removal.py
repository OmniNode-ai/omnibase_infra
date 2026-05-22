# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for OMN-11564 dead subsystem removal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.helpers import util_kafka

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_duplication_sweep_module() -> object:
    module_path = REPO_ROOT / "scripts" / "ci" / "run_duplication_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "run_duplication_sweep_omn_11564",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_omn_11564_unwired_subsystems_stay_removed() -> None:
    assert (
        importlib.util.find_spec(
            "omnibase_infra.services.session_registry.graph_projector"
        )
        is None
    )
    assert (
        importlib.util.find_spec("omnibase_infra.services.session_registry.store")
        is None
    )
    assert (
        importlib.util.find_spec(
            "omnibase_infra.plugins.examples.plugin_json_normalizer_error_handling"
        )
        is None
    )

    assert not hasattr(util_kafka, "normalize_ipv6_bootstrap_server")

    duplication_sweep = _load_duplication_sweep_module()
    assert duplication_sweep.ALL_CHECKS == ["D1", "D2"]
    assert not hasattr(duplication_sweep, "check_d3_migration_conflicts")
    assert not hasattr(duplication_sweep, "check_d4_model_collisions")
