# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Paths come from the caller, the overlay or the catalog, never from code (OMN-21757)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.runtime.gateway_canary_probe import _build_parser as probe_parser
from omnibase_infra.runtime.gateway_forwarder import _build_parser as forwarder_parser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SLICE_PATHS = (
    "src/omnibase_infra/clients/runtime_skill_client.py",
    "src/omnibase_infra/docker/catalog/generator.py",
    "src/omnibase_infra/nodes/node_bus_forwarder_effect/services/service_gateway_egress_health.py",
    "src/omnibase_infra/nodes/node_bus_forwarder_effect/services/service_gateway_lane_mirror_health.py",
    "src/omnibase_infra/nodes/node_scope_extract_compute/models",
    "src/omnibase_infra/nodes/node_scope_file_read_effect/models",
    "src/omnibase_infra/nodes/node_scope_workflow_orchestrator/models",
    "src/omnibase_infra/runtime/gateway_canary_probe.py",
    "src/omnibase_infra/runtime/models/model_local_runtime_ingress_config.py",
)


@pytest.mark.unit
def test_canary_probe_takes_every_path_from_its_arguments() -> None:
    with pytest.raises(SystemExit):
        probe_parser().parse_args(
            ["--config", "forwarder.yaml", "--broker-ref-map", "refs.yaml"]
        )
    args = probe_parser().parse_args(
        [
            "--config",
            "forwarder.yaml",
            "--broker-ref-map",
            "refs.yaml",
            "--state-file",
            "probe-state.json",
            "--egress-health-file",
            "egress.json",
            "--lane-mirror-health-file",
            "lane-mirror.json",
        ]
    )
    assert args.state_file == Path("probe-state.json")
    assert args.egress_health_file == Path("egress.json")
    assert args.lane_mirror_health_file == Path("lane-mirror.json")


@pytest.mark.unit
def test_forwarder_takes_the_health_paths_from_its_arguments() -> None:
    with pytest.raises(SystemExit):
        forwarder_parser().parse_args(["--config", "forwarder.yaml"])


@pytest.mark.unit
def test_contracts_and_di_reports_no_literal_path_on_the_slice() -> None:
    pytest.importorskip("omnibase_core.validation.validator_contracts_and_di")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_core.nodes.node_validation_report_write_effect.runtime_validator_run",
            "omnibase_core.validation.validator_contracts_and_di:ValidatorContractsAndDi",
            *_SLICE_PATHS,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Traceback" not in completed.stderr, completed.stderr
    assert "literal_path" not in completed.stdout + completed.stderr
