# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the wiring-report ratchet node (OMN-26335).

Loads the node's real contract from disk, resolves each handler through the
module path the contract declares, reads the counts from the installed boot
catalog and checks them against the baseline the contract holds.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.models.wiring_ratchet import (
    ModelWiringCounts,
    ModelWiringReportRatchetRequest,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_wiring_report_ratchet_compute/contract.yaml"
)


def _handler(operation: str) -> Any:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entries = {
        entry["operation"]: entry["handler"]
        for entry in contract["handler_routing"]["handlers"]
    }
    declared = entries[operation]
    module = importlib.import_module(declared["module"])
    return getattr(module, declared["name"])()


def test_installed_boot_catalog_stays_within_the_contract_baseline() -> None:
    """The counts read from the installed catalog are at or below the contract baseline."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    baseline = ModelWiringCounts(**contract["metadata"]["wiring_ratchet"]["baseline"])
    counts = _handler("wiring_report_counts.read").handle(discover_contracts())
    report = _handler("wiring_report_ratchet.check").handle(
        ModelWiringReportRatchetRequest(
            observed=counts,
            baseline=baseline,
            base_baseline=None,
            scope="installed boot catalog, bound handlers",
        )
    )
    assert not any(f.severity == "FAIL" for f in report.findings), [
        f.message for f in report.findings
    ]


def test_a_count_above_the_baseline_fails_through_the_bound_handler() -> None:
    """The handler the contract binds refuses a count one above the baseline."""
    baseline = ModelWiringCounts(
        failed=0, duplicates=1, quarantined=0, unproduced_subscriptions=1
    )
    observed = baseline.model_copy(update={"duplicates": baseline.duplicates + 1})
    report = _handler("wiring_report_ratchet.check").handle(
        ModelWiringReportRatchetRequest(
            observed=observed,
            baseline=baseline,
            base_baseline=None,
            scope="bound handler",
        )
    )
    assert any(f.severity == "FAIL" for f in report.findings)
