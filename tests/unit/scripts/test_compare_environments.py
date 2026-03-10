# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for scripts/compare-environments.py — parity checker."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


@pytest.mark.unit
def test_parity_report_serializes_to_json() -> None:
    from compare_environments import (
        ModelParityFinding,
        ModelParityReport,
        ModelParitySummary,
    )

    finding = ModelParityFinding(
        check_id="credential_parity",
        severity="CRITICAL",
        title="Wrong POSTGRES_USER for omniintelligence-credentials",
        detail="k8s secret has 'postgres'; expected 'role_omniintelligence'",
        local_value="role_omniintelligence",
        cloud_value="postgres",
        auto_fixable=False,
        fix_hint="Re-seed /dev/omniintelligence/ in Infisical and force-resync the InfisicalSecret",
    )
    report = ModelParityReport(
        run_id="abc123",
        generated_at="2026-03-10T00:00:00Z",
        mode="check",
        checks_run=["credential_parity"],
        findings=[finding],
        summary=ModelParitySummary(
            critical_count=1, warning_count=0, info_count=0, checks_skipped=[]
        ),
    )
    raw = report.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["findings"][0]["severity"] == "CRITICAL"
    assert parsed["summary"]["critical_count"] == 1
