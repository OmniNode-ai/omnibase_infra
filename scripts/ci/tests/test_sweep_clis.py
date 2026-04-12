#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unit tests for run_compliance_sweep.py, run_duplication_sweep.py, run_contract_sweep.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing scripts from parent dir
SCRIPTS_CI = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_CI))

FIXTURES = Path(__file__).parent / "fixtures"
COMPLIANCE_REPO = FIXTURES / "compliance_repo"
DUPLICATION_REPO = FIXTURES / "duplication_repo"


# ---------------------------------------------------------------------------
# compliance sweep tests
# ---------------------------------------------------------------------------


class TestComplianceSweep:
    def test_finds_hardcoded_topic(self):
        from run_compliance_sweep import sweep_repo

        handlers_scanned, findings = sweep_repo(COMPLIANCE_REPO, ["hardcoded-topics"])
        hardcoded = [f for f in findings if f["violation_type"] == "HARDCODED_TOPIC"]
        assert len(hardcoded) >= 1
        assert any("onex.evt" in f["message"] for f in hardcoded)

    def test_finds_transport_import(self):
        from run_compliance_sweep import sweep_repo

        _, findings = sweep_repo(COMPLIANCE_REPO, ["undeclared-transport"])
        transport = [f for f in findings if f["violation_type"] == "UNDECLARED_TRANSPORT"]
        assert len(transport) >= 1
        assert any("httpx" in f["message"] for f in transport)

    def test_finds_logic_in_node(self):
        from run_compliance_sweep import sweep_repo

        _, findings = sweep_repo(COMPLIANCE_REPO, ["logic-in-node"])
        logic = [f for f in findings if f["violation_type"] == "LOGIC_IN_NODE"]
        assert len(logic) >= 1

    def test_clean_handler_no_violations(self):
        from run_compliance_sweep import sweep_repo

        # Only scan for hardcoded topics — handler_clean.py has none
        _, findings = sweep_repo(COMPLIANCE_REPO, ["hardcoded-topics"])
        # handler_clean.py should not be in findings
        clean_findings = [f for f in findings if "handler_clean" in f["file"]]
        assert len(clean_findings) == 0

    def test_exit_0_on_no_findings(self, tmp_path):
        clean_repo = tmp_path / "clean_repo" / "handlers"
        clean_repo.mkdir(parents=True)
        (clean_repo / "handler_ok.py").write_text("def run(): pass\n")

        from run_compliance_sweep import main

        result = main(["--repo", str(tmp_path / "clean_repo"), "--checks", "hardcoded-topics"])
        assert result == 0

    def test_exit_1_on_findings(self):
        from run_compliance_sweep import main

        result = main(
            [
                "--repo",
                str(COMPLIANCE_REPO),
                "--checks",
                "hardcoded-topics",
                "--fail-on-severity",
                "error",
            ]
        )
        assert result == 1

    def test_json_output(self, capsys):
        import json as json_mod

        from run_compliance_sweep import main

        main(
            [
                "--repo",
                str(COMPLIANCE_REPO),
                "--checks",
                "hardcoded-topics",
                "--json",
            ]
        )
        captured = capsys.readouterr()
        data = json_mod.loads(captured.out)
        assert "findings" in data
        assert "status" in data
        assert data["sweep"] == "compliance_sweep"

    def test_warning_only_does_not_fail_on_error_threshold(self, tmp_path):
        # UNDECLARED_TRANSPORT is severity=warning, --fail-on-severity=error → exit 0
        transport_repo = tmp_path / "transport_repo" / "src" / "handlers"
        transport_repo.mkdir(parents=True)
        (transport_repo / "handler_t.py").write_text("import httpx\n")

        from run_compliance_sweep import main

        result = main(
            [
                "--repo",
                str(tmp_path / "transport_repo"),
                "--checks",
                "undeclared-transport",
                "--fail-on-severity",
                "error",
            ]
        )
        assert result == 0

    def test_warning_fails_on_warning_threshold(self, tmp_path):
        transport_repo = tmp_path / "transport_repo" / "src" / "handlers"
        transport_repo.mkdir(parents=True)
        (transport_repo / "handler_t.py").write_text("import httpx\n")

        from run_compliance_sweep import main

        result = main(
            [
                "--repo",
                str(tmp_path / "transport_repo"),
                "--checks",
                "undeclared-transport",
                "--fail-on-severity",
                "warning",
            ]
        )
        assert result == 1

    def test_missing_repo_is_skipped(self, capsys):
        from run_compliance_sweep import main

        result = main(["--repo", "/nonexistent/path/repo"])
        assert result == 0  # no findings when repo not found


# ---------------------------------------------------------------------------
# duplication sweep tests
# ---------------------------------------------------------------------------


class TestDuplicationSweep:
    def test_d1_finds_duplicate_table(self):
        from run_duplication_sweep import check_d1_drizzle_tables

        result = check_d1_drizzle_tables(DUPLICATION_REPO)
        assert result["status"] == "FAIL"
        assert result["finding_count"] >= 1
        table_names = [f["table"] for f in result["findings"]]
        assert "users" in table_names

    def test_d2_finds_topic_conflict(self):
        from run_duplication_sweep import check_d2_topic_conflicts

        result = check_d2_topic_conflicts(DUPLICATION_REPO)
        assert result["status"] == "FAIL"
        assert result["finding_count"] >= 1
        topics = [f["topic"] for f in result["findings"]]
        assert "onex.evt.omniclaude.prompt-submitted.v1" in topics

    def test_d2_no_conflict_when_producer_matches(self, tmp_path):
        from run_duplication_sweep import check_d2_topic_conflicts

        # Create fixture where both topics have omniclaude as producer
        hooks_dir = tmp_path / "omniclaude" / "src" / "omniclaude" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "topics.py").write_text(
            'class T:\n    A = "onex.evt.omniclaude.session.v1"\n'
        )
        bounds_dir = (
            tmp_path
            / "onex_change_control"
            / "src"
            / "onex_change_control"
            / "boundaries"
        )
        bounds_dir.mkdir(parents=True)
        (bounds_dir / "kafka_boundaries.yaml").write_text(
            "boundaries:\n"
            "  - topic_name: onex.evt.omniclaude.session.v1\n"
            "    producer_repo: omniclaude\n"
        )
        result = check_d2_topic_conflicts(tmp_path)
        assert result["status"] == "PASS"

    def test_d1_pass_when_no_duplicates(self, tmp_path):
        from run_duplication_sweep import check_d1_drizzle_tables

        shared = tmp_path / "omnidash" / "shared"
        shared.mkdir(parents=True)
        (shared / "schema-a.ts").write_text('export const foo = pgTable("foo_table", {});\n')
        (shared / "schema-b.ts").write_text('export const bar = pgTable("bar_table", {});\n')
        result = check_d1_drizzle_tables(tmp_path)
        assert result["status"] == "PASS"

    def test_json_output(self, capsys):
        import json as json_mod

        from run_duplication_sweep import main

        main(["--omni-home", str(DUPLICATION_REPO), "--checks", "D1", "--json"])
        captured = capsys.readouterr()
        data = json_mod.loads(captured.out)
        assert data["sweep"] == "duplication_sweep"
        assert "results" in data

    def test_exit_code_on_fail(self):
        from run_duplication_sweep import main

        result = main(
            ["--omni-home", str(DUPLICATION_REPO), "--checks", "D1,D2", "--fail-on-severity", "error"]
        )
        assert result == 1

    def test_missing_omni_home_returns_2(self):
        from run_duplication_sweep import main

        result = main(["--omni-home", "/nonexistent/omni_home"])
        assert result == 2


# ---------------------------------------------------------------------------
# contract sweep tests
# ---------------------------------------------------------------------------


class TestContractSweep:
    def test_drift_mode_skips_missing_repos(self, tmp_path):
        from run_contract_sweep import run_drift_mode

        result = run_drift_mode(
            omni_home=tmp_path,
            repos=["nonexistent_repo"],
            sensitivity="STANDARD",
            check_boundaries=False,
        )
        assert result["status"] in ("clean",)
        assert "nonexistent_repo" in result["repos_not_found"]

    def test_drift_mode_baseline_missing(self, tmp_path):
        from run_contract_sweep import run_drift_mode

        # Create a minimal repo structure
        (tmp_path / "my_repo" / "src").mkdir(parents=True)
        # No snapshot file → baseline_missing
        result = run_drift_mode(
            omni_home=tmp_path,
            repos=["my_repo"],
            sensitivity="STANDARD",
            check_boundaries=False,
        )
        assert "my_repo" in result["baseline_missing"]

    def test_check_boundaries_missing_file(self, tmp_path):
        from run_contract_sweep import _check_boundaries

        # No boundaries file → returns empty list
        result = _check_boundaries(tmp_path)
        assert result == []

    def test_json_output_structure(self, capsys, tmp_path):
        import json as json_mod

        from run_contract_sweep import main

        main([
            "--omni-home", str(tmp_path),
            "--repos", "nonexistent",
            "--mode", "drift",
            "--json",
            "--no-check-boundaries",
        ])
        captured = capsys.readouterr()
        data = json_mod.loads(captured.out)
        assert data["sweep"] == "contract_sweep"
        assert "drift" in data
        assert "status" in data

    def test_exit_0_when_clean(self, tmp_path):
        from run_contract_sweep import main

        result = main([
            "--omni-home", str(tmp_path),
            "--repos", "nonexistent",
            "--mode", "drift",
            "--no-check-boundaries",
        ])
        assert result == 0

    def test_missing_omni_home_returns_2(self):
        from run_contract_sweep import main

        result = main(["--omni-home", "/nonexistent/path"])
        assert result == 2
