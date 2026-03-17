# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the topic drift checker script."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_topic_drift import (
    _extract_suffix_values,
    _is_infrastructure_topic,
    main,
)


@pytest.mark.unit
class TestExtractSuffixValues:
    """Tests for AST-based SUFFIX_* extraction."""

    def test_extracts_simple_string_constants(self, tmp_path: Path) -> None:
        """Extracts SUFFIX_* = "string" assignments."""
        py_file = tmp_path / "suffixes.py"
        py_file.write_text(
            textwrap.dedent("""\
                SUFFIX_NODE_REGISTRATION: str = "onex.evt.platform.node-registration.v1"
                SUFFIX_NODE_HEARTBEAT: str = "onex.evt.platform.node-heartbeat.v1"
                NOT_A_SUFFIX: str = "should-be-ignored"
            """)
        )
        result = _extract_suffix_values(py_file)
        assert result == {
            "SUFFIX_NODE_REGISTRATION": "onex.evt.platform.node-registration.v1",
            "SUFFIX_NODE_HEARTBEAT": "onex.evt.platform.node-heartbeat.v1",
        }

    def test_ignores_non_string_assignments(self, tmp_path: Path) -> None:
        """Non-string SUFFIX_* values are ignored."""
        py_file = tmp_path / "suffixes.py"
        py_file.write_text(
            textwrap.dedent("""\
                SUFFIX_COUNT: int = 42
                SUFFIX_REAL: str = "onex.evt.platform.test.v1"
            """)
        )
        result = _extract_suffix_values(py_file)
        assert result == {"SUFFIX_REAL": "onex.evt.platform.test.v1"}


@pytest.mark.unit
class TestIsInfrastructureTopic:
    """Tests for infrastructure topic filtering."""

    def test_dlq_topic_excluded(self) -> None:
        assert _is_infrastructure_topic("onex.dlq.intents.v1") is True

    def test_broadcast_topic_excluded(self) -> None:
        assert _is_infrastructure_topic("dev.broadcast") is True

    def test_normal_topic_not_excluded(self) -> None:
        assert (
            _is_infrastructure_topic("onex.evt.platform.node-registration.v1") is False
        )


@pytest.mark.unit
class TestDriftDetection:
    """End-to-end drift detection tests using synthetic files."""

    def _setup_files(
        self,
        tmp_path: Path,
        suffix_content: str,
        contracts: dict[str, str],
    ) -> tuple[Path, Path]:
        """Create synthetic suffix file and contract directories.

        Returns:
            Tuple of (contracts_dir, constants_file).
        """
        constants_file = tmp_path / "platform_topic_suffixes.py"
        constants_file.write_text(suffix_content)

        contracts_dir = tmp_path / "nodes"
        for node_name, contract_content in contracts.items():
            node_dir = contracts_dir / node_name
            node_dir.mkdir(parents=True)
            (node_dir / "contract.yaml").write_text(contract_content)

        return contracts_dir, constants_file

    def test_no_drift_exit_0(self, tmp_path: Path) -> None:
        """When all constants are in contracts, exit 0."""
        contracts_dir, constants_file = self._setup_files(
            tmp_path,
            suffix_content=textwrap.dedent("""\
                SUFFIX_A: str = "onex.evt.platform.topic-a.v1"
            """),
            contracts={
                "node_a": textwrap.dedent("""\
                    name: "node_a"
                    event_bus:
                      subscribe_topics:
                        - "onex.evt.platform.topic-a.v1"
                """),
            },
        )

        with patch(
            "sys.argv",
            [
                "check_topic_drift.py",
                "--contracts-dir",
                str(contracts_dir),
                "--constants-file",
                str(constants_file),
            ],
        ):
            assert main() == 0

    def test_orphaned_constant_exit_1(self, tmp_path: Path) -> None:
        """When a constant is not in any contract, exit 1."""
        contracts_dir, constants_file = self._setup_files(
            tmp_path,
            suffix_content=textwrap.dedent("""\
                SUFFIX_A: str = "onex.evt.platform.topic-a.v1"
                SUFFIX_ORPHAN: str = "onex.evt.platform.orphan.v1"
            """),
            contracts={
                "node_a": textwrap.dedent("""\
                    name: "node_a"
                    event_bus:
                      subscribe_topics:
                        - "onex.evt.platform.topic-a.v1"
                """),
            },
        )

        with patch(
            "sys.argv",
            [
                "check_topic_drift.py",
                "--contracts-dir",
                str(contracts_dir),
                "--constants-file",
                str(constants_file),
            ],
        ):
            assert main() == 1

    def test_undeclared_topic_warning_exit_0(self, tmp_path: Path) -> None:
        """Contract topics not in constants are warnings only (exit 0)."""
        contracts_dir, constants_file = self._setup_files(
            tmp_path,
            suffix_content=textwrap.dedent("""\
                SUFFIX_A: str = "onex.evt.platform.topic-a.v1"
            """),
            contracts={
                "node_a": textwrap.dedent("""\
                    name: "node_a"
                    event_bus:
                      subscribe_topics:
                        - "onex.evt.platform.topic-a.v1"
                      publish_topics:
                        - "onex.evt.platform.extra-topic.v1"
                """),
            },
        )

        with patch(
            "sys.argv",
            [
                "check_topic_drift.py",
                "--contracts-dir",
                str(contracts_dir),
                "--constants-file",
                str(constants_file),
            ],
        ):
            assert main() == 0

    def test_dlq_constants_excluded(self, tmp_path: Path) -> None:
        """DLQ SUFFIX_* constants are excluded from orphan detection."""
        contracts_dir, constants_file = self._setup_files(
            tmp_path,
            suffix_content=textwrap.dedent("""\
                SUFFIX_A: str = "onex.evt.platform.topic-a.v1"
                SUFFIX_DLQ: str = "onex.dlq.agent-actions.v1"
            """),
            contracts={
                "node_a": textwrap.dedent("""\
                    name: "node_a"
                    event_bus:
                      subscribe_topics:
                        - "onex.evt.platform.topic-a.v1"
                """),
            },
        )

        with patch(
            "sys.argv",
            [
                "check_topic_drift.py",
                "--contracts-dir",
                str(contracts_dir),
                "--constants-file",
                str(constants_file),
            ],
        ):
            # DLQ constant is excluded, so no orphans
            assert main() == 0
