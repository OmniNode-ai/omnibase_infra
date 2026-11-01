# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The plan's scoped write topics come from the node contract (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.developer_principal import topics

_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_developer_principal_plan_compute"
    / "contract.yaml"
)


@pytest.mark.unit
class TestScopedWriteTopicsContract:
    def test_topics_equal_the_contract_declaration(self) -> None:
        declared = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
        assert frozenset(declared["scoped_write_topics"]) == topics.SCOPED_WRITE_TOPICS

    @pytest.mark.parametrize(
        "document",
        ["other: 1\n", "scoped_write_topics: []\n", "scoped_write_topics: [a.b]\n"],
    )
    def test_contract_without_command_topics_is_refused(
        self, document: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "contract.yaml"
        path.write_text(document, encoding="utf-8")
        with pytest.raises(ValueError, match="scoped_write_topics"):
            topics._contract_scoped_write_topics(path)
