# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The cost API reads its snapshot topics from a declared file (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.cost_api import ModelCostApiTopics, load_cost_api_topics
from omnibase_infra.services.cost_api import snapshot_cache

_TOPICS_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "services"
    / "cost_api"
    / "cost_api_topics.yaml"
)


def _declared_block() -> dict[str, object]:
    return yaml.safe_load(_TOPICS_FILE.read_text(encoding="utf-8"))["topics"]


@pytest.mark.unit
class TestCostApiTopicsContract:
    def test_loaded_topics_equal_the_declared_block(self) -> None:
        assert load_cost_api_topics().model_dump() == _declared_block()

    def test_snapshot_cache_constants_are_the_declared_topics(self) -> None:
        declared = _declared_block()
        assert declared["summary"] == snapshot_cache.TOPIC_COST_SUMMARY
        assert declared["by_repo"] == snapshot_cache.TOPIC_COST_BY_REPO
        assert declared["token_usage"] == snapshot_cache.TOPIC_COST_TOKEN_USAGE

    @pytest.mark.parametrize("key", sorted(_declared_block()))
    def test_block_missing_a_topic_fails_to_load(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        del block[key]
        path = tmp_path / "topics.yaml"
        path.write_text(yaml.safe_dump({"topics": block}), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            ModelCostApiTopics.from_file(path)

    def test_non_snapshot_topic_is_refused(self, tmp_path: Path) -> None:
        block = _declared_block()
        block["summary"] = "onex.evt.platform.node-introspection.v1"
        path = tmp_path / "topics.yaml"
        path.write_text(yaml.safe_dump({"topics": block}), encoding="utf-8")
        with pytest.raises(ValueError, match="summary"):
            ModelCostApiTopics.from_file(path)

    def test_file_without_a_topics_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "topics.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ProtocolConfigurationError):
            ModelCostApiTopics.from_file(path)
