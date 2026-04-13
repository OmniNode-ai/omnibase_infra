# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for quality-assessment topic registration [OMN-8144].

Verifies that the new quality-scoring pipeline topics are exported from
omnibase_infra.topics and present in ALL_INTELLIGENCE_TOPIC_SPECS.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_quality_assessment_cmd_suffix_exported() -> None:
    """SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD must be importable from topics."""
    from omnibase_infra.topics import (
        SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD,
    )


@pytest.mark.integration
def test_quality_assessment_completed_suffix_exported() -> None:
    """SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED must be importable from topics."""
    from omnibase_infra.topics import (
        SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED,
    )


@pytest.mark.integration
def test_quality_assessment_topics_in_all_intelligence_specs() -> None:
    """Both quality-assessment topics must appear in ALL_INTELLIGENCE_TOPIC_SPECS."""
    from omnibase_infra.topics import (
        SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD,
        SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED,
    )
    from omnibase_infra.topics.platform_topic_suffixes import (
        ALL_INTELLIGENCE_TOPIC_SPECS,
    )

    registered_suffixes = {spec.suffix for spec in ALL_INTELLIGENCE_TOPIC_SPECS}
    assert SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD in registered_suffixes, (
        f"{SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD} not in ALL_INTELLIGENCE_TOPIC_SPECS"
    )
    assert SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED in registered_suffixes, (
        f"{SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED} not in ALL_INTELLIGENCE_TOPIC_SPECS"
    )


@pytest.mark.integration
def test_quality_assessment_topics_in_init_all() -> None:
    """Both quality-assessment topics must appear in omnibase_infra.topics.__all__."""
    import omnibase_infra.topics as topics_pkg

    assert "SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD" in topics_pkg.__all__
    assert "SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED" in topics_pkg.__all__


@pytest.mark.integration
def test_quality_assessment_cmd_topic_name_format() -> None:
    """Quality-assessment cmd topic must follow onex.cmd.* naming convention."""
    from omnibase_infra.topics import SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD

    assert SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_CMD.startswith("onex.cmd.")


@pytest.mark.integration
def test_quality_assessment_completed_topic_name_format() -> None:
    """Quality-assessment completed topic must follow onex.evt.* naming convention."""
    from omnibase_infra.topics import SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED

    assert SUFFIX_INTELLIGENCE_QUALITY_ASSESSMENT_COMPLETED.startswith("onex.evt.")
