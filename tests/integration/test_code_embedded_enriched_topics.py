# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for code-embedded and code-enriched topic constants.

Verifies that the new topic suffix constants introduced in this PR are
importable from the public package API and hold the expected string values.

Related:
    - OMN-8XXX: Register SUFFIX_INTELLIGENCE_CODE_EMBEDDED / _ENRICHED topics
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


class TestCodeEmbeddedEnrichedTopicsImportable:
    """Verify constants are importable from the public topics API."""

    def test_suffix_intelligence_code_embedded_importable(self) -> None:
        from omnibase_infra.topics import (
            SUFFIX_INTELLIGENCE_CODE_EMBEDDED,
        )

    def test_suffix_intelligence_code_enriched_importable(self) -> None:
        from omnibase_infra.topics import (
            SUFFIX_INTELLIGENCE_CODE_ENRICHED,
        )


class TestCodeEmbeddedEnrichedTopicValues:
    """Verify exact topic string values."""

    def test_suffix_intelligence_code_embedded_value(self) -> None:
        from omnibase_infra.topics import SUFFIX_INTELLIGENCE_CODE_EMBEDDED

        assert (
            SUFFIX_INTELLIGENCE_CODE_EMBEDDED
            == "onex.evt.omniintelligence.code-embedded.v1"
        )

    def test_suffix_intelligence_code_enriched_value(self) -> None:
        from omnibase_infra.topics import SUFFIX_INTELLIGENCE_CODE_ENRICHED

        assert (
            SUFFIX_INTELLIGENCE_CODE_ENRICHED
            == "onex.evt.omniintelligence.code-enriched.v1"
        )
