# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration-level unit tests for topic registry consolidation.

Validates the end-to-end correctness of the ProtocolTopicRegistry
consolidation: all keys resolvable, no legacy TOPIC_* imports remain,
and the registry is the single source of truth for topic strings.

Related:
    - OMN-5839: Topic registry consolidation epic
    - OMN-5848: Final verification ticket

.. versionadded:: 0.24.0
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]

# Matches ``from omnibase_infra.event_bus.topic_constants import ... TOPIC_...``
# in BOTH single-line and multiline parenthesized form. The original single-line
# grep regex (``...import.*TOPIC_`` matched per line) missed the parenthesized
# form below because the ``TOPIC_*`` name lands on a different line than the
# ``import (`` token (false-green):
#
#     from omnibase_infra.event_bus.topic_constants import (
#         TOPIC_DELEGATION_REQUEST,
#     )
#
# Two non-overlapping alternatives keep the match scoped to the single import
# statement so it cannot bleed into an unrelated later ``TOPIC_`` reference:
#   1. multiline: ``import (`` ... up to the closing ``)`` containing ``TOPIC_``
#   2. single-line: ``import`` ... up to end-of-line containing ``TOPIC_``
_TOPIC_CONSTANT_IMPORT_RE = re.compile(
    r"from\s+omnibase_infra\.event_bus\.topic_constants\s+import\s+"
    r"(?:\([^)]*\bTOPIC_[^)]*\)|[^\n(]*\bTOPIC_[^\n]*)",
    re.DOTALL,
)


def _find_topic_constant_imports(text: str) -> bool:
    """Return True if *text* imports any ``TOPIC_*`` name from topic_constants.

    Catches both single-line and multiline parenthesized import forms.
    """
    return _TOPIC_CONSTANT_IMPORT_RE.search(text) is not None


@pytest.mark.unit
class TestTopicRegistryIntegration:
    """End-to-end verification of topic registry consolidation."""

    def test_all_keys_resolvable(self) -> None:
        """Every topic key in __all__ resolves to a valid onex.* topic."""
        registry = ServiceTopicRegistry.from_defaults()
        for key_name in topic_keys.__all__:
            key = getattr(topic_keys, key_name)
            topic = registry.resolve(key)
            assert topic.startswith("onex."), (
                f"Topic for {key} doesn't match naming: {topic}"
            )

    def test_no_topic_constant_imports_remain_in_src(self) -> None:
        """No src/ file imports TOPIC_* from topic_constants.

        Reads each ``src/**/*.py`` whole-file (not line-by-line) and applies a
        multiline-aware matcher so parenthesized imports are caught. The previous
        single-line grep was false-green against the parenthesized form.
        """
        src_root = REPO_ROOT / "src"
        offenders: list[str] = []
        for py_file in sorted(src_root.rglob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            if _find_topic_constant_imports(text):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
        assert offenders == [], (
            "Remaining TOPIC_* imports from topic_constants in src/:\n"
            + "\n".join(offenders)
        )

    def test_guard_matcher_fires_on_planted_multiline_violation(self) -> None:
        """Regression: the matcher must catch the multiline parenthesized form.

        This is the exact shape the original single-line grep missed. If the
        matcher silently false-greens again, this assertion fails.
        """
        planted_multiline = (
            "from omnibase_infra.event_bus.topic_constants import (\n"
            "    TOPIC_DELEGATION_REQUEST,\n"
            ")\n"
        )
        planted_singleline = "from omnibase_infra.event_bus.topic_constants import TOPIC_DELEGATION_FAILED\n"
        # Both real violation shapes must be detected.
        assert _find_topic_constant_imports(planted_multiline), (
            "matcher failed to detect a multiline parenthesized TOPIC_* import "
            "(false-green regression)"
        )
        assert _find_topic_constant_imports(planted_singleline), (
            "matcher failed to detect a single-line TOPIC_* import"
        )
        # A topic_constants import that pulls ONLY non-TOPIC_ helpers (e.g. the
        # DLQ builders) must NOT be flagged — those are legitimate and stay.
        legitimate_dlq = (
            "from omnibase_infra.event_bus.topic_constants import (\n"
            "    build_dlq_topic,\n"
            "    parse_dlq_topic,\n"
            ")\n"
        )
        assert not _find_topic_constant_imports(legitimate_dlq), (
            "matcher false-positived on a legitimate DLQ-helper-only import"
        )

    def test_no_wiring_health_monitored_imports_remain_in_src(self) -> None:
        """No src/ file imports WIRING_HEALTH_MONITORED_TOPICS."""
        result = subprocess.run(
            [
                "grep",
                "-rn",
                r"from omnibase_infra.event_bus.topic_constants import.*WIRING_HEALTH",
                "src/",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.stdout == "", (
            f"Remaining WIRING_HEALTH imports in src/:\n{result.stdout}"
        )

    def test_topic_constants_only_has_dlq_and_topic_constants(self) -> None:
        """topic_constants.py __all__ contains only DLQ items, TOPIC_ constants, and functions."""
        from omnibase_infra.event_bus.topic_constants import __all__ as tc_all

        for name in tc_all:
            is_dlq = name.startswith(("DLQ_", "_DLQ_"))
            is_topic = name.startswith("TOPIC_")
            is_func = name in {
                "build_dlq_topic",
                "parse_dlq_topic",
                "is_dlq_topic",
                "get_dlq_topic_for_original",
                "derive_dlq_topic_for_event_type",
            }
            assert is_dlq or is_topic or is_func, (
                f"Unexpected item in topic_constants.__all__: {name}"
            )

    def test_registry_key_count_matches_topic_keys(self) -> None:
        """Registry has exactly as many entries as topic_keys.__all__."""
        registry = ServiceTopicRegistry.from_defaults()
        assert len(registry.all_keys()) == len(topic_keys.__all__)

    def test_all_topic_strings_are_unique(self) -> None:
        """No two keys resolve to the same topic string."""
        registry = ServiceTopicRegistry.from_defaults()
        topics = [registry.resolve(k) for k in registry.all_keys()]
        assert len(set(topics)) == len(topics), (
            "Duplicate topic strings detected in registry"
        )
