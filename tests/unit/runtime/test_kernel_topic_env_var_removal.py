# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD-first: ONEX_INPUT_TOPIC and ONEX_OUTPUT_TOPIC must not be read via os.getenv.

OMN-8784: Topic catalog wiring — derive ONEX_INPUT_TOPIC etc. from contract
subscriptions, remove env vars.

The 5 topic-name env vars must be removed from handler source and replaced with
contract-declared topic resolution. This test gates that migration.

Patterns checked in service_kernel.py:
    - os.getenv("ONEX_INPUT_TOPIC", ...)
    - os.getenv("ONEX_OUTPUT_TOPIC", ...)
    - os.environ.get("ONEX_INPUT_TOPIC", ...)
    - os.environ.get("ONEX_OUTPUT_TOPIC", ...)

After migration, these env vars must be absent. If set, the kernel must hard-fail
at startup (not silently ignore or fall back).
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

# Patterns for topic env var reads that must be absent after migration
TOPIC_ENV_VAR_PATTERNS: list[str] = [
    r'os\.getenv\s*\(\s*["\']ONEX_INPUT_TOPIC["\']',
    r'os\.getenv\s*\(\s*["\']ONEX_OUTPUT_TOPIC["\']',
    r'os\.environ\.get\s*\(\s*["\']ONEX_INPUT_TOPIC["\']',
    r'os\.environ\.get\s*\(\s*["\']ONEX_OUTPUT_TOPIC["\']',
    r'os\.environ\s*\[\s*["\']ONEX_INPUT_TOPIC["\']',
    r'os\.environ\s*\[\s*["\']ONEX_OUTPUT_TOPIC["\']',
]

# Files that must not contain topic env var reads after migration
CHECKED_FILES: list[str] = [
    "src/omnibase_infra/runtime/service_kernel.py",
]


class TestTopicEnvVarRemoval:
    """OMN-8784: verify topic env vars removed from kernel source."""

    @pytest.mark.parametrize("rel_path", CHECKED_FILES)
    def test_no_topic_env_var_reads(self, rel_path: str) -> None:
        """service_kernel.py must not read ONEX_INPUT_TOPIC or ONEX_OUTPUT_TOPIC.

        After OMN-8784 migration, topics are derived from contract-declared
        subscriptions/publishes. Reading them from env vars bypasses the contract
        system and breaks TopicCatalogManager auto-wiring.

        The migration requires: contract declares subscribe_topics -> kernel reads
        topic from contract -> hard-fail if TopicCatalogManager resolution fails.
        """
        repo_root = Path(__file__).resolve().parents[3]
        file_path = repo_root / rel_path

        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")

        content = file_path.read_text()
        violations: list[str] = []

        for line_num, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in TOPIC_ENV_VAR_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f"  Line {line_num}: {line.strip()}")

        assert not violations, (
            f"Found topic env var reads in {rel_path} (OMN-8784):\n"
            + "\n".join(violations)
            + "\n\nTopics must be derived from contract subscriptions/publishes. "
            "Remove os.getenv('ONEX_INPUT_TOPIC') and os.getenv('ONEX_OUTPUT_TOPIC') "
            "and replace with contract-declared topic resolution. Hard-fail at startup "
            "if the deprecated env vars are still set."
        )

    @pytest.mark.parametrize("rel_path", CHECKED_FILES)
    def test_startup_assert_env_vars_absent(self, rel_path: str) -> None:
        """After migration, kernel must assert ONEX_INPUT/OUTPUT_TOPIC are unset.

        The migration DoD requires: migrated node asserts at startup that the
        deprecated env var is absent. This test verifies that service_kernel.py
        contains a deprecation guard for both ONEX_INPUT_TOPIC and ONEX_OUTPUT_TOPIC.
        """
        repo_root = Path(__file__).resolve().parents[3]
        file_path = repo_root / rel_path

        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")

        content = file_path.read_text()

        assert "ONEX_INPUT_TOPIC" in content and "ONEX_OUTPUT_TOPIC" in content, (
            f"{rel_path} must contain deprecation guard references for "
            "ONEX_INPUT_TOPIC and ONEX_OUTPUT_TOPIC (OMN-8784). "
            "The kernel must hard-fail if these deprecated vars are set."
        )

        # Verify they appear in a deprecation/assertion context, not as getenv reads
        deprecated_guard_pattern = re.compile(
            r"ONEX_INPUT_TOPIC.*deprecat|deprecat.*ONEX_INPUT_TOPIC|"
            r"ONEX_INPUT_TOPIC.*forbidden|forbidden.*ONEX_INPUT_TOPIC|"
            r"ONEX_INPUT_TOPIC.*removed|removed.*ONEX_INPUT_TOPIC|"
            r"_DEPRECATED_TOPIC_ENV_VARS|DEPRECATED_TOPIC",
            re.IGNORECASE,
        )
        assert deprecated_guard_pattern.search(content), (
            f"{rel_path} must have a deprecation guard for ONEX_INPUT_TOPIC "
            "and ONEX_OUTPUT_TOPIC. The guard must raise ProtocolConfigurationError "
            "if these env vars are set. (OMN-8784)"
        )
