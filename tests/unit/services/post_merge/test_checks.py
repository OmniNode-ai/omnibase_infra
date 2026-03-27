# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for post-merge check stages.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.github.model_pr_merged_event import ModelPRMergedEvent
from omnibase_infra.services.post_merge.checks import (
    _check_boundary_imports,
    _check_enum_changes,
    _check_hardcoded_secrets,
    _check_missing_error_handling,
    _check_naming_conventions,
    _check_topic_name_changes,
    run_hostile_review,
    run_integration_check,
)
from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)


def _make_event(**kwargs: object) -> ModelPRMergedEvent:
    """Helper to create a ModelPRMergedEvent with defaults."""
    defaults: dict[str, object] = {
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 42,
        "base_ref": "main",
        "head_ref": "feature/test",
        "merge_sha": "abc123def456",
        "author": "testuser",
        "changed_files": [],
        "ticket_ids": [],
        "title": "test PR",
    }
    defaults.update(kwargs)
    return ModelPRMergedEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hostile Review: Secret Detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckHardcodedSecrets:
    """Tests for _check_hardcoded_secrets."""

    def test_detects_aws_key(self) -> None:
        diff = "+++ b/config.py\n+AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'\n"
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) == 1
        assert findings[0].severity == EnumFindingSeverity.CRITICAL
        assert "AWS access key" in findings[0].title

    def test_detects_api_key(self) -> None:
        diff = '+++ b/settings.py\n+api_key = "sk-1234567890abcdef"\n'
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) == 1
        assert "API key" in findings[0].title

    def test_detects_private_key(self) -> None:
        diff = "+++ b/cert.pem\n+-----BEGIN RSA PRIVATE KEY-----\n"
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) == 1

    def test_ignores_removed_lines(self) -> None:
        diff = "+++ b/config.py\n-AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'\n"
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) == 0

    def test_no_false_positive_on_clean_diff(self) -> None:
        diff = '+++ b/main.py\n+def hello():\n+    print("hello world")\n'
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) == 0

    def test_tracks_file_path(self) -> None:
        diff = "+++ b/src/secrets.py\n+api_key = 'sk_live_abcdefghij12345678'\n"
        event = _make_event()
        findings = _check_hardcoded_secrets(diff, event)
        assert len(findings) >= 1
        assert findings[0].file_path == "src/secrets.py"


# ---------------------------------------------------------------------------
# Hostile Review: Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckMissingErrorHandling:
    """Tests for _check_missing_error_handling."""

    def test_detects_bare_except(self) -> None:
        diff = (
            "+++ b/handler.py\n"
            "+    try:\n"
            "+        do_something()\n"
            "+    except:\n"
            "+        pass\n"
        )
        event = _make_event()
        findings = _check_missing_error_handling(diff, event)
        assert len(findings) == 1
        assert findings[0].severity == EnumFindingSeverity.MEDIUM

    def test_ignores_typed_except(self) -> None:
        diff = "+++ b/handler.py\n+    except ValueError:\n"
        event = _make_event()
        findings = _check_missing_error_handling(diff, event)
        assert len(findings) == 0

    def test_ignores_non_python_files(self) -> None:
        diff = "+++ b/handler.js\n+    except:\n"
        event = _make_event()
        findings = _check_missing_error_handling(diff, event)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Hostile Review: Naming Conventions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckNamingConventions:
    """Tests for _check_naming_conventions."""

    def test_flags_basemodel_without_model_prefix(self) -> None:
        diff = "+++ b/src/models.py\n+class UserData(BaseModel):\n"
        event = _make_event()
        findings = _check_naming_conventions(diff, event)
        assert len(findings) == 1
        assert "UserData" in findings[0].title

    def test_accepts_model_prefix(self) -> None:
        diff = "+++ b/src/models.py\n+class ModelUserData(BaseModel):\n"
        event = _make_event()
        findings = _check_naming_conventions(diff, event)
        assert len(findings) == 0

    def test_accepts_config_prefix(self) -> None:
        diff = "+++ b/src/config.py\n+class ConfigApp(BaseModel):\n"
        event = _make_event()
        findings = _check_naming_conventions(diff, event)
        assert len(findings) == 0

    def test_accepts_enum_prefix(self) -> None:
        diff = "+++ b/src/enums.py\n+class EnumStatus(BaseModel):\n"
        event = _make_event()
        findings = _check_naming_conventions(diff, event)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Hostile Review: Async Entry Point
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHostileReview:
    """Tests for run_hostile_review."""

    @pytest.mark.asyncio
    async def test_skips_without_token(self) -> None:
        event = _make_event()
        findings = await run_hostile_review(event, github_token="")
        assert findings == []


# ---------------------------------------------------------------------------
# Integration Check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckBoundaryImports:
    """Tests for _check_boundary_imports."""

    def test_flags_init_changes(self) -> None:
        event = _make_event(
            changed_files=["src/omnibase_infra/__init__.py", "src/other.py"]
        )
        findings = _check_boundary_imports(event)
        assert len(findings) == 1
        assert findings[0].stage == EnumCheckStage.INTEGRATION_CHECK
        assert findings[0].severity == EnumFindingSeverity.INFO

    def test_no_findings_without_init_changes(self) -> None:
        event = _make_event(changed_files=["src/module.py"])
        findings = _check_boundary_imports(event)
        assert len(findings) == 0


@pytest.mark.unit
class TestCheckTopicNameChanges:
    """Tests for _check_topic_name_changes."""

    def test_flags_topic_suffix_changes(self) -> None:
        event = _make_event(
            changed_files=[
                "src/omnibase_infra/topics/platform_topic_suffixes.py",
            ]
        )
        findings = _check_topic_name_changes(event)
        assert len(findings) == 1
        assert findings[0].severity == EnumFindingSeverity.HIGH

    def test_flags_topics_yaml_changes(self) -> None:
        event = _make_event(changed_files=["src/omnibase_infra/services/topics.yaml"])
        findings = _check_topic_name_changes(event)
        assert len(findings) == 1

    def test_no_findings_without_topic_changes(self) -> None:
        event = _make_event(changed_files=["src/foo.py"])
        findings = _check_topic_name_changes(event)
        assert len(findings) == 0


@pytest.mark.unit
class TestCheckEnumChanges:
    """Tests for _check_enum_changes."""

    def test_flags_enum_file_changes(self) -> None:
        event = _make_event(changed_files=["src/omnibase_infra/enums/enum_status.py"])
        findings = _check_enum_changes(event)
        assert len(findings) == 1
        assert findings[0].severity == EnumFindingSeverity.MEDIUM

    def test_ignores_non_src_enum_files(self) -> None:
        event = _make_event(changed_files=["tests/enum_test.py"])
        findings = _check_enum_changes(event)
        assert len(findings) == 0


@pytest.mark.unit
class TestRunIntegrationCheck:
    """Tests for run_integration_check."""

    @pytest.mark.asyncio
    async def test_aggregates_all_sub_checks(self) -> None:
        event = _make_event(
            changed_files=[
                "src/omnibase_infra/__init__.py",
                "src/omnibase_infra/topics/platform_topic_suffixes.py",
                "src/omnibase_infra/enums/enum_status.py",
            ]
        )
        findings = await run_integration_check(event)
        # Should have findings from all 3 sub-checks
        assert len(findings) == 3
        stages = {f.stage for f in findings}
        assert stages == {EnumCheckStage.INTEGRATION_CHECK}
