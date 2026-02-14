# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the full check catalog (OMN-2151).

Tests:
- Check registry completeness (all 12 checks registered)
- SubprocessCheckExecutor (mypy, ruff, pytest, CI)
- Risk check executors (RISK-001, RISK-002, RISK-003)
- Measurement check executors (COST-001, TIME-001)
- Artifact/replay check executors (VAL-001, VAL-002)
- Check executor config and base class

Note: Some test data intentionally contains references to unsafe
pattern strings because the tests verify the unsafe operations
detector identifies them correctly. These are test fixture data
written to temporary files, not actual code execution.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
    ModelPatternCandidate,
)
from omnibase_infra.validation.checks.check_artifact import (
    EXPECTED_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    CheckArtifactCompleteness,
    CheckReplaySanity,
)
from omnibase_infra.validation.checks.check_executor import (
    ModelCheckExecutorConfig,
    SubprocessCheckExecutor,
)
from omnibase_infra.validation.checks.check_measurement import (
    CheckCostTokenDelta,
    CheckTimeWallClockDelta,
)
from omnibase_infra.validation.checks.check_registry import (
    CHECK_CATALOG_ORDER,
    CHECK_REGISTRY,
    get_check_executor,
)
from omnibase_infra.validation.checks.check_risk import (
    DEFAULT_DIFF_SIZE_THRESHOLD,
    SENSITIVE_PATH_PATTERNS,
    UNSAFE_PATTERNS,
    CheckRiskDiffSize,
    CheckRiskSensitivePaths,
    CheckRiskUnsafeOperations,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers
# ============================================================================


def _make_candidate(
    source_path: str = "/tmp/test",  # noqa: S108
    changed_files: tuple[str, ...] = (),
    risk_tags: tuple[str, ...] = (),
) -> ModelPatternCandidate:
    """Create a pattern candidate for testing."""
    return ModelPatternCandidate(
        candidate_id=uuid4(),
        pattern_id=uuid4(),
        source_path=source_path,
        changed_files=changed_files,
        risk_tags=risk_tags,
    )


def _default_config() -> ModelCheckExecutorConfig:
    """Create a default executor config."""
    return ModelCheckExecutorConfig(working_dir="/tmp", timeout_ms=5000.0)  # noqa: S108


# ============================================================================
# Check Registry Completeness
# ============================================================================


class TestCheckRegistryCompleteness:
    """Verify the check registry contains all 12 checks."""

    def test_registry_has_12_entries(self) -> None:
        """Registry contains exactly 12 check executors."""
        assert len(CHECK_REGISTRY) == 12

    def test_catalog_order_has_12_entries(self) -> None:
        """Catalog order tuple contains exactly 12 check codes."""
        assert len(CHECK_CATALOG_ORDER) == 12

    def test_all_catalog_checks_in_registry(self) -> None:
        """Every check code in the catalog order is in the registry."""
        for code in CHECK_CATALOG_ORDER:
            assert code in CHECK_REGISTRY, f"Missing check: {code}"

    def test_get_check_executor_returns_executor(self) -> None:
        """get_check_executor returns an executor for known codes."""
        for code in CHECK_CATALOG_ORDER:
            executor = get_check_executor(code)
            assert executor is not None, f"get_check_executor returned None for {code}"
            assert executor.check_code == code

    def test_get_check_executor_returns_none_for_unknown(self) -> None:
        """get_check_executor returns None for unknown codes."""
        result = get_check_executor("CHECK-UNKNOWN-999")
        assert result is None

    @pytest.mark.parametrize(
        ("code", "severity"),
        [
            ("CHECK-PY-001", EnumCheckSeverity.REQUIRED),
            ("CHECK-PY-002", EnumCheckSeverity.REQUIRED),
            ("CHECK-TEST-001", EnumCheckSeverity.REQUIRED),
            ("CHECK-TEST-002", EnumCheckSeverity.RECOMMENDED),
            ("CHECK-VAL-001", EnumCheckSeverity.RECOMMENDED),
            ("CHECK-VAL-002", EnumCheckSeverity.REQUIRED),
            ("CHECK-RISK-001", EnumCheckSeverity.REQUIRED),
            ("CHECK-RISK-002", EnumCheckSeverity.RECOMMENDED),
            ("CHECK-RISK-003", EnumCheckSeverity.REQUIRED),
            ("CHECK-OUT-001", EnumCheckSeverity.REQUIRED),
            ("CHECK-COST-001", EnumCheckSeverity.INFORMATIONAL),
            ("CHECK-TIME-001", EnumCheckSeverity.INFORMATIONAL),
        ],
    )
    def test_check_severity_matches_catalog(
        self, code: str, severity: EnumCheckSeverity
    ) -> None:
        """Each check executor has the correct severity from the catalog."""
        executor = CHECK_REGISTRY[code]
        assert executor.severity == severity


# ============================================================================
# SubprocessCheckExecutor
# ============================================================================


class TestSubprocessCheckExecutor:
    """Tests for SubprocessCheckExecutor."""

    def test_properties(self) -> None:
        """Properties return constructor values."""
        executor = SubprocessCheckExecutor(
            check_code="CHECK-TEST",
            label="Test Label",
            severity=EnumCheckSeverity.REQUIRED,
            command="echo ok",
        )
        assert executor.check_code == "CHECK-TEST"
        assert executor.label == "Test Label"
        assert executor.severity == EnumCheckSeverity.REQUIRED

    @pytest.mark.asyncio
    async def test_execute_passing_command(self) -> None:
        """A passing command (exit 0) produces a passed result."""
        executor = SubprocessCheckExecutor(
            check_code="CHECK-ECHO",
            label="Echo test",
            severity=EnumCheckSeverity.REQUIRED,
            command="echo hello",
        )
        candidate = _make_candidate()
        config = _default_config()

        result = await executor.execute(candidate, config)

        assert result.passed is True
        assert result.check_code == "CHECK-ECHO"
        assert result.duration_ms > 0
        assert "succeeded" in result.message

    @pytest.mark.asyncio
    async def test_execute_failing_command(self) -> None:
        """A failing command (non-zero exit) produces a failed result."""
        executor = SubprocessCheckExecutor(
            check_code="CHECK-FAIL",
            label="Fail test",
            severity=EnumCheckSeverity.REQUIRED,
            command="false",
        )
        candidate = _make_candidate()
        config = _default_config()

        result = await executor.execute(candidate, config)

        assert result.passed is False
        assert "failed" in result.message


# ============================================================================
# Risk Checks
# ============================================================================


class TestCheckRiskSensitivePaths:
    """Tests for CHECK-RISK-001: Sensitive paths detection."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckRiskSensitivePaths()
        assert check.check_code == "CHECK-RISK-001"
        assert check.severity == EnumCheckSeverity.REQUIRED

    @pytest.mark.asyncio
    async def test_no_sensitive_paths_passes(self) -> None:
        """No sensitive paths in changed_files results in PASS."""
        candidate = _make_candidate(
            changed_files=("src/utils/helper.py", "src/models/user.py")
        )
        result = await CheckRiskSensitivePaths().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_sensitive_paths_without_tags_fails(self) -> None:
        """Sensitive paths without risk tags results in FAIL."""
        candidate = _make_candidate(
            changed_files=("src/auth/login.py", "src/models/user.py"),
            risk_tags=(),
        )
        result = await CheckRiskSensitivePaths().execute(candidate, _default_config())
        assert result.passed is False
        assert "sensitive" in result.message.lower()

    @pytest.mark.asyncio
    async def test_sensitive_paths_with_security_tag_passes(self) -> None:
        """Sensitive paths with 'security' risk tag results in PASS."""
        candidate = _make_candidate(
            changed_files=("src/auth/login.py",),
            risk_tags=("security",),
        )
        result = await CheckRiskSensitivePaths().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_env_file_is_sensitive(self) -> None:
        """A .env file triggers sensitive path detection."""
        candidate = _make_candidate(
            changed_files=("config/.env.production",),
        )
        result = await CheckRiskSensitivePaths().execute(candidate, _default_config())
        assert result.passed is False

    def test_sensitive_patterns_are_valid_regex(self) -> None:
        """All sensitive path patterns compile as valid regexes."""
        import re

        for pattern in SENSITIVE_PATH_PATTERNS:
            compiled = re.compile(pattern)
            assert compiled is not None


class TestCheckRiskDiffSize:
    """Tests for CHECK-RISK-002: Diff size threshold."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckRiskDiffSize()
        assert check.check_code == "CHECK-RISK-002"
        assert check.severity == EnumCheckSeverity.RECOMMENDED

    @pytest.mark.asyncio
    async def test_within_threshold_passes(self) -> None:
        """Diff within threshold passes."""
        candidate = _make_candidate(
            changed_files=tuple(f"file_{i}.py" for i in range(10))
        )
        result = await CheckRiskDiffSize().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exceeds_threshold_fails(self) -> None:
        """Diff exceeding threshold fails."""
        candidate = _make_candidate(
            changed_files=tuple(f"file_{i}.py" for i in range(600))
        )
        result = await CheckRiskDiffSize().execute(candidate, _default_config())
        assert result.passed is False
        assert "exceeds" in result.message

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """Custom threshold is respected."""
        candidate = _make_candidate(
            changed_files=tuple(f"file_{i}.py" for i in range(5))
        )
        result = await CheckRiskDiffSize(threshold=3).execute(
            candidate, _default_config()
        )
        assert result.passed is False

    def test_default_threshold_value(self) -> None:
        """Default threshold matches the module constant."""
        assert DEFAULT_DIFF_SIZE_THRESHOLD == 500


class TestCheckRiskUnsafeOperations:
    """Tests for CHECK-RISK-003: Unsafe operations detector."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckRiskUnsafeOperations()
        assert check.check_code == "CHECK-RISK-003"
        assert check.severity == EnumCheckSeverity.REQUIRED

    @pytest.mark.asyncio
    async def test_no_python_files_passes(self) -> None:
        """No Python files means nothing to scan, passes."""
        candidate = _make_candidate(changed_files=("readme.md", "config.yaml"))
        result = await CheckRiskUnsafeOperations().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_safe_python_passes(self, tmp_path: Path) -> None:
        """Python file without unsafe patterns passes."""
        safe_file = tmp_path / "safe.py"
        safe_file.write_text("def hello():\n    return 'world'\n")

        candidate = _make_candidate(
            source_path=str(tmp_path),
            changed_files=("safe.py",),
        )
        result = await CheckRiskUnsafeOperations().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unsafe_import_pattern_fails(self, tmp_path: Path) -> None:
        """Python file with __import__() call fails."""
        unsafe_file = tmp_path / "unsafe.py"
        # Write test fixture with __import__ pattern for detector to find
        unsafe_file.write_text("mod = __import__('os')\n")

        candidate = _make_candidate(
            source_path=str(tmp_path),
            changed_files=("unsafe.py",),
        )
        result = await CheckRiskUnsafeOperations().execute(candidate, _default_config())
        assert result.passed is False
        assert "__import__" in result.message

    @pytest.mark.asyncio
    async def test_nonexistent_file_skipped(self) -> None:
        """A non-existent Python file in changed_files is gracefully skipped."""
        candidate = _make_candidate(
            source_path="/nonexistent/path",
            changed_files=("missing.py",),
        )
        result = await CheckRiskUnsafeOperations().execute(candidate, _default_config())
        # No crash, passes because nothing to scan
        assert result.passed is True

    def test_unsafe_patterns_are_valid_regex(self) -> None:
        """All unsafe patterns compile as valid regexes."""
        import re

        for pattern, _desc in UNSAFE_PATTERNS:
            compiled = re.compile(pattern)
            assert compiled is not None


# ============================================================================
# Measurement Checks
# ============================================================================


class TestCheckCostTokenDelta:
    """Tests for CHECK-COST-001: Token delta vs baseline."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckCostTokenDelta()
        assert check.check_code == "CHECK-COST-001"
        assert check.severity == EnumCheckSeverity.INFORMATIONAL

    @pytest.mark.asyncio
    async def test_always_passes(self) -> None:
        """Informational check always passes."""
        candidate = _make_candidate(changed_files=("a.py", "b.py"))
        result = await CheckCostTokenDelta().execute(candidate, _default_config())
        assert result.passed is True
        assert "tokens" in result.message.lower()

    @pytest.mark.asyncio
    async def test_with_baseline(self) -> None:
        """With a baseline, message includes delta."""
        candidate = _make_candidate(changed_files=("a.py",))
        result = await CheckCostTokenDelta(baseline_tokens=100).execute(
            candidate, _default_config()
        )
        assert result.passed is True
        assert "delta" in result.message.lower()

    @pytest.mark.asyncio
    async def test_without_baseline(self) -> None:
        """Without a baseline, message indicates no baseline."""
        candidate = _make_candidate()
        result = await CheckCostTokenDelta().execute(candidate, _default_config())
        assert result.passed is True
        assert "no baseline" in result.message.lower()


class TestCheckTimeWallClockDelta:
    """Tests for CHECK-TIME-001: Wall-clock delta vs baseline."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckTimeWallClockDelta()
        assert check.check_code == "CHECK-TIME-001"
        assert check.severity == EnumCheckSeverity.INFORMATIONAL

    @pytest.mark.asyncio
    async def test_always_passes(self) -> None:
        """Informational check always passes."""
        candidate = _make_candidate()
        result = await CheckTimeWallClockDelta().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_with_baseline(self) -> None:
        """With a baseline, message includes baseline reference."""
        candidate = _make_candidate()
        result = await CheckTimeWallClockDelta(baseline_ms=5000.0).execute(
            candidate, _default_config()
        )
        assert result.passed is True
        assert "baseline" in result.message.lower()


# ============================================================================
# Artifact / Replay Checks
# ============================================================================


class TestCheckReplaySanity:
    """Tests for CHECK-VAL-001: Deterministic replay sanity."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckReplaySanity()
        assert check.check_code == "CHECK-VAL-001"
        assert check.severity == EnumCheckSeverity.RECOMMENDED

    @pytest.mark.asyncio
    async def test_no_files_passes(self) -> None:
        """No changed files passes."""
        candidate = _make_candidate()
        result = await CheckReplaySanity().execute(candidate, _default_config())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_nondeterministic_pattern_still_passes(self, tmp_path: Path) -> None:
        """File with random module usage passes (RECOMMENDED, flags but does not block)."""
        rand_file = tmp_path / "rand_use.py"
        rand_file.write_text("import random\nval = random.randint(1, 10)\n")

        candidate = _make_candidate(
            source_path=str(tmp_path),
            changed_files=("rand_use.py",),
        )
        result = await CheckReplaySanity().execute(candidate, _default_config())
        # RECOMMENDED check -- passes but flags
        assert result.passed is True
        assert "non-deterministic" in result.message.lower()


class TestCheckArtifactCompleteness:
    """Tests for CHECK-VAL-002: Artifact completeness."""

    def test_check_properties(self) -> None:
        """Properties match the catalog definition."""
        check = CheckArtifactCompleteness()
        assert check.check_code == "CHECK-VAL-002"
        assert check.severity == EnumCheckSeverity.REQUIRED

    @pytest.mark.asyncio
    async def test_missing_artifact_dir_fails(self) -> None:
        """Missing artifact directory causes FAIL."""
        candidate = _make_candidate()
        check = CheckArtifactCompleteness(artifact_dir=Path("/nonexistent/artifacts"))
        result = await check.execute(candidate, _default_config())
        assert result.passed is False
        assert "does not exist" in result.message

    @pytest.mark.asyncio
    async def test_all_artifacts_present_passes(self, tmp_path: Path) -> None:
        """All required artifacts present causes PASS."""
        # Create required artifacts
        for name in REQUIRED_ARTIFACTS:
            (tmp_path / name).write_text("test")

        # Create expected artifacts
        for name in EXPECTED_ARTIFACTS:
            artifact_path = tmp_path / name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("test")

        candidate = _make_candidate()
        check = CheckArtifactCompleteness(artifact_dir=tmp_path)
        result = await check.execute(candidate, _default_config())
        assert result.passed is True
        assert "all required" in result.message.lower()

    @pytest.mark.asyncio
    async def test_missing_required_artifact_fails(self, tmp_path: Path) -> None:
        """Missing required artifact causes FAIL."""
        # Create only one of the required artifacts
        (tmp_path / REQUIRED_ARTIFACTS[0]).write_text("test")

        candidate = _make_candidate()
        check = CheckArtifactCompleteness(artifact_dir=tmp_path)
        result = await check.execute(candidate, _default_config())
        assert result.passed is False
        assert "missing required" in result.message.lower()

    @pytest.mark.asyncio
    async def test_missing_optional_still_passes(self, tmp_path: Path) -> None:
        """Missing optional artifacts still passes (with note)."""
        for name in REQUIRED_ARTIFACTS:
            (tmp_path / name).write_text("test")

        candidate = _make_candidate()
        check = CheckArtifactCompleteness(artifact_dir=tmp_path)
        result = await check.execute(candidate, _default_config())
        assert result.passed is True
        assert "missing optional" in result.message.lower()


# ============================================================================
# ModelCheckExecutorConfig
# ============================================================================


class TestModelCheckExecutorConfig:
    """Tests for ModelCheckExecutorConfig model."""

    def test_default_values(self) -> None:
        """Default config has sensible defaults."""
        config = ModelCheckExecutorConfig()
        assert config.working_dir == "."
        assert config.timeout_ms == 120_000.0
        assert config.env_overrides == ()

    def test_frozen(self) -> None:
        """Config is frozen (immutable)."""
        config = ModelCheckExecutorConfig()
        with pytest.raises(Exception):
            config.working_dir = "/other"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are forbidden."""
        with pytest.raises(Exception):
            ModelCheckExecutorConfig(unknown_field="x")  # type: ignore[call-arg]
