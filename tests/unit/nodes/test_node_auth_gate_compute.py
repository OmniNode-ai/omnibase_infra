# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeAuthGateCompute and HandlerAuthGate.

This module validates the COMPUTE node and handler that evaluates a 10-step
authorization cascade. Tests cover every cascade step, emergency override
hardening, path glob matching, expiry, and the declarative node pattern.

Test Coverage per Ticket OMN-2125:
    1. Edit without auth -> denied (step 4)
    2. Edit outside scope -> denied (step 7)
    3. Emergency override without reason -> rejected (step 2)
    4. Emergency override with reason -> allowed + banner (step 2)
    5. 10min expires -> denied (step 9)
    6. Whitelisted paths bypass all checks (step 1)
    7. All 10 cascade steps exercised

Related:
    - OMN-2125: Auth Gate Nodes — Work Authorization Compute Node
    - OMN-2006: Auth Gate Nodes (parent)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_auth_decision import EnumAuthDecision
from omnibase_infra.enums.enum_auth_source import EnumAuthSource
from omnibase_infra.nodes.node_auth_gate_compute import (
    HandlerAuthGate,
    NodeAuthGateCompute,
    RegistryInfraAuthGateCompute,
)
from omnibase_infra.nodes.node_auth_gate_compute.models import (
    ModelAuthGateDecision,
    ModelAuthGateRequest,
    ModelContractWorkAuthorization,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Path Constants
# =============================================================================


def _get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent.parent


_PROJECT_ROOT = _get_project_root()
NODE_DIR = _PROJECT_ROOT / "src/omnibase_infra/nodes/node_auth_gate_compute"
CONTRACT_PATH = NODE_DIR / "contract.yaml"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a simple mock container for node initialization."""
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerAuthGate:
    """Create a HandlerAuthGate instance for testing."""
    return HandlerAuthGate(mock_container)


@pytest.fixture
def now() -> datetime:
    """Current UTC time for deterministic testing."""
    return datetime(2026, 2, 10, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def run_id() -> UUID:
    """A fixed run_id for tests."""
    return uuid4()


@pytest.fixture
def valid_auth(run_id: UUID, now: datetime) -> ModelContractWorkAuthorization:
    """Create a valid, non-expired authorization with common defaults."""
    return ModelContractWorkAuthorization(
        run_id=run_id,
        allowed_tools=("Edit", "Write"),
        allowed_paths=("src/**/*.py",),
        repo_scopes=("omnibase_infra",),
        source=EnumAuthSource.EXPLICIT,
        expires_at=now + timedelta(hours=4),
        reason="Test authorization",
    )


# =============================================================================
# TestStep1WhitelistedPaths
# =============================================================================


class TestStep1WhitelistedPaths:
    """Step 1: Whitelisted paths -> allow regardless of auth state."""

    def test_plan_file_allowed_without_auth(self, handler: HandlerAuthGate) -> None:
        """Plan files bypass all auth checks."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="/workspace/my_feature.plan.md",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 1
        assert decision.reason_code == "whitelisted_path"

    def test_memory_file_allowed_without_auth(self, handler: HandlerAuthGate) -> None:
        """Memory files bypass all auth checks."""
        request = ModelAuthGateRequest(
            tool_name="Write",
            target_path="/home/user/.claude/memory/notes.md",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 1

    def test_project_memory_allowed(self, handler: HandlerAuthGate) -> None:
        """Project-specific memory files bypass auth checks."""
        request = ModelAuthGateRequest(
            tool_name="Write",
            target_path="/home/user/.claude/projects/my-proj/memory/MEMORY.md",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 1

    def test_memory_md_allowed(self, handler: HandlerAuthGate) -> None:
        """MEMORY.md files bypass auth checks."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="/workspace/.claude/projects/test/memory/MEMORY.md",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 1

    def test_non_whitelisted_path_falls_through(self, handler: HandlerAuthGate) -> None:
        """Non-whitelisted paths proceed to step 2+."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
        )
        decision = handler.evaluate(request)

        assert decision.step != 1

    def test_empty_path_not_whitelisted(self, handler: HandlerAuthGate) -> None:
        """Empty path is not whitelisted."""
        request = ModelAuthGateRequest(
            tool_name="Bash",
            target_path="",
        )
        decision = handler.evaluate(request)

        assert decision.step != 1

    def test_whitelist_traversal_blocked(self, handler: HandlerAuthGate) -> None:
        """Path traversal through whitelisted dir is normalized and rejected."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="/a/.claude/memory/../../../../etc/passwd",
        )
        decision = handler.evaluate(request)

        # Should NOT be whitelisted after normalization resolves to /etc/passwd
        assert decision.step != 1


# =============================================================================
# TestStep2EmergencyOverride
# =============================================================================


class TestStep2EmergencyOverride:
    """Step 2: Emergency override hardening."""

    def test_emergency_override_without_reason_denied(
        self, handler: HandlerAuthGate
    ) -> None:
        """Emergency override without ONEX_UNSAFE_REASON -> denied."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            emergency_override_active=True,
            emergency_override_reason="",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 2
        assert decision.reason_code == "emergency_no_reason"
        assert "reason is required" in decision.reason.lower()

    def test_emergency_override_with_reason_soft_deny(
        self, handler: HandlerAuthGate
    ) -> None:
        """Emergency override with reason -> soft_deny + banner."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            emergency_override_active=True,
            emergency_override_reason="Hotfix for critical production bug",
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.SOFT_DENY
        assert decision.step == 2
        assert decision.reason_code == "emergency_override"
        assert decision.banner  # Banner must be non-empty
        assert "EMERGENCY OVERRIDE ACTIVE" in decision.banner
        assert "10 minutes" in decision.banner

    def test_emergency_override_banner_mentions_renewal(
        self, handler: HandlerAuthGate
    ) -> None:
        """Banner mentions that renewal requires manual /authorize."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            emergency_override_active=True,
            emergency_override_reason="Hotfix",
        )
        decision = handler.evaluate(request)

        assert "/authorize" in decision.banner

    def test_emergency_override_soft_deny_is_permitted(
        self, handler: HandlerAuthGate
    ) -> None:
        """Soft deny is still permitted (tool invocation proceeds)."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            emergency_override_active=True,
            emergency_override_reason="Hotfix",
        )
        decision = handler.evaluate(request)

        assert decision.decision.is_permitted()
        assert bool(decision) is True

    def test_no_emergency_override_falls_through(
        self, handler: HandlerAuthGate
    ) -> None:
        """Without emergency override, cascade continues to step 3+."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            emergency_override_active=False,
        )
        decision = handler.evaluate(request)

        assert decision.step >= 3


# =============================================================================
# TestStep3NoRunId
# =============================================================================


class TestStep3NoRunId:
    """Step 3: No run_id determinable -> deny."""

    def test_no_run_id_denied(self, handler: HandlerAuthGate) -> None:
        """No run_id -> denied."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=None,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 3
        assert decision.reason_code == "no_run_id"


# =============================================================================
# TestStep4NoAuthorization
# =============================================================================


class TestStep4NoAuthorization:
    """Step 4: No authorization contract found -> deny."""

    def test_no_authorization_denied(self, handler: HandlerAuthGate) -> None:
        """Edit without auth -> denied (ticket test case)."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=uuid4(),
            authorization=None,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 4
        assert decision.reason_code == "no_authorization"


# =============================================================================
# TestStep5RunIdMismatch
# =============================================================================


class TestStep5RunIdMismatch:
    """Step 5: Authorization run_id != request run_id -> deny."""

    def test_run_id_mismatch_denied(
        self,
        handler: HandlerAuthGate,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Run ID mismatch -> denied."""
        different_run_id = uuid4()
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=different_run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 5
        assert decision.reason_code == "run_id_mismatch"


# =============================================================================
# TestStep6ToolNotAllowed
# =============================================================================


class TestStep6ToolNotAllowed:
    """Step 6: Tool not in allowed_tools -> deny."""

    def test_tool_not_allowed_denied(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Bash not in allowed_tools -> denied."""
        request = ModelAuthGateRequest(
            tool_name="Bash",
            target_path="src/main.py",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 6
        assert decision.reason_code == "tool_not_allowed"

    def test_tool_name_case_sensitive(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Lowercase 'edit' is denied when only titlecase 'Edit' is allowed.

        Tool matching is case-sensitive: callers must use canonical tool
        names exactly as declared in allowed_tools.
        """
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit",),
            allowed_paths=("src/**/*.py",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="edit",  # lowercase — should NOT match "Edit"
            target_path="src/main.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 6
        assert decision.reason_code == "tool_not_allowed"

    def test_allowed_tool_passes(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Edit is in allowed_tools -> passes this step."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/foo.py",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.step >= 7 or decision.decision == EnumAuthDecision.ALLOW


# =============================================================================
# TestStep7PathNotAllowed
# =============================================================================


class TestStep7PathNotAllowed:
    """Step 7: Path not matching allowed_paths glob -> deny."""

    def test_path_outside_scope_denied(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Edit outside scope -> denied (ticket test case)."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="docs/README.md",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 7
        assert decision.reason_code == "path_not_allowed"

    def test_path_in_scope_passes(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Path matching glob passes."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/omnibase_infra/nodes/node.py",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.step >= 8 or decision.decision == EnumAuthDecision.ALLOW

    def test_single_star_does_not_match_nested_path(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Single * in allowed_paths must not match across /."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit",),
            allowed_paths=("src/*.py",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/deep/nested/secret.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 7
        assert decision.reason_code == "path_not_allowed"

    def test_empty_path_skips_check_for_non_file_tools(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Empty target_path skips path check for non-file tools (e.g., Bash)."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Bash",),
            allowed_paths=("src/**/*.py",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="Bash",
            target_path="",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        # Non-file tool with empty path should not be denied at step 7
        assert decision.step != 7 or decision.decision != EnumAuthDecision.DENY

    def test_empty_path_denied_for_file_targeting_tools(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """File-targeting tools with empty target_path are denied at step 7."""
        for tool_name in ("Edit", "Write", "Read", "NotebookEdit", "MultiEdit"):
            auth = ModelContractWorkAuthorization(
                run_id=run_id,
                allowed_tools=(tool_name,),
                allowed_paths=("src/**/*.py",),
                repo_scopes=(),
                source=EnumAuthSource.EXPLICIT,
                expires_at=now + timedelta(hours=4),
            )
            request = ModelAuthGateRequest(
                tool_name=tool_name,
                target_path="",
                run_id=run_id,
                authorization=auth,
                now=now,
            )
            decision = handler.evaluate(request)

            assert decision.decision == EnumAuthDecision.DENY, (
                f"Expected DENY for {tool_name} with empty path"
            )
            assert decision.step == 7, (
                f"Expected step 7 for {tool_name} with empty path"
            )
            assert decision.reason_code == "file_tool_missing_path", (
                f"Expected file_tool_missing_path for {tool_name}"
            )


# =============================================================================
# TestStep8RepoNotInScope
# =============================================================================


class TestStep8RepoNotInScope:
    """Step 8: Repo not in repo_scopes -> deny."""

    def test_repo_not_in_scope_denied(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Cross-repo outside scope -> denied."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            target_repo="other_repo",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 8
        assert decision.reason_code == "repo_not_in_scope"

    def test_repo_in_scope_passes(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Repo in scope passes."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            target_repo="omnibase_infra",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.step >= 9 or decision.decision == EnumAuthDecision.ALLOW

    def test_empty_target_repo_skips_check(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        valid_auth: ModelContractWorkAuthorization,
        now: datetime,
    ) -> None:
        """Empty target_repo skips repo scope check."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            target_repo="",
            run_id=run_id,
            authorization=valid_auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.step != 8 or decision.decision != EnumAuthDecision.DENY

    def test_empty_repo_scopes_skips_check(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Empty repo_scopes on auth means no repo restriction."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write"),
            allowed_paths=("src/**/*.py",),
            repo_scopes=(),  # No repo restriction
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            target_repo="any_repo",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        # Should pass step 8 (empty repo_scopes means no restriction)
        assert decision.step != 8 or decision.decision != EnumAuthDecision.DENY


# =============================================================================
# TestStep9AuthExpired
# =============================================================================


class TestStep9AuthExpired:
    """Step 9: Auth expired -> deny."""

    def test_expired_auth_denied(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """10min expires -> denied (ticket test case)."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write"),
            allowed_paths=("src/**/*.py",),
            repo_scopes=(),
            source=EnumAuthSource.EMERGENCY_OVERRIDE,
            expires_at=now - timedelta(minutes=1),  # Already expired
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 9
        assert decision.reason_code == "auth_expired"

    def test_auth_expiring_exactly_now_denied(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Auth expiring at exactly now -> denied (>= comparison)."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write"),
            allowed_paths=("**",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now,  # Exactly now
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.DENY
        assert decision.step == 9

    def test_not_expired_passes(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Auth not yet expired passes step 9."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write"),
            allowed_paths=("**",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.step == 10


# =============================================================================
# TestStep10AllChecksPassed
# =============================================================================


class TestStep10AllChecksPassed:
    """Step 10: All checks pass -> allow."""

    def test_fully_authorized_allowed(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Fully authorized request -> allowed at step 10."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write"),
            allowed_paths=("**",),
            repo_scopes=(),
            source=EnumAuthSource.EXPLICIT,
            expires_at=now + timedelta(hours=4),
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 10
        assert decision.reason_code == "all_checks_passed"
        assert bool(decision) is True

    def test_ticket_pipeline_auth_with_matching_files(
        self,
        handler: HandlerAuthGate,
        run_id: UUID,
        now: datetime,
    ) -> None:
        """Ticket pipeline auth with matching files -> allowed."""
        auth = ModelContractWorkAuthorization(
            run_id=run_id,
            allowed_tools=("Edit", "Write", "Bash"),
            allowed_paths=(
                "src/omnibase_infra/nodes/node_auth_gate_compute/*",
                "tests/unit/nodes/test_node_auth_gate_compute.py",
            ),
            repo_scopes=("omnibase_infra",),
            source=EnumAuthSource.TICKET_PIPELINE,
            expires_at=now + timedelta(hours=2),
        )
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/omnibase_infra/nodes/node_auth_gate_compute/node.py",
            target_repo="omnibase_infra",
            run_id=run_id,
            authorization=auth,
            now=now,
        )
        decision = handler.evaluate(request)

        assert decision.decision == EnumAuthDecision.ALLOW
        assert decision.step == 10


# =============================================================================
# TestModelAuthGateDecisionBool
# =============================================================================


class TestModelAuthGateDecisionBool:
    """Test custom __bool__ behavior on ModelAuthGateDecision."""

    def test_allow_is_truthy(self) -> None:
        """ALLOW decision is truthy."""
        decision = ModelAuthGateDecision(
            decision=EnumAuthDecision.ALLOW,
            step=10,
            reason="Allowed",
            reason_code="all_checks_passed",
        )
        assert bool(decision) is True

    def test_deny_is_falsy(self) -> None:
        """DENY decision is falsy."""
        decision = ModelAuthGateDecision(
            decision=EnumAuthDecision.DENY,
            step=3,
            reason="Denied",
            reason_code="no_run_id",
        )
        assert bool(decision) is False

    def test_soft_deny_is_truthy(self) -> None:
        """SOFT_DENY decision is truthy (tool proceeds with banner)."""
        decision = ModelAuthGateDecision(
            decision=EnumAuthDecision.SOFT_DENY,
            step=2,
            reason="Emergency override",
            reason_code="emergency_override",
            banner="EMERGENCY OVERRIDE ACTIVE",
        )
        assert bool(decision) is True


# =============================================================================
# TestContractWorkAuthorizationExpiry
# =============================================================================


class TestContractWorkAuthorizationExpiry:
    """Test authorization expiry logic."""

    def test_not_expired(self) -> None:
        """Non-expired auth returns False."""
        auth = ModelContractWorkAuthorization(
            run_id=uuid4(),
            allowed_tools=("Edit",),
            allowed_paths=("**",),
            source=EnumAuthSource.EXPLICIT,
            expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        )
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert auth.is_expired(now=now) is False

    def test_expired(self) -> None:
        """Expired auth returns True."""
        auth = ModelContractWorkAuthorization(
            run_id=uuid4(),
            allowed_tools=("Edit",),
            allowed_paths=("**",),
            source=EnumAuthSource.EXPLICIT,
            expires_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        now = datetime(2026, 6, 1, tzinfo=UTC)
        assert auth.is_expired(now=now) is True

    def test_is_frozen(self) -> None:
        """Authorization model is immutable."""
        from pydantic import ValidationError

        auth = ModelContractWorkAuthorization(
            run_id=uuid4(),
            allowed_tools=("Edit",),
            allowed_paths=("**",),
            source=EnumAuthSource.EXPLICIT,
            expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        )
        with pytest.raises((TypeError, ValidationError)):
            auth.reason = "changed"  # type: ignore[misc]

    def test_naive_datetime_rejected(self) -> None:
        """Timezone-naive expires_at is rejected by AwareDatetime."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="timezone"):
            ModelContractWorkAuthorization(
                run_id=uuid4(),
                allowed_tools=("Edit",),
                allowed_paths=("**",),
                source=EnumAuthSource.EXPLICIT,
                expires_at=datetime(2026, 12, 31),  # No tzinfo
            )


# =============================================================================
# TestEnumAuthDecision
# =============================================================================


class TestEnumAuthDecision:
    """Test EnumAuthDecision helper methods."""

    def test_allow_is_permitted(self) -> None:
        assert EnumAuthDecision.ALLOW.is_permitted() is True

    def test_deny_is_not_permitted(self) -> None:
        assert EnumAuthDecision.DENY.is_permitted() is False

    def test_soft_deny_is_permitted(self) -> None:
        assert EnumAuthDecision.SOFT_DENY.is_permitted() is True

    def test_str_serialization(self) -> None:
        assert str(EnumAuthDecision.ALLOW) == "allow"
        assert str(EnumAuthDecision.DENY) == "deny"
        assert str(EnumAuthDecision.SOFT_DENY) == "soft_deny"


# =============================================================================
# TestHandlerExecute
# =============================================================================


class TestHandlerExecute:
    """Test the execute() envelope interface."""

    @pytest.mark.anyio
    async def test_execute_with_dict_payload(self, handler: HandlerAuthGate) -> None:
        """execute() accepts dict payload and returns ModelHandlerOutput."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
        )
        envelope: dict[str, object] = {
            "operation": "auth_gate.evaluate",
            "payload": request.model_dump(mode="json"),
            "correlation_id": str(uuid4()),
        }
        result = await handler.execute(envelope)

        assert result.result is not None
        assert isinstance(result.result, ModelAuthGateDecision)

    @pytest.mark.anyio
    async def test_execute_with_model_payload(self, handler: HandlerAuthGate) -> None:
        """execute() accepts model instance payload."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="/workspace/my_feature.plan.md",
        )
        envelope: dict[str, object] = {
            "operation": "auth_gate.evaluate",
            "payload": request,
        }
        result = await handler.execute(envelope)

        assert result.result is not None
        assert result.result.decision == EnumAuthDecision.ALLOW
        assert result.result.step == 1

    @pytest.mark.anyio
    async def test_execute_missing_payload_raises(
        self, handler: HandlerAuthGate
    ) -> None:
        """execute() raises ValueError when envelope has no payload."""
        envelope: dict[str, object] = {
            "operation": "auth_gate.evaluate",
        }
        with pytest.raises(ValueError, match="missing required 'payload'"):
            await handler.execute(envelope)

    @pytest.mark.anyio
    async def test_execute_invalid_correlation_id_falls_back(
        self, handler: HandlerAuthGate
    ) -> None:
        """execute() falls back to uuid4() for invalid correlation_id."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="/workspace/my_feature.plan.md",
        )
        envelope: dict[str, object] = {
            "operation": "auth_gate.evaluate",
            "payload": request,
            "correlation_id": "not-a-uuid",
        }
        result = await handler.execute(envelope)

        assert result.result is not None
        assert result.correlation_id is not None

    @pytest.mark.anyio
    async def test_execute_wrong_operation_raises(
        self, handler: HandlerAuthGate
    ) -> None:
        """execute() raises ValueError for unsupported operation."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
        )
        envelope: dict[str, object] = {
            "operation": "auth_gate.unknown",
            "payload": request,
        }
        with pytest.raises(ValueError, match="Unsupported operation"):
            await handler.execute(envelope)

    @pytest.mark.anyio
    async def test_execute_missing_operation_raises(
        self, handler: HandlerAuthGate
    ) -> None:
        """execute() raises ValueError when envelope has no operation."""
        request = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/main.py",
        )
        envelope: dict[str, object] = {
            "payload": request,
        }
        with pytest.raises(ValueError, match="Unsupported operation"):
            await handler.execute(envelope)

    @pytest.mark.anyio
    async def test_execute_invalid_payload_wraps_validation_error(
        self, handler: HandlerAuthGate
    ) -> None:
        """execute() wraps ValidationError as ValueError for malformed payloads."""
        envelope: dict[str, object] = {
            "operation": "auth_gate.evaluate",
            "payload": {"tool_name": 123},  # wrong type triggers ValidationError
        }
        with pytest.raises(ValueError, match="validation error"):
            await handler.execute(envelope)


# =============================================================================
# TestGlobToRegexEdgeCases
# =============================================================================


class TestGlobToRegexEdgeCases:
    """Edge case tests for _glob_to_regex in security-sensitive path matching."""

    def test_leading_double_star(self, handler: HandlerAuthGate) -> None:
        """Leading **/ matches any path prefix."""
        pattern = handler._glob_to_regex("**/foo.py")
        assert pattern.match("foo.py")
        assert pattern.match("a/foo.py")
        assert pattern.match("a/b/c/foo.py")
        assert not pattern.match("foo.pyc")

    def test_trailing_double_star(self, handler: HandlerAuthGate) -> None:
        """Trailing ** matches any path suffix."""
        pattern = handler._glob_to_regex("src/**")
        assert pattern.match("src/")
        assert pattern.match("src/foo.py")
        assert pattern.match("src/a/b/c.py")

    def test_multiple_double_star_segments(self, handler: HandlerAuthGate) -> None:
        """Multiple **/ segments in a single pattern."""
        pattern = handler._glob_to_regex("src/**/tests/**/*.py")
        assert pattern.match("src/tests/test_main.py")
        assert pattern.match("src/a/b/tests/c/d/test_main.py")
        assert not pattern.match("src/tests/test_main.txt")

    def test_regex_metacharacters_in_path(self, handler: HandlerAuthGate) -> None:
        """Paths with regex metacharacters are escaped correctly."""
        pattern = handler._glob_to_regex("src/file.name+extra(1).py")
        assert pattern.match("src/file.name+extra(1).py")
        # Should NOT match without the literal dot/parens
        assert not pattern.match("src/fileXname+extra(1)Xpy")

    def test_brackets_in_pattern(self, handler: HandlerAuthGate) -> None:
        """Square brackets in path are escaped, not treated as regex class."""
        pattern = handler._glob_to_regex("src/[backup]/file.py")
        assert pattern.match("src/[backup]/file.py")
        assert not pattern.match("src/b/file.py")

    def test_single_star_does_not_cross_slash(self, handler: HandlerAuthGate) -> None:
        """Single * does not match across directory separators."""
        pattern = handler._glob_to_regex("src/*.py")
        assert pattern.match("src/foo.py")
        assert not pattern.match("src/sub/foo.py")

    def test_path_traversal_normalized(self, handler: HandlerAuthGate) -> None:
        """Path with .. segments is normalized before matching."""
        assert not handler._path_matches_globs("src/../../etc/passwd", ("src/**",))

    def test_question_mark_matches_single_non_slash(
        self, handler: HandlerAuthGate
    ) -> None:
        """? matches exactly one non-/ character."""
        pattern = handler._glob_to_regex("src/?.py")
        assert pattern.match("src/a.py")
        assert not pattern.match("src/ab.py")
        assert not pattern.match("src//.py")

    def test_too_many_double_star_segments_raises(
        self, handler: HandlerAuthGate
    ) -> None:
        """Patterns with more than 2 ** segments raise ValueError (ReDoS guard)."""
        # 3 ** segments exceeds _MAX_DOUBLE_STAR_SEGMENTS=2
        with pytest.raises(ValueError, match="exceeding maximum"):
            handler._glob_to_regex("a/**/b/**/c/**")

    def test_null_byte_rejected_in_path_matches_globs(
        self, handler: HandlerAuthGate
    ) -> None:
        """Paths containing null bytes are rejected by _path_matches_globs."""
        assert not handler._path_matches_globs("src/main.py\x00evil", ("src/**",))

    def test_null_byte_rejected_in_is_whitelisted_path(
        self, handler: HandlerAuthGate
    ) -> None:
        """Paths containing null bytes are rejected by _is_whitelisted_path.

        A path like "src/main.py\\x00.plan.md" would pass fnmatch (matching
        *.plan.md) but the OS truncates at the null byte, effectively
        granting access to "src/main.py" under a whitelisted pattern.
        """
        assert not handler._is_whitelisted_path("src/main.py\x00.plan.md")

    def test_path_exceeding_path_max_rejected(self, handler: HandlerAuthGate) -> None:
        """Paths exceeding PATH_MAX (4096) characters are rejected."""
        long_path = "a/" * 3000  # 6000 characters, well over 4096
        assert not handler._path_matches_globs(long_path, ("**",))

    def test_deeply_nested_plan_file_rejected_by_whitelist_depth(
        self, handler: HandlerAuthGate
    ) -> None:
        """Paths with >10 '/' separators are rejected even if they match a whitelist pattern.

        _MAX_WHITELIST_DEPTH (10) prevents abuse via deeply nested paths
        that exploit the permissive fnmatch * (which matches across /).
        """
        # 12 slashes — exceeds _MAX_WHITELIST_DEPTH=10
        deep_path = "a/b/c/d/e/f/g/h/i/j/k/l/evil.plan.md"
        assert not handler._is_whitelisted_path(deep_path)

    def test_plan_file_within_whitelist_depth_allowed(
        self, handler: HandlerAuthGate
    ) -> None:
        """Paths within depth limit that match whitelist ARE whitelisted."""
        # 3 slashes — well within _MAX_WHITELIST_DEPTH=10
        shallow_path = "workspace/feature/spec.plan.md"
        assert handler._is_whitelisted_path(shallow_path)


# =============================================================================
# TestNodeDeclarativePattern
# =============================================================================


class TestNodeDeclarativePattern:
    """Test that the node follows the declarative pattern."""

    def test_extends_node_compute(self) -> None:
        """Node extends NodeCompute base class."""
        from omnibase_core.nodes.node_compute import NodeCompute

        assert issubclass(NodeAuthGateCompute, NodeCompute)

    def test_no_custom_logic(self, mock_container: MagicMock) -> None:
        """Node has no custom methods beyond base class."""
        node = NodeAuthGateCompute(mock_container)
        assert not hasattr(node, "evaluate")
        assert not hasattr(node, "_is_whitelisted_path")


# =============================================================================
# TestRegistryInfraAuthGateCompute
# =============================================================================


class TestRegistryInfraAuthGateCompute:
    """Test registry factory methods."""

    def test_get_node_class(self) -> None:
        """get_node_class returns the node class."""
        cls = RegistryInfraAuthGateCompute.get_node_class()
        assert cls is NodeAuthGateCompute

    def test_create_node(self, mock_container: MagicMock) -> None:
        """create_node creates instance."""
        node = RegistryInfraAuthGateCompute.create_node(mock_container)
        assert isinstance(node, NodeAuthGateCompute)


# =============================================================================
# TestContractValidation
# =============================================================================


class TestContractValidation:
    """Test contract.yaml configuration."""

    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        """Load contract.yaml data."""
        if not CONTRACT_PATH.exists():
            pytest.skip(f"Contract file not found: {CONTRACT_PATH}")
        with open(CONTRACT_PATH) as f:
            data: dict = yaml.safe_load(f)
        return data

    def test_node_type_is_compute_generic(self, contract_data: dict) -> None:
        """Node type must be COMPUTE_GENERIC."""
        assert contract_data.get("node_type") == "COMPUTE_GENERIC"

    def test_contract_version_valid(self, contract_data: dict) -> None:
        """Contract version follows semver object structure."""
        cv = contract_data.get("contract_version", {})
        assert isinstance(cv, dict)
        assert isinstance(cv.get("major"), int)
        assert isinstance(cv.get("minor"), int)
        assert isinstance(cv.get("patch"), int)

    def test_handler_routing_configured(self, contract_data: dict) -> None:
        """Handler routing is configured for auth gate."""
        hr = contract_data.get("handler_routing", {})
        assert hr.get("routing_strategy") == "operation_match"
        handlers = hr.get("handlers", [])
        assert len(handlers) >= 1
        assert handlers[0].get("handler_type") == "auth_gate"
        assert "auth_gate.evaluate" in handlers[0].get("supported_operations", [])

    def test_input_model_configured(self, contract_data: dict) -> None:
        """Input model is ModelAuthGateRequest."""
        im = contract_data.get("input_model", {})
        assert im.get("name") == "ModelAuthGateRequest"

    def test_output_model_configured(self, contract_data: dict) -> None:
        """Output model is ModelAuthGateDecision."""
        om = contract_data.get("output_model", {})
        assert om.get("name") == "ModelAuthGateDecision"


# =============================================================================
# TestHandlerProperties
# =============================================================================


class TestHandlerProperties:
    """Test handler lifecycle and properties."""

    def test_handler_type(self, handler: HandlerAuthGate) -> None:
        assert handler.handler_type == EnumHandlerType.COMPUTE_HANDLER

    def test_handler_category(self, handler: HandlerAuthGate) -> None:
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE

    @pytest.mark.anyio
    async def test_initialize_sets_flag(self, handler: HandlerAuthGate) -> None:
        assert handler._initialized is False
        await handler.initialize({})
        assert handler._initialized is True

    @pytest.mark.anyio
    async def test_shutdown_clears_flag(self, handler: HandlerAuthGate) -> None:
        await handler.initialize({})
        await handler.shutdown()
        assert handler._initialized is False


# =============================================================================
# TestModelAuthGateRequestSanitization
# =============================================================================


class TestModelAuthGateRequestSanitization:
    """Test model-level control character sanitization on target_path/target_repo."""

    def test_control_chars_stripped_from_target_path(self) -> None:
        """Control characters are stripped from target_path at model boundary."""
        req = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/\x00main\x1f.py",
        )
        assert req.target_path == "src/main.py"

    def test_control_chars_stripped_from_target_repo(self) -> None:
        """Control characters are stripped from target_repo at model boundary."""
        req = ModelAuthGateRequest(
            tool_name="Edit",
            target_repo="my\x07repo\x1b",
        )
        assert req.target_repo == "myrepo"

    def test_unicode_zero_width_stripped(self) -> None:
        """Unicode zero-width and formatting chars are stripped."""
        req = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/\u200bmain\ufeff.py",
        )
        assert req.target_path == "src/main.py"

    def test_control_chars_stripped_from_emergency_override_reason(self) -> None:
        """Control characters are stripped from emergency_override_reason at model boundary."""
        req = ModelAuthGateRequest(
            tool_name="Edit",
            emergency_override_active=True,
            emergency_override_reason="Hotfix\x00for\x1fbug",
        )
        assert req.emergency_override_reason == "Hotfixforbug"

    def test_clean_values_unchanged(self) -> None:
        """Clean values pass through unchanged."""
        req = ModelAuthGateRequest(
            tool_name="Edit",
            target_path="src/utils/helper.py",
            target_repo="omnibase_infra",
        )
        assert req.target_path == "src/utils/helper.py"
        assert req.target_repo == "omnibase_infra"

    def test_target_path_max_length_enforced(self) -> None:
        """target_path exceeding max_length (8192) is rejected at model boundary."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelAuthGateRequest(
                tool_name="Edit",
                target_path="a" * 8193,
            )

    def test_emergency_override_reason_max_length_enforced(self) -> None:
        """emergency_override_reason exceeding max_length (1000) is rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelAuthGateRequest(
                tool_name="Edit",
                emergency_override_active=True,
                emergency_override_reason="x" * 1001,
            )

    def test_tool_name_max_length_enforced(self) -> None:
        """tool_name exceeding max_length (200) is rejected at model boundary."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelAuthGateRequest(tool_name="A" * 201)

    def test_target_repo_max_length_enforced(self) -> None:
        """target_repo exceeding max_length (500) is rejected at model boundary."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelAuthGateRequest(
                tool_name="Edit",
                target_repo="r" * 501,
            )
