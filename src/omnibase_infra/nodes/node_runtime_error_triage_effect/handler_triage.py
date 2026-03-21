# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime error triage logic (OMN-5650).

Processes ModelRuntimeErrorEvent payloads from Kafka, performs action-level
dedup via Valkey, attempts auto-fixes for known categories, creates Linear
tickets for non-auto-fixable errors, and returns ModelRuntimeErrorTriageResult.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime

from omnibase_infra.enums.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.enum_triage_action import (
    EnumTriageAction,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.enum_triage_action_status import (
    EnumTriageActionStatus,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.model_triage_result import (
    ModelRuntimeErrorTriageResult,
)


class RuntimeErrorTriageDispatcher:
    """Dispatches triage actions for runtime container errors.

    Triage logic:
    1. Check action dedup via Valkey (24h TTL on fingerprint)
    2. Classify by error_category
    3. For auto-fixable categories, attempt fix with guardrails
    4. For non-auto-fixable, create Linear ticket
    5. Return ModelRuntimeErrorTriageResult
    """

    def __init__(
        self,
        valkey_url: str | None = None,
        linear_api_key: str | None = None,
        linear_team_id: str | None = None,
    ) -> None:
        # Valkey client uses a runtime-imported type not available at type-check
        # time. Using object avoids the Any-type gate.
        self._valkey: object = None
        self._linear_api_key = linear_api_key or os.environ.get("LINEAR_API_KEY", "")
        self._linear_team_id = linear_team_id or os.environ.get("LINEAR_TEAM_ID", "")
        self._environment = os.environ.get("ONEX_ENVIRONMENT", "local")
        self._init_valkey(valkey_url)

    def _init_valkey(self, valkey_url: str | None) -> None:
        """Initialise Valkey client for action-level dedup."""
        try:
            import redis

            host = os.environ.get("VALKEY_HOST", "localhost")
            port = int(os.environ.get("VALKEY_PORT", "16379"))
            password = os.environ.get("VALKEY_PASSWORD")
            self._valkey = redis.Redis(
                host=host,
                port=port,
                password=password,
                socket_timeout=5,
                decode_responses=True,
            )
        except (ImportError, Exception):  # noqa: BLE001
            pass

    def _is_deduped(self, fingerprint: str) -> tuple[bool, str]:
        """Check if this fingerprint has already been actioned within the TTL."""
        if self._valkey is None:
            return False, ""
        try:
            key = f"triage_action:{fingerprint}"
            existing = self._valkey.get(key)  # type: ignore[attr-defined]
            if existing:
                return (
                    True,
                    f"fingerprint already actioned within 24h (value={existing})",
                )
            return False, ""
        except Exception:  # noqa: BLE001
            return False, ""

    def _mark_actioned(self, fingerprint: str, action: str) -> None:
        """Mark this fingerprint as actioned in Valkey with 24h TTL."""
        if self._valkey is None:
            return
        try:
            key = f"triage_action:{fingerprint}"
            self._valkey.setex(key, 86400, action)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def triage_event(self, event: dict[str, object]) -> ModelRuntimeErrorTriageResult:
        """Triage a runtime error event.

        Args:
            event: Dict payload from Kafka (ModelRuntimeErrorEvent fields).

        Returns:
            ModelRuntimeErrorTriageResult with the action taken.
        """
        from uuid import UUID

        event_id = UUID(
            str(event.get("event_id", "00000000-0000-0000-0000-000000000000"))
        )
        fingerprint = str(event.get("fingerprint", ""))
        error_category = str(event.get("error_category", "UNKNOWN"))
        container = str(event.get("container", ""))
        severity = str(event.get("severity", "MEDIUM"))
        error_message = str(event.get("error_message", ""))
        now = datetime.now(UTC)

        # Step 1: Action-level dedup
        deduped, dedup_reason = self._is_deduped(fingerprint)
        if deduped:
            return ModelRuntimeErrorTriageResult(
                event_id=event_id,
                fingerprint=fingerprint,
                action=EnumTriageAction.DEDUPED,
                action_status=EnumTriageActionStatus.SUCCESS,
                triaged_at=now,
                dedup_reason=dedup_reason,
                error_category=EnumRuntimeErrorCategory(error_category),
                container=container,
                severity=severity,
            )

        # Step 2: Attempt auto-fix for known categories
        if error_category == "MISSING_TOPIC" and self._environment == "local":
            result = self._attempt_missing_topic_fix(event)
            if result is not None:
                self._mark_actioned(fingerprint, "AUTO_FIXED")
                return ModelRuntimeErrorTriageResult(
                    event_id=event_id,
                    fingerprint=fingerprint,
                    action=EnumTriageAction.AUTO_FIXED,
                    action_status=EnumTriageActionStatus.SUCCESS
                    if result["verified"]
                    else EnumTriageActionStatus.FAILED,
                    triaged_at=now,
                    auto_fix_type="rpk_topic_create",
                    auto_fix_command=result.get("command"),
                    auto_fix_result=result.get("output"),
                    auto_fix_verified=result["verified"],
                    error_category=EnumRuntimeErrorCategory(error_category),
                    container=container,
                    severity=severity,
                    operator_attention_required=not result["verified"],
                )

        # Step 3: Create Linear ticket for non-auto-fixable errors
        ticket_result = self._create_linear_ticket(event)
        self._mark_actioned(fingerprint, "TICKET_CREATED")

        return ModelRuntimeErrorTriageResult(
            event_id=event_id,
            fingerprint=fingerprint,
            action=EnumTriageAction.TICKET_CREATED,
            action_status=EnumTriageActionStatus.SUCCESS
            if ticket_result.get("created")
            else EnumTriageActionStatus.FAILED,
            triaged_at=now,
            ticket_id=ticket_result.get("ticket_id"),
            ticket_url=ticket_result.get("ticket_url"),
            error_category=EnumRuntimeErrorCategory(error_category),
            container=container,
            severity=severity,
            operator_attention_required=error_category == "UNKNOWN",
            notes=f"Auto-created from runtime error: {error_message[:200]}",
        )

    def _attempt_missing_topic_fix(
        self, event: dict[str, object]
    ) -> dict[str, object] | None:
        """Attempt to auto-fix a MISSING_TOPIC error by creating the topic.

        Only attempts if:
        - missing_topic_name is present
        - environment is local
        - rpk is available

        Returns dict with fix results, or None if fix not applicable.
        """
        topic_name = event.get("missing_topic_name")
        if not topic_name:
            return None

        import shutil

        rpk = shutil.which("rpk")
        if not rpk:
            return None

        cmd = [rpk, "topic", "create", str(topic_name), "-p", "6"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            command_str = " ".join(cmd)
            output = result.stdout + result.stderr

            verify_cmd = [rpk, "topic", "describe", str(topic_name)]
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            verified = verify_result.returncode == 0

            return {
                "command": command_str,
                "output": output.strip()[:500],
                "verified": verified,
            }
        except Exception:  # noqa: BLE001
            return None

    def _create_linear_ticket(self, event: dict[str, object]) -> dict[str, object]:
        """Create a Linear ticket for a non-auto-fixable error."""
        if not self._linear_api_key or not self._linear_team_id:
            return {
                "created": False,
                "ticket_id": None,
                "ticket_url": None,
                "reason": "LINEAR_API_KEY or LINEAR_TEAM_ID not configured",
            }

        error_category = event.get("error_category", "UNKNOWN")
        container = event.get("container", "unknown")
        error_message = str(event.get("error_message", ""))[:500]
        fingerprint = str(event.get("fingerprint", ""))[:16]
        severity = event.get("severity", "MEDIUM")

        priority_map = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4,
        }
        priority = priority_map.get(str(severity), 3)

        title = f"[Auto-triage] {error_category} in {container} (fp:{fingerprint}...)"
        description = (
            f"## Runtime Error Auto-Triage\n\n"
            f"**Category:** {error_category}\n"
            f"**Container:** {container}\n"
            f"**Severity:** {severity}\n"
            f"**Fingerprint:** {event.get('fingerprint', '')}\n\n"
            f"### Error Message\n```\n{error_message}\n```\n\n"
            f"### Raw Log Line\n```\n{str(event.get('raw_line', ''))[:1000]}\n```\n\n"
            f"---\n*Auto-created by NodeRuntimeErrorTriageEffect (OMN-5650)*"
        )

        import urllib.request

        query = """
        mutation IssueCreate($input: IssueCreateInput!) {
            issueCreate(input: $input) {
                success
                issue {
                    identifier
                    url
                }
            }
        }
        """
        variables = {
            "input": {
                "teamId": self._linear_team_id,
                "title": title[:200],
                "description": description,
                "priority": priority,
            }
        }

        try:
            payload = json.dumps({"query": query, "variables": variables}).encode(
                "utf-8"
            )
            req = urllib.request.Request(
                "https://api.linear.app/graphql",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._linear_api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                result = json.loads(resp.read())
                issue_data = result.get("data", {}).get("issueCreate", {})
                if issue_data.get("success"):
                    issue = issue_data.get("issue", {})
                    return {
                        "created": True,
                        "ticket_id": issue.get("identifier"),
                        "ticket_url": issue.get("url"),
                    }
                return {"created": False, "ticket_id": None, "ticket_url": None}
        except Exception:  # noqa: BLE001
            return {"created": False, "ticket_id": None, "ticket_url": None}


# Backwards-compatible alias for contract.yaml handler_routing reference
HandlerRuntimeErrorTriage = RuntimeErrorTriageDispatcher
