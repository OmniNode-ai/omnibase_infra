# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""File-backed stub implementation of ProtocolProjectTracker.

Used when LINEAR_API_KEY / LINEAR_TOKEN is not set. Satisfies the full
protocol shape with JSON-backed persistence under state_root/project_tracker_stub.json.
Not intended to prove full Linear behavioral equivalence — protocol-shape
substitute for offline/local operation only.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.adapters.project_tracker.model_stub_comment import ModelStubComment
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue
from omnibase_infra.adapters.project_tracker.model_stub_project import ModelStubProject


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StubHealthStatus:
    """Minimal health status for local stub — satisfies ProtocolServiceHealthStatus shape."""

    def __init__(self) -> None:
        self.service_id = "local-stub-project-tracker"
        self.status = "healthy"
        self.diagnostics: dict[str, str] = {}


class LocalStubProjectTracker:
    """JSON-backed local stub satisfying ProtocolProjectTracker.

    State stored in {state_root}/project_tracker_stub.json.
    Issue identifiers are STUB-{counter} (monotonically incrementing).
    Writes are crash-consistent via atomic rename.
    """

    def __init__(self, state_root: Path | None = None) -> None:
        if state_root is None:
            state_root = Path.home() / ".onex_state" / "local-tracker"
        self._state_root = Path(state_root)
        self._state_file = self._state_root / "project_tracker_stub.json"
        self._lock = threading.Lock()
        self._connected = False

    # -- internal persistence --

    def _load(self) -> dict[str, object]:
        if not self._state_file.exists():
            return {"issues": {}, "projects": {}, "comments": {}, "counter": 0}
        data: dict[str, object] = json.loads(
            self._state_file.read_text(encoding="utf-8")
        )
        return data

    def _save(self, state: dict[str, object]) -> None:
        self._state_root.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self._state_file)

    def _issue_from_dict(self, d: dict[str, object]) -> ModelStubIssue:
        raw_labels = d.get("labels", [])
        labels: list[str] = (
            [str(x) for x in raw_labels] if isinstance(raw_labels, list) else []
        )
        return ModelStubIssue(
            id=str(d["id"]),
            identifier=str(d["identifier"]),
            title=str(d["title"]),
            description=str(d["description"])
            if d.get("description") is not None
            else None,
            state=str(d.get("state", "todo")),
            priority=str(d["priority"]) if d.get("priority") is not None else None,
            assignee=str(d["assignee"]) if d.get("assignee") is not None else None,
            labels=labels,
            team=str(d["team"]) if d.get("team") is not None else None,
            project_id=str(d["project_id"])
            if d.get("project_id") is not None
            else None,
            url=str(d["url"]) if d.get("url") is not None else None,
            created_at=datetime.fromisoformat(str(d["created_at"])),
            updated_at=datetime.fromisoformat(str(d["updated_at"])),
        )

    def _comment_from_dict(self, d: dict[str, object]) -> ModelStubComment:
        return ModelStubComment(
            id=str(d["id"]),
            body=str(d["body"]),
            author=str(d.get("author", "stub")),
            created_at=datetime.fromisoformat(str(d["created_at"])),
        )

    def _project_from_dict(self, d: dict[str, object]) -> ModelStubProject:
        raw_progress = d.get("progress", 0.0)
        progress = (
            float(raw_progress) if isinstance(raw_progress, (int, float)) else 0.0
        )
        return ModelStubProject(
            id=str(d["id"]),
            name=str(d["name"]),
            description=str(d["description"])
            if d.get("description") is not None
            else None,
            state=str(d["state"]) if d.get("state") is not None else None,
            progress=progress,
            url=str(d["url"]) if d.get("url") is not None else None,
        )

    # -- ProtocolProjectTracker lifecycle --

    async def connect(self) -> bool:
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._connected = True
        return True

    async def health_check(self) -> StubHealthStatus:
        return StubHealthStatus()

    async def get_capabilities(self) -> list[str]:
        return ["read", "write"]

    async def close(self, timeout_seconds: float = 30.0) -> None:
        self._connected = False

    # -- Domain operations --

    async def list_issues(
        self, filters: dict[str, str] | None = None, limit: int = 50
    ) -> list[ModelStubIssue]:
        with self._lock:
            state = self._load()
        issues_map = state.get("issues", {})
        issues = [
            self._issue_from_dict(v)
            for v in (issues_map.values() if isinstance(issues_map, dict) else [])
            if isinstance(v, dict)
        ]
        if filters:
            for key, val in filters.items():
                issues = [i for i in issues if getattr(i, key, None) == val]
        return issues[:limit]

    async def get_issue(self, issue_id: str) -> ModelStubIssue:
        with self._lock:
            state = self._load()
        issues_map = state.get("issues", {})
        for d in issues_map.values() if isinstance(issues_map, dict) else []:
            if not isinstance(d, dict):
                continue
            if d.get("id") == issue_id or d.get("identifier") == issue_id:
                return self._issue_from_dict(d)
        raise KeyError(f"Issue not found: {issue_id}")

    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        team: str | None = None,
    ) -> ModelStubIssue:
        with self._lock:
            state = self._load()
            raw_counter = state.get("counter", 0)
            counter = (
                int(raw_counter) if isinstance(raw_counter, (int, float, str)) else 0
            ) + 1
            state["counter"] = counter
            issue_id = str(uuid.uuid4())
            identifier = f"STUB-{counter}"
            now = _now_iso()
            row: dict[str, object] = {
                "id": issue_id,
                "identifier": identifier,
                "title": title,
                "description": description,
                "state": "todo",
                "priority": priority,
                "assignee": assignee,
                "labels": labels or [],
                "team": team,
                "project_id": None,
                "url": None,
                "created_at": now,
                "updated_at": now,
            }
            issues_map = state.get("issues", {})
            if isinstance(issues_map, dict):
                issues_map[issue_id] = row
            state["issues"] = issues_map
            self._save(state)
        return self._issue_from_dict(row)

    async def update_issue(
        self, issue_id: str, updates: dict[str, str]
    ) -> ModelStubIssue:
        with self._lock:
            state = self._load()
            issues_map = state.get("issues", {})
            target_key: str | None = None
            for k, d in issues_map.items() if isinstance(issues_map, dict) else []:
                if isinstance(d, dict) and (
                    d.get("id") == issue_id or d.get("identifier") == issue_id
                ):
                    target_key = k
                    break
            if target_key is None:
                raise KeyError(f"Issue not found: {issue_id}")
            if not isinstance(issues_map, dict):
                raise KeyError(f"Issue not found: {issue_id}")
            existing = issues_map[target_key]
            row: dict[str, object] = (
                dict(existing) if isinstance(existing, dict) else {}
            )
            row.update(updates)
            row["updated_at"] = _now_iso()
            issues_map[target_key] = row
            self._save(state)
        return self._issue_from_dict(row)

    async def search_issues(self, query: str, limit: int = 50) -> list[ModelStubIssue]:
        with self._lock:
            state = self._load()
        q = query.lower()
        issues_map = state.get("issues", {})
        results = [
            self._issue_from_dict(d)
            for d in (issues_map.values() if isinstance(issues_map, dict) else [])
            if isinstance(d, dict)
            and (
                q in str(d.get("title", "")).lower()
                or q in str(d.get("description") or "").lower()
            )
        ]
        return results[:limit]

    async def add_comment(self, issue_id: str, body: str) -> ModelStubComment:
        with self._lock:
            state = self._load()
            issues_map = state.get("issues", {})
            found = any(
                isinstance(d, dict)
                and (d.get("id") == issue_id or d.get("identifier") == issue_id)
                for d in (issues_map.values() if isinstance(issues_map, dict) else [])
            )
            if not found:
                raise KeyError(f"Issue not found: {issue_id}")
            comment_id = str(uuid.uuid4())
            now = _now_iso()
            cd: dict[str, object] = {
                "id": comment_id,
                "issue_id": issue_id,
                "body": body,
                "author": "stub",
                "created_at": now,
            }
            comments_map = state.get("comments", {})
            if isinstance(comments_map, dict):
                comments_map[comment_id] = cd
            state["comments"] = comments_map
            self._save(state)
        return self._comment_from_dict(cd)

    async def get_project(self, project_id: str) -> ModelStubProject:
        with self._lock:
            state = self._load()
        projects_map = state.get("projects", {})
        if isinstance(projects_map, dict) and project_id in projects_map:
            proj = projects_map[project_id]
            if isinstance(proj, dict):
                return self._project_from_dict(proj)
        raise KeyError(f"Project not found: {project_id}")

    async def list_projects(self, limit: int = 50) -> list[ModelStubProject]:
        with self._lock:
            state = self._load()
        projects_map = state.get("projects", {})
        return [
            self._project_from_dict(v)
            for v in list(
                projects_map.values() if isinstance(projects_map, dict) else []
            )[:limit]
            if isinstance(v, dict)
        ]
