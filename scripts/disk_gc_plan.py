# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""disk_gc_plan.py — Pure, deterministic removal-plan resolver for disk-gc.sh (OMN-13008).

This module is the *only* place that decides what the .201 docker/disk GC removes.
`disk-gc.sh` collects docker inventory and pipes it in via env vars; this planner
emits a JSON plan; the shell only executes that plan. Keeping the decision logic
here (not in bash) makes the safety guarantees unit-testable without docker.

Inputs:
  KEEP_LIST  (env)   path to keep-list.yaml (small — env is fine)
  stdin      (JSON)  {"images_ndjson": "...", "ps_ndjson": "...", "inuse": "..."}
                     where each value is the raw multi-line output of the matching
                     `docker ... --format '{{json .}}'` command. Inventory is passed
                     on stdin (NOT env) because a host with many images blows past
                     ARG_MAX when the blobs are env vars (`Argument list too long`).

Output: a single JSON object on stdout:
  {
    "min_age_days": int,
    "remove_image_ids": [str, ...],
    "remove_container_ids": [str, ...],
    "kept_reasons": {image_id: reason, ...}   # for dry-run transparency
  }

Safety invariants (enforced + tested in tests/unit/scripts/test_disk_gc_plan.py):
  - never remove an image whose repo matches a keep_image_repos substring
  - never remove an image whose tag matches keep_image_tags exactly
  - never remove an image referenced by any container when protect_running is true
  - never remove anything younger than min_age_days
  - retain the newest superseded_image_keep_generations of each kept repo
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime, timezone
from typing import Any

import yaml

# docker prints CreatedAt like: "2026-05-29 14:03:11 -0400 EDT"
_CREATED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})")


def _parse_created_at(value: str, now: datetime) -> float:
    """Return age in days for a docker CreatedAt string. Unparseable → 0.0 (treat as new → keep)."""
    if not value:
        return 0.0
    m = _CREATED_RE.match(value.strip())
    if not m:
        return 0.0
    dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S %z")
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def _load_ndjson(blob: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in (blob or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _repo_protected(repo: str, keep_repos: list[str]) -> bool:
    return any(sub in repo for sub in keep_repos)


def build_plan(
    keep_list: dict[str, Any],
    images: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    inuse_refs: set[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure planner. See module docstring for invariants."""
    now = now or datetime.now(UTC)

    keep_repos: list[str] = keep_list.get("keep_image_repos", []) or []
    keep_tags: set[str] = set(keep_list.get("keep_image_tags", []) or [])
    protect_running: bool = bool(keep_list.get("protect_running", True))
    keep_generations: int = int(keep_list.get("superseded_image_keep_generations", 2))
    min_age_days: int = int(keep_list.get("min_age_days", 3))

    remove_image_ids: list[str] = []
    kept_reasons: dict[str, str] = {}

    # Group superseded candidates per repo so we can keep the N newest.
    superseded_by_repo: dict[str, list[tuple[float, str]]] = {}

    for img in images:
        image_id = img.get("ID", "") or img.get("Id", "")
        repo = img.get("Repository", "") or ""
        tag = img.get("Tag", "") or ""
        created = img.get("CreatedAt", "") or ""
        age = _parse_created_at(created, now)
        ref = f"{repo}:{tag}" if repo and tag and repo != "<none>" else image_id

        if age < min_age_days:
            kept_reasons[image_id] = (
                f"younger than min_age_days ({age:.1f}<{min_age_days})"
            )
            continue
        if tag in keep_tags and tag and tag != "<none>":
            kept_reasons[image_id] = f"tag '{tag}' in keep_image_tags"
            continue
        if protect_running and (ref in inuse_refs or image_id in inuse_refs):
            kept_reasons[image_id] = "referenced by a container (protect_running)"
            continue

        is_dangling = repo == "<none>" or repo == "" or tag == "<none>"

        if is_dangling:
            # Dangling + old + not in use → safe to remove.
            remove_image_ids.append(image_id)
            continue

        if _repo_protected(repo, keep_repos):
            # Kept repo: this is a superseded generation candidate (older than
            # min_age, not a keep_tag, not in use). Defer to per-repo retention.
            superseded_by_repo.setdefault(repo, []).append((age, image_id))
            continue

        # Untracked repo, old, not in use, not dangling. Conservative default:
        # KEEP. We only reap explicitly-tracked (keep_image_repos) superseded
        # generations and dangling images, never random third-party images.
        kept_reasons[image_id] = "repo not in keep_image_repos (conservative keep)"

    # Per-repo retention: keep the newest `keep_generations`, mark the rest for removal.
    for repo, entries in superseded_by_repo.items():
        # newest first (smallest age first)
        entries.sort(key=lambda e: e[0])
        for idx, (age, image_id) in enumerate(entries):
            if idx < keep_generations:
                kept_reasons[image_id] = (
                    f"within newest {keep_generations} generations of '{repo}'"
                )
            else:
                remove_image_ids.append(image_id)

    # Stopped containers older than min_age_days are removable.
    remove_container_ids: list[str] = []
    for c in containers:
        state = (c.get("State", "") or "").lower()
        status = (c.get("Status", "") or "").lower()
        created = c.get("CreatedAt", "") or ""
        cid = c.get("ID", "") or c.get("Id", "")
        is_stopped = state in {"exited", "dead", "created"} or status.startswith(
            "exited"
        )
        if not is_stopped:
            continue
        if _parse_created_at(created, now) < min_age_days:
            continue
        remove_container_ids.append(cid)

    # Reconcile: the SAME image id can surface in multiple `docker image ls` rows
    # (one per repo:tag). If any row produced a keep reason for an id, KEEP WINS —
    # a protected id must never be in the remove list even if another tag routed it
    # to removal. Also dedupe while preserving order.
    seen: set[str] = set()
    safe_remove: list[str] = []
    for image_id in remove_image_ids:
        if image_id in kept_reasons:
            continue
        if image_id in seen:
            continue
        seen.add(image_id)
        safe_remove.append(image_id)
    remove_image_ids = safe_remove

    return {
        "min_age_days": min_age_days,
        "remove_image_ids": remove_image_ids,
        "remove_container_ids": remove_container_ids,
        "kept_reasons": kept_reasons,
    }


def main() -> int:
    keep_list_path = os.environ["KEEP_LIST"]
    with open(keep_list_path, encoding="utf-8") as fh:
        keep_list = yaml.safe_load(fh) or {}

    raw = sys.stdin.read()
    payload = json.loads(raw) if raw.strip() else {}

    images = _load_ndjson(payload.get("images_ndjson", ""))
    containers = _load_ndjson(payload.get("ps_ndjson", ""))
    inuse = {
        line.strip() for line in payload.get("inuse", "").splitlines() if line.strip()
    }

    plan = build_plan(keep_list, images, containers, inuse)
    json.dump(plan, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
