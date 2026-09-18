# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""disk_gc_plan.py — Pure, deterministic removal-plan resolver for disk-gc.sh (OMN-13008).

This module is the *only* place that decides what the .201 docker/disk GC removes.
`disk-gc.sh` collects docker inventory and pipes it in via env vars; this planner
emits a JSON plan; the shell only executes that plan. Keeping the decision logic
here (not in bash) makes the safety guarantees unit-testable without docker.

Inputs:
  KEEP_LIST       (env)   path to keep-list.yaml (small — env is fine)
  GITHUB_TOKEN    (env)   optional; enables PR-state lookup for disposable CI tags
  GITHUB_REPO     (env)   optional; "owner/repo" for the PR-state lookup
  stdin           (JSON)  {"images_ndjson": "...", "ps_ndjson": "...", "inuse": "..."}
                          where each value is the raw multi-line output of the matching
                          `docker ... --format '{{json .}}'` command. Inventory is passed
                          on stdin (NOT env) because a host with many images blows past
                          ARG_MAX when the blobs are env vars (`Argument list too long`).

Output: a single JSON object on stdout:
  {
    "min_age_days": int,
    "builder_cache_max_size": str,   # OMN-16367: docker-parseable size string
                                     # (e.g. "100GB") passed to
                                     # `docker builder prune --max-used-space`.
                                     # A CEILING on retained build cache. The
                                     # previous age-only bound could not shrink
                                     # the cache at all: `until=` filters on LAST
                                     # ACCESS, and every large record is refreshed
                                     # by the next build's cache hit.
    "remove_image_ids": [str, ...],
    "remove_image_refs": {image_id: [repo:tag, ...]},  # OMN-15804: every tag
                                                         # reference an id carries,
                                                         # so the executor can
                                                         # untag-then-remove a
                                                         # multi-tag image instead
                                                         # of `docker rmi <id>`,
                                                         # which docker refuses
                                                         # ("referenced in
                                                         # multiple repositories")
                                                         # once >1 repo:tag point
                                                         # at the same id. Empty
                                                         # list = dangling (no tag
                                                         # to untag), remove by id.
    "remove_container_ids": [str, ...],
    "kept_reasons": {image_id: reason, ...}   # for dry-run transparency
  }

Safety invariants (enforced + tested in tests/unit/scripts/test_disk_gc_plan.py):
  - never remove an image whose repo matches a keep_image_repos substring
  - never remove an image whose tag matches keep_image_tags exactly
  - never remove an image referenced by any container when protect_running is true
  - never remove anything younger than min_age_days
  - retain the newest superseded_image_keep_generations of each kept repo

In-use matching (OMN-15804): `inuse_refs` entries can be a repo:tag string OR an
image ID, and image IDs can appear TRUNCATED (12 hex chars, as printed by
`docker ps --format '{{.Image}}'` when a container was started directly by ID)
or FULL (`sha256:<64hex>`, as printed by `docker image ls --no-trunc`). Exact
string equality misses the truncated-vs-full case, so matching also compares
the normalized 12-hex-char short form of both sides.

PR-state reaping invariants (OMN-13225, tested in test_disk_gc_pr_state.py):
  - ghcr CI tags pr-<N> / sha-* are DISPOSABLE and bypassed the age/generation window
    ONLY when their associated PR is merged or closed (via pr_state_lookup)
  - protect_running, keep_image_tags, and keep_image_repos always take precedence —
    a running-lane image is NEVER removed even if its PR is merged
  - when pr_state_lookup is None (no GitHub token), the PR-state stage is skipped
    entirely (safe default: keep all disposable tags, fall through to age/generation logic)
  - any lookup error for a tag → SKIP_AMBIGUITY (keep the image; default safe)
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import yaml

# docker prints CreatedAt like: "2026-05-29 14:03:11 -0400 EDT"
_CREATED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})")

# Tags that indicate a disposable CI artifact (pr-<N> or sha-<hex>).
# These are CI build artifacts, NOT rollback targets. They are eligible for
# PR-state reaping independent of age / generation retention.
_PR_TAG_RE = re.compile(r"^pr-(\d+)$")
_SHA_TAG_RE = re.compile(r"^sha-[0-9a-f]{6,40}$")

# Sentinel values returned by pr_state_lookup
PR_STATE_MERGED = "merged"
PR_STATE_CLOSED = "closed"
PR_STATE_OPEN = "open"
PR_STATE_UNKNOWN = "unknown"  # lookup error or PR number not found → keep

# OMN-16367: default ceiling on retained BuildKit cache, as a docker-parseable
# size string. Overridable per host via keep-list `builder_cache_max_size`.
DEFAULT_BUILDER_CACHE_MAX_SIZE = "100GB"

# OMN-16367 (DoD item 2): build-stamped tag shapes for per-sha image families.
# Every shape here was read off .201 on 2026-09-18, not invented:
#   20260918T111918Z-e49dea8f   deploy agent, <stamp>-<sha>
#   ffd61901-20260918T111918Z   deploy agent, <sha>-<stamp>
#   sha-ffd6190                 bare sha- tag
#   dev-sha-ffd6190             prefixed sha- tag (ECR)
#   ffd619017106...1222aa6      bare 40-hex content tag (ECR)
#
# DELIBERATELY NOT MATCHED: the loose `<word>-<hex>` shape (`main-f3d4029`,
# `runtime-fc61f41242dcc01b`). It is a per-sha tag too, but a pattern that broad
# would also match ordinary meaningful tags, and an unmatched tag simply means
# KEEP -- the safe direction. Narrow it in, never out.
# The DEFAULT shapes. They are defaults, not policy: the operative list is
# `generation_tag_patterns` in the keep-list, so widening what counts as a build
# stamp is a config review rather than a code change.
DEFAULT_GENERATION_TAG_PATTERNS: tuple[str, ...] = (
    r"^\d{8}T\d{6}Z?-[0-9a-f]{7,40}$",  # 20260918T111918Z-e49dea8f
    r"^[0-9a-f]{7,40}-\d{8}T\d{6}Z?$",  # ffd61901-20260918T111918Z
    r"^sha-[0-9a-f]{6,40}$",  # sha-ffd6190
    r"^[A-Za-z0-9._-]+-sha-[0-9a-f]{6,40}$",  # dev-sha-ffd6190
    r"^[0-9a-f]{40}$",  # bare 40-hex content tag
)

# How many newest generations of a per-sha family to retain, and the floor below
# which nothing is reaped however superseded. The floor is in HOURS, not days:
# the deploy agent rebuilds roughly hourly, so a day-scale floor keeps ~72
# generations permanently ineligible and a keep-newest-N window never bites.
DEFAULT_GENERATION_KEEP = 3
DEFAULT_GENERATION_MIN_AGE_HOURS = 6

# docker prints SI sizes ("3.66GB", "611MB", "24.73kB").
_SIZE_RE = re.compile(r"^\s*([0-9.]+)\s*([a-zA-Z]*B)\s*$")
_SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _parse_size(value: str) -> int:
    """Docker size string -> bytes. Unparseable -> 0, never an exception.

    A wrong total is worse than a missing one only if it is silently wrong, so an
    unreadable size contributes nothing rather than guessing.
    """
    m = _SIZE_RE.match(value or "")
    if not m:
        return 0
    try:
        return int(float(m.group(1)) * _SIZE_UNITS.get(m.group(2).upper(), 0))
    except (TypeError, ValueError):
        return 0


def _is_build_stamped_tag(tag: str, patterns: Sequence[str]) -> bool:
    """True if the tag is a per-build artifact rather than a meaningful pointer.

    Moving pointers (`dev`, `main`), release names and the keep tags all fail this
    check and are therefore never reaped by the generation rule. An unparseable
    pattern is skipped rather than raising: a broken config must not turn a GC
    into a crash, and skipping lands on KEEP.
    """
    for pattern in patterns:
        try:
            if re.match(pattern, tag):
                return True
        except re.error:
            continue
    return False


def _is_generation_bounded_repo(repo: str, patterns: list[str]) -> bool:
    """Substring match, same convention as keep_image_repos.

    Opt-in by repo: a repo nobody listed keeps its previous behaviour entirely.
    This is what holds the governed stability / judge / lakshman lanes out of
    scope structurally rather than by a name check that could drift.
    """
    return any(pattern and pattern in repo for pattern in patterns)


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


_HEX_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")


def _short_id(value: str) -> str | None:
    """Return the 12-hex-char short form of an image ID, or None if `value`
    is not a hex image ID at all (OMN-15804).

    `docker image ls --no-trunc` prints `sha256:<64hex>`; `docker ps --format
    '{{.Image}}'` prints the bare 12-hex short form when a container was
    started directly by image ID (not by repo:tag). Only hex-ID-shaped
    strings are normalized — a repo:tag ref (e.g. "myrepo:v1") never matches
    this shape, so it can never false-positive-collide with a short id.
    """
    v = value.strip()
    if v.startswith("sha256:"):
        v = v[len("sha256:") :]
    if not _HEX_ID_RE.match(v):
        return None
    return v[:12]


def _is_in_use(image_id: str, ref: str, inuse_refs: set[str]) -> bool:
    if image_id in inuse_refs or ref in inuse_refs:
        return True
    short = _short_id(image_id)
    if short is None:
        return False
    return any(short == _short_id(entry) for entry in inuse_refs)


def _is_disposable_ci_tag(tag: str) -> bool:
    """Return True if the tag looks like a disposable CI artifact (pr-N or sha-*)."""
    return bool(_PR_TAG_RE.match(tag) or _SHA_TAG_RE.match(tag))


def _pr_number_from_tag(tag: str) -> int | None:
    """Extract the PR number from a pr-<N> tag, or None if not a pr-tag."""
    m = _PR_TAG_RE.match(tag)
    return int(m.group(1)) if m else None


def make_github_pr_state_lookup(
    github_token: str, github_repo: str
) -> Callable[[int], str]:
    """Return a PR-state lookup function backed by the GitHub REST API.

    The returned callable accepts a PR number and returns one of:
      PR_STATE_MERGED, PR_STATE_CLOSED, PR_STATE_OPEN, PR_STATE_UNKNOWN.

    Any network/HTTP error → PR_STATE_UNKNOWN (safe: keep the image).
    This function is NOT called in unit tests; tests inject a mock lookup.
    """

    def _lookup(pr_number: int) -> str:
        url = f"https://api.github.com/repos/{github_repo}/pulls/{pr_number}"
        req = urllib.request.Request(  # noqa: S310 — URL is always https://api.github.com
            url,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read())
        except Exception:  # noqa: BLE001 — any network/auth/parse error → safe keep
            return PR_STATE_UNKNOWN

        state = data.get("state", "")
        merged = data.get("merged", False)
        if merged:
            return PR_STATE_MERGED
        if state == "closed":
            return PR_STATE_CLOSED
        if state == "open":
            return PR_STATE_OPEN
        return PR_STATE_UNKNOWN

    return _lookup


def build_plan(
    keep_list: dict[str, Any],
    images: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    inuse_refs: set[str],
    now: datetime | None = None,
    pr_state_lookup: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Pure planner. See module docstring for invariants.

    Args:
        keep_list:       Parsed keep-list.yaml dict.
        images:          List of docker image dicts from `docker image ls --format json`.
        containers:      List of docker container dicts from `docker ps --format json`.
        inuse_refs:      Set of repo:tag or image-id strings currently referenced by
                         any container (running or stopped). Used for protect_running.
        now:             Reference time for age calculations. Defaults to UTC now.
        pr_state_lookup: Optional callable(pr_number: int) -> str. When provided,
                         disposable CI tags (pr-<N> / sha-*) whose PR is merged or
                         closed are marked REMOVABLE regardless of age/generation.
                         When None, the PR-state stage is skipped entirely (safe default).
                         Lookup errors return PR_STATE_UNKNOWN → image is KEPT.
    """
    now = now or datetime.now(UTC)

    keep_repos: list[str] = keep_list.get("keep_image_repos", []) or []
    keep_tags: set[str] = set(keep_list.get("keep_image_tags", []) or [])
    protect_running: bool = bool(keep_list.get("protect_running", True))
    keep_generations: int = int(keep_list.get("superseded_image_keep_generations", 2))
    min_age_days: int = int(keep_list.get("min_age_days", 3))
    # OMN-16367: single source of truth for the builder-cache ceiling. Default
    # sized to the lab's working set, NOT to the 3.6 TB disk -- BuildKit's own
    # default derives a 341.8 GiB reserved floor from volume size, which is
    # untenable beside 674 GB of volumes and the containerd layer store.
    builder_cache_max_size: str = str(
        keep_list.get("builder_cache_max_size", DEFAULT_BUILDER_CACHE_MAX_SIZE)
    )
    # OMN-16367 DoD item 2: per-sha families to bound by generation count.
    # Empty list (the default) is a real off switch -- nothing changes.
    generation_bounded_repos: list[str] = (
        keep_list.get("generation_bounded_repos", []) or []
    )
    generation_keep: int = int(
        keep_list.get("generation_keep", DEFAULT_GENERATION_KEEP)
    )
    generation_min_age_hours: float = float(
        keep_list.get("generation_min_age_hours", DEFAULT_GENERATION_MIN_AGE_HOURS)
    )
    generation_tag_patterns: Sequence[str] = (
        keep_list.get("generation_tag_patterns") or DEFAULT_GENERATION_TAG_PATTERNS
    )

    remove_image_ids: list[str] = []
    kept_reasons: dict[str, str] = {}

    # Every repo:tag ref seen for a given image id, across ALL rows (not just
    # removal candidates) — OMN-15804: the executor needs the full tag set to
    # untag-then-remove a multi-tag image (`docker rmi <id>` alone fails with
    # "referenced in multiple repositories" once >1 tag points at one id).
    refs_by_id: dict[str, list[str]] = {}

    # Group superseded candidates per repo so we can keep the N newest.
    superseded_by_repo: dict[str, list[tuple[float, str]]] = {}
    # OMN-16367: separate bucket, because this class is bounded by GENERATION
    # COUNT on an hours-scale floor rather than by the day-scale min_age_days.
    generations_by_repo: dict[str, list[tuple[float, str]]] = {}

    for img in images:
        image_id = img.get("ID", "") or img.get("Id", "")
        repo = img.get("Repository", "") or ""
        tag = img.get("Tag", "") or ""
        created = img.get("CreatedAt", "") or ""
        age = _parse_created_at(created, now)
        ref = f"{repo}:{tag}" if repo and tag and repo != "<none>" else image_id

        if repo and tag and repo != "<none>" and tag != "<none>":
            tag_list = refs_by_id.setdefault(image_id, [])
            if ref not in tag_list:
                tag_list.append(ref)
        else:
            refs_by_id.setdefault(image_id, [])

        # --- Hard safety guards (always win, regardless of PR state) ---

        if tag in keep_tags and tag and tag != "<none>":
            kept_reasons[image_id] = f"tag '{tag}' in keep_image_tags"
            continue
        if protect_running and _is_in_use(image_id, ref, inuse_refs):
            kept_reasons[image_id] = "referenced by a container (protect_running)"
            continue

        # --- PR-state fast-path for disposable CI tags (OMN-13225) ---
        # Check this BEFORE the age gate: these tags bypass age/generation logic
        # when their PR is merged/closed. The hard safety guards above already ran.

        if (
            pr_state_lookup is not None
            and _is_disposable_ci_tag(tag)
            and repo
            and repo != "<none>"
        ):
            pr_number = _pr_number_from_tag(tag)
            if pr_number is not None:
                # pr-<N> tag: look up the PR state directly.
                state = pr_state_lookup(pr_number)
                if state in (PR_STATE_MERGED, PR_STATE_CLOSED):
                    remove_image_ids.append(image_id)
                    continue
                # open or unknown → keep; fall through to age/generation path
                kept_reasons[image_id] = (
                    f"disposable CI tag pr-{pr_number} state={state} (not merged/closed)"
                )
                continue
            # sha-* tag: no PR number to look up; age-based path only.
            # Fall through to normal age/generation logic below.

        # --- Generation-bounded per-sha families (OMN-16367, DoD item 2) ---
        # Placed BEFORE the day-scale age gate on purpose. At the deploy agent's
        # roughly-hourly cadence every generation is younger than min_age_days,
        # so the gate below would keep all of them -- which is precisely why 60
        # live generations of onex-lab/omninode-runtime (219.3 GB) existed on
        # 2026-09-18. keep_image_tags and protect_running are both checked ABOVE
        # this point and still win.
        if (
            generation_bounded_repos
            and repo
            and repo != "<none>"
            and tag
            and tag != "<none>"
            and _is_generation_bounded_repo(repo, generation_bounded_repos)
            and _is_build_stamped_tag(tag, generation_tag_patterns)
        ):
            if age * 24.0 < generation_min_age_hours:
                kept_reasons[image_id] = (
                    f"younger than generation_min_age_hours "
                    f"({age * 24.0:.1f}h<{generation_min_age_hours}h)"
                )
                continue
            generations_by_repo.setdefault(repo, []).append((age, image_id))
            continue

        # --- Age gate (applies to all remaining images) ---

        if age < min_age_days:
            kept_reasons[image_id] = (
                f"younger than min_age_days ({age:.1f}<{min_age_days})"
            )
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

    # Per-sha families: keep the newest `generation_keep`, reap the rest.
    for repo, gen_entries in generations_by_repo.items():
        gen_entries.sort(key=lambda e: e[0])  # newest first (smallest age first)
        for idx, (_age, image_id) in enumerate(gen_entries):
            if idx < generation_keep:
                kept_reasons[image_id] = (
                    f"newest {generation_keep} generation(s) of per-sha family "
                    f"'{repo}' (rollback headroom)"
                )
            else:
                remove_image_ids.append(image_id)

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

    remove_image_refs: dict[str, list[str]] = {
        image_id: refs_by_id.get(image_id, []) for image_id in remove_image_ids
    }

    # OMN-16367: summed nominal size of the removal set, so a dry run prints a
    # number a host-rollout consent row can cite. NOMINAL, not reclaim: shared
    # base layers collapse, and a prior measured pass on this host recovered 17.6%
    # of nominal. The caller must not present this as expected disk returned.
    size_by_id: dict[str, int] = {}
    for image in images:
        iid = str(image.get("ID", ""))
        if iid and iid not in size_by_id:
            size_by_id[iid] = _parse_size(str(image.get("Size", "")))
    remove_nominal_bytes = sum(size_by_id.get(i, 0) for i in remove_image_ids)

    return {
        "min_age_days": min_age_days,
        "builder_cache_max_size": builder_cache_max_size,
        "remove_image_ids": remove_image_ids,
        "remove_image_refs": remove_image_refs,
        "remove_nominal_bytes": remove_nominal_bytes,
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

    # PR-state lookup: enabled only when both GITHUB_TOKEN and GITHUB_REPO are set.
    github_token = os.environ.get("GITHUB_TOKEN", "")
    github_repo = os.environ.get("GITHUB_REPO", "")
    pr_state_lookup = (
        make_github_pr_state_lookup(github_token, github_repo)
        if github_token and github_repo
        else None
    )

    plan = build_plan(
        keep_list, images, containers, inuse, pr_state_lookup=pr_state_lookup
    )
    json.dump(plan, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
