#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ONE runtime-change path classifier for this repository (OMN-18664).

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
Two callers now ask the same question -- "does this set of changed files change
what the dev lane RUNS?" -- and they must never be able to disagree about the
answer.

1. ``scripts/trigger_rebuild_on_merge.py`` asks it about a merged pull request,
   to decide whether to publish a redeploy start command. A ``true`` there is
   what eventually produces a ``lab-pass-receipt-compose-dev-<sha>`` artifact.
2. ``scripts/ci/release_train.py`` asks it about each commit between a receipted
   commit and the default branch's head, to decide whether that receipt still
   describes the runtime HEAD carries. A ``false`` there is what lets the train
   inherit an ancestor's receipt.

The two answers are the same fact read from opposite ends, so a second path list
would mean the train could inherit a receipt across a commit the trigger
considered runtime-affecting -- a cut on a lane proof that does not describe the
code being cut. There is exactly one list, and it lives here.

WHY IT IS STDLIB-ONLY, DELIBERATELY
-----------------------------------
``trigger_rebuild_on_merge.py`` imports click and pydantic at module scope and
runs under ``uv run`` with the project installed. ``release_train.py`` is
stdlib-plus-pyyaml by design and runs on bare ``python3`` before any project
install. A train that had to import the trigger module to reach the classifier
would inherit that dependency and stop running before it decided anything. So
the shared surface is carved out into a module both can import, and the trigger
re-exports every name from here so its own callers and tests are unchanged.

The canonical deploy-path classifier itself is still omniclaude's deploy-gate
validator, loaded by path. Nothing about the union changes here; it only moved.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

#: A callable taking the changed-file list and returning the runtime-relevant
#: subset. Structurally the signature of omniclaude's ``find_runtime_paths``.
RuntimePathClassifier = Callable[[list[str]], list[str]]


def load_runtime_path_classifier(path: Path) -> RuntimePathClassifier:
    """Load the exact deploy-gate runtime-path classifier used by hosted CI.

    Runtime deployment scope has one owner: omniclaude's deploy-gate validator.
    Loading its ``find_runtime_paths`` callable keeps the post-merge publisher
    aligned with the required deploy gate instead of maintaining a second path
    allowlist that can silently drift.
    """
    if not path.is_file():
        raise ValueError(f"runtime path validator does not exist: {path}")

    module_name = "_canonical_deploy_path_classifier"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load runtime path validator: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ValueError(f"invalid runtime path validator {path}: {exc}") from exc

    classifier = getattr(module, "find_runtime_paths", None)
    if not callable(classifier):
        raise ValueError(
            f"runtime path validator {path} does not define find_runtime_paths"
        )
    return cast("RuntimePathClassifier", classifier)


# OMN-18072: lane-STATE paths the canonical deploy-gate classifier does not
# match, and is right not to match. Its question is "does this PR need deploy
# EVIDENCE"; this trigger's question is the wider "does this merge change what
# the lane RUNS". A migration runner, the migration corpus it applies and the
# runtime policy env are all lane state that a rebuild applies and that nothing
# else applies.
#
# MEASURED: omnibase_infra#3352 (0e9106da) changed
# scripts/run-forward-migrations.sh and deleted files under
# docker/migrations/_blocked/. The trigger ran on that merge (run 34330947198)
# and reported "No rebuild trigger: no runtime_change label or runtime path
# changes detected", so the seam has never executed on the dev lane.
#
# This supplements the canonical list HERE rather than widening it in
# omniclaude, because that list is also the required deploy gate on four repos
# and answers a different question. The canonical result is unioned, never
# replaced or narrowed.
LANE_STATE_PATH_PATTERNS: tuple[str, ...] = (
    # The forward-migration runner every compose up executes against the lane DB.
    "scripts/run-forward-migrations.sh",
    # The migration corpus that runner applies, including the _blocked/ holding
    # area -- moving a file out of it is exactly what changes the lane.
    "docker/migrations/**",
    # The compose model itself (already canonical for *.yml; declared here so
    # the trigger does not depend on that overlap staying true) and the lane's
    # runtime policy env, which the runtime reads at start.
    "docker/docker-compose*.yml",
    "docker/docker-compose*.yaml",
    "docker/runtime-policy.env",
    # The deploy agent's own source and launcher (OMN-18200). The process that
    # builds, recreates and verifies the lane is lane state by the same argument
    # the migration runner is. Without this, a fix to the agent cannot reach the
    # agent: no command is published, so it takes no job, and self_update has
    # only PRE_ACCEPT and POST_TERMINAL boundaries -- both job-driven, neither
    # reached at startup. Measured on omnibase_infra#3520 (``ead1f59b``), the
    # fix to the agent's own lab_overlay build: run 34815067432 declined, and
    # the lab host's clone stayed at ``8fd25217``, behind that fix. Restarting
    # the unit does not help, because it re-execs the same stale clone.
    #
    # Deliberately the package and the launcher, not ``scripts/deploy-agent/**``:
    # the agent's own tests change no lane behaviour, and every match here costs
    # a full dev-lane rebuild.
    "scripts/deploy-agent/deploy_agent/**",
    "scripts/deploy-agent/deploy/**",
    # OMN-18572. The dev lane's `onex-api` service, in the OMNINODE_INFRA tree.
    #
    # This one matches a path that does not exist in this repository, and that
    # is deliberate rather than a mistake: the publisher is shared, and for a
    # caller whose --source-repo is omninode_infra these are the paths that
    # decide what the lane runs. The lane resolves `image: ${ONEX_API_IMAGE}`,
    # and the lab-overlay applier builds that image from exactly this directory
    # in the archived overlay tree (`lab_overlay.API_DOCKERFILE`/`API_CONTEXT`).
    #
    # The canonical classifier is right to miss it. Its `RUNTIME_PATH_PATTERNS`
    # carry `docker/Dockerfile*` and `docker/**/*.Dockerfile`, and neither
    # matches `docker/onex-api/Dockerfile` -- three segments against a
    # two-segment pattern, and no `.Dockerfile` suffix -- while nothing at all
    # matches `docker/onex-api/main.py`. That list answers "does this PR need
    # deploy EVIDENCE" for the CLOUD plane, whose onex-api image is built and
    # pinned by a separate workflow entirely.
    #
    # Measured cost of its absence: omninode_infra#1523 merged 2026-09-17
    # 09:00:53Z and the lane was still running that squash's PARENT at 11:15Z,
    # with tenant creation on the lab impossible throughout.
    "docker/onex-api/**",
)


def _matches_pattern(path: str, pattern: str) -> bool:
    """Segment-wise, repo-root-anchored glob match.

    ``PurePosixPath.match`` is right-anchored (it matches a SUFFIX of the path)
    and does not treat ``**`` as recursive, so it both over-matches a nested
    ``a/b/docker/runtime-policy.env`` and under-matches
    ``docker/migrations/_blocked/README.md``. Both are wrong for a path list
    that decides whether a lane gets rebuilt, so the match is spelled out:
    a trailing ``/**`` matches everything beneath that directory, and every
    other pattern matches segment for segment from the repo root.
    """
    if pattern.endswith("/**"):
        return path.startswith(pattern[: -len("**")])
    pattern_parts = pattern.split("/")
    path_parts = path.split("/")
    if len(pattern_parts) != len(path_parts):
        return False
    return all(
        fnmatch.fnmatchcase(actual, expected)
        for actual, expected in zip(path_parts, pattern_parts, strict=True)
    )


def find_lane_state_paths(changed_files: list[str]) -> list[str]:
    """Return the changed files that are lane STATE this trigger must rebuild for.

    Order-preserving and de-duplicated, so the union below reads as the
    canonical hits followed by the supplementary ones.
    """
    hits: list[str] = []
    for path in changed_files:
        if not isinstance(path, str) or not path.strip():
            continue
        candidate = path.strip()
        if (
            any(_matches_pattern(candidate, p) for p in LANE_STATE_PATH_PATTERNS)
            and candidate not in hits
        ):
            hits.append(candidate)
    return hits


def classify_runtime_paths(
    changed_files: list[str], classifier: RuntimePathClassifier
) -> list[str]:
    """Run and validate the canonical classifier's output fail-closed.

    OMN-18072: the canonical result is then UNIONED with the lane-state
    supplement above. The validation stays ahead of the union so a broken
    canonical classifier still fails closed rather than being papered over by a
    supplementary hit.
    """
    runtime_paths = classifier(changed_files)
    if not isinstance(runtime_paths, list) or any(
        not isinstance(path, str) or not path.strip() for path in runtime_paths
    ):
        raise ValueError("runtime path validator returned an invalid path list")
    combined = list(runtime_paths)
    for path in find_lane_state_paths(changed_files):
        if path not in combined:
            combined.append(path)
    return combined
