#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Cross-repo dashboard/web topic parity checker (OMN-15863).

Repoints the dead "Topic Subscription Parity" half of
`.github/workflows/topic-parity.yml`, which referenced three omnidash paths
that no longer exist (`omnidash/server/read-model-consumer.ts`,
`omnidash/server/event-bus-health-poller.ts`, `omnidash/shared/topics.ts` —
omnidash dropped its Kafka-consuming server and moved to an HTTP-polling
projection-read architecture; see `omnidash/CLAUDE.md`).

This script checks the two frontend topic-constant files that are actually
live today against their real producer surfaces:

1. **omnidash** (`shared/types/topics.ts`, `onex.snapshot.projection.*`
   topics) against every producer `contract.yaml` under omnimarket,
   omnibase_infra, and omniintelligence. A frontend `TOPICS` entry with no
   matching producer contract anywhere is a dead reference — FAIL.
   A producer contract with no frontend constant is normal (backend can ship
   ahead of the widget) — advisory only.

2. **omniweb** (`lib/topics.ts` vs `contracts/topics.yaml`) — both files are
   supposed to declare the exact same `onex.evt.omniweb.*` topic set
   (enforced inside omniweb's own CI by `scripts/lint-topics.ts`). This is a
   redundant cross-repo safety net: a mismatch here means omniweb's own gate
   is either broken or was bypassed.

Usage::

    # CI mode: exit 0 if consistent, exit 1 with a human-readable diff
    python scripts/check_dashboard_topic_parity.py --check

    # Verbose: also print backend-ahead-of-frontend advisories
    python scripts/check_dashboard_topic_parity.py --check --verbose

Paths are resolved relative to the omni_home root, exactly like
`scripts/check-topic-parity.py`: default is
``Path(__file__).resolve().parents[2]`` (this file lives at
``<omni_home>/omnibase_infra/scripts/<this file>`` — it is HOSTED here
because omni_home's `no-functional-code` pre-commit gate forbids .py/.sh
in omni_home itself and routes scripts to `omnibase_infra/scripts/`, the
OMN-4922 precedent — but it is DRIVEN by
`omni_home/.github/workflows/topic-parity.yml`), override via the
``OMNI_HOME`` env var or ``--omni-home`` (required in CI, where the
workflow assembles the sibling checkouts under a synthetic omni_home
root before invoking this script from the omnibase_infra checkout).

Exit codes:
    0 -- parity: every frontend topic reference has a real producer, and
         omniweb's two topic files agree
    1 -- REAL drift found (see printed diff)
    2 -- tooling/environment error (a required repo or file is missing)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

TOPIC_LITERAL_RE = re.compile(r"'(onex\.[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+)'")


def _default_omni_home() -> Path:
    env_value = os.environ.get("OMNI_HOME")
    if env_value:
        return Path(env_value).resolve()
    # Script lives at <omni_home>/omnibase_infra/scripts/<this file>
    return Path(__file__).resolve().parents[2]


def _extract_topics_block(path: Path, const_name: str) -> set[str]:
    """Extract `onex.*` string literals from a named TS const block.

    Returns an empty set (not an error) if the file or block is missing —
    callers that require the file decide whether that's fatal.
    """
    if not path.exists():
        return set()
    content = path.read_text()
    pattern = re.compile(
        rf"(?:export\s+)?const\s+{re.escape(const_name)}\s*(?::\s*\w+(?:\[\])?)?\s*=\s*\{{(.*?)\n\}}",
        re.DOTALL,
    )
    match = pattern.search(content)
    block = match.group(1) if match else content
    return {m.group(1) for m in TOPIC_LITERAL_RE.finditer(block)}


def _extract_topics_from_tree(repo_root: Path, filename_glob: str) -> set[str]:
    """Regex-scan every matching file under repo_root/src for onex.* literals.

    Mirrors check-topic-parity.py's approach: this is a drift *detector*, not
    a schema validator — a plain string scan is deliberate so it catches
    topics regardless of which YAML key they're declared under.

    NOTE: ``Path.rglob`` does not recurse into symlinked directories on the
    Python version this script is pinned to in CI (3.12; symlink-following
    ``**`` recursion is a 3.13+ addition). If a caller assembles the
    producer repos via symlinks rather than real directories/checkouts, this
    will silently return an empty set instead of failing loudly — the CI
    workflow moves checkouts into place for exactly this reason.
    """
    src_root = repo_root / "src"
    if not src_root.exists():
        return set()
    topics: set[str] = set()
    for contract_path in src_root.rglob(filename_glob):
        content = contract_path.read_text(errors="replace")
        topics.update(re.findall(r"onex\.[a-z][a-z0-9_.-]*\.v\d+", content))
    return topics


def _extract_yaml_topic_names(path: Path) -> set[str]:
    """Extract `name: onex.*` entries from a topics.yaml registry file."""
    if not path.exists():
        return set()
    content = path.read_text()
    return set(
        re.findall(r"^\s*-?\s*name:\s*(onex\.[a-z0-9_.-]+)\s*$", content, re.MULTILINE)
    )


class ModelDashboardTopicParityPaths:
    """Resolved filesystem paths required by the dashboard topic checker."""

    def __init__(self, omni_home: Path) -> None:
        self.omni_home = omni_home
        self.omnidash_topics_ts = (
            omni_home / "omnidash" / "shared" / "types" / "topics.ts"
        )
        self.producer_repos = [
            omni_home / "omnimarket",
            omni_home / "omnibase_infra",
            omni_home / "omniintelligence",
        ]
        self.omniweb_lib_topics_ts = omni_home / "omniweb" / "lib" / "topics.ts"
        self.omniweb_contracts_topics_yaml = (
            omni_home / "omniweb" / "contracts" / "topics.yaml"
        )


def check_dashboard_parity(
    paths: ModelDashboardTopicParityPaths, verbose: bool = False
) -> int:
    errors: list[str] = []
    advisories: list[str] = []

    # -------------------------------------------------------------------
    # Check 1: omnidash frontend TOPICS vs producer contract.yaml files
    # -------------------------------------------------------------------
    if not paths.omnidash_topics_ts.exists():
        print(
            f"ERROR: omnidash topics file not found: {paths.omnidash_topics_ts}",
            file=sys.stderr,
        )
        return 2

    frontend_topics = _extract_topics_block(paths.omnidash_topics_ts, "TOPICS")
    frontend_topics = {t for t in frontend_topics if t.startswith("onex.snapshot.")}
    if not frontend_topics:
        print(
            f"ERROR: No onex.snapshot.* topics found in {paths.omnidash_topics_ts} "
            "— parser or source likely broken, failing closed.",
            file=sys.stderr,
        )
        return 2

    any_producer_repo_present = any(r.exists() for r in paths.producer_repos)
    if not any_producer_repo_present:
        print(
            "ERROR: none of the producer repos "
            f"({', '.join(str(r) for r in paths.producer_repos)}) are checked out "
            "— cannot verify producer coverage, failing closed.",
            file=sys.stderr,
        )
        return 2

    producer_topics: set[str] = set()
    for repo in paths.producer_repos:
        producer_topics.update(_extract_topics_from_tree(repo, "contract.yaml"))
    producer_topics = {t for t in producer_topics if t.startswith("onex.snapshot.")}

    orphaned_frontend = sorted(frontend_topics - producer_topics)
    backend_ahead = sorted(producer_topics - frontend_topics)

    if orphaned_frontend:
        errors.append(
            "omnidash shared/types/topics.ts references onex.snapshot.* topics with "
            "NO producing contract.yaml in omnimarket/omnibase_infra/omniintelligence:"
        )
        for t in orphaned_frontend:
            errors.append(f"  + {t}")

    if backend_ahead and verbose:
        advisories.append(
            "ADVISORY: producer contract.yaml topics not yet in omnidash TOPICS "
            "(backend ahead of frontend, OK):"
        )
        for t in backend_ahead:
            advisories.append(f"  ~ {t}")

    # -------------------------------------------------------------------
    # Check 2: omniweb lib/topics.ts vs contracts/topics.yaml
    # -------------------------------------------------------------------
    omniweb_present = (
        paths.omniweb_lib_topics_ts.exists()
        or paths.omniweb_contracts_topics_yaml.exists()
    )
    if omniweb_present:
        if not paths.omniweb_lib_topics_ts.exists():
            print(
                f"ERROR: omniweb checked out but missing {paths.omniweb_lib_topics_ts}",
                file=sys.stderr,
            )
            return 2
        if not paths.omniweb_contracts_topics_yaml.exists():
            print(
                f"ERROR: omniweb checked out but missing {paths.omniweb_contracts_topics_yaml}",
                file=sys.stderr,
            )
            return 2

        web_const_topics = _extract_topics_block(paths.omniweb_lib_topics_ts, "TOPICS")
        web_registry_topics = _extract_yaml_topic_names(
            paths.omniweb_contracts_topics_yaml
        )

        const_only = sorted(web_const_topics - web_registry_topics)
        registry_only = sorted(web_registry_topics - web_const_topics)

        if const_only:
            errors.append(
                "omniweb lib/topics.ts declares topics NOT in contracts/topics.yaml:"
            )
            for t in const_only:
                errors.append(f"  + {t}")
        if registry_only:
            errors.append(
                "omniweb contracts/topics.yaml declares topics NOT in lib/topics.ts:"
            )
            for t in registry_only:
                errors.append(f"  + {t}")
    else:
        print(
            "NOTE: omniweb not checked out — skipping omniweb topic-file parity "
            "(covered independently by omniweb's own lint-topics.ts CI gate).",
        )

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    if verbose:
        print(f"omnidash frontend onex.snapshot.* topics: {len(frontend_topics)}")
        print(f"producer-contract onex.snapshot.* topics: {len(producer_topics)}")
        if omniweb_present:
            print(f"omniweb lib/topics.ts topics:            {len(web_const_topics)}")
            print(
                f"omniweb contracts/topics.yaml topics:    {len(web_registry_topics)}"
            )
        print()

    for line in advisories:
        print(line)
    if advisories:
        print()

    if errors:
        print("DASHBOARD TOPIC PARITY FAILURE")
        print("=" * 60)
        for line in errors:
            print(line)
        print()
        print("To fix:")
        print("  1. omnidash orphan: add the producer contract.yaml entry, OR")
        print("     remove the dead constant from shared/types/topics.ts")
        print("  2. omniweb mismatch: sync lib/topics.ts and contracts/topics.yaml")
        return 1

    print("OK: Dashboard/web topic parity check passed")
    print(
        f"  omnidash TOPICS.ts:        {len(frontend_topics)} onex.snapshot.* topics, all producer-backed"
    )
    if omniweb_present:
        print(
            f"  omniweb topics:            {len(web_const_topics)} topics, lib/topics.ts == contracts/topics.yaml"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-repo dashboard/web topic parity checker (OMN-15863)"
    )
    parser.add_argument(
        "--check", action="store_true", required=True, help="Run parity check (CI mode)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show advisory backend-ahead diffs"
    )
    parser.add_argument(
        "--omni-home",
        type=Path,
        default=None,
        help="Override omni_home root (default: OMNI_HOME env var, else parents[1])",
    )
    args = parser.parse_args()

    omni_home = args.omni_home.resolve() if args.omni_home else _default_omni_home()
    paths = ModelDashboardTopicParityPaths(omni_home)
    return check_dashboard_parity(paths, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
