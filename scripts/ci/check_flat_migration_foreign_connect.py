# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail CI on a NEW flat migration with an unledgered cross-DB `\\connect` (OMN-15819).

The k8s Job that applies `docker/migrations/forward/*.sql` (flat, -maxdepth 1
-- `omninode_infra` repo, `k8s/migrations/omnibase-infra-migrate.yaml`) owns
exactly one database, `omnibase_infra`. Its flat loop's `psql -f` apply is
gated on the file's first `\\connect` directive naming that same database;
a file whose `\\connect` names anything else has NO execution path in that
Job, in that loop or any other. OMN-15819 found two files (098/099) that had
silently accreted a false "applied" ledger row for exactly that reason.

This is the STATIC, pre-merge half of the fix (the runner-side companion
lives in the `omninode_infra` repo). Every existing cross-DB flat file is
listed in ``docker/migrations/forward/cross-database-flat-migrations.yaml``
with a citation -- see that file's own docstring for the two dispositions
(``undeliverable`` / ``grandfathered``) and why the distinction matters. A
NEW cross-DB flat file that is not in the manifest is a hard, fail-closed
reject: the fix is a node-owned migration under
``docker/migrations/forward/nodes/<node>/``, which connects to the target
database directly as its own role.

Both directions are enforced -- a live cross-DB file missing from the
manifest, and a manifest entry that no longer matches live reality (file
gone, or no longer cross-DB) -- exactly the AC2 pattern
``tests/ci/test_flat_node_migration_shape_parity.py`` (OMN-15384) already
uses for its own ledger.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
MANIFEST_PATH = FORWARD_DIR / "cross-database-flat-migrations.yaml"

# The one database the k8s Job's flat loop ever \connects to for its own
# `psql -f` apply (DB_NAME in omninode_infra's
# k8s/migrations/omnibase-infra-migrate.yaml; the identical default in this
# repo's own scripts/run-forward-migrations.sh POSTGRES_DB and every
# docker-compose.infra.yml POSTGRES_DB binding for this service).
RUNNER_OWN_DATABASE = "omnibase_infra"

_VALID_DISPOSITIONS = frozenset({"undeliverable", "grandfathered"})

# The manifest is a CLOSED ledger as of OMN-15819 (this gate's own
# authorship, commit 6083b76b4): "a NEW cross-DB flat file is a hard,
# fail-closed reject" (module docstring) means what it says regardless of
# disposition or citation -- a manifest entry alone must never be able to
# authorize a brand-new cross-DB flat migration, because the manifest is
# ordinary repo-tracked YAML a PR can edit in the same diff that adds the
# offending file. Without this, `check()` verified only that a live
# cross-DB file's manifest entry *exists and its connect_target matches* --
# it never asked whether the entry was itself new, so a PR could add both
# the file and a plausible-looking manifest entry (any disposition) in one
# shot and pass clean (found in review, OMN-15819 CodeRabbit thread
# r3749990788). Pinning the exact filenames the manifest was seeded with at
# gate-authorship time closes that regardless of what the YAML says.
#
# `check()` takes this as an explicit, opt-in `frozen_seed` kwarg (default
# None = unrestricted) rather than a hardcoded default so every OTHER test
# in this suite -- which exercises stale-entry / target-drift / malformed
# scenarios against synthetic, non-production filenames -- is unaffected;
# only `main()` (the real CI entrypoint) and the tests that specifically
# cover this closed-ledger property pass it.
MANIFEST_FROZEN_SEED: frozenset[str] = frozenset(
    {
        "098_create_omninode_internal_schema.sql",
        "099_create_omninode_internal_live_events.sql",
        "083_create_log_entries.sql",
        "096_grant_role_omnidash_omnidash_analytics.sql",
        "097_grant_app_dashboard_connect_omnidash_analytics.sql",
    }
)

# Same directive shape the k8s Job's own awk one-liner recognizes:
# `awk '$1 == "\\connect" { print $2; exit }'` -- first line whose first
# whitespace-delimited field is the literal token `\connect`, first match
# wins. awk's default field splitting strips leading horizontal whitespace,
# so a line like `  \connect other_db` still has `$1 == "\\connect"` in the
# runner. `[^\S\r\n]*` mirrors that (leading spaces/tabs, not newlines) so
# an indented directive cannot silently read as "no \connect" to this gate
# while the runner still executes it as a real cross-DB connect. Mirrored
# here (not shelled out to awk) so this gate has no bash dependency and is
# independently testable.
_CONNECT_DIRECTIVE = re.compile(r"^[^\S\r\n]*\\connect\s+(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class ManifestEntry:
    file: str
    connect_target: str
    disposition: str
    citation: str


@dataclass(frozen=True)
class Violation:
    file: str
    reason: str

    def describe(self) -> str:
        return f"{self.file}: {self.reason}"


def flat_migration_connect_target(sql_path: Path) -> str | None:
    """The file's first `\\connect <db>` directive target, or None if it has none."""
    match = _CONNECT_DIRECTIVE.search(sql_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def flat_migration_files(forward_dir: Path = FORWARD_DIR) -> list[Path]:
    """Top-level (-maxdepth 1) `*.sql` files -- excludes `nodes/` deliberately.

    Matches the k8s Job's own discovery expression
    (`find "$MIGRATION_DIR" -maxdepth 1 -name '*.sql' -type f`): a file under
    `forward/nodes/<node>/` is a DIFFERENT corpus, applied by a DIFFERENT
    loop that connects directly to its own target database, and is never in
    scope for this gate.
    """
    return sorted(forward_dir.glob("*.sql"))


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, ManifestEntry]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        msg = f"{manifest_path} must be a mapping with a top-level `entries:` list"
        raise AssertionError(msg)
    entries: dict[str, ManifestEntry] = {}
    for raw_entry in raw["entries"]:
        entry = ManifestEntry(
            file=raw_entry["file"],
            connect_target=raw_entry["connect_target"],
            disposition=raw_entry["disposition"],
            citation=raw_entry["citation"],
        )
        if entry.disposition not in _VALID_DISPOSITIONS:
            msg = (
                f"{manifest_path}: {entry.file} has unknown disposition "
                f"{entry.disposition!r} (expected one of {sorted(_VALID_DISPOSITIONS)})"
            )
            raise AssertionError(msg)
        if not entry.citation.strip():
            msg = f"{manifest_path}: {entry.file} has an empty citation"
            raise AssertionError(msg)
        if entry.file in entries:
            msg = f"{manifest_path}: duplicate entry for {entry.file}"
            raise AssertionError(msg)
        entries[entry.file] = entry
    return entries


def check(
    forward_dir: Path = FORWARD_DIR,
    manifest_path: Path = MANIFEST_PATH,
    runner_own_database: str = RUNNER_OWN_DATABASE,
    *,
    frozen_seed: frozenset[str] | None = None,
) -> list[Violation]:
    """Fail-closed, both directions. Empty return == gate passes.

    ``frozen_seed``, when provided, additionally rejects any manifest entry
    for a filename outside that set -- see ``MANIFEST_FROZEN_SEED`` above.
    ``None`` (the default) skips that check entirely, which is what every
    synthetic-fixture test in this suite that is not specifically about the
    closed-ledger property wants.
    """
    manifest = load_manifest(manifest_path)
    live_cross_db: dict[str, str] = {}
    for sql_path in flat_migration_files(forward_dir):
        target = flat_migration_connect_target(sql_path)
        if target is not None and target != runner_own_database:
            live_cross_db[sql_path.name] = target

    violations: list[Violation] = []
    try:
        manifest_display = manifest_path.relative_to(REPO_ROOT)
    except ValueError:
        manifest_display = manifest_path

    for filename, target in sorted(live_cross_db.items()):
        entry = manifest.get(filename)
        if entry is None:
            violations.append(
                Violation(
                    file=filename,
                    reason=(
                        f"flat migration carries `\\connect {target}` (foreign to "
                        f"the runner's own database {runner_own_database!r}) but has "
                        f"NO entry in {manifest_display}. This "
                        "file has no execution path in the k8s Job (OMN-15819) -- "
                        "author a node-owned replacement under "
                        "docker/migrations/forward/nodes/<node>/ instead of adding "
                        "a flat cross-DB migration."
                    ),
                )
            )
        elif frozen_seed is not None and filename not in frozen_seed:
            violations.append(
                Violation(
                    file=filename,
                    reason=(
                        f"flat migration carries `\\connect {target}` and has a "
                        f"matching entry in {manifest_display} (disposition="
                        f"{entry.disposition!r}), but {filename!r} is not part of "
                        "the frozen OMN-15819 seed set -- a NEW cross-DB flat "
                        "migration is a hard reject regardless of disposition or "
                        "citation; a manifest entry alone cannot authorize one. "
                        "Author a node-owned replacement under "
                        "docker/migrations/forward/nodes/<node>/ instead."
                    ),
                )
            )
        elif entry.connect_target != target:
            violations.append(
                Violation(
                    file=filename,
                    reason=(
                        f"manifest says connect_target={entry.connect_target!r} but "
                        f"the file now targets {target!r} -- update the manifest "
                        "entry to match live reality"
                    ),
                )
            )

    for filename, entry in sorted(manifest.items()):
        if filename not in live_cross_db:
            reason = (
                "manifest entry has no live counterpart -- the file is gone, or no "
                "longer carries a foreign \\connect. Remove the stale entry."
            )
            violations.append(Violation(file=filename, reason=reason))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forward-dir",
        type=Path,
        default=FORWARD_DIR,
        help="docker/migrations/forward directory to scan",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="cross-database-flat-migrations.yaml path",
    )
    args = parser.parse_args(argv)

    violations = check(
        forward_dir=args.forward_dir,
        manifest_path=args.manifest,
        frozen_seed=MANIFEST_FROZEN_SEED,
    )
    if not violations:
        print("OK: no un-ledgered cross-DB flat migrations (OMN-15819)")
        return 0

    print("FAIL: cross-DB flat migration gate (OMN-15819)", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation.describe()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
