# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regenerate the frozen application-database SQL baseline (OMN-15361).

The baseline records violations that already existed in deployed SQL when it was
frozen, so the dev->main promotion boundary does not re-litigate the whole
migration corpus. See the ratchet comment in ``check_application_database_sql``.

Deterministic by construction: entries are content-keyed and emitted in sorted
order, so regenerating against an unchanged tree reproduces the file byte for
byte and the freeze is auditable in review.

The baseline is SHRINK-ONLY. Regenerate to *remove* entries after fixing
violations; never to absorb new ones. Run with ``--check`` in CI or locally to
assert the committed file matches what the current tree produces.

    uv run python scripts/ci/generate_application_database_sql_baseline.py \\
        --base-revision origin/main \\
        --ownership-manifest <path> [--ownership-manifest <path> ...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# Importable both as `python -m scripts.ci.generate_...` and as a direct script
# path; the bootstrap must precede the package import, not sit under __main__.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.check_application_database_sql import (
    _BASELINE_PATH,
    load_sql_baseline,
    validate_changed_sql,
    violation_key,
)

_HEADER = """\
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-15361: frozen baseline for the application-database SQL gate.
#
# WHY THIS FILE EXISTS: the gate lints SQL *changed against the PR base*. On a
# dev PR that is a handful of files, so each file was only ever scanned in its
# own small PR. At the dev->main promotion boundary the base is main, so the
# entire accumulated migration corpus counts as changed and every latent
# violation fires at once -- none newly authored, all already deployed.
# Rewriting deployed migrations to satisfy a gate at release time is the more
# dangerous path, so those pre-existing violations are recorded here and
# soft-passed. This mirrors the OMN-14443 deploy-gate grandfather ratchet.
#
# THE RATCHET: a violation NOT listed here is held to the full bar and fails
# closed. An entry whose file is deleted, or whose violation stops firing on a
# file the run actually linted, is STALE and FAILS the gate -- so the list can
# only shrink.
#
# BURN-DOWN ONLY: never hand-add an entry. Fix violations, then regenerate with
# scripts/ci/generate_application_database_sql_baseline.py to shrink the list.
# Widening it defeats the ratchet.
#
# Entries are keyed by sha256 of the "<path>: <message>" violation line -- content,
# never line numbers, so unrelated SQL edits cannot silently re-key an entry.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", default="HEAD")
    parser.add_argument(
        "--ownership-manifest",
        action="append",
        type=Path,
        default=[],
        help="Typed ownership manifest; repeat for every authoritative source",
    )
    parser.add_argument("--output", type=Path, default=_BASELINE_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed baseline differs from this tree",
    )
    return parser


def render_baseline(violations: tuple[str, ...], *, generated_at: str) -> str:
    """Render the baseline document; sorted and content-keyed, so it is stable."""
    lines = [_HEADER, f'generated_at: "{generated_at}"', f"count: {len(violations)}"]
    if not violations:
        lines.append("violations: []")
        return "\n".join(lines) + "\n"
    lines.append("violations:")
    for violation in sorted(violations):
        path, _, message = violation.partition(": ")
        lines.append(f'  - key: "{violation_key(violation)}"')
        lines.append(f'    path: "{path}"')
        lines.append(f"    violation: {message.strip()!r}")
    return "\n".join(lines) + "\n"


def _without_stamp(text: str) -> str:
    """Drop the generated_at line; it is provenance, not semantic content."""
    return "\n".join(
        line for line in text.splitlines() if not line.startswith("generated_at:")
    )


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.resolve()

    # Regenerate against the RAW gate verdict: pass an explicitly absent baseline
    # so the current snapshot cannot fold itself back in and freeze forever.
    outcome = validate_changed_sql(
        repository,
        args.base_revision,
        args.head_revision,
        ownership_manifest_paths=tuple(
            path if path.is_absolute() else repository / path
            for path in args.ownership_manifest
        ),
        baseline_path=Path("/nonexistent-baseline-force-raw-verdict"),
    )

    existing = load_sql_baseline(args.output)
    fresh_keys = {violation_key(violation) for violation in outcome.violations}
    grew = sorted(fresh_keys - set(existing))

    if args.check:
        rendered = render_baseline(outcome.violations, generated_at="")
        committed = (
            args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        )
        if _without_stamp(rendered) != _without_stamp(committed):
            print(
                f"application_database_sql_baseline=DRIFT: {args.output.name} does not "
                f"match this tree ({len(outcome.violations)} raw violations). "
                "Regenerate it, and confirm the change only SHRINKS the list."
            )
            return 1
        print(
            f"application_database_sql_baseline=OK ({len(outcome.violations)} entries)"
        )
        return 0

    if existing and grew:
        print(
            f"application_database_sql_baseline=REFUSED: regenerating would ADD "
            f"{len(grew)} entr{'y' if len(grew) == 1 else 'ies'} not in the committed "
            "baseline. The baseline is shrink-only -- fix the new violations instead."
        )
        for key in grew[:10]:
            print(f"  + {key[:12]}")
        return 1

    args.output.write_text(
        render_baseline(
            outcome.violations,
            generated_at=datetime.now(UTC).strftime("%Y-%m-%d"),
        ),
        encoding="utf-8",
    )
    removed = len(set(existing) - fresh_keys)
    print(
        f"application_database_sql_baseline=WROTE {len(outcome.violations)} entries "
        f"to {args.output} (removed {removed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
