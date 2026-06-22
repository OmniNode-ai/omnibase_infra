# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""check_evidence_provenance_probe.py — Pre-commit gate: reject a "deployed SHA"
claim in a handoff/OCC/evidence doc that lacks an adjacent provenance-probe
citation (OMN-13030).

THE CLASS FIX: deploy/handoff docs routinely assert "the runtime is now running
SHA <x>" copied from a build log or memory, never from the image itself. The
ground truth is baked into the image at /app/build-provenance.json. This gate
makes a deployed-SHA claim unwritable unless the same document shows the output
of probing that file from the running container.

RULE: Any staged markdown file that contains a "deployed sha" / "deployed
digest" / "running sha" style claim MUST carry a provenance-probe citation
within a small window of lines around the claim. The accepted probe is a
`docker exec <container> cat .../build-provenance.json` invocation (the EFFECT
that reads the image's own provenance manifest), optionally with `ssh ...`
prefix for a remote host.

Exit codes:
  0 — every deployed-SHA claim has an adjacent provenance-probe citation
  1 — one or more claims lack the adjacent probe

Usage (pre-commit):
  python scripts/check_evidence_provenance_probe.py --staged
  python scripts/check_evidence_provenance_probe.py --file path/to/handoff.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# A concrete deployed-SHA claim: a deployed/running/promoted/live/redeployed
# assertion that carries an ACTUAL SHA value on the same line. Abstract mentions
# of the words "deployed sha" (schema field descriptions, artifact-file captions,
# status definitions) are NOT claims — only an asserted concrete digest is. The
# value must be a >=7-hex-char git short/long SHA or a `sha256:` image digest.
_CLAIM_VERB = r"(?:deployed|running|promoted|live|redeployed)"
_SHA_WORD = r"(?:sha|digest|revision|commit)"
_SHA_VALUE = r"(?:sha256:[0-9a-f]{12,}|\b[0-9a-f]{7,40}\b)"
_DEPLOYED_SHA_CLAIM = re.compile(
    rf"\b{_CLAIM_VERB}\b[^\n]{{0,80}}?\b{_SHA_WORD}\b[^\n]{{0,40}}?{_SHA_VALUE}",
    re.IGNORECASE,
)

# The accepted provenance probe: a docker-exec read of the image's baked
# build-provenance manifest. Accepts an optional ssh prefix for remote hosts.
_PROVENANCE_PROBE = re.compile(
    r"docker\s+exec\b[^\n]*\bcat\b[^\n]*build-provenance\.json",
    re.IGNORECASE,
)

# Number of lines on each side of a claim within which a probe must appear for
# the claim to be considered substantiated. A probe block immediately above or
# below the claim is the expected shape.
_PROBE_WINDOW_LINES = 15

# Files exempt from the gate (this script itself, test fixtures, the schema doc
# which references build-provenance.json structurally, and docs/templates/ which
# DEMONSTRATE the required shape rather than assert a real deployment).
_EXEMPT_PATH_PATTERNS = [
    re.compile(r"^tests/"),
    re.compile(r"^docs/templates/"),
    re.compile(r"scripts/check_evidence_provenance_probe\.py$"),
    re.compile(r"build-provenance-schema\.json$"),
]


def _is_exempt(path: str) -> bool:
    return any(p.search(path) for p in _EXEMPT_PATH_PATTERNS)


def _probe_in_window(lines: list[str], claim_idx: int) -> bool:
    lo = max(0, claim_idx - _PROBE_WINDOW_LINES)
    hi = min(len(lines), claim_idx + _PROBE_WINDOW_LINES + 1)
    window = "\n".join(lines[lo:hi])
    return bool(_PROVENANCE_PROBE.search(window))


def check_file(path: Path) -> list[str]:
    """Check a single file. Returns a list of violation messages."""
    violations: list[str] = []
    rel = str(path)

    if _is_exempt(rel):
        return violations

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations  # file disappeared between staging and check

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        # Markdown headings and blockquote meta-lines describe structure/guidance,
        # not a runtime deployment assertion. A real "deployed SHA" claim is a
        # normal statement or table row.
        if stripped.startswith(("#", ">")):
            continue
        if not _DEPLOYED_SHA_CLAIM.search(line):
            continue
        if _probe_in_window(lines, idx):
            continue
        violations.append(
            f"{rel}:{idx + 1}: deployed-SHA claim without an adjacent "
            f"provenance-probe citation. Add the output of "
            f"`docker exec <runtime-container> cat /app/build-provenance.json` "
            f"within {_PROBE_WINDOW_LINES} lines of the claim so the deployed "
            f"SHA is proven against the image's baked provenance manifest, not "
            f"copied from a build log or memory (OMN-13030). Claim: "
            f"{line.strip()[:80]!r}"
        )

    return violations


def check_staged() -> list[str]:
    """Check all staged markdown files for unsubstantiated deployed-SHA claims."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: git diff failed: {exc}", file=sys.stderr)
        return []

    staged_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    violations: list[str] = []

    for rel_path in staged_files:
        if not rel_path.endswith(".md"):
            continue
        path = Path(rel_path)
        if not path.exists():
            continue
        violations.extend(check_file(path))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-commit gate: reject deployed-SHA claims in evidence/handoff "
            "docs that lack an adjacent provenance-probe citation (OMN-13030)"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--staged",
        action="store_true",
        help="Check all staged markdown files (pre-commit mode)",
    )
    group.add_argument(
        "--file",
        type=Path,
        dest="file_path",
        metavar="FILE",
        help="Check a single file",
    )
    args = parser.parse_args(argv)

    if args.staged:
        violations = check_staged()
    else:
        violations = check_file(args.file_path)

    if violations:
        print(
            "EVIDENCE-PROVENANCE GATE: deployed-SHA claim without adjacent "
            "provenance-probe citation",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nProve the deployed SHA against the image: "
            "`docker exec <container> cat /app/build-provenance.json` "
            "(OMN-13030).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
