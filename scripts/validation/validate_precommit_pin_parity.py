#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
CI/pre-commit check: pin-parity ratchet between .pre-commit-config.yaml and the
CI workflow(s) that independently pin the SAME omnibase_core validator
(OMN-14667, WS7 fan-out #3 of OMN-14655; DRIFT-3 recurrence guard).

The problem this guards: a pre-commit `repo:` hook pins an omnibase_core `rev:`
that clones the validator at one SHA, while a dedicated CI ratchet workflow
(`uv run --with "omnibase-core @ git+...@<sha>"`) pins the SAME validator at a
DIFFERENT SHA. Both surfaces then enforce a DIFFERENT frozen baseline, so a
change that is green locally can be red in CI (or vice-versa) purely because the
two pins drifted -- staleness by construction. This gate fails closed the moment
a pinned pair diverges, on either side.

Adaptation from the omnimarket/omniclaude canary (which scanned a single
`ci.yml`): omnibase_infra pins its per-validator core SHAs in *dedicated* gate
workflow files, one validator per file (canonical-inference-gate.yml,
url-authority-gate.yml, ...), and ci.yml pins NO validator core SHA. So each
PIN_PAIRS row names the SPECIFIC CI workflow file to scan, giving a strict 1:1
hook<->gate comparison. A flat "scan every workflow for any core pin" (the
canary's shape) would cross-contaminate here -- one hook's rev would be compared
against every other validator's unrelated pin and false-fail.

PIN_PAIRS below is a small, explicitly-verified table -- add a new pair only
after confirming (by hand, via `git diff <old-rev> <new-rev>` in omnibase_core)
that both sides really do reference the same validator, not two independently
pinned tools that happen to share an upstream repo.

KNOWN DRIFT (intentionally NOT yet enforced here): the `check-url-authority`
pre-commit hook pins omnibase_core `be4f95460dcd6865264ab0713a5b1cc48b41aef9`
while `.github/workflows/url-authority-gate.yml` pins the same
`validator_url_authority` at `8a53a063bd28f643d08b4cbbc6dd5c7c9f6435df`. That is
a live DRIFT-3 this gate is designed to catch, but resolving it requires bumping
one side's core SHA (a change with its own baseline blast-radius) and is tracked
separately (see PR body / follow-up ticket). Add the url-authority row to
PIN_PAIRS in the SAME change that converges the two SHAs, so this gate lands
green and then stays converged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"

# (pre-commit hook id, pre-commit repo URL, CI workflow file (repo-relative),
#  validator module the pair references [for humans/audit only]) -> both sides
# must resolve to the identical pinned omnibase_core SHA.
PIN_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "check-canonical-inference",
        "https://github.com/OmniNode-ai/omnibase_core",
        ".github/workflows/canonical-inference-gate.yml",
        "validator_canonical_inference",
    ),
)

_CI_PIN_RE = re.compile(
    r"omnibase-core\s*@\s*git\+https://github\.com/OmniNode-ai/omnibase_core"
    r"(?:\.git)?@([0-9a-f]{40})"
)


def _find_hook_rev(config: dict[str, Any], hook_id: str, repo_url: str) -> str | None:
    for repo in config.get("repos", []):
        if repo.get("repo") != repo_url:
            continue
        for hook in repo.get("hooks", []):
            if hook.get("id") == hook_id:
                rev = repo.get("rev")
                return str(rev) if rev is not None else None
    return None


def _find_ci_pins(ci_text: str) -> list[str]:
    return [m.group(1) for m in _CI_PIN_RE.finditer(ci_text)]


def main() -> int:
    if not CONFIG_PATH.is_file():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr)
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    violations: list[str] = []

    for hook_id, repo_url, ci_workflow_rel, validator in PIN_PAIRS:
        ci_workflow_path = REPO_ROOT / ci_workflow_rel
        if not ci_workflow_path.is_file():
            violations.append(
                f"pin-parity: CI workflow {ci_workflow_rel!r} not found "
                f"(hook {hook_id!r}, validator {validator!r}) -- "
                "update PIN_PAIRS or restore the workflow."
            )
            continue

        precommit_rev = _find_hook_rev(config, hook_id, repo_url)
        if precommit_rev is None:
            violations.append(
                f"pin-parity: hook id={hook_id!r} not found under repo={repo_url!r} "
                f"in {CONFIG_PATH.name} -- update PIN_PAIRS or the config."
            )
            continue

        ci_pins = _find_ci_pins(ci_workflow_path.read_text(encoding="utf-8"))
        if not ci_pins:
            violations.append(
                f"pin-parity: no CI-pinned SHA found in {ci_workflow_rel} "
                f"(hook {hook_id!r}, validator {validator!r}) -- "
                "update PIN_PAIRS or restore the CI pin."
            )
            continue

        mismatched = sorted({p for p in ci_pins if p != precommit_rev})
        if mismatched:
            violations.append(
                f"pin-parity: hook {hook_id!r} pins rev={precommit_rev} in "
                f"{CONFIG_PATH.name}, but {ci_workflow_rel} pins {mismatched} "
                f"for the same validator ({validator}). These must match -- "
                "bump whichever side is stale."
            )

    if violations:
        print(f"FAIL: {len(violations)} pin-parity violation(s):\n")
        for v in violations:
            print(f"  {v}\n")
        return 1

    print("OK: all pinned revs in PIN_PAIRS match their CI-pinned counterpart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
