#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-push deploy-scope DoD parity gate (OMN-14681, WS7 fan-out of OMN-14655).

Shift the hosted **Deploy Gate** left. The hosted required check
(`deploy-gate` / `deploy-gate.yml` -> the canonical
`validate_pr_deploy_required.py` reusable, CLAUDE.md PR CI Requirements Sec.4)
was the ONLY surface that caught omnibase_infra#2319: a PR that touched
``handler_wiring.py`` (a deploy-scoped runtime surface) with **no** deploy-scope
DoD item in its OCC contract. The drift was invisible locally until the PR hit
CI. This hook runs the deterministic, locally-decidable half of that gate at
``git push`` time so the same gap is caught before the push leaves the machine.

DRY, NOT a re-implementation (root CLAUDE.md Rule #7a / OMN-14655 DRIFT-3):
this hook does not copy the runtime path patterns or the deploy-evidence rule.
It **imports the exact canonical validator CI runs** --
``omniclaude/.github/actions/deploy-gate/validate_pr_deploy_required.py`` --
resolved from the ``OMNI_HOME`` sibling clone, and calls its
``find_runtime_paths`` / ``parse_evidence_metadata`` / ``has_deploy_evidence``
functions directly. There is exactly one rule set; the local hook and the
hosted gate cannot drift on the deploy-scope patterns or the evidence rule.

Tri-state decision (fail-loud on the locally-decidable gaps, NEVER false-red):

  * ``SKIP_NO_RUNTIME``  -- no deploy-scoped file changed. Exit 0 (parity with the
    hosted gate's ``skipped`` branch).
  * ``FAIL_NO_TICKET``   -- deploy-scoped surface touched but the push cites no
    ``OMN-XXXX`` ticket at all. Deterministic CI failure; exit 1.
  * ``FAIL_NO_EVIDENCE`` -- deploy-scoped surface touched, a ticket IS cited, its
    OCC contract IS present locally, but it declares NO deploy-scope DoD evidence.
    This is exactly the omnibase_infra#2319 gap, now caught locally; exit 1.
  * ``PASS_EVIDENCE``    -- deploy-scoped surface touched, cited ticket's local OCC
    contract declares deploy-scope DoD evidence. Exit 0 (parity pass).
  * ``NOTICE_COMPANION_UNMERGED`` -- deploy-scoped surface touched, a ticket IS
    cited, but its OCC contract is NOT resolvable in the local
    ``onex_change_control`` clone (the OCC companion is authored/merged
    separately, so absence-locally is NOT proof-of-absence-in-PR). Emits a loud,
    actionable NOTICE and exits 0. This is the deliberate boundary: the FULL
    PR-body + Evidence-Source OCC-ref resolution (``gh pr view`` /
    ``checkout-occ-contracts.sh``) stays a CI-only concern (OMN-14655 split
    principle: deploy-gate's PR-body/OCC-checkout half is CI-only). Failing red
    here would false-fail every legitimate push whose companion has not merged.

FAIL-LOUD (root CLAUDE.md Rule #8, OMN-14655 principle #4): a gate that cannot
run must be indistinguishable from a failing one. If the diff base cannot be
resolved, or the canonical validator cannot be imported from the ``OMNI_HOME``
sibling clone, this hook HARD-ERRORS (exit 1) with a remediation message -- it
never degrades to a green skip.

Honest drift note: CI resolves the validator at ``omniclaude@main``; this local
hook imports the ``OMNI_HOME/omniclaude`` sibling clone (tracking ``dev``). For
``RUNTIME_PATH_PATTERNS`` and the evidence rule these are effectively identical,
but a same-session ``git -C "$OMNI_HOME/omniclaude" pull`` keeps them converged.

Manual / test invocation (all git+gh gathering is bypassed when the three
override args are supplied, which is how the unit test drives the pure core):

    uv run python scripts/ci/check_deploy_scope_dod.py \
        --changed-files "src/omnibase_infra/runtime/service_kernel.py" \
        --pr-body "Closes OMN-1234" \
        --contracts-dir /path/to/onex_change_control/contracts
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# Relative location of the canonical deploy-gate validator inside the omniclaude
# sibling clone. This is the SAME file the hosted reusable workflow sparse-checks
# out and executes (deploy-gate-reusable.yml), so importing it here is byte-DRY.
_CANONICAL_VALIDATOR_RELPATH = Path(
    "omniclaude/.github/actions/deploy-gate/validate_pr_deploy_required.py"
)
# Local OCC contracts tree, relative to OMNI_HOME. The hosted gate reads contracts
# from onex_change_control (OMN-11423); locally we read the sibling clone.
_OCC_CONTRACTS_RELPATH = Path("onex_change_control/contracts")

_DEFAULT_BASE_REF = "origin/dev"

# Decision outcomes.
SKIP_NO_RUNTIME = "SKIP_NO_RUNTIME"
FAIL_NO_TICKET = "FAIL_NO_TICKET"
FAIL_NO_EVIDENCE = "FAIL_NO_EVIDENCE"
PASS_EVIDENCE = "PASS_EVIDENCE"
NOTICE_COMPANION_UNMERGED = "NOTICE_COMPANION_UNMERGED"


class DeployScopeHookError(RuntimeError):
    """Raised when the hook cannot run and must fail loud (exit 1)."""


@dataclass(frozen=True)
class DeployScopeDecision:
    """Outcome of classifying one push against the deploy-scope DoD rule."""

    outcome: str
    exit_code: int
    runtime_hits: tuple[str, ...]
    tickets: tuple[str, ...]
    detail: str


def _log(message: str) -> None:
    print(f"[deploy-scope-dod] {message}", file=sys.stderr)


def _run_git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DeployScopeHookError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def resolve_omni_home(repo_root: Path) -> Path:
    """Resolve OMNI_HOME so the canonical omniclaude validator can be imported.

    Prefers the ``OMNI_HOME`` env var; falls back to walking up from the repo
    root looking for a directory that contains the canonical validator. This
    keeps the hook correct from BOTH the canonical clone
    (``$OMNI_HOME/omnibase_infra``) and a nested worktree
    (``$OMNI_HOME/omni_worktrees/<ticket>/omnibase_infra``). Fails loud when no
    candidate contains the validator (OMN-14655 principle #4).
    """
    candidates: list[Path] = []
    env_home = os.environ.get("OMNI_HOME")
    if env_home:
        candidates.append(Path(env_home))
    candidates.extend([repo_root, *repo_root.parents])

    for candidate in candidates:
        if (candidate / _CANONICAL_VALIDATOR_RELPATH).is_file():
            return candidate

    raise DeployScopeHookError(
        "cannot locate the canonical deploy-gate validator "
        f"({_CANONICAL_VALIDATOR_RELPATH}) under OMNI_HOME or any parent of "
        f"{repo_root}. This hook mirrors the hosted deploy-gate by importing the "
        "SAME validator CI runs; it does not re-implement the rule. "
        "REMEDIATION: ensure the omniclaude sibling clone exists under OMNI_HOME "
        '(git -C "$OMNI_HOME/omniclaude" pull --ff-only) and OMNI_HOME is set.'
    )


def load_canonical_validator(omni_home: Path) -> ModuleType:
    """Import the canonical deploy-gate validator module by file path (DRY)."""
    validator_path = omni_home / _CANONICAL_VALIDATOR_RELPATH
    spec = importlib.util.spec_from_file_location(
        "_deploy_gate_canonical_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise DeployScopeHookError(
            f"could not build an import spec for {validator_path}"
        )
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so the module's own @dataclass decorators can resolve
    # cls.__module__ via sys.modules (dataclasses._is_type looks it up there).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - defensive import guard
        sys.modules.pop(spec.name, None)
        raise DeployScopeHookError(
            f"failed to import canonical validator {validator_path}: {exc}"
        ) from exc
    return module


def _resolve_base_sha(repo_root: Path, base_ref: str) -> str:
    """Resolve the merge-base against ``base_ref``, best-effort fetch first.

    Mirrors scripts/hooks/prepush_smart_tests.sh: an online push refreshes the
    base ref; offline is tolerated only when it already resolves locally; an
    entirely unresolvable base HARD-ERRORS rather than diffing against nothing.
    """
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "fetch",
            "--quiet",
            "origin",
            base_ref.removeprefix("origin/"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    verify = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{base_ref}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        raise DeployScopeHookError(
            f"base ref '{base_ref}' could not be resolved. "
            f"REMEDIATION: fetch it (git fetch origin {base_ref.removeprefix('origin/')}) "
            "or set DEPLOY_SCOPE_BASE_REF to a resolvable ref."
        )
    merge_base = _run_git(["merge-base", base_ref, "HEAD"], repo_root).strip()
    if not merge_base:
        raise DeployScopeHookError(
            f"no common ancestor between '{base_ref}' and HEAD. "
            f"REMEDIATION: rebase your branch onto {base_ref}."
        )
    return merge_base


def compute_changed_files(repo_root: Path, base_ref: str) -> list[str]:
    """Return the files changed on this branch relative to ``base_ref``."""
    base_sha = _resolve_base_sha(repo_root, base_ref)
    diff = _run_git(["diff", "--name-only", base_sha, "HEAD"], repo_root)
    return [line for line in diff.splitlines() if line.strip()]


def resolve_pr_body(repo_root: Path, base_ref: str) -> str:
    """Best-effort text proxy for the PR body: live PR body if a PR exists,
    else the branch commit messages (what the author will paste into the PR).

    Never fatal: a gh/network failure falls back to commit messages so the hook
    stays offline-tolerant. The ticket-citation this text feeds is deterministic
    either way because the branch/commit convention embeds an ``OMN-`` id.
    """
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root).strip()
    gh_body = _try_gh_pr_body(repo_root, branch)
    if gh_body is not None:
        return gh_body
    try:
        base_sha = _resolve_base_sha(repo_root, base_ref)
        commit_log = _run_git(
            ["log", "--format=%s%n%b", f"{base_sha}..HEAD"], repo_root
        )
    except DeployScopeHookError:
        commit_log = ""
    # Branch name typically encodes the ticket too (jonah/omn-14681-...).
    return f"{branch}\n{commit_log}"


def _try_gh_pr_body(repo_root: Path, branch: str) -> str | None:
    result = subprocess.run(
        ["gh", "pr", "view", branch, "--json", "body", "-q", ".body"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=15,
    )
    if result.returncode != 0:
        return None
    body = result.stdout.strip()
    return body or None


def classify_deploy_scope(
    validator: ModuleType,
    changed_files: list[str],
    pr_body: str,
    contracts_dir: Path,
) -> DeployScopeDecision:
    """Pure decision core -- imports zero git/gh state, so it is unit-testable.

    Reuses the canonical validator's own functions so the deploy-scope patterns
    and the deploy-evidence rule are single-sourced with the hosted gate.
    """
    runtime_hits = validator.find_runtime_paths(changed_files)
    if not runtime_hits:
        return DeployScopeDecision(
            outcome=SKIP_NO_RUNTIME,
            exit_code=0,
            runtime_hits=(),
            tickets=(),
            detail="No deploy-scoped runtime paths touched.",
        )

    metadata = validator.parse_evidence_metadata(pr_body)
    if metadata.source and metadata.ticket:
        tickets = [metadata.ticket]
    else:
        tickets = [f"OMN-{m}" for m in validator.TICKET_PATTERN.findall(pr_body)]
    # De-duplicate while preserving order.
    tickets = list(dict.fromkeys(tickets))

    if not tickets:
        return DeployScopeDecision(
            outcome=FAIL_NO_TICKET,
            exit_code=1,
            runtime_hits=tuple(runtime_hits),
            tickets=(),
            detail=(
                "Deploy-scoped surface touched but this push cites no OMN-XXXX "
                "ticket. The hosted deploy-gate will fail on the same gap."
            ),
        )

    present = [t for t in tickets if (contracts_dir / f"{t}.yaml").is_file()]
    if not present:
        return DeployScopeDecision(
            outcome=NOTICE_COMPANION_UNMERGED,
            exit_code=0,
            runtime_hits=tuple(runtime_hits),
            tickets=tuple(tickets),
            detail=(
                "Deploy-scoped surface touched; cited ticket(s) "
                f"{tickets} have no OCC contract in the local "
                f"{contracts_dir} clone (companion likely not merged yet). "
                "Cannot prove the gap locally -- the hosted deploy-gate resolves "
                "the OCC companion via Evidence-Source and makes the final call."
            ),
        )

    if any(validator.has_deploy_evidence(contracts_dir / f"{t}.yaml") for t in present):
        return DeployScopeDecision(
            outcome=PASS_EVIDENCE,
            exit_code=0,
            runtime_hits=tuple(runtime_hits),
            tickets=tuple(present),
            detail=(
                f"Deploy-scoped surface touched; ticket(s) {present} declare "
                "deploy-scope DoD evidence in their OCC contract."
            ),
        )

    return DeployScopeDecision(
        outcome=FAIL_NO_EVIDENCE,
        exit_code=1,
        runtime_hits=tuple(runtime_hits),
        tickets=tuple(present),
        detail=(
            f"Deploy-scoped surface touched; cited ticket(s) {present} have an "
            "OCC contract but declare NO deploy-scope DoD evidence "
            "(the omnibase_infra#2319 gap)."
        ),
    )


def render(decision: DeployScopeDecision, validator: ModuleType) -> None:
    """Print a human-actionable report for the decision to stderr."""
    if decision.outcome == SKIP_NO_RUNTIME:
        return  # quiet parity with the hosted gate's skip branch

    _log(f"outcome={decision.outcome}")
    _log(f"deploy-scoped files: {list(decision.runtime_hits)}")
    _log(decision.detail)

    if decision.exit_code != 0:
        _log("")
        _log(getattr(validator, "DEPLOY_EVIDENCE_GUIDANCE", ""))
        _log("")
        _log(
            "This is the LOCAL mirror of the required hosted `deploy-gate` check. "
            "Fix it before pushing: add a deploy-scope DoD item to the cited "
            "ticket's OCC contract (onex_change_control/contracts/OMN-XXXX.yaml)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local pre-push mirror of the hosted deploy-gate deploy-scope DoD "
            "check (OMN-14681)."
        )
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Space-separated changed-file override (bypasses git; for tests).",
    )
    parser.add_argument(
        "--pr-body",
        default=None,
        help="PR-body-proxy override (bypasses gh/git; for tests).",
    )
    parser.add_argument(
        "--contracts-dir",
        default=None,
        help="OCC contracts dir override (bypasses OMNI_HOME resolution).",
    )
    parser.add_argument(
        "--omni-home",
        default=None,
        help="OMNI_HOME override for locating the canonical validator.",
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("DEPLOY_SCOPE_BASE_REF", _DEFAULT_BASE_REF),
        help="git ref to diff against (default: origin/dev).",
    )
    args = parser.parse_args(argv)

    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        _log(f"ERROR: not inside a git worktree: {exc}")
        return 1

    try:
        omni_home = (
            Path(args.omni_home) if args.omni_home else resolve_omni_home(repo_root)
        )
        validator = load_canonical_validator(omni_home)

        if args.changed_files is not None:
            changed_files = [f for f in args.changed_files.split() if f]
        else:
            changed_files = compute_changed_files(repo_root, args.base_ref)

        if args.contracts_dir is not None:
            contracts_dir = Path(args.contracts_dir)
        else:
            contracts_dir = omni_home / _OCC_CONTRACTS_RELPATH

        if args.pr_body is not None:
            pr_body = args.pr_body
        else:
            pr_body = resolve_pr_body(repo_root, args.base_ref)
    except DeployScopeHookError as exc:
        _log(f"ERROR (fail-loud): {exc}")
        return 1

    decision = classify_deploy_scope(
        validator=validator,
        changed_files=changed_files,
        pr_body=pr_body,
        contracts_dir=contracts_dir,
    )
    render(decision, validator)
    return decision.exit_code


if __name__ == "__main__":
    sys.exit(main())
