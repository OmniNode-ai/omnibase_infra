#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-push deploy-scope DoD parity gate (OMN-14681, WS7 fan-out of OMN-14655).

Shift the hosted **Deploy Gate** left. The hosted required check
(`deploy-gate` / `deploy-gate.yml` -> the canonical
`validate_pr_deploy_required.py` reusable, CLAUDE.md PR CI Requirements Sec.4)
was the ONLY surface that caught omnibase_infra#2319: a PR that touched
``handler_wiring.py`` (a deploy-scoped runtime surface) with **no** deploy-scope
DoD item in its ticket contract. The drift was invisible locally until the PR hit
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

Tri-state decision (fail-loud on the locally-decidable gaps, NEVER false-red).
The cited tickets' contracts are this repository's own ``contracts/<ticket>.yaml``
-- the directory the hosted gate reads at the pull request head
(``contract-source: caller``, OMN-20074) -- so every gap below is decidable here:

  * ``SKIP_NO_RUNTIME``  -- no deploy-scoped file changed. Exit 0 (parity with the
    hosted gate's ``skipped`` branch).
  * ``FAIL_NO_TICKET``   -- deploy-scoped surface touched but the push cites no
    ``OMN-XXXX`` ticket at all. Deterministic CI failure; exit 1.
  * ``FAIL_NO_EVIDENCE`` -- deploy-scoped surface touched and either no cited
    ticket has a ``contracts/<ticket>.yaml``, or a contract is present but declares
    NO deploy-scope DoD evidence (the omnibase_infra#2319 gap); exit 1.
  * ``PASS_EVIDENCE``    -- deploy-scoped surface touched, a cited ticket's contract
    declares deploy-scope DoD evidence. Exit 0 (parity pass).

FAIL-LOUD (root CLAUDE.md Rule #8, OMN-14655 principle #4): a gate that cannot
run must be indistinguishable from a failing one. If the diff base cannot be
resolved, or the canonical validator cannot be imported from the ``OMNI_HOME``
sibling clone, this hook HARD-ERRORS (exit 1) with a remediation message -- it
never degrades to a green skip.

Honest drift note: CI resolves the validator at ``omniclaude@main``; this local
hook imports the ``OMNI_HOME/omniclaude`` sibling clone (tracking ``dev``). For
``RUNTIME_PATH_PATTERNS`` and the evidence rule these are effectively identical,
but a same-session ``git -C "$OMNI_HOME/omniclaude" pull`` keeps them converged.

Ticket-contract leg (OMN-20870). The hosted ``repo-evidence / dod-verify``
context (omnibase_core ``receipt-gate.yml``, job ``dod-verify``, called by
``.github/workflows/call-repo-evidence-gate.yml``) refuses a pull request whose
cited ticket has no ``contracts/<ticket>.yaml`` at the head, and the verifier it
installs refuses a check whose value is prose
(``INVALID_CHECK_VALUE_NOT_A_COMMAND``). Both reds were reaching CI on every new
PR that missed them. This leg refuses the same two classes at ``git push``:

  * ``FAIL_NO_CONTRACT`` -- a ticket cited in the branch's commit subjects (the
    local stand-in for the PR title the hosted job reads) has no
    ``contracts/<ticket>.yaml`` in the pushed HEAD commit. The file is named.
  * ``FAIL_CHECK_NOT_A_COMMAND`` -- a ``command`` / ``test_passes`` check in a
    head-side contract is judged prose by the verifier's own shape guard. The
    check is named by contract and evidence id.

The shape verdict is not re-implemented here. It is the released verifier's
``_invalid_check_value_reason``, run from ``omnimarket==<verifier-version>``,
where the version is read from the caller workflow so the local and hosted
verdicts use one pin. The leg runs no bound check: it only judges shape. Bot
authorship is not consulted -- a push has no GitHub-verified author, and the
hosted job's bot exemptions are derived from that metadata alone, so they stay
CI-only.

Manual / test invocation (all git+gh gathering is bypassed when the three
override args are supplied, which is how the unit test drives the pure core):

    uv run python scripts/ci/check_deploy_scope_dod.py \
        --changed-files "src/omnibase_infra/runtime/service_kernel.py" \
        --pr-body "Closes OMN-1234" \
        --contracts-dir /path/to/contracts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import yaml

# Relative location of the canonical deploy-gate validator inside the omniclaude
# sibling clone. This is the SAME file the hosted reusable workflow sparse-checks
# out and executes (deploy-gate-reusable.yml), so importing it here is byte-DRY.
_CANONICAL_VALIDATOR_RELPATH = Path(
    "omniclaude/.github/actions/deploy-gate/validate_pr_deploy_required.py"
)
# This repository's own ticket contracts. The hosted deploy gate reads
# ``contracts/<ticket>.yaml`` at the pull request head (``contract-source:
# caller``, OMN-20074); locally we read the same directory of the working tree.
_CONTRACTS_RELPATH = Path("contracts")

_DEFAULT_BASE_REF = "origin/dev"

# Decision outcomes.
SKIP_NO_RUNTIME = "SKIP_NO_RUNTIME"
FAIL_NO_TICKET = "FAIL_NO_TICKET"
FAIL_NO_EVIDENCE = "FAIL_NO_EVIDENCE"
PASS_EVIDENCE = "PASS_EVIDENCE"


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


#: The canonical-validator surface this hook actually calls. Checked after the
#: import so a STALE sibling clone fails with its own name instead of an
#: ``AttributeError`` raised several frames deep in ``classify_deploy_scope``.
#:
#: OMN-16989: the resolver's fallback walks EVERY parent of the repo root, so
#: whichever directory named ``omniclaude`` sits nearest above the checkout wins
#: -- with no assertion that it is the sibling of THIS repo's canonical clone or
#: that it is anywhere near current. Measured on `.201` over the OMN-16991
#: remote leg: the transplanted tree lives at
#: ``/data/omninode/onex-prepush/runs/<id>/tree``, the walk reached
#: ``/data/omninode`` and imported an omniclaude clone pinned at omniclaude#1600
#: (the host's up-to-date clone, #1963, is at ``/data/omninode/omni_home`` and is
#: NOT an ancestor). Four tests then failed on
#: ``module '_deploy_gate_canonical_validator' has no attribute
#: 'parse_evidence_metadata'`` -- a red about another repo's clone freshness on
#: whichever host ran the suite, with nothing in the message saying so.
_REQUIRED_VALIDATOR_ATTRS = (
    "find_runtime_paths",
    "parse_evidence_metadata",
    "TICKET_PATTERN",
)


def load_canonical_validator(omni_home: Path) -> ModuleType:
    """Import the canonical deploy-gate validator module by file path (DRY).

    Raises ``DeployScopeHookError`` when the resolved clone does not expose the
    surface this hook calls -- a named, actionable failure instead of a silent
    dependence on another repo's clone freshness.
    """
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
    missing = [a for a in _REQUIRED_VALIDATOR_ATTRS if not hasattr(module, a)]
    if missing:
        sys.modules.pop(spec.name, None)
        raise DeployScopeHookError(
            f"the canonical deploy-gate validator at {validator_path} is STALE: "
            f"it does not expose {', '.join(missing)}. This hook mirrors the "
            "hosted gate by importing that exact file, so an out-of-date clone "
            "silently changes the verdict. "
            'REMEDIATION: git -C "$(dirname "$(dirname "$(dirname "$(dirname '
            f'"$(dirname {validator_path})")")")")" pull --ff-only, or set '
            "OMNI_HOME to the clone you intend this hook to read."
        )
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


def contracts_dir_for(repo_root: Path) -> Path:
    """The directory the hosted deploy gate reads the cited tickets' contracts from."""
    return repo_root / _CONTRACTS_RELPATH


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
        missing = ", ".join(f"contracts/{t}.yaml" for t in tickets)
        return DeployScopeDecision(
            outcome=FAIL_NO_EVIDENCE,
            exit_code=1,
            runtime_hits=tuple(runtime_hits),
            tickets=tuple(tickets),
            detail=(
                "Deploy-scoped surface touched; cited ticket(s) "
                f"{tickets} have no contract in this repository ({missing}). "
                "The hosted deploy-gate reads the contract at the pull request "
                "head and fails on the same gap."
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
                "deploy-scope DoD evidence in their contract."
            ),
        )

    return DeployScopeDecision(
        outcome=FAIL_NO_EVIDENCE,
        exit_code=1,
        runtime_hits=tuple(runtime_hits),
        tickets=tuple(present),
        detail=(
            f"Deploy-scoped surface touched; cited ticket(s) {present} have a "
            "contract but declare NO deploy-scope DoD evidence "
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
            "ticket's contract (contracts/OMN-XXXX.yaml)."
        )


# ---------------------------------------------------------------------------
# Ticket-contract leg (OMN-20870)
# ---------------------------------------------------------------------------

#: The repo-evidence caller; its ``verifier-version`` input is the one pin.
_REPO_EVIDENCE_CALLER_RELPATH = Path(".github/workflows/call-repo-evidence-gate.yml")
#: The same ticket pattern the hosted job greps out of the PR title.
_CITED_TICKET_RE = re.compile(r"OMN-[0-9]+")
#: The check types the verifier hands to its shape guard before running them.
_COMMAND_CHECK_TYPES = ("command", "test_passes")
#: The Python the hosted job installs the verifier into.
_VERIFIER_PYTHON = "3.13"
#: Runs inside the released verifier's environment: reads a JSON list of
#: check values on stdin, writes the guard's reason (or null) for each.
_VERIFIER_SHAPE_PROGRAM = (
    "import json, sys\n"
    "from omnimarket.nodes.node_dod_verify.services.evidence_collector "
    "import _invalid_check_value_reason\n"
    "request = json.load(sys.stdin)\n"
    "json.dump([_invalid_check_value_reason(v, cwd=request['cwd']) "
    "for v in request['values']], sys.stdout)\n"
)

PASS_CONTRACTS = "PASS_CONTRACTS"
NOTICE_NO_TICKET_CITED = "NOTICE_NO_TICKET_CITED"
FAIL_NO_CONTRACT = "FAIL_NO_CONTRACT"
FAIL_CHECK_NOT_A_COMMAND = "FAIL_CHECK_NOT_A_COMMAND"

#: Judges a batch of check values; returns one reason (or ``None``) per value.
ShapeJudge = Callable[[Sequence[str]], list[str | None]]


@dataclass(frozen=True)
class TicketContractDecision:
    """Outcome of judging the cited tickets' head-side contracts."""

    outcome: str
    exit_code: int
    tickets: tuple[str, ...]
    findings: tuple[str, ...]


def cited_tickets(subjects: str) -> list[str]:
    """Ticket ids cited in ``subjects``, sorted and unique like the hosted grep."""
    return sorted(set(_CITED_TICKET_RE.findall(subjects)))


def resolve_branch_subjects(repo_root: Path, base_ref: str) -> str:
    """Commit subjects on this branch's own line since ``base_ref``.

    ``--first-parent --no-merges`` keeps the subjects of a merged-in base
    branch out, so a branch that merged dev does not cite dev's tickets.
    """
    base_sha = _resolve_base_sha(repo_root, base_ref)
    return _run_git(
        ["log", "--first-parent", "--no-merges", "--format=%s", f"{base_sha}..HEAD"],
        repo_root,
    )


def read_head_contract(repo_root: Path, ticket: str) -> str | None:
    """``contracts/<ticket>.yaml`` as committed at HEAD, or ``None`` if absent.

    Reads the commit being pushed, not the working tree, so an uncommitted
    contract does not pass a push that does not carry it.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:contracts/{ticket}.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def command_checks(contract_text: str) -> list[tuple[str, str]]:
    """``(evidence id, check value)`` for every check the verifier shape-guards.

    Mirrors the verifier's selection: ``check_type`` in ``command`` /
    ``test_passes``, value from ``command`` else ``check_value``.
    """
    try:
        contract = yaml.safe_load(contract_text)
    except yaml.YAMLError as exc:
        raise DeployScopeHookError(f"contract is not valid YAML: {exc}") from exc
    if not isinstance(contract, dict):
        raise DeployScopeHookError("contract is not a YAML mapping")
    items = contract.get("dod_evidence") or []
    if not isinstance(items, list):
        raise DeployScopeHookError("dod_evidence is not a list")
    found: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id", "unknown"))
        for check in item.get("checks") or []:
            if not isinstance(check, dict):
                continue
            if check.get("check_type") not in _COMMAND_CHECK_TYPES:
                continue
            value = check.get("command") or check.get("check_value") or ""
            found.append((evidence_id, str(value)))
    return found


def classify_ticket_contracts(
    tickets: Sequence[str],
    read_contract: Callable[[str], str | None],
    judge: ShapeJudge,
) -> TicketContractDecision:
    """Pure decision core: presence first, then the verifier's shape guard."""
    if not tickets:
        return TicketContractDecision(
            outcome=NOTICE_NO_TICKET_CITED,
            exit_code=0,
            tickets=(),
            findings=(
                "No OMN-XXXX ticket in this branch's commit subjects. The hosted "
                "repo-evidence / dod-verify reads the PR title, which must cite one.",
            ),
        )

    contracts: dict[str, str] = {}
    missing: list[str] = []
    for ticket in tickets:
        text = read_contract(ticket)
        if text is None:
            missing.append(ticket)
        else:
            contracts[ticket] = text
    if missing:
        return TicketContractDecision(
            outcome=FAIL_NO_CONTRACT,
            exit_code=1,
            tickets=tuple(tickets),
            findings=tuple(
                f"this push cites {t} but carries no contracts/{t}.yaml"
                for t in missing
            ),
        )

    labelled: list[tuple[str, str]] = []
    values: list[str] = []
    for ticket, text in contracts.items():
        try:
            checks = command_checks(text)
        except DeployScopeHookError as exc:
            return TicketContractDecision(
                outcome=FAIL_CHECK_NOT_A_COMMAND,
                exit_code=1,
                tickets=tuple(tickets),
                findings=(f"contracts/{ticket}.yaml: {exc}",),
            )
        for evidence_id, value in checks:
            labelled.append(
                (f"contracts/{ticket}.yaml dod_evidence {evidence_id}", value)
            )
            values.append(value)

    reasons = judge(values) if values else []
    if len(reasons) != len(values):
        raise DeployScopeHookError(
            f"the verifier judged {len(reasons)} of {len(values)} check values"
        )
    findings = [
        f"{label}: INVALID_CHECK_VALUE_NOT_A_COMMAND: {reason} (check value: {value!r})"
        for (label, value), reason in zip(labelled, reasons, strict=True)
        if reason is not None
    ]
    if findings:
        return TicketContractDecision(
            outcome=FAIL_CHECK_NOT_A_COMMAND,
            exit_code=1,
            tickets=tuple(tickets),
            findings=tuple(findings),
        )
    return TicketContractDecision(
        outcome=PASS_CONTRACTS,
        exit_code=0,
        tickets=tuple(tickets),
        findings=(),
    )


def read_verifier_version(repo_root: Path) -> str:
    """The ``verifier-version`` the repo-evidence caller pins."""
    caller = repo_root / _REPO_EVIDENCE_CALLER_RELPATH
    try:
        workflow = yaml.safe_load(caller.read_text(encoding="utf-8"))
        version = workflow["jobs"]["repo-evidence"]["with"]["verifier-version"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise DeployScopeHookError(
            f"cannot read verifier-version from {_REPO_EVIDENCE_CALLER_RELPATH}: {exc}"
        ) from exc
    if not isinstance(version, str) or not version.strip():
        raise DeployScopeHookError(
            f"{_REPO_EVIDENCE_CALLER_RELPATH} pins no verifier-version"
        )
    return version.strip()


def verifier_shape_argv(verifier_version: str) -> list[str]:
    """The command that runs the released verifier's shape guard."""
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        _VERIFIER_PYTHON,
        "--with",
        f"omnimarket=={verifier_version}",
        "python",
        "-I",
        "-c",
        _VERIFIER_SHAPE_PROGRAM,
    ]


def pinned_verifier_judge(verifier_version: str, cwd: Path) -> ShapeJudge:
    """A judge that asks the pinned released verifier, failing loud if it cannot."""

    def judge(values: Sequence[str]) -> list[str | None]:
        result = subprocess.run(
            verifier_shape_argv(verifier_version),
            input=json.dumps({"values": list(values), "cwd": str(cwd)}),
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )
        if result.returncode != 0:
            raise DeployScopeHookError(
                f"omnimarket=={verifier_version} shape guard did not run "
                f"(rc={result.returncode}): {result.stderr.strip()[-2000:]}"
            )
        try:
            reasons = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DeployScopeHookError(
                f"omnimarket=={verifier_version} shape guard returned no JSON: {exc}"
            ) from exc
        if not isinstance(reasons, list) or not all(
            r is None or isinstance(r, str) for r in reasons
        ):
            raise DeployScopeHookError(
                f"omnimarket=={verifier_version} shape guard returned {reasons!r}"
            )
        return list(reasons)

    return judge


def render_ticket_contracts(decision: TicketContractDecision) -> None:
    """Print the ticket-contract leg's verdict to stderr."""
    if decision.outcome == PASS_CONTRACTS:
        _log(f"ticket contracts: {list(decision.tickets)} present and command-shaped")
        return
    _log(f"outcome={decision.outcome}")
    for finding in decision.findings:
        _log(finding)
    if decision.exit_code != 0:
        _log(
            "This is the LOCAL mirror of the required hosted `repo-evidence / "
            "dod-verify` check, which refuses the same head. Commit "
            "contracts/<ticket>.yaml with command-shaped check values before pushing."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local pre-push mirror of the hosted deploy-gate deploy-scope DoD "
            "check (OMN-14681) and of repo-evidence / dod-verify's ticket-contract "
            "presence and check-shape refusals (OMN-20870)."
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
        help="Ticket contracts dir override (default: <repo root>/contracts).",
    )
    parser.add_argument(
        "--omni-home",
        default=None,
        help="OMNI_HOME override for locating the canonical validator.",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Commit-subject override for the ticket-contract leg (bypasses git).",
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
            contracts_dir = contracts_dir_for(repo_root)

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

    try:
        subjects = (
            args.subjects
            if args.subjects is not None
            else resolve_branch_subjects(repo_root, args.base_ref)
        )
        contract_decision = classify_ticket_contracts(
            tickets=cited_tickets(subjects),
            read_contract=lambda ticket: read_head_contract(repo_root, ticket),
            judge=pinned_verifier_judge(read_verifier_version(repo_root), repo_root),
        )
    except DeployScopeHookError as exc:
        _log(f"ERROR (fail-loud): {exc}")
        return 1
    render_ticket_contracts(contract_decision)
    return max(decision.exit_code, contract_decision.exit_code)


if __name__ == "__main__":
    sys.exit(main())
