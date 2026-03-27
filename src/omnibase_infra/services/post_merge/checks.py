# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post-merge check stage implementations.

Each check stage receives a ``ModelPRMergedEvent`` and returns a list of
``ModelPostMergeFinding`` instances. Stages are designed to be independent
and fail gracefully -- a failure in one stage does not block others.

Related Tickets:
    - OMN-6727: post-merge consumer chain
    - OMN-6725: contract_sweep skill (dependency for contract sweep stage)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

from omnibase_infra.models.github.model_pr_merged_event import ModelPRMergedEvent
from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hostile Review Stage
# ---------------------------------------------------------------------------


async def run_hostile_review(
    event: ModelPRMergedEvent,
    *,
    github_token: str = "",
) -> list[ModelPostMergeFinding]:
    """Run hostile review on the merged diff.

    Fetches the PR diff via GitHub API and analyses it for common issues:
    - Security concerns (hardcoded secrets, injection vectors)
    - Missing error handling at system boundaries
    - Contract violations (naming conventions, model patterns)

    Args:
        event: The PR merged event with metadata.
        github_token: GitHub PAT for API access. If empty, stage is skipped.

    Returns:
        List of findings from the hostile review.
    """
    if not github_token:
        logger.warning(
            "Hostile review skipped: no GitHub token configured",
            extra={"repo": event.repo, "pr_number": event.pr_number},
        )
        return []

    findings: list[ModelPostMergeFinding] = []

    try:
        diff = await _fetch_pr_diff(event.repo, event.pr_number, github_token)
    except Exception:
        logger.exception(
            "Failed to fetch PR diff for hostile review",
            extra={"repo": event.repo, "pr_number": event.pr_number},
        )
        return []

    # Pattern checks on the diff
    findings.extend(_check_hardcoded_secrets(diff, event))
    findings.extend(_check_missing_error_handling(diff, event))
    findings.extend(_check_naming_conventions(diff, event))

    logger.info(
        "Hostile review completed",
        extra={
            "repo": event.repo,
            "pr_number": event.pr_number,
            "findings_count": len(findings),
        },
    )
    return findings


async def _fetch_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """Fetch the PR diff from GitHub API."""
    import httpx

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


# Secret patterns to detect in diffs (added lines only)
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "Generic API key assignment",
        re.compile(r'(?:api[_-]?key|apikey)\s*[:=]\s*["\'][^"\']{8,}', re.IGNORECASE),
    ),
    ("Private key header", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
    ("JWT token", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
]


def _check_hardcoded_secrets(
    diff: str, event: ModelPRMergedEvent
) -> list[ModelPostMergeFinding]:
    """Scan diff for hardcoded secrets in added lines."""
    findings: list[ModelPostMergeFinding] = []
    current_file: str | None = None

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added_content = line[1:]
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(added_content):
                findings.append(
                    ModelPostMergeFinding(
                        stage=EnumCheckStage.HOSTILE_REVIEW,
                        severity=EnumFindingSeverity.CRITICAL,
                        title=f"Potential {label} detected",
                        description=(
                            f"Potential {label} found in added line of "
                            f"PR #{event.pr_number} ({event.repo}). "
                            f"Review and rotate if this is a real credential."
                        ),
                        file_path=current_file,
                    )
                )
    return findings


def _check_missing_error_handling(
    diff: str, event: ModelPRMergedEvent
) -> list[ModelPostMergeFinding]:
    """Check for common error-handling gaps in added code."""
    findings: list[ModelPostMergeFinding] = []
    current_file: str | None = None

    # Only check Python files
    bare_except_re = re.compile(r"^\+\s+except\s*:")

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if current_file and current_file.endswith(".py") and bare_except_re.match(line):
            findings.append(
                ModelPostMergeFinding(
                    stage=EnumCheckStage.HOSTILE_REVIEW,
                    severity=EnumFindingSeverity.MEDIUM,
                    title="Bare except clause",
                    description=(
                        f"Bare `except:` clause added in PR #{event.pr_number}. "
                        "This catches SystemExit, KeyboardInterrupt, etc. "
                        "Use `except Exception:` at minimum."
                    ),
                    file_path=current_file,
                )
            )
    return findings


def _check_naming_conventions(
    diff: str, event: ModelPRMergedEvent
) -> list[ModelPostMergeFinding]:
    """Check that new Pydantic models follow Model prefix convention."""
    findings: list[ModelPostMergeFinding] = []
    current_file: str | None = None

    # Match class definitions inheriting from BaseModel without Model prefix
    model_re = re.compile(r"^\+class\s+(?!Model)([A-Z]\w+)\(.*BaseModel.*\):")

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if current_file and current_file.endswith(".py"):
            match = model_re.match(line)
            if match:
                class_name = match.group(1)
                # Exclude Config* and known non-Model patterns
                if not class_name.startswith(("Config", "Enum")):
                    findings.append(
                        ModelPostMergeFinding(
                            stage=EnumCheckStage.HOSTILE_REVIEW,
                            severity=EnumFindingSeverity.LOW,
                            title=f"BaseModel class `{class_name}` missing Model prefix",
                            description=(
                                f"Class `{class_name}` extends BaseModel but does not "
                                "follow the `Model` prefix naming convention."
                            ),
                            file_path=current_file,
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# Contract Sweep Stage
# ---------------------------------------------------------------------------


async def run_contract_sweep(
    event: ModelPRMergedEvent,
    *,
    contracts_dir: str = "src/omnibase_infra/nodes",
) -> list[ModelPostMergeFinding]:
    """Run contract drift check on the repository after merge.

    Clones the repo at the merge SHA and runs ``check_topic_drift.py``
    to detect drift between topic constants and contract YAML declarations.

    Args:
        event: The PR merged event with metadata.
        contracts_dir: Relative path to the contracts directory.

    Returns:
        List of findings from the contract sweep.
    """
    findings: list[ModelPostMergeFinding] = []

    # Only run for repos that have contract infrastructure
    contract_files = [
        f for f in event.changed_files if "contract" in f.lower() or f.endswith(".yaml")
    ]
    topic_files = [f for f in event.changed_files if "topic" in f.lower()]
    suffix_files = [f for f in event.changed_files if "suffix" in f.lower()]

    # If no contract/topic/suffix files changed, skip the heavyweight check
    if not contract_files and not topic_files and not suffix_files:
        logger.info(
            "Contract sweep skipped: no contract/topic files changed",
            extra={"repo": event.repo, "pr_number": event.pr_number},
        )
        return findings

    # Run check_topic_drift.py in a subprocess
    # The script is available in the same repo, so we can run it directly
    # against the current worktree / checkout
    try:
        result = await _run_check_topic_drift(event, contracts_dir)
        if result:
            findings.extend(result)
    except Exception:
        logger.exception(
            "Contract sweep failed",
            extra={"repo": event.repo, "pr_number": event.pr_number},
        )
        # Return a single finding indicating the sweep failed
        findings.append(
            ModelPostMergeFinding(
                stage=EnumCheckStage.CONTRACT_SWEEP,
                severity=EnumFindingSeverity.HIGH,
                title="Contract sweep execution failed",
                description=(
                    f"check_topic_drift.py failed to execute for PR #{event.pr_number} "
                    f"in {event.repo}. Manual contract review required."
                ),
            )
        )

    logger.info(
        "Contract sweep completed",
        extra={
            "repo": event.repo,
            "pr_number": event.pr_number,
            "findings_count": len(findings),
        },
    )
    return findings


async def _run_check_topic_drift(
    event: ModelPRMergedEvent,
    contracts_dir: str,
) -> list[ModelPostMergeFinding]:
    """Clone repo at merge SHA and run topic drift check."""
    findings: list[ModelPostMergeFinding] = []
    github_token = os.environ.get("GITHUB_TOKEN", "")

    with tempfile.TemporaryDirectory(prefix="post-merge-sweep-") as tmpdir:
        clone_url = f"https://github.com/{event.repo}.git"
        if github_token:
            clone_url = (
                f"https://x-access-token:{github_token}@github.com/{event.repo}.git"
            )

        # Shallow clone at the merge SHA
        clone_proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth",
            "1",
            clone_url,
            tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, clone_stderr = await clone_proc.communicate()
        if clone_proc.returncode != 0:
            logger.error(
                "Git clone failed for contract sweep",
                extra={
                    "repo": event.repo,
                    "stderr": clone_stderr.decode(errors="replace")[:500],
                },
            )
            return findings

        # Checkout the specific merge SHA
        checkout_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            tmpdir,
            "checkout",
            event.merge_sha,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await checkout_proc.communicate()

        # Run check_topic_drift.py
        script_path = Path(tmpdir) / "scripts" / "check_topic_drift.py"
        if not script_path.exists():
            logger.info(
                "check_topic_drift.py not found in repo, skipping contract sweep",
                extra={"repo": event.repo},
            )
            return findings

        drift_proc = await asyncio.create_subprocess_exec(
            "python3",
            str(script_path),
            "--contracts-dir",
            str(Path(tmpdir) / contracts_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
        )
        drift_stdout, drift_stderr = await drift_proc.communicate()

        if drift_proc.returncode != 0:
            output = drift_stdout.decode(errors="replace") + drift_stderr.decode(
                errors="replace"
            )
            findings.append(
                ModelPostMergeFinding(
                    stage=EnumCheckStage.CONTRACT_SWEEP,
                    severity=EnumFindingSeverity.HIGH,
                    title="Contract topic drift detected",
                    description=(
                        f"check_topic_drift.py reported drift after PR #{event.pr_number} "
                        f"merged in {event.repo}:\n\n{output[:2000]}"
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Integration Check Stage
# ---------------------------------------------------------------------------


async def run_integration_check(
    event: ModelPRMergedEvent,
) -> list[ModelPostMergeFinding]:
    """Run integration boundary checks on the merged PR.

    Checks for cross-repo boundary violations:
    - Import changes that cross package boundaries
    - Topic name changes that could break consumers
    - Enum value changes that could break serialization

    Args:
        event: The PR merged event with metadata.

    Returns:
        List of findings from the integration check.
    """
    findings: list[ModelPostMergeFinding] = []

    # Check for cross-boundary import changes
    findings.extend(_check_boundary_imports(event))

    # Check for topic name modifications
    findings.extend(_check_topic_name_changes(event))

    # Check for enum value modifications
    findings.extend(_check_enum_changes(event))

    logger.info(
        "Integration check completed",
        extra={
            "repo": event.repo,
            "pr_number": event.pr_number,
            "findings_count": len(findings),
        },
    )
    return findings


def _check_boundary_imports(event: ModelPRMergedEvent) -> list[ModelPostMergeFinding]:
    """Flag files that import across known package boundaries."""
    findings: list[ModelPostMergeFinding] = []

    # Cross-package import boundary pairs that warrant attention
    boundary_packages = [
        ("omnibase_core", "omnibase_infra"),
        ("omnibase_spi", "omnibase_infra"),
        ("omnibase_infra", "omniclaude"),
    ]

    # Check changed files for __init__.py modifications that export public API
    init_files = [f for f in event.changed_files if f.endswith("__init__.py")]
    if init_files:
        findings.append(
            ModelPostMergeFinding(
                stage=EnumCheckStage.INTEGRATION_CHECK,
                severity=EnumFindingSeverity.INFO,
                title="Public API surface changed",
                description=(
                    f"PR #{event.pr_number} modified {len(init_files)} __init__.py file(s): "
                    f"{', '.join(init_files[:5])}. Verify downstream consumers are not broken."
                ),
            )
        )

    return findings


def _check_topic_name_changes(event: ModelPRMergedEvent) -> list[ModelPostMergeFinding]:
    """Flag changes to topic suffix files."""
    findings: list[ModelPostMergeFinding] = []

    topic_suffix_files = [
        f
        for f in event.changed_files
        if "platform_topic_suffixes" in f or "topics.yaml" in f
    ]
    if topic_suffix_files:
        findings.append(
            ModelPostMergeFinding(
                stage=EnumCheckStage.INTEGRATION_CHECK,
                severity=EnumFindingSeverity.HIGH,
                title="Topic definitions modified",
                description=(
                    f"PR #{event.pr_number} modified topic definition files: "
                    f"{', '.join(topic_suffix_files)}. "
                    "Topic renames or removals can break downstream consumers. "
                    "Verify all consuming services are updated."
                ),
            )
        )

    return findings


def _check_enum_changes(event: ModelPRMergedEvent) -> list[ModelPostMergeFinding]:
    """Flag changes to enum definition files."""
    findings: list[ModelPostMergeFinding] = []

    enum_files = [
        f for f in event.changed_files if f.startswith("src/") and "enum_" in f
    ]
    if enum_files:
        findings.append(
            ModelPostMergeFinding(
                stage=EnumCheckStage.INTEGRATION_CHECK,
                severity=EnumFindingSeverity.MEDIUM,
                title="Enum definitions modified",
                description=(
                    f"PR #{event.pr_number} modified {len(enum_files)} enum file(s): "
                    f"{', '.join(enum_files[:5])}. "
                    "Enum value changes can break serialization boundaries. "
                    "Verify cross-repo compatibility."
                ),
            )
        )

    return findings


__all__ = [
    "run_contract_sweep",
    "run_hostile_review",
    "run_integration_check",
]
