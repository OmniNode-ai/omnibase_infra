# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-6790: every committed GLM endpoint must be the z.ai CODING PLAN surface.

We hold an ACTIVE z.ai **GLM Coding Plan**. z.ai runs two different products on
one host, and they are not two spellings of one endpoint:

* Coding Plan (**ours**) is served at ``https://api.z.ai/api/coding/paas/v4``.
* Pay-as-you-go (NOT ours) is served at ``https://api.z.ai/api/paas/v4``.

Presenting our Coding-Plan key to the pay-as-you-go surface returns::

    HTTP 429 {"code":"1113","message":"Insufficient balance or no resource
              package. Please recharge."}

That is a **WRONG-ENDPOINT signal, not a billing fact.** It is never a reason to
fund the account. This has now been rediscovered three times (OMN-14625,
OMN-16891, OMN-17193) because the fact lived only in a memory note and an env
override -- never in a contract with a test. This test is that test, on the
infra side; omnimarket's routing contract is the authority for the URL itself
(``configs/bifrost_delegation.yaml``, backend ``cloud-glm``), guarded by
``tests/unit/delegation/test_glm_coding_plan_endpoint_omn6790.py``.

Live probe 2026-08-30, one key (sha12 ``27fecebdd647``), one-token completions,
same host and same second -- the only variable is the path:

    /api/coding/paas/v4   glm-5.3, glm-5.3-flash, glm-5-turbo, glm-4.6,
                          glm-4.5   -> 200, each echoing its own model id
    /api/paas/v4          all of the above                    -> 429 / 1113
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The only z.ai surface our plan entitles us to.
_CODING_PREFIX = "https://api.z.ai/api/coding/paas/v4"

# Any z.ai URL committed anywhere in this repo.
_ZAI_URL = re.compile(r"https://api\.z\.ai/[A-Za-z0-9/_.\-]*")

# Scanned trees. Committed config/compose/contract surfaces only -- this guard
# is about what we ship, not about vendored copies or scratch worktrees.
_SCAN_DIRS = ("docker", "config", "contracts", "scripts", "deploy")

_EXCLUDED_PARTS = frozenset({".git", "node_modules", ".venv", "workspace", ".claude"})


def _iter_committed_files() -> list[Path]:
    out: list[Path] = []
    for rel in _SCAN_DIRS:
        root = _REPO_ROOT / rel
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _EXCLUDED_PARTS & set(path.parts):
                continue
            if path.suffix.lower() not in {
                ".yml",
                ".yaml",
                ".env",
                ".example",
                ".sh",
                ".json",
                ".py",
            }:
                continue
            out.append(path)
    return out


def _zai_urls() -> list[tuple[Path, int, str]]:
    found: list[tuple[Path, int, str]] = []
    for path in _iter_committed_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "api.z.ai" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ZAI_URL.finditer(line):
                found.append((path, lineno, match.group(0)))
    return found


def test_repo_actually_declares_a_glm_endpoint() -> None:
    """Guard the guard: if nothing matches, the scanner silently proves nothing."""
    assert _zai_urls(), (
        "No z.ai URL found anywhere under "
        f"{_SCAN_DIRS} -- this test would pass vacuously. Either the scan roots "
        "moved or the GLM endpoint was removed; fix the scanner, do not delete "
        "this assertion. (An empty result is not evidence of absence.)"
    )


def test_every_committed_glm_endpoint_is_the_coding_plan_surface() -> None:
    """No committed z.ai URL may sit on the pay-as-you-go surface."""
    offenders = [
        (path, lineno, url)
        for path, lineno, url in _zai_urls()
        if not url.startswith(_CODING_PREFIX)
    ]
    if offenders:
        rendered = "\n".join(
            f"  {path.relative_to(_REPO_ROOT)}:{lineno}: {url}"
            for path, lineno, url in offenders
        )
        pytest.fail(
            "GLM endpoint(s) are NOT on the z.ai Coding Plan surface.\n"
            f"{rendered}\n\n"
            "We hold a GLM CODING PLAN. The required base URL is:\n"
            f"    {_CODING_PREFIX}\n"
            "The bare https://api.z.ai/api/paas/v4 is the PAY-AS-YOU-GO product, "
            "which we do not hold. It answers our Coding-Plan key with HTTP 429 "
            'code 1113 "Insufficient balance or no resource package" -- that is a '
            "WRONG-ENDPOINT signal, NOT a billing fact, and never a reason to add "
            "funds. Authority for this URL is omnimarket "
            "configs/bifrost_delegation.yaml (backend cloud-glm). See OMN-6790."
        )


def test_judge_lane_glm_default_is_the_coding_plan_surface() -> None:
    """The judge lane's compose default specifically (regression: OMN-16891)."""
    compose = _REPO_ROOT / "docker" / "docker-compose.judge.yml"
    line = next(
        (
            ln
            for ln in compose.read_text(encoding="utf-8").splitlines()
            if "LLM_GLM_URL:" in ln
        ),
        None,
    )
    assert line is not None, f"LLM_GLM_URL default vanished from {compose.name}"
    assert _CODING_PREFIX in line, (
        f"judge lane LLM_GLM_URL default must fall back to the Coding Plan surface "
        f"({_CODING_PREFIX}); got: {line.strip()}"
    )
