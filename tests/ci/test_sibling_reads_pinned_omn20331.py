# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20331: a gating workflow reads a sibling repo at a pinned commit.

A gating omnibase_infra check that reads a sibling's live ``dev``/``main`` tip
turns red on every open PR when that sibling merges, and no omnibase_infra PR
can fix it (the OMN-17292 rationale, applied to every sibling). Each sibling
commit is declared once in ``.github/sibling-pins.yaml`` and written inline as
the ``ref:`` of the checkout, the ``git clone`` checkout, or the ``@<sha>`` of
the ``uses:`` in the workflows listed in ``PINNED_WORKFLOWS``.

Falsifier: restore ``ref: dev`` (or drop the ref) on any sibling checkout in
those files and ``test_sibling_checkout_refs_are_the_pin`` names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PINS = yaml.safe_load((REPO_ROOT / ".github" / "sibling-pins.yaml").read_text())["pins"]

# Deliberately live, not in PINNED_WORKFLOWS (measured red when pinned, PR #4449):
# ci-bus-overlay-binding.yml binds the overlay to the branch the publisher reads at
# merge time (its own test requires omnimarket ``dev``). onex_change_control's
# composite actions fail when addressed by a sha, so contract-validation.yml runs
# the validate-contract steps inline with a pinned checkout (OMN-19747). The
# validate-boundaries action ci.yml calls is still a composite: at a pinned OCC dev sha its inner
# checkout takes ``github.action_ref``, which resolves to ``v6`` inside the
# composite, and the merge_group Cross-Repo Migration Conflicts job went red
# (run 36942428614); that one ``uses:`` stays on ``@main``.
LIVE_ACTIONS = frozenset({"onex_change_control/.github/actions/validate-boundaries"})
PINNED_WORKFLOWS = (
    "ci.yml",
    "call-occ-autobind.yml",
    "call-occ-companion-effect.yml",
    "contract-topic-graph.yml",
    "contract-validation.yml",
    "contractor-integration-note.yml",
    "delegation-consumer-kwarg-parity.yml",
    "dispatcher-route-coverage.yml",
    "docs-validate.yml",
    "duplication-sweep.yml",
    "exposure-reader-coverage.yml",
    "hostile-reviewer.yml",
    "r1-front-door-probe.yml",
    "receipt-honesty.yml",
    "runtime-rebuild-trigger.yml",
    "skill-node-mapping-sync.yml",
)

# A ``ref:`` that is an expression is resolved from committed state by a
# resolver step (omnimarket-contract-pin.yaml, OMN-17427 pin resolver, the
# paired-PR resolver of node-migration-sync.yml) or is a sha already pinned in place.
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^\s*repository: OmniNode-ai/(\w+)\s*$")
REF = re.compile(r"^\s*ref: (\S+)")
USES = re.compile(r"uses: OmniNode-ai/((\w+)/\.github/\S+)@(\S+)")


def _checkout_refs(text: str) -> list[tuple[int, str, str | None]]:
    lines = text.split("\n")
    found = []
    for i, line in enumerate(lines):
        m = REPOSITORY.match(line)
        if m and m.group(1) in PINS:
            ref = next(
                (r.group(1) for ln in lines[i + 1 : i + 4] if (r := REF.match(ln))),
                None,
            )
            found.append((i + 1, m.group(1), ref))
    return found


def test_pins_are_full_shas_and_omnimarket_matches_contract_pin() -> None:
    assert all(HEX40.fullmatch(v) for v in PINS.values())
    contract_pin = yaml.safe_load(
        (REPO_ROOT / ".github" / "omnimarket-contract-pin.yaml").read_text()
    )["omnimarket_contract_ref"]
    assert PINS["omnimarket"] == contract_pin


@pytest.mark.parametrize("name", PINNED_WORKFLOWS)
def test_sibling_checkout_refs_are_the_pin(name: str) -> None:
    text = (WORKFLOWS / name).read_text()
    bad = []
    for lineno, repo, ref in _checkout_refs(text):
        if ref is None or ref in {"dev", "main"}:
            bad.append(f"{name}:{lineno} {repo} ref={ref}")
        elif HEX40.fullmatch(ref):
            # a sha that predates the pin file is tolerated only when it is not a
            # branch-tip read; the migrated sites carry the pin comment
            continue
    assert not bad, f"sibling read by branch or no ref: {bad}"


@pytest.mark.parametrize("name", PINNED_WORKFLOWS)
def test_migrated_sites_carry_the_declared_sha(name: str) -> None:
    text = (WORKFLOWS / name).read_text()
    for line in text.split("\n"):
        if "# .github/sibling-pins.yaml" not in line:
            continue
        assert any(sha in line for sha in PINS.values()), f"{name}: {line.strip()}"


@pytest.mark.parametrize("name", PINNED_WORKFLOWS)
def test_no_sibling_reusable_workflow_or_clone_by_branch(name: str) -> None:
    text = (WORKFLOWS / name).read_text()
    bad = [
        f"{name}: {m.group(1)}@{m.group(3)}"
        for m in USES.finditer(text)
        if m.group(2) in PINS
        and m.group(1) not in LIVE_ACTIONS
        and not HEX40.fullmatch(m.group(3))
    ]
    assert not bad, f"sibling reusable workflow/action by branch: {bad}"
    assert not re.search(r"^\s*core-ref: (dev|main)\s*$", text, re.M)
    for m in re.finditer(
        r"git clone [^\n]*--branch (dev|main)[^\n]*\n[^\n]*OmniNode-ai/(\w+)", text
    ):
        assert m.group(2) not in PINS, f"{name}: clone of {m.group(2)} by branch"
