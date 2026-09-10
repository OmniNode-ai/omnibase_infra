# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18115 — a provisioning comment must name the seam that actually grants.

OMN-18060 (#3352) moved the least-privilege GRANT for ``chain_canary_reader``
into ``scripts/run-forward-migrations.sh``'s ``LOGIN_ONLY_ROLE_GRANT_MAP`` seam,
which re-asserts it on every compose up. The comments that had described the
grant were not moved with it: four shipped provisioning surfaces went on
attributing it to "migration 104" — an ordinal that is BURNED (OMN-17923 retired
``104_create_validator_ro_role.sql`` and its record forbids reuse), naming a file
that does not exist and never will.

That is not a cosmetic staleness. The comment is the only prose a reader meets at
the provisioning seam, so it is what a reader believes. OMN-18115 was filed as a
High defect against a settled decision, quoting one of these comments verbatim as
its evidence that the grant was "applied out of band" — while the seam that
issues it sat in ``scripts/``, outside the directories that ticket's grep
covered. The live dev lane's own migration one-shot had logged the grant being
asserted. A second lane then propagated the same mistake forward into a new
shipped paragraph.

So the rule these tests hold is: **at a provisioning surface, the prose that
describes a grant must name the authority that issues it.** Concretely, for every
principal in the runner's grant map:

* no provisioning comment may attribute its grants to a NUMBERED MIGRATION —
  no migration issues them, and the one most recently named is a burned ordinal;
* a comment that DOES describe the grant must name the real authority, either
  ``run-forward-migrations.sh`` or ``LOGIN_ONLY_ROLE_GRANT_MAP``;
* no comment may assert the grant is unissued, missing, or applied out of band.

Scope is the provisioning surfaces themselves — the compose files that carry the
credential variable and the fresh-volume bootstrap script — not every file in the
repo that happens to mention the role. Those are the seams a reader consults when
asking "what creates this authorization", and they are the four that were wrong.

Companion static pins on the seam's SHAPE: ``test_login_only_role_grants_omn18060.py``.
Live proofs that the grant lands: ``tests/scripts/test_login_only_role_grants_live_omn18060.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

# The provisioning surfaces: every compose file that carries a granted
# principal's credential variable, plus the fresh-volume bootstrap script that
# mints the role. Resolved by glob rather than listed, so a new lane's compose
# file is covered the day it is added instead of the day someone remembers.
PROVISIONING_SURFACES: tuple[Path, ...] = (
    *sorted((REPO_ROOT / "docker").glob("docker-compose*.yml")),
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "000_create_multiple_databases.sh",
)

# The authority that issues these grants. Either spelling identifies it.
AUTHORITY_TOKENS = ("run-forward-migrations.sh", "LOGIN_ONLY_ROLE_GRANT_MAP")

# Prose that describes a grant. A comment block naming a granted principal and
# any of these is making a claim about authorization, and owes the authority.
GRANT_WORDS = ("grant", "select", "connect", "usage", "privilege")

# The seam that MINTS these principals on a fresh volume. Compose-only, like
# the grant seam: the k8s Job never runs either, which is the asymmetry the
# comments have to make legible.
CREDENTIAL_SEAM = "000_create_multiple_databases.sh"

# A numbered migration, in any of the spellings the stale comments used.
NUMBERED_MIGRATION_RE = re.compile(r"migration\s+\d+", re.IGNORECASE)

# Assertions that the grant does not exist. Each of these was true before #3352
# and is false after it; one of them shipped into a comment regardless.
ABSENCE_CLAIMS = (
    "applied out of band",
    "no file ever issued",
    "not reproducible",
    "issued by no migration",
    "no migration issues it",
)


def _granted_roles() -> list[str]:
    """The principals the runner's grant map actually grants to.

    Derived from the shipped runner rather than hardcoded, so adding a second
    entry to the map brings its comments under this rule automatically.
    """
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("for grant_role_entry in") + len("for grant_role_entry in")
    body = text[start : text.index("; do", start)]
    roles = re.findall(r'"([a-z0-9_]+):[a-z0-9_]+\.[a-z0-9_]+:[a-z0-9_,]+"', body)
    assert roles, (
        "no principals parsed out of the runner's LOGIN_ONLY_ROLE_GRANT_MAP. "
        "Either the seam moved or this parse went stale — a rule with an empty "
        "subject passes vacuously, which is the failure mode being avoided."
    )
    return roles


def _comment_blocks(path: Path) -> list[tuple[int, str]]:
    """Contiguous runs of comment lines, as (1-based first line, text).

    A block is the unit a reader takes in, so it is the unit the rule applies
    to: a sentence naming the role and a sentence naming the migration are one
    claim even when they are two lines.
    """
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    first_line = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            if not current:
                first_line = lineno
            current.append(stripped.lstrip("#").strip())
        elif current:
            blocks.append((first_line, " ".join(current)))
            current = []
    if current:
        blocks.append((first_line, " ".join(current)))
    return blocks


def _blocks_naming_a_granted_role() -> list[tuple[Path, int, str]]:
    """Every provisioning comment block that names a granted principal."""
    roles = _granted_roles()
    found: list[tuple[Path, int, str]] = []
    for surface in PROVISIONING_SURFACES:
        if not surface.exists():
            continue
        for lineno, block in _comment_blocks(surface):
            if any(role in block for role in roles):
                found.append((surface, lineno, block))
    return found


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


@pytest.mark.unit
def test_the_rule_has_a_subject() -> None:
    """Positive control: the scan finds the blocks it is supposed to judge.

    Without this, deleting every comment — or breaking the glob, or breaking the
    grant-map parse — would turn all three rules below green while proving
    nothing. An empty result is not evidence of absence.
    """
    blocks = _blocks_naming_a_granted_role()
    assert len(blocks) >= 4, (
        "expected at least the four provisioning comment blocks that describe "
        f"chain_canary_reader, found {len(blocks)}: "
        f"{[(_relative(p), n) for p, n, _ in blocks]}"
    )
    surfaces = {p for p, _, _ in blocks}
    assert len(surfaces) >= 3, (
        "expected the blocks to span at least the two lane compose files and "
        f"the bootstrap script, found only {sorted(_relative(p) for p in surfaces)}"
    )


@pytest.mark.unit
def test_no_provisioning_comment_credits_a_numbered_migration() -> None:
    """No migration issues these grants; the last one named is a burned ordinal."""
    offenders = [
        (_relative(path), lineno, NUMBERED_MIGRATION_RE.search(block).group(0))  # type: ignore[union-attr]
        for path, lineno, block in _blocks_naming_a_granted_role()
        if NUMBERED_MIGRATION_RE.search(block)
    ]
    assert not offenders, (
        "a provisioning comment attributes a granted principal's grants to a "
        f"numbered migration: {offenders}. No migration issues them — "
        "scripts/run-forward-migrations.sh does, on every compose up, through "
        "LOGIN_ONLY_ROLE_GRANT_MAP (OMN-18060, #3352). Ordinal 104 is BURNED by "
        "OMN-17923 and 105 is the link-5 ledger_chain relation, not this grant. "
        "Name the seam that issues the grant, not a file that does not exist."
    )


@pytest.mark.unit
def test_a_comment_that_describes_the_grant_names_its_authority() -> None:
    """Describing authorization obliges you to say what confers it."""
    offenders = [
        (_relative(path), lineno)
        for path, lineno, block in _blocks_naming_a_granted_role()
        if any(word in block.lower() for word in GRANT_WORDS)
        and not any(token in block for token in AUTHORITY_TOKENS)
    ]
    assert not offenders, (
        "a provisioning comment describes a granted principal's privileges "
        f"without naming what issues them: {offenders}. Name "
        "scripts/run-forward-migrations.sh or its LOGIN_ONLY_ROLE_GRANT_MAP "
        "seam. A reader who cannot find the authority concludes there is none — "
        "which is exactly how OMN-18115 came to be filed."
    )


@pytest.mark.unit
def test_no_provisioning_comment_claims_the_grant_is_unissued() -> None:
    """The grant is issued and re-asserted on every compose up. Say so."""
    offenders = [
        (_relative(path), lineno, claim)
        for path, lineno, block in _blocks_naming_a_granted_role()
        for claim in ABSENCE_CLAIMS
        if claim in block.lower()
    ]
    assert not offenders, (
        "a provisioning comment asserts a granted principal's grant is missing "
        f"or was applied out of band: {offenders}. It is issued by "
        "scripts/run-forward-migrations.sh and re-asserted on every compose up, "
        "with a readback that fails the run when the grant did not take "
        "(OMN-18060). The .201 dev lane's own forward-migration one-shot logs "
        "it: 'ok chain_canary_reader CONNECT + USAGE on public + SELECT "
        "(correlation_id,state) on public.delegation_workflow_state asserted'."
    )


@pytest.mark.unit
def test_the_comment_states_which_lane_provisions_the_principal() -> None:
    """The two-lane split must be legible where the credential is configured.

    The grant seam and the credential seam are both compose-only. The k8s Job
    applies the flat SQL corpus and never runs the shell runner, so on the
    managed (RDS) lane the principal has no row at all. A reader who does not
    know that reads the absence as breakage and reaches for a migration -- which
    is the shape of OMN-18115. Naming the minting seam makes the asymmetry
    readable at the place the question is asked.
    """
    offenders = [
        (_relative(path), lineno)
        for path, lineno, block in _blocks_naming_a_granted_role()
        if any(word in block.lower() for word in GRANT_WORDS)
        and CREDENTIAL_SEAM not in block
        and path.name != CREDENTIAL_SEAM
    ]
    assert not offenders, (
        "a provisioning comment describes a granted principal without naming "
        f"what mints it: {offenders}. Name {CREDENTIAL_SEAM} (fresh volume) "
        "alongside the runner's credential seam (warm volume), so a reader can "
        "tell a compose lane from the managed lane, where the principal is "
        "absent by construction and needs no grant."
    )
