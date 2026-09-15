# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 107 is RETIRED from the flat forward stream (OMN-17486).

``107_create_action_authorization_nonce_claim.sql`` (landed by ``18b539f0f``,
#3549) declares two cluster preconditions and refuses, by design, when either is
absent: the ``rsd_action_authorization_claim`` restricted principal, which it
guard-creates and cannot create as a ``NOCREATEROLE`` identity, and the
``action_authorization_claim`` schema, which it asserts and never creates. Both
refusals name the privileged application-database provisioning seam as the
remediation.

On the ``onex-dev`` serving RDS neither precondition holds, and the seam does not
carry either one. ``deploy-onex-staging`` run 34943161516 (2026-09-15T07:47:34Z,
head ``07293d45``) recorded the refusal verbatim at the migration's own line 54::

    ERROR:  rsd_action_authorization_claim does not exist on this cluster and
            the executing role role_omnibase_infra cannot create it

That is a live statement from the cluster itself, stronger than a probe. The Job
exhausted its backoff and every ``deploy-onex-staging`` run since #3549 has died
at "Run database migrations", which is migration-order 1 of 6 and upstream of
the overlay apply and the boot gate -- so nothing at all reaches ``onex-dev``.
The seam cannot be taught this in one change either: ``provision-cluster-roles.sh``
creates principals without ``NOINHERIT`` (this migration rejects an inheriting
role on its very next branch), and nothing in the managed lane provisions a
SCHEMA at all.

Retired, not rewritten. A migration that skips when its precondition is unmet is
fail-open; 103's header rejects that masking skip, so does OMN-17923's
retirement of 104, and so does this one. The repository has no same-database
retirement mechanism for a flat file -- the cross-database tombstone applies only
to files whose ``\\connect`` names a foreign database, ``skip-manifest.yaml`` is
read only by the compose-lane runner and never by the k8s Job, and
``_ledger/migration-supersessions.tsv`` is node-only -- so the retirement is the
smallest thing the k8s Job honours: the forward file and its rollback are REMOVED
from the corpus, and the record of why, and of what must be true before the
migration is re-issued, lives in ``_ledger/retired-flat-migrations.tsv``.

This module is the gate over that record. It pins:

* the record names 107, its rollback, the commit the bytes were retired from,
  the digests of those bytes, and OMN-17486 as the condition for re-issue;
* both retired files are out of the corpus and ordinal 107 is BURNED, so a lane
  that somehow recorded 107 can never be confused with one that applied the
  re-issue;
* the surviving stream at or below 107 is byte-identical to the stream as it
  stood before #3549 landed, so the retirement removed exactly these two files;
* the ``action_authorization_claim`` schema DECLARATION survives in every shipped
  instance. Declaration is not provisioning; it is harmless until the seam
  carries it, and it is the input a re-issue starts from.

Scope note. The family this migration serves (OMN-17463 first-effect ledger
authority) is NOT beta scope by operator ruling of 2026-09-14, and is revisited
after beta. Staging delivery is beta-critical. The retirement is what keeps a
non-beta migration from holding the beta-critical lane.

Ticket: OMN-17486. Precedent: OMN-17923 (the 104 retirement this follows).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from scripts.check_schema_fingerprint import compute_migration_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations"
FORWARD_DIR = MIGRATIONS_DIR / "forward"
ROLLBACK_DIR = MIGRATIONS_DIR / "rollback"
RECORD = FORWARD_DIR / "_ledger" / "retired-flat-migrations.tsv"
FINGERPRINT_ARTIFACT = MIGRATIONS_DIR / "schema_fingerprint.sha256"
INSTANCES_DIR = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"

RETIRED_FORWARD = "107_create_action_authorization_nonce_claim.sql"
RETIRED_ROLLBACK = "rollback_107_create_action_authorization_nonce_claim.sql"
RETIRED_FROM_COMMIT = "18b539f0f76fc4c5df1966d34ca4a2ebeba51f8f"
RETIRED_FORWARD_SHA256 = (
    "df0fc6bc0dac9a5ef7abfc173dc32549a0a3c4d93066ba19db65953c9dfb0e5c"
)
RETIRED_ROLLBACK_SHA256 = (
    "3eb317695f138b0071e450cbb80b6df7fc0985f1e9630df837f47f500dc40895"
)
RETIRING_TICKET = "OMN-17486"
RETIRED_ORDINAL = 107
SURVIVING_HIGH_WATER = 106
# The stream as it stood BEFORE #3549 landed 107: the fingerprint stamped at
# 18b539f0f's parent. The retirement must reproduce it exactly -- that equality
# is the proof that the stream through 106 is byte-unchanged and that the
# retirement removed 107 and nothing else.
PRE_3549_PARENT_COMMIT = "d10b5d712009db952fb12f0e6b85234babf142f4"
PRE_3549_STREAM_SHA256 = (
    "3f8b15b6ace73ca12e6f862804effb61cb0df41aa98095eda8cf16ba5bb3c47e"
)
PRE_3549_STREAM_FILE_COUNT = 90
SCHEMA = "action_authorization_claim"
SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
# The staging run whose own log carries the refusal this retirement answers.
BLOCKED_STAGING_RUN = "34943161516"

COLUMNS = (
    "retired_forward",
    "retired_rollback",
    "retired_from_commit",
    "forward_sha256",
    "rollback_sha256",
    "tickets",
    "reissue_condition",
    "reason",
)


def _rows() -> list[dict[str, str]]:
    assert RECORD.is_file(), f"{RECORD} is missing: the retirement has no record"
    rows: list[dict[str, str]] = []
    for lineno, raw in enumerate(RECORD.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        assert len(fields) == len(COLUMNS), (
            f"{RECORD.name}:{lineno} has {len(fields)} tab-separated fields; "
            f"the record schema is {COLUMNS}"
        )
        rows.append(dict(zip(COLUMNS, fields, strict=True)))
    assert rows, f"{RECORD.name} carries no retirement row"
    return rows


def _row_for_107() -> dict[str, str]:
    matches = [row for row in _rows() if row["retired_forward"] == RETIRED_FORWARD]
    assert len(matches) == 1, (
        f"{RECORD.name} must carry exactly one row for {RETIRED_FORWARD}; "
        f"found {len(matches)}"
    )
    return matches[0]


def _ordinal(name: str) -> int | None:
    match = re.match(r"^(?:rollback_)?(\d{3})_", name)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_names_the_retired_bytes_precisely() -> None:
    row = _row_for_107()
    assert row["retired_rollback"] == RETIRED_ROLLBACK
    assert row["retired_from_commit"] == RETIRED_FROM_COMMIT, (
        "the record must name the full commit the retired bytes came from, so a "
        "re-issue starts from `git show <commit>:<path>` rather than from memory"
    )
    assert row["forward_sha256"] == RETIRED_FORWARD_SHA256
    assert row["rollback_sha256"] == RETIRED_ROLLBACK_SHA256


@pytest.mark.unit
def test_record_cites_the_ticket_and_names_the_reissue_condition() -> None:
    row = _row_for_107()
    assert RETIRING_TICKET in set(row["tickets"].split(",")), (
        f"the row cites {row['tickets']}; it must cite {RETIRING_TICKET}"
    )
    condition = row["reissue_condition"]
    assert RETIRING_TICKET in condition, (
        "the re-issue condition must name OMN-17486: the migration comes back "
        "only once the provisioning seam creates both the restricted principal "
        "and the schema BEFORE the migrate Job runs"
    )
    assert re.search(r"\bnew (?:ordinal|number)\b", condition, re.IGNORECASE), (
        "the re-issue condition must state that the migration returns as a NEW "
        "number; ordinal 107 is burned"
    )
    for field in ("reason", "reissue_condition"):
        assert row[field].strip(), f"{field} is empty"


@pytest.mark.unit
def test_record_names_both_unmet_preconditions_not_just_the_role() -> None:
    """Naming only the role would understate what a re-issue has to wait for.

    The role is the refusal the staging log happens to show first, because the
    migration checks it first. The schema assertion two statements later is a
    SECOND unmet precondition with a different remediation: nothing in the
    managed lane provisions a schema at all, while the role at least has a seam
    that could be taught to create it. A record that names one and not the other
    would send the re-issue back into the same failure one statement further on.
    """
    row = _row_for_107()
    haystack = f"{row['reason']} {row['reissue_condition']}"
    assert "rsd_action_authorization_claim" in haystack, (
        "the record must name the restricted principal the migration cannot create"
    )
    assert re.search(rf"\b{SCHEMA}\b\s+schema|schema\s+\b{SCHEMA}\b", haystack), (
        "the record must name the schema precondition as well as the role; they "
        "are two separate unmet preconditions with two different remediations"
    )


@pytest.mark.unit
def test_record_rejects_the_masking_skip_in_its_own_words() -> None:
    row = _row_for_107()
    assert re.search(r"fail[- ]open", row["reason"], re.IGNORECASE), (
        "the reason must say why the file was retired rather than rewritten to "
        "skip when its precondition is unmet: a fail-open migration is worse "
        "than a retired one"
    )


@pytest.mark.unit
def test_record_cites_the_staging_run_that_recorded_the_refusal() -> None:
    """The evidence is a live cluster refusal, not an inference from the code."""
    row = _row_for_107()
    assert BLOCKED_STAGING_RUN in row["reason"], (
        "the reason must cite the deploy-onex-staging run whose log carries the "
        "refusal, so the claim 'this blocked staging' is checkable"
    )


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retired_files_are_absent_from_the_corpus() -> None:
    for row in _rows():
        forward = FORWARD_DIR / row["retired_forward"]
        rollback = ROLLBACK_DIR / row["retired_rollback"]
        assert not forward.exists(), (
            f"{forward.name} is recorded as retired but is still in the forward "
            "stream; the k8s Job would apply it"
        )
        assert not rollback.exists(), (
            f"{rollback.name} is recorded as retired but is still in the rollback set"
        )


@pytest.mark.unit
def test_retired_ordinal_is_burned_not_reused() -> None:
    reused = sorted(
        path.name
        for directory in (FORWARD_DIR, ROLLBACK_DIR)
        for path in directory.glob("*.sql")
        if _ordinal(path.name) == RETIRED_ORDINAL
    )
    assert reused == [], (
        f"ordinal {RETIRED_ORDINAL} is retired and must never be reused; found "
        f"{reused}. Re-issue the migration as a NEW number so a lane that "
        f"recorded {RETIRED_ORDINAL} can never be mistaken for one that applied "
        "the re-issue."
    )


@pytest.mark.unit
def test_stream_tops_out_at_106_until_the_reissue() -> None:
    ordinals = sorted(
        ordinal
        for path in FORWARD_DIR.glob("*.sql")
        if (ordinal := _ordinal(path.name)) is not None
    )
    assert ordinals, f"no ordinal-prefixed migrations under {FORWARD_DIR}"
    at_or_below_retired = [o for o in ordinals if o <= RETIRED_ORDINAL]
    assert max(at_or_below_retired) == SURVIVING_HIGH_WATER, (
        f"the surviving stream at or below {RETIRED_ORDINAL} tops out at "
        f"{max(at_or_below_retired)}, expected {SURVIVING_HIGH_WATER}"
    )


@pytest.mark.unit
def test_surviving_stream_is_byte_identical_to_the_pre_3549_stream() -> None:
    """The stream through 106 is byte-unchanged: the retirement removed ONLY 107.

    ``compute_migration_fingerprint`` hashes each forward ``*.sql`` by name and
    content, so equality with the value stamped at #3549's parent commit is a
    byte-level statement about the surviving corpus, not merely a count.

    Scoped to ordinals at or below the retired one, following the OMN-16964
    correction to the 104 gate: the invariant this module owns is "the
    retirement removed 107 and nothing in 001..106 changed", which stays true
    forever. Fingerprinting the whole directory would conflate it with "no
    forward migration may ever land again", which is not asserted anywhere and
    which the re-issue this record contemplates would immediately break.
    """
    fingerprint, count = compute_migration_fingerprint(
        FORWARD_DIR,
        include=lambda path: (ordinal := _ordinal(path.name)) is not None
        and ordinal <= RETIRED_ORDINAL,
    )
    assert (fingerprint, count) == (
        PRE_3549_STREAM_SHA256,
        PRE_3549_STREAM_FILE_COUNT,
    ), (
        f"the surviving forward stream at or below {RETIRED_ORDINAL} fingerprints "
        f"as {fingerprint} over {count} files; the stream before #3549 (commit "
        f"{PRE_3549_PARENT_COMMIT[:9]}) was {PRE_3549_STREAM_SHA256} over "
        f"{PRE_3549_STREAM_FILE_COUNT}. Either the retirement removed more than "
        "107, or something else in 001..106 changed and needs its own restamp."
    )


@pytest.mark.unit
def test_fingerprint_artifact_matches_the_surviving_stream() -> None:
    """The artifact is what CI verifies; it must describe the stream WITHOUT 107.

    A stale artifact would mean the retirement changed the corpus without
    re-stamping, and ``check_schema_fingerprint.py verify`` would fail CI for a
    reason that reads like drift rather than like this retirement.
    """
    fingerprint, count = compute_migration_fingerprint(FORWARD_DIR)
    content = FINGERPRINT_ARTIFACT.read_text(encoding="utf-8")
    assert f"sha256:{fingerprint}" in content, (
        "schema_fingerprint.sha256 does not match the on-disk forward stream; "
        "run `python scripts/check_schema_fingerprint.py stamp`"
    )
    assert f"migration_file_count: {count}" in content
    assert not any(path.name == RETIRED_FORWARD for path in FORWARD_DIR.glob("*.sql"))


# ---------------------------------------------------------------------------
# The declaration survives
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("instance", SHIPPED_INSTANCES)
def test_claim_schema_declaration_survives_the_retirement(instance: str) -> None:
    """Retiring the MIGRATION does not withdraw the SCHEMA declaration.

    The declaration is harmless until the seam provisions it, and it is what a
    re-issue starts from. Withdrawing it here would make the re-issue start from
    zero, and would also drop the domain and owner decisions that were already
    made for this schema.
    """
    document = yaml.safe_load(
        (INSTANCES_DIR / f"{instance}.yaml").read_text(encoding="utf-8")
    )
    schemas = document["databases"]["application"]["schemas"]
    assert SCHEMA in schemas, (
        f"{instance}.yaml no longer declares the {SCHEMA} schema; the retirement "
        "of migration 107 must leave the topology declaration intact"
    )
    assert schemas[SCHEMA]["domain"] == "OMNINODE_INTERNAL"
    assert schemas[SCHEMA]["owner"] == "owner_omninode_internal"
