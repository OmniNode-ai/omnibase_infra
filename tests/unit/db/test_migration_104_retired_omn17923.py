# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 104 is RETIRED from the flat forward stream (OMN-17923).

``104_create_validator_ro_role.sql`` (landed by OMN-17792, ``cb122fefb``) guard-
creates the ``validator_ro`` cluster role and, when the role is absent and the
executing identity cannot create it, re-raises ``insufficient_privilege`` with a
named remediation -- by design, so the migration never fails open. On the
``onex-dev`` serving RDS that design meets a lane where the precondition is
unmet: the flat loop runs as ``role_omnibase_infra`` (``NOCREATEROLE`` by
contract, read back live 2026-09-04T20:13:34Z), and the seam that is supposed to
create the role first -- ``omninode_infra scripts/provision-cluster-roles.sh``,
the "Provision topology cluster roles (OMN-17347)" step -- reads the
``omninode_infra`` projection of the topology, which is pinned to a commit that
predates the ``validator_ro`` declaration. So the very next
``deploy-onex-staging`` migration step would abort on 104, upstream of the
overlay apply and the boot gate, exactly the way 103 aborted run 33341217605 on
2026-08-30 before the seam existed.

The repository has no same-database retirement mechanism for a flat file: the
cross-database tombstone applies only to files whose ``\\connect`` names a
foreign database, ``skip-manifest.yaml`` is read only by the compose-lane runner
(never by the k8s Job), and ``_ledger/migration-supersessions.tsv`` is node-only.
So the retirement is the smallest thing the k8s Job honours -- the file and its
rollback are REMOVED from the corpus -- and the record of WHY, and of WHAT must
be true before the migration is re-issued, lives in
``_ledger/retired-flat-migrations.tsv``. This module is the gate over that
record. It pins:

* the record names 104, its rollback, the commit the bytes were retired from,
  the digests of those bytes, and OMN-17923 as the condition for re-issue;
* neither retired file is in the corpus, and ordinal 104 is BURNED -- the
  migration comes back as a NEW number, never as 104 again, so a lane that
  somehow recorded 104 can never be confused with one that applied the re-issue;
* the stream still tops out at 103, byte-identical in count to what the
  fingerprint artifact records, so the retirement removed exactly the retired
  file and nothing else;
* the ``validator_ro`` principal DECLARATION survives in every shipped instance.
  Declaration is not provisioning; it is harmless until the seam carries it, and
  it is the thing OMN-17923 exists to deliver.

The retired migration was NOT rewritten to skip when the role cannot be created.
A fail-open migration is worse than a retired one; 103's own header rejects the
masking skip and so does this one.

Ticket: OMN-17923 (blocks OMN-17792). Origin: OMN-17792 (#3190).
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

RETIRED_FORWARD = "104_create_validator_ro_role.sql"
RETIRED_ROLLBACK = "rollback_104_create_validator_ro_role.sql"
RETIRED_FROM_COMMIT = "cb122fefb541c199751ad8ffa50efbb00aee003f"
RETIRED_FORWARD_SHA256 = (
    "ede81614e04b021a462abacc6cb1183cd6cf8aba2e80739d05276d0377a8224a"
)
RETIRED_ROLLBACK_SHA256 = (
    "33c020bcddd77b08d243ed560d5e6c0d27bdffbc15e0f83e860ec3997696cfa0"
)
RETIRING_TICKET = "OMN-17923"
ORIGIN_TICKET = "OMN-17792"
SURVIVING_HIGH_WATER = 103
# The stream as it stood BEFORE #3190 landed 104: the fingerprint stamped at
# cb122fefb's parent (71177a327, #3188). The retirement must reproduce it
# exactly -- that equality is the proof that the stream through 103 is
# byte-unchanged and that the retirement removed 104 and nothing else.
PRE_3190_PARENT_COMMIT = "71177a327dbf4d1a2c62d186c5ebc61a726c887a"
PRE_3190_STREAM_SHA256 = (
    "79aa3056a6ad65eaf8899cfbb689c9cbe13b2088f3af40f970710890270d1143"
)
PRE_3190_STREAM_FILE_COUNT = 88
ROLE = "validator_ro"
SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

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


def _row_for_104() -> dict[str, str]:
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
    row = _row_for_104()
    assert row["retired_rollback"] == RETIRED_ROLLBACK
    assert row["retired_from_commit"] == RETIRED_FROM_COMMIT, (
        "the record must name the full commit the retired bytes came from, so "
        "the re-issue can start from `git show <commit>:<path>` rather than from "
        "memory"
    )
    assert row["forward_sha256"] == RETIRED_FORWARD_SHA256
    assert row["rollback_sha256"] == RETIRED_ROLLBACK_SHA256


@pytest.mark.unit
def test_record_cites_both_tickets_and_names_the_reissue_condition() -> None:
    row = _row_for_104()
    tickets = set(row["tickets"].split(","))
    assert {RETIRING_TICKET, ORIGIN_TICKET} <= tickets, (
        f"the row cites {sorted(tickets)}; it must cite {RETIRING_TICKET} (the "
        f"retirement) and {ORIGIN_TICKET} (the origin)"
    )
    condition = row["reissue_condition"]
    assert RETIRING_TICKET in condition, (
        "the re-issue condition must name OMN-17923: the migration comes back "
        "only once the omninode_infra projection of the topology carries "
        "validator_ro and the OMN-17347 seam creates the role BEFORE the Job"
    )
    assert re.search(r"\bnew (?:ordinal|number)\b", condition, re.IGNORECASE), (
        "the re-issue condition must state that the migration returns as a NEW "
        "number; ordinal 104 is burned"
    )
    for field in ("reason", "reissue_condition"):
        assert row[field].strip(), f"{field} is empty"


@pytest.mark.unit
def test_record_rejects_the_masking_skip_in_its_own_words() -> None:
    row = _row_for_104()
    assert re.search(r"fail[- ]open", row["reason"], re.IGNORECASE), (
        "the reason must say why the file was retired rather than rewritten to "
        "skip on insufficient_privilege: a fail-open migration is worse than a "
        "retired one"
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
    for row in _rows():
        ordinal = _ordinal(row["retired_forward"])
        assert ordinal is not None, row["retired_forward"]
        reused = sorted(
            path.name
            for directory in (FORWARD_DIR, ROLLBACK_DIR)
            for path in directory.glob("*.sql")
            if _ordinal(path.name) == ordinal
        )
        assert reused == [], (
            f"ordinal {ordinal:03d} is retired and must never be reused; found "
            f"{reused}. Re-issue the migration as a NEW number so a lane that "
            f"recorded {ordinal:03d} can never be mistaken for one that applied "
            "the re-issue."
        )


@pytest.mark.unit
def test_stream_tops_out_at_103_until_the_reissue() -> None:
    ordinals = sorted(
        ordinal
        for path in FORWARD_DIR.glob("*.sql")
        if (ordinal := _ordinal(path.name)) is not None
    )
    assert ordinals, f"no ordinal-prefixed migrations under {FORWARD_DIR}"
    assert SURVIVING_HIGH_WATER in ordinals, (
        "103_create_tenant_projection_writer_role.sql must survive the retirement "
        "untouched; it is applied on the onex-dev RDS (2026-08-31T15:47:28Z)"
    )
    below_or_at_retired = [ordinal for ordinal in ordinals if ordinal <= 104]
    assert max(below_or_at_retired) == SURVIVING_HIGH_WATER, (
        f"the surviving stream at or below 104 tops out at "
        f"{max(below_or_at_retired):03d}, expected {SURVIVING_HIGH_WATER:03d}"
    )


@pytest.mark.unit
def test_surviving_stream_is_byte_identical_to_the_pre_3190_stream() -> None:
    """The stream through 103 is byte-unchanged: the retirement removed ONLY 104.

    ``compute_migration_fingerprint`` hashes every forward ``*.sql`` by name and
    content, so equality with the value stamped at #3190's parent commit is a
    byte-level statement about the whole surviving corpus, not just a count.
    A new migration, or any edit to 001..103, changes this value and must land
    as its own change with its own restamp -- never folded into a retirement.
    """
    fingerprint, count = compute_migration_fingerprint(FORWARD_DIR)
    assert (fingerprint, count) == (
        PRE_3190_STREAM_SHA256,
        PRE_3190_STREAM_FILE_COUNT,
    ), (
        f"the surviving forward stream fingerprints as {fingerprint} over {count} "
        f"files; the stream before #3190 (parent {PRE_3190_PARENT_COMMIT[:9]}) "
        f"was {PRE_3190_STREAM_SHA256} over {PRE_3190_STREAM_FILE_COUNT}. "
        "Either the retirement removed more than 104, or something else in "
        "001..103 changed and needs its own restamp."
    )


@pytest.mark.unit
def test_fingerprint_artifact_matches_the_surviving_stream() -> None:
    """The artifact is what CI verifies; it must describe the stream WITHOUT 104.

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
def test_validator_ro_declaration_survives_the_retirement(instance: str) -> None:
    """Retiring the MIGRATION does not withdraw the PRINCIPAL.

    The declaration is harmless until the seam provisions it, and it is the
    input OMN-17923's projection resync exists to carry to the seam. Withdrawing
    it here would make the re-issue start from zero.
    """
    document = yaml.safe_load(
        (INSTANCES_DIR / f"{instance}.yaml").read_text(encoding="utf-8")
    )
    principals = document["databases"]["application"]["principals"]
    assert ROLE in principals, (
        f"{instance}.yaml no longer declares {ROLE}; the retirement of migration "
        "104 must leave the topology principal declaration intact"
    )
    assert principals[ROLE]["bypass_rls"] is False
