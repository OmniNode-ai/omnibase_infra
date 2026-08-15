# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``onex occ`` stamp/validate of OCC PR metadata (OMN-14190).

These exercise the REAL CLI wiring (click ``CliRunner``) against the REAL
canonical OCC stamp schema (``omnibase_compat.contracts.pr_occ_stamp``);
nothing is mocked. The two load-bearing behaviors the ticket requires are:

* ``validate`` is **fail-closed** — a missing/malformed ``Evidence-Source`` or
  ``Evidence-Ticket`` exits non-zero with an actionable message.
* ``stamp`` is **idempotent** — re-running never produces a duplicate Evidence
  block (``stamp(stamp(x)) == stamp(x)``).

Also covers ``compute-contract-hash`` (OMN-15711 / FM5): the load-bearing
behavior there is that the CLI's output is byte-for-byte identical to calling
``validator_receipt_gate``'s hash functions directly -- proving the CLI
reimplements no hashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnibase_core.validation.validator_receipt_gate import (
    compute_contract_entry_sha256,
    compute_contract_sha256,
)
from omnibase_infra.cli.cli_occ import (
    compute_receipt_contract_hashes,
    occ,
    parse_evidence_source_token,
    stamp_pr_body,
    validate_pr_body,
)

_VALID_BODY = (
    "Human-authored summary paragraph.\n"
    "\n"
    "Some more prose about the change.\n"
    "\n"
    "Evidence-Ticket: OMN-14190\n"
    "Evidence-Source: OCC#1408\n"
)


# ---------------------------------------------------------------------------
# validate_pr_body — pure fail-closed assertions
# ---------------------------------------------------------------------------


def test_validate_pr_body_accepts_complete_stamp() -> None:
    assert validate_pr_body(_VALID_BODY) == []


def test_validate_pr_body_accepts_commit_sha_source() -> None:
    body = "body\nEvidence-Ticket: OMN-1\nEvidence-Source: 7420667f\n"
    assert validate_pr_body(body) == []


def test_validate_pr_body_flags_missing_source() -> None:
    problems = validate_pr_body("body\nEvidence-Ticket: OMN-14190\n")
    assert len(problems) == 1
    assert "Evidence-Source" in problems[0]


def test_validate_pr_body_flags_missing_ticket() -> None:
    problems = validate_pr_body("body\nEvidence-Source: OCC#1408\n")
    assert len(problems) == 1
    assert "Evidence-Ticket" in problems[0]


def test_validate_pr_body_flags_malformed_source() -> None:
    # A present-but-malformed source parses to None → treated as missing.
    problems = validate_pr_body(
        "body\nEvidence-Source: not-a-ref\nEvidence-Ticket: OMN-1\n"
    )
    assert len(problems) == 1
    assert "Evidence-Source" in problems[0]


def test_validate_pr_body_flags_fully_unstamped_body() -> None:
    problems = validate_pr_body("Just a plain PR body with no evidence lines.\n")
    assert len(problems) == 2


# ---------------------------------------------------------------------------
# parse_evidence_source_token — reuse of Piece-2 for token validation
# ---------------------------------------------------------------------------


def test_parse_evidence_source_token_occ_pr() -> None:
    source = parse_evidence_source_token("OCC#1408")
    assert source.render_token() == "OCC#1408"


def test_parse_evidence_source_token_commit_sha() -> None:
    source = parse_evidence_source_token("7420667f")
    assert source.render_token() == "7420667f"


def test_parse_evidence_source_token_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="invalid --evidence-source"):
        parse_evidence_source_token("garbage-value")


# ---------------------------------------------------------------------------
# stamp_pr_body — idempotency + merge semantics
# ---------------------------------------------------------------------------


def test_stamp_inserts_block_into_unstamped_body() -> None:
    stamped = stamp_pr_body(
        "A human-authored summary.\n",
        tickets=["OMN-14190"],
        evidence_source="OCC#1408",
    )
    assert "Evidence-Ticket: OMN-14190" in stamped
    assert "Evidence-Source: OCC#1408" in stamped
    assert "A human-authored summary." in stamped


def test_stamp_is_idempotent() -> None:
    once = stamp_pr_body(
        "Summary.\n", tickets=["OMN-14190"], evidence_source="OCC#1408"
    )
    twice = stamp_pr_body(once, tickets=["OMN-14190"], evidence_source="OCC#1408")
    assert once == twice
    # Exactly one Evidence-Ticket / Evidence-Source line — no duplication.
    assert twice.count("Evidence-Ticket: OMN-14190") == 1
    assert twice.count("Evidence-Source: OCC#1408") == 1


def test_stamp_idempotent_on_already_stamped_body_with_no_new_args() -> None:
    # Re-stamping a complete body with no new args canonicalizes to a fixpoint.
    once = stamp_pr_body(_VALID_BODY)
    twice = stamp_pr_body(once)
    assert once == twice
    assert twice.count("Evidence-Ticket: OMN-14190") == 1
    assert twice.count("Evidence-Source: OCC#1408") == 1


def test_stamp_merges_new_ticket_without_dropping_existing() -> None:
    stamped = stamp_pr_body(_VALID_BODY, tickets=["OMN-99999"])
    assert stamped.count("Evidence-Ticket: OMN-14190") == 1
    assert stamped.count("Evidence-Ticket: OMN-99999") == 1


def test_stamp_dedupes_repeated_ticket() -> None:
    stamped = stamp_pr_body(
        "Summary.\n",
        tickets=["OMN-14190", "OMN-14190"],
        evidence_source="OCC#1408",
    )
    assert stamped.count("Evidence-Ticket: OMN-14190") == 1


# ---------------------------------------------------------------------------
# CLI wiring — validate subcommand
# ---------------------------------------------------------------------------


def test_cli_validate_file_passes(tmp_path: Path) -> None:
    body_file = tmp_path / "pr_body.md"
    body_file.write_text(_VALID_BODY, encoding="utf-8")
    result = CliRunner().invoke(occ, ["validate", str(body_file)])
    assert result.exit_code == 0, result.output


def test_cli_validate_file_fails_closed(tmp_path: Path) -> None:
    body_file = tmp_path / "pr_body.md"
    body_file.write_text("No stamp here.\n", encoding="utf-8")
    result = CliRunner().invoke(occ, ["validate", str(body_file)])
    assert result.exit_code == 1
    assert "Evidence-Source" in result.output
    assert "Evidence-Ticket" in result.output


def test_cli_validate_stdin_passes() -> None:
    result = CliRunner().invoke(occ, ["validate"], input=_VALID_BODY)
    assert result.exit_code == 0, result.output


def test_cli_validate_stdin_fails_closed() -> None:
    result = CliRunner().invoke(occ, ["validate"], input="nope\n")
    assert result.exit_code == 1


def test_cli_validate_reports_all_files(tmp_path: Path) -> None:
    good = tmp_path / "good.md"
    good.write_text(_VALID_BODY, encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("missing\n", encoding="utf-8")
    result = CliRunner().invoke(occ, ["validate", str(good), str(bad)])
    assert result.exit_code == 1
    assert str(bad) in result.output
    assert str(good) not in result.output


def test_cli_validate_stdin_conflicts_with_files(tmp_path: Path) -> None:
    body_file = tmp_path / "pr_body.md"
    body_file.write_text(_VALID_BODY, encoding="utf-8")
    result = CliRunner().invoke(occ, ["validate", "--stdin", str(body_file)])
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


# ---------------------------------------------------------------------------
# CLI wiring — stamp subcommand
# ---------------------------------------------------------------------------


def test_cli_stamp_in_place_writes_file(tmp_path: Path) -> None:
    body_file = tmp_path / "pr_body.md"
    body_file.write_text("Summary of the change.\n", encoding="utf-8")
    result = CliRunner().invoke(
        occ,
        [
            "stamp",
            str(body_file),
            "--ticket",
            "OMN-14190",
            "--evidence-source",
            "OCC#1408",
            "--in-place",
        ],
    )
    assert result.exit_code == 0, result.output
    written = body_file.read_text(encoding="utf-8")
    assert "Evidence-Ticket: OMN-14190" in written
    assert "Evidence-Source: OCC#1408" in written


def test_cli_stamp_stdout_roundtrip_is_valid() -> None:
    runner = CliRunner()
    stamped = runner.invoke(
        occ,
        ["stamp", "--ticket", "OMN-14190", "--evidence-source", "OCC#1408"],
        input="Summary.\n",
    )
    assert stamped.exit_code == 0, stamped.output
    # The stamped output must pass validate.
    checked = runner.invoke(occ, ["validate"], input=stamped.output)
    assert checked.exit_code == 0, checked.output


def test_cli_stamp_rejects_bad_source() -> None:
    result = CliRunner().invoke(
        occ,
        ["stamp", "--ticket", "OMN-14190", "--evidence-source", "garbage"],
        input="Summary.\n",
    )
    assert result.exit_code != 0
    assert "invalid --evidence-source" in result.output


def test_cli_stamp_in_place_requires_file() -> None:
    result = CliRunner().invoke(
        occ,
        ["stamp", "--ticket", "OMN-14190", "--in-place"],
        input="Summary.\n",
    )
    assert result.exit_code != 0
    assert "--in-place requires a FILE" in result.output


# ---------------------------------------------------------------------------
# compute-contract-hash (OMN-15711 / FM5)
# ---------------------------------------------------------------------------

_CONTRACT_YAML = """\
ticket_id: OMN-15711
schema_version: "1.0"
title: Example contract for hash CLI tests
dod_evidence:
  - id: dod-omn-15711-example
    description: Example evidence item.
    check_command: "true"
  - id: dod-omn-15711-second
    description: Second evidence item.
    check_command: "true"
"""


@pytest.fixture
def contract_file(tmp_path: Path) -> Path:
    path = tmp_path / "OMN-15711.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


def test_compute_receipt_contract_hashes_matches_gate_whole_file_hash(
    contract_file: Path,
) -> None:
    result = compute_receipt_contract_hashes(contract_file)
    assert result == {
        "contract_sha256": f"sha256:{compute_contract_sha256(contract_file)}"
    }


def test_compute_receipt_contract_hashes_matches_gate_entry_hash(
    contract_file: Path,
) -> None:
    contract_data = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    result = compute_receipt_contract_hashes(
        contract_file, evidence_item_id="dod-omn-15711-example"
    )
    assert result == {
        "contract_sha256": f"sha256:{compute_contract_sha256(contract_file)}",
        "contract_entry_sha256": compute_contract_entry_sha256(
            contract_data, "dod-omn-15711-example"
        ),
    }


def test_compute_receipt_contract_hashes_unknown_entry_id_raises(
    contract_file: Path,
) -> None:
    from omnibase_core.validation.validator_receipt_gate import (
        ContractEntryNotFoundError,
    )

    with pytest.raises(ContractEntryNotFoundError):
        compute_receipt_contract_hashes(
            contract_file, evidence_item_id="does-not-exist"
        )


def test_cli_compute_contract_hash_whole_file_matches_gate(contract_file: Path) -> None:
    result = CliRunner().invoke(occ, ["compute-contract-hash", str(contract_file)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "contract_sha256": f"sha256:{compute_contract_sha256(contract_file)}"
    }


def test_cli_compute_contract_hash_with_entry_id_matches_gate(
    contract_file: Path,
) -> None:
    contract_data = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    result = CliRunner().invoke(
        occ,
        [
            "compute-contract-hash",
            str(contract_file),
            "--evidence-item-id",
            "dod-omn-15711-second",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "contract_sha256": f"sha256:{compute_contract_sha256(contract_file)}",
        "contract_entry_sha256": compute_contract_entry_sha256(
            contract_data, "dod-omn-15711-second"
        ),
    }


def test_cli_compute_contract_hash_unknown_entry_id_fails_closed(
    contract_file: Path,
) -> None:
    result = CliRunner().invoke(
        occ,
        [
            "compute-contract-hash",
            str(contract_file),
            "--evidence-item-id",
            "does-not-exist",
        ],
    )
    assert result.exit_code != 0
    assert "does-not-exist" in result.output


def test_cli_compute_contract_hash_missing_file_fails_closed(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        occ, ["compute-contract-hash", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code != 0
