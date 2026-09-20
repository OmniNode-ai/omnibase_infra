# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18866 -- the three probes that were named in ``PROBES_NOT_YET_WIRED``.

Every probe here carries BOTH controls, and that pairing is the point rather
than a convention. A probe only ever exercised against a healthy input has
never been shown capable of failing, and a probe only ever exercised against a
broken one has never been shown capable of passing. Either alone produces a
check whose green is uninformative, which is the exact shape Operating Rule 16
("prove a zero with a positive control") exists to refuse.

The ticket's acceptance criteria, and where each is pinned:

* AC1, one change adds the probe AND the check name --
  ``test_nothing_remains_declared_unwired`` and
  ``test_every_wired_name_has_a_callable_probe``.
* AC2, evidence on every check including passing ones --
  ``test_every_probe_emits_non_empty_evidence_on_both_verdicts``.
* AC3, a negative control per probe -- the three ``*_fails_*`` cases.
* AC4, a positive control per probe -- the three ``*_passes_*`` cases.
* AC5, indeterminate is not a pass -- the ``*_indeterminate_*`` cases, plus
  ``test_indeterminate_is_never_a_pass_for_any_probe``.

The subprocess-shaped calls are driven through the injected ``runner`` seam
rather than by monkeypatching ``subprocess.run``, so these tests need no lane,
no broker and no database, and they assert on the argv the probe BUILDS as well
as on the verdict it reaches. A test that only checked the verdict would pass
against a probe that read the wrong container.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.ci.lab_pass_receipt import (
    COMPOSE_DEV_INTEGRATION_CHECKS,
    CONSUMER_GROUP_LAG_CHECK,
    DELEGATION_GOLDEN_CHAIN_CHECK,
    MIGRATIONS_APPLIED_CHECK,
    PROBES_NOT_YET_WIRED,
    ModelBrokerAccess,
    ModelMigrationLedger,
    check_consumer_group_lag,
    check_delegation_golden_chain,
    check_migrations_applied,
    declared_forward_migrations,
    load_lag_sample,
    parse_group_list_argument,
    read_group_total_lag,
    sample_group_lag,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeRunner:
    """A ``subprocess.run``-shaped double that records the argv it was given."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        raises: Exception | None = None,
        per_call: list[tuple[int, str]] | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.per_call = list(per_call or [])
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if self.raises is not None:
            raise self.raises
        if self.per_call:
            code, out = self.per_call.pop(0)
            return subprocess.CompletedProcess(list(argv), code, out, self.stderr)
        return subprocess.CompletedProcess(
            list(argv), self.returncode, self.stdout, self.stderr
        )


LEDGER = ModelMigrationLedger(container="pg-x", database="omnibase_infra")
ACCESS = ModelBrokerAccess(container="broker-x", brokers="redpanda:9092")


def _describe(total_lag: int) -> str:
    return f"GROUP g\nSTATE Stable\nMEMBERS 1\nTOTAL-LAG {total_lag}\n"


def _canary_receipt(
    tmp_path: Path, *, success: bool, verdict: str = "chain_ok"
) -> Path:
    path = tmp_path / "chain-canary-receipt.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "success": success,
                    "verdict": verdict,
                    "detail": "one delegation, one correlation",
                    "links_proven": 4,
                    "links_total": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# AC1 -- the name and the probe landed together
# ---------------------------------------------------------------------------


def test_nothing_remains_declared_unwired() -> None:
    """The three names left the unwired list because they are wired.

    Falsifier for the inverse mistake: if a future change empties the list
    WITHOUT wiring something, the companion test below fails, so the two
    together refuse both directions of the rule-24 error.
    """
    assert PROBES_NOT_YET_WIRED == ()


def test_every_wired_name_has_a_callable_probe() -> None:
    """A declared check name with nothing computing it is the rule-24 defect."""
    probes = {
        MIGRATIONS_APPLIED_CHECK: check_migrations_applied,
        CONSUMER_GROUP_LAG_CHECK: check_consumer_group_lag,
        DELEGATION_GOLDEN_CHAIN_CHECK: check_delegation_golden_chain,
    }
    assert set(COMPOSE_DEV_INTEGRATION_CHECKS) == set(probes)
    for name, probe in probes.items():
        assert callable(probe), name


def test_the_three_names_are_not_also_http_checks() -> None:
    """Two lists declaring the same check would be two owners for one name."""
    from scripts.ci.lab_pass_receipt import COMPOSE_DEV_HTTP_CHECKS

    assert not set(COMPOSE_DEV_INTEGRATION_CHECKS) & set(COMPOSE_DEV_HTTP_CHECKS)


# ---------------------------------------------------------------------------
# migrations_applied
# ---------------------------------------------------------------------------


def test_migrations_passes_when_every_declared_id_is_recorded() -> None:
    """POSITIVE CONTROL. Shapes taken from the live lane on 2026-09-20."""
    runner = FakeRunner(stdout="docker/001_a.sql\ndocker/002_b.sql\n")
    check = check_migrations_applied(
        ["docker/001_a.sql", "docker/002_b.sql"], LEDGER, runner=runner
    )
    assert check.ok is True
    assert check.indeterminate is False
    assert "none missing" in check.evidence
    # The probe must read the container it was told to read. Asserting only the
    # verdict would pass against a probe that queried something else entirely.
    argv = runner.calls[0]
    assert argv[:3] == ["docker", "exec", "pg-x"]
    assert "omnibase_infra" in argv


def test_migrations_fails_when_a_declared_id_is_missing() -> None:
    """NEGATIVE CONTROL -- a lane serving an image whose schema never landed.

    Every readiness endpoint answers 200 in this state, which is why the
    absence has to be asked about directly.
    """
    runner = FakeRunner(stdout="docker/001_a.sql\n")
    check = check_migrations_applied(
        ["docker/001_a.sql", "docker/002_b.sql"], LEDGER, runner=runner
    )
    assert check.ok is False
    assert check.indeterminate is False
    assert "DECLARED BUT NOT APPLIED" in check.evidence
    assert "docker/002_b.sql" in check.evidence


def test_migrations_tolerates_a_lane_ahead_of_this_tree() -> None:
    """A descendant lane may carry migrations this tree does not declare.

    OMN-18388 already accepts convergence onto a DESCENDANT of the merge sha,
    so failing on an extra row would make this check contradict the convergence
    guard standing beside it in the same receipt.
    """
    runner = FakeRunner(stdout="docker/001_a.sql\ndocker/002_b.sql\ndocker/003_c.sql\n")
    check = check_migrations_applied(["docker/001_a.sql"], LEDGER, runner=runner)
    assert check.ok is True
    assert "not declared here" in check.evidence


def test_migrations_indeterminate_when_the_ledger_cannot_be_read() -> None:
    runner = FakeRunner(returncode=1, stderr="could not connect to server")
    check = check_migrations_applied(["docker/001_a.sql"], LEDGER, runner=runner)
    assert check.ok is False
    assert check.indeterminate is True


def test_migrations_indeterminate_on_zero_rows_rather_than_passing() -> None:
    """An empty ledger reads identically to a table the probe failed to query."""
    check = check_migrations_applied(
        ["docker/001_a.sql"], LEDGER, runner=FakeRunner(stdout="\n  \n")
    )
    assert check.ok is False
    assert check.indeterminate is True


def test_migrations_indeterminate_on_an_empty_declaration() -> None:
    """A vacuous comparison passes against every lane, so it is not a pass."""
    check = check_migrations_applied([], LEDGER, runner=FakeRunner(stdout="x\n"))
    assert check.ok is False
    assert check.indeterminate is True


def test_declared_forward_migrations_refuses_an_absent_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        declared_forward_migrations(tmp_path / "nope")


def test_declared_forward_migrations_refuses_an_empty_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="refusing to compare"):
        declared_forward_migrations(tmp_path)


def test_declared_forward_migrations_is_flat_and_sql_only(tmp_path: Path) -> None:
    """Matches check_schema_fingerprint.py's own glob, which is non-recursive.

    The ``*.sh`` siblings are bootstrap helpers the runner does not ledger and
    the subdirectories are node-owned streams with ledgers of their own, so
    including either would make the probe demand rows that never exist.
    """
    (tmp_path / "002_b.sql").write_text("", encoding="utf-8")
    (tmp_path / "001_a.sql").write_text("", encoding="utf-8")
    (tmp_path / "000_boot.sh").write_text("", encoding="utf-8")
    nested = tmp_path / "nodes"
    nested.mkdir()
    (nested / "900_nested.sql").write_text("", encoding="utf-8")

    assert declared_forward_migrations(tmp_path) == (
        "docker/001_a.sql",
        "docker/002_b.sql",
    )


def test_the_real_tree_matches_the_ledger_shape() -> None:
    """The declared ids are spelled the way the lane records them.

    Measured live 2026-09-20: 90 declared, 90 recorded, zero on either side of
    the difference. This asserts the SHAPE rather than the count, because the
    count moves every time a migration lands and a test that pinned it would
    fail on the next one for no reason anybody cares about.
    """
    declared = declared_forward_migrations(Path("docker/migrations/forward"))
    assert declared, "the repository declares forward migrations"
    for migration_id in declared:
        assert migration_id.startswith("docker/")
        assert migration_id.endswith(".sql")
        assert "/" not in migration_id[len("docker/") :]


# ---------------------------------------------------------------------------
# consumer_group_lag
# ---------------------------------------------------------------------------


def test_lag_passes_under_the_bound_and_not_growing() -> None:
    """POSITIVE CONTROL."""
    check = check_consumer_group_lag(
        ACCESS,
        ["g"],
        max_lag=100,
        first_sample={"g": 12},
        runner=FakeRunner(stdout=_describe(10)),
    )
    assert check.ok is True
    assert "none over bound" in check.evidence
    assert "none growing" in check.evidence


def test_lag_fails_over_the_declared_bound() -> None:
    """NEGATIVE CONTROL, first arm."""
    check = check_consumer_group_lag(
        ACCESS, ["g"], max_lag=5, runner=FakeRunner(stdout=_describe(498))
    )
    assert check.ok is False
    assert check.indeterminate is False
    assert "over bound" in check.evidence


def test_lag_fails_when_growing_even_inside_a_generous_bound() -> None:
    """NEGATIVE CONTROL, second arm -- the one that matters.

    This is the OMN-18851 shape: the savings writer sat at lag 498 for nine
    days. A bound generous enough not to flag every busy group admits that
    number, so the bound alone could not have caught it. Growth across two
    samples is scale-free and does.
    """
    check = check_consumer_group_lag(
        ACCESS,
        ["g"],
        max_lag=100_000,
        first_sample={"g": 498},
        runner=FakeRunner(stdout=_describe(601)),
    )
    assert check.ok is False
    assert "GROWING" in check.evidence
    assert "498->601" in check.evidence


def test_lag_does_not_fail_on_high_but_flat_lag() -> None:
    """A backlog being worked is not a stopped consumer.

    Flat-lag alarming is the noise that gets an alarm muted, and a muted alarm
    is how the nine-day freeze would have gone unnoticed a second time.
    """
    check = check_consumer_group_lag(
        ACCESS,
        ["g"],
        max_lag=100_000,
        first_sample={"g": 498},
        runner=FakeRunner(stdout=_describe(498)),
    )
    assert check.ok is True


def test_lag_says_so_when_it_did_not_measure_growth() -> None:
    """Without a baseline the check covers the bound ONLY, and admits it."""
    check = check_consumer_group_lag(
        ACCESS, ["g"], max_lag=100, runner=FakeRunner(stdout=_describe(10))
    )
    assert check.ok is True
    assert "growth was NOT measured" in check.evidence


def test_lag_indeterminate_when_a_group_cannot_be_read() -> None:
    check = check_consumer_group_lag(
        ACCESS,
        ["g"],
        max_lag=100,
        runner=FakeRunner(returncode=1, stderr="SASL authentication failed"),
    )
    assert check.ok is False
    assert check.indeterminate is True


def test_lag_indeterminate_when_no_groups_are_declared() -> None:
    """A lane declaring no groups is not a lane with no lag."""
    check = check_consumer_group_lag(ACCESS, [], max_lag=100, runner=FakeRunner())
    assert check.ok is False
    assert check.indeterminate is True


def test_total_lag_is_matched_by_label_not_column_offset() -> None:
    """rpk pads that table differently between versions.

    Mirrors declared_consumer_groups.parse_group_describe, which matches the
    label for the same reason. One parse rule for one output format.
    """
    padded = "GROUP      g\nSTATE           Stable\nTOTAL-LAG          42\n"
    assert read_group_total_lag(ACCESS, "g", runner=FakeRunner(stdout=padded)) == 42


def test_read_group_total_lag_refuses_output_with_no_total_lag_line() -> None:
    with pytest.raises(ValueError, match="no TOTAL-LAG"):
        read_group_total_lag(ACCESS, "g", runner=FakeRunner(stdout="STATE Dead\n"))


def test_sasl_is_all_or_nothing() -> None:
    """A partial credential is dropped by rpk and reads as an unauthenticated
    probe, so it is refused at construction rather than at the broker."""
    with pytest.raises(ValueError, match="all-or-nothing"):
        ModelBrokerAccess(container="c", brokers="redpanda:9092", sasl_username="u")


def test_the_credential_never_reaches_the_evidence() -> None:
    """A secret in evidence is a secret in an uploaded artifact."""
    access = ModelBrokerAccess(
        container="c",
        brokers="redpanda:9092",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_username="probe-user",
        sasl_password="probe-secret-value",
    )
    check = check_consumer_group_lag(
        access, ["g"], max_lag=100, runner=FakeRunner(stdout=_describe(1))
    )
    assert "probe-secret-value" not in check.evidence
    # The flags themselves must still carry it, or the probe cannot authenticate.
    assert "pass=probe-secret-value" in access.rpk_flags()


def test_a_baseline_omits_groups_it_could_not_read(tmp_path: Path) -> None:
    """A zero baseline would make any later reading look like growth."""
    runner = FakeRunner(per_call=[(0, _describe(7)), (1, "")])
    assert sample_group_lag(ACCESS, ["good", "bad"], runner=runner) == {"good": 7}


def test_an_unreadable_baseline_file_disables_growth_rather_than_faking_it(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "sample.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_lag_sample(broken) is None
    assert load_lag_sample(tmp_path / "absent.json") is None


def test_group_list_argument_accepts_commas_and_newlines() -> None:
    assert parse_group_list_argument(" a, b\nc ,, ") == ("a", "b", "c")
    assert parse_group_list_argument("") == ()


# ---------------------------------------------------------------------------
# delegation_golden_chain
# ---------------------------------------------------------------------------


def test_delegation_passes_on_a_successful_canary_receipt(tmp_path: Path) -> None:
    """POSITIVE CONTROL."""
    check = check_delegation_golden_chain(_canary_receipt(tmp_path, success=True))
    assert check.ok is True
    assert "one live delegation" in check.evidence


def test_delegation_fails_when_the_canary_reports_no_terminal(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL -- the OMN-18852 shape.

    A deployed consumer refusing the payload at its decode boundary produces
    exactly this: a delegation submitted, no terminal for its correlation.
    Reproduced live on 2026-09-20 against the dev lane, where the boundary
    refused an undeclared field.
    """
    check = check_delegation_golden_chain(
        _canary_receipt(tmp_path, success=False, verdict="terminal_missing")
    )
    assert check.ok is False
    assert check.indeterminate is False
    assert "terminal_missing" in check.evidence


def test_delegation_does_not_require_a_complete_five_link_chain_proof(
    tmp_path: Path,
) -> None:
    """Link 5 has no leg in any probe today (OMN-16964).

    Binding this check to ``chain_proof_complete`` would make it permanently
    red and, within a week, permanently ignored. The counts are carried into
    the evidence so the weaker claim is visible rather than assumed.
    """
    check = check_delegation_golden_chain(_canary_receipt(tmp_path, success=True))
    assert check.ok is True
    assert "4 of 5" in check.evidence
    assert "NOT the bar" in check.evidence


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("", "empty"),
        ("{not json", "not valid JSON"),
        ("[]", "not a JSON object"),
        ('{"nope": 1}', "no 'result' object"),
        ('{"result": {"verdict": "x"}}', "missing or not a boolean"),
    ],
)
def test_delegation_indeterminate_on_an_unusable_receipt(
    tmp_path: Path, body: str, reason: str
) -> None:
    path = tmp_path / "chain-canary-receipt.json"
    path.write_text(body, encoding="utf-8")
    check = check_delegation_golden_chain(path)
    assert check.ok is False
    assert check.indeterminate is True
    assert reason in check.evidence


def test_delegation_indeterminate_when_the_dispatch_never_ran(
    tmp_path: Path,
) -> None:
    """An absent receipt is a fact about this probe's own execution path.

    Reporting it as a lane FAILURE would blame the lane for the probe dying,
    which is the attribution error the canary's own summary step warns about.
    """
    check = check_delegation_golden_chain(tmp_path / "never-written.json")
    assert check.ok is False
    assert check.indeterminate is True
    assert "did not run" in check.evidence


# ---------------------------------------------------------------------------
# AC2 / AC5 -- properties that hold across all three
# ---------------------------------------------------------------------------


def _every_verdict(tmp_path: Path) -> list[object]:
    return [
        check_migrations_applied(
            ["docker/1.sql"], LEDGER, runner=FakeRunner(stdout="docker/1.sql\n")
        ),
        check_migrations_applied(
            ["docker/1.sql"], LEDGER, runner=FakeRunner(stdout="\n")
        ),
        check_migrations_applied(
            ["docker/1.sql"], LEDGER, runner=FakeRunner(returncode=1)
        ),
        check_consumer_group_lag(
            ACCESS, ["g"], max_lag=9, runner=FakeRunner(stdout=_describe(1))
        ),
        check_consumer_group_lag(
            ACCESS, ["g"], max_lag=0, runner=FakeRunner(stdout=_describe(1))
        ),
        check_consumer_group_lag(ACCESS, [], max_lag=9, runner=FakeRunner()),
        check_delegation_golden_chain(_canary_receipt(tmp_path, success=True)),
        check_delegation_golden_chain(_canary_receipt(tmp_path, success=False)),
        check_delegation_golden_chain(tmp_path / "absent.json"),
    ]


def test_every_probe_emits_non_empty_evidence_on_both_verdicts(
    tmp_path: Path,
) -> None:
    """AC2. An ``ok: true`` with no evidence is indistinguishable from a check
    that was never run, and the receipt model refuses it -- but the refusal
    arrives at emit time, which is too late to tell anyone what happened."""
    for check in _every_verdict(tmp_path):
        assert check.evidence.strip(), check.name  # type: ignore[attr-defined]


def test_indeterminate_is_never_a_pass_for_any_probe(tmp_path: Path) -> None:
    """AC5. An unproven premise must leave the receipt non-PASS, so the
    delivery gate stays closed rather than opening on a check nobody
    established."""
    for check in _every_verdict(tmp_path):
        if check.indeterminate:  # type: ignore[attr-defined]
            assert check.ok is False  # type: ignore[attr-defined]


def test_each_probe_names_itself_correctly(tmp_path: Path) -> None:
    """A check's name is its key on the receipt; a typo silently orphans it."""
    names = {c.name for c in _every_verdict(tmp_path)}  # type: ignore[attr-defined]
    assert names == set(COMPOSE_DEV_INTEGRATION_CHECKS)
