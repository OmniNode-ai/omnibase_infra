# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18866 -- the three probes that were named in ``PROBES_NOT_YET_WIRED``.

Every probe here carries BOTH controls, and that pairing is the point rather
than a convention. A probe only ever exercised against a healthy input has
never been shown capable of failing, and a probe only ever exercised against a
broken one has never been shown capable of passing. Either alone produces a
check whose green is uninformative, which is the exact shape Operating Rule 16
("prove a zero with a positive control") exists to refuse.

TWO of the three are wired into the compose-dev receipt. The third,
``delegation_golden_chain``, was wired and reverted within the hour: its probe
is correct and is kept, but binding the receipt to an inline chain-canary
dispatch gated DELIVERY on a separately-owned surface that was already red, so
a healthy lane was refused for staging. The last section of this module is the
red-control set for that and for the other way this shipped wrong.

The ticket's acceptance criteria, and where each is pinned:

* AC1, one change adds the probe AND the check name --
  ``test_the_unwired_list_names_exactly_what_is_unwired`` and
  ``test_every_wired_name_has_a_callable_probe``, which together refuse both
  directions: a name removed while nothing computes it, and a name left on the
  unwired list while the job emits it.
* AC2, evidence on every check including passing ones --
  ``test_every_probe_emits_non_empty_evidence_on_both_verdicts``.
* AC3, a negative control per probe -- the ``*_fails_*`` cases.
* AC4, a positive control per probe -- the ``*_passes_*`` cases.
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
    GroupSourceError,
    ModelBrokerAccess,
    ModelMigrationLedger,
    check_consumer_group_lag,
    check_delegation_golden_chain,
    check_migrations_applied,
    declared_forward_migrations,
    load_declared_groups,
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


def test_the_unwired_list_names_exactly_what_is_unwired() -> None:
    """Two of the three names left the list because they are wired; one came
    back, and the list has to say so.

    `delegation_golden_chain` was wired and then reverted within the hour --
    not because the probe was wrong but because the WIRING gated delivery on a
    separately-owned surface that was already red. Rule 24 cuts both ways: a
    name must not be removed while nothing computes it, and it must not stay
    removed while the job no longer emits it.

    Falsifier for the inverse mistake: if a future change empties this list
    without wiring something, the companion test below fails, so the two
    together refuse both directions of the error.
    """
    assert PROBES_NOT_YET_WIRED == (DELEGATION_GOLDEN_CHAIN_CHECK,)
    assert not set(PROBES_NOT_YET_WIRED) & set(COMPOSE_DEV_INTEGRATION_CHECKS), (
        "a name cannot be both wired and declared unwired"
    )


def test_every_wired_name_has_a_callable_probe() -> None:
    """A declared check name with nothing computing it is the rule-24 defect."""
    probes = {
        MIGRATIONS_APPLIED_CHECK: check_migrations_applied,
        CONSUMER_GROUP_LAG_CHECK: check_consumer_group_lag,
    }
    assert set(COMPOSE_DEV_INTEGRATION_CHECKS) == set(probes)
    for name, probe in probes.items():
        assert callable(probe), name
    # The unwired one keeps its probe too, so re-wiring it is a workflow
    # change and not a rewrite.
    assert callable(check_delegation_golden_chain)


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
    assert names == set(COMPOSE_DEV_INTEGRATION_CHECKS) | {
        DELEGATION_GOLDEN_CHAIN_CHECK
    }


# ---------------------------------------------------------------------------
# OMN-18866 follow-up: the two ways this shipped wrong in production
# ---------------------------------------------------------------------------
#
# Both of these are RED CONTROLS for real failures, not hypotheticals. On
# run 35488731787 a healthy, converged dev lane produced a FAIL receipt with
# seven checks passing and these two failing, and that refused every dev sha
# for staging under rule 24(b) until it was reverted.


def test_an_unreadable_group_source_is_not_reported_as_an_empty_declaration(
    tmp_path: Path,
) -> None:
    """RED CONTROL 1, reproducing the exact production evidence string.

    The deriving step failed silently -- exit 1, no output, because its stdout
    was swallowed by a command substitution -- so its step output was never
    set, the workflow expression delivered an empty string, and this check
    said "no consumer groups were declared for this lane". That sentence was
    FALSE: the lane declares six. The check was refusing for the right reason
    applied to the wrong fact, which is the worst kind of correct.

    The two facts now have two outcomes, and the falsifier is that they stop
    being distinguishable.
    """
    missing = tmp_path / "never-written.txt"
    with pytest.raises(GroupSourceError, match="does not exist"):
        load_declared_groups(missing)

    source_error = "the declared-group source /x/declared-groups.txt does not exist"
    unreadable = check_consumer_group_lag(
        ACCESS, [], max_lag=10, runner=FakeRunner(), source_error=source_error
    )
    genuinely_empty = check_consumer_group_lag(
        ACCESS, [], max_lag=10, runner=FakeRunner()
    )

    # Both refuse -- neither may become a pass.
    assert unreadable.ok is False and unreadable.indeterminate is True
    assert genuinely_empty.ok is False and genuinely_empty.indeterminate is True
    # ...but they must not say the same thing.
    assert unreadable.evidence != genuinely_empty.evidence
    assert "does not exist" in unreadable.evidence
    assert "read successfully and contained no groups" in genuinely_empty.evidence


def test_a_populated_group_file_restores_the_live_reading(tmp_path: Path) -> None:
    """GREEN CONTROL 1: the corrected input, in the shape the deriver writes.

    Six groups, one per line, exactly as measured on the live lane.
    """
    source = tmp_path / "declared-groups.txt"
    source.write_text(
        "\n".join(
            f"local.omnimarket-projections.{name}-writer.consume.v1"
            for name in (
                "delegation",
                "live-events",
                "registration",
                "savings",
                "tenant-credentials",
                "tenant-registry",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    groups = load_declared_groups(source)
    assert len(groups) == 6
    assert "local.omnimarket-projections.savings-writer.consume.v1" in groups

    check = check_consumer_group_lag(
        ACCESS,
        groups,
        max_lag=10000,
        first_sample=dict.fromkeys(groups, 0),
        runner=FakeRunner(stdout=_describe(0)),
    )
    assert check.ok is True
    assert "6 declared group(s)" in check.evidence


def test_a_trailing_newline_only_file_is_an_empty_declaration_not_an_error(
    tmp_path: Path,
) -> None:
    """The boundary between the two facts, pinned so it cannot drift."""
    source = tmp_path / "declared-groups.txt"
    source.write_text("\n\n  \n", encoding="utf-8")
    assert load_declared_groups(source) == ()


def test_the_delegation_check_is_declared_unwired_again(tmp_path: Path) -> None:
    """RED CONTROL 2, and the reason it is a REMOVAL rather than a repair.

    The delegation check was bound to a chain-canary dispatch fired inline by
    the job that gates delivery. The canary is separately owned and its
    scheduled runs were already red, so the binding refused delivery for shas
    with no defect in them. The probe was behaving exactly as designed; the
    WIRING was the defect.

    So the check comes off the emitted set and the name goes back on the
    unwired list. The probe, its flag and its tests all stay, because the fix
    is a wiring change -- the canary emitting its own sha-keyed receipt -- and
    not a rewrite.

    Falsifier: the name is emitted by the compose-dev job again without that
    receipt surface existing, which would re-block delivery on another lane's
    dispatch.
    """
    assert DELEGATION_GOLDEN_CHAIN_CHECK in PROBES_NOT_YET_WIRED
    assert DELEGATION_GOLDEN_CHAIN_CHECK not in COMPOSE_DEV_INTEGRATION_CHECKS

    # The probe itself is retained and still works, in both directions.
    assert (
        check_delegation_golden_chain(_canary_receipt(tmp_path, success=True)).ok
        is True
    )
    assert (
        check_delegation_golden_chain(_canary_receipt(tmp_path, success=False)).ok
        is False
    )


def test_the_emitting_job_no_longer_dispatches_the_chain_canary() -> None:
    """The workflow, not just the module, must have stopped doing it.

    Asserting only the module's constant would pass while the job still fired
    a dispatch whose failure took the receipt with it.
    """
    workflow = Path(".github/workflows/runtime-rebuild-trigger.yml").read_text(
        encoding="utf-8"
    )
    verify = workflow.split("verify-lane-converged:", 1)[1]
    assert "onex skill chain_canary" not in verify
    assert "--chain-canary-receipt" not in verify


def test_the_deriver_is_invoked_from_the_repository_root() -> None:
    """The `cd` into a subdirectory is what died silently in CI.

    Every working step in that job calls `uv run python <path-from-root>`, and
    the one step that did otherwise exited 1 in 0.086s having printed nothing.
    Pinned so nobody reintroduces the subdirectory form.
    """
    raw = Path(".github/workflows/runtime-rebuild-trigger.yml").read_text(
        encoding="utf-8"
    )
    # COMMENT LINES ARE STRIPPED FIRST, and that is not a convenience. The
    # comment beside the fixed step QUOTES the broken form in order to explain
    # it, so a naive substring assertion over the whole file fails on the
    # documentation of the very thing it is checking for. That is the rule-15
    # shape -- prose mentioning a literal a matcher reacts to -- reproduced in
    # a test rather than in a PR body, and the fix is the same: judge the
    # executable text, not the prose about it.
    executable = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert "cd scripts/runtime_build" not in executable
    assert (
        "uv run python scripts/runtime_build/declared_consumer_groups.py" in executable
    )
    # And the result must travel as a file, not a step output.
    assert "declared-groups.txt" in executable
    assert "--consumer-groups-file" in executable
    assert "steps.groups.outputs.groups" not in executable


def unwired_claim_disagreement(
    workflow_text: str, unwired: tuple[str, ...]
) -> str | None:
    """Return the disagreement between the job's prose and the unwired set.

    Rule 15 says judge the executable text and not the prose about it, and
    :func:`test_the_deriver_is_invoked_from_the_repository_root` above strips
    comments for exactly that reason. This function is the narrow exception,
    and the reason it earns one is measured rather than argued: the revert of
    `delegation_golden_chain` left the constant correct and left this job's
    comment asserting the list was empty, and that sentence survived a day of
    readers because every test here reads the constant and none read the claim
    beside it. A coverage claim nobody checks is the same defect as a check
    nobody runs, which is what OMN-18864 is about.

    It checks ONE thing and makes no attempt to parse English: an emptiness
    claim about the list cannot stand while the list is not empty.
    """
    claims_empty = "PROBES_NOT_YET_WIRED is now empty" in workflow_text
    if claims_empty and unwired:
        return (
            "the workflow states PROBES_NOT_YET_WIRED is now empty while it names "
            f"{', '.join(unwired)}"
        )
    return None


def test_the_workflow_prose_agrees_with_the_unwired_set() -> None:
    """The job may not claim coverage the receipt does not carry.

    Falsifier: the emitting job asserts the unwired list is empty while a
    check name sits on it, which is what shipped on 2026-09-20 and what no
    test here could see.
    """
    workflow = Path(".github/workflows/runtime-rebuild-trigger.yml").read_text(
        encoding="utf-8"
    )
    assert unwired_claim_disagreement(workflow, PROBES_NOT_YET_WIRED) is None


def test_the_prose_check_catches_the_drift_it_exists_for() -> None:
    """Negative control: without it the assertion above proves nothing.

    The check is one substring, so a check that had quietly stopped matching
    would look exactly like a workflow in agreement with the constant.
    """
    drifted = "# scripts/ci/lab_pass_receipt.py's PROBES_NOT_YET_WIRED is now empty.\n"
    assert (
        unwired_claim_disagreement(drifted, (DELEGATION_GOLDEN_CHAIN_CHECK,))
        is not None
    )
    # And it stays silent in the two states that are NOT the defect: an
    # emptiness claim with a genuinely empty list, and a non-empty list with
    # no claim about it.
    assert unwired_claim_disagreement(drifted, ()) is None
    assert (
        unwired_claim_disagreement(
            "# nothing said here\n", (DELEGATION_GOLDEN_CHAIN_CHECK,)
        )
        is None
    )
