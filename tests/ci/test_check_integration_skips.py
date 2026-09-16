# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the integration silent-skip false-green gate (OMN-14172).

Encodes the regression the gate exists to prevent: an `@pytest.mark.integration`
real-service test that self-skips because a PROVISIONED service (Postgres) looks
absent must turn the gate RED (case b), while a job where the same tests actually
ran — plus a legitimately-optional Kafka-broker skip — must PASS (case a).

The synthetic JUnit fixtures below mirror the exact schema pytest emits for the
two curated omnibase_infra Postgres proofs
(``test_postgres_repository_runtime_integration`` — creates its own table — and
``test_registration_storage_postgres_uuid_cast`` — the OMN-9041 UUID-cast
regression guard). The ``<skipped message=...>`` strings are the verbatim
Postgres-absence reasons those tests emit, grepped from ``tests/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci.check_integration_skips import (
    GuardConfig,
    evaluate,
    main,
    parse_junit,
    selftest,
)

_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "integration_skip_guard.yaml"
)

# Postgres provisioned: the two real-DB proofs RAN; only a Kafka-broker skip.
_JUNIT_PROVISIONED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" skipped="1">
 <testcase classname="tests.integration.runtime.db.test_postgres_repository_runtime_integration"
   name="test_execute_select_returns_rows_against_real_postgres" time="0.1"/>
 <testcase classname="tests.integration.handlers.test_registration_storage_postgres_uuid_cast"
   name="test_query_registrations_casts_node_id_uuid" time="0.1"/>
 <testcase classname="tests.integration.event_bus.test_kafka_boundary" name="test_kafka_roundtrip">
   <skipped type="pytest.skip" message="Redpanda broker not reachable at localhost:9092: connection refused"/>
 </testcase>
</testsuite></testsuites>"""

# Postgres removed / OMNIBASE_INFRA_DB_URL unset: both real-DB proofs SILENTLY
# SKIP. This is the exact false-green that would let an OMN-9041-class defect
# reach dev.
_JUNIT_SILENT_SKIP = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" skipped="2">
 <testcase classname="tests.integration.runtime.db.test_postgres_repository_runtime_integration"
   name="test_execute_select_returns_rows_against_real_postgres" time="0.0">
   <skipped type="pytest.skip" message="PostgreSQL integration tests skipped: Missing OMNIBASE_INFRA_DB_URL or required POSTGRES_* fallback variables"/>
 </testcase>
 <testcase classname="tests.integration.handlers.test_registration_storage_postgres_uuid_cast"
   name="test_query_registrations_casts_node_id_uuid" time="0.0">
   <skipped type="pytest.skip" message="PostgreSQL not available (set OMNIBASE_INFRA_DB_URL or POSTGRES_HOST+POSTGRES_PASSWORD)"/>
 </testcase>
</testsuite></testsuites>"""


@pytest.fixture
def cfg() -> GuardConfig:
    return GuardConfig.load(_CONFIG)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


@pytest.mark.unit
def test_shipped_config_is_fail_closed(cfg: GuardConfig) -> None:
    assert cfg.silent_skip_allowed is False
    assert "postgres" in cfg.required_service_patterns
    assert cfg.required_service_patterns["postgres"], "postgres needs skip patterns"
    assert cfg.require_executed_min >= 1


@pytest.mark.unit
def test_case_a_provisioned_service_passes(cfg: GuardConfig, tmp_path: Path) -> None:
    junit = _write(tmp_path, "a.xml", _JUNIT_PROVISIONED)
    stats = parse_junit([junit])
    assert stats.executed == 2  # both real-DB proofs ran
    assert evaluate(stats, cfg, strict=False) == []


@pytest.mark.unit
def test_case_b_silent_skip_goes_red(cfg: GuardConfig, tmp_path: Path) -> None:
    junit = _write(tmp_path, "b.xml", _JUNIT_SILENT_SKIP)
    stats = parse_junit([junit])
    assert stats.executed == 0
    violations = evaluate(stats, cfg, strict=False)
    assert violations, "reintroduced missing-service silent skip must be caught"
    joined = " ".join(violations)
    assert "FALSE-GREEN" in joined
    assert "postgres" in joined


@pytest.mark.unit
def test_case_b_main_exit_code_is_nonzero(cfg: GuardConfig, tmp_path: Path) -> None:
    junit = _write(tmp_path, "b.xml", _JUNIT_SILENT_SKIP)
    assert main(["--junit", str(junit), "--config", str(_CONFIG)]) == 1


@pytest.mark.unit
def test_case_a_main_exit_code_is_zero(cfg: GuardConfig, tmp_path: Path) -> None:
    junit = _write(tmp_path, "a.xml", _JUNIT_PROVISIONED)
    assert main(["--junit", str(junit), "--config", str(_CONFIG)]) == 0


@pytest.mark.unit
def test_kafka_broker_skip_is_not_a_false_green(
    cfg: GuardConfig, tmp_path: Path
) -> None:
    body = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" skipped="1">
 <testcase classname="t" name="ran" time="0.1"/>
 <testcase classname="t" name="kafka"><skipped type="pytest.skip"
   message="Redpanda broker not reachable at localhost:9092: connection refused"/></testcase>
</testsuite></testsuites>"""
    junit = _write(tmp_path, "kafka.xml", body)
    assert evaluate(parse_junit([junit]), cfg, strict=False) == []


@pytest.mark.unit
def test_catalog_manifest_skip_is_not_a_false_green(
    cfg: GuardConfig, tmp_path: Path
) -> None:
    # tests/integration/test_catalog_extra_networks.py emits this catalog-data
    # skip; it names "postgres" but is NOT a service-absence false-green.
    body = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" skipped="1">
 <testcase classname="t" name="ran" time="0.1"/>
 <testcase classname="t" name="catalog"><skipped type="pytest.skip"
   message="postgres manifest not found in catalog — skipping"/></testcase>
</testsuite></testsuites>"""
    junit = _write(tmp_path, "catalog.xml", body)
    assert evaluate(parse_junit([junit]), cfg, strict=False) == []


@pytest.mark.unit
def test_all_postgres_absence_strings_are_caught(
    cfg: GuardConfig, tmp_path: Path
) -> None:
    # Every real Postgres-absence skip string grepped from tests/ must classify
    # as a false-green (offending service == postgres), so the gate stays correct
    # if the curated set broadens.
    reasons = [
        "PostgreSQL integration tests skipped: PostgreSQL configured but not reachable",
        "PostgreSQL not available (set OMNIBASE_INFRA_DB_URL)",
        "PostgreSQL not configured or not reachable",
        "PostgreSQL not reachable",
        "PostgreSQL DSN not available",
        "Database not configured (set OMNIBASE_INFRA_DB_URL or POSTGRES_PASSWORD)",
        "Database not reachable: Connection refused",
        "No database URL configured (set OMNIBASE_INFRA_DB_URL)",
    ]
    cases = "".join(
        f'<testcase classname="t" name="c{i}"><skipped type="pytest.skip" '
        f'message="{reason}"/></testcase>'
        for i, reason in enumerate(reasons)
    )
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" tests="{len(reasons) + 1}" '
        f'skipped="{len(reasons)}">'
        '<testcase classname="t" name="ran" time="0.1"/>'
        f"{cases}</testsuite></testsuites>"
    )
    junit = _write(tmp_path, "all.xml", body)
    violations = evaluate(parse_junit([junit]), cfg, strict=False)
    assert len(violations) == len(reasons), violations
    assert all("FALSE-GREEN" in v for v in violations)


@pytest.mark.unit
def test_zero_collection_is_a_false_green(cfg: GuardConfig, tmp_path: Path) -> None:
    body = '<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="0"/></testsuites>'
    junit = _write(tmp_path, "empty.xml", body)
    violations = evaluate(parse_junit([junit]), cfg, strict=False)
    assert any("UNDER-COLLECTION" in v for v in violations)


@pytest.mark.unit
def test_missing_report_fails_closed(cfg: GuardConfig) -> None:
    assert main(["--junit", "/nonexistent/nope.xml", "--config", str(_CONFIG)]) == 2


@pytest.mark.unit
def test_strict_flags_unclassified_skip(cfg: GuardConfig, tmp_path: Path) -> None:
    body = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" skipped="1">
 <testcase classname="t" name="ran" time="0.1"/>
 <testcase classname="t" name="weird"><skipped type="pytest.skip"
   message="some brand new unrecognised reason nobody classified yet"/></testcase>
</testsuite></testsuites>"""
    junit = _write(tmp_path, "weird.xml", body)
    assert evaluate(parse_junit([junit]), cfg, strict=False) == []
    assert any(
        "UNCLASSIFIED-SKIP" in v
        for v in evaluate(parse_junit([junit]), cfg, strict=True)
    )


@pytest.mark.unit
def test_embedded_selftest_passes(cfg: GuardConfig) -> None:
    assert selftest(cfg) == 0


# =============================================================================
# The curated set is declared in two files, and they must agree (OMN-18419)
# =============================================================================
# `curated_test_paths` in the config is documented as INFORMATIONAL -- the gate
# keys on junit skip reasons, not on that list -- which is exactly why it can
# drift. The list and the ci.yml step that runs the proofs are the same
# decision written twice, and a proof dropped from the step while the list
# still names it leaves a curated Postgres proof that nothing runs, with no
# signal anywhere. This pins them to each other. It is check-only: if it fails,
# add the path to whichever of the two files is missing it.

_CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
_CURATED_PROOF_STEP_ID = "run_curated_proofs"


def _declared_curated_paths() -> list[str]:
    raw = yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {}
    paths = raw.get("curated_test_paths") or []
    assert isinstance(paths, list) and paths, (
        f"{_CONFIG} declares no curated_test_paths; the guard job would have "
        "nothing to run and its zero-collection check is the only thing that "
        "would notice"
    )
    return [str(entry) for entry in paths]


def _curated_step_script() -> str:
    """The `run` body of the ci.yml step that runs the curated proofs.

    Read from the parsed workflow by step id rather than by grepping the file,
    so a moved step or a renamed one is a resolvable failure instead of a
    silently empty match.
    """
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8")) or {}
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if step.get("id") == _CURATED_PROOF_STEP_ID:
                assert job_name == "integration-guard", (
                    f"step {_CURATED_PROOF_STEP_ID!r} moved to job {job_name!r}; "
                    "the curated Postgres proofs must run on the job that "
                    "provisions Postgres"
                )
                return str(step.get("run") or "")
    raise AssertionError(
        f"no step with id {_CURATED_PROOF_STEP_ID!r} in {_CI_WORKFLOW}; the "
        "curated Postgres proofs are not being run by any job"
    )


@pytest.mark.unit
def test_every_curated_path_is_run_by_the_guard_job() -> None:
    script = _curated_step_script()
    missing = [path for path in _declared_curated_paths() if path not in script]
    assert not missing, (
        "these paths are declared curated_test_paths but the integration-guard "
        f"step does not run them, so nothing executes them: {missing}"
    )


@pytest.mark.unit
def test_every_path_the_guard_job_runs_is_declared_curated() -> None:
    declared = _declared_curated_paths()
    script = _curated_step_script()
    selected = [token for token in script.split() if token.startswith("tests/")]
    assert selected, (
        "the integration-guard step selects no test path at all; its "
        "zero-collection check would be the only thing to notice"
    )
    undeclared = [token for token in selected if token not in declared]
    assert not undeclared, (
        "the integration-guard step runs these paths but curated_test_paths "
        f"does not declare them, so the config understates what the gate "
        f"covers: {undeclared}"
    )
