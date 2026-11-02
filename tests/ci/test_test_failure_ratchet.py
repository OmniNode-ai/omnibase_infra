# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/check_test_failure_ratchet.py and
scripts/publish_test_failure_baseline.py (OMN-13027).

Tests validate baseline parsing, expiry logic, new-cluster detection, and
the ratchet gate exit codes — all without subprocess or network calls.
"""

from __future__ import annotations

import datetime
import importlib.util
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Load scripts under test via importlib (not on sys.path by default)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> object:
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ratchet = _load("check_test_failure_ratchet")
publisher = _load("publish_test_failure_baseline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_junit(path: Path, failures: list[tuple[str, str]]) -> None:
    """Write a minimal JUnit XML with the given (classname, testname) failures."""
    root = ET.Element("testsuite", name="pytest", tests=str(len(failures)))
    for classname, name in failures:
        tc = ET.SubElement(root, "testcase", classname=classname, name=name)
        ET.SubElement(tc, "failure", message="assertion error")
    tree = ET.ElementTree(root)
    tree.write(str(path))


def _write_baseline(path: Path, clusters: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": "2026-06-17T00:00:00+00:00",
        "generated_by": "dev-baseline-publisher",
        "repo": "OmniNode-ai/omnibase_infra",
        "ttl_days": 7,
        "clusters": clusters,
    }
    with path.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=True)


def _future_expiry() -> str:
    return (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=6)
    ).isoformat()


def _past_expiry() -> str:
    return (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
    ).isoformat()


# ---------------------------------------------------------------------------
# parse_junit (shared between both scripts)
# ---------------------------------------------------------------------------


class TestParseJunit:
    def test_empty_xml_returns_empty(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        root = ET.Element("testsuite", tests="0")
        ET.ElementTree(root).write(str(junit))
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert result == {}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = ratchet.parse_junit(tmp_path / "missing.xml")  # type: ignore[attr-defined]
        assert result == {}

    def test_single_failure_parsed(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert "tests.unit" in result
        assert "tests.unit.test_foo::test_bar" in result["tests.unit"]

    def test_multiple_failures_same_cluster(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        _write_junit(
            junit,
            [
                ("tests.unit.test_foo", "test_a"),
                ("tests.unit.test_foo", "test_b"),
            ],
        )
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert len(result["tests.unit"]) == 2

    def test_failures_in_distinct_clusters(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        _write_junit(
            junit,
            [
                ("tests.unit.test_foo", "test_a"),
                ("tests.nodes.test_bar", "test_b"),
            ],
        )
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert "tests.unit" in result
        assert "tests.nodes" in result

    def test_passing_tests_excluded(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        root = ET.Element("testsuite", tests="2")
        # passing test — no failure child
        ET.SubElement(
            root, "testcase", classname="tests.unit.test_foo", name="test_pass"
        )
        # failing test
        tc = ET.SubElement(
            root, "testcase", classname="tests.unit.test_foo", name="test_fail"
        )
        ET.SubElement(tc, "failure", message="boom")
        ET.ElementTree(root).write(str(junit))
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert len(result.get("tests.unit", [])) == 1
        assert "tests.unit.test_foo::test_fail" in result["tests.unit"]

    def test_error_child_also_counts_as_failure(self, tmp_path: Path) -> None:
        junit = tmp_path / "junit.xml"
        root = ET.Element("testsuite", tests="1")
        tc = ET.SubElement(
            root, "testcase", classname="tests.unit.test_foo", name="test_err"
        )
        ET.SubElement(tc, "error", message="import error")
        ET.ElementTree(root).write(str(junit))
        result = ratchet.parse_junit(junit)  # type: ignore[attr-defined]
        assert "tests.unit" in result


# ---------------------------------------------------------------------------
# Ratchet gate — check_test_failure_ratchet.main()
# ---------------------------------------------------------------------------


class TestRatchetGate:
    def test_no_failures_passes(self, tmp_path: Path) -> None:
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(baseline, {})
        junit = tmp_path / "junit.xml"
        root = ET.Element("testsuite", tests="0")
        ET.ElementTree(root).write(str(junit))
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline)]
        )
        assert rc == 0

    def test_failure_in_baseline_passes(self, tmp_path: Path) -> None:
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(
            baseline,
            {
                "tests.unit": {
                    "ticket": "OMN-99999",
                    "expires_at": _future_expiry(),
                    "count": 1,
                    "examples": ["tests.unit.test_foo::test_bar"],
                    "first_seen": "2026-06-17T00:00:00+00:00",
                }
            },
        )
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline)]
        )
        assert rc == 0

    def test_new_failure_not_in_baseline_fails(self, tmp_path: Path) -> None:
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(baseline, {})  # empty baseline
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_new")])
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline)]
        )
        assert rc == 1

    def test_expired_entry_fails_without_allow_expired(self, tmp_path: Path) -> None:
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(
            baseline,
            {
                "tests.unit": {
                    "ticket": "OMN-99999",
                    "expires_at": _past_expiry(),  # expired
                    "count": 1,
                    "examples": [],
                    "first_seen": "2026-06-01T00:00:00+00:00",
                }
            },
        )
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline)]
        )
        assert rc == 1

    def test_expired_entry_passes_with_allow_expired(self, tmp_path: Path) -> None:
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(
            baseline,
            {
                "tests.unit": {
                    "ticket": "OMN-99999",
                    "expires_at": _past_expiry(),
                    "count": 1,
                    "examples": [],
                    "first_seen": "2026-06-01T00:00:00+00:00",
                }
            },
        )
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline), "--allow-expired"]
        )
        assert rc == 0

    def test_missing_baseline_treated_as_empty_strict(self, tmp_path: Path) -> None:
        """Missing baseline = empty baseline (strictest): a failure with no
        baseline is a new cluster -> gate blocks (rc 1); zero failures -> rc 0
        (the bootstrap PR that introduces the gate before dev has a baseline)."""
        junit = tmp_path / "junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        rc = ratchet.main(  # type: ignore[attr-defined]
            [
                "--junit-xml",
                str(junit),
                "--baseline",
                str(tmp_path / "nonexistent.yaml"),
            ]
        )
        assert rc == 1
        empty_junit = tmp_path / "empty.xml"
        _write_junit(empty_junit, [])
        rc2 = ratchet.main(  # type: ignore[attr-defined]
            [
                "--junit-xml",
                str(empty_junit),
                "--baseline",
                str(tmp_path / "nonexistent.yaml"),
            ]
        )
        assert rc2 == 0

    def test_partial_overlap_new_cluster_fails(self, tmp_path: Path) -> None:
        """Some failures in baseline, one new cluster — gate must fail."""
        baseline = tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        _write_baseline(
            baseline,
            {
                "tests.unit": {
                    "ticket": "OMN-99999",
                    "expires_at": _future_expiry(),
                    "count": 1,
                    "examples": [],
                    "first_seen": "2026-06-17T00:00:00+00:00",
                }
            },
        )
        junit = tmp_path / "junit.xml"
        _write_junit(
            junit,
            [
                ("tests.unit.test_foo", "test_old"),  # in baseline
                ("tests.nodes.test_new", "test_brand_new"),  # NOT in baseline
            ],
        )
        rc = ratchet.main(  # type: ignore[attr-defined]
            ["--junit-xml", str(junit), "--baseline", str(baseline)]
        )
        assert rc == 1


# ---------------------------------------------------------------------------
# Baseline publisher helpers
# ---------------------------------------------------------------------------


class TestBaselinePublisher:
    def test_prune_expired_removes_old_entries(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        past = (now - datetime.timedelta(days=1)).isoformat()
        future = (now + datetime.timedelta(days=6)).isoformat()
        clusters = {
            "old_cluster": {"expires_at": past, "ticket": "OMN-1"},
            "live_cluster": {"expires_at": future, "ticket": "OMN-2"},
        }
        result = publisher.prune_expired(clusters, now)  # type: ignore[attr-defined]
        assert "live_cluster" in result
        assert "old_cluster" not in result

    def test_prune_expired_keeps_all_valid(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        future = (now + datetime.timedelta(days=6)).isoformat()
        clusters = {
            "a": {"expires_at": future, "ticket": "OMN-1"},
            "b": {"expires_at": future, "ticket": "OMN-2"},
        }
        result = publisher.prune_expired(clusters, now)  # type: ignore[attr-defined]
        assert set(result.keys()) == {"a", "b"}

    def test_load_baseline_missing_returns_empty(self, tmp_path: Path) -> None:
        result = publisher.load_baseline(tmp_path / "missing.yaml")  # type: ignore[attr-defined]
        assert result == {"generated_at": "", "clusters": {}}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.yaml"
        data = {
            "generated_at": "2026-06-17T00:00:00",
            "clusters": {"tests.unit": {"ticket": "OMN-1", "count": 3}},
        }
        publisher.save_baseline(path, data)  # type: ignore[attr-defined]
        loaded = publisher.load_baseline(path)  # type: ignore[attr-defined]
        assert loaded["clusters"]["tests.unit"]["ticket"] == "OMN-1"

    def test_main_dry_run_no_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run must not write any files."""
        monkeypatch.chdir(tmp_path)
        # Create empty JUnit (no failures)
        junit = tmp_path / "dev-baseline-junit.xml"
        root = ET.Element("testsuite", tests="0")
        ET.ElementTree(root).write(str(junit))
        # Create baseline dir
        baseline_dir = tmp_path / "config" / "validation"
        baseline_dir.mkdir(parents=True)

        rc = publisher.main(  # type: ignore[attr-defined]
            ["--skip-run", "--dry-run", "--repo", "OmniNode-ai/omnibase_infra"]
        )
        assert rc == 0
        # No baseline written
        assert not (baseline_dir / "test-failure-baseline.yaml").exists()

    def test_main_writes_baseline_when_no_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        junit = tmp_path / "dev-baseline-junit.xml"
        root = ET.Element("testsuite", tests="0")
        ET.ElementTree(root).write(str(junit))
        (tmp_path / "config" / "validation").mkdir(parents=True)

        rc = publisher.main(  # type: ignore[attr-defined]
            ["--skip-run", "--repo", "OmniNode-ai/omnibase_infra"]
        )
        assert rc == 0
        baseline_path = (
            tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        )
        assert baseline_path.exists()
        with baseline_path.open() as fh:
            data = yaml.safe_load(fh)
        assert data["clusters"] == {}

    def test_main_adds_new_cluster_without_ticket_when_no_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        junit = tmp_path / "dev-baseline-junit.xml"
        _write_junit(junit, [("tests.unit.test_foo", "test_bar")])
        (tmp_path / "config" / "validation").mkdir(parents=True)

        rc = publisher.main(  # type: ignore[attr-defined]
            [
                "--skip-run",
                "--repo",
                "OmniNode-ai/omnibase_infra",
                "--linear-api-key",
                "",
            ]
        )
        assert rc == 0
        baseline_path = (
            tmp_path / "config" / "validation" / "test-failure-baseline.yaml"
        )
        with baseline_path.open() as fh:
            data = yaml.safe_load(fh)
        assert "tests.unit" in data["clusters"]
        # No ticket filed (no API key) — ticket field is empty string
        assert data["clusters"]["tests.unit"]["ticket"] == ""

    def test_main_removes_resolved_clusters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clusters that now pass should be removed from the baseline."""
        monkeypatch.chdir(tmp_path)
        baseline_dir = tmp_path / "config" / "validation"
        baseline_dir.mkdir(parents=True)
        baseline_path = baseline_dir / "test-failure-baseline.yaml"

        # Existing baseline with a cluster that will now pass
        _write_baseline(
            baseline_path,
            {
                "tests.unit": {
                    "ticket": "OMN-99999",
                    "expires_at": _future_expiry(),
                    "count": 1,
                    "examples": [],
                    "first_seen": "2026-06-17T00:00:00+00:00",
                }
            },
        )

        # JUnit with NO failures (tests now pass)
        junit = tmp_path / "dev-baseline-junit.xml"
        root = ET.Element("testsuite", tests="0")
        ET.ElementTree(root).write(str(junit))

        rc = publisher.main(  # type: ignore[attr-defined]
            ["--skip-run", "--repo", "OmniNode-ai/omnibase_infra"]
        )
        assert rc == 0
        with baseline_path.open() as fh:
            data = yaml.safe_load(fh)
        assert "tests.unit" not in data["clusters"]


# ---------------------------------------------------------------------------
# The publisher workflow itself (OMN-22128)
#
# The ratchet gate's own refusal text tells a blocked PR to "run
# dev-baseline-publisher on dev to add them to the baseline". That remedy was
# inoperable for 4.5 months: all 138 scheduled runs from 2026-06-18 to
# 2026-11-02 ended non-success, in two eras with two causes, and
# config/validation/test-failure-baseline.yaml on dev still carried only its
# own creation commit. Both causes are asserted here so neither can come back
# silently, since a `cancelled` conclusion is not a failure and alarms nothing.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLISHER_WORKFLOW = (
    _REPO_ROOT / ".github" / "workflows" / "dev-baseline-publisher.yml"
)
_BASELINE_FILE = (
    _REPO_ROOT / "config" / "validation" / "test-failure-baseline.yaml"
)

#: Test time measured for the publisher's own marker selection on clean dev,
#: summed across the 15 parallel splits of run 41446149536 (head f30b9f461).
#: The publisher runs that selection serially in one process, so this is its
#: floor, not its target.
_MEASURED_SERIAL_SUITE_MINUTES = 86


def _publisher_job() -> dict[str, object]:
    with _PUBLISHER_WORKFLOW.open() as fh:
        workflow = yaml.safe_load(fh)
    job = workflow["jobs"]["publish-baseline"]
    assert isinstance(job, dict)
    return job


class TestPublisherWorkflow:
    """The publisher must be able to finish, and to land what it finished."""

    def test_publisher_timeout_clears_the_measured_suite_floor(self) -> None:
        """Era two (2026-09-07..2026-11-02, 70 runs): cancelled mid-pytest.

        Run 41380955296 ran 03:03:29Z-03:33:47Z and its post-job cleanup
        terminated orphan uv, python3, pytest and psql processes, so the job
        died inside the test run and never reached the push at all. The job
        timeout has to clear the serial suite with headroom.
        """
        timeout = _publisher_job()["timeout-minutes"]
        assert isinstance(timeout, int), (
            "publish-baseline must declare an integer timeout-minutes; "
            f"got {timeout!r}"
        )
        assert timeout >= _MEASURED_SERIAL_SUITE_MINUTES, (
            f"publish-baseline allows {timeout} minutes but the suite it runs "
            f"serially measured {_MEASURED_SERIAL_SUITE_MINUTES} minutes of "
            "test time on clean dev. A timeout at or under the suite kills the "
            "job mid-pytest and reports `cancelled`, which no alarm reads."
        )

    def test_publisher_push_rebases_and_retries_onto_current_dev(self) -> None:
        """Era one (2026-06-18..2026-09-06, 68 runs): push rejected.

        Run 34008081727 finished the suite at 29m09s, wrote and committed the
        baseline as 8cca07378 (1 file changed, 19 insertions, 25 deletions),
        then died on "! [rejected] dev -> dev (fetch first)". A job that spends
        an hour on tests always races a dev merge, so a single bare push loses
        the baseline it just computed.
        """
        steps = _publisher_job()["steps"]
        assert isinstance(steps, list)
        push_steps = [
            step
            for step in steps
            if isinstance(step, dict) and "git push" in str(step.get("run", ""))
        ]
        assert push_steps, "no step in publish-baseline pushes the baseline"

        for step in push_steps:
            run = str(step["run"])
            assert "git rebase origin/dev" in run, (
                f"step {step.get('name')!r} pushes the baseline without "
                "rebasing onto the current origin/dev, so a merge landing "
                "during the suite rejects it non-fast-forward and the "
                "baseline is lost (run 34008081727)"
            )
            assert "git fetch origin dev" in run, (
                f"step {step.get('name')!r} rebases without fetching first, "
                "so it rebases onto a stale origin/dev"
            )
            # The retry must actually loop, not just rebase once and give up.
            assert "for attempt in" in run, (
                f"step {step.get('name')!r} has no retry loop around the "
                "push, so a second concurrent merge still loses the baseline"
            )

    def test_committed_baseline_loads_through_the_publishers_own_loader(
        self,
    ) -> None:
        """AC4's repo-side half: whatever is committed must stay parseable.

        The publisher writes this file and the PR gate reads it, so a baseline
        that the publisher's own loader cannot read would strand every open PR
        with no way to baseline a dev-side failure.
        """
        assert _BASELINE_FILE.exists(), f"{_BASELINE_FILE} is missing"
        baseline = publisher.load_baseline(_BASELINE_FILE)  # type: ignore[attr-defined]
        assert isinstance(baseline, dict)
        for field in ("generated_at", "generated_by", "repo", "ttl_days"):
            assert field in baseline, (
                f"committed baseline has no {field!r}; the publisher writes "
                "all four and the gate's summary step reads them"
            )
        assert isinstance(baseline.get("clusters", {}), dict), (
            "clusters must be a mapping of cluster key to entry"
        )
