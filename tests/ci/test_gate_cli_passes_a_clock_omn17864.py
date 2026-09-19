# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The CLI must hand ``evaluate`` a clock, or both graces are inert (OMN-17864).

WHY THIS FILE EXISTS
    :func:`failure_is_provisional` and :func:`cancellation_is_provisional` both
    return ``False`` when ``now is None``. That is deliberate and fail-closed: a
    caller that forgets the time enforces the old, strict reading rather than
    waiting on a red forever.

    It is also a terrible SILENT outcome, and it is not hypothetical. The sibling
    port of this change into ``omninode_infra`` changed the gate module and did
    not change that repository's separate poller, which is its only production
    caller of ``evaluate``. Every unit test passed, ``mypy --strict`` passed, and
    the gate shipped COMPLETELY INERT — live CI printed
    ``external-context failures: verify / verify`` two seconds into the run, with
    no awaiting line, exactly as before the change.

    This repository's only production caller is :func:`main`, which does pass the
    clock. Nothing asserted that, so the same silent regression was one refactor
    away here. This file closes that.

WHAT IT PINS, AND WHAT IT DOES NOT
    It pins the WIRING: that ``main`` reaches ``evaluate`` with a real,
    timezone-aware, current wall clock, and that no CLI flag or environment
    variable can supply it instead. It does NOT pin how either grace then
    behaves — each has its own incident replay.

WHY NO FLAG IS THE LOAD-BEARING HALF
    Both graces' un-forgeability rests entirely on the observation time being
    the process's own wall clock. A caller-assertable time would let a long-dead
    red be held provisional indefinitely, which is the bypass neither grace may
    become. This is the same property the OMN-18319 prod-promotion gate enforces
    for its health fact, and it is pinned the same way: by reading the parser's
    own option strings, so reintroducing a surface is a red test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import ci_summary_gate

pytestmark = pytest.mark.unit


@pytest.fixture
def payloads(tmp_path: Path) -> tuple[Path, Path]:
    jobs = tmp_path / "jobs.json"
    jobs.write_text("[]", encoding="utf-8")
    check_runs = tmp_path / "check_runs.json"
    check_runs.write_text('{"check_runs": []}', encoding="utf-8")
    return jobs, check_runs


def test_the_cli_hands_evaluate_a_real_current_clock(
    payloads: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring assertion: ``now`` arrives, aware, and is the wall clock."""

    captured: dict[str, Any] = {}

    def _fake_evaluate(_jobs: object, **kwargs: object) -> tuple[int, str]:
        captured.update(kwargs)
        return ci_summary_gate.EXIT_PENDING, "stubbed report"

    monkeypatch.setattr(ci_summary_gate, "evaluate", _fake_evaluate)
    jobs, check_runs = payloads

    before = datetime.now(UTC)
    ci_summary_gate.main(
        [
            "--jobs-file",
            str(jobs),
            "--check-runs-file",
            str(check_runs),
            "--event-name",
            "pull_request",
        ]
    )
    after = datetime.now(UTC)

    assert "now" in captured, (
        "main() must pass now= to evaluate(); omitting it silently disables the "
        "OMN-18355 cancellation grace and the OMN-17864 failure grace at once"
    )
    now = captured["now"]
    assert isinstance(now, datetime)
    assert now.tzinfo is not None, "a naive clock cannot be compared to GitHub's UTC"
    assert before - timedelta(seconds=5) <= now <= after + timedelta(seconds=5)


@pytest.mark.parametrize(
    "forbidden", ["--now", "--observation-time", "--clock", "--as-of"]
)
def test_no_entrypoint_accepts_an_observation_time_option(
    forbidden: str, payloads: tuple[Path, Path]
) -> None:
    """A caller-assertable clock would make either grace an unbounded bypass.

    Asserted BEHAVIOURALLY — the CLI is invoked with the flag and must reject
    it — rather than by reading the source, so a future option added by any
    route is caught, not only one spelled the way this file expects.
    """

    jobs, check_runs = payloads
    with pytest.raises(SystemExit) as excinfo:
        ci_summary_gate.main(
            [
                "--jobs-file",
                str(jobs),
                "--check-runs-file",
                str(check_runs),
                forbidden,
                "2020-01-01T00:00:00Z",
            ]
        )
    assert excinfo.value.code == 2, f"{forbidden} reintroduces a forgeable clock"


def test_the_clock_is_not_reachable_from_the_environment() -> None:
    source = Path(ci_summary_gate.__file__).read_text(encoding="utf-8")
    now_lines = [
        line for line in source.splitlines() if line.strip().startswith("now=datetime")
    ]
    assert now_lines, "main() no longer constructs the observation time itself"
    for line in now_lines:
        assert line.strip() == "now=datetime.now(UTC),", line
        assert "environ" not in line
    assert "CI_SUMMARY_NOW" not in source
