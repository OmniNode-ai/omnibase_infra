# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 -- the overlay applier's records and shared test doubles.

The cluster apply these tests used to pin is deleted (OMN-23521). What remains
here is the record reader contract the agent's HTTP surface serves, and the
``FakeRunner`` and ``_applier`` doubles the other overlay tests import.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.lab_overlay import (
    OMNIMARKET_VERSION_PROGRAM,
    LabOverlayApplier,
    ModelLabOverlayCapture,
    ModelLabOverlayCheck,
    load_latest_record,
    load_record,
    record_path,
)

pytestmark = pytest.mark.unit

SHA = "b" * 40
STAMP = "20260912T010203Z"
MANIFEST_SHA = "c" * 40


class FakeRunner:
    """Records every argv and answers from a caller-supplied script.

    The applier funnels every external command through one callable precisely so
    the failure modes that matter -- a garbage-collected pin, an apply that exits
    non-zero, a lane whose pods keep the previous image -- are reachable without a
    lab host, a Docker daemon or a cluster.
    """

    def __init__(self, answers: dict[str, tuple[int, str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        argv = list(argv)
        self.calls.append(argv)
        joined = " ".join(argv)
        for needle, (code, out) in self.answers.items():
            if needle in joined:
                return subprocess.CompletedProcess(argv, code, out, "")
        return subprocess.CompletedProcess(argv, 0, self._default(joined), "")

    @staticmethod
    def _default(joined: str) -> str:
        if "rev-parse origin/dev" in joined:
            return MANIFEST_SHA
        if "importlib.metadata" in joined:
            return "0.4.61"
        return ""

    def argv_containing(self, needle: str) -> list[list[str]]:
        return [call for call in self.calls if needle in " ".join(call)]


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in overlay source clone."""
    source = tmp_path / "overlay-source"
    (source / "docker").mkdir(parents=True)
    return source


def _applier(
    tmp_path: Path,
    overlay_source: Path,
    runner: FakeRunner,
    *,
    env: dict[str, str] | None = None,
) -> LabOverlayApplier:
    applier = LabOverlayApplier(
        state_dir=tmp_path / "state",
        repo_dir=tmp_path / "omnibase_infra",
        overlay_source_dir=overlay_source,
        env=dict(env or {}),
        runner=runner,
    )

    # `archive_overlay` shells out to git and tar, neither of which exists as a
    # real repository here. Stub the SOURCE materialisation only: everything the
    # tests are about happens after it.
    def _archive(destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "docker").mkdir(exist_ok=True)
        return MANIFEST_SHA

    applier.archive_overlay = _archive  # type: ignore[method-assign]
    return applier


def test_load_record_distinguishes_absent_from_malformed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert load_record(state, SHA) is None

    path = record_path(state, SHA)
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_record(state, SHA)


def test_load_latest_record_returns_none_when_nothing_was_ever_written(
    tmp_path: Path,
) -> None:
    # OMN-18399. Directory doesn't exist yet -- no apply has ever run.
    assert load_latest_record(tmp_path / "state") is None


def test_load_latest_record_picks_the_newest_finished_at_not_filename_order(
    tmp_path: Path,
) -> None:
    """OMN-18399. Sorted by ``finished_at``, not by sha or mtime -- a record
    written earlier in wall-clock time for a numerically-larger sha must not
    read as latest."""
    state = tmp_path / "state"
    directory = state / "lab-overlay"
    directory.mkdir(parents=True)
    older_sha = "a" * 40
    newer_sha = "0" * 40  # sorts BEFORE older_sha as a filename
    (directory / f"{older_sha}.json").write_text(
        json.dumps(
            {"sha": older_sha, "finished_at": "2026-09-15T08:00:00Z", "checks": []}
        )
    )
    (directory / f"{newer_sha}.json").write_text(
        json.dumps(
            {"sha": newer_sha, "finished_at": "2026-09-15T09:00:00Z", "checks": []}
        )
    )
    latest = load_latest_record(state)
    assert latest is not None
    assert latest["sha"] == newer_sha


def test_load_latest_record_raises_on_a_malformed_record(tmp_path: Path) -> None:
    """OMN-18399. Same contract as ``load_record``: unreadable is a finding, not
    an absence, and must not be papered over as "just skip the bad one" -- that
    could silently report a stale record as the latest."""
    state = tmp_path / "state"
    directory = state / "lab-overlay"
    directory.mkdir(parents=True)
    (directory / f"{SHA}.json").write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_latest_record(state)


def test_load_latest_record_does_not_descend_into_the_work_subdirectory(
    tmp_path: Path,
) -> None:
    """OMN-18399. ``lab-overlay/work/`` holds the applier's own scratch files
    during an in-flight apply; it must never be mistaken for a record."""
    state = tmp_path / "state"
    directory = state / "lab-overlay"
    work = directory / "work"
    work.mkdir(parents=True)
    (work / "stray.json").write_text("not a record")
    assert load_latest_record(state) is None


def test_a_check_without_evidence_is_refused() -> None:
    """An ok:true with no evidence is indistinguishable from a check that was
    never run, which is the shape the receipt contract exists to refuse."""
    with pytest.raises(ValueError, match="evidence is required"):
        ModelLabOverlayCheck(name="x", ok=True, evidence="")


def test_the_version_probe_reads_the_installed_distribution() -> None:
    """Measured 2026-09-12 on BOTH lanes: ``omnimarket.__version__`` is the
    literal 0.1.0 in every image, while ``importlib.metadata`` reports 0.4.61 on
    the compose lane and 0.4.30 on the lab lane -- the two numbers this ticket is
    about. A comparison built on ``__version__`` would have compared 0.1.0 against
    0.1.0 and PASSED on the exact discrepancy it exists to catch."""
    assert "importlib.metadata" in OMNIMARKET_VERSION_PROGRAM
    assert "__version__" not in OMNIMARKET_VERSION_PROGRAM


# ---------------------------------------------------------------------------
# OMN-18354 -- the api build must NAME the runtime image it derives from.
#
# These pin the condition that took the lab lane out of service on 2026-09-14.
# omninode_infra#1443 made stage 0 of ``docker/onex-api/Dockerfile`` a
# ``FROM ${RUNTIME_IMAGE}`` with NO default -- deliberately, so a build cannot
# silently certify a package closure nobody deployed. This caller passed no
# build argument, so from that merge onward every api build exited 1 and the
# whole run failed at ``images_pinned``: deploy-agent records be80676c
# (02:56Z), 3f59b19c (03:21Z) and 8fd25217 (03:42Z), three consecutive runs,
# each with the runtime family rebuilt and the api image NOT rebuilt.
#
# That is exactly the state OMN-18354 AC4 names -- a runtime repin that leaves
# the previously built api image deployed against it -- reached through the
# build rather than through the pin. The lane sat at onex-api 0.4.75 against
# runtime 0.4.79 until a hand-run build supplied the argument.
# ---------------------------------------------------------------------------


def captured(sha: str, stamp: str) -> ModelLabOverlayCapture:
    """A capture that pinned the runtime image, for the agent-level doubles."""
    return ModelLabOverlayCapture(
        sha=sha,
        stamp=stamp,
        runtime_image=f"runtime:{stamp}-{sha[:8]}",
        infra_migrate_image=f"infra-migrate:{stamp}-{sha[:8]}",
        pin_error=None,
        compose_omnimarket_version=None,
        compose_version_error=None,
    )


def build_passes(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """The applier's ``_run`` for a double: every command exits 0."""
    return subprocess.CompletedProcess(list(argv), 0, "", "")
