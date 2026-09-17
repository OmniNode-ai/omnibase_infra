# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18545 -- the repair build is a build, and nothing else.

``build_repair_migrate_image`` is the failing deploy job's entry into the lab
overlay module. It exists because this module is the only thing in the repository
that builds a replacement CLOUD-migrate image from the overlay tree -- the image
the compose dev lane's migration preflight runs, and therefore the image a failed
preflight needs rebuilt.

WHY IT IS NOT JUST ``apply``. The apply PROMOTES the runtime pin
(``omnibase-infra-omninode-runtime:latest``) on the premise that "this agent just
built it for the compose lane from the merged sha". On a FAILED job that premise
can be false -- a job that dies before or during the image build leaves that tag
at the previous deploy's image -- and the apply's own readback checks cannot
catch it, because ``deployed_image`` compares the lane against the tag the same
run just minted and ``runtime_omnimarket_version`` compares it against the
compose lane, which on a failed deploy is also still the old image. A full apply
on the failing path would therefore roll the persistent k3s lane to a tag NAMING
the merged sha while it ran the previous commit's binary, and report PASS.

These tests assert the negative space, which is the whole point: no promotion, no
lane apply, no containerd import, no store binding, and a record that can never
read as a lab pass. A test that only checked "the build ran" would pass against
exactly the implementation this method exists to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deploy_agent.lab_overlay import (
    CLOUD_MIGRATE_IMAGE_NAME,
    LAB_LANE_VALUE,
    LabOverlayRefusalError,
)

from .test_lab_overlay_omn18200 import (
    MANIFEST_SHA,
    SHA,
    STAMP,
    FakeRunner,
    _applier,
)

pytestmark = pytest.mark.unit

EXPECTED_TAG = f"{CLOUD_MIGRATE_IMAGE_NAME}:{MANIFEST_SHA[:8]}-{STAMP}"


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in omninode_infra clone, the same shape the apply tests use.

    Declared here rather than imported from the neighbouring module: importing a
    fixture by name re-binds it in this namespace and every test signature then
    reads as a redefinition, which ruff refuses and which hides whether the
    fixture or the parameter is in play.
    """
    source = tmp_path / "omninode_infra"
    (source / "k8s" / "onex-lab").mkdir(parents=True)
    (source / "k8s" / "onex-lab" / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
    return source


def _record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checks(record: dict[str, object]) -> list[dict[str, object]]:
    checks = record["checks"]
    assert isinstance(checks, list)
    return checks


def test_it_builds_the_cloud_migrate_image_and_records_the_tag(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.build_repair_migrate_image(
        sha=SHA, stamp=STAMP, correlation_id="cid-repair"
    )

    builds = runner.argv_containing("docker build")
    assert len(builds) == 1, f"expected exactly one build, got {builds}"
    assert "-t" in builds[0]
    assert builds[0][builds[0].index("-t") + 1] == EXPECTED_TAG, (
        "the repair build must mint the same tag shape a successful apply would, "
        "so the operator pins one name whichever path produced the image"
    )

    record = _record(path)
    assert record["sha"] == SHA
    assert record["lane"] == LAB_LANE_VALUE
    assert record["agent_command_id"] == "cid-repair"
    built = next(
        check
        for check in _checks(record)
        if check["name"] == "cloud_migrate_repair_image_built"
    )
    assert built["ok"] is True
    assert EXPECTED_TAG in str(built["evidence"])


def test_it_never_reads_as_a_lab_pass(tmp_path: Path, overlay_source: Path) -> None:
    """The record shares a path and a shape with the apply's, so the only thing
    keeping a receipt from calling it PASS is a check that always fails.

    The receipt derives PASS iff every check passed. A repair record that passed
    everything would report a lane advanced to a sha it was never advanced to --
    a worse lie than writing no record at all.
    """
    applier = _applier(tmp_path, overlay_source, FakeRunner())

    record = _record(
        applier.build_repair_migrate_image(
            sha=SHA, stamp=STAMP, correlation_id="cid-repair"
        )
    )

    applied = next(
        check for check in _checks(record) if check["name"] == "lab_overlay_applied"
    )
    assert applied["ok"] is False
    assert "not attempted" in str(applied["evidence"])
    assert not all(check["ok"] for check in _checks(record))


def test_it_promotes_nothing_applies_nothing_and_imports_nothing(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The negative space, asserted by name.

    Each of these would be a real defect, not a stylistic difference: a promotion
    mislabels a stale runtime image with the merged sha, an apply rolls the
    persistent lane off a failed deploy, and a containerd import writes a pin
    into the lane's content store on a run that pins nothing.
    """
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    applier.build_repair_migrate_image(
        sha=SHA, stamp=STAMP, correlation_id="cid-repair"
    )

    assert runner.argv_containing("docker tag") == [], "the runtime pin was promoted"
    assert runner.argv_containing("apply_lab_lane.sh") == [], "the lane was applied"
    assert runner.argv_containing("images import") == [], (
        "the repair build imported into the k3s content store; the compose lane "
        "reads the host Docker daemon and this run pins nothing"
    )
    assert runner.argv_containing("kubectl") == []
    # `docker save` is piped, so it goes through the popen double rather than the
    # runner -- asserting it on the runner would be a zero that can never be
    # anything else.
    assert applier._popen.refs == []


def test_the_negative_assertions_above_are_not_vacuous(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The positive control for the test above (rule 16).

    Every call that test asserts ABSENT is asserted PRESENT here, from a full
    apply driven through the same doubles. Without this, a renamed argv or a
    mistyped needle would turn each of those assertions into a zero that can
    never be anything else -- which is exactly how the first draft of this file
    "proved" the repair build skipped a containerd import that the apply path
    also appeared to skip.
    """
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid-apply")

    assert len(runner.argv_containing("docker tag")) == 1
    assert len(runner.argv_containing("apply_lab_lane.sh")) == 1
    assert len(runner.argv_containing("images import")) == 4
    assert runner.argv_containing("kubectl") != []
    assert len(applier._popen.refs) == 4


def test_the_store_binding_is_never_written(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The binding carries the lab store identity to ``apply_lab_lane.sh``. This
    path does not run that script, so the file must never be created -- a 0600
    credential file written for no consumer is blast radius for nothing."""
    applier = _applier(tmp_path, overlay_source, FakeRunner())

    applier.build_repair_migrate_image(
        sha=SHA, stamp=STAMP, correlation_id="cid-repair"
    )

    binding = tmp_path / "state" / "lab-overlay" / "work" / "store-binding.env"
    assert not binding.exists()


def test_a_failing_build_is_a_failing_check_not_an_exception(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Same contract as ``apply``: the failure is data. A raise here would reach
    the agent's deploy ``except`` block, which has already recorded the job's real
    error, and would skip the terminal publish."""
    runner = FakeRunner({"docker build": (1, "")})
    applier = _applier(tmp_path, overlay_source, runner)

    record = _record(
        applier.build_repair_migrate_image(
            sha=SHA, stamp=STAMP, correlation_id="cid-repair"
        )
    )

    built = next(
        check
        for check in _checks(record)
        if check["name"] == "cloud_migrate_repair_image_built"
    )
    assert built["ok"] is False
    assert LabOverlayRefusalError.__name__ in str(built["evidence"])


def test_an_unresolvable_overlay_source_still_writes_a_record(
    tmp_path: Path, overlay_source: Path
) -> None:
    """ "The overlay source could not be read" and "nobody ran the repair" are
    different facts, and the record is the only place they are told apart."""
    applier = _applier(tmp_path, overlay_source, FakeRunner())

    def _refuse(destination: Path) -> str:
        msg = "git archive refused: origin/dev is unreachable"
        raise LabOverlayRefusalError(msg)

    applier.archive_overlay = _refuse  # type: ignore[method-assign]

    record = _record(
        applier.build_repair_migrate_image(
            sha=SHA, stamp=STAMP, correlation_id="cid-repair"
        )
    )

    names = [check["name"] for check in _checks(record)]
    assert names == ["overlay_source_resolved", "lab_overlay_applied"]
    assert not any(check["ok"] for check in _checks(record))
