# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The image-build ceiling carries a term for the machine it runs on (OMN-18615).

THE TWO KILLS THIS FILE REPRODUCES
----------------------------------

MEASURED on the ``.201`` dev lane, 2026-09-17. Both are the SAME rebuild, for
``omnimarket#2622`` squash ``387d5fe4cbfa5d04baf0872880d70fa83268e35e``:

    b046ce97-f2f8-43b0-9302-c4579a4b06de  16:19:40Z -> 16:37:41Z  1081s  load1 ~97
    499ab623-2fcc-4482-8afa-10846a06977d  17:23:32Z -> 17:41:35Z  1083s  load1 ~60

Both logged the byte-identical derivation:

    1080s = 60 work steps of 'docker/Dockerfile.runtime' x 15s/step (one shared
    BuildKit solve) + 9 buildable service(s) in profile 'runtime' x 20s/image
    (export), floor 600s

That identity across a 37-point spread in load average is the defect. Both of
the OMN-18072 terms -- the Dockerfile's work-step count and the profile's
buildable-service count -- are properties of the REPOSITORY. Neither is a
property of the machine the build is about to run on, so the number cannot
move when the machine does. The same tree built in about 6m44s at 11:43Z the
same morning under a 1240s ceiling with a warm cache.

WHY A WIDER FLAT NUMBER IS NOT THE FIX. That is what OMN-18072 already did to
OMN-18057's 300s, and it bought one morning. A constant re-opens this failure
one number later, which is the argument ``build_budget``'s own module docstring
makes about the constant it replaced.

THE DIRECTION OF THE ADJUSTMENT IS ASYMMETRIC, and deliberately so. Over-
estimating a ceiling only delays a kill; under-estimating one kills a healthy
build, which is the failure OMN-18072 existed to remove and the failure this
ticket is. So an UNREADABLE host condition widens the ceiling rather than
narrowing it -- and the widening is BOUNDED by a hard upper bound that no
combination of terms can exceed (AC5).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent.build_budget import (
    HARD_UPPER_BOUND_SECONDS,
    EnumBuildOutcome,
    ModelBuildProgress,
    derive_image_build_budget,
    parse_build_progress,
)
from deploy_agent.host_conditions import (
    EnumBuildCacheState,
    ModelHostConditions,
    probe_host_conditions,
)

# The derivation both 2026-09-17 kills logged, and the elapsed each ran before
# the kill. Named once so the assertions below read as "the recorded incident"
# rather than as bare literals.
RECORDED_BUILD_STEPS = 60
RECORDED_BUILDABLE_SERVICES = 9
RECORDED_PER_STEP_SECONDS = 15
RECORDED_PER_IMAGE_SECONDS = 20
RECORDED_FLOOR_SECONDS = 600
RECORDED_CEILING_SECONDS = 1080

# (correlation id, elapsed before the kill, .201 load1 at the time)
RECORDED_KILLS: tuple[tuple[str, int, float], ...] = (
    ("b046ce97-f2f8-43b0-9302-c4579a4b06de", 1081, 97.0),
    ("499ab623-2fcc-4482-8afa-10846a06977d", 1083, 60.0),
)

# The `.201` dev lane host. 32 cores, so load1 97 is a saturation ratio of ~3.
LAB_HOST_CPU_COUNT = 32

# The same tree, built warm on an idle host earlier the same morning.
WARM_IDLE_BUILD_SECONDS = 404


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _recorded_model(tmp_path: Path) -> tuple[str, ...]:
    """The build model that derives to exactly the recorded 1080s."""
    _write(
        tmp_path,
        "docker/Dockerfile.runtime",
        "FROM scratch\n"
        + "".join(f"RUN echo {i}\n" for i in range(RECORDED_BUILD_STEPS)),
    )
    body = ["services:"]
    for index in range(RECORDED_BUILDABLE_SERVICES):
        body.append(f"  svc{index}:")
        body.append("    profiles: [runtime, full]")
        body.append("    build:")
        body.append("      context: ..")
        body.append("      dockerfile: docker/Dockerfile.runtime")
    compose = _write(tmp_path, "docker/docker-compose.yml", "\n".join(body) + "\n")
    return (str(compose),)


def _derive(compose_files: tuple[str, ...], host: ModelHostConditions | None) -> object:
    return derive_image_build_budget(
        compose_files,
        "runtime",
        per_step_seconds=RECORDED_PER_STEP_SECONDS,
        per_image_seconds=RECORDED_PER_IMAGE_SECONDS,
        floor_seconds=RECORDED_FLOOR_SECONDS,
        host=host,
    )


def _idle_warm() -> ModelHostConditions:
    return probe_host_conditions(
        loadavg_reader=lambda: (1.2, 1.1, 1.0),
        cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
        builder_cache_reader=lambda: EnumBuildCacheState.WARM,
    )


def _contended_cold(load1: float) -> ModelHostConditions:
    return probe_host_conditions(
        loadavg_reader=lambda: (load1, load1, load1),
        cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
        builder_cache_reader=lambda: EnumBuildCacheState.COLD,
    )


@pytest.mark.unit
class TestAC3TheRecordedKillsAreFixtures:
    """AC3: both 2026-09-17 kills are reproduced before the fix is asserted."""

    def test_the_repository_only_model_returns_1080_for_both_kills(
        self, tmp_path: Path
    ) -> None:
        """The OMN-18072 derivation is blind to the host, by construction.

        This is the RED half made permanent: it pins the OLD formula's answer,
        so a future change that quietly drops the host terms shows up here as
        the incident's own number coming back.
        """
        compose_files = _recorded_model(tmp_path)
        unadjusted = _derive(compose_files, None)
        assert unadjusted.timeout_seconds == RECORDED_CEILING_SECONDS
        for correlation_id, elapsed, _load1 in RECORDED_KILLS:
            assert elapsed > unadjusted.timeout_seconds, (
                f"{correlation_id} ran {elapsed}s, which must exceed the "
                f"{unadjusted.timeout_seconds}s ceiling that killed it"
            )

    def test_the_two_kills_derived_the_identical_number_at_different_loads(
        self, tmp_path: Path
    ) -> None:
        """The falsifier the ticket names: identical ceilings fail AC1."""
        compose_files = _recorded_model(tmp_path)
        ceilings = {
            correlation_id: _derive(compose_files, None).timeout_seconds
            for correlation_id, _elapsed, _load1 in RECORDED_KILLS
        }
        assert len(set(ceilings.values())) == 1
        assert set(ceilings.values()) == {RECORDED_CEILING_SECONDS}


@pytest.mark.unit
class TestAC1TheCeilingMovesWithTheMachine:
    """AC1: the derivation carries at least one property of the host."""

    def test_idle_warm_and_contended_cold_derive_different_ceilings(
        self, tmp_path: Path
    ) -> None:
        """The ticket's falsifier, stated exactly: an identical number fails."""
        compose_files = _recorded_model(tmp_path)
        idle = _derive(compose_files, _idle_warm())
        contended = _derive(compose_files, _contended_cold(97.0))
        assert idle.timeout_seconds != contended.timeout_seconds
        assert contended.timeout_seconds > idle.timeout_seconds

    def test_both_recorded_kills_would_have_been_granted_their_elapsed(
        self, tmp_path: Path
    ) -> None:
        """A fix that still kills the builds it was written for is not a fix."""
        compose_files = _recorded_model(tmp_path)
        for correlation_id, elapsed, load1 in RECORDED_KILLS:
            budget = _derive(compose_files, _contended_cold(load1))
            assert budget.timeout_seconds > elapsed, (
                f"{correlation_id} ran {elapsed}s under load1 {load1}; the "
                f"corrected ceiling is {budget.timeout_seconds}s and would "
                "have killed it again"
            )

    def test_load_alone_moves_the_ceiling(self, tmp_path: Path) -> None:
        """Contention is a term in its own right, not only a cache proxy."""
        compose_files = _recorded_model(tmp_path)
        quiet = probe_host_conditions(
            loadavg_reader=lambda: (2.0, 2.0, 2.0),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        )
        busy = probe_host_conditions(
            loadavg_reader=lambda: (97.0, 97.0, 97.0),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        )
        assert (
            _derive(compose_files, busy).timeout_seconds
            > _derive(compose_files, quiet).timeout_seconds
        )

    def test_cache_state_alone_moves_the_ceiling(self, tmp_path: Path) -> None:
        """A pruned BuildKit cache is the likelier dominant term (ticket §why)."""
        compose_files = _recorded_model(tmp_path)
        warm = probe_host_conditions(
            loadavg_reader=lambda: (2.0, 2.0, 2.0),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        )
        cold = probe_host_conditions(
            loadavg_reader=lambda: (2.0, 2.0, 2.0),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: EnumBuildCacheState.COLD,
        )
        assert (
            _derive(compose_files, cold).timeout_seconds
            > _derive(compose_files, warm).timeout_seconds
        )

    def test_the_idle_warm_ceiling_still_clears_the_measured_warm_build(
        self, tmp_path: Path
    ) -> None:
        """Adapting must not tighten the idle case below what it already ran."""
        compose_files = _recorded_model(tmp_path)
        idle = _derive(compose_files, _idle_warm())
        assert idle.timeout_seconds >= RECORDED_CEILING_SECONDS
        assert idle.timeout_seconds > WARM_IDLE_BUILD_SECONDS


@pytest.mark.unit
class TestAC1UnreadableHostConditionsWiden:
    """An unreadable machine property must never TIGHTEN the ceiling."""

    def test_unreadable_loadavg_is_unknown_and_widens(self, tmp_path: Path) -> None:
        compose_files = _recorded_model(tmp_path)

        def _raises() -> tuple[float, float, float]:
            raise OSError("no loadavg on this platform")

        unknown = probe_host_conditions(
            loadavg_reader=_raises,
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        )
        assert unknown.load1 is None
        assert (
            _derive(compose_files, unknown).timeout_seconds
            > _derive(compose_files, _idle_warm()).timeout_seconds
        )

    def test_unreadable_cache_state_is_unknown_and_widens(self, tmp_path: Path) -> None:
        compose_files = _recorded_model(tmp_path)

        def _raises() -> EnumBuildCacheState:
            raise OSError("docker not reachable")

        unknown = probe_host_conditions(
            loadavg_reader=lambda: (1.2, 1.1, 1.0),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=_raises,
        )
        assert unknown.cache_state is EnumBuildCacheState.UNKNOWN
        assert (
            _derive(compose_files, unknown).timeout_seconds
            > _derive(compose_files, _idle_warm()).timeout_seconds
        )

    def test_a_zero_or_absent_cpu_count_does_not_divide_by_zero(
        self, tmp_path: Path
    ) -> None:
        compose_files = _recorded_model(tmp_path)
        conditions = probe_host_conditions(
            loadavg_reader=lambda: (10.0, 10.0, 10.0),
            cpu_count_reader=lambda: 0,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        )
        assert conditions.saturation is None
        assert _derive(compose_files, conditions).timeout_seconds > 0

    def test_the_probe_never_raises(self) -> None:
        """A ceiling that cannot be derived at all would block every deploy."""

        def _raises() -> object:
            raise RuntimeError("boom")

        conditions = probe_host_conditions(
            loadavg_reader=_raises,  # type: ignore[arg-type]
            cpu_count_reader=_raises,  # type: ignore[arg-type]
            builder_cache_reader=_raises,  # type: ignore[arg-type]
        )
        assert conditions.cache_state is EnumBuildCacheState.UNKNOWN


@pytest.mark.unit
class TestAC5BoundedAndFailClosedOnMutation:
    """AC5 (labelled): adapting must stay BOUNDED."""

    def test_no_combination_of_terms_exceeds_the_hard_upper_bound(
        self, tmp_path: Path
    ) -> None:
        compose_files = _recorded_model(tmp_path)
        absurd = probe_host_conditions(
            loadavg_reader=lambda: (1_000_000.0, 1.0, 1.0),
            cpu_count_reader=lambda: 1,
            builder_cache_reader=lambda: EnumBuildCacheState.COLD,
        )
        budget = _derive(compose_files, absurd)
        assert budget.timeout_seconds <= HARD_UPPER_BOUND_SECONDS

    def test_an_enormous_repository_model_is_also_bounded(self, tmp_path: Path) -> None:
        """The bound covers the model terms too, not only the host multiplier."""
        _write(
            tmp_path,
            "docker/Dockerfile.runtime",
            "FROM scratch\n" + "".join(f"RUN echo {i}\n" for i in range(100_000)),
        )
        compose = _write(
            tmp_path,
            "docker/docker-compose.yml",
            "services:\n"
            "  svc:\n"
            "    profiles: [runtime]\n"
            "    build:\n"
            "      context: ..\n"
            "      dockerfile: docker/Dockerfile.runtime\n",
        )
        budget = _derive((str(compose),), _contended_cold(97.0))
        assert budget.timeout_seconds <= HARD_UPPER_BOUND_SECONDS

    def test_the_floor_still_holds_under_an_idle_host(self, tmp_path: Path) -> None:
        """A widening term must never be able to read as a narrowing one."""
        _write(tmp_path, "docker/Dockerfile.runtime", "FROM scratch\nRUN echo one\n")
        compose = _write(
            tmp_path,
            "docker/docker-compose.yml",
            "services:\n"
            "  svc:\n"
            "    profiles: [runtime]\n"
            "    build:\n"
            "      context: ..\n"
            "      dockerfile: docker/Dockerfile.runtime\n",
        )
        budget = _derive((str(compose),), _idle_warm())
        assert budget.timeout_seconds >= RECORDED_FLOOR_SECONDS

    def test_the_hard_upper_bound_is_above_the_recorded_kills(self) -> None:
        """A bound below the incident would re-create it as a constant."""
        for _cid, elapsed, _load1 in RECORDED_KILLS:
            assert elapsed < HARD_UPPER_BOUND_SECONDS


@pytest.mark.unit
class TestAC2TheKillMessageNamesObservedProgress:
    """AC2: a kill says which term was exceeded and at what observed rate."""

    def test_parses_completed_steps_and_exported_images(self) -> None:
        progress = parse_build_progress(
            "#1 [internal] load build definition\n"
            "#1 DONE 0.1s\n"
            "#12 [builder 3/9] RUN uv sync\n"
            "#12 DONE 41.2s\n"
            "#13 [builder 4/9] COPY src/ ./src/\n"
            "#13 DONE 2.0s\n"
            "#30 exporting to image\n"
            "#30 naming to docker.io/library/omnibase-infra-omninode-runtime\n"
            "#30 naming to docker.io/library/omnibase-infra-runtime-effects\n"
        )
        assert progress.steps_completed == 3
        assert progress.images_exported == 2

    def test_a_step_with_no_done_line_is_not_counted_as_completed(self) -> None:
        progress = parse_build_progress(
            "#12 [builder 3/9] RUN uv sync\n#12 DONE 41.2s\n#13 [builder 4/9] RUN slow\n"
        )
        assert progress.steps_completed == 1

    def test_unreadable_output_is_an_explicit_nothing_not_a_zero(self) -> None:
        progress = parse_build_progress(None)
        assert progress.observed is False
        assert progress.steps_completed == 0

    def test_a_stalled_build_is_distinguishable_from_a_slow_one(self) -> None:
        stalled = ModelBuildProgress(
            observed=True, steps_completed=0, images_exported=0
        )
        slow = ModelBuildProgress(observed=True, steps_completed=55, images_exported=0)
        stalled_text = stalled.describe(
            elapsed_seconds=1081, assumed_steps=60, assumed_per_step_seconds=15
        )
        slow_text = slow.describe(
            elapsed_seconds=1081, assumed_steps=60, assumed_per_step_seconds=15
        )
        assert stalled_text != slow_text
        assert "0/60" in stalled_text
        assert "55/60" in slow_text

    def test_the_description_names_observed_rate_against_the_assumed_rate(
        self,
    ) -> None:
        text = ModelBuildProgress(
            observed=True, steps_completed=55, images_exported=0
        ).describe(elapsed_seconds=1100, assumed_steps=60, assumed_per_step_seconds=15)
        assert "20.0s/step observed" in text
        assert "15s/step assumed" in text

    def test_no_observation_says_so_rather_than_implying_a_stall(self) -> None:
        text = ModelBuildProgress(
            observed=False, steps_completed=0, images_exported=0
        ).describe(elapsed_seconds=1081, assumed_steps=60, assumed_per_step_seconds=15)
        assert "no build progress output was captured" in text


@pytest.mark.unit
class TestAC4OutcomeTokensAreDistinct:
    """AC4: budget exhausted and build errored are different tokens."""

    def test_the_two_outcomes_have_distinct_tokens(self) -> None:
        assert (
            EnumBuildOutcome.BUDGET_EXHAUSTED.value
            != EnumBuildOutcome.BUILD_ERRORED.value
        )

    def test_neither_token_is_a_substring_of_the_other(self) -> None:
        """A caller that classifies by substring must not match both."""
        exhausted = EnumBuildOutcome.BUDGET_EXHAUSTED.value
        errored = EnumBuildOutcome.BUILD_ERRORED.value
        assert exhausted not in errored
        assert errored not in exhausted

    def test_classify_reads_the_token_off_an_agent_error_string(self) -> None:
        assert (
            EnumBuildOutcome.classify(
                f"{EnumBuildOutcome.BUDGET_EXHAUSTED.value}: runtime image build "
                "for profile 'runtime' exceeded its 1080s ceiling"
            )
            is EnumBuildOutcome.BUDGET_EXHAUSTED
        )
        assert (
            EnumBuildOutcome.classify(
                f"{EnumBuildOutcome.BUILD_ERRORED.value}: Docker compose build failed"
            )
            is EnumBuildOutcome.BUILD_ERRORED
        )

    def test_an_unrelated_error_classifies_to_none_not_to_a_default(self) -> None:
        assert EnumBuildOutcome.classify("lane lock contended") is None
        assert EnumBuildOutcome.classify("") is None
        assert EnumBuildOutcome.classify(None) is None


@pytest.mark.unit
class TestTheBudgetReportsItsOwnHostTerms:
    """A ceiling a reader cannot explain is the one that produced this ticket."""

    def test_describe_names_the_host_terms_it_applied(self, tmp_path: Path) -> None:
        compose_files = _recorded_model(tmp_path)
        text = _derive(compose_files, _contended_cold(97.0)).describe()
        assert "load1" in text
        assert "cache" in text

    def test_describe_says_so_when_no_host_conditions_were_read(
        self, tmp_path: Path
    ) -> None:
        compose_files = _recorded_model(tmp_path)
        text = _derive(compose_files, None).describe()
        assert "no host conditions" in text
