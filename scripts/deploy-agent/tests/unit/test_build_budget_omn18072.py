# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime IMAGE-BUILD ceiling is derived from the build model (OMN-18072).

MEASURED, from the deploy agent's own dev-lane job history on 2026-09-09
(state dir ``/data/omninode/deploy-agent/state/jobs-dev``, journal
``deploy-agent-dev``):

    6c323639  build start 01:44:30.717Z, first core container Created
              01:48:24.652Z  =>  runtime image build <= 233.9s, job SUCCEEDED
    79171e79  build start 09:05:39.640Z, nine runtime images exported
              09:09:08.891Z (t+209.3s), killed 09:10:39.674Z  =>  300.0s
    2788af33  build start 09:12:08.616Z, killed 09:17:08.902Z  =>  300.3s,
              with a WARM BuildKit cache from the run above

Two consecutive sanctioned rebuilds died at exactly the flat constant, the
second with a warm cache, so the cache is not the variable: this build simply
runs close enough to 300s that ordinary host load decides the outcome. Two of
the three observations are right-censored at 300s, which is why the ceiling is
derived from the build MODEL rather than from a percentile over a history whose
two longest entries were killed instead of measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent.build_budget import (
    BuildBudgetError,
    ModelBuildBudget,
    count_build_steps,
    derive_image_build_budget,
)

# The flat constant that killed both live rebuilds. Named once so the guards
# below read as "not the number that failed" rather than a bare literal.
FAILED_FLAT_CONSTANT_SECONDS = 300
# The longest runtime image build this history actually completed.
MEASURED_SUCCESSFUL_BUILD_SECONDS = 234


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
class TestCountBuildSteps:
    def test_counts_only_work_instructions(self, tmp_path: Path) -> None:
        """FROM/ARG/ENV/WORKDIR cost no wall time and must not inflate the ceiling."""
        dockerfile = _write(
            tmp_path,
            "Dockerfile",
            "FROM python:3.12-slim AS builder\n"
            "ARG X=1\n"
            "ENV Y=2\n"
            "WORKDIR /app\n"
            "RUN echo one\n"
            "COPY src/ ./src/\n"
            "ADD thing.tar /thing\n"
            "USER nobody\n"
            'CMD ["true"]\n',
        )
        assert count_build_steps(dockerfile) == 3

    def test_backslash_continuations_are_one_step(self, tmp_path: Path) -> None:
        dockerfile = _write(
            tmp_path,
            "Dockerfile",
            "FROM scratch\n"
            "RUN set -eu; \\\n"
            "    echo a; \\\n"
            "    echo b\n"
            "RUN echo tail\n",
        )
        assert count_build_steps(dockerfile) == 2

    def test_heredoc_body_is_not_scanned_for_instructions(self, tmp_path: Path) -> None:
        """A shell line inside RUN ... <<'EOF' that starts with RUN is not a step.

        Dockerfile.runtime really does write a helper script this way; a naive
        line-prefix count reads its body as further instructions and inflates
        the derived ceiling with steps that do not exist.
        """
        dockerfile = _write(
            tmp_path,
            "Dockerfile",
            "FROM scratch\n"
            "RUN cat > /usr/local/bin/helper <<'EOF'\n"
            "RUN this is shell text, not an instruction\n"
            "COPY neither is this\n"
            "EOF\n"
            "RUN chmod +x /usr/local/bin/helper\n",
        )
        assert count_build_steps(dockerfile) == 2

    def test_unreadable_dockerfile_raises_rather_than_returning_zero(
        self, tmp_path: Path
    ) -> None:
        """A silent 0 would collapse the ceiling to its floor undetectably."""
        with pytest.raises(BuildBudgetError, match="cannot read Dockerfile"):
            count_build_steps(tmp_path / "definitely-absent")


@pytest.mark.unit
class TestDeriveImageBuildBudget:
    def _model(self, tmp_path: Path, *, services: int, steps: int) -> Path:
        _write(
            tmp_path,
            "docker/Dockerfile.runtime",
            "FROM scratch\n" + "".join(f"RUN echo {i}\n" for i in range(steps)),
        )
        body = ["services:"]
        for index in range(services):
            body.append(f"  svc{index}:")
            body.append("    profiles: [runtime, full]")
            body.append("    build:")
            body.append("      context: ..")
            body.append("      dockerfile: docker/Dockerfile.runtime")
        body.append("  postgres:")
        body.append("    profiles: [core, full]")
        body.append("    image: postgres:16")
        return _write(tmp_path, "docker/compose.yml", "\n".join(body) + "\n")

    def test_ceiling_is_solve_plus_export_read_from_the_model(
        self, tmp_path: Path
    ) -> None:
        compose = self._model(tmp_path, services=9, steps=60)
        budget = derive_image_build_budget(
            (str(compose),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        assert budget.build_steps == 60
        assert len(budget.buildable_services) == 9
        # 60 * 15 (one shared solve) + 9 * 20 (per-image export)
        assert budget.timeout_seconds == 1080

    def test_ceiling_moves_when_the_model_moves(self, tmp_path: Path) -> None:
        """A tenth runtime service or a new RUN step must raise the ceiling.

        This is the whole point of deriving rather than declaring: the number
        cannot go stale behind a growing image.
        """
        base = derive_image_build_budget(
            (str(self._model(tmp_path / "a", services=9, steps=60)),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        grown = derive_image_build_budget(
            (str(self._model(tmp_path / "b", services=10, steps=70)),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        assert grown.timeout_seconds > base.timeout_seconds

    def test_floor_holds_when_the_model_reads_small(self, tmp_path: Path) -> None:
        compose = self._model(tmp_path, services=1, steps=1)
        budget = derive_image_build_budget(
            (str(compose),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        assert budget.timeout_seconds == 600

    def test_profile_with_no_buildable_service_reports_why(
        self, tmp_path: Path
    ) -> None:
        compose = self._model(tmp_path, services=9, steps=60)
        budget = derive_image_build_budget(
            (str(compose),),
            "core",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        assert budget.buildable_services == ()
        assert budget.dockerfile is None
        assert "no service in profile 'core' declares a build:" in budget.describe()

    def test_unreadable_compose_raises_rather_than_falling_back(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BuildBudgetError, match="cannot read compose file"):
            derive_image_build_budget(
                (str(tmp_path / "absent.yml"),),
                "runtime",
                per_step_seconds=15,
                per_image_seconds=20,
                floor_seconds=600,
            )

    def test_describe_names_the_dockerfile_and_both_terms(self, tmp_path: Path) -> None:
        """The derivation is logged verbatim, the way the compose-up one is."""
        compose = self._model(tmp_path, services=9, steps=60)
        text = derive_image_build_budget(
            (str(compose),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        ).describe()
        assert "Dockerfile.runtime" in text
        assert "60 work steps" in text
        assert "9 buildable service(s)" in text
        assert "floor 600s" in text

    def test_budget_is_frozen(self, tmp_path: Path) -> None:
        budget = derive_image_build_budget(
            (str(self._model(tmp_path, services=9, steps=60)),),
            "runtime",
            per_step_seconds=15,
            per_image_seconds=20,
            floor_seconds=600,
        )
        assert isinstance(budget, ModelBuildBudget)
        with pytest.raises(Exception):
            budget.timeout_seconds = 1  # type: ignore[misc]


@pytest.mark.unit
class TestLiveRuntimeModel:
    """The ceiling the dev lane will actually run under, read from the repo."""

    def test_live_runtime_ceiling_clears_every_measured_build(self) -> None:
        from deploy_agent import executor as executor_mod

        # Resolved through the module attribute so the conftest repoint onto
        # THIS checkout applies (REPO_DIR is a deploy-host path).
        budget = executor_mod.runtime_image_build_budget("runtime")

        assert Path(executor_mod.COMPOSE_FILE).name == "docker-compose.infra.yml"
        # The regression guard: not the constant that killed both live rebuilds,
        # and comfortably clear of the one build this history completed.
        assert budget.timeout_seconds != FAILED_FLAT_CONSTANT_SECONDS
        assert budget.timeout_seconds > FAILED_FLAT_CONSTANT_SECONDS
        assert budget.timeout_seconds > 2 * MEASURED_SUCCESSFUL_BUILD_SECONDS
        # Derived, not floored: the live model is big enough that the floor is
        # not what produced this number.
        assert budget.timeout_seconds > budget.floor_seconds
        assert budget.dockerfile is not None
        assert budget.dockerfile.endswith("docker/Dockerfile.runtime")
        assert len(budget.buildable_services) >= 9

    def test_compose_up_ceiling_still_bounds_the_build_from_above(self) -> None:
        """A build ceiling above the compose-up ceiling would be incoherent.

        The build does no health-gated waiting; it must fail sooner than the
        phase that does.
        """
        from deploy_agent import executor as executor_mod
        from deploy_agent.events import EnumRuntimeLane, Scope
        from deploy_agent.executor import services_for_scope

        build = executor_mod.runtime_image_build_budget("runtime")
        compose_up = executor_mod.runtime_compose_up_budget(
            EnumRuntimeLane.DEV, services_for_scope(Scope.RUNTIME)
        )
        assert build.timeout_seconds < compose_up.timeout_seconds
