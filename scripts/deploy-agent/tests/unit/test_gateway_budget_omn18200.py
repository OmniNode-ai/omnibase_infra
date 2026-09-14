# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The gateway deploy ceiling is derived from the gateway model, not a flat
floor-plus-constant (OMN-18200 residual 3, same class as OMN-18057/OMN-18072).

``GATEWAY_DEPLOY_TIMEOUT_SECONDS`` was
``RUNTIME_IMAGE_BUILD_FLOOR_SECONDS + 300`` (900s), unrelated to the gateway's
own build or reload cost. On the only rebuild since ``omnibase_infra#3524``
merged, ``_deploy_gateway_lane`` was killed at exactly 900s: roughly 780s went
to a cold image build, leaving the recreate half (reload + verify) no budget
at all. These tests pin the derivation -- build and recreate budgeted
separately, so a cold build can never consume the recreate half -- not a
number.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent.build_budget import BuildBudgetError, ModelBuildBudget
from deploy_agent.gateway_budget import (
    GatewayBudgetError,
    ModelGatewayDeployBudget,
    derive_gateway_deploy_budget,
    read_exec_reload_wait_timeout,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _gateway_compose(tmp_path: Path, *, runtime_steps: int, sidecar_steps: int) -> Path:
    _write(
        tmp_path,
        "docker/Dockerfile.runtime",
        "FROM scratch\n" + "".join(f"RUN echo {i}\n" for i in range(runtime_steps)),
    )
    _write(
        tmp_path,
        "docker/gateway/dns-bastion/Dockerfile",
        "FROM alpine:3.20\n" + "".join(f"RUN echo {i}\n" for i in range(sidecar_steps)),
    )
    return _write(
        tmp_path,
        "docker/docker-compose.gateway.yml",
        "\n".join(
            [
                "services:",
                "  gateway-dns-bastion:",
                "    build:",
                "      context: ./gateway/dns-bastion",
                "  gateway-forwarder:",
                "    image: '${GATEWAY_IMAGE:?required}'",
                "    build:",
                "      context: ..",
                "      dockerfile: docker/Dockerfile.runtime",
                "    depends_on:",
                "      gateway-dns-bastion:",
                "        condition: service_healthy",
                "",
            ]
        ),
    )


def _service_unit(tmp_path: Path, *, wait_timeout: str | None = "120") -> Path:
    wait_clause = f" --wait-timeout {wait_timeout}" if wait_timeout else ""
    return _write(
        tmp_path,
        "docker/gateway/onex-gateway-forwarder.service",
        "\n".join(
            [
                "[Service]",
                "Type=oneshot",
                (
                    "ExecStart=/usr/bin/docker compose -f x.yml up -d --no-build "
                    f"--wait{wait_clause} gateway-forwarder"
                ),
                (
                    "ExecReload=/usr/bin/docker compose -f x.yml up -d --no-build "
                    f"--force-recreate --wait{wait_clause} gateway-forwarder"
                ),
                "",
            ]
        ),
    )


def _service_unit_no_exec_reload(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        "docker/gateway/onex-gateway-forwarder.service",
        "[Service]\nType=oneshot\nExecStart=/usr/bin/true\n",
    )


DEFAULT_KWARGS = {
    "per_step_seconds": 15,
    "per_image_seconds": 20,
    "build_floor_seconds": 600,
    "reload_margin_seconds": 120,
    "reload_floor_seconds": 180,
}


@pytest.mark.unit
class TestReadExecReloadWaitTimeout:
    def test_reads_the_declared_value(self, tmp_path: Path) -> None:
        unit = _service_unit(tmp_path, wait_timeout="120")
        assert read_exec_reload_wait_timeout(unit) == 120

    def test_a_different_value_is_read_back_exactly(self, tmp_path: Path) -> None:
        """Pins the read as parsing, not a hardcoded 120 in disguise."""
        unit = _service_unit(tmp_path, wait_timeout="45")
        assert read_exec_reload_wait_timeout(unit) == 45

    def test_missing_wait_timeout_raises(self, tmp_path: Path) -> None:
        unit = _service_unit(tmp_path, wait_timeout=None)
        with pytest.raises(GatewayBudgetError, match="no --wait-timeout"):
            read_exec_reload_wait_timeout(unit)

    def test_missing_exec_reload_line_raises(self, tmp_path: Path) -> None:
        unit = _service_unit_no_exec_reload(tmp_path)
        with pytest.raises(GatewayBudgetError, match="no ExecReload="):
            read_exec_reload_wait_timeout(unit)

    def test_unreadable_unit_raises_rather_than_returning_a_default(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(GatewayBudgetError, match="cannot read systemd unit"):
            read_exec_reload_wait_timeout(tmp_path / "absent.service")


@pytest.mark.unit
class TestDeriveGatewayDeployBudget:
    def test_the_live_incident_shape_is_no_longer_a_bare_900s(
        self, tmp_path: Path
    ) -> None:
        """Reproduces the failing shape: the real gateway Dockerfile is 60 work
        steps (measured on the checked-out runtime Dockerfile the gateway
        shares) and two buildable services. Under the OLD floor-plus-constant
        (900s flat) that alone consumed ~780s of the 900s budget on the one
        live rebuild, leaving the recreate half nothing. The derived budget
        must reserve the recreate half ON TOP of whatever the build costs,
        so the total exceeds the old flat constant instead of being bounded
        by it.
        """
        compose = _gateway_compose(tmp_path, runtime_steps=60, sidecar_steps=4)
        unit = _service_unit(tmp_path, wait_timeout="120")
        budget = derive_gateway_deploy_budget(
            (str(compose),), "gateway", unit, **DEFAULT_KWARGS
        )
        old_flat_constant_seconds = 900
        assert budget.timeout_seconds > old_flat_constant_seconds
        # The recreate half is intact, not eaten by the build's own cost.
        assert budget.recreate_seconds == 240  # 120 wait-timeout + 120 margin
        assert budget.timeout_seconds == budget.build.timeout_seconds + 240

    def test_a_cold_build_does_not_shrink_the_recreate_budget(
        self, tmp_path: Path
    ) -> None:
        """The whole point of budgeting separately: a bigger Dockerfile must
        raise the TOTAL, never take seconds away from the recreate half.
        """
        small = derive_gateway_deploy_budget(
            (str(_gateway_compose(tmp_path / "a", runtime_steps=10, sidecar_steps=1)),),
            "gateway",
            _service_unit(tmp_path / "a"),
            **DEFAULT_KWARGS,
        )
        large = derive_gateway_deploy_budget(
            (
                str(
                    _gateway_compose(tmp_path / "b", runtime_steps=200, sidecar_steps=1)
                ),
            ),
            "gateway",
            _service_unit(tmp_path / "b"),
            **DEFAULT_KWARGS,
        )
        assert large.timeout_seconds > small.timeout_seconds
        assert large.recreate_seconds == small.recreate_seconds == 240

    def test_recreate_floor_holds_against_a_suspiciously_small_wait_timeout(
        self, tmp_path: Path
    ) -> None:
        compose = _gateway_compose(tmp_path, runtime_steps=10, sidecar_steps=1)
        unit = _service_unit(tmp_path, wait_timeout="5")
        budget = derive_gateway_deploy_budget(
            (str(compose),), "gateway", unit, **DEFAULT_KWARGS
        )
        assert budget.reload_wait_timeout_seconds == 5
        # 5 + 120 margin = 125, below the 180s floor -- the floor must win.
        assert budget.recreate_seconds == 180

    def test_unreadable_compose_raises_rather_than_falling_back(
        self, tmp_path: Path
    ) -> None:
        unit = _service_unit(tmp_path, wait_timeout="120")
        with pytest.raises(BuildBudgetError, match="cannot read compose file"):
            derive_gateway_deploy_budget(
                (str(tmp_path / "absent.yml"),), "gateway", unit, **DEFAULT_KWARGS
            )

    def test_unreadable_unit_raises_rather_than_falling_back(
        self, tmp_path: Path
    ) -> None:
        compose = _gateway_compose(tmp_path, runtime_steps=10, sidecar_steps=1)
        with pytest.raises(GatewayBudgetError, match="cannot read systemd unit"):
            derive_gateway_deploy_budget(
                (str(compose),),
                "gateway",
                tmp_path / "absent.service",
                **DEFAULT_KWARGS,
            )

    def test_describe_names_both_halves_and_their_source(self, tmp_path: Path) -> None:
        compose = _gateway_compose(tmp_path, runtime_steps=60, sidecar_steps=4)
        unit = _service_unit(tmp_path, wait_timeout="120")
        text = derive_gateway_deploy_budget(
            (str(compose),), "gateway", unit, **DEFAULT_KWARGS
        ).describe()
        assert "build" in text
        assert "recreate 240s" in text
        assert "--wait-timeout 120s" in text
        assert str(unit) in text

    def test_budget_is_frozen(self, tmp_path: Path) -> None:
        compose = _gateway_compose(tmp_path, runtime_steps=10, sidecar_steps=1)
        unit = _service_unit(tmp_path, wait_timeout="120")
        budget = derive_gateway_deploy_budget(
            (str(compose),), "gateway", unit, **DEFAULT_KWARGS
        )
        assert isinstance(budget, ModelGatewayDeployBudget)
        assert isinstance(budget.build, ModelBuildBudget)
        with pytest.raises(Exception):
            budget.timeout_seconds = 1  # type: ignore[misc]


@pytest.mark.unit
class TestAgainstTheRealCheckedOutGatewayModel:
    """Exercises the derivation against the ACTUAL repo files the deploy
    invokes, not a synthetic fixture -- the same discipline
    ``test_build_budget_omn18072`` and ``test_compose_budget_omn18057`` apply
    by importing the real compose file for their lane-level tests. This is
    what proves the 900s flat constant really was undersized for THIS
    Dockerfile family, not just for a constructed worst case.
    """

    def test_real_model_exceeds_the_old_flat_constant(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        compose = repo_root / "docker" / "docker-compose.gateway.yml"
        unit = repo_root / "docker" / "gateway" / "onex-gateway-forwarder.service"
        assert compose.is_file(), compose
        assert unit.is_file(), unit
        budget = derive_gateway_deploy_budget(
            (str(compose),), "gateway", unit, **DEFAULT_KWARGS
        )
        old_flat_constant_seconds = 900
        assert budget.timeout_seconds > old_flat_constant_seconds
        assert budget.reload_wait_timeout_seconds == 120
        assert len(budget.build.buildable_services) == 2
