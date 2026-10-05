# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20490: a rung on a unified-pool endpoint never asks for more than its share.

A llama.cpp server under ``--kv-unified --parallel N`` shares one token pool
between its N slots and reports that pool as every slot's ``n_ctx``. Four
concurrent requests that each keep prompt plus output within ``pool / N`` fit;
one that outgrows it can exhaust the pool and the server answers HTTP 500
``Context size has been exceeded``. A binding's ``max_tokens`` is the output
side of that sum, so it is held here to the capacity its lane overlay declares
for the endpoint (``endpoint_capacity``: pool, slots, prompt headroom).

Every number comes from the committed overlays. The test covers each committed
lane overlay and the contract the renderer writes from it, so a binding added
to another lane, or a renderer that stopped writing the overlay's ceiling, is
caught the same way as an edit to the dev lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_DIR = ROOT / "docker" / "lane-overlays"

pytestmark = pytest.mark.unit
sys.path.insert(0, str(ROOT / "src"))

from omnibase_infra.models.delegation.model_bifrost_lane_endpoint_capacity import (
    ModelBifrostLaneEndpointCapacity,
)
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

_DEV_LANE = "dev"
_DEV_202_LANE = "dev-202"


def _load(path: Path) -> ModelBifrostLaneOverlay:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelBifrostLaneOverlay.model_validate(raw)


def _committed_overlays() -> dict[str, ModelBifrostLaneOverlay]:
    paths = sorted(OVERLAY_DIR.glob("*.bifrost.yaml"))
    assert paths, f"no committed lane overlay under {OVERLAY_DIR}"
    return {path.name: _load(path) for path in paths}


def _declared_capacities(
    overlays: dict[str, ModelBifrostLaneOverlay],
) -> dict[str, ModelBifrostLaneEndpointCapacity]:
    """Every endpoint any committed overlay declares a capacity for.

    A pool endpoint is a property of the server, not of one lane: a lane that
    binds the same ``host:port`` without restating the capacity is held to the
    capacity another lane declared. Two lanes declaring different capacities
    for one server cannot both be right, so that is refused here.
    """
    declared: dict[str, ModelBifrostLaneEndpointCapacity] = {}
    for name, overlay in overlays.items():
        for capacity in overlay.endpoint_capacity:
            known = declared.setdefault(capacity.endpoint, capacity)
            assert known == capacity, (
                f"{name} declares {capacity.endpoint} as {capacity!r}, which "
                f"contradicts the earlier declaration {known!r}"
            )
    return declared


def _capacity_for(
    endpoint_url: str, capacities: dict[str, ModelBifrostLaneEndpointCapacity]
) -> ModelBifrostLaneEndpointCapacity | None:
    return next(
        (capacity for capacity in capacities.values() if capacity.serves(endpoint_url)),
        None,
    )


def _share_violation(
    *,
    where: str,
    backend_id: str,
    max_tokens: int,
    capacity: ModelBifrostLaneEndpointCapacity,
) -> str | None:
    """The refusal text for one binding, or None when it fits its share."""
    if max_tokens + capacity.prompt_headroom_tokens <= capacity.per_slot_tokens:
        return None
    return (
        f"{where}: backend {backend_id!r} on {capacity.endpoint} asks for "
        f"max_tokens={max_tokens}, which with the {capacity.prompt_headroom_tokens}"
        f"-token prompt headroom is {max_tokens + capacity.prompt_headroom_tokens} "
        f"tokens, over the per-slot share of {capacity.per_slot_tokens} "
        f"({capacity.pool_tokens} pool tokens / {capacity.parallel_slots} slots). "
        f"Lower max_tokens to at most {capacity.max_output_tokens}."
    )


def _overlay_violations(
    name: str,
    overlay: ModelBifrostLaneOverlay,
    capacities: dict[str, ModelBifrostLaneEndpointCapacity],
) -> tuple[list[str], int]:
    """Refusals for one overlay's bindings, and how many bindings were checked."""
    violations: list[str] = []
    checked = 0
    for binding in overlay.backends:
        capacity = _capacity_for(binding.endpoint_url, capacities)
        if capacity is None:
            continue
        checked += 1
        violation = _share_violation(
            where=name,
            backend_id=binding.backend_key,
            max_tokens=binding.max_tokens,
            capacity=capacity,
        )
        if violation is not None:
            violations.append(violation)
    return violations, checked


def _synthetic_base_contract(
    overlay: ModelBifrostLaneOverlay,
    capacities: dict[str, ModelBifrostLaneEndpointCapacity],
) -> dict[str, object]:
    """A base contract declaring each backend the overlay rebinds, ceiling at the pool.

    The omnimarket base contract is not importable here. The backends an
    overlay ADDS need no base entry; the ones it REBINDS get one whose own
    ``max_tokens`` is the whole pool, so a renderer that let the base ceiling
    through would show up as a violation below.
    """
    pool_ceiling = max(capacity.pool_tokens for capacity in capacities.values())
    tier = "local" if overlay.locale is EnumBifrostLaneLocale.LAB else "cloud"
    return {
        "backends": [
            {
                "backend_id": binding.backend_key,
                "model_name": binding.advertised_model,
                "tier": tier,
                "max_tokens": pool_ceiling,
            }
            for binding in overlay.backends
            if not binding.declares_new_backend
        ]
    }


def _rendered_violations(
    name: str,
    overlay_path: Path,
    overlay: ModelBifrostLaneOverlay,
    capacities: dict[str, ModelBifrostLaneEndpointCapacity],
    tmp_path: Path,
) -> tuple[list[str], int]:
    base = tmp_path / f"{name}.base.yaml"
    base.write_text(
        yaml.safe_dump(_synthetic_base_contract(overlay, capacities)),
        encoding="utf-8",
    )
    target = tmp_path / f"{name}.rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=base,
        overlay_path=overlay_path,
        target_path=target,
        environ={},
        verify_endpoints=False,
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    violations: list[str] = []
    checked = 0
    for backend in rendered["backends"]:
        endpoint_url = backend.get("endpoint_url")
        if not isinstance(endpoint_url, str):
            continue
        capacity = _capacity_for(endpoint_url, capacities)
        if capacity is None:
            continue
        checked += 1
        violation = _share_violation(
            where=f"{name} rendered",
            backend_id=backend["backend_id"],
            max_tokens=backend["max_tokens"],
            capacity=capacity,
        )
        if violation is not None:
            violations.append(violation)
    return violations, checked


def test_every_committed_binding_on_a_pool_endpoint_fits_its_share() -> None:
    overlays = _committed_overlays()
    capacities = _declared_capacities(overlays)
    assert capacities, (
        "no committed overlay declares an endpoint_capacity, so this test "
        "would check nothing"
    )

    violations: list[str] = []
    checked = 0
    for name, overlay in overlays.items():
        found, count = _overlay_violations(name, overlay, capacities)
        violations.extend(found)
        checked += count

    assert not violations, "\n".join(violations)
    # Positive control for the zero above: the dev lane really binds a backend
    # on a declared pool endpoint, so "no violations" was reached by checking it.
    assert checked >= 1, "no committed binding sits on a declared pool endpoint"


def test_every_declared_pool_endpoint_is_bound_by_a_committed_backend() -> None:
    overlays = _committed_overlays()
    capacities = _declared_capacities(overlays)
    bound = {
        capacity.endpoint
        for overlay in overlays.values()
        for binding in overlay.backends
        if (capacity := _capacity_for(binding.endpoint_url, capacities)) is not None
    }
    assert set(capacities) == bound, (
        f"endpoint_capacity declares {sorted(set(capacities) - bound)} but no "
        "committed backend is bound there: a stale declaration holds nothing"
    )


def test_every_overlay_that_binds_a_pool_endpoint_renders_within_its_share(
    tmp_path: Path,
) -> None:
    overlays = _committed_overlays()
    capacities = _declared_capacities(overlays)

    violations: list[str] = []
    rendered_bindings = 0
    for name, overlay in overlays.items():
        if not any(
            _capacity_for(binding.endpoint_url, capacities)
            for binding in overlay.backends
        ):
            continue
        found, count = _rendered_violations(
            name, OVERLAY_DIR / name, overlay, capacities, tmp_path
        )
        violations.extend(found)
        rendered_bindings += count

    assert not violations, "\n".join(violations)
    assert rendered_bindings >= 1, "no rendered backend sits on a pool endpoint"


def test_dev_202_overlay_declares_no_backend_by_design() -> None:
    """The dev-202 lane's zero-local-backend isolation is stated, not skipped.

    OMN-19505: it must not bind the pool endpoint on its own host, so the
    share check has nothing to hold it to. Binding one is a decision of its
    own; this refuses it here rather than letting the capacity test pass it
    by never meeting it.
    """
    overlays = _committed_overlays()
    capacities = _declared_capacities(overlays)
    dev_202 = overlays[f"{_DEV_202_LANE}.bifrost.yaml"]

    assert dev_202.lane == _DEV_202_LANE
    assert dev_202.locale is EnumBifrostLaneLocale.CLOUD
    assert dev_202.backends == ()
    assert [
        binding.backend_key
        for binding in dev_202.backends
        if _capacity_for(binding.endpoint_url, capacities) is not None
    ] == []


def test_the_dev_lane_binds_a_pool_endpoint_so_the_checks_above_are_not_vacuous() -> (
    None
):
    overlays = _committed_overlays()
    capacities = _declared_capacities(overlays)
    dev = overlays[f"{_DEV_LANE}.bifrost.yaml"]

    on_pool = [
        binding.backend_key
        for binding in dev.backends
        if _capacity_for(binding.endpoint_url, capacities) is not None
    ]
    assert on_pool, "the dev lane binds no backend on a declared pool endpoint"


# --- positive controls: the checker refuses what it is meant to refuse -------

_CONTROL_CAPACITY = ModelBifrostLaneEndpointCapacity(
    endpoint="10.0.0.9:8000",
    pool_tokens=1_200,
    parallel_slots=3,
    prompt_headroom_tokens=100,
)
_CONTROL_URL = "http://10.0.0.9:8000/v1/chat/completions"


def _control_overlay(max_tokens: int) -> ModelBifrostLaneOverlay:
    return ModelBifrostLaneOverlay.model_validate(
        {
            "schema_version": "bifrost_lane_overlay.v3",
            "lane": "control",
            "locale": "lab",
            "backends": [
                {
                    "backend_id": "control-chat",
                    "endpoint_url": _CONTROL_URL,
                    "served_model_id": "control-model",
                    "parameter_count": "1B",
                    "context_window": _CONTROL_CAPACITY.pool_tokens,
                    "max_tokens": max_tokens,
                    "timeout_ms": 1_000,
                }
            ],
        }
    )


def test_checker_accepts_a_ceiling_that_exactly_fills_the_share() -> None:
    overlay = _control_overlay(_CONTROL_CAPACITY.max_output_tokens)
    violations, checked = _overlay_violations(
        "control", overlay, {_CONTROL_CAPACITY.endpoint: _CONTROL_CAPACITY}
    )
    assert (violations, checked) == ([], 1)


def test_checker_refuses_a_ceiling_one_token_past_the_share() -> None:
    overlay = _control_overlay(_CONTROL_CAPACITY.max_output_tokens + 1)
    violations, checked = _overlay_violations(
        "control", overlay, {_CONTROL_CAPACITY.endpoint: _CONTROL_CAPACITY}
    )
    assert checked == 1
    assert len(violations) == 1
    assert f"at most {_CONTROL_CAPACITY.max_output_tokens}" in violations[0]


def test_checker_refuses_a_ceiling_that_is_the_whole_share() -> None:
    """The shape the dev lane shipped with: max_tokens equal to the whole share."""
    overlay = _control_overlay(_CONTROL_CAPACITY.per_slot_tokens)
    violations, _ = _overlay_violations(
        "control", overlay, {_CONTROL_CAPACITY.endpoint: _CONTROL_CAPACITY}
    )
    assert len(violations) == 1


def test_checker_ignores_an_endpoint_with_no_declared_capacity() -> None:
    overlay = _control_overlay(_CONTROL_CAPACITY.pool_tokens)
    violations, checked = _overlay_violations("control", overlay, {})
    assert (violations, checked) == ([], 0)
