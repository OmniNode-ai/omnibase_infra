# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OpenRouter secret name resolves on the lanes that hold it [OMN-16891].

``llm.openrouter.api_key`` is the logical ref every OpenRouter backend in
omnimarket's ``bifrost_delegation.yaml`` declares. On a deployed lane that ref
is resolved SOLELY through this repo's per-lane ``secret_resolver_mappings``:
every lane sets ``enable_convention_fallback: false``, so if the mapping names
an env var the host does not export, the tier is credential-dead with no alias
path to rescue it.

Live evidence, 2026-08-28 (names and LENGTHS only — no secret value read):

* ``.201`` host ``~/.omnibase/.env`` exports ``OPENROUTER_API_KEY`` (len 73).
  It does NOT define ``OPEN_ROUTER_API_KEY`` (len 0).
* Every runtime container on the dev and judge lanes reported len 0 for BOTH
  names — ``omninode-runtime``, ``omninode-runtime-effects``,
  ``omninode-judge-runtime``. The judge compose passes
  ``OPEN_ROUTER_API_KEY: ${OPEN_ROUTER_API_KEY:-}``, which expands to EMPTY
  because the host defines the other spelling.

So the OpenRouter rung has never been able to carry work on any deployed lane.
OMN-13943 and OMN-15048 asserted the opposite ("canonical ``~/.omnibase/.env``
declares ``OPEN_ROUTER_API_KEY``") and propagated that spelling across both
repos; that premise is false on the runtime host. This test pins the corrected
name as a DATA INVARIANT over the committed lane contracts so the spelling
cannot drift back silently — a mapping is either resolvable or it is a lie.

Convergence direction: ``OPENROUTER_API_KEY`` — the provider's own documented
variable name, and the only spelling that carries a real value on the host.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_POLICY = _REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
_RUNTIME_POLICY_ENV = _REPO_ROOT / "docker" / "runtime-policy.env"
_JUDGE_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.judge.yml"
_MODEL_REGISTRY = _REPO_ROOT / "docker" / "catalog" / "model_registry.yaml"

_OPENROUTER_REF = "llm.openrouter.api_key"

# The one canonical spelling. Live-verified as the name holding a real value on
# the .201 host (2026-08-28).
_CANONICAL_ENV = "OPENROUTER_API_KEY"

# The spelling OMN-13943/OMN-15048 propagated on a false premise. It resolves
# to nothing on every surface probed, so nothing may name it.
_RETIRED_ENV = "OPEN_ROUTER_API_KEY"


def _load_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.unit
class TestOpenRouterSecretNameConvergence:
    """One spelling, on every surface that names the OpenRouter secret."""

    def test_every_lane_mapping_names_the_canonical_env_var(self) -> None:
        """Each lane's ``llm.openrouter.api_key`` mapping must be resolvable.

        ``enable_convention_fallback`` is false on every lane, so this mapping
        is the ONLY resolution path — a wrong name here is a dead tier, not a
        degraded one.
        """
        lanes: dict[str, Any] = _load_yaml(_RUNTIME_POLICY)["profiles"]
        offenders: dict[str, str] = {}
        checked: list[str] = []

        for lane_name, lane in lanes.items():
            for mapping in lane.get("secret_resolver_mappings") or []:
                if mapping.get("logical_name") != _OPENROUTER_REF:
                    continue
                source_path = mapping.get("source", {}).get("source_path")
                checked.append(lane_name)
                if source_path != _CANONICAL_ENV:
                    offenders[lane_name] = str(source_path)

        assert checked, (
            f"no lane in {_RUNTIME_POLICY.name} declares a {_OPENROUTER_REF!r} "
            "mapping — the OpenRouter rung cannot resolve on any lane"
        )
        assert not offenders, (
            f"lane(s) map {_OPENROUTER_REF!r} to a non-canonical env var: "
            f"{offenders}. The .201 host exports {_CANONICAL_ENV!r}; "
            f"{_RETIRED_ENV!r} is defined nowhere and resolves to empty."
        )

    def test_generated_lane_env_blobs_name_the_canonical_env_var(self) -> None:
        """The rendered ``*_SECRET_RESOLVER_CONFIG_JSON`` blobs must agree.

        ``docker/runtime-policy.env`` carries the same mappings pre-rendered as
        JSON. Drift between the contract and the rendered blob would let a
        redeploy silently reintroduce the dead name.
        """
        text = _RUNTIME_POLICY_ENV.read_text(encoding="utf-8")
        offenders: dict[str, str] = {}

        for line in text.splitlines():
            if "_SECRET_RESOLVER_CONFIG_JSON=" not in line:
                continue
            var_name, _, raw = line.partition("=")
            raw = raw.strip()
            if not (raw.startswith("'") and raw.endswith("'")):
                continue
            blob = json.loads(raw[1:-1])
            for mapping in blob.get("mappings", []):
                if mapping.get("logical_name") != _OPENROUTER_REF:
                    continue
                source_path = mapping.get("source", {}).get("source_path")
                if source_path != _CANONICAL_ENV:
                    offenders[var_name] = str(source_path)

        assert not offenders, (
            "rendered lane secret-resolver blobs still map "
            f"{_OPENROUTER_REF!r} to a dead env var: {offenders}"
        )

    def test_no_committed_surface_names_the_retired_spelling(self) -> None:
        """``OPEN_ROUTER_API_KEY`` must not survive on any wiring surface.

        Keeping the retired spelling anywhere is worse than deleting it: it
        names a variable no host defines, so it reads as configured while
        resolving to nothing (the exact failure OMN-13943 introduced).
        """
        surfaces = (
            _RUNTIME_POLICY,
            _RUNTIME_POLICY_ENV,
            _JUDGE_COMPOSE,
            _MODEL_REGISTRY,
        )
        offenders: dict[str, list[int]] = {}

        # Word-boundary match so OPENROUTER_API_KEY never counts as a hit.
        pattern = re.compile(rf"\b{_RETIRED_ENV}\b")
        for path in surfaces:
            hits = [
                lineno
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                )
                if pattern.search(line)
            ]
            if hits:
                offenders[str(path.relative_to(_REPO_ROOT))] = hits

        assert not offenders, (
            f"{_RETIRED_ENV!r} still appears on wiring surfaces (file -> lines): "
            f"{offenders}. It resolves to empty on every probed host and lane."
        )

    def test_judge_compose_passes_the_canonical_var_through(self) -> None:
        """The judge lane must forward the name the host actually exports.

        The judge lane is the only lane whose compose file passes an OpenRouter
        variable through explicitly, so it is the one place a rename could be
        half-applied and still look wired.
        """
        text = _JUDGE_COMPOSE.read_text(encoding="utf-8")
        assert _CANONICAL_ENV in text, (
            f"{_JUDGE_COMPOSE.name} does not pass {_CANONICAL_ENV!r} through to "
            "the judge runtime — its OpenRouter-backed judge cannot authenticate"
        )
