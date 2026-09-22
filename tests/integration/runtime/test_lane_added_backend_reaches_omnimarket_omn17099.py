# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17099: a lane-added backend survives the real consumer, and stays house.

The renderer may now append a backend the base contract does not declare. Two
things have to be true of that artifact on the OTHER side of the seam, in
omnimarket, or the freedom is either useless or dangerous:

1. The consumer accepts it. omnimarket validates the whole rendered contract
   against its own backend schema at load; an added entry missing a field that
   schema requires would take every task class down at runtime start.
2. A credential the lane declares is a HOUSE credential. omnimarket derives the
   house set from every resolved backend's reference
   (``customer_key_terminus.house_credential_refs``), and the customer-key
   terminus refuses a customer-attributed route that would authenticate with
   one (OMN-17082, C10). This drives that refusal through the real modules, so
   the new freedom demonstrably cannot bind a house key to customer work.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

omnimarket_loader = pytest.importorskip(
    "omnimarket.adapters.llm.bifrost.config_loader_bifrost_delegation"
)
terminus = pytest.importorskip("omnimarket.routing.customer_key_terminus")

_ROOT = Path(__file__).resolve().parents[3]
_DEV_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_ADDED_REF = "llm.omn17099-added.api_key"


def _packaged_base_contract() -> Path:
    import importlib.resources

    path = Path(
        str(
            importlib.resources.files("omnimarket").joinpath(
                "configs/bifrost_delegation.yaml"
            )
        )
    )
    assert path.is_file(), path
    return path


def _overlay_with_added_backend(tmp_path: Path) -> Path:
    overlay = yaml.safe_load(_DEV_OVERLAY.read_text(encoding="utf-8"))
    overlay["backends"].append(
        {
            "backend_id": "cloud-omn17099-added",
            "endpoint_url": "https://inference.example.test/v1/chat/completions",
            "served_model_id": "added-model",
            "parameter_count": "7B",
            "context_window": 32_768,
            "max_tokens": 8_192,
            "timeout_ms": 60_000,
            "provider": "openrouter",
            "tier": "cheap_cloud",
            "credential": {"kind": "secret_ref", "secret_ref": _ADDED_REF},
            "capabilities": ["code_generation"],
        }
    )
    path = tmp_path / "dev.bifrost.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path


def test_the_added_backend_loads_and_its_credential_is_house(tmp_path: Path) -> None:
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=_packaged_base_contract(),
        overlay_path=_overlay_with_added_backend(tmp_path),
        target_path=target,
        environ={},
        verify_endpoints=False,
    )

    config = omnimarket_loader.load_bifrost_delegation_config(
        config_path=target, overlay_path=None
    )
    by_id = {backend.backend_id: backend for backend in config.backends}
    added = by_id["cloud-omn17099-added"]
    assert added.resolved_secret_ref == _ADDED_REF
    assert added.endpoint_url == "https://inference.example.test/v1/chat/completions"

    resolved = {
        backend.backend_id: SimpleNamespace(
            api_key_ref=backend.resolved_secret_ref, api_key_env=None
        )
        for backend in config.backends
    }
    house = terminus.house_credential_refs(resolved)
    assert _ADDED_REF in house

    with pytest.raises(terminus.CustomerKeyRefusedError) as excinfo:
        terminus.enforce_customer_key_terminus(
            tenant_id="customer-tenant-omn17099",
            task_type="code_generation",
            correlation_id=uuid4(),
            surface=terminus.EnumDelegationSurface.CLOUD,
            api_key_ref=_ADDED_REF,
            api_key_env=None,
            backend_ref="cloud-omn17099-added",
            house_refs=house,
        )
    assert (
        excinfo.value.refusal.reason
        is terminus.EnumCustomerKeyRefusalReason.HOUSE_CREDENTIAL_ON_CUSTOMER_PATH
    )

    # Control: the same route on the house/untenanted path is not refused.
    terminus.enforce_customer_key_terminus(
        tenant_id=None,
        task_type="code_generation",
        correlation_id=uuid4(),
        surface=terminus.EnumDelegationSurface.CLOUD,
        api_key_ref=_ADDED_REF,
        api_key_env=None,
        backend_ref="cloud-omn17099-added",
        house_refs=house,
    )
