# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the product-repository option from CLI mapping to sweep output.

The real CLI registry, argument parser, request model and handler are joined
here. Only the external GitHub, Linear and verifier boundaries use doubles.
The JSON round trip represents the input/output payloads used by dispatch.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from omnibase_infra.cli.cli_skill import _parse_skill_args, load_skill_registry
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16106_product_repo_forward_arm import (
    _PRODUCT_REPO,
    _merged_with_contract,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_17342_backfill_arm import (
    _FakeLinear,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("include_product_repo", [True, False])
async def test_cli_product_repos_reaches_sweep_verification(
    include_product_repo: bool,
) -> None:
    mapping = load_skill_registry().get("evidence_autoclose_sweep")
    assert mapping is not None
    args: tuple[str, ...] = ("--backfill-lookback-hours", "0")
    if include_product_repo:
        args += ("--product-repos", f"{_PRODUCT_REPO}, OmniNode-ai/omnibase_core")
    payload = _parse_skill_args(mapping, args)
    correlation_id = uuid4()
    payload["correlation_id"] = str(correlation_id)
    request = ModelEvidenceAutocloseSweepRequest.model_validate_json(
        json.dumps(payload)
    )
    gh = _merged_with_contract()
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(request)
    readback = ModelEvidenceAutocloseSweepResult.model_validate_json(
        result.model_dump_json()
    )

    assert readback.correlation_id == correlation_id
    assert readback.dry_run is True
    assert linear.state_updates == []
    assert linear.comments == []
    if include_product_repo:
        assert request.product_repos == (_PRODUCT_REPO, "OmniNode-ai/omnibase_core")
        assert gh.dod_verify == ["OMN-21146"]
        assert readback.companions_scanned == 1
        assert readback.outcomes[0].ticket_id == "OMN-21146"
        assert readback.outcomes[0].companion_pr_url == (
            f"https://github.com/{_PRODUCT_REPO}/pull/3993"
        )
        assert any(
            path.startswith(f"repos/{_PRODUCT_REPO}/pulls/3993/files")
            for path in gh.paths
        )
        assert any("OmniNode-ai/omnibase_core/pulls?" in path for path in gh.paths)
    else:
        assert request.product_repos == ()
        assert gh.dod_verify == []
        assert readback.companions_scanned == 0
        assert readback.outcomes == ()
        assert not any(_PRODUCT_REPO in path for path in gh.paths)
