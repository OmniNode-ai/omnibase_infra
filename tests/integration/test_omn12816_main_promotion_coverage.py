# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.integration


def test_omn12816_llm_inference_contract_declares_extra_body() -> None:
    contract_path = (
        Path(__file__).parents[2]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )

    contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    # OMN-18385 moved this to patch 1: the api-key field on the node's
    # request models became a secret-wrapped type, so a dump of one emits a
    # mask instead of the key. The wire shape did not change. The pin is an
    # exact equality on purpose -- it makes any contract movement on a
    # promoted node show up in a PR diff rather than drifting unnoticed --
    # so moving it here is the intended workflow, not a way around the test.
    assert contract["contract_version"] == {"major": 1, "minor": 5, "patch": 1}
    assert "extra_body" in contract["input_model"]["description"]
