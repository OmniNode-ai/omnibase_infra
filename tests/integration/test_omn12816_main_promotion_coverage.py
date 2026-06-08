# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
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

    assert contract["contract_version"] == {"major": 1, "minor": 4, "patch": 1}
    assert "extra_body" in contract["input_model"]["description"]
