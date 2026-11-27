# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856: the shipped runtime_config.yaml loads through the kernel's own loader.

The dev lane bind-mounts the host ``contracts/`` over the image's copy, so the
yaml a runtime boots with is validated by whatever model the image carries. This
test loads the real file through ``load_runtime_config`` (the call the entrypoint
makes) and not through a fragment of the model, so a policy value the model
refuses fails here with the message the boot printed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.service_kernel import load_runtime_config

_CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
_RUNTIME_CONFIG = _CONTRACTS / "runtime" / "runtime_config.yaml"


def test_shipped_runtime_config_loads_through_the_kernel_loader() -> None:
    config = load_runtime_config(_CONTRACTS)
    declared = yaml.safe_load(_RUNTIME_CONFIG.read_text(encoding="utf-8"))
    assert config.boot_subscriptions.policy == declared["boot_subscriptions"]["policy"]


def test_a_boot_policy_the_model_refuses_fails_the_load_naming_the_field(
    tmp_path: Path,
) -> None:
    declared = yaml.safe_load(_RUNTIME_CONFIG.read_text(encoding="utf-8"))
    declared["boot_subscriptions"] = {"policy": "everything"}
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "runtime_config.yaml").write_text(yaml.safe_dump(declared))
    with pytest.raises(Exception, match=r"boot_subscriptions\.policy"):
        load_runtime_config(tmp_path)
