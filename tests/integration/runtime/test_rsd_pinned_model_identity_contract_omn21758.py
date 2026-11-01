# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The packaged RSD pin contract on disk drives the real RSD runtime models (OMN-21758)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from pydantic import ValidationError

import omnibase_infra
from omnibase_infra.models.rsd.model_rsd_pinned_model_identity import (
    ModelRsdPinnedModelIdentity,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_observation import (
    ModelRsdLiveDelegationObservation,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(omnibase_infra.__file__).parent
    / "contracts"
    / "rsd_pinned_model_identity.yaml"
)


def test_pin_comes_from_the_contract_file_on_disk() -> None:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    declared = document["pinned_model_identity"]
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    assert pinned.model_dump() == declared


def test_real_runtime_model_accepts_the_pin_and_refuses_a_substitute() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    revision = "0" * 40
    accepted = ModelRsdLiveDelegationObservation(
        installed_public_rsd_revision_sha=revision,
        observed_model_id=pinned.observed_model,
    )
    assert accepted.observed_model_id == pinned.observed_model
    with pytest.raises(ValidationError, match="contract pin"):
        ModelRsdLiveDelegationObservation(
            installed_public_rsd_revision_sha=revision,
            observed_model_id=pinned.served_model,
        )


def test_built_wheel_loads_the_pin_outside_the_source_tree(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (wheel,) = wheels.glob("*.whl")
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        assert (
            "omnibase_infra/contracts/rsd_pinned_model_identity.yaml"
            in archive.namelist()
        )
        archive.extractall(installed)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import omnibase_infra; "
            "from omnibase_infra.models.rsd.model_rsd_pinned_model_identity "
            "import ModelRsdPinnedModelIdentity; "
            "from omnibase_infra.runtime.models.model_rsd_live_delegation_observation "
            "import ModelRsdLiveDelegationObservation; "
            "import sys; "
            "assert Path(omnibase_infra.__file__).is_relative_to(sys.argv[1]); "
            "pin = ModelRsdPinnedModelIdentity.from_contract(); "
            "observation = ModelRsdLiveDelegationObservation("
            "installed_public_rsd_revision_sha='0'*40, observed_model_id=pin.observed_model); "
            "assert observation.observed_model_id == pin.observed_model; "
            "print('wheel RSD pin readback passed')",
            str(installed),
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(installed), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "wheel RSD pin readback passed"
