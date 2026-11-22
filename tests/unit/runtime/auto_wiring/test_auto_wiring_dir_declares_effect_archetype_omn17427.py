# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""src/omnibase_infra/runtime/auto_wiring/ declares the effect archetype for K4 (OMN-17427).

The check-no-io-outside-effects hook judges a module against the nearest
contract.yaml above it. With no contract above the auto-wiring package, its
nodeless-module fallback (any ``handler_*.py``) judged handler_wiring.py as
non-EFFECT, so every PR that touched that file failed on its environment,
clock and database surfaces, which are the module's job.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_no_io_outside_effects import (
    ValidatorNoIoOutsideEffects,
)

_AUTO_WIRING = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "auto_wiring"
)
_IO_MODULE = _AUTO_WIRING / "handler_wiring.py"


@pytest.mark.unit
def test_auto_wiring_contract_declares_the_effect_archetype() -> None:
    contract = yaml.safe_load((_AUTO_WIRING / "contract.yaml").read_text())

    assert contract["node_type"] == "EFFECT_GENERIC"
    assert contract["descriptor"]["node_archetype"] == "effect"


@pytest.mark.unit
def test_wiring_module_passes_under_the_auto_wiring_contract() -> None:
    result = ValidatorNoIoOutsideEffects().validate(
        [_AUTO_WIRING / "contract.yaml", _IO_MODULE]
    )

    assert result.is_valid, result.issues


@pytest.mark.unit
def test_same_module_fails_without_the_archetype_declaration(tmp_path: Path) -> None:
    # Deliberately hostile outer context: the negative control must not inherit
    # EFFECT even if this test runs beneath a checkout with such a contract.
    shutil.copy(_AUTO_WIRING / "contract.yaml", tmp_path / "contract.yaml")
    non_effect = tmp_path / "non_effect"
    non_effect.mkdir()
    (non_effect / "contract.yaml").write_text(
        "node_type: COMPUTE_GENERIC\ndescriptor:\n  node_archetype: compute\n"
    )
    package = non_effect / "src" / "omnibase_infra" / "runtime" / "auto_wiring"
    package.mkdir(parents=True)
    module = package / _IO_MODULE.name
    shutil.copy(_IO_MODULE, module)
    local_contract = package / "contract.yaml"
    shutil.copy(_AUTO_WIRING / "contract.yaml", local_contract)

    validator = ValidatorNoIoOutsideEffects(root=tmp_path)
    control = validator.validate([local_contract, module])
    assert control.is_valid, control.issues
    local_contract.unlink()

    result = validator.validate([module])

    assert not result.is_valid, result.issues
    assert {issue.code for issue in result.issues} == {
        "env_read",
        "clock",
        "import_database",
        "file_read",
        "direct_bus",
        "direct_adapter",
    }
    assert all(
        issue.file_path == module.relative_to(tmp_path) for issue in result.issues
    )


@pytest.mark.unit
@pytest.mark.parametrize("placement", ["sibling", "nested"])
def test_non_effect_components_keep_their_io_checks(
    tmp_path: Path, placement: str
) -> None:
    effect = tmp_path / "auto_wiring"
    effect.mkdir()
    shutil.copy(_AUTO_WIRING / "contract.yaml", effect / "contract.yaml")
    component = (effect if placement == "nested" else tmp_path) / "compute"
    component.mkdir()
    (component / "contract.yaml").write_text(
        "node_type: COMPUTE_GENERIC\ndescriptor:\n  node_archetype: compute\n"
    )
    module = component / "probe.py"
    module.write_text("import os\nvalue = os.environ.get('ONEX_PROBE')\n")

    result = ValidatorNoIoOutsideEffects(root=tmp_path).validate([module])

    assert not result.is_valid, result.issues
    assert [
        (issue.code, issue.file_path, issue.line_number) for issue in result.issues
    ] == [("env_read", module.relative_to(tmp_path), 2)]
