# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""src/omnibase_infra/handlers/ declares the effect archetype for K4 (OMN-17427).

The check-no-io-outside-effects hook judges a module against the nearest
contract.yaml above it. With no contract above the infra-owned handler package,
its nodeless-module fallback judged every handler there as non-EFFECT, so an
I/O import in a bound handler failed a rule the package is exempt from.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_no_io_outside_effects import (
    ValidatorNoIoOutsideEffects,
)

_HANDLERS = Path("src/omnibase_infra/handlers")
_IO_HANDLER = _HANDLERS / "handler_http.py"


@pytest.mark.unit
def test_handlers_contract_declares_the_effect_archetype() -> None:
    contract = yaml.safe_load((_HANDLERS / "contract.yaml").read_text())

    assert contract["node_type"] == "EFFECT_GENERIC"
    assert contract["descriptor"]["node_archetype"] == "effect"


@pytest.mark.unit
def test_handler_with_io_import_passes_under_the_handlers_contract() -> None:
    result = ValidatorNoIoOutsideEffects().validate(
        [_HANDLERS / "contract.yaml", _IO_HANDLER]
    )

    assert result.is_valid, result.issues


@pytest.mark.unit
def test_same_handler_fails_without_the_archetype_declaration(tmp_path: Path) -> None:
    handlers = tmp_path / "src" / "omnibase_infra" / "handlers"
    handlers.mkdir(parents=True)
    module = handlers / _IO_HANDLER.name
    shutil.copy(_IO_HANDLER, module)

    result = ValidatorNoIoOutsideEffects().validate([module])

    assert not result.is_valid
