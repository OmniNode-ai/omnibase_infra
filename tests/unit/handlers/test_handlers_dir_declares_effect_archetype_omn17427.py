# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""src/omnibase_infra/handlers/ declares the effect archetype for K4 (OMN-17427).

The check-no-io-outside-effects hook judges a module against the nearest
contract.yaml above it. With no contract above the infra-owned handler package,
its nodeless-module fallback judged every handler there as non-EFFECT, so an
I/O import in a bound handler failed a rule the package is exempt from.
"""

import ast
import importlib
import shutil
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.validation.model_canonical_file_shape_text import (
    ModelCanonicalFileShapeText,
)
from omnibase_core.models.validation.model_canonical_file_shape_tree import (
    ModelCanonicalFileShapeTree,
)
from omnibase_core.validation.checker_canonical_file_shape import (
    file_rule_refusals,
    load_rule_set,
)
from omnibase_core.validation.validator_canonical_file_shape import (
    ValidatorCanonicalFileShape,
)
from omnibase_core.validation.validator_no_io_outside_effects import (
    ValidatorNoIoOutsideEffects,
)
from omnibase_infra.handlers.placement.model_placement_command import (
    ModelPlacementCommand,
)

_HANDLERS = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "handlers"
_COMPUTE_CONTRACT = (
    "node_type: COMPUTE_GENERIC\ndescriptor:\n  node_archetype: compute\n"
)


@pytest.mark.unit
def test_epoch_guard_contract_binds_an_executable_compute_handler() -> None:
    package = _HANDLERS / "placement" / "handler_placement_epoch_guard"
    contract_path = package / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    binding = contract["handler"]
    handler_class = getattr(importlib.import_module(binding["module"]), binding["name"])
    guard = handler_class()
    new = ModelPlacementCommand(node_key="node_a", placement_epoch=2, runtime_key="new")
    old = new.model_copy(update={"placement_epoch": 1, "runtime_key": "old"})

    assert guard.accept(new)
    assert not guard.accept(old)
    assert guard.highest("node_a") == 2
    assert guard.accept(new)
    assert guard.accept(old.model_copy(update={"node_key": "node_b"}))
    assert contract["descriptor"]["node_archetype"] == "compute"

    root = _HANDLERS.parents[2]
    module_path = (package / "handler.py").relative_to(root).as_posix()
    contract_name = contract_path.relative_to(root).as_posix()
    rules = load_rule_set(ValidatorCanonicalFileShape().contract)

    def refusals(document: dict[str, object]) -> dict[str, tuple[str, str]]:
        tree = ModelCanonicalFileShapeTree(
            files=(module_path, contract_name),
            handler_class_paths=(module_path,),
            contract_texts=(
                ModelCanonicalFileShapeText(
                    path=contract_name, text=yaml.safe_dump(document)
                ),
            ),
        )
        return file_rule_refusals(tree, rules)

    assert module_path not in refusals(contract)
    del contract["handler"]
    assert refusals(contract)[module_path][0] == "handler_outside_node"


@pytest.mark.unit
def test_handlers_contract_declares_the_effect_archetype() -> None:
    contract = yaml.safe_load((_HANDLERS / "contract.yaml").read_text())

    assert contract["node_type"] == "EFFECT_GENERIC"
    assert contract["descriptor"]["node_archetype"] == "effect"


@pytest.mark.unit
def test_same_handler_fails_without_the_archetype_declaration(tmp_path: Path) -> None:
    # An outer EFFECT must not make the negative control pass accidentally.
    shutil.copy(_HANDLERS / "contract.yaml", tmp_path / "contract.yaml")
    non_effect = tmp_path / "non_effect"
    non_effect.mkdir()
    (non_effect / "contract.yaml").write_text(_COMPUTE_CONTRACT)
    handlers = non_effect / "src" / "omnibase_infra" / "handlers"
    handlers.mkdir(parents=True)
    module = handlers / "handler_probe.py"
    module.write_text("import httpx\n")
    local_contract = handlers / "contract.yaml"
    shutil.copy(_HANDLERS / "contract.yaml", local_contract)

    validator = ValidatorNoIoOutsideEffects(root=tmp_path)
    control = validator.validate([local_contract, module])
    assert control.is_valid, control.issues
    local_contract.unlink()

    result = validator.validate([module])

    assert not result.is_valid, result.issues
    assert [(issue.code, issue.file_path) for issue in result.issues] == [
        ("import_network", module.relative_to(tmp_path))
    ]


@pytest.mark.unit
@pytest.mark.parametrize("placement", ["sibling", "nested"])
def test_non_effect_components_keep_their_io_checks(
    tmp_path: Path, placement: str
) -> None:
    handlers = tmp_path / "handlers"
    handlers.mkdir()
    shutil.copy(_HANDLERS / "contract.yaml", handlers / "contract.yaml")
    compute = (handlers if placement == "nested" else tmp_path) / "compute"
    compute.mkdir()
    (compute / "contract.yaml").write_text(_COMPUTE_CONTRACT)
    module = compute / "probe.py"
    module.write_text("import httpx\n")

    result = ValidatorNoIoOutsideEffects(root=tmp_path).validate([module])

    assert not result.is_valid, result.issues
    assert [(issue.code, issue.file_path) for issue in result.issues] == [
        ("import_network", module.relative_to(tmp_path))
    ]


@pytest.mark.unit
def test_contract_location_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    test_handlers_contract_declares_the_effect_archetype()


@pytest.mark.unit
def test_declared_compute_handlers_do_not_inherit_effect(
    tmp_path: Path,
) -> None:
    compute_modules: list[Path] = []
    for module in _HANDLERS.rglob("*.py"):
        tree = ast.parse(module.read_text())
        if any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "EnumHandlerTypeCategory"
            and node.value.attr in {"COMPUTE", "NONDETERMINISTIC_COMPUTE"}
            for node in ast.walk(tree)
        ):
            compute_modules.append(module)

    assert compute_modules, "positive control: the package contains compute handlers"
    for index, module in enumerate(sorted(compute_modules)):
        contract_path = next(
            parent / "contract.yaml"
            for parent in module.parents
            if (parent / "contract.yaml").is_file()
        )
        contract = yaml.safe_load(contract_path.read_text())
        assert contract["descriptor"]["node_archetype"] != "effect", module
        package = tmp_path / f"compute_{index}"
        package.mkdir()
        shutil.copy(contract_path, package / "contract.yaml")
        probe = package / "handler_probe.py"
        probe.write_text("import httpx\n")
        result = ValidatorNoIoOutsideEffects(root=tmp_path).validate([probe])
        assert not result.is_valid, module
        assert [issue.code for issue in result.issues] == ["import_network"]
