# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime handler for contract validation compute (canonical definition B)."""

from __future__ import annotations

import ast
import re

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_core.services.service_contract_validator import ServiceContractValidator
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)


def _namespace_violations(topics: tuple[str, ...]) -> list[str]:
    """Refuse a single consumer set containing physical and canonical names."""
    bare: list[str] = []
    prefixed: list[str] = []
    for topic in topics:
        if re.fullmatch(r"onex\.(?:evt|cmd|dlq|intent|snapshot)\..+\.v[1-9]\d*", topic):
            bare.append(topic)
        elif re.fullmatch(
            r"[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)*\.onex\."
            r"(?:evt|cmd|dlq|intent|snapshot)\..+\.v[1-9]\d*",
            topic,
        ):
            prefixed.append(topic)
    if bare and prefixed:
        return [
            "mixed consumer topic namespace: prefixed="
            f"{sorted(set(prefixed))!r}; unprefixed={sorted(set(bare))!r}"
        ]
    return []


class ConsumerTopicVisitor(ast.NodeVisitor):
    """Resolve literal consumer declarations without executing scanned source.

    Names and collection concatenation are resolved within their lexical scope.
    Dynamic expressions remain unknown; topics in separate consumers are never
    combined into a synthetic consumer set.
    """

    def __init__(self) -> None:
        self.values: dict[str, tuple[str, ...]] = {}
        self.strings: dict[str, str] = {}
        self.consumer_names = {"AIOKafkaConsumer", "KafkaConsumer", "KafkaTransport"}
        self.violations: list[str] = []

    def _topics(self, node: ast.AST) -> tuple[str, ...]:
        return self._value(node) or ()

    def _string(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.strings.get(node.id)
        if isinstance(node, ast.Attribute):
            return self.strings.get(ast.unparse(node))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._string(node.left), self._string(node.right)
            if left is not None and right is not None:
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts = [
                self._string(
                    part.value if isinstance(part, ast.FormattedValue) else part
                )
                for part in node.values
            ]
            if all(part is not None for part in parts):
                return "".join(part for part in parts if part is not None)
        return None

    def _value(self, node: ast.AST) -> tuple[str, ...] | None:
        scalar = self._string(node)
        if scalar is not None:
            return (scalar,)
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Attribute):
            return self.values.get(ast.unparse(node))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return tuple(topic for item in node.elts for topic in self._topics(item))
        if isinstance(node, ast.Starred):
            return self._value(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._value(node.left), self._value(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in self.consumer_names:
                self.consumer_names.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._value(node.value)
        scalar = self._string(node.value)
        for target in node.targets:
            if isinstance(target, (ast.Name, ast.Attribute)):
                key = ast.unparse(target)
                self.values.pop(key, None)
                self.strings.pop(key, None)
                if scalar is not None:
                    self.strings[key] = scalar
                elif value is not None:
                    self.values[key] = value

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if isinstance(node.target, (ast.Name, ast.Attribute)):
                key = ast.unparse(node.target)
                value = self._value(node.value)
                scalar = self._string(node.value)
                self.values.pop(key, None)
                self.strings.pop(key, None)
                if scalar is not None:
                    self.strings[key] = scalar
                elif value is not None:
                    self.values[key] = value

    def _visit_scope(
        self,
        node: ast.AST,
    ) -> None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        outer = self.values
        outer_strings = self.strings
        self.values = outer.copy()
        self.strings = outer_strings.copy()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for argument in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                self.values.pop(argument.arg, None)
                self.strings.pop(argument.arg, None)
        for statement in node.body:
            self.visit(statement)
        self.values = outer
        self.strings = outer_strings

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._visit_scope(node)
        else:
            super().visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else (function.attr if isinstance(function, ast.Attribute) else "")
        )
        factory = (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in self.consumer_names
            and name == "from_bootstrap"
        )
        if name in self.consumer_names or name == "subscribe" or factory:
            topics = tuple(
                topic
                for keyword in node.keywords
                if keyword.arg == "topics"
                for topic in self._topics(keyword.value)
            )
            if name in self.consumer_names - {"KafkaTransport"} or name == "subscribe":
                topics += tuple(
                    topic for arg in node.args for topic in self._topics(arg)
                )
            self.violations.extend(
                f"line {node.lineno}: {violation}"
                for violation in _namespace_violations(topics)
            )
        self.generic_visit(node)


class HandlerContractValidate:
    """Canonical def-B handler for deterministic contract validation."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelContractValidateInput
    ) -> ModelContractValidationResult:
        """Validate a contract through the runtime compute-node boundary."""
        if request.consumer_topics is not None or request.consumer_source is not None:
            if request.consumer_source is not None:
                visitor = ConsumerTopicVisitor()
                try:
                    visitor.visit(ast.parse(request.consumer_source))
                    violations = visitor.violations
                except SyntaxError as exc:
                    violations = [f"could not parse consumer declarations: {exc}"]
            else:
                assert request.consumer_topics is not None
                violations = _namespace_violations(request.consumer_topics)
            return ModelContractValidationResult(
                is_valid=not violations,
                score=0.0 if violations else 1.0,
                violations=violations,
                interface_version=ModelSemVer(major=1, minor=1, patch=0),
            )
        validator = ServiceContractValidator()
        if request.model_code is not None:
            assert request.contract_content is not None
            return validator.validate_model_compliance(
                request.model_code,
                request.contract_content,
            )
        if request.file_path is not None:
            return validator.validate_contract_file(
                request.file_path,
                request.contract_type,
                request.base_dir,
            )
        assert request.contract_content is not None
        return validator.validate_contract_yaml(
            request.contract_content,
            request.contract_type,
        )


__all__ = ["HandlerContractValidate"]
