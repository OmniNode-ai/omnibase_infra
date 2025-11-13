#!/usr/bin/env python3
"""
Node Validator for ONEX v2.0 Mixin-Enhanced Nodes.

Validates generated nodes through 6 comprehensive stages:
1. Syntax validation - Python compile check (<10ms)
2. AST validation - Structure and method signatures (<20ms)
3. Import resolution - Verify imports can be resolved (<50ms)
4. Type checking - Optional mypy integration (1-3s)
5. ONEX compliance - Mixin verification and patterns (<100ms)
6. Security scanning - Dangerous pattern detection (<100ms)

Performance targets: <200ms without type checking, <3s with type checking.

Example:
    >>> validator = NodeValidator(enable_type_checking=False)
    >>> results = await validator.validate_generated_node(
    ...     node_file_content=generated_code,
    ...     contract=enhanced_contract
    ... )
    >>> for result in results:
    ...     print(result)
    ...     if not result.passed:
    ...         print(f"  Errors: {result.errors}")
"""

import ast
import importlib
import importlib.util
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..models_contract import ModelEnhancedContract
from .models import EnumValidationStage, ModelValidationResult

logger = logging.getLogger(__name__)


class NodeValidator:
    """
    Comprehensive validator for mixin-enhanced ONEX v2.0 nodes.

    Runs multiple validation stages to ensure generated nodes are:
    - Syntactically correct
    - Structurally sound (proper class/method signatures)
    - Import-resolvable (all imports available)
    - Type-safe (optional mypy checking)
    - ONEX compliant (mixins present, proper patterns)
    - Secure (no dangerous patterns)

    Performance optimized:
    - Fast stages run first (fail-fast)
    - Expensive stages (mypy) optional
    - Parallel validation where possible

    Example:
        >>> validator = NodeValidator(
        ...     enable_type_checking=False,
        ...     enable_security_scan=True
        ... )
        >>> results = await validator.validate_generated_node(
        ...     node_file_content=code,
        ...     contract=contract
        ... )
        >>> passed = all(r.passed for r in results)
    """

    # Security pattern definitions
    DANGEROUS_PATTERNS = [
        (r"\beval\s*\(", "eval() call detected - security risk"),
        (r"\bexec\s*\(", "exec() call detected - security risk"),
        (r"\b__import__\s*\(", "__import__() call detected - use regular imports"),
        (r"os\.system\s*\(", "os.system() call detected - use subprocess instead"),
        (
            r"subprocess\..*shell\s*=\s*True",
            "subprocess with shell=True - potential command injection",
        ),
    ]

    SUSPICIOUS_PATTERNS = [
        (r"\bpickle\.", "pickle usage - consider safer alternatives"),
        (r"yaml\.load\s*\((?!.*Loader=)", "unsafe yaml.load() - use yaml.safe_load()"),
    ]

    SECRET_PATTERNS = [
        (r"password\s*=\s*['\"](?!{|<)", "Hardcoded password detected"),
        (r"api[_-]?key\s*=\s*['\"](?!{|<)", "Hardcoded API key detected"),
        (r"secret\s*=\s*['\"](?!{|<)", "Hardcoded secret detected"),
        (r"token\s*=\s*['\"](?!{|<)", "Hardcoded token detected"),
    ]

    def __init__(
        self,
        enable_type_checking: bool = False,
        enable_security_scan: bool = True,
        mypy_config_path: Optional[Path] = None,
    ):
        """
        Initialize node validator.

        Args:
            enable_type_checking: Enable mypy type checking (slower, ~1-3s)
            enable_security_scan: Enable security pattern scanning (recommended)
            mypy_config_path: Path to mypy config file (optional)
        """
        self.enable_type_checking = enable_type_checking
        self.enable_security_scan = enable_security_scan
        self.mypy_config_path = mypy_config_path
        self.logger = logging.getLogger(__name__)

    async def validate_generated_node(
        self,
        node_file_content: str,
        contract: ModelEnhancedContract,
    ) -> list[ModelValidationResult]:
        """
        Run all validation stages on generated node.

        Executes stages in optimized order (fast-fail):
        1. Syntax (fastest, most critical)
        2. AST (fast, required for later stages)
        3. Imports (medium speed)
        4. ONEX compliance (medium speed)
        5. Security (fast)
        6. Type checking (slowest, optional)

        Args:
            node_file_content: Generated node.py file contents
            contract: Contract used for generation (with mixins)

        Returns:
            List of validation results, one per stage

        Example:
            >>> validator = NodeValidator()
            >>> results = await validator.validate_generated_node(code, contract)
            >>> if all(r.passed for r in results):
            ...     print("✅ All validation stages passed")
        """
        results: list[ModelValidationResult] = []

        # Handle None contract gracefully
        if contract is None:
            self.logger.warning("Contract is None - skipping validation")
            return [
                ModelValidationResult(
                    stage=EnumValidationStage.ONEX_COMPLIANCE,
                    passed=False,
                    errors=["Contract is None - cannot validate"],
                    warnings=[],
                )
            ]

        self.logger.info(
            f"Starting validation for {contract.name} "
            f"({len(node_file_content)} chars, {len(contract.mixins)} mixins)"
        )

        # Stage 1: Syntax validation (fast, critical)
        syntax_result = await self._validate_syntax(node_file_content)
        results.append(syntax_result)

        # If syntax fails, can't continue with AST-based validation
        if not syntax_result.passed:
            self.logger.warning("Syntax validation failed, skipping AST-based stages")
            return results

        # Stage 2: AST validation (requires valid syntax)
        ast_result = await self._validate_ast(node_file_content, contract)
        results.append(ast_result)

        # Stage 3: Import resolution
        import_result = await self._validate_imports(node_file_content)
        results.append(import_result)

        # Stage 4: ONEX compliance (mixin verification)
        onex_result = await self._validate_onex_compliance(node_file_content, contract)
        results.append(onex_result)

        # Stage 5: Security scan (if enabled)
        if self.enable_security_scan:
            security_result = await self._validate_security(node_file_content)
            results.append(security_result)

        # Stage 6: Type checking (optional, slow)
        if self.enable_type_checking:
            type_result = await self._validate_types(node_file_content)
            results.append(type_result)

        # Log summary
        passed_count = sum(1 for r in results if r.passed)
        total_time = sum(r.execution_time_ms for r in results)
        self.logger.info(
            f"Validation complete: {passed_count}/{len(results)} passed, "
            f"{total_time:.1f}ms total"
        )

        return results

    async def _validate_syntax(self, code: str) -> ModelValidationResult:
        """
        Validate Python syntax using compile().

        Fastest validation stage (~1-10ms). Catches all syntax errors:
        - SyntaxError
        - IndentationError
        - TabError

        Args:
            code: Source code to validate

        Returns:
            ModelValidationResult with syntax errors if any
        """
        start = time.time()
        errors = []
        suggestions = []

        try:
            compile(code, "<generated>", "exec")
            self.logger.debug("✅ Syntax validation passed")
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n    {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n    {' ' * (e.offset - 1)}^"
            errors.append(error_msg)
            suggestions.append(
                "Fix syntax error before proceeding with other validation stages"
            )
            self.logger.error(f"❌ Syntax error: {error_msg}")
        except IndentationError as e:
            error_msg = f"Line {e.lineno}: Indentation error - {e.msg}"
            errors.append(error_msg)
            suggestions.append("Check indentation consistency (use 4 spaces, not tabs)")
        except TabError as e:
            error_msg = f"Line {e.lineno}: Tab/space mixing - {e.msg}"
            errors.append(error_msg)
            suggestions.append("Convert all indentation to spaces (4 spaces per level)")
        except Exception as e:
            errors.append(f"Unexpected syntax validation error: {e!s}")
            self.logger.exception("Unexpected syntax validation error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.SYNTAX,
            passed=len(errors) == 0,
            errors=errors,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    async def _validate_ast(
        self, code: str, contract: ModelEnhancedContract
    ) -> ModelValidationResult:
        """
        Validate AST structure and required methods.

        Checks for:
        - Valid class definition
        - Required methods (__init__, initialize, shutdown)
        - Async method markers
        - Proper method signatures

        Performance: <20ms

        Args:
            code: Source code (must be syntactically valid)
            contract: Contract with node type information

        Returns:
            ModelValidationResult with structural errors if any
        """
        start = time.time()
        errors = []
        warnings = []
        suggestions = []

        try:
            tree = ast.parse(code)

            # Find main node class
            node_class = self._extract_class_from_ast(tree, contract.name)
            if not node_class:
                errors.append(f"Class '{contract.name}' not found in generated code")
                suggestions.append(f"Ensure class is named exactly '{contract.name}'")
                return ModelValidationResult(
                    stage=EnumValidationStage.AST,
                    passed=False,
                    errors=errors,
                    execution_time_ms=(time.time() - start) * 1000,
                    suggestions=suggestions,
                )

            # Extract methods
            methods = self._extract_methods_from_class(node_class)
            method_names = {m.name for m in methods}
            async_methods = {
                m.name for m in methods if isinstance(m, ast.AsyncFunctionDef)
            }

            # Check required methods
            if "__init__" not in method_names:
                errors.append("Missing required method: __init__(self, container)")
                suggestions.append(
                    "Add: def __init__(self, container: ModelContainer):\n"
                    "         super().__init__(container)"
                )

            if "initialize" not in method_names:
                errors.append("Missing required method: async def initialize(self)")
                suggestions.append(
                    "Add: async def initialize(self) -> None:\n"
                    "         await super().initialize()"
                )
            elif "initialize" not in async_methods:
                errors.append("Method 'initialize' must be async")
                suggestions.append("Change 'def initialize' to 'async def initialize'")

            if "shutdown" not in method_names:
                warnings.append("Missing recommended method: async def shutdown(self)")
                suggestions.append(
                    "Consider adding: async def shutdown(self) -> None:\n"
                    "                     await super().shutdown()"
                )
            elif "shutdown" not in async_methods:
                warnings.append("Method 'shutdown' should be async")

            # Check execute method (based on node type)
            expected_execute = f"execute_{contract.node_type.lower()}"
            if expected_execute not in method_names:
                errors.append(
                    f"Missing execute method: async def {expected_execute}(...)"
                )
                suggestions.append(
                    f"Add execute method for {contract.node_type} node type"
                )
            elif expected_execute not in async_methods:
                errors.append(f"Method '{expected_execute}' must be async")

            # Check __init__ signature
            if "__init__" in method_names:
                init_method = next(m for m in methods if m.name == "__init__")
                if not self._check_init_signature(init_method):
                    warnings.append(
                        "__init__ signature should be: def __init__(self, container: ModelContainer)"
                    )

            self.logger.debug(
                f"✅ AST validation: {len(method_names)} methods, "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )

        except SyntaxError:
            # Should not happen (syntax validated in previous stage)
            errors.append(
                "AST parsing failed (unexpected - syntax stage should catch this)"
            )
        except Exception as e:
            errors.append(f"AST validation error: {e!s}")
            self.logger.exception("AST validation error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.AST,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    async def _validate_imports(self, code: str) -> ModelValidationResult:
        """
        Validate that all imports can be resolved.

        Checks:
        - All imported modules exist
        - All imported names exist in modules
        - Special handling for omnibase_core (allowed even if not installed)

        Performance: <50ms (may be slower if imports trigger module loading)

        Args:
            code: Source code

        Returns:
            ModelValidationResult with import errors if any
        """
        start = time.time()
        errors = []
        warnings = []
        suggestions = []

        try:
            tree = ast.parse(code)

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, None))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append((module, alias.name))

            # Validate each import
            for module_name, import_name in imports:
                # Skip omnibase_core (will be available at runtime)
                if module_name.startswith("omnibase_core"):
                    self.logger.debug(f"Skipping omnibase_core import: {module_name}")
                    continue

                # Try to import module
                try:
                    if module_name:
                        spec = importlib.util.find_spec(module_name)
                        if spec is None:
                            warnings.append(
                                f"Module '{module_name}' not found (may be available at runtime)"
                            )
                        elif import_name:
                            # Try to import specific name
                            try:
                                module = importlib.import_module(module_name)
                                if not hasattr(module, import_name):
                                    warnings.append(
                                        f"'{import_name}' not found in module '{module_name}'"
                                    )
                            except ImportError:
                                warnings.append(
                                    f"Cannot import '{import_name}' from '{module_name}'"
                                )
                except (ImportError, ModuleNotFoundError, ValueError):
                    warnings.append(
                        f"Module '{module_name}' not available (may be OK if installed at runtime)"
                    )

            # Import warnings are not critical (dependencies may not be installed in dev env)
            if warnings:
                suggestions.append(
                    "Ensure all dependencies are listed in pyproject.toml or requirements.txt"
                )

            self.logger.debug(
                f"✅ Import validation: {len(imports)} imports, {len(warnings)} warnings"
            )

        except SyntaxError:
            errors.append("Cannot parse imports (syntax error)")
        except Exception as e:
            errors.append(f"Import validation error: {e!s}")
            self.logger.exception("Import validation error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.IMPORTS,
            passed=len(errors) == 0,  # Only fail on errors, not warnings
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    async def _validate_onex_compliance(
        self, code: str, contract: ModelEnhancedContract
    ) -> ModelValidationResult:
        """
        Validate ONEX v2.0 compliance and mixin presence.

        Checks:
        - Inherits from NodeEffect (or appropriate base)
        - All declared mixins present in inheritance chain
        - No duplicate mixins
        - Proper super().__init__() call
        - Proper await super().initialize() call
        - No built-in duplication (e.g., don't implement circuit breaker if using NodeEffect's)

        Performance: <100ms

        Args:
            code: Source code
            contract: Contract with mixin declarations

        Returns:
            ModelValidationResult with compliance errors if any
        """
        start = time.time()
        errors = []
        warnings = []
        suggestions = []

        try:
            tree = ast.parse(code)

            # Find main class
            node_class = self._extract_class_from_ast(tree, contract.name)
            if not node_class:
                errors.append("Cannot validate ONEX compliance - class not found")
                return ModelValidationResult(
                    stage=EnumValidationStage.ONEX_COMPLIANCE,
                    passed=False,
                    errors=errors,
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # Check base class inheritance
            base_class_name = f"Node{contract.node_type.capitalize()}"
            has_base_class = any(
                self._get_name_from_node(base) == base_class_name
                for base in node_class.bases
            )
            if not has_base_class:
                errors.append(f"Class must inherit from {base_class_name}")
                suggestions.append(
                    f"class {contract.name}({base_class_name}, ...):\n" f"    ..."
                )

            # Extract declared mixins from inheritance
            inherited_names = [
                self._get_name_from_node(base) for base in node_class.bases
            ]

            # Check each declared mixin is present
            for mixin_decl in contract.get_enabled_mixins():
                mixin_name = mixin_decl.name
                if mixin_name not in inherited_names:
                    errors.append(
                        f"Declared mixin '{mixin_name}' not found in inheritance chain"
                    )
                    suggestions.append(
                        f"Add {mixin_name} to class inheritance:\n"
                        f"class {contract.name}({base_class_name}, {mixin_name}, ...):"
                    )

            # Check for duplicate mixins
            mixin_counts = {}
            for name in inherited_names:
                if name.startswith("Mixin") or name.startswith("mixin_"):
                    mixin_counts[name] = mixin_counts.get(name, 0) + 1

            for mixin, count in mixin_counts.items():
                if count > 1:
                    errors.append(
                        f"Duplicate mixin '{mixin}' in inheritance chain ({count} times)"
                    )

            # Check __init__ for super().__init__(container)
            methods = self._extract_methods_from_class(node_class)
            init_method = next((m for m in methods if m.name == "__init__"), None)
            if init_method:
                has_super_init = self._check_super_call(init_method, "__init__")
                if not has_super_init:
                    errors.append(
                        "Missing super().__init__(container) call in __init__"
                    )
                    suggestions.append(
                        "Add super().__init__(container) as first line in __init__"
                    )

            # Check initialize for await super().initialize()
            init_async = next((m for m in methods if m.name == "initialize"), None)
            if init_async:
                has_super_initialize = self._check_super_call(init_async, "initialize")
                if not has_super_initialize:
                    warnings.append(
                        "Missing await super().initialize() call in initialize()"
                    )
                    suggestions.append(
                        "Add await super().initialize() as first line in initialize()"
                    )

            # Check for NodeEffect built-in duplication
            # (e.g., don't implement circuit breaker if NodeEffect has it)
            builtin_features = ["circuit_breaker", "retry", "health_check"]
            for feature in builtin_features:
                if f"def {feature}" in code or f"async def {feature}" in code:
                    warnings.append(
                        f"Custom implementation of '{feature}' detected - "
                        f"NodeEffect provides this via configuration"
                    )

            self.logger.debug(
                f"✅ ONEX compliance: {len(contract.mixins)} mixins, "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )

        except SyntaxError:
            errors.append("Cannot validate ONEX compliance (syntax error)")
        except Exception as e:
            errors.append(f"ONEX compliance validation error: {e!s}")
            self.logger.exception("ONEX compliance validation error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.ONEX_COMPLIANCE,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    async def _validate_security(self, code: str) -> ModelValidationResult:
        """
        Validate code for security issues.

        Checks for:
        - Dangerous patterns (eval, exec, __import__)
        - Unsafe os.system() calls
        - subprocess with shell=True
        - Hardcoded secrets (passwords, API keys)
        - Suspicious patterns (pickle, unsafe yaml.load)

        Performance: <100ms

        Args:
            code: Source code

        Returns:
            ModelValidationResult with security issues if any
        """
        start = time.time()
        errors = []
        warnings = []
        suggestions = []

        try:
            # Check dangerous patterns (critical)
            for pattern, message in self.DANGEROUS_PATTERNS:
                matches = list(re.finditer(pattern, code))
                for match in matches:
                    line_num = code[: match.start()].count("\n") + 1
                    errors.append(f"Line {line_num}: {message}")

            # Check secret patterns (critical)
            for pattern, message in self.SECRET_PATTERNS:
                matches = list(re.finditer(pattern, code, re.IGNORECASE))
                for match in matches:
                    line_num = code[: match.start()].count("\n") + 1
                    errors.append(f"Line {line_num}: {message}")

            # Check suspicious patterns (warnings)
            for pattern, message in self.SUSPICIOUS_PATTERNS:
                matches = list(re.finditer(pattern, code))
                for match in matches:
                    line_num = code[: match.start()].count("\n") + 1
                    warnings.append(f"Line {line_num}: {message}")

            # Suggestions based on findings
            if errors:
                suggestions.append("Remove dangerous patterns and hardcoded secrets")
                suggestions.append(
                    "Use environment variables or secret management for credentials"
                )
            if "subprocess" in code:
                suggestions.append(
                    "When using subprocess, avoid shell=True and validate inputs"
                )

            self.logger.debug(
                f"✅ Security scan: {len(errors)} errors, {len(warnings)} warnings"
            )

        except Exception as e:
            errors.append(f"Security validation error: {e!s}")
            self.logger.exception("Security validation error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.SECURITY,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    async def _validate_types(self, code: str) -> ModelValidationResult:
        """
        Validate type hints using mypy.

        Optional stage (slow: 1-3s). Requires mypy to be installed.

        Args:
            code: Source code

        Returns:
            ModelValidationResult with type errors if any
        """
        start = time.time()
        errors = []
        warnings = []
        suggestions = []

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name

            try:
                # Run mypy
                cmd = ["mypy", "--strict", "--no-error-summary", tmp_path]
                if self.mypy_config_path:
                    cmd.insert(1, f"--config-file={self.mypy_config_path}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    # Parse mypy output
                    for line in result.stdout.strip().split("\n"):
                        if "error:" in line.lower():
                            # Extract error (remove temp file path)
                            error_msg = line.split("error:", 1)[1].strip()
                            errors.append(f"Type error: {error_msg}")
                        elif "note:" in line.lower():
                            note_msg = line.split("note:", 1)[1].strip()
                            warnings.append(f"Type note: {note_msg}")

                if errors:
                    suggestions.append("Add type hints to all function signatures")
                    suggestions.append(
                        "Use -> None for functions that don't return values"
                    )

                self.logger.debug(
                    f"✅ Type checking: {len(errors)} errors, {len(warnings)} notes"
                )

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            warnings.append("Type checking timed out (>30s) - skipping")
        except FileNotFoundError:
            warnings.append("mypy not found - install with: pip install mypy")
        except Exception as e:
            warnings.append(f"Type checking failed: {e!s}")
            self.logger.exception("Type checking error")

        execution_time = (time.time() - start) * 1000

        return ModelValidationResult(
            stage=EnumValidationStage.TYPE_CHECKING,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            suggestions=suggestions,
        )

    # ===== Helper Methods =====

    def _extract_class_from_ast(
        self, tree: ast.Module, class_name: str
    ) -> Optional[ast.ClassDef]:
        """
        Extract specific class definition from AST.

        Args:
            tree: Parsed AST
            class_name: Name of class to find

        Returns:
            ClassDef node if found, None otherwise
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _extract_methods_from_class(
        self, class_node: ast.ClassDef
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """
        Extract all method definitions from class.

        Args:
            class_node: ClassDef node

        Returns:
            List of method nodes
        """
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(node)
        return methods

    def _check_init_signature(self, init_method: ast.FunctionDef) -> bool:
        """
        Check if __init__ has correct signature.

        Expected: def __init__(self, container: ModelContainer)

        Args:
            init_method: __init__ method node

        Returns:
            True if signature is correct
        """
        # Should have 2 args: self, container
        if len(init_method.args.args) != 2:
            return False

        # Second arg should be named 'container'
        if init_method.args.args[1].arg != "container":
            return False

        return True

    def _check_super_call(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef, method_name: str
    ) -> bool:
        """
        Check if method contains super().method_name() call.

        Args:
            method: Method node to check
            method_name: Name of super method to look for

        Returns:
            True if super call found
        """
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                # Check for super().method_name() pattern
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Call)
                        and isinstance(node.func.value.func, ast.Name)
                        and node.func.value.func.id == "super"
                        and node.func.attr == method_name
                    ):
                        return True
        return False

    def _get_name_from_node(self, node: ast.expr) -> str:
        """
        Extract name from AST node (handles Name and Attribute nodes).

        Args:
            node: AST expression node

        Returns:
            Name string (e.g., "NodeEffect", "MixinHealthCheck")
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""


__all__ = ["NodeValidator"]
