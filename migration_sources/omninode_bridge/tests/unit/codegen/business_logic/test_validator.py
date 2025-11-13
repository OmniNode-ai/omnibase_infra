#!/usr/bin/env python3
"""
Unit tests for CodeValidator.

Tests all validation aspects:
- Syntax validation
- ONEX compliance
- Type hints
- Security checks
- Code quality
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Setup path to avoid importing codegen/__init__.py which requires omnibase_core
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Load validation_rules first
rules_path = (
    src_path / "omninode_bridge" / "codegen" / "business_logic" / "validation_rules.py"
)
spec_rules = importlib.util.spec_from_file_location(
    "omninode_bridge.codegen.business_logic.validation_rules", rules_path
)
rules_module = importlib.util.module_from_spec(spec_rules)
sys.modules["omninode_bridge.codegen.business_logic.validation_rules"] = rules_module
spec_rules.loader.exec_module(rules_module)

# Now load validator (it will find validation_rules in sys.modules)
validator_path = (
    src_path / "omninode_bridge" / "codegen" / "business_logic" / "validator.py"
)
spec_validator = importlib.util.spec_from_file_location(
    "omninode_bridge.codegen.business_logic.validator", validator_path
)
validator_module = importlib.util.module_from_spec(spec_validator)
sys.modules["omninode_bridge.codegen.business_logic.validator"] = validator_module
spec_validator.loader.exec_module(validator_module)

CodeValidator = validator_module.CodeValidator
GenerationContext = validator_module.GenerationContext
ValidationResult = validator_module.ValidationResult


class TestCodeValidator:
    """Test suite for CodeValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CodeValidator(strict=True)

    @pytest.fixture
    def lenient_validator(self):
        """Create lenient validator instance."""
        return CodeValidator(strict=False)

    @pytest.fixture
    def context(self):
        """Create generation context."""
        return GenerationContext(
            node_type="effect",
            method_name="execute_effect",
            service_name="test_service",
        )

    # ==================== Syntax Validation Tests ====================

    @pytest.mark.asyncio
    async def test_valid_syntax(self, validator, context):
        """Test validation passes for valid Python syntax."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert result.syntax_valid
        assert len(result.syntax_errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, validator, context):
        """Test validation fails for invalid Python syntax."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic.
    return {"status": "success"  # Missing closing brace
'''
        result = await validator.validate(code, context)
        assert not result.syntax_valid
        assert len(result.syntax_errors) > 0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_syntax_error_with_line_number(self, validator, context):
        """Test syntax error includes line number."""
        code = """
def test():
    x =
    return x
"""
        result = await validator.validate(code, context)
        assert not result.syntax_valid
        assert any("line" in err.lower() for err in result.syntax_errors)

    # ==================== ONEX Compliance Tests ====================

    @pytest.mark.asyncio
    async def test_onex_compliant_code(self, validator, context):
        """Test validation passes for ONEX-compliant code."""
        code = '''
from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.logging.structured import emit_log_event

async def execute_effect(self, container):
    """Execute effect logic."""
    try:
        emit_log_event("info", "Processing request")
        result = await self._process()
        return result
    except Exception as e:
        raise ModelOnexError(
            EnumCoreErrorCode.PROCESSING_ERROR,
            f"Failed to process: {e}"
        )
'''
        result = await validator.validate(code, context)
        assert result.onex_compliant
        assert len(result.onex_issues) == 0

    @pytest.mark.asyncio
    async def test_missing_model_onex_error(self, validator, context):
        """Test detection of generic exceptions instead of ModelOnexError."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    if error:
        raise Exception("Something went wrong")
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.onex_compliant
        assert any("ModelOnexError" in issue for issue in result.onex_issues)

    @pytest.mark.asyncio
    async def test_missing_emit_log_event(self, validator, context):
        """Test detection of logger usage instead of emit_log_event."""
        code = '''
import logging
logger = logging.getLogger(__name__)

async def execute_effect(self, container):
    """Execute effect logic."""
    logger.info("Processing request")
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.onex_compliant
        assert any("emit_log_event" in issue for issue in result.onex_issues)

    # ==================== Type Hint Tests ====================

    @pytest.mark.asyncio
    async def test_complete_type_hints(self, validator, context):
        """Test validation passes for complete type hints."""
        code = '''
from typing import Dict, Any

async def execute_effect(self, container: Any) -> Dict[str, Any]:
    """Execute effect logic."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert result.has_type_hints
        assert len(result.type_hint_issues) == 0

    @pytest.mark.asyncio
    async def test_missing_return_type_hint(self, validator, context):
        """Test detection of missing return type hint."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.has_type_hints
        assert any("return type hint" in issue for issue in result.type_hint_issues)

    @pytest.mark.asyncio
    async def test_missing_argument_type_hint(self, validator, context):
        """Test detection of missing argument type hints."""
        code = '''
from typing import Dict, Any

async def execute_effect(self, container) -> Dict[str, Any]:
    """Execute effect logic."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.has_type_hints
        assert any("container" in issue for issue in result.type_hint_issues)

    @pytest.mark.asyncio
    async def test_magic_methods_skip_type_hints(self, validator, context):
        """Test that magic methods don't require type hints."""
        code = '''
def __init__(self):
    """Initialize."""
    self.data = {}

def __str__(self):
    """String representation."""
    return str(self.data)
'''
        result = await validator.validate(code, context)
        # Should pass because magic methods are excepted
        assert result.has_type_hints

    # ==================== Security Tests ====================

    @pytest.mark.asyncio
    async def test_secure_code(self, validator, context):
        """Test validation passes for secure code."""
        code = '''
import os

async def execute_effect(self, container):
    """Execute effect logic."""
    api_key = os.getenv("API_KEY")
    password = os.environ.get("PASSWORD")
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert result.security_clean
        assert len(result.security_issues) == 0

    @pytest.mark.asyncio
    async def test_hardcoded_password(self, validator, context):
        """Test detection of hardcoded password."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    password = "my_secret_password"
    api_key = "sk-1234567890"
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.security_clean
        assert any("password" in issue.lower() for issue in result.security_issues)

    @pytest.mark.asyncio
    async def test_sql_injection_f_string(self, validator, context):
        """Test detection of SQL injection via f-strings."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    user_id = container.get("user_id")
    query = f"SELECT * FROM users WHERE id = {user_id}"
    await conn.execute(query)
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.security_clean
        assert any("SQL injection" in issue for issue in result.security_issues)

    @pytest.mark.asyncio
    async def test_dangerous_eval(self, validator, context):
        """Test detection of dangerous eval() usage."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    code_str = container.get("code")
    result = eval(code_str)
    return {"result": result}
'''
        result = await validator.validate(code, context)
        assert not result.security_clean
        assert any("eval" in issue.lower() for issue in result.security_issues)

    @pytest.mark.asyncio
    async def test_dangerous_exec(self, validator, context):
        """Test detection of dangerous exec() usage."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    code_str = container.get("code")
    exec(code_str)
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.security_clean
        assert any("exec" in issue.lower() for issue in result.security_issues)

    # ==================== Code Quality Tests ====================

    @pytest.mark.asyncio
    async def test_good_quality_code(self, validator, context):
        """Test validation passes for good quality code."""
        code = '''
from typing import Dict, Any

async def execute_effect(self, container: Any) -> Dict[str, Any]:
    """
    Execute effect logic.

    Processes the container data and returns results.
    """
    result = await self._process(container)
    return result
'''
        result = await validator.validate(code, context)
        assert len(result.quality_issues) == 0
        assert result.complexity_score <= 10

    @pytest.mark.asyncio
    async def test_high_complexity(self, validator, context):
        """Test detection of high cyclomatic complexity."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    if x and y:
        if a or b:
            for i in range(10):
                if c:
                    while d:
                        if e and f:
                            if g or h:
                                return True
    return False
'''
        result = await validator.validate(code, context)
        assert any("complexity" in issue.lower() for issue in result.quality_issues)
        assert result.complexity_score > 10

    @pytest.mark.asyncio
    async def test_missing_docstring(self, validator, context):
        """Test detection of missing docstring."""
        code = """
async def execute_effect(self, container):
    return {"status": "success"}
"""
        result = await validator.validate(code, context)
        assert any("docstring" in issue.lower() for issue in result.quality_issues)

    @pytest.mark.asyncio
    async def test_short_docstring(self, validator, context):
        """Test detection of short docstring."""
        code = '''
async def execute_effect(self, container):
    """Short."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert any(
            "short docstring" in issue.lower() for issue in result.quality_issues
        )

    @pytest.mark.asyncio
    async def test_long_function(self, validator, context):
        """Test detection of overly long function."""
        # Generate a function with >50 lines
        lines = ["async def execute_effect(self, container):"]
        lines.append('    """Execute effect logic."""')
        for i in range(60):
            lines.append(f"    x{i} = {i}")
        lines.append('    return {"status": "success"}')
        code = "\n".join(lines)

        result = await validator.validate(code, context)
        assert any("too long" in issue.lower() for issue in result.quality_issues)
        assert result.line_count > 50

    # ==================== Quality Score Tests ====================

    @pytest.mark.asyncio
    async def test_quality_score_perfect_code(self, validator, context):
        """Test quality score for perfect code."""
        code = '''
from typing import Dict, Any
from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.logging.structured import emit_log_event

async def execute_effect(self, container: Any) -> Dict[str, Any]:
    """
    Execute effect logic with proper error handling.

    This method processes container data safely.
    """
    try:
        emit_log_event("info", "Processing request")
        result = await self._process(container)
        return result
    except Exception as e:
        raise ModelOnexError(
            EnumCoreErrorCode.PROCESSING_ERROR,
            f"Failed: {e}"
        )
'''
        result = await validator.validate(code, context)
        assert result.quality_score >= 0.9
        assert result.passed

    @pytest.mark.asyncio
    async def test_quality_score_poor_code(self, validator, context):
        """Test quality score for poor code."""
        code = """
def bad_function():
    password = "hardcoded"
    eval("dangerous")
    return x
"""
        result = await validator.validate(code, context)
        assert result.quality_score < 0.5
        assert not result.passed

    # ==================== Strict vs Lenient Mode Tests ====================

    @pytest.mark.asyncio
    async def test_strict_mode_fails_on_warnings(self, validator, context):
        """Test strict mode fails on non-critical issues."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    logger.info("test")
    return {"status": "success"}
'''
        result = await validator.validate(code, context)
        assert not result.passed  # Fails due to missing emit_log_event

    @pytest.mark.asyncio
    async def test_lenient_mode_passes_on_warnings(self, lenient_validator, context):
        """Test lenient mode passes on non-critical issues."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    logger.info("test")
    return {"status": "success"}
'''
        result = await lenient_validator.validate(code, context)
        assert result.passed  # Passes because syntax and security are OK

    @pytest.mark.asyncio
    async def test_lenient_mode_fails_on_critical(self, lenient_validator, context):
        """Test lenient mode still fails on critical issues."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    password = "hardcoded_secret"
    return {"status": "success"}
'''
        result = await lenient_validator.validate(code, context)
        assert not result.passed  # Fails due to security issue

    # ==================== Integration Tests ====================

    @pytest.mark.asyncio
    async def test_validation_result_structure(self, validator, context):
        """Test ValidationResult has all expected fields."""
        code = '''
async def execute_effect(self, container):
    """Execute effect logic."""
    return {"status": "success"}
'''
        result = await validator.validate(code, context)

        # Check all fields are present
        assert isinstance(result.passed, bool)
        assert isinstance(result.quality_score, float)
        assert 0.0 <= result.quality_score <= 1.0
        assert isinstance(result.syntax_valid, bool)
        assert isinstance(result.onex_compliant, bool)
        assert isinstance(result.has_type_hints, bool)
        assert isinstance(result.security_clean, bool)
        assert isinstance(result.issues, list)
        assert isinstance(result.complexity_score, int)
        assert isinstance(result.line_count, int)

    @pytest.mark.asyncio
    async def test_all_issues_aggregated(self, validator, context):
        """Test that all issues are aggregated in issues list."""
        code = """
def bad():
    password = "secret"
    eval("x")
    return y
"""
        result = await validator.validate(code, context)

        # All component issues should be in the main issues list
        all_component_issues = (
            result.syntax_errors
            + result.onex_issues
            + result.type_hint_issues
            + result.security_issues
            + result.quality_issues
        )
        assert len(result.issues) == len(all_component_issues)


class TestValidationRules:
    """Test validation rules module."""

    def test_hardcoded_secret_detection(self):
        """Test hardcoded secret detection."""
        # Direct import of validation_rules module
        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "omninode_bridge"
            / "codegen"
            / "business_logic"
            / "validation_rules.py"
        )
        spec = importlib.util.spec_from_file_location("validation_rules", rules_path)
        rules_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_module)

        code = """
password = "my_password"
api_key = os.getenv("API_KEY")
secret = "hardcoded_secret"
"""
        issues = rules_module.check_hardcoded_secrets(code)
        assert len(issues) >= 1
        assert any("password" in issue.lower() for issue in issues)

    def test_dangerous_patterns_detection(self):
        """Test dangerous pattern detection."""
        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "omninode_bridge"
            / "codegen"
            / "business_logic"
            / "validation_rules.py"
        )
        spec = importlib.util.spec_from_file_location("validation_rules", rules_path)
        rules_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_module)

        code = """
eval("x + y")
exec("print('hello')")
"""
        issues = rules_module.check_dangerous_patterns(code)
        assert len(issues) >= 2
        assert any("eval" in issue.lower() for issue in issues)
        assert any("exec" in issue.lower() for issue in issues)

    def test_onex_compliance_check(self):
        """Test ONEX compliance checking."""
        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "omninode_bridge"
            / "codegen"
            / "business_logic"
            / "validation_rules.py"
        )
        spec = importlib.util.spec_from_file_location("validation_rules", rules_path)
        rules_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_module)

        # Non-compliant code
        code = """
raise Exception("error")
logger.info("message")
"""
        compliant, issues = rules_module.check_onex_compliance(code)
        assert not compliant
        assert len(issues) > 0

    def test_complexity_estimation(self):
        """Test complexity estimation."""
        rules_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "omninode_bridge"
            / "codegen"
            / "business_logic"
            / "validation_rules.py"
        )
        spec = importlib.util.spec_from_file_location("validation_rules", rules_path)
        rules_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_module)

        simple_code = "return x"
        complex_code = """
if x and y:
    for i in range(10):
        if a or b:
            while c:
                pass
"""
        assert rules_module.estimate_complexity(simple_code) == 1
        assert rules_module.estimate_complexity(complex_code) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
