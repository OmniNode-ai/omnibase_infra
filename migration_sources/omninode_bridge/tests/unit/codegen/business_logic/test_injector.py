#!/usr/bin/env python3
"""
Unit tests for CodeInjector.

Tests stub detection, code injection, indentation preservation,
and error handling.
"""

import pytest

from omninode_bridge.codegen.business_logic.injector import (
    CodeInjectionError,
    CodeInjector,
)
from omninode_bridge.codegen.business_logic.models import StubInfo


class TestCodeInjector:
    """Test suite for CodeInjector."""

    @pytest.fixture
    def injector(self):
        """Create CodeInjector instance."""
        return CodeInjector()

    @pytest.fixture
    def sample_template_todo_pass(self):
        """Sample template with TODO + pass stub."""
        return '''class NodeExample:
    """Example node."""

    async def execute_effect(
        self,
        contract: ModelContractEffect,
        correlation_id: UUID,
    ) -> ModelEffectResult:
        """Execute the effect operation."""
        # TODO: Implement business logic
        pass

    def other_method(self):
        """Another method."""
        return "existing"
'''

    @pytest.fixture
    def sample_template_implementation_required(self):
        """Sample template with IMPLEMENTATION REQUIRED marker."""
        return '''class NodeExample:
    """Example node."""

    async def process_data(
        self,
        data: dict,
        context: dict,
    ) -> dict:
        """Process data."""
        # IMPLEMENTATION REQUIRED
        pass
'''

    @pytest.fixture
    def sample_template_not_implemented(self):
        """Sample template with NotImplementedError."""
        return '''class NodeExample:
    """Example node."""

    def validate_input(self, data: dict) -> bool:
        """Validate input data."""
        raise NotImplementedError("TODO: Implement validation")
'''

    @pytest.fixture
    def sample_template_simple_pass(self):
        """Sample template with simple pass."""
        return '''def helper_function(value: int) -> int:
    """Helper function."""
    pass
'''

    @pytest.fixture
    def sample_generated_code(self):
        """Sample LLM-generated code (indented method body)."""
        return """try:
    result = await self._perform_operation(contract, correlation_id)
    return ModelEffectResult(
        success=True,
        data=result,
        correlation_id=correlation_id,
    )
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise ModelOnexError(f"Effect execution failed: {e}") from e"""

    # ==================== Test: Stub Detection ====================

    def test_find_stubs_todo_pass(self, injector, sample_template_todo_pass):
        """Test finding stubs with TODO + pass pattern."""
        stubs = injector.find_stubs(sample_template_todo_pass, "example.py")

        assert len(stubs) == 1
        assert stubs[0].method_name == "execute_effect"
        assert stubs[0].file_path == "example.py"
        assert stubs[0].line_start == 4
        assert "async def execute_effect" in stubs[0].signature
        assert "Execute the effect operation" in stubs[0].docstring

    def test_find_stubs_implementation_required(
        self, injector, sample_template_implementation_required
    ):
        """Test finding stubs with IMPLEMENTATION REQUIRED marker."""
        stubs = injector.find_stubs(sample_template_implementation_required)

        assert len(stubs) == 1
        assert stubs[0].method_name == "process_data"
        assert "async def process_data" in stubs[0].signature

    def test_find_stubs_not_implemented(
        self, injector, sample_template_not_implemented
    ):
        """Test finding stubs with NotImplementedError."""
        stubs = injector.find_stubs(sample_template_not_implemented)

        assert len(stubs) == 1
        assert stubs[0].method_name == "validate_input"
        assert "def validate_input" in stubs[0].signature

    def test_find_stubs_simple_pass(self, injector, sample_template_simple_pass):
        """Test finding stubs with simple pass."""
        stubs = injector.find_stubs(sample_template_simple_pass)

        assert len(stubs) == 1
        assert stubs[0].method_name == "helper_function"

    def test_find_stubs_no_stubs(self, injector):
        """Test finding stubs when none exist."""
        code = """class Complete:
    def method(self):
        return "implemented"
"""
        stubs = injector.find_stubs(code)
        assert len(stubs) == 0

    def test_find_stubs_multiple(self, injector):
        """Test finding multiple stubs in one file."""
        code = """class MultiStub:
    def method1(self):
        # TODO: Implement
        pass

    def method2(self):
        raise NotImplementedError()

    def method3(self):
        # IMPLEMENTATION REQUIRED
        pass
"""
        stubs = injector.find_stubs(code)
        assert len(stubs) == 3
        method_names = [stub.method_name for stub in stubs]
        assert "method1" in method_names
        assert "method2" in method_names
        assert "method3" in method_names

    def test_find_stubs_avoids_duplicates(self, injector):
        """Test that duplicate methods are not returned multiple times."""
        code = """def method(self):
    # TODO: Implement
    pass
"""
        stubs = injector.find_stubs(code)
        # Even though multiple patterns might match, should only return once
        assert len(stubs) == 1

    # ==================== Test: Code Injection ====================

    def test_inject_code_basic(
        self, injector, sample_template_todo_pass, sample_generated_code
    ):
        """Test basic code injection."""
        stubs = injector.find_stubs(sample_template_todo_pass)
        assert len(stubs) == 1

        modified = injector.inject_code(
            template_code=sample_template_todo_pass,
            generated_code=sample_generated_code,
            stub_info=stubs[0],
        )

        # Check that generated code is present
        assert "try:" in modified
        assert "await self._perform_operation" in modified
        assert "ModelEffectResult" in modified

        # Check that stub marker is gone
        assert "# TODO: Implement business logic" not in modified
        assert "pass" not in modified or "pass" in sample_generated_code

        # Check that other methods are preserved
        assert "def other_method" in modified
        assert 'return "existing"' in modified

    def test_inject_code_preserves_indentation(
        self, injector, sample_template_todo_pass, sample_generated_code
    ):
        """Test that code injection preserves indentation."""
        stubs = injector.find_stubs(sample_template_todo_pass)

        modified = injector.inject_code(
            template_code=sample_template_todo_pass,
            generated_code=sample_generated_code,
            stub_info=stubs[0],
        )

        # Check that indentation is correct (should match method body - 8 spaces)
        lines = modified.split("\n")
        for line in lines:
            if "try:" in line and line.strip() == "try:":
                # Should have 8 spaces (2 levels of indentation)
                assert line.startswith("        try:")
                break
        else:
            pytest.fail("Generated code not found in modified template")

    def test_inject_code_with_line_numbers(self, injector):
        """Test injection using explicit line numbers in StubInfo."""
        template = """def method(self):
    # TODO: Implement
    pass
"""
        generated = "return 42"

        stub_info = StubInfo(
            file_path="test.py",
            method_name="method",
            stub_code="# TODO: Implement\n    pass",
            line_start=1,
            line_end=3,
            signature="def method(self)",
            docstring=None,
        )

        modified = injector.inject_code(
            template_code=template, generated_code=generated, stub_info=stub_info
        )

        assert "return 42" in modified
        assert "# TODO: Implement" not in modified

    def test_inject_code_method_not_found(self, injector):
        """Test injection fails when method not found."""
        template = """def other_method(self):
    return 1
"""
        stub_info = StubInfo(
            file_path="test.py",
            method_name="nonexistent_method",
            stub_code="pass",
            line_start=1,
            line_end=1,
            signature="def nonexistent_method(self)",
            docstring=None,
        )

        with pytest.raises(CodeInjectionError) as exc_info:
            injector.inject_code(
                template_code=template, generated_code="return 2", stub_info=stub_info
            )

        assert "nonexistent_method" in str(exc_info.value)

    def test_inject_code_empty_generated(self, injector, sample_template_todo_pass):
        """Test injection with empty generated code."""
        stubs = injector.find_stubs(sample_template_todo_pass)

        modified = injector.inject_code(
            template_code=sample_template_todo_pass,
            generated_code="",
            stub_info=stubs[0],
        )

        # Should still work, just removes the stub
        assert "# TODO: Implement business logic" not in modified

    # ==================== Test: Indentation Handling ====================

    def test_extract_indentation_spaces(self, injector):
        """Test extracting indentation with spaces."""
        code = "    def method(self):\n        pass"
        indentation = injector._extract_indentation(code, 4)
        assert indentation == "    "

    def test_extract_indentation_tabs(self, injector):
        """Test extracting indentation with tabs."""
        code = "\t\tdef method(self):\n\t\t\tpass"
        indentation = injector._extract_indentation(code, 2)
        assert indentation == "\t\t"

    def test_format_with_indentation(self, injector):
        """Test formatting code with indentation."""
        code = "line1\nline2\n    nested"
        indentation = "    "

        formatted = injector._format_with_indentation(code, indentation)

        expected = "    line1\n    line2\n        nested"
        assert formatted == expected

    def test_format_with_indentation_preserves_empty_lines(self, injector):
        """Test that empty lines are preserved during formatting."""
        code = "line1\n\nline2"
        indentation = "    "

        formatted = injector._format_with_indentation(code, indentation)

        lines = formatted.split("\n")
        assert lines[1] == ""  # Empty line preserved

    # ==================== Test: Method End Detection ====================

    def test_find_method_end_simple(self, injector):
        """Test finding end of simple method."""
        code = """class Example:
    def method1(self):
        return 1

    def method2(self):
        return 2
"""
        # Start at method1 definition
        start_pos = code.index("def method1")
        end_pos = injector._find_method_end(code, start_pos)

        # Should end before method2
        method_text = code[start_pos:end_pos]
        assert "method1" in method_text
        assert "method2" not in method_text

    def test_find_method_end_nested(self, injector):
        """Test finding end of method with nested blocks."""
        code = """class Example:
    def method(self):
        if True:
            for i in range(10):
                print(i)
        return True

    def next_method(self):
        pass
"""
        start_pos = code.index("def method(self)")
        end_pos = injector._find_method_end(code, start_pos)

        method_text = code[start_pos:end_pos]
        assert "def method(self)" in method_text
        assert "return True" in method_text
        assert "def next_method" not in method_text

    def test_find_method_end_at_file_end(self, injector):
        """Test finding end of method at end of file."""
        code = """class Example:
    def method(self):
        return 1
"""
        start_pos = code.index("def method")
        end_pos = injector._find_method_end(code, start_pos)

        # Should extend to end of code
        assert end_pos == len(code)

    # ==================== Test: Edge Cases ====================

    def test_inject_multiline_docstring(self, injector):
        """Test injection with multiline docstring."""
        template = '''def method(self):
    """
    This is a multiline
    docstring.
    """
    # TODO: Implement
    pass
'''
        stubs = injector.find_stubs(template)
        assert len(stubs) == 1

        modified = injector.inject_code(
            template_code=template, generated_code="return True", stub_info=stubs[0]
        )

        assert "return True" in modified
        assert "This is a multiline" in modified  # Docstring preserved

    def test_inject_async_method(self, injector):
        """Test injection with async method."""
        template = '''async def async_method(self) -> int:
    """Async method."""
    # TODO: Implement
    pass
'''
        stubs = injector.find_stubs(template)
        assert len(stubs) == 1
        assert "async def" in stubs[0].signature

        modified = injector.inject_code(
            template_code=template,
            generated_code="return await self.fetch()",
            stub_info=stubs[0],
        )

        assert "return await self.fetch()" in modified

    def test_inject_complex_signature(self, injector):
        """Test injection with complex method signature."""
        template = '''def complex_method(
    self,
    arg1: str,
    arg2: Optional[int] = None,
    *args: Any,
    **kwargs: dict,
) -> Tuple[bool, str]:
    """Complex signature."""
    # TODO: Implement
    pass
'''
        stubs = injector.find_stubs(template)
        assert len(stubs) == 1

        modified = injector.inject_code(
            template_code=template,
            generated_code="return (True, 'success')",
            stub_info=stubs[0],
        )

        assert "return (True, 'success')" in modified

    def test_stub_info_line_numbers_accuracy(self, injector):
        """Test that line numbers in StubInfo are accurate."""
        template = """# Line 1
class Example:  # Line 2
    def method1(self):  # Line 3
        pass  # Line 4

    def method2(self):  # Line 6
        # TODO: Implement  # Line 7
        pass  # Line 8
"""
        stubs = injector.find_stubs(template)
        stub = next(s for s in stubs if s.method_name == "method2")

        assert stub.line_start == 6
        assert stub.line_end == 8


# ==================== Integration Tests ====================


class TestCodeInjectorIntegration:
    """Integration tests for CodeInjector with real-world scenarios."""

    @pytest.fixture
    def injector(self):
        """Create CodeInjector instance."""
        return CodeInjector()

    def test_full_workflow(self, injector):
        """Test complete workflow: find stubs -> inject code."""
        template = '''class NodeDatabaseEffect:
    """Database effect node."""

    async def execute_effect(
        self,
        contract: ModelContractEffect,
        correlation_id: UUID,
    ) -> ModelEffectResult:
        """Execute database operation."""
        # TODO: Implement business logic
        pass

    async def validate_connection(self) -> bool:
        """Validate database connection."""
        # IMPLEMENTATION REQUIRED
        pass
'''

        generated_execute = """try:
    query = contract.input_state.get("query")
    result = await self.db.execute(query)
    return ModelEffectResult(
        success=True,
        data=result,
        correlation_id=correlation_id,
    )
except Exception as e:
    raise ModelOnexError(f"Database error: {e}") from e"""

        generated_validate = """try:
    await self.db.ping()
    return True
except Exception:
    return False"""

        # Step 1: Find all stubs
        stubs = injector.find_stubs(template, "node.py")
        assert len(stubs) == 2

        # Step 2: Inject code for first stub
        modified = template
        execute_stub = next(s for s in stubs if s.method_name == "execute_effect")
        modified = injector.inject_code(modified, generated_execute, execute_stub)

        # Step 3: Inject code for second stub (need to find again after modification)
        stubs_updated = injector.find_stubs(modified, "node.py")
        validate_stub = next(
            s for s in stubs_updated if s.method_name == "validate_connection"
        )
        modified = injector.inject_code(modified, generated_validate, validate_stub)

        # Verify both injections
        assert "await self.db.execute(query)" in modified
        assert "await self.db.ping()" in modified
        assert "# TODO: Implement business logic" not in modified
        assert "# IMPLEMENTATION REQUIRED" not in modified

    def test_inject_preserves_class_structure(self, injector):
        """Test that injection preserves class structure."""
        template = '''class NodeExample:
    """Example node."""

    def __init__(self):
        """Initialize."""
        self.state = {}

    async def execute_effect(self) -> dict:
        """Execute effect."""
        # TODO: Implement
        pass

    def cleanup(self):
        """Cleanup resources."""
        self.state = {}
'''

        stubs = injector.find_stubs(template)
        modified = injector.inject_code(
            template, "return {'status': 'success'}", stubs[0]
        )

        # Verify structure preserved
        assert "def __init__(self):" in modified
        assert "self.state = {}" in modified
        assert "def cleanup(self):" in modified
        assert "return {'status': 'success'}" in modified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
