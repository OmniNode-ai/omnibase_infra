#!/usr/bin/env python3
"""Unit tests for code generation event schemas.

Tests cover:
- Valid instantiation with all required fields
- Default values work correctly
- Field validation (e.g., confidence 0.0-1.0, limit 1-20)
- Schema version defaults to "1.0"
- JSON serialization/deserialization
- Invalid data raises ValidationError
- Edge cases and boundary conditions
"""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.events.codegen_schemas import (
    CodegenAnalysisRequest,
    CodegenAnalysisResponse,
    CodegenMixinRequest,
    CodegenMixinResponse,
    CodegenPatternRequest,
    CodegenPatternResponse,
    CodegenStatusEvent,
    CodegenValidationRequest,
    CodegenValidationResponse,
)
from omninode_bridge.events.enums import (
    EnumAnalysisType,
    EnumNodeType,
    EnumSessionStatus,
    EnumValidationType,
)


# Fixtures for common test data
@pytest.fixture
def correlation_id():
    """Generate a correlation ID for testing."""
    return uuid4()


@pytest.fixture
def session_id():
    """Generate a session ID for testing."""
    return uuid4()


@pytest.fixture
def test_timestamp():
    """Generate a timestamp for testing."""
    return datetime.now(UTC)


class TestCodegenAnalysisRequest:
    """Test suite for CodegenAnalysisRequest schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="# PRD: New Feature\n\nImplement feature X",
        )

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.prd_content == "# PRD: New Feature\n\nImplement feature X"
        assert request.analysis_type == EnumAnalysisType.FULL  # Default value
        assert request.workspace_context == {}  # Default value
        assert isinstance(request.timestamp, datetime)
        assert request.schema_version == "1.0"

    def test_valid_creation_with_all_fields(
        self, correlation_id, session_id, test_timestamp
    ):
        """Test valid instantiation with all fields specified."""
        workspace_context = {
            "project_root": "/path/to/project",
            "files": ["file1.py", "file2.py"],
        }

        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="# PRD Content",
            analysis_type=EnumAnalysisType.PARTIAL,
            workspace_context=workspace_context,
            timestamp=test_timestamp,
            schema_version="1.0",
        )

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.prd_content == "# PRD Content"
        assert request.analysis_type == EnumAnalysisType.PARTIAL
        assert request.workspace_context == workspace_context
        assert request.timestamp == test_timestamp
        assert request.schema_version == "1.0"

    def test_default_values(self, correlation_id, session_id):
        """Test that default values work correctly."""
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="Content",
        )

        assert request.analysis_type == EnumAnalysisType.FULL
        assert request.workspace_context == {}
        assert isinstance(request.timestamp, datetime)
        assert request.schema_version == "1.0"

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenAnalysisRequest(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "prd_content" in error_fields

    def test_json_serialization(self, correlation_id, session_id):
        """Test JSON serialization."""
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="Content",
        )

        json_str = request.model_dump_json()
        parsed_data = json.loads(json_str)

        assert parsed_data["correlation_id"] == str(correlation_id)
        assert parsed_data["session_id"] == str(session_id)
        assert parsed_data["prd_content"] == "Content"
        assert parsed_data["schema_version"] == "1.0"

    def test_json_deserialization(self, correlation_id, session_id):
        """Test JSON deserialization."""
        data = {
            "correlation_id": str(correlation_id),
            "session_id": str(session_id),
            "prd_content": "Content",
            "analysis_type": "quick",
        }

        request = CodegenAnalysisRequest.model_validate(data)

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.prd_content == "Content"
        assert request.analysis_type == EnumAnalysisType.QUICK

    def test_roundtrip_serialization(self, correlation_id, session_id):
        """Test serialization and deserialization roundtrip."""
        original = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="Content",
            analysis_type=EnumAnalysisType.PARTIAL,
        )

        data = original.model_dump()
        restored = CodegenAnalysisRequest.model_validate(data)

        assert restored.correlation_id == original.correlation_id
        assert restored.session_id == original.session_id
        assert restored.prd_content == original.prd_content
        assert restored.analysis_type == original.analysis_type


class TestCodegenAnalysisResponse:
    """Test suite for CodegenAnalysisResponse schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        analysis_result = {"requirements": ["req1", "req2"], "architecture": {}}

        response = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result=analysis_result,
            confidence=0.85,
            processing_time_ms=150,
        )

        assert response.correlation_id == correlation_id
        assert response.session_id == session_id
        assert response.analysis_result == analysis_result
        assert response.confidence == 0.85
        assert response.processing_time_ms == 150
        assert isinstance(response.timestamp, datetime)
        assert response.schema_version == "1.0"

    def test_confidence_validation_lower_bound(self, correlation_id, session_id):
        """Test that confidence accepts 0.0."""
        response = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result={},
            confidence=0.0,
            processing_time_ms=100,
        )

        assert response.confidence == 0.0

    def test_confidence_validation_upper_bound(self, correlation_id, session_id):
        """Test that confidence accepts 1.0."""
        response = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result={},
            confidence=1.0,
            processing_time_ms=100,
        )

        assert response.confidence == 1.0

    def test_confidence_validation_below_range(self, correlation_id, session_id):
        """Test that confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenAnalysisResponse(
                correlation_id=correlation_id,
                session_id=session_id,
                analysis_result={},
                confidence=-0.1,
                processing_time_ms=100,
            )

        errors = exc_info.value.errors()
        assert any("confidence" in str(e) for e in errors)

    def test_confidence_validation_above_range(self, correlation_id, session_id):
        """Test that confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenAnalysisResponse(
                correlation_id=correlation_id,
                session_id=session_id,
                analysis_result={},
                confidence=1.1,
                processing_time_ms=100,
            )

        errors = exc_info.value.errors()
        assert any("confidence" in str(e) for e in errors)

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenAnalysisResponse(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "analysis_result" in error_fields
        assert "confidence" in error_fields
        assert "processing_time_ms" in error_fields

    def test_json_roundtrip(self, correlation_id, session_id):
        """Test JSON serialization and deserialization roundtrip."""
        original = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result={"key": "value"},
            confidence=0.92,
            processing_time_ms=250,
        )

        json_str = original.model_dump_json()
        restored = CodegenAnalysisResponse.model_validate_json(json_str)

        assert restored.correlation_id == original.correlation_id
        assert restored.confidence == original.confidence
        assert restored.processing_time_ms == original.processing_time_ms


class TestCodegenValidationRequest:
    """Test suite for CodegenValidationRequest schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="class MyNode:\n    pass",
            node_type=EnumNodeType.EFFECT,
        )

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.code_content == "class MyNode:\n    pass"
        assert request.node_type == EnumNodeType.EFFECT
        assert request.contracts == []  # Default value
        assert request.validation_type == EnumValidationType.FULL  # Default value
        assert isinstance(request.timestamp, datetime)
        assert request.schema_version == "1.0"

    def test_valid_creation_with_all_fields(
        self, correlation_id, session_id, test_timestamp
    ):
        """Test valid instantiation with all fields specified."""
        contracts = [{"name": "contract1", "version": "1.0"}]

        request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="code",
            node_type=EnumNodeType.COMPUTE,
            contracts=contracts,
            validation_type=EnumValidationType.COMPLIANCE,
            timestamp=test_timestamp,
        )

        assert request.node_type == EnumNodeType.COMPUTE
        assert request.contracts == contracts
        assert request.validation_type == EnumValidationType.COMPLIANCE
        assert request.timestamp == test_timestamp

    def test_node_type_values(self, correlation_id, session_id):
        """Test different node type values."""
        for node_type in [
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ]:
            request = CodegenValidationRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                code_content="code",
                node_type=node_type,
            )
            assert request.node_type == node_type

    def test_validation_type_values(self, correlation_id, session_id):
        """Test different validation type values."""
        for validation_type in [
            EnumValidationType.FULL,
            EnumValidationType.SYNTAX,
            EnumValidationType.COMPLIANCE,
            EnumValidationType.SEMANTIC,
        ]:
            request = CodegenValidationRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                code_content="code",
                node_type=EnumNodeType.EFFECT,
                validation_type=validation_type,
            )
            assert request.validation_type == validation_type

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenValidationRequest(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "code_content" in error_fields
        assert "node_type" in error_fields

    def test_json_serialization(self, correlation_id, session_id):
        """Test JSON serialization."""
        contracts = [{"name": "test"}]
        request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="code",
            node_type=EnumNodeType.EFFECT,
            contracts=contracts,
        )

        json_str = request.model_dump_json()
        parsed_data = json.loads(json_str)

        assert parsed_data["correlation_id"] == str(correlation_id)
        assert parsed_data["contracts"] == contracts


class TestCodegenValidationResponse:
    """Test suite for CodegenValidationResponse schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        validation_result = {"errors": [], "warnings": []}

        response = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result=validation_result,
            quality_score=0.88,
            onex_compliance_score=0.95,
            is_valid=True,
            processing_time_ms=200,
        )

        assert response.correlation_id == correlation_id
        assert response.session_id == session_id
        assert response.validation_result == validation_result
        assert response.quality_score == 0.88
        assert response.onex_compliance_score == 0.95
        assert response.is_valid is True
        assert response.processing_time_ms == 200
        assert isinstance(response.timestamp, datetime)
        assert response.schema_version == "1.0"

    def test_quality_score_validation_bounds(self, correlation_id, session_id):
        """Test quality score boundary validation."""
        # Test lower bound
        response = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.0,
            onex_compliance_score=0.5,
            is_valid=False,
            processing_time_ms=100,
        )
        assert response.quality_score == 0.0

        # Test upper bound
        response = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=1.0,
            onex_compliance_score=1.0,
            is_valid=True,
            processing_time_ms=100,
        )
        assert response.quality_score == 1.0

    def test_quality_score_validation_out_of_bounds(self, correlation_id, session_id):
        """Test that quality score out of bounds raises ValidationError."""
        with pytest.raises(ValidationError):
            CodegenValidationResponse(
                correlation_id=correlation_id,
                session_id=session_id,
                validation_result={},
                quality_score=1.5,
                onex_compliance_score=0.5,
                is_valid=True,
                processing_time_ms=100,
            )

    def test_onex_compliance_score_validation_bounds(self, correlation_id, session_id):
        """Test ONEX compliance score boundary validation."""
        # Test lower bound
        response = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.5,
            onex_compliance_score=0.0,
            is_valid=False,
            processing_time_ms=100,
        )
        assert response.onex_compliance_score == 0.0

        # Test upper bound
        response = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.5,
            onex_compliance_score=1.0,
            is_valid=True,
            processing_time_ms=100,
        )
        assert response.onex_compliance_score == 1.0

    def test_boolean_is_valid_field(self, correlation_id, session_id):
        """Test is_valid boolean field."""
        response_valid = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.9,
            onex_compliance_score=0.9,
            is_valid=True,
            processing_time_ms=100,
        )
        assert response_valid.is_valid is True

        response_invalid = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.5,
            onex_compliance_score=0.5,
            is_valid=False,
            processing_time_ms=100,
        )
        assert response_invalid.is_valid is False

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenValidationResponse(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "validation_result" in error_fields
        assert "quality_score" in error_fields
        assert "onex_compliance_score" in error_fields
        assert "is_valid" in error_fields
        assert "processing_time_ms" in error_fields


class TestCodegenPatternRequest:
    """Test suite for CodegenPatternRequest schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        request = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="A node that processes user data",
            node_type=EnumNodeType.COMPUTE,
        )

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.node_description == "A node that processes user data"
        assert request.node_type == EnumNodeType.COMPUTE
        assert request.limit == 5  # Default value
        assert isinstance(request.timestamp, datetime)
        assert request.schema_version == "1.0"

    def test_limit_default_value(self, correlation_id, session_id):
        """Test that limit defaults to 5."""
        request = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="desc",
            node_type=EnumNodeType.EFFECT,
        )
        assert request.limit == 5

    def test_limit_validation_lower_bound(self, correlation_id, session_id):
        """Test that limit accepts 1 (lower bound)."""
        request = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="desc",
            node_type=EnumNodeType.EFFECT,
            limit=1,
        )
        assert request.limit == 1

    def test_limit_validation_upper_bound(self, correlation_id, session_id):
        """Test that limit accepts 20 (upper bound)."""
        request = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="desc",
            node_type=EnumNodeType.EFFECT,
            limit=20,
        )
        assert request.limit == 20

    def test_limit_validation_below_range(self, correlation_id, session_id):
        """Test that limit below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenPatternRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                node_description="desc",
                node_type=EnumNodeType.EFFECT,
                limit=0,
            )

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_limit_validation_above_range(self, correlation_id, session_id):
        """Test that limit above 20 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenPatternRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                node_description="desc",
                node_type=EnumNodeType.EFFECT,
                limit=21,
            )

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenPatternRequest(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "node_description" in error_fields
        assert "node_type" in error_fields

    def test_json_roundtrip(self, correlation_id, session_id):
        """Test JSON serialization and deserialization roundtrip."""
        original = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="desc",
            node_type=EnumNodeType.REDUCER,
            limit=10,
        )

        json_str = original.model_dump_json()
        restored = CodegenPatternRequest.model_validate_json(json_str)

        assert restored.correlation_id == original.correlation_id
        assert restored.node_description == original.node_description
        assert restored.limit == original.limit


class TestCodegenPatternResponse:
    """Test suite for CodegenPatternResponse schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        pattern_result = [
            {"node_name": "NodeExample", "similarity": 0.85},
            {"node_name": "NodeSimilar", "similarity": 0.72},
        ]

        response = CodegenPatternResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            pattern_result=pattern_result,
            total_matches=15,
            processing_time_ms=120,
        )

        assert response.correlation_id == correlation_id
        assert response.session_id == session_id
        assert response.pattern_result == pattern_result
        assert response.total_matches == 15
        assert response.processing_time_ms == 120
        assert isinstance(response.timestamp, datetime)
        assert response.schema_version == "1.0"

    def test_empty_pattern_result(self, correlation_id, session_id):
        """Test with empty pattern result list."""
        response = CodegenPatternResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            pattern_result=[],
            total_matches=0,
            processing_time_ms=50,
        )

        assert response.pattern_result == []
        assert response.total_matches == 0

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenPatternResponse(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "pattern_result" in error_fields
        assert "total_matches" in error_fields
        assert "processing_time_ms" in error_fields

    def test_json_serialization(self, correlation_id, session_id):
        """Test JSON serialization."""
        pattern_result = [{"node": "test", "score": 0.9}]

        response = CodegenPatternResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            pattern_result=pattern_result,
            total_matches=1,
            processing_time_ms=100,
        )

        json_str = response.model_dump_json()
        parsed_data = json.loads(json_str)

        assert parsed_data["pattern_result"] == pattern_result
        assert parsed_data["total_matches"] == 1


class TestCodegenMixinRequest:
    """Test suite for CodegenMixinRequest schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        requirements = ["caching", "metrics", "logging"]

        request = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=requirements,
            node_type=EnumNodeType.EFFECT,
        )

        assert request.correlation_id == correlation_id
        assert request.session_id == session_id
        assert request.requirements == requirements
        assert request.node_type == EnumNodeType.EFFECT
        assert request.existing_mixins == []  # Default value
        assert isinstance(request.timestamp, datetime)
        assert request.schema_version == "1.0"

    def test_valid_creation_with_existing_mixins(self, correlation_id, session_id):
        """Test with existing mixins specified."""
        requirements = ["caching", "metrics"]
        existing_mixins = ["LoggingMixin", "MetricsMixin"]

        request = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=requirements,
            node_type=EnumNodeType.COMPUTE,
            existing_mixins=existing_mixins,
        )

        assert request.existing_mixins == existing_mixins

    def test_empty_requirements_list(self, correlation_id, session_id):
        """Test with empty requirements list."""
        request = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=[],
            node_type=EnumNodeType.REDUCER,
        )

        assert request.requirements == []

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenMixinRequest(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "requirements" in error_fields
        assert "node_type" in error_fields

    def test_json_roundtrip(self, correlation_id, session_id):
        """Test JSON serialization and deserialization roundtrip."""
        requirements = ["feature1", "feature2"]
        existing_mixins = ["Mixin1"]

        original = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=requirements,
            node_type=EnumNodeType.ORCHESTRATOR,
            existing_mixins=existing_mixins,
        )

        json_str = original.model_dump_json()
        restored = CodegenMixinRequest.model_validate_json(json_str)

        assert restored.requirements == original.requirements
        assert restored.existing_mixins == original.existing_mixins


class TestCodegenMixinResponse:
    """Test suite for CodegenMixinResponse schema."""

    def test_valid_creation_with_required_fields(self, correlation_id, session_id):
        """Test valid instantiation with all required fields."""
        mixin_recommendations = [
            {"mixin": "CachingMixin", "rationale": "Improves performance"},
            {"mixin": "LoggingMixin", "rationale": "Enables debugging"},
        ]

        response = CodegenMixinResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            mixin_recommendations=mixin_recommendations,
            total_recommendations=2,
            processing_time_ms=180,
        )

        assert response.correlation_id == correlation_id
        assert response.session_id == session_id
        assert response.mixin_recommendations == mixin_recommendations
        assert response.total_recommendations == 2
        assert response.processing_time_ms == 180
        assert isinstance(response.timestamp, datetime)
        assert response.schema_version == "1.0"

    def test_empty_recommendations(self, correlation_id, session_id):
        """Test with empty recommendations list."""
        response = CodegenMixinResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            mixin_recommendations=[],
            total_recommendations=0,
            processing_time_ms=50,
        )

        assert response.mixin_recommendations == []
        assert response.total_recommendations == 0

    def test_missing_required_fields(self, correlation_id):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenMixinResponse(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "mixin_recommendations" in error_fields
        assert "total_recommendations" in error_fields
        assert "processing_time_ms" in error_fields

    def test_json_serialization(self, correlation_id, session_id):
        """Test JSON serialization."""
        mixin_recommendations = [{"mixin": "TestMixin", "score": 0.9}]

        response = CodegenMixinResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            mixin_recommendations=mixin_recommendations,
            total_recommendations=1,
            processing_time_ms=100,
        )

        json_str = response.model_dump_json()
        parsed_data = json.loads(json_str)

        assert parsed_data["mixin_recommendations"] == mixin_recommendations
        assert parsed_data["total_recommendations"] == 1


class TestCodegenStatusEvent:
    """Test suite for CodegenStatusEvent schema."""

    def test_valid_creation_with_required_fields(self, session_id):
        """Test valid instantiation with all required fields."""
        event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PROCESSING,
            progress_percentage=25.5,
            message="Analyzing PRD requirements",
        )

        assert event.session_id == session_id
        assert event.status == EnumSessionStatus.PROCESSING
        assert event.progress_percentage == 25.5
        assert event.message == "Analyzing PRD requirements"
        assert event.metadata == {}  # Default value
        assert isinstance(event.timestamp, datetime)
        assert event.schema_version == "1.0"

    def test_valid_creation_with_metadata(self, session_id):
        """Test with metadata specified."""
        metadata = {"current_step": "requirement_extraction", "errors": []}

        event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PROCESSING,
            progress_percentage=75.0,
            message="Generating code",
            metadata=metadata,
        )

        assert event.metadata == metadata

    def test_progress_percentage_validation_lower_bound(self, session_id):
        """Test that progress percentage accepts 0.0."""
        event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PENDING,
            progress_percentage=0.0,
            message="Starting",
        )
        assert event.progress_percentage == 0.0

    def test_progress_percentage_validation_upper_bound(self, session_id):
        """Test that progress percentage accepts 100.0."""
        event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.COMPLETED,
            progress_percentage=100.0,
            message="Completed",
        )
        assert event.progress_percentage == 100.0

    def test_progress_percentage_validation_below_range(self, session_id):
        """Test that progress percentage below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenStatusEvent(
                session_id=session_id,
                status=EnumSessionStatus.PENDING,
                progress_percentage=-0.1,
                message="Invalid",
            )

        errors = exc_info.value.errors()
        assert any("progress_percentage" in str(e) for e in errors)

    def test_progress_percentage_validation_above_range(self, session_id):
        """Test that progress percentage above 100.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenStatusEvent(
                session_id=session_id,
                status=EnumSessionStatus.COMPLETED,
                progress_percentage=100.1,
                message="Invalid",
            )

        errors = exc_info.value.errors()
        assert any("progress_percentage" in str(e) for e in errors)

    def test_status_values(self, session_id):
        """Test different valid status values."""
        valid_statuses = [
            EnumSessionStatus.PENDING,
            EnumSessionStatus.PROCESSING,
            EnumSessionStatus.COMPLETED,
            EnumSessionStatus.FAILED,
            EnumSessionStatus.CANCELLED,
        ]

        for status in valid_statuses:
            event = CodegenStatusEvent(
                session_id=session_id,
                status=status,
                progress_percentage=50.0,
                message=f"Status: {status.value}",
            )
            assert event.status == status

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenStatusEvent()

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "session_id" in error_fields
        assert "status" in error_fields
        assert "progress_percentage" in error_fields
        assert "message" in error_fields

    def test_json_roundtrip(self, session_id):
        """Test JSON serialization and deserialization roundtrip."""
        metadata = {"step": "validation", "errors_count": 0}

        original = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PROCESSING,
            progress_percentage=80.0,
            message="Validating code",
            metadata=metadata,
        )

        json_str = original.model_dump_json()
        restored = CodegenStatusEvent.model_validate_json(json_str)

        assert restored.session_id == original.session_id
        assert restored.status == original.status
        assert restored.progress_percentage == original.progress_percentage
        assert restored.message == original.message
        assert restored.metadata == original.metadata


class TestSchemaVersionConsistency:
    """Test suite for schema version consistency across all schemas."""

    def test_all_schemas_default_to_version_1_0(self, correlation_id, session_id):
        """Test that all schemas default to version 1.0."""
        # Request schemas
        analysis_req = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="content",
        )
        assert analysis_req.schema_version == "1.0"

        validation_req = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="code",
            node_type=EnumNodeType.EFFECT,
        )
        assert validation_req.schema_version == "1.0"

        pattern_req = CodegenPatternRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            node_description="desc",
            node_type=EnumNodeType.COMPUTE,
        )
        assert pattern_req.schema_version == "1.0"

        mixin_req = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=["req"],
            node_type=EnumNodeType.EFFECT,
        )
        assert mixin_req.schema_version == "1.0"

        # Response schemas
        analysis_resp = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result={},
            confidence=0.8,
            processing_time_ms=100,
        )
        assert analysis_resp.schema_version == "1.0"

        validation_resp = CodegenValidationResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            validation_result={},
            quality_score=0.8,
            onex_compliance_score=0.9,
            is_valid=True,
            processing_time_ms=100,
        )
        assert validation_resp.schema_version == "1.0"

        pattern_resp = CodegenPatternResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            pattern_result=[],
            total_matches=0,
            processing_time_ms=100,
        )
        assert pattern_resp.schema_version == "1.0"

        mixin_resp = CodegenMixinResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            mixin_recommendations=[],
            total_recommendations=0,
            processing_time_ms=100,
        )
        assert mixin_resp.schema_version == "1.0"

        # Status event
        status_event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PENDING,
            progress_percentage=0.0,
            message="Starting",
        )
        assert status_event.schema_version == "1.0"


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_uuid_string_conversion(self, correlation_id, session_id):
        """Test UUID to string conversion in JSON serialization."""
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="content",
        )

        json_str = request.model_dump_json()
        parsed = json.loads(json_str)

        # UUIDs should be serialized as strings
        assert isinstance(parsed["correlation_id"], str)
        assert isinstance(parsed["session_id"], str)
        assert UUID(parsed["correlation_id"]) == correlation_id
        assert UUID(parsed["session_id"]) == session_id

    def test_datetime_serialization(self, correlation_id, session_id):
        """Test datetime serialization in JSON."""
        timestamp = datetime.now(UTC)
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="content",
            timestamp=timestamp,
        )

        json_str = request.model_dump_json()
        parsed = json.loads(json_str)

        # Datetime should be serialized as ISO format string
        assert isinstance(parsed["timestamp"], str)

    def test_empty_dict_fields(self, correlation_id, session_id):
        """Test schemas with empty dict fields."""
        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content="content",
            workspace_context={},
        )
        assert request.workspace_context == {}

        response = CodegenAnalysisResponse(
            correlation_id=correlation_id,
            session_id=session_id,
            analysis_result={},
            confidence=0.5,
            processing_time_ms=100,
        )
        assert response.analysis_result == {}

    def test_empty_list_fields(self, correlation_id, session_id):
        """Test schemas with empty list fields."""
        request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="code",
            node_type=EnumNodeType.EFFECT,
            contracts=[],
        )
        assert request.contracts == []

        request = CodegenMixinRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            requirements=[],
            node_type=EnumNodeType.EFFECT,
            existing_mixins=[],
        )
        assert request.requirements == []
        assert request.existing_mixins == []

    def test_very_long_string_fields(self, correlation_id, session_id):
        """Test schemas with very long string fields."""
        long_content = "x" * 100000  # 100KB of content

        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content=long_content,
        )
        assert len(request.prd_content) == 100000

        validation_request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content=long_content,
            node_type=EnumNodeType.EFFECT,
        )
        assert len(validation_request.code_content) == 100000

    def test_complex_nested_dict_fields(self, correlation_id, session_id):
        """Test schemas with complex nested dictionary fields."""
        complex_metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "nested_dict": {"key": "value"},
                    }
                }
            }
        }

        event = CodegenStatusEvent(
            session_id=session_id,
            status=EnumSessionStatus.PROCESSING,
            progress_percentage=50.0,
            message="Processing",
            metadata=complex_metadata,
        )
        assert event.metadata == complex_metadata

    def test_special_characters_in_string_fields(self, correlation_id, session_id):
        """Test schemas with special characters in string fields."""
        special_content = 'Content with "quotes", \\backslashes\\, and\nnewlines'

        request = CodegenAnalysisRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            prd_content=special_content,
        )
        assert request.prd_content == special_content

        # Test JSON roundtrip with special characters
        json_str = request.model_dump_json()
        restored = CodegenAnalysisRequest.model_validate_json(json_str)
        assert restored.prd_content == special_content


class TestEnumValidation:
    """Test suite for enum field validation."""

    def test_invalid_analysis_type(self, correlation_id, session_id):
        """Test that invalid analysis_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenAnalysisRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                prd_content="content",
                analysis_type="invalid_type",  # type: ignore
            )

        errors = exc_info.value.errors()
        assert any("analysis_type" in str(e) for e in errors)

    def test_invalid_validation_type(self, correlation_id, session_id):
        """Test that invalid validation_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenValidationRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                code_content="code",
                node_type=EnumNodeType.EFFECT,
                validation_type="invalid_type",  # type: ignore
            )

        errors = exc_info.value.errors()
        assert any("validation_type" in str(e) for e in errors)

    def test_invalid_node_type(self, correlation_id, session_id):
        """Test that invalid node_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenValidationRequest(
                correlation_id=correlation_id,
                session_id=session_id,
                code_content="code",
                node_type="invalid_node",  # type: ignore
            )

        errors = exc_info.value.errors()
        assert any("node_type" in str(e) for e in errors)

    def test_invalid_session_status(self, session_id):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CodegenStatusEvent(
                session_id=session_id,
                status="invalid_status",  # type: ignore
                progress_percentage=50.0,
                message="Test",
            )

        errors = exc_info.value.errors()
        assert any("status" in str(e) for e in errors)

    def test_enum_serialization_as_strings(self, correlation_id, session_id):
        """Test that enums serialize as strings in JSON."""
        request = CodegenValidationRequest(
            correlation_id=correlation_id,
            session_id=session_id,
            code_content="code",
            node_type=EnumNodeType.COMPUTE,
            validation_type=EnumValidationType.SYNTAX,
        )

        json_str = request.model_dump_json()
        parsed = json.loads(json_str)

        # Enums should serialize as their string values
        assert parsed["node_type"] == "compute"
        assert parsed["validation_type"] == "syntax"

    def test_enum_deserialization_from_strings(self, correlation_id, session_id):
        """Test that enums can be deserialized from string values."""
        data = {
            "correlation_id": str(correlation_id),
            "session_id": str(session_id),
            "code_content": "code",
            "node_type": "reducer",  # String value
            "validation_type": "compliance",  # String value
        }

        request = CodegenValidationRequest.model_validate(data)

        # Should deserialize to proper enum types
        assert request.node_type == EnumNodeType.REDUCER
        assert request.validation_type == EnumValidationType.COMPLIANCE
        assert isinstance(request.node_type, EnumNodeType)
        assert isinstance(request.validation_type, EnumValidationType)
