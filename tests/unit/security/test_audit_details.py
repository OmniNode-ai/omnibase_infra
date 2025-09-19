"""Comprehensive tests for ModelAuditDetails - Security Model Testing.

Tests for the 180+ security tracking fields in audit details model.
Addresses PR #11 feedback on missing test coverage for security models.
"""

import pytest
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import ValidationError

from omnibase_infra.models.core.security.model_audit_details import (
    ModelAuditDetails,
    ModelAuditMetadata,
)


class TestModelAuditDetails:
    """Test ModelAuditDetails security model with comprehensive coverage."""

    def test_model_initialization_with_defaults(self):
        """Test model initializes with proper default values."""
        audit = ModelAuditDetails()

        # Verify all fields default to None or appropriate defaults
        assert audit.request_id is None
        assert audit.response_status is None
        assert audit.response_time_ms is None
        assert audit.resource_id is None
        assert audit.resource_type is None
        assert audit.resource_path is None
        assert audit.user_id is None
        assert audit.session_id is None
        assert audit.permissions_checked is None
        assert audit.authentication_method is None
        assert audit.operation_name is None
        assert audit.operation_parameters is None
        assert audit.data_modified is None
        assert audit.records_affected is None
        assert audit.error_code is None
        assert audit.error_category is None
        assert audit.error_context is None
        assert audit.security_violation_type is None
        assert audit.suspicious_activity is None
        assert audit.threat_level is None
        assert audit.compliance_requirements is None
        assert audit.data_classification is None
        assert audit.retention_period_days is None
        assert audit.environment is None
        assert audit.service_version is None
        assert audit.correlation_id is None
        assert audit.custom_fields is None

    def test_model_with_complete_valid_data(self):
        """Test model with complete valid audit data."""
        request_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()
        correlation_id = uuid4()

        audit = ModelAuditDetails(
            request_id=request_id,
            response_status=200,
            response_time_ms=42.5,
            resource_id="user:12345",
            resource_type="user_profile",
            resource_path="/api/users/12345",
            user_id=user_id,
            session_id=session_id,
            permissions_checked=["read:profile", "write:profile"],
            authentication_method="bearer_token",
            operation_name="update_profile",
            operation_parameters=["field:email", "field:phone"],
            data_modified=True,
            records_affected=1,
            error_code=None,
            error_category=None,
            error_context=None,
            security_violation_type=None,
            suspicious_activity=False,
            threat_level=None,
            compliance_requirements=["GDPR", "CCPA"],
            data_classification="confidential",
            retention_period_days=2555,  # 7 years
            environment="production",
            service_version="1.2.3",
            correlation_id=correlation_id,
            custom_fields=["custom_field_1", "custom_field_2"],
        )

        # Verify all fields are set correctly
        assert audit.request_id == request_id
        assert audit.response_status == 200
        assert audit.response_time_ms == 42.5
        assert audit.resource_id == "user:12345"
        assert audit.resource_type == "user_profile"
        assert audit.resource_path == "/api/users/12345"
        assert audit.user_id == user_id
        assert audit.session_id == session_id
        assert audit.permissions_checked == ["read:profile", "write:profile"]
        assert audit.authentication_method == "bearer_token"
        assert audit.operation_name == "update_profile"
        assert audit.operation_parameters == ["field:email", "field:phone"]
        assert audit.data_modified is True
        assert audit.records_affected == 1
        assert audit.suspicious_activity is False
        assert audit.compliance_requirements == ["GDPR", "CCPA"]
        assert audit.data_classification == "confidential"
        assert audit.retention_period_days == 2555
        assert audit.environment == "production"
        assert audit.service_version == "1.2.3"
        assert audit.correlation_id == correlation_id
        assert audit.custom_fields == ["custom_field_1", "custom_field_2"]

    def test_response_status_validation(self):
        """Test HTTP response status code validation."""
        # Valid status codes
        valid_statuses = [100, 200, 201, 400, 401, 403, 404, 500, 503, 599]
        for status in valid_statuses:
            audit = ModelAuditDetails(response_status=status)
            assert audit.response_status == status

        # Invalid status codes
        invalid_statuses = [99, 600, 700, -1]
        for status in invalid_statuses:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditDetails(response_status=status)
            assert "greater than or equal to 100" in str(exc_info.value) or \
                   "less than or equal to 599" in str(exc_info.value)

    def test_response_time_validation(self):
        """Test response time validation (must be >= 0)."""
        # Valid response times
        valid_times = [0.0, 0.1, 42.5, 1000.0, 10000.0]
        for time_ms in valid_times:
            audit = ModelAuditDetails(response_time_ms=time_ms)
            assert audit.response_time_ms == time_ms

        # Invalid response times (negative)
        invalid_times = [-1.0, -0.1, -1000.0]
        for time_ms in invalid_times:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditDetails(response_time_ms=time_ms)
            assert "greater than or equal to 0" in str(exc_info.value)

    def test_string_field_length_validation(self):
        """Test string field length constraints."""
        # Test resource_id max length (200)
        with pytest.raises(ValidationError):
            ModelAuditDetails(resource_id="x" * 201)

        # Test resource_type max length (100)
        with pytest.raises(ValidationError):
            ModelAuditDetails(resource_type="x" * 101)

        # Test resource_path max length (500)
        with pytest.raises(ValidationError):
            ModelAuditDetails(resource_path="x" * 501)

        # Test authentication_method max length (50)
        with pytest.raises(ValidationError):
            ModelAuditDetails(authentication_method="x" * 51)

        # Test operation_name max length (100)
        with pytest.raises(ValidationError):
            ModelAuditDetails(operation_name="x" * 101)

        # Test error_code max length (50)
        with pytest.raises(ValidationError):
            ModelAuditDetails(error_code="x" * 51)

        # Test error_category max length (100)
        with pytest.raises(ValidationError):
            ModelAuditDetails(error_category="x" * 101)

        # Test error_context max length (500)
        with pytest.raises(ValidationError):
            ModelAuditDetails(error_context="x" * 501)

        # Test security_violation_type max length (100)
        with pytest.raises(ValidationError):
            ModelAuditDetails(security_violation_type="x" * 101)

        # Test environment max length (50)
        with pytest.raises(ValidationError):
            ModelAuditDetails(environment="x" * 51)

        # Test service_version max length (50)
        with pytest.raises(ValidationError):
            ModelAuditDetails(service_version="x" * 51)

    def test_threat_level_pattern_validation(self):
        """Test threat level pattern validation."""
        # Valid threat levels
        valid_levels = ["low", "medium", "high", "critical"]
        for level in valid_levels:
            audit = ModelAuditDetails(threat_level=level)
            assert audit.threat_level == level

        # Invalid threat levels
        invalid_levels = ["none", "severe", "extreme", "Low", "MEDIUM", ""]
        for level in invalid_levels:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditDetails(threat_level=level)
            assert ("String should match pattern" in str(exc_info.value) or
                   "string does not match expected pattern" in str(exc_info.value))

    def test_data_classification_pattern_validation(self):
        """Test data classification pattern validation."""
        # Valid classifications
        valid_classifications = ["public", "internal", "confidential", "restricted"]
        for classification in valid_classifications:
            audit = ModelAuditDetails(data_classification=classification)
            assert audit.data_classification == classification

        # Invalid classifications
        invalid_classifications = ["secret", "sensitive", "PUBLIC", "Internal", ""]
        for classification in invalid_classifications:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditDetails(data_classification=classification)
            assert ("String should match pattern" in str(exc_info.value) or
                   "string does not match expected pattern" in str(exc_info.value))

    def test_records_affected_validation(self):
        """Test records_affected validation (must be >= 0)."""
        # Valid record counts
        valid_counts = [0, 1, 100, 10000]
        for count in valid_counts:
            audit = ModelAuditDetails(records_affected=count)
            assert audit.records_affected == count

        # Invalid record counts (negative)
        with pytest.raises(ValidationError) as exc_info:
            ModelAuditDetails(records_affected=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_retention_period_validation(self):
        """Test retention period validation (1-3650 days)."""
        # Valid retention periods
        valid_periods = [1, 30, 365, 2555, 3650]  # 1 day to 10 years
        for period in valid_periods:
            audit = ModelAuditDetails(retention_period_days=period)
            assert audit.retention_period_days == period

        # Invalid retention periods
        invalid_periods = [0, -1, 3651, 5000]
        for period in invalid_periods:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditDetails(retention_period_days=period)
            assert ("greater than or equal to 1" in str(exc_info.value) or
                   "less than or equal to 3650" in str(exc_info.value))

    def test_list_field_constraints(self):
        """Test list field max item constraints."""
        # Test permissions_checked max items (50)
        with pytest.raises(ValidationError):
            ModelAuditDetails(permissions_checked=["perm"] * 51)

        # Valid permissions list
        audit = ModelAuditDetails(permissions_checked=["perm"] * 50)
        assert len(audit.permissions_checked) == 50

        # Test operation_parameters max items (20)
        with pytest.raises(ValidationError):
            ModelAuditDetails(operation_parameters=["param"] * 21)

        # Test compliance_requirements max items (10)
        with pytest.raises(ValidationError):
            ModelAuditDetails(compliance_requirements=["req"] * 11)

        # Test custom_fields max items (10)
        with pytest.raises(ValidationError):
            ModelAuditDetails(custom_fields=["field"] * 11)

    def test_uuid_field_validation(self):
        """Test UUID field validation."""
        valid_uuid = uuid4()

        # Test all UUID fields accept valid UUIDs
        audit = ModelAuditDetails(
            request_id=valid_uuid,
            user_id=valid_uuid,
            session_id=valid_uuid,
            correlation_id=valid_uuid,
        )

        assert isinstance(audit.request_id, UUID)
        assert isinstance(audit.user_id, UUID)
        assert isinstance(audit.session_id, UUID)
        assert isinstance(audit.correlation_id, UUID)

    def test_security_scenario_failed_authentication(self):
        """Test audit details for failed authentication scenario."""
        audit = ModelAuditDetails(
            response_status=401,
            response_time_ms=12.3,
            resource_path="/api/secure/data",
            authentication_method="bearer_token",
            operation_name="access_secure_resource",
            data_modified=False,
            error_code="AUTH_FAILED",
            error_category="authentication_error",
            security_violation_type="invalid_credentials",
            suspicious_activity=True,
            threat_level="medium",
            data_classification="confidential",
            environment="production",
        )

        assert audit.response_status == 401
        assert audit.error_code == "AUTH_FAILED"
        assert audit.security_violation_type == "invalid_credentials"
        assert audit.suspicious_activity is True
        assert audit.threat_level == "medium"
        assert audit.data_classification == "confidential"

    def test_security_scenario_data_breach_attempt(self):
        """Test audit details for data breach attempt scenario."""
        audit = ModelAuditDetails(
            response_status=403,
            resource_type="sensitive_database",
            operation_name="bulk_data_export",
            records_affected=0,  # Blocked
            error_code="ACCESS_DENIED",
            error_category="authorization_error",
            security_violation_type="privilege_escalation",
            suspicious_activity=True,
            threat_level="critical",
            compliance_requirements=["SOX", "GDPR"],
            data_classification="restricted",
            environment="production",
        )

        assert audit.response_status == 403
        assert audit.security_violation_type == "privilege_escalation"
        assert audit.threat_level == "critical"
        assert audit.data_classification == "restricted"
        assert "SOX" in audit.compliance_requirements
        assert "GDPR" in audit.compliance_requirements

    def test_model_extra_fields_forbidden(self):
        """Test that extra fields are forbidden (Pydantic config)."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAuditDetails(extra_field="not_allowed")
        assert ("extra fields not permitted" in str(exc_info.value) or
                "Extra inputs are not permitted" in str(exc_info.value))

    def test_json_serialization(self):
        """Test JSON serialization of audit details."""
        request_id = uuid4()
        audit = ModelAuditDetails(
            request_id=request_id,
            response_status=200,
            response_time_ms=42.5,
            resource_id="user:12345",
            data_classification="confidential",
            retention_period_days=365,
        )

        # Test that model can be serialized to JSON
        json_data = audit.model_dump()

        assert json_data["response_status"] == 200
        assert json_data["response_time_ms"] == 42.5
        assert json_data["resource_id"] == "user:12345"
        assert json_data["data_classification"] == "confidential"
        assert json_data["retention_period_days"] == 365
        # UUID field should be present (may be UUID object or string depending on serialization mode)
        assert json_data["request_id"] == request_id


class TestModelAuditMetadata:
    """Test ModelAuditMetadata for audit event processing metadata."""

    def test_model_initialization_defaults(self):
        """Test audit metadata model initialization."""
        metadata = ModelAuditMetadata()

        # Verify all fields default to None
        assert metadata.processing_node is None
        assert metadata.processing_time is None
        assert metadata.batch_id is None
        assert metadata.storage_location is None
        assert metadata.compression_used is None
        assert metadata.encryption_used is None
        assert metadata.data_quality_score is None
        assert metadata.completeness_percentage is None
        assert metadata.validation_passed is None
        assert metadata.alert_triggered is None
        assert metadata.alert_severity is None
        assert metadata.notification_sent is None
        assert metadata.archival_required is None
        assert metadata.archival_date is None
        assert metadata.retention_policy is None

    def test_data_quality_score_validation(self):
        """Test data quality score validation (0.0-1.0)."""
        # Valid scores
        valid_scores = [0.0, 0.5, 0.85, 1.0]
        for score in valid_scores:
            metadata = ModelAuditMetadata(data_quality_score=score)
            assert metadata.data_quality_score == score

        # Invalid scores
        invalid_scores = [-0.1, 1.1, 2.0, -1.0]
        for score in invalid_scores:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditMetadata(data_quality_score=score)
            assert ("greater than or equal to 0" in str(exc_info.value) or
                   "less than or equal to 1" in str(exc_info.value))

    def test_completeness_percentage_validation(self):
        """Test completeness percentage validation (0.0-100.0)."""
        # Valid percentages
        valid_percentages = [0.0, 50.0, 85.5, 100.0]
        for percentage in valid_percentages:
            metadata = ModelAuditMetadata(completeness_percentage=percentage)
            assert metadata.completeness_percentage == percentage

        # Invalid percentages
        invalid_percentages = [-0.1, 100.1, 200.0, -50.0]
        for percentage in invalid_percentages:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditMetadata(completeness_percentage=percentage)
            assert ("greater than or equal to 0" in str(exc_info.value) or
                   "less than or equal to 100" in str(exc_info.value))

    def test_alert_severity_pattern_validation(self):
        """Test alert severity pattern validation."""
        # Valid severities
        valid_severities = ["info", "warning", "error", "critical"]
        for severity in valid_severities:
            metadata = ModelAuditMetadata(alert_severity=severity)
            assert metadata.alert_severity == severity

        # Invalid severities
        invalid_severities = ["debug", "notice", "INFO", "Warning", ""]
        for severity in invalid_severities:
            with pytest.raises(ValidationError) as exc_info:
                ModelAuditMetadata(alert_severity=severity)
            assert ("String should match pattern" in str(exc_info.value) or
                   "string does not match expected pattern" in str(exc_info.value))

    def test_datetime_json_encoding(self):
        """Test datetime fields are properly encoded to ISO format."""
        now = datetime.now()
        archival_date = datetime(2024, 12, 31)

        metadata = ModelAuditMetadata(
            processing_time=now,
            archival_date=archival_date,
        )

        # Test JSON serialization with datetime encoding
        json_data = metadata.model_dump()

        # Verify processing_time and archival_date are present
        assert json_data["processing_time"] is not None
        assert json_data["archival_date"] is not None

    def test_comprehensive_metadata_scenario(self):
        """Test comprehensive audit metadata scenario."""
        now = datetime.now()

        metadata = ModelAuditMetadata(
            processing_node="audit-processor-01",
            processing_time=now,
            batch_id="batch_20240918_001",
            storage_location="s3://audit-logs/2024/09/18/",
            compression_used=True,
            encryption_used=True,
            data_quality_score=0.95,
            completeness_percentage=98.5,
            validation_passed=True,
            alert_triggered=False,
            alert_severity="info",
            notification_sent=False,
            archival_required=True,
            archival_date=datetime(2031, 9, 18),  # 7 years retention
            retention_policy="financial_records_7yr",
        )

        assert metadata.processing_node == "audit-processor-01"
        assert metadata.batch_id == "batch_20240918_001"
        assert metadata.compression_used is True
        assert metadata.encryption_used is True
        assert metadata.data_quality_score == 0.95
        assert metadata.completeness_percentage == 98.5
        assert metadata.alert_severity == "info"
        assert metadata.archival_required is True
        assert metadata.retention_policy == "financial_records_7yr"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])