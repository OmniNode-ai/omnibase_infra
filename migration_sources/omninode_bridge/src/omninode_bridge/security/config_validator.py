"""Production security configuration validator for comprehensive security compliance."""

import os
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger

logger = structlog.get_logger(__name__)


class SecurityValidationResult(BaseModel):
    """Result of security validation check."""

    check_name: str
    status: str  # pass, fail, warning, info
    severity: str  # critical, high, medium, low, info
    message: str
    recommendation: str | None = None
    current_value: str | None = None
    expected_value: str | None = None
    details: dict[str, Any] | None = None


class SecurityConfiguration(BaseModel):
    """Model for security configuration validation."""

    environment: str
    service_name: str
    validation_timestamp: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    security_score: float
    compliance_level: str
    results: list[SecurityValidationResult]
    recommendations: list[str]


class ProductionSecurityValidator:
    """Comprehensive production security configuration validator."""

    def __init__(
        self,
        environment: str = None,
        service_name: str = "omninode_bridge",
        strict_mode: bool = None,
    ):
        """Initialize production security validator.

        Args:
            environment: Deployment environment
            service_name: Service name for audit logging
            strict_mode: Enable strict validation (auto-detected for production)
        """
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )
        self.service_name = service_name

        # Auto-detect strict mode for production
        if strict_mode is None:
            self.strict_mode = self.environment == "production"
        else:
            self.strict_mode = strict_mode

        # Initialize audit logger
        self.audit_logger = get_audit_logger(service_name)

        # Validation results
        self.results: list[SecurityValidationResult] = []

    def validate_all(self) -> SecurityConfiguration:
        """Run comprehensive security validation checks.

        Returns:
            SecurityConfiguration with validation results
        """
        logger.info(
            "Starting comprehensive security validation",
            environment=self.environment,
            strict_mode=self.strict_mode,
        )

        # Clear previous results
        self.results = []

        # Run all validation categories
        self._validate_environment_variables()
        self._validate_authentication_security()
        self._validate_encryption_security()
        self._validate_network_security()
        self._validate_database_security()
        self._validate_kafka_security()
        self._validate_ssl_tls_configuration()
        self._validate_cors_configuration()
        self._validate_rate_limiting_configuration()
        self._validate_audit_logging_configuration()
        self._validate_api_security()
        self._validate_webhook_security()

        # Calculate security metrics
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == "pass"])
        failed_checks = len([r for r in self.results if r.status == "fail"])
        warnings = len([r for r in self.results if r.status == "warning"])

        # Calculate security score (0-100)
        if total_checks == 0:
            security_score = 0.0
        else:
            # Weight different severities
            score = 0
            for result in self.results:
                if result.status == "pass":
                    if result.severity == "critical":
                        score += 20
                    elif result.severity == "high":
                        score += 15
                    elif result.severity == "medium":
                        score += 10
                    elif result.severity == "low":
                        score += 5
                    else:  # info
                        score += 2
                elif result.status == "warning":
                    if result.severity == "critical":
                        score += 10
                    elif result.severity == "high":
                        score += 8
                    elif result.severity == "medium":
                        score += 5
                    elif result.severity == "low":
                        score += 3
                    else:  # info
                        score += 1
                # Failed checks contribute 0 points

            max_score = sum(
                [
                    (
                        20
                        if r.severity == "critical"
                        else (
                            15
                            if r.severity == "high"
                            else (
                                10
                                if r.severity == "medium"
                                else 5 if r.severity == "low" else 2
                            )
                        )
                    )
                    for r in self.results
                ],
            )

            security_score = (score / max_score * 100) if max_score > 0 else 0.0

        # Determine compliance level
        if security_score >= 95:
            compliance_level = "excellent"
        elif security_score >= 85:
            compliance_level = "good"
        elif security_score >= 70:
            compliance_level = "acceptable"
        elif security_score >= 50:
            compliance_level = "poor"
        else:
            compliance_level = "critical"

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Create configuration result
        config = SecurityConfiguration(
            environment=self.environment,
            service_name=self.service_name,
            validation_timestamp=datetime.now(UTC),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            security_score=security_score,
            compliance_level=compliance_level,
            results=self.results,
            recommendations=recommendations,
        )

        # Log validation summary
        self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY_AUDIT,
            severity=AuditSeverity.HIGH if failed_checks > 0 else AuditSeverity.LOW,
            additional_data={
                "component": "security_configuration_validator",
                "environment": self.environment,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "warnings": warnings,
                "security_score": security_score,
                "compliance_level": compliance_level,
            },
            message=f"Security validation completed: {compliance_level} compliance ({security_score:.1f}%)",
        )

        logger.info(
            "Security validation completed",
            environment=self.environment,
            total_checks=total_checks,
            passed=passed_checks,
            failed=failed_checks,
            warnings=warnings,
            security_score=security_score,
            compliance_level=compliance_level,
        )

        return config

    def _add_result(
        self,
        check_name: str,
        status: str,
        severity: str,
        message: str,
        recommendation: str = None,
        current_value: str = None,
        expected_value: str = None,
        details: dict[str, Any] = None,
    ):
        """Add validation result."""
        result = SecurityValidationResult(
            check_name=check_name,
            status=status,
            severity=severity,
            message=message,
            recommendation=recommendation,
            current_value=current_value,
            expected_value=expected_value,
            details=details,
        )
        self.results.append(result)

    def _validate_environment_variables(self):
        """Validate critical environment variables."""
        # Critical environment variables for production
        critical_vars = [
            "POSTGRES_PASSWORD",
            "API_KEY",
            "JWT_SECRET_KEY",
            "WEBHOOK_SIGNING_SECRET",
        ]

        # High priority environment variables
        high_priority_vars = [
            "API_KEY_ENCRYPTION_SEED",
            "CORS_ALLOWED_ORIGINS",
            "ENVIRONMENT",
        ]

        # Validate critical variables
        for var in critical_vars:
            value = os.getenv(var)
            if not value:
                self._add_result(
                    check_name=f"environment_variable_{var.lower()}",
                    status="fail",
                    severity="critical",
                    message=f"Critical environment variable {var} is not set",
                    recommendation=f"Set {var} environment variable with a secure value",
                    expected_value="secure_value",
                    current_value="not_set",
                )
            elif var in [
                "POSTGRES_PASSWORD",
                "JWT_SECRET_KEY",
                "WEBHOOK_SIGNING_SECRET",
            ]:
                # Validate password/key strength
                if len(value) < 32:
                    self._add_result(
                        check_name=f"environment_variable_{var.lower()}_strength",
                        status="fail",
                        severity="high",
                        message=f"{var} is too short (minimum 32 characters)",
                        recommendation=f"Use a longer, more secure value for {var}",
                        expected_value=">=32_characters",
                        current_value=f"{len(value)}_characters",
                    )
                elif value in ["default", "test", "development", "changeme"]:
                    self._add_result(
                        check_name=f"environment_variable_{var.lower()}_default",
                        status="fail",
                        severity="critical",
                        message=f"{var} appears to be using a default value",
                        recommendation=f"Change {var} to a unique, secure value",
                        current_value="default_value",
                        expected_value="unique_secure_value",
                    )
                else:
                    self._add_result(
                        check_name=f"environment_variable_{var.lower()}",
                        status="pass",
                        severity="critical",
                        message=f"{var} is properly configured",
                    )

        # Validate high priority variables
        for var in high_priority_vars:
            value = os.getenv(var)
            if not value:
                self._add_result(
                    check_name=f"environment_variable_{var.lower()}",
                    status="warning",
                    severity="high",
                    message=f"Important environment variable {var} is not set",
                    recommendation=f"Consider setting {var} for enhanced security",
                )
            else:
                self._add_result(
                    check_name=f"environment_variable_{var.lower()}",
                    status="pass",
                    severity="high",
                    message=f"{var} is configured",
                )

    def _validate_authentication_security(self):
        """Validate authentication configuration."""
        # API Key validation
        api_key = os.getenv("API_KEY")
        if api_key:
            # Check if API key is using a secure format
            if len(api_key) < 32:
                self._add_result(
                    check_name="api_key_length",
                    status="fail",
                    severity="high",
                    message="API key is too short",
                    recommendation="Use an API key with at least 32 characters",
                    current_value=f"{len(api_key)}_characters",
                    expected_value=">=32_characters",
                )
            else:
                self._add_result(
                    check_name="api_key_length",
                    status="pass",
                    severity="high",
                    message="API key length is adequate",
                )

        # JWT Secret validation
        jwt_secret = os.getenv("JWT_SECRET_KEY")
        if jwt_secret:
            # Check JWT secret entropy
            if self._check_entropy(jwt_secret) < 4.0:
                self._add_result(
                    check_name="jwt_secret_entropy",
                    status="warning",
                    severity="medium",
                    message="JWT secret key has low entropy",
                    recommendation="Use a more random JWT secret key",
                )
            else:
                self._add_result(
                    check_name="jwt_secret_entropy",
                    status="pass",
                    severity="medium",
                    message="JWT secret key has adequate entropy",
                )

        # API Key rotation configuration
        rotation_interval = os.getenv("API_KEY_ROTATION_INTERVAL_HOURS")
        if rotation_interval:
            try:
                hours = int(rotation_interval)
                if self.environment == "production" and hours > 168:  # 1 week
                    self._add_result(
                        check_name="api_key_rotation_interval",
                        status="warning",
                        severity="medium",
                        message="API key rotation interval is too long for production",
                        recommendation="Consider rotating API keys more frequently in production",
                        current_value=f"{hours}_hours",
                        expected_value="<=168_hours",
                    )
                else:
                    self._add_result(
                        check_name="api_key_rotation_interval",
                        status="pass",
                        severity="medium",
                        message="API key rotation interval is appropriate",
                    )
            except ValueError:
                self._add_result(
                    check_name="api_key_rotation_interval",
                    status="fail",
                    severity="low",
                    message="Invalid API key rotation interval",
                    recommendation="Set a valid numeric value for API_KEY_ROTATION_INTERVAL_HOURS",
                )

    def _validate_encryption_security(self):
        """Validate encryption configuration."""
        # Check for encryption seed
        encryption_seed = os.getenv("API_KEY_ENCRYPTION_SEED")
        if not encryption_seed:
            self._add_result(
                check_name="encryption_seed",
                status="fail",
                severity="high",
                message="API key encryption seed is not configured",
                recommendation="Set API_KEY_ENCRYPTION_SEED for secure key storage",
            )
        elif len(encryption_seed) < 32:
            self._add_result(
                check_name="encryption_seed_length",
                status="fail",
                severity="high",
                message="Encryption seed is too short",
                recommendation="Use an encryption seed with at least 32 characters",
                current_value=f"{len(encryption_seed)}_characters",
                expected_value=">=32_characters",
            )
        else:
            self._add_result(
                check_name="encryption_seed",
                status="pass",
                severity="high",
                message="Encryption seed is properly configured",
            )

    def _validate_network_security(self):
        """Validate network security configuration."""
        # Check if HTTPS is enforced
        if self.environment == "production":
            # In production, should not allow HTTP
            allow_http = os.getenv("ALLOW_HTTP", "false").lower()
            if allow_http == "true":
                self._add_result(
                    check_name="https_enforcement",
                    status="fail",
                    severity="critical",
                    message="HTTP is allowed in production environment",
                    recommendation="Disable HTTP and enforce HTTPS in production",
                    current_value="http_allowed",
                    expected_value="https_only",
                )
            else:
                self._add_result(
                    check_name="https_enforcement",
                    status="pass",
                    severity="critical",
                    message="HTTPS enforcement is properly configured",
                )

        # Validate allowed hosts
        allowed_hosts = os.getenv("ALLOWED_HOSTS")
        if self.environment == "production" and not allowed_hosts:
            self._add_result(
                check_name="allowed_hosts",
                status="warning",
                severity="medium",
                message="Allowed hosts not configured in production",
                recommendation="Configure ALLOWED_HOSTS for production security",
            )

    def _validate_database_security(self):
        """Validate database security configuration."""
        # Check SSL mode
        ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
        if self.environment == "production" and ssl_mode not in [
            "require",
            "verify-ca",
            "verify-full",
        ]:
            self._add_result(
                check_name="postgres_ssl_mode",
                status="fail",
                severity="high",
                message="PostgreSQL SSL mode is not secure for production",
                recommendation="Set POSTGRES_SSL_MODE to 'require', 'verify-ca', or 'verify-full'",
                current_value=ssl_mode,
                expected_value="require|verify-ca|verify-full",
            )
        else:
            self._add_result(
                check_name="postgres_ssl_mode",
                status="pass",
                severity="high",
                message="PostgreSQL SSL mode is properly configured",
            )

        # Check for certificate validation
        ssl_cert = os.getenv("POSTGRES_SSL_CERT")
        ssl_key = os.getenv("POSTGRES_SSL_KEY")
        ssl_ca = os.getenv("POSTGRES_SSL_CA")

        if self.environment == "production" and ssl_mode in [
            "verify-ca",
            "verify-full",
        ]:
            if not ssl_ca:
                self._add_result(
                    check_name="postgres_ssl_ca",
                    status="fail",
                    severity="high",
                    message="PostgreSQL SSL CA certificate not configured",
                    recommendation="Configure POSTGRES_SSL_CA for certificate validation",
                )
            else:
                self._add_result(
                    check_name="postgres_ssl_ca",
                    status="pass",
                    severity="high",
                    message="PostgreSQL SSL CA certificate is configured",
                )

    def _validate_kafka_security(self):
        """Validate Kafka security configuration."""
        kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")

        # Check for SSL in production
        if self.environment == "production":
            if "ssl://" not in kafka_servers and "sasl_ssl://" not in kafka_servers:
                self._add_result(
                    check_name="kafka_ssl",
                    status="warning",
                    severity="medium",
                    message="Kafka connection may not be using SSL in production",
                    recommendation="Use SSL-enabled Kafka brokers in production",
                    current_value=kafka_servers,
                )
            else:
                self._add_result(
                    check_name="kafka_ssl",
                    status="pass",
                    severity="medium",
                    message="Kafka connection is using SSL",
                )

        # Check for authentication
        kafka_username = os.getenv("KAFKA_USERNAME")
        kafka_password = os.getenv("KAFKA_PASSWORD")

        if self.environment == "production":
            if not kafka_username or not kafka_password:
                self._add_result(
                    check_name="kafka_authentication",
                    status="warning",
                    severity="medium",
                    message="Kafka authentication not configured",
                    recommendation="Configure Kafka authentication for production",
                )
            else:
                self._add_result(
                    check_name="kafka_authentication",
                    status="pass",
                    severity="medium",
                    message="Kafka authentication is configured",
                )

    def _validate_ssl_tls_configuration(self):
        """Validate SSL/TLS configuration."""
        # Check for SSL certificate configuration
        ssl_cert_path = os.getenv("SSL_CERT_PATH")
        ssl_key_path = os.getenv("SSL_KEY_PATH")

        if self.environment == "production":
            if not ssl_cert_path or not ssl_key_path:
                self._add_result(
                    check_name="ssl_certificates",
                    status="warning",
                    severity="high",
                    message="SSL certificates not configured",
                    recommendation="Configure SSL certificates for production",
                )
            else:
                # Check if certificate files exist
                if os.path.exists(ssl_cert_path) and os.path.exists(ssl_key_path):
                    self._add_result(
                        check_name="ssl_certificates",
                        status="pass",
                        severity="high",
                        message="SSL certificates are properly configured",
                    )
                else:
                    self._add_result(
                        check_name="ssl_certificates",
                        status="fail",
                        severity="high",
                        message="SSL certificate files not found",
                        recommendation="Ensure SSL certificate files exist at specified paths",
                    )

    def _validate_cors_configuration(self):
        """Validate CORS configuration."""
        cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")

        if self.environment == "production":
            if "*" in cors_origins:
                self._add_result(
                    check_name="cors_wildcard",
                    status="fail",
                    severity="high",
                    message="CORS wildcard origin allowed in production",
                    recommendation="Replace wildcard with specific allowed origins",
                    current_value="wildcard",
                    expected_value="specific_origins",
                )
            elif not cors_origins:
                self._add_result(
                    check_name="cors_origins",
                    status="fail",
                    severity="medium",
                    message="CORS origins not configured",
                    recommendation="Configure specific CORS origins for production",
                )
            else:
                # Check for HTTPS origins
                origins = [origin.strip() for origin in cors_origins.split(",")]
                http_origins = [
                    origin for origin in origins if origin.startswith("http://")
                ]

                if http_origins:
                    self._add_result(
                        check_name="cors_https",
                        status="warning",
                        severity="medium",
                        message="Non-HTTPS origins in CORS configuration",
                        recommendation="Use HTTPS origins only in production",
                        current_value=str(http_origins),
                    )
                else:
                    self._add_result(
                        check_name="cors_configuration",
                        status="pass",
                        severity="medium",
                        message="CORS configuration is secure",
                    )

    def _validate_rate_limiting_configuration(self):
        """Validate rate limiting configuration."""
        # Check rate limiting settings
        rate_limit_enabled = os.getenv("ENABLE_RATE_LIMITING", "true").lower()

        if rate_limit_enabled != "true":
            self._add_result(
                check_name="rate_limiting_enabled",
                status="fail",
                severity="high",
                message="Rate limiting is disabled",
                recommendation="Enable rate limiting for security protection",
                current_value="disabled",
                expected_value="enabled",
            )
        else:
            self._add_result(
                check_name="rate_limiting_enabled",
                status="pass",
                severity="high",
                message="Rate limiting is enabled",
            )

        # Check rate limits for production
        if self.environment == "production":
            hook_rate_limit = os.getenv("HOOK_PROCESSING_RATE_LIMIT")
            if hook_rate_limit:
                try:
                    limit = int(hook_rate_limit.split("/")[0])
                    if limit > 1000:  # More than 1000 requests
                        self._add_result(
                            check_name="rate_limit_strictness",
                            status="warning",
                            severity="medium",
                            message="Rate limit may be too permissive for production",
                            recommendation="Consider lowering rate limits in production",
                            current_value=hook_rate_limit,
                        )
                    else:
                        self._add_result(
                            check_name="rate_limit_strictness",
                            status="pass",
                            severity="medium",
                            message="Rate limit is appropriately configured",
                        )
                except (ValueError, IndexError):
                    self._add_result(
                        check_name="rate_limit_format",
                        status="warning",
                        severity="low",
                        message="Rate limit format may be invalid",
                        recommendation="Check rate limit configuration format",
                    )

    def _validate_audit_logging_configuration(self):
        """Validate audit logging configuration."""
        # Check if audit logging is enabled
        audit_enabled = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower()

        if audit_enabled != "true":
            self._add_result(
                check_name="audit_logging_enabled",
                status="fail",
                severity="critical",
                message="Audit logging is disabled",
                recommendation="Enable audit logging for security compliance",
                current_value="disabled",
                expected_value="enabled",
            )
        else:
            self._add_result(
                check_name="audit_logging_enabled",
                status="pass",
                severity="critical",
                message="Audit logging is enabled",
            )

        # Check log level
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if self.environment == "production" and log_level == "DEBUG":
            self._add_result(
                check_name="log_level_production",
                status="warning",
                severity="medium",
                message="Debug logging enabled in production",
                recommendation="Use INFO or WARNING log level in production",
                current_value=log_level,
                expected_value="INFO|WARNING|ERROR",
            )

    def _validate_api_security(self):
        """Validate API security configuration."""
        # Check for API documentation exposure
        if self.environment == "production":
            disable_docs = os.getenv("DISABLE_API_DOCS", "true").lower()
            if disable_docs != "true":
                self._add_result(
                    check_name="api_docs_exposure",
                    status="warning",
                    severity="medium",
                    message="API documentation may be exposed in production",
                    recommendation="Disable API documentation in production",
                    current_value="enabled",
                    expected_value="disabled",
                )
            else:
                self._add_result(
                    check_name="api_docs_exposure",
                    status="pass",
                    severity="medium",
                    message="API documentation is properly disabled",
                )

    def _validate_webhook_security(self):
        """Validate webhook security configuration."""
        # Check webhook signing
        webhook_secret = os.getenv("WEBHOOK_SIGNING_SECRET")
        if not webhook_secret:
            self._add_result(
                check_name="webhook_signing_secret",
                status="fail",
                severity="high",
                message="Webhook signing secret not configured",
                recommendation="Configure WEBHOOK_SIGNING_SECRET for webhook security",
            )
        elif webhook_secret in ["default", "test", "development"]:
            self._add_result(
                check_name="webhook_signing_secret_default",
                status="fail",
                severity="high",
                message="Webhook signing secret appears to be a default value",
                recommendation="Use a unique, secure webhook signing secret",
                current_value="default_value",
                expected_value="unique_secure_value",
            )
        else:
            self._add_result(
                check_name="webhook_signing_secret",
                status="pass",
                severity="high",
                message="Webhook signing secret is properly configured",
            )

        # Check webhook IP restrictions
        allowed_ips = os.getenv("WEBHOOK_ALLOWED_IPS")
        if self.environment == "production" and not allowed_ips:
            self._add_result(
                check_name="webhook_ip_restrictions",
                status="warning",
                severity="medium",
                message="Webhook IP restrictions not configured",
                recommendation="Configure WEBHOOK_ALLOWED_IPS for enhanced security",
            )

    def _check_entropy(self, value: str) -> float:
        """Calculate entropy of a string."""
        if not value:
            return 0.0

        # Calculate character frequency
        char_count = {}
        for char in value:
            char_count[char] = char_count.get(char, 0) + 1

        # Calculate entropy using Shannon entropy formula
        import math

        length = len(value)
        entropy = 0.0
        for count in char_count.values():
            probability = count / length
            entropy -= probability * math.log2(probability) if probability > 0 else 0

        return entropy

    def _generate_recommendations(self) -> list[str]:
        """Generate prioritized recommendations based on validation results."""
        recommendations = []

        # Critical failures first
        critical_failures = [
            r for r in self.results if r.status == "fail" and r.severity == "critical"
        ]
        if critical_failures:
            recommendations.append(
                "CRITICAL: Address critical security failures immediately:",
            )
            for result in critical_failures:
                if result.recommendation:
                    recommendations.append(f"  - {result.recommendation}")

        # High severity issues
        high_severity_issues = [
            r
            for r in self.results
            if r.status in ["fail", "warning"] and r.severity == "high"
        ]
        if high_severity_issues:
            recommendations.append(
                "HIGH PRIORITY: Address high severity security issues:",
            )
            for result in high_severity_issues:
                if result.recommendation:
                    recommendations.append(f"  - {result.recommendation}")

        # Environment-specific recommendations
        if self.environment == "production":
            recommendations.extend(
                [
                    "PRODUCTION SECURITY:",
                    "  - Ensure all secrets are unique and not default values",
                    "  - Use HTTPS for all external communications",
                    "  - Configure proper SSL/TLS certificates",
                    "  - Enable comprehensive audit logging",
                    "  - Restrict CORS to specific origins",
                    "  - Use strong authentication and encryption",
                ],
            )

        # General best practices
        recommendations.extend(
            [
                "GENERAL BEST PRACTICES:",
                "  - Regularly rotate API keys and secrets",
                "  - Monitor security logs for suspicious activity",
                "  - Keep security configurations up to date",
                "  - Review and test security configurations regularly",
            ],
        )

        return recommendations

    def export_report(self, format: str = "json") -> str:
        """Export validation report in specified format.

        Args:
            format: Export format ('json', 'yaml', 'text')

        Returns:
            Formatted report string
        """
        config = self.validate_all()

        if format.lower() == "json":
            return config.model_dump_json(indent=2)
        elif format.lower() == "text":
            return self._format_text_report(config)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _format_text_report(self, config: SecurityConfiguration) -> str:
        """Format validation report as text."""
        lines = [
            "OmniNode Bridge Security Validation Report",
            "=" * 50,
            f"Environment: {config.environment}",
            f"Service: {config.service_name}",
            f"Validation Time: {config.validation_timestamp}",
            "",
            "SUMMARY",
            "-------",
            f"Security Score: {config.security_score:.1f}%",
            f"Compliance Level: {config.compliance_level.upper()}",
            f"Total Checks: {config.total_checks}",
            f"Passed: {config.passed_checks}",
            f"Failed: {config.failed_checks}",
            f"Warnings: {config.warnings}",
            "",
            "VALIDATION RESULTS",
            "------------------",
        ]

        # Group results by severity
        for severity in ["critical", "high", "medium", "low", "info"]:
            severity_results = [r for r in config.results if r.severity == severity]
            if severity_results:
                lines.append(f"\n{severity.upper()} SEVERITY:")
                for result in severity_results:
                    status_icon = (
                        "✓"
                        if result.status == "pass"
                        else "✗" if result.status == "fail" else "⚠"
                    )
                    lines.append(
                        f"  {status_icon} {result.check_name}: {result.message}",
                    )
                    if result.recommendation and result.status != "pass":
                        lines.append(f"    → {result.recommendation}")

        # Add recommendations
        if config.recommendations:
            lines.extend(
                [
                    "",
                    "RECOMMENDATIONS",
                    "---------------",
                ],
            )
            lines.extend(config.recommendations)

        return "\n".join(lines)


def create_security_validator(
    environment: str = None,
    service_name: str = "omninode_bridge",
    **kwargs,
) -> ProductionSecurityValidator:
    """Create security validator with environment-specific configuration.

    Args:
        environment: Deployment environment
        service_name: Service name for audit logging
        **kwargs: Additional validator options

    Returns:
        ProductionSecurityValidator instance
    """
    return ProductionSecurityValidator(
        environment=environment,
        service_name=service_name,
        **kwargs,
    )


def validate_production_security() -> SecurityConfiguration:
    """Quick function to validate production security configuration.

    Returns:
        SecurityConfiguration with validation results
    """
    validator = create_security_validator()
    return validator.validate_all()


def export_security_report(format: str = "json") -> str:
    """Export security validation report.

    Args:
        format: Export format ('json', 'text')

    Returns:
        Formatted security report
    """
    validator = create_security_validator()
    return validator.export_report(format)
