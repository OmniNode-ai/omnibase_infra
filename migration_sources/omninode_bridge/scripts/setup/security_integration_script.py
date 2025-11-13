#!/usr/bin/env python3
"""
Comprehensive Security Integration Script for OmniNode Bridge

This script integrates all security features into the existing hook receiver
and provides comprehensive security validation and testing.
"""

import asyncio
import os
import sys
from typing import Any

from omninode_bridge.security import (  # Configuration validation; Authentication components; Security middleware; Request signing; Audit logging
    AuditEventType,
    AuditSeverity,
    create_jwt_authenticator,
    create_security_validator,
    create_webhook_security,
    export_security_report,
    get_api_key_manager,
    get_audit_logger,
    get_environment_cors_config,
    get_security_headers_config,
    setup_automatic_rotation,
)

# Imports are now fixed to use correct package paths


class SecurityIntegrationManager:
    """Manages the integration of all security features."""

    def __init__(self, environment: str = None):
        """Initialize security integration manager."""
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )
        self.service_name = "omninode_bridge_security_integration"
        self.audit_logger = get_audit_logger(self.service_name)

        print(
            f"🔒 Initializing Security Integration for environment: {self.environment}",
        )

    async def validate_security_configuration(self) -> dict[str, Any]:
        """Validate current security configuration."""
        print("\n📋 Validating Security Configuration...")

        try:
            validator = create_security_validator(
                environment=self.environment,
                service_name=self.service_name,
            )

            config = validator.validate_all()

            print("✅ Security validation completed")
            print(f"   Security Score: {config.security_score:.1f}%")
            print(f"   Compliance Level: {config.compliance_level.upper()}")
            print(
                f"   Checks: {config.passed_checks} passed, {config.failed_checks} failed, {config.warnings} warnings",
            )

            # Log critical failures
            critical_failures = [
                r
                for r in config.results
                if r.status == "fail" and r.severity == "critical"
            ]
            if critical_failures:
                print("\n❌ Critical Security Issues Found:")
                for failure in critical_failures:
                    print(f"   - {failure.message}")
                    if failure.recommendation:
                        print(f"     → {failure.recommendation}")

            # Log warnings
            high_warnings = [
                r
                for r in config.results
                if r.status == "warning" and r.severity in ["critical", "high"]
            ]
            if high_warnings:
                print("\n⚠️  High Priority Warnings:")
                for warning in high_warnings:
                    print(f"   - {warning.message}")

            return config.model_dump()

        except Exception as e:
            print(f"❌ Security validation failed: {e}")
            return {"error": str(e)}

    async def setup_authentication_security(self) -> dict[str, Any]:
        """Setup comprehensive authentication security."""
        print("\n🔐 Setting up Authentication Security...")

        results = {
            "api_key_manager": False,
            "jwt_authenticator": False,
            "auto_rotation": False,
        }

        try:
            # Setup API Key Manager
            print("   Setting up API Key Manager...")
            api_key_manager = await get_api_key_manager(self.service_name)
            if api_key_manager:
                results["api_key_manager"] = True
                print("   ✅ API Key Manager initialized")

                # Setup automatic rotation
                if self.environment == "production":
                    await setup_automatic_rotation(
                        self.service_name,
                        check_interval_hours=1,
                    )
                    results["auto_rotation"] = True
                    print("   ✅ Automatic key rotation enabled")

        except Exception as e:
            print(f"   ⚠️  API Key Manager setup failed: {e}")

        try:
            # Setup JWT Authenticator
            print("   Setting up JWT Authenticator...")
            jwt_authenticator = create_jwt_authenticator(
                environment=self.environment,
                service_name=self.service_name,
            )
            if jwt_authenticator:
                results["jwt_authenticator"] = True
                print("   ✅ JWT Authenticator initialized")

        except Exception as e:
            print(f"   ⚠️  JWT Authenticator setup failed: {e}")

        return results

    def setup_middleware_security(self) -> dict[str, Any]:
        """Setup security middleware components."""
        print("\n🛡️  Setting up Security Middleware...")

        results = {
            "security_headers": False,
            "cors_security": False,
            "webhook_security": False,
        }

        try:
            # Test security headers configuration
            print("   Configuring Security Headers...")
            headers_config = get_security_headers_config(self.environment)
            if headers_config:
                results["security_headers"] = True
                print("   ✅ Security Headers configuration ready")

        except Exception as e:
            print(f"   ⚠️  Security Headers setup failed: {e}")

        try:
            # Test CORS security configuration
            print("   Configuring CORS Security...")
            cors_config = get_environment_cors_config(self.environment)
            if cors_config:
                results["cors_security"] = True
                print("   ✅ CORS Security configuration ready")

                # Validate CORS origins
                cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
                if cors_origins:
                    origins = [origin.strip() for origin in cors_origins.split(",")]
                    print(f"   📝 Configured CORS origins: {len(origins)} origins")

                    if self.environment == "production":
                        # Check for security issues in production
                        if "*" in cors_origins:
                            print("   ⚠️  WARNING: Wildcard CORS origin in production!")

                        http_origins = [o for o in origins if o.startswith("http://")]
                        if http_origins:
                            print(
                                f"   ⚠️  WARNING: Non-HTTPS origins in production: {http_origins}",
                            )

        except Exception as e:
            print(f"   ⚠️  CORS Security setup failed: {e}")

        try:
            # Setup webhook security
            print("   Configuring Webhook Security...")
            signing_secret = os.getenv("WEBHOOK_SIGNING_SECRET", "")
            if signing_secret:
                webhook_security = create_webhook_security(
                    signing_secret=signing_secret,
                    service_name=self.service_name,
                )
                if webhook_security:
                    results["webhook_security"] = True
                    print("   ✅ Webhook Security initialized")

                    # Check IP restrictions
                    allowed_ips = os.getenv("WEBHOOK_ALLOWED_IPS", "")
                    if allowed_ips:
                        ip_list = [ip.strip() for ip in allowed_ips.split(",")]
                        print(
                            f"   📝 Webhook IP restrictions: {len(ip_list)} IPs allowed",
                        )
                    else:
                        print("   ⚠️  No IP restrictions configured for webhooks")
            else:
                print("   ⚠️  No webhook signing secret configured")

        except Exception as e:
            print(f"   ⚠️  Webhook Security setup failed: {e}")

        return results

    def validate_environment_security(self) -> dict[str, Any]:
        """Validate environment-specific security settings."""
        print(f"\n🌍 Validating {self.environment.title()} Environment Security...")

        validation_results = {
            "environment": self.environment,
            "required_vars_present": [],
            "missing_vars": [],
            "security_issues": [],
            "recommendations": [],
        }

        # Define required environment variables by environment
        if self.environment == "production":
            required_vars = [
                "POSTGRES_PASSWORD",
                "JWT_SECRET_KEY",
                "WEBHOOK_SIGNING_SECRET",
                "API_KEY_ENCRYPTION_SEED",
                "CORS_ALLOWED_ORIGINS",
            ]
            security_recommendations = [
                "Use HTTPS for all external communications",
                "Enable SSL for database connections",
                "Configure proper CORS origins (no wildcards)",
                "Set up proper SSL certificates",
                "Enable comprehensive audit logging",
                "Use strong, unique secrets (minimum 32 characters)",
            ]
        elif self.environment == "staging":
            required_vars = [
                "POSTGRES_PASSWORD",
                "JWT_SECRET_KEY",
                "WEBHOOK_SIGNING_SECRET",
            ]
            security_recommendations = [
                "Use staging-specific secrets (not production)",
                "Test SSL configurations",
                "Validate CORS settings for staging domains",
            ]
        else:  # development
            required_vars = [
                "POSTGRES_PASSWORD",
            ]
            security_recommendations = [
                "Use development-specific secrets",
                "Enable debug logging for troubleshooting",
                "Test security configurations locally",
            ]

        # Check required environment variables
        for var in required_vars:
            value = os.getenv(var)
            if value:
                validation_results["required_vars_present"].append(var)
                print(f"   ✅ {var} is set")

                # Additional security checks
                if var in [
                    "POSTGRES_PASSWORD",
                    "JWT_SECRET_KEY",
                    "WEBHOOK_SIGNING_SECRET",
                ]:
                    if len(value) < 32:
                        validation_results["security_issues"].append(
                            f"{var} is too short (< 32 characters)",
                        )
                        print(f"   ⚠️  {var} is too short for secure use")
                    elif value.lower() in [
                        "default",
                        "test",
                        "development",
                        "changeme",
                        "password",
                    ]:
                        validation_results["security_issues"].append(
                            f"{var} appears to be a default value",
                        )
                        print(f"   ⚠️  {var} appears to be using a default value")
            else:
                validation_results["missing_vars"].append(var)
                print(f"   ❌ {var} is missing")

        # Add environment-specific recommendations
        validation_results["recommendations"] = security_recommendations

        # Additional security checks
        if self.environment == "production":
            # Check SSL configuration
            ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
            if ssl_mode not in ["require", "verify-ca", "verify-full"]:
                validation_results["security_issues"].append(
                    "PostgreSQL SSL mode not secure for production",
                )
                print(
                    f"   ⚠️  PostgreSQL SSL mode '{ssl_mode}' not secure for production",
                )

            # Check CORS configuration
            cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
            if "*" in cors_origins:
                validation_results["security_issues"].append(
                    "CORS wildcard origin in production",
                )
                print("   ⚠️  CORS wildcard origin detected in production")

        return validation_results

    async def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        print("\n📊 Generating Comprehensive Security Report...")

        try:
            # Run security validation
            validator = create_security_validator(
                environment=self.environment,
                service_name=self.service_name,
            )

            # Export report
            report_path = export_security_report("text")
            print(f"   ✅ Security report generated: {report_path}")

            # Also generate JSON report for automation
            json_report = validator.export_report("json")
            json_path = f"security_report_{self.environment}.json"
            with open(json_path, "w") as f:
                f.write(json_report)
            print(f"   ✅ JSON security report generated: {json_path}")

            return report_path

        except Exception as e:
            print(f"   ❌ Failed to generate security report: {e}")
            return ""

    def print_security_summary(
        self,
        auth_results: dict[str, Any],
        middleware_results: dict[str, Any],
        env_results: dict[str, Any],
    ):
        """Print comprehensive security summary."""
        print("\n" + "=" * 60)
        print("🔒 COMPREHENSIVE SECURITY INTEGRATION SUMMARY")
        print("=" * 60)

        print(f"\n🌍 Environment: {self.environment.upper()}")

        # Authentication Summary
        print("\n🔐 Authentication Security:")
        print(
            f"   API Key Manager: {'✅' if auth_results.get('api_key_manager') else '❌'}",
        )
        print(
            f"   JWT Authenticator: {'✅' if auth_results.get('jwt_authenticator') else '❌'}",
        )
        print(
            f"   Auto Key Rotation: {'✅' if auth_results.get('auto_rotation') else '❌'}",
        )

        # Middleware Summary
        print("\n🛡️  Security Middleware:")
        print(
            f"   Security Headers: {'✅' if middleware_results.get('security_headers') else '❌'}",
        )
        print(
            f"   CORS Security: {'✅' if middleware_results.get('cors_security') else '❌'}",
        )
        print(
            f"   Webhook Security: {'✅' if middleware_results.get('webhook_security') else '❌'}",
        )

        # Environment Summary
        print("\n🌍 Environment Configuration:")
        print(
            f"   Required Variables: {len(env_results.get('required_vars_present', []))}/{len(env_results.get('required_vars_present', [])) + len(env_results.get('missing_vars', []))}",
        )
        print(f"   Security Issues: {len(env_results.get('security_issues', []))}")

        if env_results.get("missing_vars"):
            print("\n❌ Missing Variables:")
            for var in env_results["missing_vars"]:
                print(f"   - {var}")

        if env_results.get("security_issues"):
            print("\n⚠️  Security Issues:")
            for issue in env_results["security_issues"]:
                print(f"   - {issue}")

        if env_results.get("recommendations"):
            print("\n💡 Recommendations:")
            for rec in env_results["recommendations"][:5]:  # Show top 5
                print(f"   - {rec}")

        # Overall Status
        total_components = 6  # auth(3) + middleware(3)
        working_components = sum(
            [
                auth_results.get("api_key_manager", 0),
                auth_results.get("jwt_authenticator", 0),
                auth_results.get("auto_rotation", 0),
                middleware_results.get("security_headers", 0),
                middleware_results.get("cors_security", 0),
                middleware_results.get("webhook_security", 0),
            ],
        )

        security_percentage = (working_components / total_components) * 100
        missing_vars_count = len(env_results.get("missing_vars", []))
        security_issues_count = len(env_results.get("security_issues", []))

        print("\n📊 Overall Security Status:")
        print(
            f"   Components Working: {working_components}/{total_components} ({security_percentage:.0f}%)",
        )
        print(f"   Configuration Issues: {missing_vars_count + security_issues_count}")

        if (
            security_percentage >= 80
            and missing_vars_count == 0
            and security_issues_count == 0
        ):
            print("   Status: 🟢 EXCELLENT - Production Ready")
        elif security_percentage >= 60 and missing_vars_count <= 2:
            print("   Status: 🟡 GOOD - Minor issues to address")
        elif security_percentage >= 40:
            print("   Status: 🟠 FAIR - Multiple issues need attention")
        else:
            print("   Status: 🔴 CRITICAL - Major security gaps")

        print("\n" + "=" * 60)

    async def run_comprehensive_security_integration(self):
        """Run complete security integration process."""
        print("🚀 Starting Comprehensive Security Integration for OmniNode Bridge")
        print("=" * 70)

        try:
            # 1. Validate current configuration
            config_validation = await self.validate_security_configuration()

            # 2. Setup authentication
            auth_results = await self.setup_authentication_security()

            # 3. Setup middleware
            middleware_results = self.setup_middleware_security()

            # 4. Validate environment
            env_results = self.validate_environment_security()

            # 5. Generate reports
            report_path = await self.generate_security_report()

            # 6. Print summary
            self.print_security_summary(auth_results, middleware_results, env_results)

            # 7. Log completion
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_AUDIT,
                severity=AuditSeverity.LOW,
                additional_data={
                    "component": "security_integration",
                    "environment": self.environment,
                    "auth_components": sum(auth_results.values()),
                    "middleware_components": sum(middleware_results.values()),
                    "missing_vars": len(env_results.get("missing_vars", [])),
                    "security_issues": len(env_results.get("security_issues", [])),
                },
                message="Comprehensive security integration completed",
            )

            print("\n✅ Security integration completed successfully!")
            if report_path:
                print(f"📄 Detailed report: {report_path}")

            return True

        except Exception as e:
            print(f"\n❌ Security integration failed: {e}")

            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.CRITICAL,
                additional_data={
                    "component": "security_integration",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Security integration failed: {e}",
            )

            return False


async def main():
    """Main entry point for security integration."""
    import argparse

    parser = argparse.ArgumentParser(description="OmniNode Bridge Security Integration")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default=os.getenv("ENVIRONMENT", "development"),
        help="Deployment environment",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate security report without integration",
    )

    args = parser.parse_args()

    manager = SecurityIntegrationManager(environment=args.environment)

    if args.report_only:
        print("📊 Generating security report only...")
        config_validation = await manager.validate_security_configuration()
        report_path = await manager.generate_security_report()
        print(f"✅ Security report generated: {report_path}")
    else:
        success = await manager.run_comprehensive_security_integration()

        if not success:
            sys.exit(1)

    print("\n🔒 Security integration process completed!")


if __name__ == "__main__":
    asyncio.run(main())
