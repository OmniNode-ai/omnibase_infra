#!/usr/bin/env python3
"""
Metadata Stamping Service Configuration Validator
Validates docker-compose configuration and environment setup
"""

import subprocess
import sys
from pathlib import Path

import yaml


class ConfigValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docker_compose_file = project_root / "docker-compose.metadata-stamping.yml"
        self.env_file = project_root / ".env.metadata-stamping"
        self.main_compose_file = project_root / "docker-compose.yml"

        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def validate_docker_compose_syntax(self) -> bool:
        """Validate docker-compose syntax"""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose_file), "config"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.info.append("✅ Docker-compose syntax is valid")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"❌ Docker-compose syntax error: {e.stderr}")
            return False
        except FileNotFoundError:
            self.errors.append("❌ docker-compose command not found")
            return False

    def validate_docker_compose_structure(self) -> bool:
        """Validate docker-compose file structure"""
        if not self.docker_compose_file.exists():
            self.errors.append(
                f"❌ Docker-compose file not found: {self.docker_compose_file}"
            )
            return False

        try:
            with open(self.docker_compose_file) as f:
                config = yaml.safe_load(f)

            # Check for version attribute (should be removed in new format)
            if "version" in config:
                self.warnings.append(
                    "⚠️  'version' attribute found - consider removing for latest Docker Compose"
                )

            # Check services
            if "services" not in config:
                self.errors.append("❌ No 'services' section found")
                return False

            # Check metadata-stamping service
            if "metadata-stamping" not in config["services"]:
                self.errors.append("❌ 'metadata-stamping' service not found")
                return False

            service = config["services"]["metadata-stamping"]

            # Validate required sections
            required_sections = [
                "build",
                "environment",
                "ports",
                "networks",
                "depends_on",
                "healthcheck",
            ]
            for section in required_sections:
                if section not in service:
                    self.errors.append(
                        f"❌ Missing required section '{section}' in metadata-stamping service"
                    )

            # Check environment variables
            env_vars = service.get("environment", {})
            required_env_vars = [
                "METADATA_STAMPING_SERVICE_HOST",
                "METADATA_STAMPING_SERVICE_PORT",
                "METADATA_STAMPING_DB_HOST",
                "METADATA_STAMPING_DB_PASSWORD",
            ]

            for var in required_env_vars:
                if var not in env_vars:
                    self.errors.append(
                        f"❌ Missing required environment variable: {var}"
                    )

            # Check port configuration
            ports = service.get("ports", [])
            if len(ports) < 2:
                self.warnings.append(
                    "⚠️  Expected at least 2 port mappings (service + metrics)"
                )

            # Check network configuration
            networks = service.get("networks", [])
            if "omninode-bridge-network" not in networks:
                self.errors.append(
                    "❌ Service not connected to 'omninode-bridge-network'"
                )

            # Check external network definition
            if "networks" in config:
                network_config = config["networks"].get("omninode-bridge-network", {})
                if not network_config.get("external", False):
                    self.warnings.append(
                        "⚠️  Network should be marked as external: true"
                    )

            self.info.append("✅ Docker-compose structure validation passed")
            return len(self.errors) == 0

        except yaml.YAMLError as e:
            self.errors.append(f"❌ YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Validation error: {e}")
            return False

    def validate_environment_file(self) -> bool:
        """Validate environment file"""
        if not self.env_file.exists():
            self.warnings.append(f"⚠️  Environment file not found: {self.env_file}")
            return True  # Not critical for validation

        try:
            with open(self.env_file) as f:
                content = f.read()

            # Check for key environment variables
            required_vars = [
                "METADATA_STAMPING_SERVICE_NAME",
                "METADATA_STAMPING_EXTERNAL_PORT",
                "METADATA_STAMPING_METRICS_PORT",
            ]

            for var in required_vars:
                if f"{var}=" not in content:
                    self.warnings.append(f"⚠️  Missing environment variable: {var}")

            # Check for port conflicts
            if "METADATA_STAMPING_EXTERNAL_PORT=8053" in content:
                self.warnings.append(
                    "⚠️  Port 8053 may conflict with archon-intelligence service"
                )

            self.info.append("✅ Environment file validation passed")
            return True

        except Exception as e:
            self.errors.append(f"❌ Environment file validation error: {e}")
            return False

    def validate_dockerfile_exists(self) -> bool:
        """Check if Dockerfile exists"""
        dockerfile_paths = [
            self.project_root
            / "src/omninode_bridge/services/metadata_stamping/Dockerfile",
            self.project_root / "docker/metadata-stamping/Dockerfile",
        ]

        for dockerfile in dockerfile_paths:
            if dockerfile.exists():
                self.info.append(f"✅ Dockerfile found: {dockerfile}")
                return True

        self.errors.append("❌ No Dockerfile found for metadata stamping service")
        return False

    def validate_topic_integration(self) -> bool:
        """Validate Kafka topic integration"""
        if not self.main_compose_file.exists():
            self.warnings.append(
                "⚠️  Main docker-compose.yml not found - cannot validate topic integration"
            )
            return True

        try:
            with open(self.main_compose_file) as f:
                content = f.read()

            # Check for metadata stamping topics
            metadata_topics = [
                "metadata-stamp-created",
                "metadata-stamp-validated",
                "metadata-batch-processed",
            ]

            for topic in metadata_topics:
                if topic not in content:
                    self.warnings.append(
                        f"⚠️  Metadata topic '{topic}' not found in main docker-compose"
                    )

            if "metadata" in content:
                self.info.append("✅ Metadata topics found in main docker-compose")

            return True

        except Exception as e:
            self.errors.append(f"❌ Topic validation error: {e}")
            return False

    def validate_service_dependencies(self) -> bool:
        """Validate service dependencies"""
        required_services = ["postgres", "redpanda", "consul"]

        if not self.main_compose_file.exists():
            self.warnings.append(
                "⚠️  Cannot validate service dependencies - main docker-compose not found"
            )
            return True

        try:
            with open(self.main_compose_file) as f:
                config = yaml.safe_load(f)

            services = config.get("services", {})

            for service in required_services:
                if service not in services:
                    self.errors.append(
                        f"❌ Required dependency service '{service}' not found in main docker-compose"
                    )
                else:
                    self.info.append(f"✅ Dependency service '{service}' found")

            return len(self.errors) == 0

        except Exception as e:
            self.errors.append(f"❌ Service dependency validation error: {e}")
            return False

    def run_validation(self) -> bool:
        """Run all validations"""
        print("🔍 Validating Metadata Stamping Service Configuration...\n")

        validations = [
            ("Docker Compose File Structure", self.validate_docker_compose_structure),
            ("Docker Compose Syntax", self.validate_docker_compose_syntax),
            ("Environment File", self.validate_environment_file),
            ("Dockerfile Existence", self.validate_dockerfile_exists),
            ("Kafka Topic Integration", self.validate_topic_integration),
            ("Service Dependencies", self.validate_service_dependencies),
        ]

        all_passed = True
        for name, validation_func in validations:
            print(f"📋 {name}...")
            try:
                result = validation_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.errors.append(f"❌ {name} validation failed: {e}")
                all_passed = False

        return all_passed

    def print_results(self):
        """Print validation results"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION RESULTS")
        print("=" * 60)

        if self.info:
            print("\n✅ SUCCESS:")
            for msg in self.info:
                print(f"  {msg}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for msg in self.warnings:
                print(f"  {msg}")

        if self.errors:
            print("\n❌ ERRORS:")
            for msg in self.errors:
                print(f"  {msg}")

        print("\n📈 SUMMARY:")
        print(f"  ✅ Successes: {len(self.info)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  ❌ Errors: {len(self.errors)}")

        if self.errors:
            print(f"\n🚨 VALIDATION FAILED - {len(self.errors)} error(s) found")
            return False
        elif self.warnings:
            print(
                f"\n⚠️  VALIDATION PASSED WITH WARNINGS - {len(self.warnings)} warning(s)"
            )
            return True
        else:
            print("\n🎉 VALIDATION PASSED - All checks successful!")
            return True


def main():
    """Main validation function"""
    project_root = Path(__file__).parent.parent

    print("🔧 Metadata Stamping Service Configuration Validator")
    print(f"📁 Project Root: {project_root}")
    print("-" * 60)

    validator = ConfigValidator(project_root)

    # Run validation
    validation_passed = validator.run_validation()

    # Print results
    success = validator.print_results()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
