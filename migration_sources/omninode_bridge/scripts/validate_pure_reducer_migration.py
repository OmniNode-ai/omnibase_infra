#!/usr/bin/env python3
"""
Pure Reducer Migration Validation Script.

This script validates that the Pure Reducer migration completed successfully by:
1. Checking database schema (tables, indexes, constraints)
2. Validating service health (canonical store, projection materializer, etc.)
3. Verifying data consistency (canonical vs projection counts)
4. Testing basic workflow operations
5. Checking performance metrics

Usage:
    python scripts/validate_pure_reducer_migration.py

Exit Codes:
    0 - All validations passed
    1 - One or more validations failed
    2 - Critical error (e.g., database unreachable)
"""

import asyncio
import os
import sys
from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

import asyncpg
import httpx

# Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),  # Load from environment
    "database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
}

SERVICE_URLS = {
    "orchestrator": "http://localhost:8053",
    "canonical_store": "http://localhost:8080",
    "projection_materializer": "http://localhost:8081",
    "reducer_service": "http://localhost:8082",
}

# ANSI color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"  {text}")


class MigrationValidator:
    """Validates Pure Reducer migration completion."""

    def __init__(self):
        self.conn: Optional[asyncpg.Connection] = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.validation_errors = []
        self.validation_warnings = []

    async def connect_database(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            self.conn = await asyncpg.connect(**POSTGRES_CONFIG)
            print_success("Connected to PostgreSQL database")
            return True
        except Exception as e:
            print_error(f"Failed to connect to database: {e}")
            return False

    async def close(self) -> None:
        """Close connections."""
        if self.conn:
            await self.conn.close()
        await self.http_client.aclose()

    async def validate_schema(self) -> bool:
        """Validate database schema changes."""
        print_header("Database Schema Validation")

        required_tables = [
            "workflow_state",
            "workflow_projection",
            "projection_watermarks",
            "action_dedup_log",
        ]

        all_valid = True

        for table in required_tables:
            exists = await self.conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                table,
            )

            if exists:
                print_success(f"Table '{table}' exists")
            else:
                print_error(f"Table '{table}' missing")
                self.validation_errors.append(f"Missing table: {table}")
                all_valid = False

        # Validate workflow_state structure
        print_info("Validating workflow_state schema...")
        columns = await self.conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'workflow_state'
            ORDER BY ordinal_position
            """
        )

        expected_columns = {
            "workflow_key": "text",
            "version": "bigint",
            "state": "jsonb",
            "updated_at": "timestamp with time zone",
            "schema_version": "integer",
            "provenance": "jsonb",
        }

        for col_name, col_type in expected_columns.items():
            col = next((c for c in columns if c["column_name"] == col_name), None)
            if col:
                if col_type in col["data_type"]:
                    print_success(f"  Column '{col_name}' ({col['data_type']})")
                else:
                    print_error(
                        f"  Column '{col_name}' has wrong type: {col['data_type']} (expected {col_type})"
                    )
                    all_valid = False
            else:
                print_error(f"  Column '{col_name}' missing")
                all_valid = False

        # Validate indexes
        print_info("Validating indexes...")
        indexes = await self.conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename IN ('workflow_state', 'workflow_projection', 'action_dedup_log')
            """
        )

        required_indexes = [
            "idx_workflow_state_version",
            "idx_workflow_state_updated",
            "idx_projection_namespace",
            "idx_projection_tag",
            "idx_action_dedup_expires",
        ]

        for idx in required_indexes:
            if any(i["indexname"] == idx for i in indexes):
                print_success(f"  Index '{idx}' exists")
            else:
                print_warning(f"  Index '{idx}' missing (performance may be impacted)")
                self.validation_warnings.append(f"Missing index: {idx}")

        return all_valid

    async def validate_services(self) -> bool:
        """Validate service health."""
        print_header("Service Health Validation")

        all_healthy = True

        for service_name, service_url in SERVICE_URLS.items():
            try:
                response = await self.http_client.get(f"{service_url}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get("status", "unknown")

                    if status == "healthy":
                        print_success(f"Service '{service_name}' is healthy")

                        # Check orchestrator mode
                        if service_name == "orchestrator":
                            mode = health_data.get("mode", "unknown")
                            if mode == "event_driven":
                                print_success(
                                    "  Orchestrator running in event-driven mode"
                                )
                            elif mode == "legacy":
                                print_warning(
                                    "  Orchestrator running in legacy mode (EventBus unavailable?)"
                                )
                                self.validation_warnings.append(
                                    "Orchestrator in legacy mode"
                                )
                            else:
                                print_warning(f"  Orchestrator mode: {mode}")
                    else:
                        print_error(f"Service '{service_name}' is unhealthy: {status}")
                        all_healthy = False
                else:
                    print_error(
                        f"Service '{service_name}' returned status {response.status_code}"
                    )
                    all_healthy = False
            except httpx.ConnectError:
                print_error(
                    f"Service '{service_name}' is not reachable at {service_url}"
                )
                self.validation_errors.append(f"Service unreachable: {service_name}")
                all_healthy = False
            except Exception as e:
                print_error(f"Error checking service '{service_name}': {e}")
                all_healthy = False

        return all_healthy

    async def validate_data_consistency(self) -> bool:
        """Validate data consistency between canonical and projection stores."""
        print_header("Data Consistency Validation")

        all_consistent = True

        # Check record counts
        canonical_count = await self.conn.fetchval(
            "SELECT COUNT(*) FROM workflow_state"
        )
        projection_count = await self.conn.fetchval(
            "SELECT COUNT(*) FROM workflow_projection"
        )

        print_info(f"Canonical store records: {canonical_count}")
        print_info(f"Projection store records: {projection_count}")

        if canonical_count == projection_count:
            print_success("Record counts match")
        else:
            diff = abs(canonical_count - projection_count)
            if diff <= 5:  # Allow small lag
                print_warning(
                    f"Record count mismatch: {diff} records (may be projection lag)"
                )
                self.validation_warnings.append(f"Record count lag: {diff}")
            else:
                print_error(f"Significant record count mismatch: {diff} records")
                self.validation_errors.append(f"Record count mismatch: {diff}")
                all_consistent = False

        # Check version consistency
        version_mismatches = await self.conn.fetch(
            """
            SELECT
                ws.workflow_key,
                ws.version AS canonical_version,
                wp.version AS projection_version,
                ws.version - wp.version AS version_lag
            FROM workflow_state ws
            LEFT JOIN workflow_projection wp USING (workflow_key)
            WHERE ws.version != wp.version
            ORDER BY version_lag DESC
            LIMIT 10
            """
        )

        if version_mismatches:
            max_lag = max(m["version_lag"] for m in version_mismatches)
            print_warning(f"Found {len(version_mismatches)} workflows with version lag")
            print_info(f"  Max version lag: {max_lag}")

            if max_lag > 10:
                print_error("Excessive version lag detected (>10 versions)")
                self.validation_errors.append(f"Excessive version lag: {max_lag}")
                all_consistent = False
            else:
                print_warning("Acceptable version lag (eventual consistency)")
        else:
            print_success("All versions consistent")

        # Check for orphaned projections
        orphaned = await self.conn.fetchval(
            """
            SELECT COUNT(*)
            FROM workflow_projection wp
            LEFT JOIN workflow_state ws USING (workflow_key)
            WHERE ws.workflow_key IS NULL
            """
        )

        if orphaned > 0:
            print_error(f"Found {orphaned} orphaned projection records")
            self.validation_errors.append(f"Orphaned projections: {orphaned}")
            all_consistent = False
        else:
            print_success("No orphaned projection records")

        return all_consistent

    async def validate_workflow_operations(self) -> bool:
        """Validate basic workflow operations."""
        print_header("Workflow Operations Validation")

        all_valid = True

        # Test workflow creation
        test_namespace = f"test-migration-{uuid4()}"

        try:
            response = await self.http_client.post(
                f"{SERVICE_URLS['orchestrator']}/stamp",
                json={
                    "content": "Migration validation test content",
                    "file_path": f"/test/migration-{uuid4()}.txt",
                    "namespace": test_namespace,
                },
            )

            if response.status_code == 200:
                result = response.json()
                workflow_key = result.get("workflow_key")
                file_hash = result.get("file_hash")

                print_success("Workflow creation successful")
                print_info(f"  Workflow key: {workflow_key}")
                print_info(f"  File hash: {file_hash}")

                # Verify canonical state created
                await asyncio.sleep(0.5)  # Allow for async processing

                canonical_state = await self.conn.fetchrow(
                    "SELECT workflow_key, version FROM workflow_state WHERE workflow_key = $1",
                    workflow_key,
                )

                if canonical_state:
                    print_success("Canonical state created")
                    print_info(f"  Version: {canonical_state['version']}")
                else:
                    print_error("Canonical state NOT created")
                    self.validation_errors.append("Canonical state not created")
                    all_valid = False

                # Verify projection created (eventual consistency - may lag)
                await asyncio.sleep(1.0)  # Allow projection to materialize

                projection = await self.conn.fetchrow(
                    "SELECT workflow_key, version, tag FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )

                if projection:
                    print_success("Projection created")
                    print_info(f"  Version: {projection['version']}")
                    print_info(f"  Tag: {projection['tag']}")
                else:
                    print_warning("Projection not yet created (may be projection lag)")
                    self.validation_warnings.append("Projection lag detected")

            else:
                print_error(f"Workflow creation failed: HTTP {response.status_code}")
                print_info(f"  Response: {response.text}")
                self.validation_errors.append(
                    f"Workflow creation failed: {response.status_code}"
                )
                all_valid = False

        except Exception as e:
            print_error(f"Workflow operation test failed: {e}")
            self.validation_errors.append(f"Workflow test failed: {e}")
            all_valid = False

        return all_valid

    async def validate_metrics(self) -> bool:
        """Validate performance metrics."""
        print_header("Metrics Validation")

        all_valid = True

        # Check orchestrator metrics
        try:
            response = await self.http_client.get(
                f"{SERVICE_URLS['orchestrator']}/metrics"
            )
            if response.status_code == 200:
                metrics = response.text

                # Check for Pure Reducer metrics
                expected_metrics = [
                    "canonical_store_state_commits_total",
                    "canonical_store_state_conflicts_total",
                    "canonical_store_commit_latency_ms",
                ]

                for metric in expected_metrics:
                    if metric in metrics:
                        print_success(f"Metric '{metric}' present")
                    else:
                        print_warning(
                            f"Metric '{metric}' not found (may not be used yet)"
                        )

            else:
                print_warning(f"Metrics endpoint returned {response.status_code}")

        except Exception as e:
            print_warning(f"Could not validate metrics: {e}")

        return all_valid

    async def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print_header("Pure Reducer Migration Validation")
        print_info(f"Validation started at: {datetime.now(UTC).isoformat()}")

        try:
            # Connect to database
            if not await self.connect_database():
                return False

            # Run validations
            schema_valid = await self.validate_schema()
            services_valid = await self.validate_services()
            data_valid = await self.validate_data_consistency()
            workflow_valid = await self.validate_workflow_operations()
            metrics_valid = await self.validate_metrics()

            # Summary
            print_header("Validation Summary")

            all_valid = (
                schema_valid
                and services_valid
                and data_valid
                and workflow_valid
                and metrics_valid
            )

            if all_valid and not self.validation_errors:
                print_success("✓ All validations PASSED")
            else:
                print_error("✗ Some validations FAILED")

            if self.validation_errors:
                print("\nErrors:")
                for error in self.validation_errors:
                    print_error(f"  {error}")

            if self.validation_warnings:
                print("\nWarnings:")
                for warning in self.validation_warnings:
                    print_warning(f"  {warning}")

            print_info(f"\nValidation completed at: {datetime.now(UTC).isoformat()}")

            return all_valid and not self.validation_errors

        finally:
            await self.close()


async def main() -> int:
    """Main entry point."""
    validator = MigrationValidator()

    try:
        success = await validator.run_all_validations()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        return 2
    except Exception as e:
        print_error(f"Critical error during validation: {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
