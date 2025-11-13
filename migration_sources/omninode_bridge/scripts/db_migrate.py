#!/usr/bin/env python3
"""Database migration management script for OmniNode Bridge.

This script provides safe database migration operations with validation,
backup, and rollback capabilities for production deployments.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psycopg2


class DatabaseMigrationManager:
    """Manages database migrations with safety checks and rollback capabilities."""

    def __init__(self, database_url: str | None = None):
        """Initialize migration manager with database connection."""
        self.database_url = database_url or self._get_database_url()
        self.backup_dir = Path("backups/database")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try different environment variable names
        url = (
            os.getenv("OMNINODE_BRIDGE_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'your_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5436')}/"
            f"{os.getenv('POSTGRES_DATABASE', 'omninode_bridge')}"
        )

        if not url or "your_password" in url:
            raise ValueError(
                "Database URL not configured. Set OMNINODE_BRIDGE_DATABASE_URL "
                "or individual POSTGRES_* environment variables.",
            )

        return url

    def validate_connection(self) -> bool:
        """Validate database connection."""
        print("🔍 Validating database connection...")
        try:
            conn = psycopg2.connect(self.database_url)
            conn.close()
            print("✅ Database connection validated")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def create_backup(self, backup_name: str | None = None) -> str:
        """Create database backup before migration."""
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"omninode_bridge_backup_{timestamp}.sql"

        backup_path = self.backup_dir / backup_name

        print(f"💾 Creating database backup: {backup_path}")

        # Extract connection parameters from URL
        import urllib.parse

        parsed = urllib.parse.urlparse(self.database_url)

        cmd = [
            "pg_dump",
            f"--host={parsed.hostname}",
            f"--port={parsed.port or 5432}",
            f"--username={parsed.username}",
            f"--dbname={parsed.path.lstrip('/')}",
            "--verbose",
            "--clean",
            "--if-exists",
            "--create",
            f"--file={backup_path}",
        ]

        # Set password via environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = parsed.password

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"✅ Backup created successfully: {backup_path}")
                return str(backup_path)
            else:
                print(f"❌ Backup failed: {result.stderr}")
                return ""
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return ""

    def validate_migrations(self) -> bool:
        """Validate migration scripts syntax and dependencies."""
        print("🔍 Validating migration scripts...")

        try:
            # Check Alembic configuration
            result = subprocess.run(
                ["alembic", "check"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )

            if result.returncode == 0:
                print("✅ Migration scripts validated")
                return True
            else:
                print(f"❌ Migration validation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Migration validation failed: {e}")
            return False

    def get_current_revision(self) -> str | None:
        """Get current database revision."""
        try:
            result = subprocess.run(
                ["alembic", "current", "--verbose"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )

            if result.returncode == 0:
                # Parse output to extract revision
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "Rev:" in line:
                        return line.split("Rev:")[-1].strip()
                return "empty"  # No migrations applied
            return None
        except Exception as e:
            print(f"❌ Failed to get current revision: {e}")
            return None

    def apply_migrations(
        self,
        target_revision: str = "head",
        dry_run: bool = False,
    ) -> bool:
        """Apply database migrations with safety checks."""
        if not self.validate_connection():
            return False

        if not self.validate_migrations():
            return False

        current_revision = self.get_current_revision()
        print(f"📊 Current revision: {current_revision}")
        print(f"🎯 Target revision: {target_revision}")

        if dry_run:
            print("🔍 Dry run mode - showing SQL that would be executed:")
            cmd = ["alembic", "upgrade", target_revision, "--sql"]
        else:
            # Create backup before applying migrations
            backup_path = self.create_backup()
            if not backup_path:
                print("❌ Backup failed - aborting migration")
                return False

            print(f"🚀 Applying migrations to {target_revision}...")
            cmd = ["alembic", "upgrade", target_revision]

        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                text=True,
                capture_output=True,
                check=False,
            )

            if result.returncode == 0:
                if dry_run:
                    print("✅ Dry run completed successfully")
                    print("SQL that would be executed:")
                    print(result.stdout)
                else:
                    print("✅ Migrations applied successfully")
                return True
            else:
                print(f"❌ Migration failed: {result.stderr}")
                if not dry_run:
                    print(f"💡 Restore from backup: {backup_path}")
                return False
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

    def rollback_migration(self, target_revision: str) -> bool:
        """Rollback to specific revision."""
        if not self.validate_connection():
            return False

        current_revision = self.get_current_revision()
        print(f"📊 Current revision: {current_revision}")
        print(f"⏪ Rolling back to: {target_revision}")

        # Create backup before rollback
        backup_path = self.create_backup()
        if not backup_path:
            print("❌ Backup failed - aborting rollback")
            return False

        try:
            result = subprocess.run(
                ["alembic", "downgrade", target_revision],
                cwd=Path(__file__).parent.parent,
                text=True,
                capture_output=True,
                check=False,
            )

            if result.returncode == 0:
                print("✅ Rollback completed successfully")
                return True
            else:
                print(f"❌ Rollback failed: {result.stderr}")
                print(f"💡 Restore from backup: {backup_path}")
                return False
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False

    def show_migration_history(self) -> None:
        """Show migration history."""
        try:
            result = subprocess.run(
                ["alembic", "history", "--verbose"],
                cwd=Path(__file__).parent.parent,
                text=True,
                capture_output=True,
                check=False,
            )

            if result.returncode == 0:
                print("📚 Migration History:")
                print(result.stdout)
            else:
                print(f"❌ Failed to get history: {result.stderr}")
        except Exception as e:
            print(f"❌ Failed to get history: {e}")

    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore database from backup."""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return False

        print(f"🔄 Restoring database from backup: {backup_path}")

        # Extract connection parameters from URL
        import urllib.parse

        parsed = urllib.parse.urlparse(self.database_url)

        cmd = [
            "psql",
            f"--host={parsed.hostname}",
            f"--port={parsed.port or 5432}",
            f"--username={parsed.username}",
            f"--dbname={parsed.path.lstrip('/')}",
            f"--file={backup_path}",
        ]

        # Set password via environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = parsed.password

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print("✅ Database restored successfully")
                return True
            else:
                print(f"❌ Restore failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False


def main():
    """Main entry point for database migration management."""
    parser = argparse.ArgumentParser(
        description="OmniNode Bridge Database Migration Manager",
    )

    parser.add_argument(
        "command",
        choices=["migrate", "rollback", "history", "validate", "backup", "restore"],
        help="Migration command to execute",
    )

    parser.add_argument(
        "--target",
        default="head",
        help="Target revision for migrate/rollback (default: head)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show SQL that would be executed without applying changes",
    )

    parser.add_argument(
        "--backup-path",
        help="Path to backup file for restore operation",
    )

    parser.add_argument(
        "--database-url",
        help="Database URL (overrides environment variables)",
    )

    args = parser.parse_args()

    try:
        manager = DatabaseMigrationManager(args.database_url)

        if args.command == "migrate":
            success = manager.apply_migrations(args.target, args.dry_run)
        elif args.command == "rollback":
            if args.target == "head":
                print("❌ Target revision required for rollback")
                sys.exit(1)
            success = manager.rollback_migration(args.target)
        elif args.command == "history":
            manager.show_migration_history()
            success = True
        elif args.command == "validate":
            success = manager.validate_migrations() and manager.validate_connection()
        elif args.command == "backup":
            backup_path = manager.create_backup()
            success = bool(backup_path)
        elif args.command == "restore":
            if not args.backup_path:
                print("❌ Backup path required for restore operation")
                sys.exit(1)
            success = manager.restore_from_backup(args.backup_path)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"❌ Migration manager failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
