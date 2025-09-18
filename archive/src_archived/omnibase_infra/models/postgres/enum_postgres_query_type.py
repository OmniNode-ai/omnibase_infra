"""PostgreSQL query type enumeration."""

from enum import Enum


class EnumPostgresQueryType(str, Enum):
    """PostgreSQL query type enumeration."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"  # Data Definition Language (CREATE, DROP, ALTER, etc.)
    DCL = "dcl"  # Data Control Language (GRANT, REVOKE, etc.)
    TCL = "tcl"  # Transaction Control Language (COMMIT, ROLLBACK, etc.)
    GENERAL = "general"  # General/mixed queries
