"""
SQL Validator
Validates and sanitizes SQL queries for security and safety.
"""

import logging
import re
from typing import Dict, List, Optional, Any
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Token
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)


class SQLValidator:
    """Validates SQL queries for read-only access and safety."""

    def __init__(
        self,
        allowed_operations: List[str] = None,
        forbidden_keywords: List[str] = None,
        max_rows: int = 1000,
        max_joins: int = 3,
        pii_columns: List[str] = None
    ):
        """
        Initialize SQL validator.

        Args:
            allowed_operations: List of allowed SQL operations (default: SELECT only)
            forbidden_keywords: List of forbidden SQL keywords
            max_rows: Maximum allowed LIMIT value
            max_joins: Maximum number of joins allowed
            pii_columns: List of PII column names to block
        """
        self.allowed_operations = allowed_operations or ["SELECT"]
        self.forbidden_keywords = forbidden_keywords or [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"
        ]
        self.max_rows = max_rows
        self.max_joins = max_joins
        self.pii_columns = [col.lower() for col in (pii_columns or [])]

    def validate(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query against security rules.

        Args:
            sql: SQL query string

        Returns:
            Dictionary with validation result
        """
        # Clean SQL
        sql_clean = sql.strip()

        if not sql_clean:
            return {
                "valid": False,
                "error": "Empty SQL query",
                "sanitized_sql": None
            }

        # Parse SQL
        try:
            parsed = sqlparse.parse(sql_clean)

            if not parsed:
                return {
                    "valid": False,
                    "error": "Invalid SQL syntax",
                    "sanitized_sql": None
                }

            statement = parsed[0]

        except Exception as e:
            return {
                "valid": False,
                "error": f"SQL parsing error: {str(e)}",
                "sanitized_sql": None
            }

        # Check operation type
        operation_check = self._check_operation_type(statement)
        if not operation_check["valid"]:
            return operation_check

        # Check forbidden keywords
        forbidden_check = self._check_forbidden_keywords(sql_clean)
        if not forbidden_check["valid"]:
            return forbidden_check

        # Check for LIMIT clause
        limit_check = self._check_limit_clause(statement, sql_clean)
        if not limit_check["valid"]:
            return limit_check

        # Check number of joins
        join_check = self._check_joins(sql_clean)
        if not join_check["valid"]:
            return join_check

        # Check for PII columns
        pii_check = self._check_pii_columns(sql_clean)
        if not pii_check["valid"]:
            return pii_check

        # Get sanitized SQL (with LIMIT enforced)
        sanitized_sql = limit_check.get("sanitized_sql", sql_clean)

        return {
            "valid": True,
            "sanitized_sql": sanitized_sql,
            "warnings": limit_check.get("warnings", [])
        }

    def _check_operation_type(self, statement) -> Dict[str, Any]:
        """Check if SQL operation is allowed."""
        # Get first token (operation type)
        first_token = statement.token_first(skip_ws=True, skip_cm=True)

        if not first_token:
            return {
                "valid": False,
                "error": "Unable to determine SQL operation type"
            }

        operation = first_token.value.upper()

        if operation not in self.allowed_operations:
            return {
                "valid": False,
                "error": f"Operation '{operation}' not allowed. Only {', '.join(self.allowed_operations)} operations are permitted."
            }

        return {"valid": True}

    def _check_forbidden_keywords(self, sql: str) -> Dict[str, Any]:
        """Check for forbidden keywords."""
        sql_upper = sql.upper()

        for keyword in self.forbidden_keywords:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(keyword) + r'\b'

            if re.search(pattern, sql_upper):
                return {
                    "valid": False,
                    "error": f"Forbidden keyword '{keyword}' detected"
                }

        return {"valid": True}

    def _check_limit_clause(self, statement, sql: str) -> Dict[str, Any]:
        """Check and enforce LIMIT clause."""
        sql_upper = sql.upper()
        warnings = []

        # Check if LIMIT exists
        has_limit = "LIMIT" in sql_upper

        if not has_limit:
            # Add LIMIT clause
            sanitized_sql = sql.rstrip(';').rstrip() + f" LIMIT {self.max_rows}"
            warnings.append(f"Added LIMIT {self.max_rows} to query")

            return {
                "valid": True,
                "sanitized_sql": sanitized_sql,
                "warnings": warnings
            }

        # Extract LIMIT value
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)

        if limit_match:
            limit_value = int(limit_match.group(1))

            if limit_value > self.max_rows:
                # Replace with max allowed
                sanitized_sql = re.sub(
                    r'LIMIT\s+\d+',
                    f'LIMIT {self.max_rows}',
                    sql,
                    flags=re.IGNORECASE
                )

                warnings.append(
                    f"LIMIT reduced from {limit_value} to {self.max_rows}"
                )

                return {
                    "valid": True,
                    "sanitized_sql": sanitized_sql,
                    "warnings": warnings
                }

        return {
            "valid": True,
            "sanitized_sql": sql,
            "warnings": warnings
        }

    def _check_joins(self, sql: str) -> Dict[str, Any]:
        """Check number of JOIN clauses."""
        sql_upper = sql.upper()

        # Count different types of joins
        join_patterns = [
            r'\bJOIN\b',
            r'\bLEFT\s+JOIN\b',
            r'\bRIGHT\s+JOIN\b',
            r'\bINNER\s+JOIN\b',
            r'\bOUTER\s+JOIN\b',
            r'\bCROSS\s+JOIN\b'
        ]

        total_joins = 0

        for pattern in join_patterns:
            total_joins += len(re.findall(pattern, sql_upper))

        # Check for CROSS JOIN (usually forbidden)
        if re.search(r'\bCROSS\s+JOIN\b', sql_upper):
            return {
                "valid": False,
                "error": "CROSS JOIN is not allowed (potential cartesian product)"
            }

        if total_joins > self.max_joins:
            return {
                "valid": False,
                "error": f"Too many JOINs ({total_joins}). Maximum allowed: {self.max_joins}"
            }

        return {"valid": True}

    def _check_pii_columns(self, sql: str) -> Dict[str, Any]:
        """Check for PII columns in query."""
        if not self.pii_columns:
            return {"valid": True}

        sql_lower = sql.lower()

        for pii_col in self.pii_columns:
            # Check if PII column is referenced
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(pii_col) + r'\b'

            if re.search(pattern, sql_lower):
                return {
                    "valid": False,
                    "error": f"Query references PII column '{pii_col}' which is not allowed"
                }

        return {"valid": True}

    def extract_table_names(self, sql: str) -> List[str]:
        """
        Extract table names from SQL query.

        Args:
            sql: SQL query

        Returns:
            List of table names
        """
        try:
            parsed = sqlparse.parse(sql)

            if not parsed:
                return []

            tables = []

            for statement in parsed:
                tables.extend(self._extract_tables_from_statement(statement))

            return list(set(tables))

        except Exception as e:
            logger.error(f"Error extracting table names: {e}")
            return []

    def _extract_tables_from_statement(self, statement) -> List[str]:
        """Extract table names from a parsed statement."""
        tables = []

        from_seen = False

        for token in statement.tokens:
            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.append(identifier.get_name())
                elif isinstance(token, Identifier):
                    tables.append(token.get_name())

                from_seen = False

            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True

        return tables


def main():
    """Example usage."""
    import json

    validator = SQLValidator(
        max_rows=1000,
        max_joins=3,
        pii_columns=["ssn", "password", "credit_card"]
    )

    # Test queries
    test_queries = [
        "SELECT * FROM sales",
        "SELECT * FROM sales LIMIT 2000",
        "SELECT * FROM users WHERE password = 'test'",
        "INSERT INTO users VALUES ('test')",
        "SELECT a.*, b.* FROM sales a CROSS JOIN customers b",
        "SELECT * FROM sales WHERE date > '2024-01-01' LIMIT 100"
    ]

    print("=== SQL Validation Tests ===\n")

    for query in test_queries:
        print(f"Query: {query}")
        result = validator.validate(query)
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
