"""
DuckDB Catalog Manager
Handles DuckDB database initialization, Parquet file registration, and query execution.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBCatalog:
    """Manages DuckDB connection and table catalog."""

    def __init__(
        self,
        database_path: str = "data/catalog.duckdb",
        parquet_dir: str = "data/processed",
        memory_limit: str = "2GB",
        threads: int = 4,
        auto_register: bool = False,
    ):
        """
        Initialize DuckDB catalog.

        Args:
            database_path: Path to DuckDB database file (use :memory: for in-memory)
            parquet_dir: Directory containing Parquet files
            memory_limit: Memory limit for DuckDB
            threads: Number of threads for query execution
        """
        self.database_path = database_path
        self.parquet_dir = Path(parquet_dir)
        self.connection = None
        self.memory_limit = memory_limit
        self.threads = threads
        self.tables = {}

        # Initialize connection
        self._initialize_connection()

        # Optionally register Parquet files automatically (disabled by default)
        if auto_register and self.parquet_dir.exists():
            self.register_parquet_files()

    def _initialize_connection(self):
        """Initialize DuckDB connection with configuration."""
        try:
            self.connection = duckdb.connect(
                database=self.database_path,
                read_only=False
            )

            # Configure DuckDB
            self.connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self.connection.execute(f"SET threads TO {self.threads}")

            # Enable Parquet parallel reading
            self.connection.execute("SET enable_object_cache=true")

            logger.info(f"DuckDB connection initialized: {self.database_path}")

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB connection: {e}")
            raise

    def register_parquet_file(self, parquet_path: str, table_name: Optional[str] = None) -> str:
        """
        Register a Parquet file as a DuckDB view.

        Args:
            parquet_path: Path to Parquet file
            table_name: Table name (default: filename without extension)

        Returns:
            Name of registered table
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if table_name is None:
            table_name = parquet_path.stem
            # Prefer suffix after first underscore, so files like "<uuid>_<sheet>" map to the sheet name
            if "_" in table_name:
                table_name = table_name.split("_", 1)[1]

        # Clean table name (DuckDB naming rules)
        #  - replace invalid characters with underscore
        #  - ensure it does not start with a digit by prefixing 't_'
        cleaned = table_name.replace('-', '_').replace('.', '_')
        cleaned = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in cleaned)
        if not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = f"t_{cleaned}"
        table_name = cleaned

        try:
            # Create view from Parquet file
            query = f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_parquet('{str(parquet_path).replace(chr(92), '/')}')
            """

            self.connection.execute(query)

            # Store table info
            self.tables[table_name] = {
                "parquet_path": str(parquet_path),
                "registered_at": datetime.now().isoformat()
            }

            logger.info(f"Registered table: {table_name} from {parquet_path.name}")

            return table_name

        except Exception as e:
            logger.error(f"Error registering Parquet file {parquet_path.name}: {e}")
            raise

    def register_parquet_files(self, pattern: str = "*.parquet") -> List[str]:
        """
        Register all Parquet files in the parquet directory.

        Args:
            pattern: Glob pattern for Parquet files

        Returns:
            List of registered table names
        """
        if not self.parquet_dir.exists():
            logger.warning(f"Parquet directory not found: {self.parquet_dir}")
            return []

        parquet_files = list(self.parquet_dir.glob(pattern))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {self.parquet_dir}")
            return []

        logger.info(f"Registering {len(parquet_files)} Parquet files...")

        registered_tables = []

        for parquet_file in parquet_files:
            try:
                table_name = self.register_parquet_file(str(parquet_file))
                registered_tables.append(table_name)
            except Exception as e:
                logger.error(f"Failed to register {parquet_file.name}: {e}")
                continue

        logger.info(f"✓ Registered {len(registered_tables)} tables")

        return registered_tables

    def list_tables(self) -> List[str]:
        """
        Get list of all registered tables.

        Returns:
            List of table names
        """
        try:
            result = self.connection.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of column dictionaries with name and type
        """
        try:
            query = f"DESCRIBE {table_name}"
            result = self.connection.execute(query).fetchdf()

            columns = [
                {
                    "name": row["column_name"],
                    "type": row["column_type"],
                    "null": row["null"]
                }
                for _, row in result.iterrows()
            ]

            return columns

        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        try:
            # Get row count
            count_result = self.connection.execute(
                f"SELECT COUNT(*) as count FROM {table_name}"
            ).fetchone()
            row_count = count_result[0] if count_result else 0

            # Get schema
            schema = self.get_table_schema(table_name)

            # Get sample data
            sample_df = self.connection.execute(
                f"SELECT * FROM {table_name} LIMIT 5"
            ).fetchdf()

            return {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(schema),
                "columns": schema,
                "sample_data": sample_df.to_dict(orient='records'),
                "parquet_path": self.tables.get(table_name, {}).get("parquet_path")
            }

        except Exception as e:
            logger.error(f"Error getting info for {table_name}: {e}")
            return {}

    def execute_query(
        self,
        query: str,
        timeout: int = 10,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute a SQL query with safety limits.

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds
            max_rows: Maximum number of rows to return

        Returns:
            Dictionary with query results
        """
        start_time = datetime.now()

        try:
            # Add LIMIT if not present
            if "LIMIT" not in query.upper():
                query = query.rstrip(';') + f" LIMIT {max_rows}"

            # Set query timeout (if supported by DuckDB version)
            try:
                self.connection.execute(f"SET query_timeout={timeout * 1000}")  # milliseconds
            except Exception:
                # Older DuckDB versions don't support query_timeout
                pass

            # Execute query
            result_df = self.connection.execute(query).fetchdf()

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "rows": result_df.to_dict(orient='records'),
                "row_count": len(result_df),
                "column_names": result_df.columns.tolist(),
                "execution_time_seconds": execution_time,
                "query": query
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"Query execution failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "query": query
            }

    def get_catalog_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tables in catalog.

        Returns:
            Dictionary with catalog summary
        """
        tables = self.list_tables()

        summary = {
            "table_count": len(tables),
            "tables": {}
        }

        for table_name in tables:
            try:
                info = self.get_table_info(table_name)
                summary["tables"][table_name] = {
                    "row_count": info.get("row_count", 0),
                    "column_count": info.get("column_count", 0),
                    "columns": [c["name"] for c in info.get("columns", [])]
                }
            except Exception as e:
                logger.error(f"Error getting summary for {table_name}: {e}")
                continue

        return summary

    def close(self):
        """Close DuckDB connection."""
        if self.connection:
            self.connection.close()
            logger.info("DuckDB connection closed")


def main():
    """Example usage and CLI entry point."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize catalog
    catalog = DuckDBCatalog()

    # Print summary
    summary = catalog.get_catalog_summary()

    print("\n=== DuckDB Catalog Summary ===")
    print(f"Total tables: {summary['table_count']}")
    print("\nTables:")
    for table_name, info in summary['tables'].items():
        print(f"  - {table_name}: {info['row_count']} rows, {info['column_count']} columns")
        print(f"    Columns: {', '.join(info['columns'][:5])}" +
              ("..." if len(info['columns']) > 5 else ""))

    # Interactive query mode if no args
    if len(sys.argv) == 1:
        print("\n=== Interactive Query Mode ===")
        print("Enter SQL queries (or 'quit' to exit):\n")

        while True:
            try:
                query = input("SQL> ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query:
                    continue

                result = catalog.execute_query(query)

                if result["success"]:
                    print(f"\n✓ Query executed in {result['execution_time_seconds']:.3f}s")
                    print(f"Rows returned: {result['row_count']}")

                    if result['rows']:
                        df = pd.DataFrame(result['rows'])
                        print(df.to_string())
                else:
                    print(f"\n✗ Error: {result['error']}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    catalog.close()


if __name__ == "__main__":
    main()
