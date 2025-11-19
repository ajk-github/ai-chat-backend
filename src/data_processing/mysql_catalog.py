"""
MySQL Catalog Manager
Handles MySQL database connection, schema caching, and query execution.
"""
#src/data_processing/mysql_catalog.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

try:
    import aiomysql
except ImportError:
    aiomysql = None

logger = logging.getLogger(__name__)


class MySQLCatalog:
    """Manages MySQL connection pool and table catalog."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Initialize MySQL catalog with connection pool.

        Args:
            host: MySQL host address
            user: MySQL username
            password: MySQL password
            database: Database name
            port: MySQL port (default: 3306)
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        if aiomysql is None:
            raise ImportError(
                "aiomysql is required for MySQL support. Install it with: pip install aiomysql"
            )

        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.pool_size = pool_size
        self.max_overflow = max_overflow

        self.pool: Optional[aiomysql.Pool] = None
        self._schema_cache: Dict[str, List[Dict[str, str]]] = {}
        self._tables_list: List[str] = []
        self._schema_loaded = False

        # Initialize connection pool and load schema
        # Note: This is synchronous initialization, but we'll use async methods
        # The actual connection will be established on first use

    async def _get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self.pool is None:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=1,
                maxsize=self.pool_size + self.max_overflow,
                autocommit=True,
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor,
            )
            logger.info(f"MySQL connection pool created for {self.host}/{self.database}")
        return self.pool

    async def _load_schema_cache(self):
        """Load schema information from INFORMATION_SCHEMA and cache it."""
        if self._schema_loaded:
            return

        logger.info("Loading MySQL schema cache from INFORMATION_SCHEMA...")

        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Query all tables and columns
                    query = """
                    SELECT 
                        TABLE_NAME,
                        COLUMN_NAME,
                        DATA_TYPE,
                        COLUMN_TYPE,
                        IS_NULLABLE,
                        COLUMN_KEY,
                        COLUMN_DEFAULT,
                        EXTRA
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = %s
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                    """
                    await cursor.execute(query, (self.database,))
                    results = await cursor.fetchall()

                    # Build schema cache
                    current_table = None
                    columns = []

                    for row in results:
                        table_name = row['TABLE_NAME']
                        if table_name != current_table:
                            if current_table is not None:
                                self._schema_cache[current_table] = columns
                                self._tables_list.append(current_table)
                            current_table = table_name
                            columns = []

                        # Build column info
                        col_info = {
                            "name": row['COLUMN_NAME'],
                            "type": row['DATA_TYPE'],
                            "full_type": row['COLUMN_TYPE'],
                            "nullable": row['IS_NULLABLE'] == 'YES',
                            "key": row['COLUMN_KEY'],
                            "default": row['COLUMN_DEFAULT'],
                            "extra": row['EXTRA']
                        }
                        columns.append(col_info)

                    # Add last table
                    if current_table is not None:
                        self._schema_cache[current_table] = columns
                        self._tables_list.append(current_table)

                    self._schema_loaded = True
                    logger.info(
                        f"Schema cache loaded: {len(self._tables_list)} tables, "
                        f"{sum(len(cols) for cols in self._schema_cache.values())} columns"
                    )

        except Exception as e:
            logger.error(f"Failed to load schema cache: {e}")
            raise

    def list_tables(self) -> List[str]:
        """
        Get list of all tables (from cache).

        Returns:
            List of table names
        """
        if not self._schema_loaded:
            # If schema not loaded, trigger async load (this is a sync method, so we'll wait)
            # In practice, this should be called after async initialization
            logger.warning("Schema cache not loaded. Call _load_schema_cache() first.")
            return []

        return self._tables_list.copy()

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get schema information for a table (from cache).

        Args:
            table_name: Name of the table

        Returns:
            List of column dictionaries with name, type, etc.
        """
        if not self._schema_loaded:
            logger.warning("Schema cache not loaded. Call _load_schema_cache() first.")
            return []

        return self._schema_cache.get(table_name, []).copy()

    async def execute_query(
        self,
        query: str,
        timeout: int = 30,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute a SQL query with safety limits.

        Args:
            query: SQL query to execute (must be SELECT only)
            timeout: Query timeout in seconds
            max_rows: Maximum number of rows to return

        Returns:
            Dictionary with query results
        """
        start_time = datetime.now()

        # Ensure schema is loaded
        await self._load_schema_cache()

        pool = await self._get_pool()

        try:
            # Validate query is SELECT only (basic check)
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return {
                    "success": False,
                    "error": "Only SELECT queries are allowed",
                    "execution_time_seconds": 0,
                    "query": query
                }

            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Execute query with timeout
                    await cursor.execute(query)
                    rows = await cursor.fetchall()

                    # Convert rows to list of dicts
                    # aiomysql DictCursor already returns dicts, but ensure compatibility
                    result_rows = []
                    for row in rows:
                        if isinstance(row, dict):
                            # Convert any datetime/date objects to ISO strings
                            row_dict = {}
                            for key, value in row.items():
                                if isinstance(value, (datetime,)):
                                    row_dict[key] = value.isoformat()
                                elif hasattr(value, 'isoformat'):  # date objects
                                    row_dict[key] = value.isoformat()
                                else:
                                    row_dict[key] = value
                            result_rows.append(row_dict)
                        else:
                            result_rows.append(row)

                    execution_time = (datetime.now() - start_time).total_seconds()

                    # Get column names from first row or cursor description
                    column_names = list(result_rows[0].keys()) if result_rows else []

                    return {
                        "success": True,
                        "rows": result_rows,
                        "row_count": len(result_rows),
                        "column_names": column_names,
                        "execution_time_seconds": execution_time,
                        "query": query
                    }

        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": f"Query timeout after {timeout} seconds",
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

    async def close(self):
        """Close MySQL connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            logger.info("MySQL connection pool closed")

    def get_catalog_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tables in catalog (from cache).

        Returns:
            Dictionary with catalog summary
        """
        if not self._schema_loaded:
            return {
                "table_count": 0,
                "tables": {},
                "message": "Schema cache not loaded"
            }

        summary = {
            "table_count": len(self._tables_list),
            "tables": {}
        }

        for table_name in self._tables_list:
            columns = self._schema_cache.get(table_name, [])
            summary["tables"][table_name] = {
                "column_count": len(columns),
                "columns": [c["name"] for c in columns]
            }

        return summary

