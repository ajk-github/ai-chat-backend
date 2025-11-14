"""
Database Connection Manager
Manages multiple MySQL database connections and catalogs.
"""
import os
import logging
from typing import Dict, Optional
from data_processing.mysql_catalog import MySQLCatalog

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages multiple database connections."""
    
    _instance: Optional['DatabaseManager'] = None
    _catalogs: Dict[str, MySQLCatalog] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database manager - catalogs are created lazily on first use."""
        pass
    
    def get_catalog(self, database_id: str) -> MySQLCatalog:
        """
        Get or create MySQL catalog for a specific database.
        
        Database configurations are read from environment variables:
        - DB_{DATABASE_ID}_HOST
        - DB_{DATABASE_ID}_USER
        - DB_{DATABASE_ID}_PASSWORD
        - DB_{DATABASE_ID}_DATABASE
        - DB_{DATABASE_ID}_PORT (optional, defaults to 3306)
        
        Example for database_id="TELOS":
        - DB_TELOS_HOST
        - DB_TELOS_USER
        - DB_TELOS_PASSWORD
        - DB_TELOS_DATABASE
        - DB_TELOS_PORT
        
        Args:
            database_id: Database identifier (e.g., "TELOS", "CLIENT1", etc.)
            
        Returns:
            MySQLCatalog instance for the specified database
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Normalize database_id to uppercase for environment variable lookup
        db_id_upper = database_id.upper()
        
        # Check if catalog already exists
        if db_id_upper in self._catalogs:
            return self._catalogs[db_id_upper]
        
        # Get database credentials from environment
        db_host = os.getenv(f"DB_{db_id_upper}_HOST")
        db_user = os.getenv(f"DB_{db_id_upper}_USER")
        db_password = os.getenv(f"DB_{db_id_upper}_PASSWORD")
        db_name = os.getenv(f"DB_{db_id_upper}_DATABASE")
        db_port = int(os.getenv(f"DB_{db_id_upper}_PORT", "3306"))
        
        # Validate required credentials
        if not all([db_host, db_user, db_password, db_name]):
            missing = [k for k, v in {
                f"DB_{db_id_upper}_HOST": db_host,
                f"DB_{db_id_upper}_USER": db_user,
                f"DB_{db_id_upper}_PASSWORD": db_password,
                f"DB_{db_id_upper}_DATABASE": db_name,
            }.items() if not v]
            raise ValueError(
                f"Missing required database environment variables for '{database_id}': {', '.join(missing)}. "
                f"Please set them in your .env file or environment."
            )
        
        # Create catalog instance
        catalog = MySQLCatalog(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port,
            pool_size=5,
            max_overflow=10,
        )
        
        # Cache the catalog
        self._catalogs[db_id_upper] = catalog
        logger.info(f"MySQL catalog initialized for database '{database_id}' at {db_host}/{db_name}")
        
        return catalog
    
    def list_available_databases(self) -> list[str]:
        """
        List all available database configurations from environment variables.
        
        Returns:
            List of database IDs that have complete configurations
        """
        available = []
        env_vars = os.environ
        
        # Find all DB_*_HOST variables
        db_ids = set()
        for key in env_vars:
            if key.startswith("DB_") and key.endswith("_HOST"):
                # Extract database ID (e.g., "DB_TELOS_HOST" -> "TELOS")
                db_id = key[3:-5]  # Remove "DB_" prefix and "_HOST" suffix
                db_ids.add(db_id)
        
        # Check which ones have complete configurations
        for db_id in db_ids:
            if all([
                os.getenv(f"DB_{db_id}_HOST"),
                os.getenv(f"DB_{db_id}_USER"),
                os.getenv(f"DB_{db_id}_PASSWORD"),
                os.getenv(f"DB_{db_id}_DATABASE"),
            ]):
                available.append(db_id)
        
        return sorted(available)
    
    async def close_all(self):
        """Close all database connection pools."""
        for db_id, catalog in self._catalogs.items():
            try:
                # Use MySQLCatalog's close method if available
                if hasattr(catalog, 'close'):
                    await catalog.close()
                elif hasattr(catalog, 'pool') and catalog.pool:
                    catalog.pool.close()
                    await catalog.pool.wait_closed()
                logger.info(f"Closed connection pool for database '{db_id}'")
            except Exception as e:
                logger.warning(f"Error closing connection pool for database '{db_id}': {e}")
        
        self._catalogs.clear()


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


def get_mysql_catalog_for_database(database_id: str) -> MySQLCatalog:
    """
    Get MySQL catalog for a specific database.
    
    Args:
        database_id: Database identifier (e.g., "TELOS", "CLIENT1")
        
    Returns:
        MySQLCatalog instance
    """
    manager = get_database_manager()
    return manager.get_catalog(database_id)

