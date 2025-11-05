"""
Chat Data Manager
Manages per-chat data storage, file processing, and DuckDB catalog isolation.
"""
#src/data_processing/chat_data_manager.py
import logging
import shutil
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import OrderedDict

from data_processing.duckdb_catalog import DuckDBCatalog
from data_processing.excel_converter import ExcelConverter
from data_processing.csv_converter import CSVConverter
from data_processing.schema_profiler import SchemaProfiler
from data_processing.file_validator import FileValidator, FileValidationError

logger = logging.getLogger(__name__)


class ChatDataManager:
    """Manages data storage and processing for individual chat sessions."""

    # Maximum number of catalogs to cache (prevents memory leaks)
    MAX_CACHE_SIZE = 50

    def __init__(self, base_data_dir: str = "data/chats", max_cache_size: int = None):
        """
        Initialize Chat Data Manager.

        Args:
            base_data_dir: Base directory for all chat data
            max_cache_size: Maximum number of catalogs to cache (default: 50)
        """
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)

        self.file_validator = FileValidator()
        self.excel_converter = ExcelConverter()
        self.csv_converter = CSVConverter()
        self.schema_profiler = SchemaProfiler()

        # LRU cache for active catalogs (OrderedDict for LRU eviction)
        # OrderedDict maintains insertion order, oldest entries are evicted first
        self.max_cache_size = max_cache_size or self.MAX_CACHE_SIZE
        self._catalog_cache: OrderedDict[str, DuckDBCatalog] = OrderedDict()

    def create_chat_workspace(self, chat_id: str) -> Dict[str, str]:
        """
        Create directory structure for a new chat.

        Args:
            chat_id: Chat session ID

        Returns:
            Dictionary with created directory paths
        """
        chat_dir = self.base_data_dir / chat_id

        # Create directory structure
        raw_dir = chat_dir / "raw"
        processed_dir = chat_dir / "processed"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata_path = chat_dir / "metadata.json"
        metadata = {
            "chat_id": chat_id,
            "created_at": datetime.now().isoformat(),
            "files": [],
            "total_storage_bytes": 0
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created workspace for chat: {chat_id}")

        return {
            "chat_dir": str(chat_dir),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "catalog_path": str(chat_dir / "catalog.duckdb"),
            "metadata_path": str(metadata_path)
        }

    def process_uploaded_file(
        self,
        chat_id: str,
        file_path: str,
        original_filename: str,
        current_file_count: int = 0,
        current_user_storage: int = 0
    ) -> Dict[str, Any]:
        """
        Process uploaded file and convert to Parquet.

        Args:
            chat_id: Chat session ID
            file_path: Temporary path to uploaded file
            original_filename: Original filename from upload
            current_file_count: Number of files already in chat
            current_user_storage: Total user storage in bytes

        Returns:
            Dictionary with processing results

        Raises:
            FileValidationError: If validation fails
        """
        logger.info(f"Processing file for chat {chat_id}: {original_filename}")

        # Validate file
        validation_result = self.file_validator.validate_file(
            file_path=file_path,
            original_filename=original_filename,
            current_chat_file_count=current_file_count,
            current_user_storage=current_user_storage
        )

        # Create workspace if it doesn't exist
        chat_dir = self.base_data_dir / chat_id
        if not chat_dir.exists():
            self.create_chat_workspace(chat_id)

        # Generate file ID
        file_id = str(uuid.uuid4())
        sanitized_name = validation_result['sanitized_filename']
        extension = validation_result['extension']

        # Move file to raw directory
        raw_path = chat_dir / "raw" / f"{file_id}{extension}"
        shutil.copy2(file_path, raw_path)

        logger.info(f"File saved to: {raw_path}")

        # Convert to Parquet
        processed_dir = chat_dir / "processed"

        try:
            if extension in ['.xlsb', '.xlsx', '.xls']:
                # Use Excel converter
                result = self._process_excel_file(raw_path, processed_dir, file_id)

            elif extension in ['.csv', '.tsv']:
                # Use CSV converter
                delimiter = '\t' if extension == '.tsv' else None
                result = self._process_csv_file(raw_path, processed_dir, file_id, delimiter)

            elif extension == '.parquet':
                # Direct Parquet file - just copy and profile
                result = self._process_parquet_file(raw_path, processed_dir, file_id)

            else:
                raise ValueError(f"Unsupported file format: {extension}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            # Cleanup raw file on error
            if raw_path.exists():
                raw_path.unlink()
            raise

        # Register tables in DuckDB catalog
        catalog = self.get_chat_catalog(chat_id)
        for table in result['tables']:
            try:
                catalog.register_parquet_file(
                    parquet_path=table['parquet_path'],
                    table_name=table['table_name']
                )
            except Exception as e:
                logger.error(f"Error registering table {table['table_name']}: {e}")

        # Delete raw file after successful conversion (we only need Parquet files)
        try:
            if raw_path.exists():
                raw_path.unlink()
                logger.info(f"Deleted raw file after successful conversion: {raw_path.name}")
        except Exception as e:
            logger.warning(f"Failed to delete raw file {raw_path.name}: {e}")
            # Don't fail the whole operation if cleanup fails

        # Update metadata
        self._update_metadata(chat_id, {
            "file_id": file_id,
            "original_name": original_filename,
            "sanitized_name": sanitized_name,
            "extension": extension,
            "size_bytes": validation_result['size_bytes'],
            "uploaded_at": datetime.now().isoformat(),
            "tables": [t['table_name'] for t in result['tables']],
            "status": "completed"
        })

        logger.info(f"✓ File processed successfully: {original_filename}")

        return {
            "file_id": file_id,
            "original_name": original_filename,
            "tables_created": result['tables'],
            "total_rows": result['total_rows'],
            "total_size_bytes": validation_result['size_bytes']
        }

    def _process_excel_file(
        self,
        file_path: Path,
        output_dir: Path,
        file_id: str
    ) -> Dict[str, Any]:
        """Process Excel file to Parquet."""
        self.excel_converter.output_dir = output_dir

        # Convert all sheets
        result = self.excel_converter.convert_excel_file(str(file_path))

        # Profile each sheet
        tables = []
        total_rows = 0

        for sheet_result in result['sheets']:
            parquet_path = sheet_result['parquet_path']

            # Generate schema profile
            profile_path = self.schema_profiler.profile_parquet(parquet_path)

            tables.append({
                "table_name": sheet_result['table_name'],
                "parquet_path": parquet_path,
                "schema_path": profile_path,
                "row_count": sheet_result['row_count'],
                "column_count": sheet_result['column_count']
            })

            total_rows += sheet_result['row_count']

        return {
            "tables": tables,
            "total_rows": total_rows,
            "sheet_count": len(tables)
        }

    def _process_csv_file(
        self,
        file_path: Path,
        output_dir: Path,
        file_id: str,
        delimiter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process CSV file to Parquet."""
        self.csv_converter.output_dir = output_dir

        # Use file_id as table name for CSV (single table)
        table_name = f"table_{file_id.replace('-', '_')}"

        result = self.csv_converter.convert_to_parquet(
            csv_path=str(file_path),
            table_name=table_name,
            delimiter=delimiter
        )

        # Profile the table
        profile_path = self.schema_profiler.profile_parquet(result['parquet_path'])

        tables = [{
            "table_name": result['table_name'],
            "parquet_path": result['parquet_path'],
            "schema_path": profile_path,
            "row_count": result['row_count'],
            "column_count": result['column_count']
        }]

        return {
            "tables": tables,
            "total_rows": result['row_count'],
            "sheet_count": 1
        }

    def _process_parquet_file(
        self,
        file_path: Path,
        output_dir: Path,
        file_id: str
    ) -> Dict[str, Any]:
        """Process existing Parquet file (just copy and profile)."""
        table_name = f"table_{file_id.replace('-', '_')}"
        parquet_path = output_dir / f"{table_name}.parquet"

        # Copy to processed directory
        shutil.copy2(file_path, parquet_path)

        # Profile
        profile_path = self.schema_profiler.profile_parquet(str(parquet_path))

        # Get row count
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(parquet_path))
        row_count = pf.metadata.num_rows
        column_count = pf.metadata.num_columns

        tables = [{
            "table_name": table_name,
            "parquet_path": str(parquet_path),
            "schema_path": profile_path,
            "row_count": row_count,
            "column_count": column_count
        }]

        return {
            "tables": tables,
            "total_rows": row_count,
            "sheet_count": 1
        }

    def get_chat_catalog(self, chat_id: str) -> DuckDBCatalog:
        """
        Get or create DuckDB catalog for chat.
        
        Uses LRU cache with automatic eviction to prevent memory leaks.
        When cache is full, oldest entries are evicted and their connections closed.

        Args:
            chat_id: Chat session ID

        Returns:
            DuckDBCatalog instance for this chat
        """
        # Check cache first - move to end (most recently used)
        if chat_id in self._catalog_cache:
            # Move to end (mark as recently used)
            catalog = self._catalog_cache.pop(chat_id)
            self._catalog_cache[chat_id] = catalog
            logger.debug(f"Cache hit for chat: {chat_id}")
            return catalog

        # Ensure workspace exists
        chat_dir = self.base_data_dir / chat_id
        if not chat_dir.exists():
            self.create_chat_workspace(chat_id)

        # Create new catalog
        catalog_path = chat_dir / "catalog.duckdb"
        processed_dir = chat_dir / "processed"

        catalog = DuckDBCatalog(
            database_path=str(catalog_path),
            parquet_dir=str(processed_dir)
        )

        # Evict oldest entries if cache is full
        if len(self._catalog_cache) >= self.max_cache_size:
            self._evict_oldest_catalog()

        # Add to cache (at end, most recently used)
        self._catalog_cache[chat_id] = catalog

        logger.info(f"Initialized catalog for chat: {chat_id} (cache size: {len(self._catalog_cache)}/{self.max_cache_size})")

        return catalog

    def _evict_oldest_catalog(self):
        """
        Evict the oldest catalog from cache and close its connection.
        
        This prevents memory leaks by ensuring connections are properly closed
        when catalogs are removed from cache.
        """
        if not self._catalog_cache:
            return

        # Get oldest entry (first in OrderedDict)
        chat_id, catalog = self._catalog_cache.popitem(last=False)
        
        try:
            # Close DuckDB connection before evicting
            catalog.close()
            logger.info(f"Evicted and closed catalog for chat: {chat_id} (cache was full)")
        except Exception as e:
            logger.warning(f"Error closing catalog connection for chat {chat_id} during eviction: {e}")
            # Continue eviction even if close fails

    def close_all_catalogs(self):
        """
        Close all cached DuckDB catalog connections.
        
        Call this when shutting down or when you want to ensure all connections are closed.
        """
        closed_count = 0
        for chat_id, catalog in list(self._catalog_cache.items()):
            try:
                catalog.close()
                closed_count += 1
            except Exception as e:
                logger.warning(f"Error closing catalog for chat {chat_id}: {e}")
        
        self._catalog_cache.clear()
        logger.info(f"Closed {closed_count} catalog connection(s)")

    def delete_chat_data(self, chat_id: str) -> bool:
        """
        Delete all data for a chat session.

        Args:
            chat_id: Chat session ID

        Returns:
            True if successful, False otherwise
        """
        chat_dir = self.base_data_dir / chat_id

        if not chat_dir.exists():
            logger.warning(f"Chat directory not found: {chat_id}")
            return False

        try:
            # Close catalog if cached in this instance and remove from cache
            if chat_id in self._catalog_cache:
                try:
                    catalog = self._catalog_cache.pop(chat_id)
                    catalog.close()
                    logger.info(f"Closed DuckDB catalog for chat {chat_id}")
                except Exception as conn_err:
                    logger.warning(f"Error closing catalog connection: {conn_err}")

            # Delete entire directory with retry logic for locked files
            # Windows may keep DuckDB files locked briefly after connection closes
            max_retries = 3
            retry_delay = 0.5  # seconds
            
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(chat_dir)
                    break
                except (PermissionError, OSError) as e:
                    # Check if it's a file lock error (EBUSY on Windows)
                    is_locked = (
                        isinstance(e, PermissionError) or 
                        (isinstance(e, OSError) and hasattr(e, 'winerror') and e.winerror == 32) or
                        (isinstance(e, OSError) and e.errno == 13)  # Permission denied
                    )
                    
                    if is_locked and attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries}: File locked, waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise

            logger.info(f"✓ Deleted all data for chat: {chat_id}")

            return True

        except PermissionError as pe:
            logger.error(f"Permission denied deleting chat data for {chat_id}: {pe}. File may be locked.")
            return False
        except Exception as e:
            logger.error(f"Error deleting chat data: {e}")
            return False

    def get_chat_storage_usage(self, chat_id: str) -> int:
        """
        Calculate total storage used by chat.

        Args:
            chat_id: Chat session ID

        Returns:
            Total bytes used
        """
        chat_dir = self.base_data_dir / chat_id

        if not chat_dir.exists():
            return 0

        total_bytes = 0

        for file in chat_dir.rglob('*'):
            if file.is_file():
                total_bytes += file.stat().st_size

        return total_bytes

    def list_chat_tables(self, chat_id: str) -> List[Dict[str, Any]]:
        """
        List all tables available in chat.

        Args:
            chat_id: Chat session ID

        Returns:
            List of table information
        """
        catalog = self.get_chat_catalog(chat_id)
        tables = catalog.list_tables()

        table_info = []

        for table_name in tables:
            info = catalog.get_table_info(table_name)
            if info:
                table_info.append({
                    "table_name": table_name,
                    "row_count": info.get("row_count", 0),
                    "column_count": info.get("column_count", 0),
                    "columns": [c["name"] for c in info.get("columns", [])]
                })

        return table_info

    def _update_metadata(self, chat_id: str, file_info: Dict[str, Any]):
        """Update metadata.json with new file information."""
        metadata_path = self.base_data_dir / chat_id / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"chat_id": chat_id, "files": [], "total_storage_bytes": 0}

        metadata["files"].append(file_info)
        metadata["total_storage_bytes"] = self.get_chat_storage_usage(chat_id)
        metadata["updated_at"] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cleanup_inactive_catalogs(self, keep_recent: int = 10):
        """
        Close and remove inactive catalogs from cache.

        Args:
            keep_recent: Number of recent catalogs to keep
        """
        if len(self._catalog_cache) <= keep_recent:
            return

        # Close and remove oldest catalogs
        chat_ids = list(self._catalog_cache.keys())
        to_remove = chat_ids[:-keep_recent]

        for chat_id in to_remove:
            try:
                self._catalog_cache[chat_id].close()
                del self._catalog_cache[chat_id]
                logger.info(f"Closed catalog for inactive chat: {chat_id}")
            except Exception as e:
                logger.error(f"Error closing catalog {chat_id}: {e}")


# Example CLI entry point removed; this module is used by the FastAPI app.
