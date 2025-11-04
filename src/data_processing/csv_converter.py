"""
CSV Converter
Converts CSV/TSV files to Parquet format for efficient querying.
"""

import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CSVConverter:
    """Converts CSV and TSV files to Parquet format."""

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize CSV converter.

        Args:
            output_dir: Directory for output Parquet files
        """
        self.output_dir = Path(output_dir)

    def convert_to_parquet(
        self,
        csv_path: str,
        table_name: str,
        delimiter: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> Dict[str, any]:
        """
        Convert CSV file to Parquet.

        Args:
            csv_path: Path to CSV file
            table_name: Name for the output table
            delimiter: CSV delimiter (auto-detected if None)
            encoding: File encoding (default: utf-8)

        Returns:
            Dictionary with conversion results
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Converting CSV to Parquet: {csv_path}")

        try:
            # Read CSV with pandas
            df = pd.read_csv(
                csv_path,
                delimiter=delimiter,
                encoding=encoding,
                low_memory=False
            )

            # Clean column names for SQL compatibility
            df.columns = [self._sanitize_column_name(col) for col in df.columns]

            # Get metadata
            row_count = len(df)
            column_count = len(df.columns)

            logger.info(f"Loaded CSV: {row_count} rows, {column_count} columns")

            # Generate output path
            parquet_path = self.output_dir / f"{table_name}.parquet"

            # Convert to Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(parquet_path))

            logger.info(f"✓ Created Parquet file: {parquet_path}")

            return {
                "table_name": table_name,
                "parquet_path": str(parquet_path),
                "row_count": row_count,
                "column_count": column_count,
                "columns": df.columns.tolist()
            }

        except Exception as e:
            logger.error(f"Error converting CSV: {e}")
            raise

    def _sanitize_column_name(self, name: str) -> str:
        """
        Sanitize column name for SQL compatibility.

        Args:
            name: Original column name

        Returns:
            Sanitized column name
        """
        # Convert to string and strip whitespace
        name = str(name).strip()

        # Replace spaces and special characters with underscore
        name = ''.join(c if c.isalnum() else '_' for c in name)

        # Remove consecutive underscores
        while '__' in name:
            name = name.replace('__', '_')

        # Remove leading/trailing underscores
        name = name.strip('_')

        # Ensure doesn't start with number
        if name and name[0].isdigit():
            name = 'col_' + name

        # Ensure not empty
        if not name:
            name = 'unnamed_column'

        return name


# Example CLI entry point removed; this module is used by the FastAPI app.
