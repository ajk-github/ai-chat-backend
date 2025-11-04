"""
Excel to Parquet Converter
Converts .xlsb, .xlsx, and .xls files to optimized Parquet format with schema profiling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyxlsb import open_workbook

logger = logging.getLogger(__name__)


class ExcelConverter:
    """Handles conversion of Excel files (.xlsb, .xlsx, .xls) to Parquet format."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        compression: str = "snappy",
        row_group_size: int = 100000,
    ):
        """
        Initialize Excel converter.

        Args:
            output_dir: Directory to save Parquet files
            compression: Compression codec (snappy, gzip, zstd, lz4)
            row_group_size: Number of rows per row group
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.row_group_size = row_group_size

    def read_xlsb(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Read .xlsb file into pandas DataFrame.

        Args:
            file_path: Path to .xlsb file
            sheet_name: Specific sheet to read (None for all sheets)

        Returns:
            DataFrame containing sheet data
        """
        try:
            with open_workbook(file_path) as wb:
                if sheet_name:
                    sheets = [sheet_name]
                else:
                    sheets = wb.sheets

                # For single sheet, return DataFrame directly
                if len(sheets) == 1:
                    with wb.get_sheet(sheets[0]) as sheet:
                        data = []
                        for row in sheet.rows():
                            data.append([cell.v for cell in row])

                        if not data:
                            return pd.DataFrame()

                        # First row as header
                        df = pd.DataFrame(data[1:], columns=data[0])
                        return df

            logger.info(f"Successfully read {file_path}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error reading .xlsb file {file_path}: {e}")
            raise

    def read_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Read Excel file (.xlsx or .xlsb) into pandas DataFrame.

        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet to read

        Returns:
            DataFrame containing sheet data
        """
        file_path = Path(file_path)

        if file_path.suffix == ".xlsb":
            return self.read_xlsb(str(file_path), sheet_name)
        else:
            # Use pandas for .xlsx
            return pd.read_excel(file_path, sheet_name=sheet_name)

    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Get list of sheet names from Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            List of sheet names
        """
        file_path = Path(file_path)

        if file_path.suffix == ".xlsb":
            with open_workbook(str(file_path)) as wb:
                return wb.sheets
        else:
            # Use pandas for .xlsx
            with pd.ExcelFile(file_path) as xls:
                return xls.sheet_names

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame for better storage and querying.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows
        df = df.dropna(how='all')

        # Clean column names (remove special chars, spaces)
        cleaned_columns = []
        for col in df.columns:
            # Convert to string and clean
            col_str = str(col).strip()

            # Replace None or empty with 'column'
            if col_str in ['None', 'none', '', 'nan']:
                col_str = 'column'

            col_str = (col_str
                .replace(' ', '_')
                .replace('-', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('/', '_')
                .replace('\\', '_')
                .replace('.', '_')
                .lower())

            cleaned_columns.append(col_str)

        # Handle duplicate column names
        seen = {}
        final_columns = []

        for col in cleaned_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)

        df.columns = final_columns

        # Infer and optimize dtypes
        df = self._optimize_dtypes(df)

        return df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for better storage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized dtypes
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        for col in df.columns:
            try:
                col_type = df[col].dtype

                # Skip if already optimal
                if col_type in ['int8', 'int16', 'int32', 'float32', 'category']:
                    continue

                # Optimize integers
                if col_type == 'int64':
                    col_min = df[col].min()
                    col_max = df[col].max()

                    if pd.notna(col_min) and pd.notna(col_max):
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype('int16')
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            df[col] = df[col].astype('int32')

                # Optimize floats
                elif col_type == 'float64':
                    df[col] = df[col].astype('float32')

                # Convert low-cardinality strings to category
                elif col_type == 'object':
                    num_unique = df[col].nunique()
                    num_total = len(df[col])

                    if num_total > 0 and num_unique / num_total < 0.5 and num_unique < 1000:
                        df[col] = df[col].astype('category')

            except Exception as e:
                # Skip optimization for problematic columns
                logger.warning(f"Could not optimize column {col}: {e}")
                continue

        return df

    def convert_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: str,
        clean: bool = True
    ) -> str:
        """
        Convert DataFrame to Parquet file.

        Args:
            df: DataFrame to convert
            output_path: Output file path
            clean: Whether to clean DataFrame before saving

        Returns:
            Path to saved Parquet file
        """
        if clean:
            df = self.clean_dataframe(df)

        # Convert problematic columns to string
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                # Convert all values to string to avoid mixed types
                df[col] = df[col].astype(str)

        # Convert to PyArrow Table for more control
        try:
            table = pa.Table.from_pandas(df)
        except Exception as e:
            logger.error(f"Error converting to PyArrow table: {e}")
            # Last resort: convert everything to string
            df = df.astype(str)
            table = pa.Table.from_pandas(df)

        # Write Parquet file
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )

        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(
            f"Saved Parquet file: {output_path} "
            f"({len(df)} rows, {file_size:.2f} MB)"
        )

        return output_path

    def convert_excel_file(
        self,
        excel_path: str,
        output_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert entire Excel file to Parquet (one file per sheet).

        Args:
            excel_path: Path to Excel file
            output_prefix: Prefix for output files (default: excel filename)

        Returns:
            Dictionary with 'sheets' key containing list of sheet results
        """
        excel_path = Path(excel_path)

        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        if output_prefix is None:
            output_prefix = excel_path.stem

        logger.info(f"Converting {excel_path.name}...")

        # Get all sheet names
        sheet_names = self.get_sheet_names(str(excel_path))
        logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")

        sheets = []

        for sheet_name in sheet_names:
            try:
                # Read sheet
                logger.info(f"Processing sheet: {sheet_name}")
                df = self.read_excel(str(excel_path), sheet_name)

                if df.empty:
                    logger.warning(f"Sheet {sheet_name} is empty, skipping")
                    continue

                # Create output path
                safe_sheet_name = (
                    sheet_name.replace(' ', '_')
                    .replace('/', '_')
                    .replace('\\', '_')
                    .lower()
                )
                output_filename = f"{output_prefix}_{safe_sheet_name}.parquet"
                output_path = self.output_dir / output_filename

                # Convert to Parquet
                self.convert_to_parquet(df, str(output_path))

                # Add sheet result
                sheets.append({
                    "table_name": safe_sheet_name,
                    "parquet_path": str(output_path),
                    "row_count": len(df),
                    "column_count": len(df.columns)
                })

            except Exception as e:
                logger.error(f"Error converting sheet {sheet_name}: {e}")
                continue

        logger.info(f"Conversion complete. Processed {len(sheets)} sheets.")
        return {"sheets": sheets}


# Example CLI entry point removed; this module is used by the FastAPI app.
