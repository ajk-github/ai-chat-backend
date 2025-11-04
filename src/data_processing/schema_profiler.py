"""
Schema Profiler
Generates schema profiles and metadata for Parquet files to help LLM understand data structure.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class SchemaProfiler:
    """Generates schema profiles for data tables."""

    def __init__(self, sample_rows: int = 10):
        """
        Initialize schema profiler.

        Args:
            sample_rows: Number of sample rows to include in profile
        """
        self.sample_rows = sample_rows

    def profile_parquet(self, parquet_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive schema profile for a Parquet file.

        Args:
            parquet_path: Path to Parquet file

        Returns:
            Dictionary containing schema profile
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logger.info(f"Profiling {parquet_path.name}...")

        # Read Parquet file
        df = pd.read_parquet(parquet_path)

        # Read Parquet metadata
        parquet_file = pq.ParquetFile(parquet_path)
        parquet_metadata = parquet_file.metadata

        profile = {
            "table_name": parquet_path.stem,
            "file_path": str(parquet_path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_mb": parquet_path.stat().st_size / (1024 * 1024),
            "created_at": datetime.now().isoformat(),
            "columns": [],
            "sample_data": [],
            "statistics": {}
        }

        # Profile each column
        for col in df.columns:
            col_profile = self._profile_column(df, col)
            profile["columns"].append(col_profile)

        # Add sample data
        sample_df = df.head(self.sample_rows)
        profile["sample_data"] = sample_df.to_dict(orient='records')

        # Overall statistics
        profile["statistics"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_cells": int(df.isnull().sum().sum()),
            "null_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        }

        logger.info(
            f"Profile complete: {profile['table_name']} "
            f"({profile['row_count']} rows, {profile['column_count']} cols)"
        )

        return profile

    def _profile_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Profile a single column.

        Args:
            df: DataFrame
            col: Column name

        Returns:
            Column profile dictionary
        """
        col_data = df[col]
        col_profile = {
            "name": col,
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_percentage": float(col_data.isnull().sum() / len(col_data) * 100),
            "unique_count": int(col_data.nunique()),
        }

        # Type-specific profiling
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update(self._profile_numeric(col_data))
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_profile.update(self._profile_datetime(col_data))
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            col_profile.update(self._profile_text(col_data))
        elif pd.api.types.is_bool_dtype(col_data):
            col_profile.update(self._profile_boolean(col_data))

        # Add sample values
        sample_values = col_data.dropna().head(5).tolist()
        col_profile["sample_values"] = [
            str(v) if not isinstance(v, (int, float, bool)) else v
            for v in sample_values
        ]

        # Semantic hints (detect common patterns)
        col_profile["semantic_hints"] = self._detect_semantic_type(col, col_data)

        return col_profile

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column."""
        return {
            "data_category": "numeric",
            "min": float(series.min()) if pd.notna(series.min()) else None,
            "max": float(series.max()) if pd.notna(series.max()) else None,
            "mean": float(series.mean()) if pd.notna(series.mean()) else None,
            "median": float(series.median()) if pd.notna(series.median()) else None,
            "std": float(series.std()) if pd.notna(series.std()) else None,
        }

    def _profile_datetime(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column."""
        return {
            "data_category": "datetime",
            "min": str(series.min()) if pd.notna(series.min()) else None,
            "max": str(series.max()) if pd.notna(series.max()) else None,
            "range_days": (series.max() - series.min()).days if pd.notna(series.min()) else None,
        }

    def _profile_text(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column."""
        # Convert to string and calculate lengths
        str_series = series.astype(str)
        lengths = str_series.str.len()

        return {
            "data_category": "text",
            "min_length": int(lengths.min()) if pd.notna(lengths.min()) else None,
            "max_length": int(lengths.max()) if pd.notna(lengths.max()) else None,
            "avg_length": float(lengths.mean()) if pd.notna(lengths.mean()) else None,
        }

    def _profile_boolean(self, series: pd.Series) -> Dict[str, Any]:
        """Profile boolean column."""
        value_counts = series.value_counts()
        return {
            "data_category": "boolean",
            "true_count": int(value_counts.get(True, 0)),
            "false_count": int(value_counts.get(False, 0)),
        }

    def _detect_semantic_type(self, col_name: str, series: pd.Series) -> List[str]:
        """
        Detect semantic meaning of column based on name and content.

        Args:
            col_name: Column name
            series: Column data

        Returns:
            List of semantic hints
        """
        hints = []
        col_lower = col_name.lower()

        # ID patterns
        if any(x in col_lower for x in ['id', '_id', 'key', 'code']):
            hints.append("identifier")

        # Name patterns
        if any(x in col_lower for x in ['name', 'title', 'label']):
            hints.append("name")

        # Date/time patterns
        if any(x in col_lower for x in ['date', 'time', 'timestamp', 'created', 'updated']):
            hints.append("temporal")

        # Financial patterns
        if any(x in col_lower for x in ['price', 'cost', 'revenue', 'amount', 'salary', 'fee']):
            hints.append("financial")

        # Quantity patterns
        if any(x in col_lower for x in ['quantity', 'count', 'qty', 'num', 'total']):
            hints.append("quantity")

        # Contact patterns
        if any(x in col_lower for x in ['email', 'phone', 'address', 'contact']):
            hints.append("contact_info")

        # Status patterns
        if any(x in col_lower for x in ['status', 'state', 'flag', 'active']):
            hints.append("status")

        # Category patterns
        if any(x in col_lower for x in ['category', 'type', 'class', 'group']):
            hints.append("category")

        # Percentage patterns
        if any(x in col_lower for x in ['percent', 'rate', 'ratio']):
            hints.append("percentage")

        return hints if hints else ["unknown"]

    def save_profile(self, profile: Dict[str, Any], output_path: str):
        """
        Save schema profile to JSON file.

        Args:
            profile: Schema profile dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        logger.info(f"Profile saved to {output_path}")

    def profile_directory(self, parquet_dir: str, output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Profile all Parquet files in a directory.

        Args:
            parquet_dir: Directory containing Parquet files
            output_dir: Directory to save profile JSON files (default: same as parquet_dir)

        Returns:
            Dictionary mapping table names to profiles
        """
        parquet_dir = Path(parquet_dir)
        output_dir = Path(output_dir) if output_dir else parquet_dir

        parquet_files = list(parquet_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {parquet_dir}")
            return {}

        logger.info(f"Found {len(parquet_files)} Parquet files to profile")

        profiles = {}

        for parquet_file in parquet_files:
            try:
                # Generate profile
                profile = self.profile_parquet(str(parquet_file))

                # Save profile
                profile_filename = f"{parquet_file.stem}_schema.json"
                profile_path = output_dir / profile_filename
                self.save_profile(profile, str(profile_path))

                profiles[profile["table_name"]] = profile

            except Exception as e:
                logger.error(f"Error profiling {parquet_file.name}: {e}")
                continue

        # Save combined catalog
        catalog_path = output_dir / "schema_catalog.json"
        catalog = {
            "generated_at": datetime.now().isoformat(),
            "table_count": len(profiles),
            "tables": {name: {
                "file_path": prof["file_path"],
                "row_count": prof["row_count"],
                "column_count": prof["column_count"],
                "columns": [c["name"] for c in prof["columns"]]
            } for name, prof in profiles.items()}
        }

        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Schema catalog saved to {catalog_path}")

        return profiles


def main():
    """Example usage and CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python schema_profiler.py <parquet_dir>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parquet_dir = sys.argv[1]
    profiler = SchemaProfiler(sample_rows=10)

    profiles = profiler.profile_directory(parquet_dir)

    print(f"\n=== Profiled {len(profiles)} tables ===")
    for table_name, profile in profiles.items():
        print(f"{table_name}: {profile['row_count']} rows, {profile['column_count']} columns")


if __name__ == "__main__":
    main()
