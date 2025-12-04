"""
KPI Metrics Tools for AI Agent

Individual tools that allow the AI agent to answer specific metric questions
without generating full reports. These tools wrap the pure calculation functions
from kpi_calculations.py.

Example user questions:
- "What is our gross collection rate?"
- "How many days in AR do we have?"
- "What's our total charges this month?"
- "Calculate the net collection rate"
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from src.utils.kpi_calculations import (
    calculate_total_visits,
    calculate_total_charges,
    calculate_total_payments,
    calculate_gross_collection_rate,
    calculate_net_collection_rate,
    calculate_days_in_ar,
    calculate_ar_aging,
    calculate_billed_ar,
    calculate_unbilled_ar,
    calculate_charges_submitted_percentage,
    get_date_range
)
from src.utils.kpi_calculator import KPICalculator
from src.data_processing.duckdb_catalog import DuckDBCatalog

logger = logging.getLogger(__name__)


class KPIMetricsTools:
    """Tools for calculating individual KPI metrics."""

    def __init__(self, catalog: DuckDBCatalog):
        """
        Initialize KPI metrics tools.

        Args:
            catalog: DuckDB catalog for querying data
        """
        self.catalog = catalog

    def _get_data(self, table_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get data from the database.

        Args:
            table_name: Optional specific table name

        Returns:
            DataFrame with data or None if error
        """
        try:
            tables = self.catalog.list_tables()
            if not tables:
                return None

            target_table = table_name if table_name and table_name in tables else tables[0]
            query = f"SELECT * FROM {target_table}"
            result = self.catalog.execute_query(query, max_rows=999999999, timeout=60)

            if not result.get("success"):
                return None

            df = pd.DataFrame(result.get("rows", []))
            return df if not df.empty else None

        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    def get_total_visits(self, table_name: Optional[str] = None, date_filter: Optional[str] = None) -> str:
        """
        Calculate total number of visits.

        Args:
            table_name: Optional table name
            date_filter: Optional date filter (e.g., "current_month", "last_week")

        Returns:
            Formatted response with total visits
        """
        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        # Apply date filter if specified
        if date_filter:
            df = self._apply_date_filter(df, date_filter)

        calculator = KPICalculator(df)
        total_visits = calculate_total_visits(df)

        period = f" ({date_filter})" if date_filter else ""
        return f"**Total Visits{period}:** {total_visits:,}"

    def get_total_charges(self, table_name: Optional[str] = None, date_filter: Optional[str] = None) -> str:
        """
        Calculate total charges (excluding negative values).

        Args:
            table_name: Optional table name
            date_filter: Optional date filter

        Returns:
            Formatted response with total charges
        """
        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        if date_filter:
            df = self._apply_date_filter(df, date_filter)

        calculator = KPICalculator(df)

        # Find charge column
        charge_col = self._find_column(df, ['charge', 'charges', 'total_charges'])
        if not charge_col:
            return "❌ No charge column found in data."

        total_charges = calculate_total_charges(df, charge_col)

        period = f" ({date_filter})" if date_filter else ""
        return f"**Total Charges{period}:** ${total_charges:,.2f}"

    def get_gross_collection_rate(self, table_name: Optional[str] = None, date_filter: Optional[str] = None) -> str:
        """
        Calculate Gross Collection Rate (GCR).

        Args:
            table_name: Optional table name
            date_filter: Optional date filter

        Returns:
            Formatted response with GCR
        """
        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        if date_filter:
            df = self._apply_date_filter(df, date_filter)

        # Find columns
        charge_col = self._find_column(df, ['charge', 'charges'])
        payment_col = self._find_column(df, ['total_payment', 'payment'])

        if not charge_col or not payment_col:
            return "❌ Required columns not found in data."

        total_charges = calculate_total_charges(df, charge_col)
        total_payments = calculate_total_payments(df, payment_col)
        gcr = calculate_gross_collection_rate(total_payments, total_charges)

        period = f" ({date_filter})" if date_filter else ""
        return f"""**Gross Collection Rate{period}:**
- Total Payments: ${total_payments:,.2f}
- Total Charges: ${total_charges:,.2f}
- **GCR: {gcr:.2f}%**"""

    def get_net_collection_rate(self, table_name: Optional[str] = None, date_filter: Optional[str] = None) -> str:
        """
        Calculate Net Collection Rate (NCR).

        Args:
            table_name: Optional table name
            date_filter: Optional date filter

        Returns:
            Formatted response with NCR
        """
        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        if date_filter:
            df = self._apply_date_filter(df, date_filter)

        # Find columns
        payment_col = self._find_column(df, ['total_payment', 'payment'])
        expected_col = self._find_column(df, ['expected', 'expected_payment'])

        if not payment_col or not expected_col:
            return "❌ Required columns not found in data."

        total_payments = calculate_total_payments(df, payment_col)
        total_expected = float(df[expected_col].sum())
        ncr = calculate_net_collection_rate(total_payments, total_expected)

        period = f" ({date_filter})" if date_filter else ""
        return f"""**Net Collection Rate{period}:**
- Total Payments: ${total_payments:,.2f}
- Total Expected: ${total_expected:,.2f}
- **NCR: {ncr:.2f}%**"""

    def get_days_in_ar(self, table_name: Optional[str] = None) -> str:
        """
        Calculate Days in AR (DAR) using Weekly KPI logic.

        Uses 365 days for full year data, or Year-to-date for current year only.

        Args:
            table_name: Optional table name

        Returns:
            Formatted response with DAR
        """
        from src.utils.kpi_calculations import calculate_days_in_ar_weekly, _get_period_days_weekly

        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        # Find columns
        balance_col = self._find_column(df, ['balance', 'ar_balance'])
        charge_col = self._find_column(df, ['charge', 'charges'])
        date_col = self._find_column(df, ['visit_date', 'date', 'dos'])

        if not balance_col or not charge_col or not date_col:
            return "❌ Required columns not found in data."

        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].notna()]

        ending_ar = float(df[balance_col].sum())
        total_charges = calculate_total_charges(df, charge_col)

        # Use new weekly DAR calculation
        dar = calculate_days_in_ar_weekly(ending_ar, total_charges, df, date_col)
        period_days = _get_period_days_weekly(df, date_col)

        return f"""**Days in AR:**
- Ending AR: ${ending_ar:,.2f}
- Total Charges: ${total_charges:,.2f}
- Period Days: {period_days} (365 for full year, or YTD for current year)
- **DAR: {dar} Days**"""

    def get_ar_summary(self, table_name: Optional[str] = None) -> str:
        """
        Get comprehensive AR summary including aging, billed/unbilled.

        Args:
            table_name: Optional table name

        Returns:
            Formatted response with AR summary
        """
        df = self._get_data(table_name)
        if df is None:
            return "❌ No data available."

        # Find columns
        balance_col = self._find_column(df, ['balance', 'ar_balance'])
        date_col = self._find_column(df, ['visit_date', 'date', 'dos'])
        status_col = self._find_column(df, ['visit_status', 'status'])

        if not balance_col:
            return "❌ Balance column not found in data."

        total_ar = float(df[balance_col].sum())

        # Calculate AR aging if date column exists
        ar_aging_text = ""
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            aging = calculate_ar_aging(
                df, date_col, balance_col,
                [(31, 60), (60, 90)]
            )
            ar_aging_text = f"""
**AR Aging:**
- 31-60 Days: ${aging.get('31-60', 0):,.2f}
- 60-90 Days: ${aging.get('60-90', 0):,.2f}"""

        # Calculate billed/unbilled if status column exists
        billed_unbilled_text = ""
        if status_col:
            billed_statuses = ['claim created', 'approved', 'reviewed', 'rejected']
            unbilled_statuses = ['on hold', 'issues pending', 'credentialing hold', 'pending auth']

            billed_ar = calculate_billed_ar(df, balance_col, status_col, billed_statuses)
            unbilled_ar = calculate_unbilled_ar(df, balance_col, status_col, unbilled_statuses)

            billed_pct = (billed_ar / total_ar * 100) if total_ar > 0 else 0
            unbilled_pct = (unbilled_ar / total_ar * 100) if total_ar > 0 else 0

            billed_unbilled_text = f"""
**Billed vs Unbilled AR:**
- Billed AR: ${billed_ar:,.2f} ({billed_pct:.2f}%)
- Unbilled AR: ${unbilled_ar:,.2f} ({unbilled_pct:.2f}%)"""

        return f"""**AR Summary:**
- Total AR: ${total_ar:,.2f}{ar_aging_text}{billed_unbilled_text}"""

    def _apply_date_filter(self, df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """
        Apply date filter to DataFrame.

        Args:
            df: Input DataFrame
            filter_type: Type of filter ("current_month", "last_week", etc.)

        Returns:
            Filtered DataFrame
        """
        date_col = self._find_column(df, ['visit_date', 'date', 'dos', 'transaction_date'])
        if not date_col:
            return df

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].notna()]

        if filter_type == "current_month":
            latest_date = df[date_col].max()
            first_of_month = latest_date.replace(day=1)
            return df[(df[date_col] >= first_of_month) & (df[date_col] <= latest_date)]

        # Add more filters as needed
        return df

    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """
        Find a column from list of candidates.

        Args:
            df: DataFrame to search
            candidates: List of possible column names

        Returns:
            Column name if found, None otherwise
        """
        # Normalize DataFrame columns
        df_cols_lower = [col.lower().replace(' ', '_') for col in df.columns]
        col_map = dict(zip(df_cols_lower, df.columns))

        for candidate in candidates:
            candidate_normalized = candidate.lower().replace(' ', '_')
            if candidate_normalized in df_cols_lower:
                return col_map[candidate_normalized]

        return None


def create_kpi_metrics_tool_functions(catalog: DuckDBCatalog) -> Dict[str, callable]:
    """
    Create individual KPI metric tool functions for LangGraph.

    Args:
        catalog: DuckDB catalog instance

    Returns:
        Dictionary of tool_name: tool_function
    """
    tools = KPIMetricsTools(catalog)

    return {
        "get_total_visits": tools.get_total_visits,
        "get_total_charges": tools.get_total_charges,
        "get_gross_collection_rate": tools.get_gross_collection_rate,
        "get_net_collection_rate": tools.get_net_collection_rate,
        "get_days_in_ar": tools.get_days_in_ar,
        "get_ar_summary": tools.get_ar_summary,
    }
