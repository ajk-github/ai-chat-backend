"""
Weekly Report Tool
LangGraph tool for generating weekly KPI reports from uploaded data.
"""
# src/agents/weekly_report_tool.py

import pandas as pd
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.utils.kpi_calculator import KPICalculator
from src.data_processing.duckdb_catalog import DuckDBCatalog

logger = logging.getLogger(__name__)


class WeeklyReportTool:
    """Tool for generating weekly KPI reports."""

    def __init__(self, catalog: DuckDBCatalog):
        """
        Initialize weekly report tool.

        Args:
            catalog: DuckDB catalog for querying data
        """
        self.catalog = catalog

    def generate_report(self, company_name: str = "Company", table_name: Optional[str] = None) -> str:
        """
        Generate a weekly KPI report from the data.

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)

        Returns:
            Formatted weekly report as text
        """
        try:
            logger.info(f"Generating weekly report for {company_name}...")

            # Get all available tables
            tables = self.catalog.list_tables()

            if not tables:
                return "❌ No data tables found. Please upload data first."

            # Use specified table or first available table
            target_table = table_name if table_name and table_name in tables else tables[0]
            logger.info(f"Using table: {target_table}")

            # Query all data from the table (no row limit for reports, longer timeout)
            query = f"SELECT * FROM {target_table}"
            result = self.catalog.execute_query(query, max_rows=999999999, timeout=60)

            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                return f"❌ Error querying data: {error_msg}"

            # Convert result to DataFrame (DuckDBCatalog returns data in "rows" key)
            df = pd.DataFrame(result.get("rows", []))

            if df.empty:
                return f"❌ No data available in the table '{target_table}'. Please ensure data is uploaded."

            logger.info(f"Loaded {len(df)} rows from {target_table}")

            # Initialize KPI calculator
            calculator = KPICalculator(df)

            # Generate report
            report = calculator.generate_weekly_report(company_name=company_name)

            # Format as text
            formatted_report = calculator.format_report_as_text(report)

            logger.info(f"Successfully generated weekly report for {company_name}")
            return formatted_report

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}", exc_info=True)
            return f"❌ Error generating weekly report: {str(e)}"

    def get_tool_description(self) -> str:
        """Get the tool description for LangGraph."""
        return """Generate a weekly KPI report with 10 key metrics.

        Use this tool when the user asks for:
        - Weekly report
        - KPI report
        - Performance metrics
        - Weekly summary
        - Commands like "/weeklyreport --{company_name}"

        Parameters:
        - company_name: Name of the company (default: "Company")
        - table_name: Optional specific table name (default: first table)

        Returns a formatted report with:
        1. Visits
        2. Charges
        3. Charges Submitted (%)
        4. Payments
        5. Gross Collection Rate (%)
        6. Net Collection Rate (%)
        7. Days in AR (DAR)
        8. A/R (31-60 Days)
        9. A/R (60+ Days)
        10. Denial vs Resolution (%)
        """


def create_weekly_report_tool_function(catalog: DuckDBCatalog):
    """
    Create the weekly report tool function for LangGraph.

    Args:
        catalog: DuckDB catalog instance

    Returns:
        Callable function for LangGraph tool
    """
    tool = WeeklyReportTool(catalog)

    def weekly_report(company_name: str = "Company", table_name: Optional[str] = None) -> str:
        """
        Generate a weekly KPI report.

        Args:
            company_name: Name of the company for the report
            table_name: Optional specific table name

        Returns:
            Formatted weekly report
        """
        return tool.generate_report(company_name=company_name, table_name=table_name)

    return weekly_report
