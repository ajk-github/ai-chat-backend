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
        Generate weekly, quarterly, and yearly KPI reports from the data.

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)

        Returns:
            Formatted weekly, quarterly, and yearly reports as text (combined)
        """
        try:
            logger.info(f"Generating weekly and quarterly reports for {company_name}...")

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

            # Generate weekly report
            weekly_report = calculator.generate_weekly_report(company_name=company_name)
            formatted_weekly = calculator.format_report_as_text(weekly_report)

            # Generate quarterly report
            quarterly_report = calculator.generate_quarterly_report(company_name=company_name)
            formatted_quarterly = calculator.format_quarterly_report_as_text(quarterly_report)

            # Generate yearly report
            yearly_report = calculator.generate_yearly_report(company_name=company_name)
            formatted_yearly = calculator.format_yearly_report_as_text(yearly_report)

            # Generate weekly breakdown (for current month)
            weekly_breakdown = calculator.generate_weekly_breakdown_report(company_name=company_name)
            formatted_weekly_breakdown = calculator.format_weekly_breakdown_as_text(weekly_breakdown)

            # Combine all four reports
            combined_report = f"""
{formatted_weekly}


═════════════════════════════════════════════════════════════════


{formatted_quarterly}


═════════════════════════════════════════════════════════════════


{formatted_yearly}


═════════════════════════════════════════════════════════════════


{formatted_weekly_breakdown}
"""

            logger.info(f"Successfully generated all reports for {company_name}")
            return combined_report.strip()

        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
            return f"❌ Error generating reports: {str(e)}"

    def get_tool_description(self) -> str:
        """Get the tool description for LangGraph."""
        return """Generate weekly, quarterly, and yearly KPI reports with comprehensive metrics.

        Use this tool when the user asks for:
        - Weekly report
        - Quarterly report
        - Yearly report
        - Year over Year report
        - KPI report
        - Performance metrics
        - Weekly summary
        - Quarterly summary
        - Commands like "/weeklyreport --{company_name}"

        Parameters:
        - company_name: Name of the company (default: "Company")
        - table_name: Optional specific table name (default: first table)

        Returns FOUR formatted reports:

        SLIDE 1 - WEEKLY REPORT:
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

        SLIDE 2 - QUARTERLY REPORT (by Year and Quarter):
        1. Visits
        2. Charges
        3. Charges Submitted (%)
        4. Payments
        5. Gross Collection Rate (%)
        6. Net Collection Rate (%)
        7. Days in AR (DAR)
        8. Billed AR
        9. Billed AR %
        10. Unbilled AR
        11. Unbilled AR %
        12. Denial vs Resolution (%)

        SLIDE 3 - YEAR OVER YEAR REPORT:
        1. Visits
        2. Charges
        3. Charges Submitted (%)
        4. Payments
        5. Gross Collection Rate (%)
        6. Net Collection Rate (%)
        7. Days in AR (DAR)
        8. Billed AR
        9. Unbilled AR
        10. Denial vs Resolution (%)

        SLIDE 4 - WEEKLY BREAKDOWN (Current Month):
        1. Weekly Collections by Week
        2. Weekly Visits by Week
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
