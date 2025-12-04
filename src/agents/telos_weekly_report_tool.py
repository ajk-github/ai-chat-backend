"""
Telos Weekly Report Tool
LangGraph tool for generating Telos-specific weekly KPI reports (11 slides).

This is a company-specific report structure. For other companies, create similar
files (e.g., company_x_weekly_report_tool.py) with their specific slide structure.
"""
# src/agents/telos_weekly_report_tool.py

import pandas as pd
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.utils.kpi_calculator import KPICalculator
from src.data_processing.duckdb_catalog import DuckDBCatalog

logger = logging.getLogger(__name__)


class TelosWeeklyReportTool:
    """
    Tool for generating Telos-specific weekly KPI reports.

    Generates 11 slides:
    - Slide 2: Billing Progress (Current Month)
    - Slide 3: KPI Metrics Weekly
    - Slide 4: KPI Metrics Quarterly
    - Slide 5: KPI Metrics Year Over Year
    - Slide 6: Weekly Breakdown
    - Slide 7: Client Summary Weekly
    - Slide 8: Client Summary Month to Date
    - Slide 9: Unbilled Status
    - Slide 10: N/A (skipped)
    - Slide 11: Top 15 Denial Categories
    """

    COMPANY_NAME = "Telos"  # Default company name for Telos reports

    def __init__(self, catalog: DuckDBCatalog):
        """
        Initialize Telos weekly report tool.

        Args:
            catalog: DuckDB catalog for querying data
        """
        self.catalog = catalog

    def generate_report(self, company_name: str = None, table_name: Optional[str] = None) -> str:
        """
        Generate complete Telos weekly report with all 11 slides.

        Args:
            company_name: Name of the company for the report header (default: "Telos")
            table_name: Specific table to query (if None, uses first available table)

        Returns:
            Formatted report with all slides as text (combined)
        """
        # Use Telos as default company name
        company_name = company_name or self.COMPANY_NAME

        try:
            logger.info(f"Generating Telos weekly report for {company_name}...")

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

            # Generate all slides
            slides = self._generate_all_slides(calculator, company_name)

            # Combine all slides
            combined_report = self._combine_slides(slides)

            logger.info(f"Successfully generated Telos report for {company_name}")
            return combined_report.strip()

        except Exception as e:
            logger.error(f"Error generating Telos report: {e}", exc_info=True)
            return f"❌ Error generating Telos report: {str(e)}"

    def _generate_all_slides(self, calculator: KPICalculator, company_name: str) -> Dict[int, str]:
        """
        Generate all individual slides.

        Args:
            calculator: KPICalculator instance with data
            company_name: Company name for headers

        Returns:
            Dictionary of slide_number: formatted_content
        """
        slides = {}

        # Slide 2: Billing Progress
        try:
            billing_progress = calculator.generate_billing_progress_report(company_name=company_name)
            slides[2] = calculator.format_billing_progress_as_text(billing_progress)
        except Exception as e:
            slides[2] = f"❌ Error generating Slide 2 (Billing Progress): {str(e)}"

        # Slide 3: KPI Metrics Weekly
        try:
            weekly_report = calculator.generate_weekly_report(company_name=company_name)
            slides[3] = calculator.format_report_as_text(weekly_report)
        except Exception as e:
            slides[3] = f"❌ Error generating Slide 3 (KPI Metrics Weekly): {str(e)}"

        # Slide 4: KPI Metrics Quarterly
        try:
            quarterly_report = calculator.generate_quarterly_report(company_name=company_name)
            slides[4] = calculator.format_quarterly_report_as_text(quarterly_report)
        except Exception as e:
            slides[4] = f"❌ Error generating Slide 4 (KPI Metrics Quarterly): {str(e)}"

        # Slide 5: KPI Metrics Year Over Year
        try:
            yearly_report = calculator.generate_yearly_report(company_name=company_name)
            slides[5] = calculator.format_yearly_report_as_text(yearly_report)
        except Exception as e:
            slides[5] = f"❌ Error generating Slide 5 (KPI Metrics Year Over Year): {str(e)}"

        # Slide 6: Weekly Breakdown
        try:
            weekly_breakdown = calculator.generate_weekly_breakdown_report(company_name=company_name)
            slides[6] = calculator.format_weekly_breakdown_as_text(weekly_breakdown)
        except Exception as e:
            slides[6] = f"❌ Error generating Slide 6 (Weekly Breakdown): {str(e)}"

        # Slide 7: Client Summary Weekly
        try:
            weekly_comparison = calculator.generate_weekly_comparison_report(company_name=company_name)
            slides[7] = calculator.format_weekly_comparison_as_text(weekly_comparison)
        except Exception as e:
            slides[7] = f"❌ Error generating Slide 7 (Client Summary Weekly): {str(e)}"

        # Slide 8: Client Summary Month to Date
        try:
            monthly_comparison = calculator.generate_monthly_comparison_report(company_name=company_name)
            slides[8] = calculator.format_monthly_comparison_as_text(monthly_comparison)
        except Exception as e:
            slides[8] = f"❌ Error generating Slide 8 (Client Summary Month to Date): {str(e)}"

        # Slide 9: Unbilled Status
        try:
            unbilled_status = calculator.generate_unbilled_status_report(company_name=company_name)
            slides[9] = calculator.format_unbilled_status_as_text(unbilled_status)
        except Exception as e:
            slides[9] = f"❌ Error generating Slide 9 (Unbilled Status): {str(e)}"

        # Slide 10: N/A (skipped)

        # Slide 11: Top 15 Denial Categories
        try:
            denial_categories = calculator.generate_denial_categories_report(company_name=company_name)
            slides[11] = calculator.format_denial_categories_as_text(denial_categories)
        except Exception as e:
            slides[11] = f"❌ Error generating Slide 11 (Top 15 Denial Categories): {str(e)}"

        return slides

    def _combine_slides(self, slides: Dict[int, str]) -> str:
        """
        Combine all slides into a single report with separators.

        Args:
            slides: Dictionary of slide_number: formatted_content

        Returns:
            Combined report string
        """
        separator = "\n\n═════════════════════════════════════════════════════════════════\n\n"

        # Combine slides in order (2, 3, 4, 5, 6, 7, 8, 9, 11)
        slide_order = [2, 3, 4, 5, 6, 7, 8, 9, 11]

        combined_parts = []
        for slide_num in slide_order:
            if slide_num in slides:
                combined_parts.append(slides[slide_num])

        return separator.join(combined_parts)

    def get_tool_description(self) -> str:
        """Get the tool description for LangGraph."""
        return """Generate Telos weekly KPI report with 11 comprehensive slides.

        Use this tool when the user asks for:
        - Weekly report
        - Quarterly report
        - Yearly report
        - Year over Year report
        - KPI report
        - Performance metrics
        - Weekly summary
        - Quarterly summary
        - Commands like "/weeklyreport" or "/telosreport"

        Parameters:
        - company_name: Name of the company (default: "Telos")
        - table_name: Optional specific table name (default: first table)

        Returns NINE slides (Slide 2-11, skipping Slide 10):

        SLIDE 2 - BILLING PROGRESS (Current Month):
        - Charges
        - Encounters
        - Insurance Payments Posted
        - Patient Payments Posted
        - A/R Claims processed
        - Authorization denials / Pending claims

        SLIDE 3 - KPI METRICS WEEKLY:
        - Visits
        - Charges
        - Charges Submitted (%)
        - Payments
        - Gross Collection Rate (%)
        - Net Collection Rate (%)
        - Days in AR (DAR)
        - A/R (31-60 Days)
        - A/R (60+ Days)
        - Denial vs Resolution (%)

        SLIDE 4 - KPI METRICS QUARTERLY:
        - Quarterly breakdown of all KPIs
        - Billed/Unbilled AR metrics

        SLIDE 5 - KPI METRICS YEAR OVER YEAR:
        - Yearly comparison of all KPIs

        SLIDE 6 - WEEKLY BREAKDOWN (Current Month):
        - Forecasted vs Actual Visits (Cumulative)
        - Forecasted vs Actual Collections (Cumulative)

        SLIDE 7 - CLIENT SUMMARY WEEKLY:
        - Collections, Charges, Visits for latest week

        SLIDE 8 - CLIENT SUMMARY MONTH TO DATE:
        - Collections, Charges, Visits for current month + last 3 months (same day range)

        SLIDE 9 - UNBILLED STATUS:
        - Visit counts by status for last completed weeks

        SLIDE 11 - TOP 15 DENIAL CATEGORIES:
        - Top 15 denial codes with visit count and expected payment by state
        - All-time data
        """


def create_telos_weekly_report_tool_function(catalog: DuckDBCatalog):
    """
    Create the Telos weekly report tool function for LangGraph.

    Args:
        catalog: DuckDB catalog instance

    Returns:
        Callable function for LangGraph tool
    """
    tool = TelosWeeklyReportTool(catalog)

    def telos_weekly_report(company_name: str = None, table_name: Optional[str] = None) -> str:
        """
        Generate a Telos weekly KPI report.

        Args:
            company_name: Name of the company for the report (default: "Telos")
            table_name: Optional specific table name

        Returns:
            Formatted Telos weekly report with 11 slides
        """
        return tool.generate_report(company_name=company_name, table_name=table_name)

    return telos_weekly_report
