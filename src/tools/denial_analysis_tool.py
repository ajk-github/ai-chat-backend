"""
Denial Analysis Tool for AI Agent

This tool allows the AI agent to generate denial analysis reports when users
ask about denials, such as:
- "/denial"
- "Show me denial analysis"
- "What are my denied claims?"
- "Denial report"

The tool generates 3 slides of denial metrics (Slide 1 implemented, Slides 2-3 pending).

IMPORTANT: Uses DuckDB SQL queries directly for accurate counting.
Empty Denial Code = NOT DENIED
Non-empty Denial Code = DENIED
"""

from typing import Optional, Dict, Any, List, Tuple
import logging

from src.data_processing.duckdb_catalog import DuckDBCatalog

logger = logging.getLogger(__name__)


class DenialAnalysisTool:
    """Tool for generating denial analysis reports using DuckDB SQL queries."""

    def __init__(self, catalog: DuckDBCatalog):
        """
        Initialize denial analysis tool.

        Args:
            catalog: DuckDB catalog for querying data
        """
        self.catalog = catalog

    def _find_denial_code_column(self, table_name: str) -> Optional[str]:
        """
        Find the denial code column in the table.

        Handles columns with spaces like "Denial Code" and various naming conventions.

        Args:
            table_name: Name of the table to search

        Returns:
            Column name if found, None otherwise
        """
        try:
            # Get table schema
            schema = self.catalog.get_table_schema(table_name)

            if not schema:
                logger.warning(f"Could not get schema for table {table_name}")
                return None

            # Get all column names (case-insensitive search)
            columns = [col['name'] for col in schema]

            # Search for denial code column (case-insensitive, handles spaces)
            # Common variations: "Denial Code", "denial_code", "denialcode", "denial", "deny_code"
            denial_patterns = [
                'denial code',
                'denial_code',
                'denialcode',
                'denial',
                'deny_code',
                'rejection_code',
                'reject_code'
            ]

            for col in columns:
                col_lower = col.lower().strip()
                if col_lower in denial_patterns:
                    logger.info(f"Found denial code column: '{col}' (exact case preserved)")
                    return col  # Return with original casing

            logger.warning(f"Denial code column not found in table {table_name}")
            logger.info(f"Available columns: {columns}")
            return None

        except Exception as e:
            logger.error(f"Error finding denial code column: {e}")
            return None

    def _find_charges_column(self, table_name: str) -> Optional[str]:
        """
        Find the charges column in the table.

        Args:
            table_name: Name of the table to search

        Returns:
            Column name if found, None otherwise
        """
        try:
            schema = self.catalog.get_table_schema(table_name)

            if not schema:
                return None

            columns = [col['name'] for col in schema]

            # Common charge column names
            charge_patterns = [
                'charges',
                'charge',
                'billed_amount',
                'billed',
                'total_charges',
                'amount',
                'billed amount'
            ]

            for col in columns:
                col_lower = col.lower().strip()
                if col_lower in charge_patterns:
                    logger.info(f"Found charges column: '{col}'")
                    return col

            logger.warning(f"Charges column not found in table {table_name}")
            return None

        except Exception as e:
            logger.error(f"Error finding charges column: {e}")
            return None

    def _find_visit_date_column(self, table_name: str) -> Optional[str]:
        """
        Find the visit_date column in the table.

        Args:
            table_name: Name of the table to search

        Returns:
            Column name if found, None otherwise
        """
        try:
            schema = self.catalog.get_table_schema(table_name)

            if not schema:
                return None

            columns = [col['name'] for col in schema]

            # Common visit date column names
            date_patterns = [
                'visit_date',
                'visit date',
                'visitdate',
                'date',
                'dos',
                'date_of_service',
                'date of service',
                'service_date',
                'service date'
            ]

            for col in columns:
                col_lower = col.lower().strip()
                if col_lower in date_patterns:
                    logger.info(f"Found visit date column: '{col}'")
                    return col

            logger.warning(f"Visit date column not found in table {table_name}")
            return None

        except Exception as e:
            logger.error(f"Error finding visit date column: {e}")
            return None

    def _find_transaction_date_column(self, table_name: str) -> Optional[str]:
        """
        Find the transaction_date column in the table.

        Args:
            table_name: Name of the table to search

        Returns:
            Column name if found, None otherwise
        """
        try:
            schema = self.catalog.get_table_schema(table_name)

            if not schema:
                return None

            columns = [col['name'] for col in schema]

            # Common transaction date column names
            date_patterns = [
                'transaction_date',
                'transaction date',
                'transactiondate',
                'payment_date',
                'payment date',
                'paymentdate',
                'date_created',
                'created_date',
                'post_date',
                'posting_date'
            ]

            for col in columns:
                col_lower = col.lower().strip()
                if col_lower in date_patterns:
                    logger.info(f"Found transaction date column: '{col}'")
                    return col

            logger.warning(f"Transaction date column not found in table {table_name}")
            return None

        except Exception as e:
            logger.error(f"Error finding transaction date column: {e}")
            return None

    def _get_value_distribution(self, table_name: str, denial_col: str) -> Tuple[List[Tuple[str, int]], int, int, int]:
        """
        Get distribution of values in the denial code column for diagnostics.

        Args:
            table_name: Name of the table
            denial_col: Name of the denial code column

        Returns:
            Tuple of (list of (value, count) tuples, null_count, empty_count, nan_count)
        """
        try:
            # Query 1: Count NULL values
            null_query = f'''
                SELECT COUNT(*) as null_count
                FROM "{table_name}"
                WHERE "{denial_col}" IS NULL
            '''
            null_result = self.catalog.execute_query(null_query, max_rows=1, timeout=30)
            null_count = null_result.get("rows", [{}])[0].get("null_count", 0) if null_result.get("success") else 0

            # Query 2: Count empty string values
            empty_query = f'''
                SELECT COUNT(*) as empty_count
                FROM "{table_name}"
                WHERE "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") = ''
            '''
            empty_result = self.catalog.execute_query(empty_query, max_rows=1, timeout=30)
            empty_count = empty_result.get("rows", [{}])[0].get("empty_count", 0) if empty_result.get("success") else 0

            # Query 3: Count 'nan' string values (pandas NaN as string)
            nan_query = f'''
                SELECT COUNT(*) as nan_count
                FROM "{table_name}"
                WHERE UPPER(TRIM("{denial_col}")) = 'NAN'
            '''
            nan_result = self.catalog.execute_query(nan_query, max_rows=1, timeout=30)
            nan_count = nan_result.get("rows", [{}])[0].get("nan_count", 0) if nan_result.get("success") else 0

            # Query 4: Get top 20 distinct non-empty values with counts (excluding 'nan')
            dist_query = f'''
                SELECT "{denial_col}" as value, COUNT(*) as count
                FROM "{table_name}"
                WHERE "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") != ''
                  AND UPPER(TRIM("{denial_col}")) != 'NAN'
                GROUP BY "{denial_col}"
                ORDER BY count DESC
                LIMIT 20
            '''
            dist_result = self.catalog.execute_query(dist_query, max_rows=20, timeout=30)

            distribution = []
            if dist_result.get("success"):
                for row in dist_result.get("rows", []):
                    value = str(row.get("value", ""))
                    count = row.get("count", 0)
                    distribution.append((value, count))

            logger.info(f"Value distribution - NULL: {null_count}, Empty: {empty_count}, NaN: {nan_count}, Real codes: {len(distribution)}")

            return distribution, null_count, empty_count, nan_count

        except Exception as e:
            logger.error(f"Error getting value distribution: {e}")
            return [], 0, 0, 0

    def _get_denial_metrics(self, table_name: str, denial_col: str, charges_col: Optional[str]) -> Dict[str, Any]:
        """
        Get denial metrics using DuckDB SQL queries.

        LOGIC (SIMPLE):
        - NOT DENIED: denial_col IS NULL OR TRIM(denial_col) = '' (empty)
        - DENIED: denial_col IS NOT NULL AND TRIM(denial_col) != '' (has a value)

        Args:
            table_name: Name of the table
            denial_col: Name of the denial code column
            charges_col: Name of the charges column (can be None)

        Returns:
            Dictionary with metrics
        """
        try:
            # Get value distribution for diagnostics
            distribution, null_count, empty_count, nan_count = self._get_value_distribution(table_name, denial_col)

            # Query 1: Total claims count
            total_query = f'SELECT COUNT(*) as total FROM "{table_name}"'
            total_result = self.catalog.execute_query(total_query, max_rows=1, timeout=30)

            if not total_result.get("success"):
                logger.error(f"Error getting total count: {total_result.get('error')}")
                return {"error": "Failed to get total count"}

            total_count = total_result.get("rows", [{}])[0].get("total", 0)
            logger.info(f"Total claims: {total_count}")

            # Query 2: DENIED claims count (NOT NULL AND NOT EMPTY AND NOT 'nan')
            # SIMPLE LOGIC: If the column has any value (not empty, not 'nan'), it's denied
            # 'nan' is the string representation of pandas/Python NaN values
            denied_query = f'''
                SELECT COUNT(*) as denied
                FROM "{table_name}"
                WHERE "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") != ''
                  AND UPPER(TRIM("{denial_col}")) != 'NAN'
            '''
            denied_result = self.catalog.execute_query(denied_query, max_rows=1, timeout=30)

            if not denied_result.get("success"):
                logger.error(f"Error getting denied count: {denied_result.get('error')}")
                return {"error": "Failed to get denied count"}

            denied_count = denied_result.get("rows", [{}])[0].get("denied", 0)
            logger.info(f"Denied claims (non-empty values): {denied_count}")

            # Query 3: Denied claims charges (if charges column exists)
            denied_charges = 0.0
            if charges_col:
                charges_query = f'''
                    SELECT SUM("{charges_col}") as total_charges
                    FROM "{table_name}"
                    WHERE "{denial_col}" IS NOT NULL
                      AND TRIM("{denial_col}") != ''
                      AND UPPER(TRIM("{denial_col}")) != 'NAN'
                '''
                charges_result = self.catalog.execute_query(charges_query, max_rows=1, timeout=30)

                if charges_result.get("success"):
                    denied_charges = charges_result.get("rows", [{}])[0].get("total_charges", 0) or 0
                    logger.info(f"Denied charges: ${denied_charges:,.2f}")

            # Calculate denial percentage
            denial_percentage = (denied_count / total_count * 100) if total_count > 0 else 0

            # Calculate NOT denied count (NULL + empty + 'nan' string)
            not_denied_count = null_count + empty_count + nan_count

            return {
                "total_count": total_count,
                "denied_count": denied_count,
                "not_denied_count": not_denied_count,
                "null_count": null_count,
                "empty_count": empty_count,
                "nan_count": nan_count,
                "denial_percentage": denial_percentage,
                "denied_charges": denied_charges,
                "distribution": distribution,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error getting denial metrics: {e}", exc_info=True)
            return {"error": str(e)}

    def generate_report(self, company_name: str = "Company", table_name: Optional[str] = None,
                       use_transaction_date: bool = False) -> str:
        """
        Generate complete denial analysis report (all slides) using DuckDB SQL queries.

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)
            use_transaction_date: If True, use transaction_date instead of visit_date

        Returns:
            Formatted denial analysis report with all slides as text
        """
        try:
            date_type = "transaction_date" if use_transaction_date else "visit_date"
            logger.info(f"Generating complete denial analysis report for {company_name} using {date_type}...")

            # Get all available tables
            tables = self.catalog.list_tables()

            if not tables:
                return "❌ No data tables found. Please upload data first."

            # Use specified table or first available table
            target_table = table_name if table_name and table_name in tables else tables[0]
            logger.info(f"Using table: {target_table}")

            # Find denial code column
            denial_col = self._find_denial_code_column(target_table)

            if not denial_col:
                # Get schema for debugging
                schema = self.catalog.get_table_schema(target_table)
                columns = [col['name'] for col in schema] if schema else []

                return f"""❌ Denial Code column not found in the data.

Available columns in table '{target_table}':
{', '.join(columns)}

Looking for columns like: 'Denial Code', 'denial_code', 'denialcode', etc.
Please ensure your data has a denial code column."""

            # Find date column based on flag
            if use_transaction_date:
                date_col = self._find_transaction_date_column(target_table)
                date_label = "Transaction Date"
            else:
                date_col = self._find_visit_date_column(target_table)
                date_label = "Visit Date"

            if not date_col:
                return f"❌ {date_label} column not found - cannot generate time-based denial reports"

            # Find charges column (optional)
            charges_col = self._find_charges_column(target_table)

            if not charges_col:
                logger.warning("Charges column not found - will show $0.00 for denied charges")

            # ===== SLIDE 1: All Time =====
            # Get denial metrics using SQL queries
            metrics = self._get_denial_metrics(target_table, denial_col, charges_col)

            if "error" in metrics:
                return f"❌ Error calculating denial metrics: {metrics.get('error')}"

            # Format Slide 1 as a markdown table
            slide1_output = f"\n{'='*60}\n"
            slide1_output += f"SLIDE 1: {company_name} - Denied Claims (All time)\n"
            slide1_output += f"{'='*60}\n\n"

            if not charges_col:
                slide1_output += "⚠️  WARNING: Charges column not found\n\n"

            # Create markdown table
            slide1_output += "| Metric | Value |\n"
            slide1_output += "|--------|-------|\n"
            slide1_output += f"| Count of Denied Claims | {metrics['denied_count']:,} |\n"
            slide1_output += f"| Total Claims | {metrics['total_count']:,} |\n"
            slide1_output += f"| Denial Percentage | {metrics['denial_percentage']:.2f}% |\n"
            slide1_output += f"| Total Denied Charges | ${metrics['denied_charges']:,.2f} |\n"
            slide1_output += f"\n{'='*60}\n"

            # ===== SLIDE 2: Year over Year =====
            slide2_output = self.generate_slide_2_year_over_year(company_name, target_table, date_col, date_label)

            # ===== SLIDE 3: Current Month =====
            slide3_output = self.generate_slide_3_current_month(company_name, target_table, date_col, date_label)

            # ===== SLIDE 4: Last Week =====
            slide4_output = self.generate_slide_4_last_week(company_name, target_table, date_col, date_label)

            # Combine all slides
            combined_output = f"{slide1_output}\n\n{slide2_output}\n\n{slide3_output}\n\n{slide4_output}"

            logger.info(f"Successfully generated complete denial analysis report for {company_name} using {date_type}")
            return combined_output

        except Exception as e:
            logger.error(f"Error generating denial analysis report: {e}", exc_info=True)
            return f"❌ Error generating denial analysis report: {str(e)}"

    def _get_available_years(self, table_name: str, visit_date_col: str) -> List[int]:
        """
        Get list of available years from the visit_date column.

        Args:
            table_name: Name of the table
            visit_date_col: Name of the visit date column

        Returns:
            List of years (sorted ascending)
        """
        try:
            query = f'''
                SELECT DISTINCT YEAR("{visit_date_col}") as year
                FROM "{table_name}"
                WHERE "{visit_date_col}" IS NOT NULL
                ORDER BY year ASC
            '''
            result = self.catalog.execute_query(query, max_rows=100, timeout=30)

            if result.get("success"):
                years = [row.get("year") for row in result.get("rows", []) if row.get("year")]
                logger.info(f"Found years: {years}")
                return years
            return []

        except Exception as e:
            logger.error(f"Error getting available years: {e}")
            return []

    def _get_denial_metrics_for_year(self, table_name: str, denial_col: str, charges_col: Optional[str],
                                      visit_date_col: str, year: int) -> Dict[str, Any]:
        """
        Get denial metrics for a specific year.

        Args:
            table_name: Name of the table
            denial_col: Name of the denial code column
            charges_col: Name of the charges column (can be None)
            visit_date_col: Name of the visit date column
            year: Year to filter by

        Returns:
            Dictionary with metrics for that year
        """
        try:
            # Query 1: Total claims for this year
            total_query = f'''
                SELECT COUNT(*) as total
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
            '''
            total_result = self.catalog.execute_query(total_query, max_rows=1, timeout=30)
            total_count = total_result.get("rows", [{}])[0].get("total", 0) if total_result.get("success") else 0

            # Query 2: Denied claims for this year
            denied_query = f'''
                SELECT COUNT(*) as denied
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
                  AND "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") != ''
                  AND UPPER(TRIM("{denial_col}")) != 'NAN'
            '''
            denied_result = self.catalog.execute_query(denied_query, max_rows=1, timeout=30)
            denied_count = denied_result.get("rows", [{}])[0].get("denied", 0) if denied_result.get("success") else 0

            # Query 3: Denied charges for this year (if charges column exists)
            denied_charges = 0.0
            if charges_col:
                charges_query = f'''
                    SELECT SUM("{charges_col}") as total_charges
                    FROM "{table_name}"
                    WHERE YEAR("{visit_date_col}") = {year}
                      AND "{denial_col}" IS NOT NULL
                      AND TRIM("{denial_col}") != ''
                      AND UPPER(TRIM("{denial_col}")) != 'NAN'
                '''
                charges_result = self.catalog.execute_query(charges_query, max_rows=1, timeout=30)
                denied_charges = charges_result.get("rows", [{}])[0].get("total_charges", 0) or 0 if charges_result.get("success") else 0

            # Calculate denial percentage
            denial_percentage = (denied_count / total_count * 100) if total_count > 0 else 0

            return {
                "year": year,
                "total_count": total_count,
                "denied_count": denied_count,
                "denial_percentage": denial_percentage,
                "denied_charges": denied_charges
            }

        except Exception as e:
            logger.error(f"Error getting denial metrics for year {year}: {e}")
            return {
                "year": year,
                "total_count": 0,
                "denied_count": 0,
                "denial_percentage": 0.0,
                "denied_charges": 0.0,
                "error": str(e)
            }

    def generate_slide_2_year_over_year(self, company_name: str = "Company", table_name: Optional[str] = None,
                                        date_col: Optional[str] = None, date_label: str = "Visit Date") -> str:
        """
        Generate Slide 2: Denied Claims (Year over Year)

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)

        Returns:
            Formatted year over year denial analysis report as text
        """
        try:
            logger.info(f"Generating year over year denial analysis for {company_name}...")

            # Get all available tables
            tables = self.catalog.list_tables()

            if not tables:
                return "❌ No data tables found. Please upload data first."

            # Use specified table or first available table
            target_table = table_name if table_name and table_name in tables else tables[0]

            # Find required columns
            denial_col = self._find_denial_code_column(target_table)
            if not denial_col:
                return "❌ Denial Code column not found - cannot generate year over year report"

            # Use provided date_col or find visit_date
            if not date_col:
                date_col = self._find_visit_date_column(target_table)
                if not date_col:
                    return "❌ Visit Date column not found - cannot generate year over year report"

            charges_col = self._find_charges_column(target_table)

            # Get available years
            years = self._get_available_years(target_table, date_col)

            if not years:
                return "❌ No valid years found in visit date column"

            # Get metrics for each year
            year_metrics = []
            for year in years:
                metrics = self._get_denial_metrics_for_year(target_table, denial_col, charges_col, date_col, year)
                year_metrics.append(metrics)

            # Format the report as a markdown table
            output = f"\n{'='*60}\n"
            output += f"SLIDE 2: {company_name} - Denied Claims (Year over Year)\n"
            output += f"Based on {date_label}\n"
            output += f"{'='*60}\n\n"

            if not charges_col:
                output += "⚠️  WARNING: Charges column not found\n\n"

            # Create markdown table with years as columns
            # Header row
            output += "| Metric |"
            for year in years:
                output += f" {year} |"
            output += "\n"

            # Separator row
            output += "|--------|"
            for _ in years:
                output += "--------|"
            output += "\n"

            # Count of Denied Claims row
            output += "| Count of Denied Claims |"
            for metrics in year_metrics:
                output += f" {metrics['denied_count']:,} |"
            output += "\n"

            # Total Claims row
            output += "| Total Claims |"
            for metrics in year_metrics:
                output += f" {metrics['total_count']:,} |"
            output += "\n"

            # Denial Percentage row
            output += "| Denial Percentage |"
            for metrics in year_metrics:
                output += f" {metrics['denial_percentage']:.2f}% |"
            output += "\n"

            # Total Denied Charges row
            output += "| Total Denied Charges |"
            for metrics in year_metrics:
                output += f" ${metrics['denied_charges']:,.2f} |"
            output += "\n"

            output += f"\n{'='*60}\n"

            logger.info(f"Successfully generated year over year denial analysis for {company_name}")
            return output

        except Exception as e:
            logger.error(f"Error generating year over year denial analysis: {e}", exc_info=True)
            return f"❌ Error generating year over year denial analysis: {str(e)}"

    def _get_current_month_info(self, table_name: str, visit_date_col: str) -> Optional[Dict[str, Any]]:
        """
        Get current month information (year and month) based on the latest date in the data.

        Args:
            table_name: Name of the table
            visit_date_col: Name of the visit date column

        Returns:
            Dictionary with year, month, and date range, or None if error
        """
        try:
            # Get the latest date in the data
            query = f'''
                SELECT MAX("{visit_date_col}") as max_date
                FROM "{table_name}"
                WHERE "{visit_date_col}" IS NOT NULL
            '''
            result = self.catalog.execute_query(query, max_rows=1, timeout=30)

            if result.get("success") and result.get("rows"):
                max_date = result.get("rows")[0].get("max_date")
                if max_date:
                    # Get year and month from max date
                    year_query = f'''
                        SELECT
                            YEAR("{visit_date_col}") as year,
                            MONTH("{visit_date_col}") as month
                        FROM "{table_name}"
                        WHERE "{visit_date_col}" = '{max_date}'
                        LIMIT 1
                    '''
                    year_result = self.catalog.execute_query(year_query, max_rows=1, timeout=30)

                    if year_result.get("success") and year_result.get("rows"):
                        year = year_result.get("rows")[0].get("year")
                        month = year_result.get("rows")[0].get("month")

                        logger.info(f"Current month determined as: {year}-{month:02d} (based on max date: {max_date})")

                        return {
                            "year": year,
                            "month": month,
                            "max_date": max_date
                        }

            return None

        except Exception as e:
            logger.error(f"Error getting current month info: {e}")
            return None

    def _get_denial_metrics_for_current_month(self, table_name: str, denial_col: str, charges_col: Optional[str],
                                               visit_date_col: str, year: int, month: int) -> Dict[str, Any]:
        """
        Get denial metrics for the current month (month-to-date).

        Args:
            table_name: Name of the table
            denial_col: Name of the denial code column
            charges_col: Name of the charges column (can be None)
            visit_date_col: Name of the visit date column
            year: Current year
            month: Current month

        Returns:
            Dictionary with metrics for current month
        """
        try:
            # Query 1: Total claims for this month
            total_query = f'''
                SELECT COUNT(*) as total
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
                  AND MONTH("{visit_date_col}") = {month}
            '''
            total_result = self.catalog.execute_query(total_query, max_rows=1, timeout=30)
            total_count = total_result.get("rows", [{}])[0].get("total", 0) if total_result.get("success") else 0

            # Query 2: Denied claims for this month
            denied_query = f'''
                SELECT COUNT(*) as denied
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
                  AND MONTH("{visit_date_col}") = {month}
                  AND "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") != ''
                  AND UPPER(TRIM("{denial_col}")) != 'NAN'
            '''
            denied_result = self.catalog.execute_query(denied_query, max_rows=1, timeout=30)
            denied_count = denied_result.get("rows", [{}])[0].get("denied", 0) if denied_result.get("success") else 0

            # Query 3: Denied charges for this month (if charges column exists)
            denied_charges = 0.0
            if charges_col:
                charges_query = f'''
                    SELECT SUM("{charges_col}") as total_charges
                    FROM "{table_name}"
                    WHERE YEAR("{visit_date_col}") = {year}
                      AND MONTH("{visit_date_col}") = {month}
                      AND "{denial_col}" IS NOT NULL
                      AND TRIM("{denial_col}") != ''
                      AND UPPER(TRIM("{denial_col}")) != 'NAN'
                '''
                charges_result = self.catalog.execute_query(charges_query, max_rows=1, timeout=30)
                denied_charges = charges_result.get("rows", [{}])[0].get("total_charges", 0) or 0 if charges_result.get("success") else 0

            # Calculate denial percentage
            denial_percentage = (denied_count / total_count * 100) if total_count > 0 else 0

            return {
                "year": year,
                "month": month,
                "total_count": total_count,
                "denied_count": denied_count,
                "denial_percentage": denial_percentage,
                "denied_charges": denied_charges
            }

        except Exception as e:
            logger.error(f"Error getting denial metrics for current month: {e}")
            return {
                "year": year,
                "month": month,
                "total_count": 0,
                "denied_count": 0,
                "denial_percentage": 0.0,
                "denied_charges": 0.0,
                "error": str(e)
            }

    def generate_slide_3_current_month(self, company_name: str = "Company", table_name: Optional[str] = None,
                                       date_col: Optional[str] = None, date_label: str = "Visit Date") -> str:
        """
        Generate Slide 3: Denied Claims (Current Month - Month to Date)

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)
            date_col: Date column to use (if None, will find visit_date)
            date_label: Label for the date type (e.g., "Visit Date" or "Transaction Date")

        Returns:
            Formatted current month denial analysis report as text
        """
        try:
            logger.info(f"Generating current month denial analysis for {company_name} using {date_label}...")

            # Get all available tables
            tables = self.catalog.list_tables()

            if not tables:
                return "❌ No data tables found. Please upload data first."

            # Use specified table or first available table
            target_table = table_name if table_name and table_name in tables else tables[0]

            # Find required columns
            denial_col = self._find_denial_code_column(target_table)
            if not denial_col:
                return "❌ Denial Code column not found - cannot generate current month report"

            # Use provided date_col or find visit_date
            if not date_col:
                date_col = self._find_visit_date_column(target_table)
                if not date_col:
                    return f"❌ {date_label} column not found - cannot generate current month report"

            charges_col = self._find_charges_column(target_table)

            # Get current month info (based on latest date in data)
            current_month_info = self._get_current_month_info(target_table, date_col)

            if not current_month_info:
                return "❌ Could not determine current month from data"

            year = current_month_info["year"]
            month = current_month_info["month"]

            # Get metrics for current month
            metrics = self._get_denial_metrics_for_current_month(
                target_table, denial_col, charges_col, date_col, year, month
            )

            # Month name mapping
            month_names = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }
            month_name = month_names.get(month, str(month))

            # Format the report as a markdown table
            output = f"\n{'='*60}\n"
            output += f"SLIDE 3: {company_name} - Denied Claims (Current Month)\n"
            output += f"Based on {date_label}\n"
            output += f"{'='*60}\n\n"
            output += f"**Period:** {month_name} {year} (Month to Date)\n\n"

            if not charges_col:
                output += "⚠️  WARNING: Charges column not found\n\n"

            # Create markdown table
            output += "| Metric | Value |\n"
            output += "|--------|-------|\n"
            output += f"| Count of Denied Claims | {metrics['denied_count']:,} |\n"
            output += f"| Total Claims | {metrics['total_count']:,} |\n"
            output += f"| Denial Percentage | {metrics['denial_percentage']:.2f}% |\n"
            output += f"| Total Denied Charges | ${metrics['denied_charges']:,.2f} |\n"
            output += f"\n{'='*60}\n"

            logger.info(f"Successfully generated current month denial analysis for {company_name}")
            return output

        except Exception as e:
            logger.error(f"Error generating current month denial analysis: {e}", exc_info=True)
            return f"❌ Error generating current month denial analysis: {str(e)}"

    def _get_last_week_info(self, table_name: str, visit_date_col: str) -> Optional[Dict[str, Any]]:
        """
        Get last week information based on the latest date in the data.
        Last week = the week containing the latest date (most recent week with data).

        Args:
            table_name: Name of the table
            visit_date_col: Name of the visit date column

        Returns:
            Dictionary with year, week number, start and end dates, or None if error
        """
        try:
            # Get the latest date in the data
            query = f'''
                SELECT MAX("{visit_date_col}") as max_date
                FROM "{table_name}"
                WHERE "{visit_date_col}" IS NOT NULL
            '''
            result = self.catalog.execute_query(query, max_rows=1, timeout=30)

            if result.get("success") and result.get("rows"):
                max_date = result.get("rows")[0].get("max_date")
                if max_date:
                    # Get the week information for the latest date (the week containing max_date)
                    week_query = f'''
                        SELECT
                            YEAR("{visit_date_col}") as last_week_year,
                            WEEK("{visit_date_col}") as last_week_number,
                            DATE_TRUNC('week', "{visit_date_col}") as week_start,
                            DATE_TRUNC('week', "{visit_date_col}") + INTERVAL 6 DAYS as week_end
                        FROM "{table_name}"
                        WHERE "{visit_date_col}" = '{max_date}'
                        LIMIT 1
                    '''
                    week_result = self.catalog.execute_query(week_query, max_rows=1, timeout=30)

                    if week_result.get("success") and week_result.get("rows"):
                        row = week_result.get("rows")[0]
                        year = row.get("last_week_year")
                        week_num = row.get("last_week_number")
                        week_start = row.get("week_start")
                        week_end = row.get("week_end")

                        logger.info(f"Last week determined as: Year {year}, Week {week_num} ({week_start} to {week_end})")

                        return {
                            "year": year,
                            "week_number": week_num,
                            "week_start": week_start,
                            "week_end": week_end
                        }

            return None

        except Exception as e:
            logger.error(f"Error getting last week info: {e}")
            return None

    def _get_denial_metrics_for_last_week(self, table_name: str, denial_col: str, charges_col: Optional[str],
                                          visit_date_col: str, year: int, week_number: int) -> Dict[str, Any]:
        """
        Get denial metrics for last week.

        Args:
            table_name: Name of the table
            denial_col: Name of the denial code column
            charges_col: Name of the charges column (can be None)
            visit_date_col: Name of the visit date column
            year: Year of the week
            week_number: Week number

        Returns:
            Dictionary with metrics for last week
        """
        try:
            # Query 1: Total claims for last week
            total_query = f'''
                SELECT COUNT(*) as total
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
                  AND WEEK("{visit_date_col}") = {week_number}
            '''
            total_result = self.catalog.execute_query(total_query, max_rows=1, timeout=30)
            total_count = total_result.get("rows", [{}])[0].get("total", 0) if total_result.get("success") else 0

            # Query 2: Denied claims for last week
            denied_query = f'''
                SELECT COUNT(*) as denied
                FROM "{table_name}"
                WHERE YEAR("{visit_date_col}") = {year}
                  AND WEEK("{visit_date_col}") = {week_number}
                  AND "{denial_col}" IS NOT NULL
                  AND TRIM("{denial_col}") != ''
                  AND UPPER(TRIM("{denial_col}")) != 'NAN'
            '''
            denied_result = self.catalog.execute_query(denied_query, max_rows=1, timeout=30)
            denied_count = denied_result.get("rows", [{}])[0].get("denied", 0) if denied_result.get("success") else 0

            # Query 3: Denied charges for last week (if charges column exists)
            denied_charges = 0.0
            if charges_col:
                charges_query = f'''
                    SELECT SUM("{charges_col}") as total_charges
                    FROM "{table_name}"
                    WHERE YEAR("{visit_date_col}") = {year}
                      AND WEEK("{visit_date_col}") = {week_number}
                      AND "{denial_col}" IS NOT NULL
                      AND TRIM("{denial_col}") != ''
                      AND UPPER(TRIM("{denial_col}")) != 'NAN'
                '''
                charges_result = self.catalog.execute_query(charges_query, max_rows=1, timeout=30)
                denied_charges = charges_result.get("rows", [{}])[0].get("total_charges", 0) or 0 if charges_result.get("success") else 0

            # Calculate denial percentage
            denial_percentage = (denied_count / total_count * 100) if total_count > 0 else 0

            return {
                "year": year,
                "week_number": week_number,
                "total_count": total_count,
                "denied_count": denied_count,
                "denial_percentage": denial_percentage,
                "denied_charges": denied_charges
            }

        except Exception as e:
            logger.error(f"Error getting denial metrics for last week: {e}")
            return {
                "year": year,
                "week_number": week_number,
                "total_count": 0,
                "denied_count": 0,
                "denial_percentage": 0.0,
                "denied_charges": 0.0,
                "error": str(e)
            }

    def generate_slide_4_last_week(self, company_name: str = "Company", table_name: Optional[str] = None,
                                   date_col: Optional[str] = None, date_label: str = "Visit Date") -> str:
        """
        Generate Slide 4: Denied Claims (Last Week)

        Args:
            company_name: Name of the company for the report header
            table_name: Specific table to query (if None, uses first available table)
            date_col: Date column to use (if None, will find visit_date)
            date_label: Label for the date type (e.g., "Visit Date" or "Transaction Date")

        Returns:
            Formatted last week denial analysis report as text
        """
        try:
            logger.info(f"Generating last week denial analysis for {company_name} using {date_label}...")

            # Get all available tables
            tables = self.catalog.list_tables()

            if not tables:
                return "❌ No data tables found. Please upload data first."

            # Use specified table or first available table
            target_table = table_name if table_name and table_name in tables else tables[0]

            # Find required columns
            denial_col = self._find_denial_code_column(target_table)
            if not denial_col:
                return "❌ Denial Code column not found - cannot generate last week report"

            # Use provided date_col or find visit_date
            if not date_col:
                date_col = self._find_visit_date_column(target_table)
                if not date_col:
                    return f"❌ {date_label} column not found - cannot generate last week report"

            charges_col = self._find_charges_column(target_table)

            # Get last week info (based on latest date in data)
            last_week_info = self._get_last_week_info(target_table, date_col)

            if not last_week_info:
                return "❌ Could not determine last week from data"

            year = last_week_info["year"]
            week_number = last_week_info["week_number"]
            week_start = last_week_info.get("week_start", "")
            week_end = last_week_info.get("week_end", "")

            # Get metrics for last week
            metrics = self._get_denial_metrics_for_last_week(
                target_table, denial_col, charges_col, date_col, year, week_number
            )

            # Format the report as a markdown table
            output = f"\n{'='*60}\n"
            output += f"SLIDE 4: {company_name} - Denied Claims (Latest Week)\n"
            output += f"Based on {date_label}\n"
            output += f"{'='*60}\n\n"
            output += f"**Period:** Week {week_number}, {year}"
            if week_start and week_end:
                output += f" ({week_start} to {week_end})"
            output += "\n\n"

            if not charges_col:
                output += "⚠️  WARNING: Charges column not found\n\n"

            # Create markdown table
            output += "| Metric | Value |\n"
            output += "|--------|-------|\n"
            output += f"| Count of Denied Claims | {metrics['denied_count']:,} |\n"
            output += f"| Total Claims | {metrics['total_count']:,} |\n"
            output += f"| Denial Percentage | {metrics['denial_percentage']:.2f}% |\n"
            output += f"| Total Denied Charges | ${metrics['denied_charges']:,.2f} |\n"
            output += f"\n{'='*60}\n"

            logger.info(f"Successfully generated last week denial analysis for {company_name}")
            return output

        except Exception as e:
            logger.error(f"Error generating last week denial analysis: {e}", exc_info=True)
            return f"❌ Error generating last week denial analysis: {str(e)}"

    def get_tool_description(self) -> str:
        """Get the tool description for LangGraph."""
        return """Generate denial analysis report with comprehensive denial metrics.

        Use this tool when the user asks for:
        - "/denial"
        - "denial analysis"
        - "denial report"
        - "denied claims"
        - "show me denials"
        - "denial metrics"
        - "claims denial"

        Parameters:
        - company_name: Name of the company (default: "Company")
        - table_name: Optional specific table name (default: first table)

        Returns formatted denial analysis report with FOUR slides:

        SLIDE 1 - DENIED CLAIMS (ALL TIME):
        1. Count of Denied Claims - Total number of claims with a denial code
        2. Total Claims Count - Total number of all claims
        3. Denial Percentage - Percentage of claims that were denied
        4. Total Denied Claims Charges - Sum of charges for all denied claims

        SLIDE 2 - DENIED CLAIMS (YEAR OVER YEAR):
        - Same metrics as Slide 1, broken down by year
        - Years displayed as columns in the table
        - Automatically detects all years in the data

        SLIDE 3 - DENIED CLAIMS (CURRENT MONTH):
        - Same metrics as Slide 1, for current month only
        - Current month determined by latest date in visit_date column
        - Shows month-to-date data

        SLIDE 4 - DENIED CLAIMS (LAST WEEK):
        - Same metrics as Slide 1, for last week only
        - Last week = week containing the latest date (most recent week with data)
        - Shows week number and date range

        Denial Logic (SIMPLE):
        - A claim is DENIED if denial_code column is NOT empty (has any value)
        - A claim is NOT DENIED if denial_code is empty or NULL

        Uses DuckDB SQL queries for accurate counting.
        Shows top denial codes and distribution for diagnostics.
        """


def create_denial_analysis_tool_function(catalog: DuckDBCatalog):
    """
    Create the denial analysis tool function for LangGraph.

    Args:
        catalog: DuckDB catalog instance

    Returns:
        Callable function for LangGraph tool
    """
    tool = DenialAnalysisTool(catalog)

    def denial_analysis(company_name: str = "Company", table_name: Optional[str] = None) -> str:
        """
        Generate a denial analysis report.

        Args:
            company_name: Name of the company for the report
            table_name: Optional specific table name

        Returns:
            Formatted denial analysis report
        """
        return tool.generate_report(company_name=company_name, table_name=table_name)

    return denial_analysis
