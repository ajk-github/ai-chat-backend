"""
Denial Analysis Calculator

This module provides denial analysis calculations for healthcare billing data.
Tracks denied claims based on denial_code column and calculates key metrics.

Author: AI Assistant
Created: 2025-12-09
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DenialAnalyzer:
    """
    Analyzes denial data from healthcare billing datasets.

    A claim is considered DENIED if the denial_code column is not empty.
    A claim is considered NOT DENIED if the denial_code column is empty/null.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DenialAnalyzer with a DataFrame.

        Args:
            df: DataFrame containing billing data with denial_code column
        """
        self.df = df.copy()
        self.denial_code_col = None
        self.charges_col = None

        # Find denial_code column
        denial_code_columns = ['denial_code', 'denialcode', 'denial', 'deny_code', 'rejection_code']
        for col in denial_code_columns:
            if col in self.df.columns:
                self.denial_code_col = col
                logger.info(f"Found denial code column: {col}")
                break

        # Find charges column
        charges_columns = ['charges', 'charge', 'billed_amount', 'billed', 'total_charges', 'amount']
        for col in charges_columns:
            if col in self.df.columns:
                self.charges_col = col
                logger.info(f"Found charges column: {col}")
                break

    def calculate_denied_claims_count(self) -> int:
        """
        Calculate the total count of denied claims.

        Formula: COUNT of rows where denial_code is not empty/null

        Returns:
            int: Count of denied claims
        """
        if self.denial_code_col is None:
            logger.warning("Denial code column not found")
            return 0

        # Count rows where denial_code is not null and not empty string
        denied_count = self.df[
            (self.df[self.denial_code_col].notna()) &
            (self.df[self.denial_code_col].astype(str).str.strip() != '')
        ].shape[0]

        logger.info(f"Denied claims count: {denied_count}")
        return denied_count

    def calculate_total_claims_count(self) -> int:
        """
        Calculate the total count of all claims.

        Returns:
            int: Total count of claims
        """
        total_count = self.df.shape[0]
        logger.info(f"Total claims count: {total_count}")
        return total_count

    def calculate_denial_percentage(self) -> float:
        """
        Calculate the denial percentage.

        Formula: (Denied Claims Count / Total Claims Count) * 100

        Returns:
            float: Denial percentage (0-100)
        """
        total_claims = self.calculate_total_claims_count()
        if total_claims == 0:
            logger.warning("No claims found in dataset")
            return 0.0

        denied_claims = self.calculate_denied_claims_count()
        denial_percentage = (denied_claims / total_claims) * 100

        logger.info(f"Denial percentage: {denial_percentage:.2f}%")
        return denial_percentage

    def calculate_denied_claims_charges(self) -> float:
        """
        Calculate the total charges of denied claims.

        Formula: SUM of charges where denial_code is not empty

        Returns:
            float: Total charges of denied claims
        """
        if self.denial_code_col is None:
            logger.warning("Denial code column not found")
            return 0.0

        if self.charges_col is None:
            logger.warning("Charges column not found")
            return 0.0

        # Filter denied claims
        denied_df = self.df[
            (self.df[self.denial_code_col].notna()) &
            (self.df[self.denial_code_col].astype(str).str.strip() != '')
        ]

        # Sum charges
        total_denied_charges = denied_df[self.charges_col].sum()

        logger.info(f"Total denied claims charges: ${total_denied_charges:,.2f}")
        return total_denied_charges

    def generate_slide_1_all_time(self, company_name: str = "Company") -> Dict[str, Any]:
        """
        Generate Slide 1: Denied Claims (All time)

        This slide shows:
        - Count of denied claims
        - Denial percentage
        - Total charges of denied claims

        Args:
            company_name: Name of the company for the report title

        Returns:
            Dict containing slide data and formatted text
        """
        try:
            # Calculate metrics
            denied_count = self.calculate_denied_claims_count()
            total_count = self.calculate_total_claims_count()
            denial_percentage = self.calculate_denial_percentage()
            denied_charges = self.calculate_denied_claims_charges()

            # Check for missing columns
            warnings = []
            if self.denial_code_col is None:
                warnings.append("denial_code column not found")
            if self.charges_col is None:
                warnings.append("charges column not found")

            # Build result
            result = {
                "slide_number": 1,
                "title": f"{company_name} - Denied Claims (All time)",
                "metrics": {
                    "denied_claims_count": denied_count,
                    "total_claims_count": total_count,
                    "denial_percentage": denial_percentage,
                    "denied_claims_charges": denied_charges
                },
                "warnings": warnings,
                "formatted_text": self._format_slide_1_text(
                    company_name, denied_count, total_count, denial_percentage, denied_charges, warnings
                )
            }

            return result

        except Exception as e:
            logger.error(f"Error generating Slide 1 (All time): {str(e)}", exc_info=True)
            return {
                "slide_number": 1,
                "title": f"{company_name} - Denied Claims (All time)",
                "error": str(e),
                "formatted_text": f"❌ Error generating Slide 1: {str(e)}"
            }

    def _format_slide_1_text(
        self,
        company_name: str,
        denied_count: int,
        total_count: int,
        denial_percentage: float,
        denied_charges: float,
        warnings: list
    ) -> str:
        """Format Slide 1 data as readable text."""
        output = f"\n{'='*60}\n"
        output += f"SLIDE 1: {company_name} - Denied Claims (All time)\n"
        output += f"{'='*60}\n\n"

        if warnings:
            output += "⚠️  WARNINGS:\n"
            for warning in warnings:
                output += f"   - {warning}\n"
            output += "\n"

        output += f"📊 Denial Metrics:\n"
        output += f"   • Count of Denied Claims:     {denied_count:,}\n"
        output += f"   • Total Claims:               {total_count:,}\n"
        output += f"   • Denial Percentage:          {denial_percentage:.2f}%\n"
        output += f"   • Total Denied Charges:       ${denied_charges:,.2f}\n"
        output += f"\n{'='*60}\n"

        return output

    def generate_all_slides(self, company_name: str = "Company") -> Dict[str, Any]:
        """
        Generate all denial analysis slides.

        Currently includes:
        - Slide 1: Denied Claims (All time)
        - Slide 2: TBD (to be implemented)
        - Slide 3: TBD (to be implemented)

        Args:
            company_name: Name of the company for the report

        Returns:
            Dict containing all slides data
        """
        try:
            slide_1 = self.generate_slide_1_all_time(company_name)

            # Placeholder for slides 2 and 3
            result = {
                "company_name": company_name,
                "total_slides": 3,
                "slides": {
                    "slide_1": slide_1,
                    "slide_2": {"status": "pending", "message": "Slide 2 implementation pending"},
                    "slide_3": {"status": "pending", "message": "Slide 3 implementation pending"}
                },
                "combined_text": slide_1.get("formatted_text", "")
            }

            return result

        except Exception as e:
            logger.error(f"Error generating denial analysis slides: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "combined_text": f"❌ Error generating denial analysis: {str(e)}"
            }


def analyze_denials(df: pd.DataFrame, company_name: str = "Company") -> Dict[str, Any]:
    """
    Convenience function to analyze denials from a DataFrame.

    Args:
        df: DataFrame containing billing data
        company_name: Name of the company for the report

    Returns:
        Dict containing all denial analysis slides
    """
    analyzer = DenialAnalyzer(df)
    return analyzer.generate_all_slides(company_name)
