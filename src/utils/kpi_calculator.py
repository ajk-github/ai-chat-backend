"""
KPI Calculator
Calculates key performance indicators for RCM (Revenue Cycle Management) weekly reports.
"""
# src/utils/kpi_calculator.py

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class KPICalculator:
    """Calculate RCM KPIs from billing data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize KPI calculator with a DataFrame.

        Args:
            df: DataFrame containing billing data
        """
        self.df = df
        self._normalize_columns()

    def _normalize_columns(self):
        """Normalize column names to lowercase and replace spaces with underscores."""
        self.df.columns = self.df.columns.str.lower().str.strip().str.replace(' ', '_')

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, handling zero division.

        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division fails

        Returns:
            Result of division or default value
        """
        try:
            if denominator == 0 or pd.isna(denominator):
                return default
            result = numerator / denominator
            return result if not pd.isna(result) else default
        except (ZeroDivisionError, TypeError):
            return default

    def calculate_total_visits(self) -> int:
        """
        Calculate total visits/encounters.

        Formula: COUNT(*) WHERE visit_status IS NOT NULL
        """
        # Try multiple column names that might represent visits
        visit_columns = ['visits', 'encounters', 'visit', 'encounter', 'visit_count']

        for col in visit_columns:
            if col in self.df.columns:
                return int(self.df[col].sum())

        # If no specific visit column, count rows where visit_status is not null
        if 'visit_status' in self.df.columns:
            visit_count = self.df['visit_status'].notna().sum()
            logger.info(f"Counting visits where visit_status IS NOT NULL: {visit_count}")
            return int(visit_count)

        # Fallback: count all rows as encounters
        logger.warning("No visit/encounter column found. Using row count as visits.")
        return len(self.df)

    def calculate_total_charges(self) -> float:
        """
        Calculate total charges.

        Formula: ∑ Charges
        """
        # Updated to match actual data column names
        charge_columns = ['charge', 'charges', 'total_charges', 'billed_amount', 'charge_amount']

        for col in charge_columns:
            if col in self.df.columns:
                return float(self.df[col].sum())

        logger.warning("No charges column found.")
        return 0.0

    def calculate_charges_submitted_pct(self) -> float:
        """
        Calculate percentage of charges submitted.

        Formula: (Claimed Charges ÷ Total Charges) × 100

        Where:
        - Claimed Charges = charges where status = 'Claim Created'
        - Total Charges = all charges (excluding certain statuses)
        """
        # Try to find status column and unit/amount columns for calculation
        status_col = None
        status_columns = ['visit_status', 'claim_status', 'status', 'submission_status', 'label']
        for col in status_columns:
            if col in self.df.columns:
                status_col = col
                break

        # Look for charge amount columns
        charge_col = None
        charge_columns = ['charge', 'charges', 'charge_amount', 'total_charge', 'billed_amount']
        for col in charge_columns:
            if col in self.df.columns:
                charge_col = col
                break

        # If we have both status and charges, calculate based on status
        if status_col and charge_col:
            try:
                # Calculate claimed charges (status = 'Claim Created')
                claimed_mask = self.df[status_col].astype(str).str.lower().str.contains('claim created|submitted|billed', na=False)
                claimed_charges = self.df.loc[claimed_mask, charge_col].sum()

                # Calculate total charges (all records, or exclude certain statuses)
                total_charges = self.df[charge_col].sum()

                return self._safe_divide(claimed_charges, total_charges, 0.0) * 100
            except Exception as e:
                logger.warning(f"Error calculating charges submitted % from status: {e}")

        # Fallback: Look for pre-calculated submitted charges column
        if 'charges_submitted' in self.df.columns:
            submitted = self.df['charges_submitted'].sum()
            total = self.calculate_total_charges()
            return self._safe_divide(submitted, total, 0.0) * 100

        # Fallback: Count records by status
        if status_col:
            try:
                submitted_count = self.df[self.df[status_col].astype(str).str.lower().str.contains('claim created|submitted|billed', na=False)].shape[0]
                total_count = len(self.df)
                return self._safe_divide(submitted_count, total_count, 0.0) * 100
            except Exception as e:
                logger.warning(f"Error calculating charges submitted % by count: {e}")

        # Default: assume all charges are submitted
        logger.warning("No submission data found. Assuming 100% submitted.")
        return 100.0

    def calculate_total_payments(self) -> float:
        """
        Calculate total payments/collections.

        Formula: ∑ Payments
        """
        # Updated to match actual data column names - "Total Payment" is the main column
        payment_columns = ['total_payment', 'payment', 'payments', 'collections', 'paid_amount', 'payment_amount']

        for col in payment_columns:
            if col in self.df.columns:
                total = float(self.df[col].sum())
                logger.info(f"  Using column '{col}' for total payments: ${total:,.2f}")
                return total

        logger.warning("No payments column found.")
        return 0.0

    def calculate_gross_collection_rate(self) -> float:
        """
        Calculate gross collection rate.

        Formula: Payments ÷ Charges × 100

        Simple calculation without filters.
        """
        # Calculate total payments (simple, no filters)
        total_payments = self.calculate_total_payments()

        # Calculate total charges (simple, no filters)
        total_charges = self.calculate_total_charges()

        result = self._safe_divide(total_payments, total_charges, 0.0) * 100
        logger.info(f"Gross Collection Rate: ${total_payments:,.2f} / ${total_charges:,.2f} = {result:.2f}%")
        return result

    def _calculate_filtered_charges(self) -> float:
        """
        Calculate total charges with business logic filters.

        Excludes:
        - status = 12
        - CPT codes: 00001, 00002, 00003, 90221, 90222
        """
        df_filtered = self.df.copy()

        # Filter out status = 12 (if status column exists)
        status_columns = ['visit_status', 'claim_status', 'status']
        for col in status_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[df_filtered[col] != 12]
                    df_filtered = df_filtered[df_filtered[col] != '12']
                except:
                    pass
                break

        # Filter out specific CPT codes (if cpt_code column exists)
        excluded_cpt_codes = ['00001', '00002', '00003', '90221', '90222']
        cpt_columns = ['cpt_code', 'cpt', 'procedure_code']
        for col in cpt_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[~df_filtered[col].astype(str).isin(excluded_cpt_codes)]
                except:
                    pass
                break

        # Calculate charges: unit × cpt_amount (not used in your data - you have direct charge column)
        # But check if there's a charge column first before trying unit × amount
        charge_columns = ['charge', 'charges', 'total_charge', 'billed_amount']
        for col in charge_columns:
            if col in df_filtered.columns:
                try:
                    return float(df_filtered[col].sum())
                except:
                    pass

        # Fallback: Calculate charges: unit × cpt_amount
        unit_col = None
        amount_col = None

        # Find unit column
        unit_columns = ['unit', 'units', 'quantity', 'qty']
        for col in unit_columns:
            if col in df_filtered.columns:
                unit_col = col
                break

        # Find amount column
        amount_columns = ['cpt_amount', 'amount', 'charge_amount', 'price', 'rate']
        for col in amount_columns:
            if col in df_filtered.columns:
                amount_col = col
                break

        # Calculate total charges
        if unit_col and amount_col:
            try:
                df_filtered[unit_col] = pd.to_numeric(df_filtered[unit_col], errors='coerce').fillna(0)
                df_filtered[amount_col] = pd.to_numeric(df_filtered[amount_col], errors='coerce').fillna(0)
                total = (df_filtered[unit_col] * df_filtered[amount_col]).sum()
                return float(total)
            except Exception as e:
                logger.warning(f"Error calculating charges from unit × amount: {e}")

        logger.warning("Could not calculate filtered charges. Using unfiltered total.")
        return self.calculate_total_charges()

    def _calculate_filtered_payments(self) -> float:
        """
        Calculate total payments with business logic filters.

        Excludes:
        - transaction_type IN (9, 10)
        - status = 12
        - CPT codes: 00001, 00002, 00003, 90221, 90222
        """
        df_filtered = self.df.copy()

        # Filter out transaction types 9 and 10 (if transaction_type column exists)
        transaction_type_columns = ['transaction_type', 'payment_type', 'trans_type']
        for col in transaction_type_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[~df_filtered[col].isin([9, 10, '9', '10'])]
                except:
                    pass
                break

        # Filter out status = 12 (if status column exists)
        status_columns = ['visit_status', 'claim_status', 'status']
        for col in status_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[df_filtered[col] != 12]
                    df_filtered = df_filtered[df_filtered[col] != '12']
                except:
                    pass
                break

        # Filter out specific CPT codes (if cpt_code column exists)
        excluded_cpt_codes = ['00001', '00002', '00003', '90221', '90222']
        cpt_columns = ['cpt_code', 'cpt', 'procedure_code']
        for col in cpt_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[~df_filtered[col].astype(str).isin(excluded_cpt_codes)]
                except:
                    pass
                break

        # Calculate total payments
        payment_columns = ['total_payment', 'payment', 'payments', 'payment_amount', 'paid_amount', 'collection']
        for col in payment_columns:
            if col in df_filtered.columns:
                try:
                    return float(df_filtered[col].sum())
                except:
                    pass

        logger.warning("Could not calculate filtered payments. Using unfiltered total.")
        return self.calculate_total_payments()

    def calculate_net_collection_rate(self) -> float:
        """
        Calculate net collection rate.

        Formula: (Total Payments / Total Expected Payments) × 100

        Simple calculation without filters:
        - Total Payments = SUM(total_payment)
        - Total Expected Payments = SUM(expected)
        """
        # Calculate total payments (simple, no filters)
        total_payments = self.calculate_total_payments()

        # Calculate total expected payments (simple, no filters)
        total_expected_payments = self._calculate_expected_payments_simple()

        # Debug logging
        logger.info(f"NCR Calculation Debug:")
        logger.info(f"  Total Payments: ${total_payments:,.2f}")
        logger.info(f"  Total Expected: ${total_expected_payments:,.2f}")
        logger.info(f"  NCR: {self._safe_divide(total_payments, total_expected_payments, 0.0) * 100:.2f}%")

        return self._safe_divide(total_payments, total_expected_payments, 0.0) * 100

    def _calculate_expected_payments_simple(self) -> float:
        """
        Calculate total expected payments - simple calculation without filters.

        Uses "Expected" column, not "Allowed".
        Just SUM(expected) across all rows.
        """
        # Calculate total expected payments - no filters
        # IMPORTANT: Use "expected" column first, NOT "allowed"
        expected_columns = ['expected', 'expected_payment', 'expected_amount', 'allowed', 'allowed_amount', 'contracted_amount']
        for col in expected_columns:
            if col in self.df.columns:
                try:
                    total = float(self.df[col].sum())
                    logger.info(f"  Using column '{col}' for expected payments: ${total:,.2f}")
                    logger.info(f"  Non-null count in '{col}': {self.df[col].notna().sum()} / {len(self.df)}")
                    return total
                except Exception as e:
                    logger.warning(f"  Failed to use column '{col}': {e}")
                    pass

        # Fallback: If no allowed amount column, try to use contractual logic
        # Net = Charges - Contractuals
        contractual_columns = ['contractuals', 'contractual', 'contractual_adjustments', 'contractual_adj']
        for col in contractual_columns:
            if col in self.df.columns:
                try:
                    charges = self.calculate_total_charges()
                    contractuals = float(self.df[col].sum())
                    return charges - contractuals
                except:
                    pass

        logger.warning("Could not find allowed amount column. Using total charges as expected payments.")
        return self.calculate_total_charges()

    def _calculate_expected_payments(self) -> float:
        """
        Calculate total expected payments (allowed amounts) with business logic filters.

        Uses allowed amount from encounter_allowed_amount table.

        Excludes:
        - status = 12
        - CPT codes: 00001, 00002, 00003, 90221, 90222
        """
        df_filtered = self.df.copy()

        # Filter out status = 12 (if status column exists)
        status_columns = ['visit_status', 'claim_status', 'status']
        for col in status_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[df_filtered[col] != 12]
                    df_filtered = df_filtered[df_filtered[col] != '12']
                except:
                    pass
                break

        # Filter out specific CPT codes (if cpt_code column exists)
        excluded_cpt_codes = ['00001', '00002', '00003', '90221', '90222']
        cpt_columns = ['cpt_code', 'cpt', 'procedure_code']
        for col in cpt_columns:
            if col in df_filtered.columns:
                try:
                    df_filtered = df_filtered[~df_filtered[col].astype(str).isin(excluded_cpt_codes)]
                except:
                    pass
                break

        # Calculate total expected payments (allowed amounts)
        allowed_columns = ['allowed', 'expected', 'allowed_amount', 'expected_payment', 'expected_amount', 'contracted_amount']
        for col in allowed_columns:
            if col in df_filtered.columns:
                try:
                    return float(df_filtered[col].sum())
                except:
                    pass

        # Fallback: If no allowed amount column, try to use contractual logic
        # Net = Charges - Contractuals
        contractual_columns = ['contractuals', 'contractual', 'contractual_adjustments', 'contractual_adj']
        for col in contractual_columns:
            if col in df_filtered.columns:
                try:
                    charges = self._calculate_filtered_charges()
                    contractuals = float(df_filtered[col].sum())
                    return charges - contractuals
                except:
                    pass

        logger.warning("Could not find allowed amount column. Using filtered charges as expected payments.")
        return self._calculate_filtered_charges()

    def calculate_days_in_ar(self) -> int:
        """
        Calculate Days in AR (DAR).

        Formula: Ending AR ÷ Avg Daily Charges

        Where:
        - Ending AR = SUM(Balance) - outstanding AR
        - Average Daily Charges = Total Charges ÷ 365 (annualized)
        """
        # Calculate Ending AR from Balance column
        ending_ar = 0.0
        balance_columns = ['balance', 'ar', 'ar_balance', 'outstanding_balance']
        for col in balance_columns:
            if col in self.df.columns:
                ending_ar = float(self.df[col].sum())
                logger.info(f"  Using column '{col}' for Ending AR: ${ending_ar:,.2f}")
                break

        # If no balance column, calculate as Charges - Payments
        if ending_ar == 0.0:
            total_charges = self.calculate_total_charges()
            total_payments = self.calculate_total_payments()
            ending_ar = total_charges - total_payments
            logger.info(f"  Calculated Ending AR as Charges - Payments: ${ending_ar:,.2f}")

        # Calculate total charges
        total_charges = self.calculate_total_charges()

        # Calculate average daily charges (annualized)
        avg_daily_charges = self._safe_divide(total_charges, 365, 0.0)

        # Calculate Days in AR
        days_in_ar = self._safe_divide(ending_ar, avg_daily_charges, 0.0)

        logger.info(f"Days in AR: ${ending_ar:,.2f} / (${total_charges:,.2f} / 365) = {days_in_ar:.1f} days")

        return int(round(days_in_ar))

    def calculate_ar_31_60_days(self) -> float:
        """
        Calculate A/R in 31-60 day bucket.

        Formula: Sum of AR aged 31-60 days
        """
        # Look for AR aging columns
        ar_31_60_columns = ['ar_31_60', 'ar_31_60_days', 'ar_bucket_31_60', 'aging_31_60']

        for col in ar_31_60_columns:
            if col in self.df.columns:
                return float(self.df[col].sum())

        # Try to calculate from date columns
        if self._has_date_and_ar_columns():
            return self._calculate_aged_ar(31, 60)

        logger.warning("No AR 31-60 days data found.")
        return 0.0

    def calculate_ar_60_plus_days(self) -> float:
        """
        Calculate A/R over 60 days.

        Formula: Sum of AR aged > 60 days
        """
        # Look for AR aging columns
        ar_60_plus_columns = ['ar_60_plus', 'ar_60+', 'ar_60_days', 'ar_over_60', 'ar_bucket_60_plus', 'aging_60_plus']

        for col in ar_60_plus_columns:
            if col in self.df.columns:
                return float(self.df[col].sum())

        # Try to calculate from date columns
        if self._has_date_and_ar_columns():
            return self._calculate_aged_ar(61, 9999)

        logger.warning("No AR 60+ days data found.")
        return 0.0

    def calculate_denial_resolution_rate(self) -> float:
        """
        Calculate denial vs resolution rate.

        Formula: (Resolved Denials ÷ Total Denials) × 100
        """
        # Look for denial and resolution columns
        resolved_denials = 0.0
        total_denials = 0.0

        # Try to find resolved denials
        resolved_columns = ['resolved_denials', 'denials_resolved', 'denial_recoveries']
        for col in resolved_columns:
            if col in self.df.columns:
                resolved_denials = float(self.df[col].sum())
                break

        # Try to find total denials
        denial_columns = ['denials', 'denial_amount', 'total_denials', 'denied_amount']
        for col in denial_columns:
            if col in self.df.columns:
                total_denials = float(self.df[col].sum())
                break

        # If we have a status column, try to count denials and resolutions
        if total_denials == 0.0:
            status_columns = ['status', 'claim_status', 'denial_status']
            for col in status_columns:
                if col in self.df.columns:
                    denial_statuses = ['denied', 'denial', 'rejected']
                    resolved_statuses = ['resolved', 'paid', 'approved', 'accepted']

                    df_lower = self.df[col].str.lower()
                    total_denials = df_lower.isin(denial_statuses).sum()
                    resolved_denials = df_lower.isin(resolved_statuses).sum()
                    break

        return self._safe_divide(resolved_denials, total_denials, 0.0) * 100

    def _has_date_and_ar_columns(self) -> bool:
        """Check if dataframe has date and AR balance columns."""
        date_columns = ['date', 'dos', 'date_of_service', 'service_date', 'transaction_date']
        ar_columns = ['ar', 'balance', 'ar_balance', 'outstanding_balance']

        has_date = any(col in self.df.columns for col in date_columns)
        has_ar = any(col in self.df.columns for col in ar_columns)

        return has_date and has_ar

    def _calculate_aged_ar(self, min_days: int, max_days: int) -> float:
        """Calculate AR for a specific aging bucket."""
        # Find date and AR columns
        date_col = None
        date_columns = ['date', 'dos', 'date_of_service', 'service_date', 'transaction_date']
        for col in date_columns:
            if col in self.df.columns:
                date_col = col
                break

        ar_col = None
        ar_columns = ['ar', 'balance', 'ar_balance', 'outstanding_balance']
        for col in ar_columns:
            if col in self.df.columns:
                ar_col = col
                break

        if date_col is None or ar_col is None:
            return 0.0

        try:
            # Calculate days from today
            df_copy = self.df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            today = datetime.now()
            df_copy['days_old'] = (today - df_copy[date_col]).dt.days

            # Filter by aging bucket
            mask = (df_copy['days_old'] >= min_days) & (df_copy['days_old'] <= max_days)
            return float(df_copy.loc[mask, ar_col].sum())
        except Exception as e:
            logger.error(f"Error calculating aged AR: {e}")
            return 0.0

    def generate_weekly_report(self, company_name: str = "Company") -> Dict[str, Any]:
        """
        Generate a complete weekly KPI report.

        Args:
            company_name: Name of the company for the report

        Returns:
            Dictionary containing all 10 KPI metrics
        """
        logger.info(f"Generating weekly report for {company_name}...")

        report = {
            "company": company_name,
            "generated_at": datetime.now().isoformat(),
            "kpis": {
                "visits": self.calculate_total_visits(),
                "charges": self.calculate_total_charges(),
                "charges_submitted_pct": round(self.calculate_charges_submitted_pct(), 2),
                "payments": self.calculate_total_payments(),
                "gross_collection_rate_pct": round(self.calculate_gross_collection_rate(), 2),
                "net_collection_rate_pct": round(self.calculate_net_collection_rate(), 2),
                "days_in_ar": self.calculate_days_in_ar(),
                "ar_31_60_days": self.calculate_ar_31_60_days(),
                "ar_60_plus_days": self.calculate_ar_60_plus_days(),
                "denial_resolution_rate_pct": round(self.calculate_denial_resolution_rate(), 2),
            }
        }

        logger.info(f"Weekly report generated successfully for {company_name}")
        return report

    def calculate_billed_ar(self) -> float:
        """
        Calculate total Billed AR (Accounts Receivable that has been billed).

        Formula: Sum of Balance where Claim Status or Visit Status indicates billed/submitted
        Uses "Balance" column which represents outstanding AR for each claim.
        """
        # Updated to use "Claim Status" and "Visit Status" from actual data
        status_columns = ['claim_status', 'visit_status', 'status', 'label']
        # Updated to use "Balance" column for AR calculation
        balance_columns = ['balance', 'ar', 'ar_balance', 'outstanding_balance']

        status_col = None
        for col in status_columns:
            if col in self.df.columns:
                status_col = col
                break

        balance_col = None
        for col in balance_columns:
            if col in self.df.columns:
                balance_col = col
                break

        if status_col and balance_col:
            try:
                # Billed AR = outstanding balance where claim has been created/submitted
                billed_mask = self.df[status_col].astype(str).str.lower().str.contains(
                    'claim created|submitted|billed|pending|claim submitted', na=False
                )
                return float(self.df.loc[billed_mask, balance_col].sum())
            except Exception as e:
                logger.warning(f"Error calculating billed AR from balance: {e}")

        # Fallback: use balance column total if available
        if balance_col:
            return float(self.df[balance_col].sum())

        # Final fallback: calculate as charges - payments
        total_charges = self.calculate_total_charges()
        total_payments = self.calculate_total_payments()
        return total_charges - total_payments

    def calculate_unbilled_ar(self) -> float:
        """
        Calculate total Unbilled AR (charges not yet billed).

        Formula: Sum of Balance where Claim Status or Visit Status indicates NOT billed/submitted
        """
        # Updated to use "Claim Status" and "Visit Status" from actual data
        status_columns = ['claim_status', 'visit_status', 'status', 'label']
        # Updated to use "Balance" column for AR calculation
        balance_columns = ['balance', 'ar', 'ar_balance', 'outstanding_balance']

        status_col = None
        for col in status_columns:
            if col in self.df.columns:
                status_col = col
                break

        balance_col = None
        for col in balance_columns:
            if col in self.df.columns:
                balance_col = col
                break

        if status_col and balance_col:
            try:
                # Unbilled AR = outstanding balance where claim has NOT been created/submitted yet
                unbilled_mask = ~self.df[status_col].astype(str).str.lower().str.contains(
                    'claim created|submitted|billed|pending|claim submitted', na=False
                )
                return float(self.df.loc[unbilled_mask, balance_col].sum())
            except Exception as e:
                logger.warning(f"Error calculating unbilled AR from balance: {e}")

        # Fallback: calculate as total charges - billed AR
        total_charges = self.calculate_total_charges()
        billed_ar = self.calculate_billed_ar()
        return max(0.0, total_charges - billed_ar)

    def calculate_billed_ar_percentage(self) -> float:
        """
        Calculate percentage of AR that has been billed.

        Formula: (Billed AR / Total Balance) × 100
        """
        billed_ar = self.calculate_billed_ar()
        unbilled_ar = self.calculate_unbilled_ar()
        total_ar = billed_ar + unbilled_ar
        return self._safe_divide(billed_ar, total_ar, 0.0) * 100

    def calculate_unbilled_ar_percentage(self) -> float:
        """
        Calculate percentage of AR that is unbilled.

        Formula: (Unbilled AR / Total Balance) × 100
        """
        billed_ar = self.calculate_billed_ar()
        unbilled_ar = self.calculate_unbilled_ar()
        total_ar = billed_ar + unbilled_ar
        return self._safe_divide(unbilled_ar, total_ar, 0.0) * 100

    def generate_weekly_breakdown_report(self, company_name: str = "Company", year: int = None, month: int = None) -> Dict[str, Any]:
        """
        Generate a weekly breakdown report for a specific month.

        Args:
            company_name: Name of the company for the report
            year: Year to report on (if None, auto-detects latest month with data)
            month: Month to report on (if None, auto-detects latest month with data)

        Returns:
            Dictionary containing weekly breakdown metrics
        """
        logger.info(f"Generating weekly breakdown report for {company_name}...")

        # Find date column
        date_col = None
        date_columns = ['visit_date', 'date', 'dos', 'date_of_service', 'service_date', 'transaction_date', 'claim_date']
        for col in date_columns:
            if col in self.df.columns:
                date_col = col
                break

        if date_col is None:
            logger.warning("No date column found. Cannot generate weekly breakdown report.")
            return {
                "company": company_name,
                "year": year,
                "month": month,
                "generated_at": datetime.now().isoformat(),
                "error": "No date field found in data.",
                "weeks": {}
            }

        try:
            # Parse dates
            df_copy = self.df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy[df_copy[date_col].notna()]

            if df_copy.empty:
                return {
                    "company": company_name,
                    "year": year,
                    "month": month,
                    "generated_at": datetime.now().isoformat(),
                    "error": "No valid dates found in data.",
                    "weeks": {}
                }

            # Auto-detect latest year/month if not specified
            if year is None or month is None:
                latest_date = df_copy[date_col].max()
                year = latest_date.year
                month = latest_date.month
                logger.info(f"Auto-detected latest month: {year}-{month:02d}")

            # Filter for specific year and month
            df_filtered = df_copy[
                (df_copy[date_col].dt.year == year) &
                (df_copy[date_col].dt.month == month)
            ]

            if df_filtered.empty:
                return {
                    "company": company_name,
                    "year": year,
                    "month": month,
                    "generated_at": datetime.now().isoformat(),
                    "error": f"No data found for {year}-{month:02d}.",
                    "weeks": {}
                }

            # Add week number (week of month)
            df_filtered['day_of_month'] = df_filtered[date_col].dt.day
            df_filtered['week_of_month'] = ((df_filtered['day_of_month'] - 1) // 7) + 1

            # Group by week of month and calculate individual week metrics first
            weekly_data = {}
            weeks_sorted = sorted(df_filtered['week_of_month'].unique())

            # Calculate individual week metrics
            week_metrics = {}
            for week_num in weeks_sorted:
                week_df = df_filtered[df_filtered['week_of_month'] == week_num]
                week_calc = KPICalculator(week_df)

                week_metrics[int(week_num)] = {
                    "visits": week_calc.calculate_total_visits(),
                    "collections": week_calc._calculate_filtered_payments(),  # Use filtered payments
                    "expected": week_calc._calculate_expected_payments(),  # Use filtered expected payments
                }

            # Calculate cumulative values
            cumulative_visits = 0
            cumulative_collections = 0
            cumulative_expected = 0

            for week_num in weeks_sorted:
                metrics = week_metrics[int(week_num)]

                # Add this week's values to cumulative totals
                cumulative_visits += metrics["visits"]
                cumulative_collections += metrics["collections"]
                cumulative_expected += metrics["expected"]

                weekly_data[f"week_{int(week_num)}"] = {
                    "week": int(week_num),
                    "visits": cumulative_visits,  # Cumulative
                    "collections": cumulative_collections,  # Cumulative
                    "forecasted": cumulative_expected,  # Cumulative expected payments
                }

            logger.info(f"Weekly breakdown report generated for {year}-{month:02d}")
            return {
                "company": company_name,
                "year": year,
                "month": month,
                "month_name": datetime(year, month, 1).strftime('%B'),
                "generated_at": datetime.now().isoformat(),
                "weeks": weekly_data
            }

        except Exception as e:
            logger.error(f"Error generating weekly breakdown report: {e}", exc_info=True)
            return {
                "company": company_name,
                "year": year,
                "month": month,
                "generated_at": datetime.now().isoformat(),
                "error": f"Error generating weekly breakdown: {str(e)}",
                "weeks": {}
            }

    def format_weekly_breakdown_as_text(self, report: Dict[str, Any]) -> str:
        """
        Format the weekly breakdown report as human-readable text.

        Args:
            report: Report dictionary from generate_weekly_breakdown_report()

        Returns:
            Formatted text weekly breakdown report
        """
        company = report["company"]
        year = report.get("year")
        month_name = report.get("month_name", "")

        if "error" in report:
            return f"❌ Error: {report['error']}"

        weeks = report.get("weeks", {})

        if not weeks:
            return f"❌ No weekly data available for {month_name} {year}."

        # Sort weeks
        sorted_weeks = sorted(weeks.items(), key=lambda x: x[1]['week'])

        # Format output
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║          {company.upper()} - WEEKLY BREAKDOWN
║          {month_name} {year}
╚══════════════════════════════════════════════════════════════╝

WEEKLY COLLECTIONS - {month_name} {year}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forecasted              Week         Week Actual
────────────────────────────────────────────────────────────────"""

        for week_key, week_data in sorted_weeks:
            week_num = week_data['week']
            forecasted = week_data.get('forecasted', 0)
            collections = week_data['collections']
            output += f"\n${forecasted:,.2f}".ljust(24) + f"Week {week_num}".ljust(13) + f"${collections:,.2f}"

        output += f"""


WEEKLY VISITS BREAKDOWN - {month_name} {year}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week                              Visits (Cumulative)
────────────────────────────────────────────────────────────────"""

        for week_key, week_data in sorted_weeks:
            week_num = week_data['week']
            visits = week_data['visits']
            output += f"\nWeek {week_num}                           {visits:,}"

        output += f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"Generated: {report['generated_at']}\n"

        return output.strip()

    def generate_yearly_report(self, company_name: str = "Company") -> Dict[str, Any]:
        """
        Generate a yearly KPI report grouped by year.

        Args:
            company_name: Name of the company for the report

        Returns:
            Dictionary containing yearly KPI metrics organized by year
        """
        logger.info(f"Generating yearly report for {company_name}...")

        # Check if Year column already exists
        has_year_col = 'year' in self.df.columns

        df_copy = self.df.copy()
        use_date_fallback = False

        if has_year_col:
            # Use pre-calculated Year column
            logger.info("Using pre-calculated Year column")
            logger.info(f"Total rows before filtering: {len(df_copy)}")

            # Ensure it's numeric
            if not pd.api.types.is_numeric_dtype(df_copy['year']):
                df_copy['year'] = pd.to_numeric(df_copy['year'], errors='coerce')

            # Remove rows with invalid year
            valid_rows = df_copy['year'].notna().sum()
            logger.info(f"Rows with valid year: {valid_rows}")

            if valid_rows == 0 or valid_rows < len(df_copy) * 0.5:
                logger.warning(f"Too many null values in Year column. Falling back to date parsing.")
                use_date_fallback = True
            else:
                df_copy = df_copy[df_copy['year'].notna()]
                logger.info(f"Total rows after filtering: {len(df_copy)}")
        else:
            use_date_fallback = True

        if use_date_fallback:
            # Fallback: Find date column and extract year
            date_col = None
            date_columns = ['visit_date', 'date', 'dos', 'date_of_service', 'service_date', 'transaction_date', 'claim_date']
            for col in date_columns:
                if col in self.df.columns:
                    date_col = col
                    break

            if date_col is None:
                logger.warning("No Year column or date column found. Cannot generate yearly report.")
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": "No Year column or date field found in data.",
                    "years": {}
                }

            try:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy = df_copy[df_copy[date_col].notna()]

                if df_copy.empty:
                    return {
                        "company": company_name,
                        "generated_at": datetime.now().isoformat(),
                        "error": "No valid dates found in data.",
                        "years": {}
                    }

                df_copy['year'] = df_copy[date_col].dt.year

            except Exception as e:
                logger.error(f"Error parsing dates: {e}")
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": f"Error parsing dates: {str(e)}",
                    "years": {}
                }

        try:
            if df_copy.empty:
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": "No valid data after filtering.",
                    "years": {}
                }

            # Group by year
            yearly_data = {}

            for year, group_df in df_copy.groupby('year'):
                # Create a temporary calculator for this year's data
                year_calc = KPICalculator(group_df)

                year_key = f"{int(year)}"
                yearly_data[year_key] = {
                    "year": int(year),
                    "kpis": {
                        "visits": year_calc.calculate_total_visits(),
                        "charges": year_calc.calculate_total_charges(),
                        "charges_submitted_pct": round(year_calc.calculate_charges_submitted_pct(), 2),
                        "payments": year_calc.calculate_total_payments(),
                        "gross_collection_rate_pct": round(year_calc.calculate_gross_collection_rate(), 2),
                        "net_collection_rate_pct": round(year_calc.calculate_net_collection_rate(), 2),
                        "days_in_ar": year_calc.calculate_days_in_ar(),
                        "billed_ar": year_calc.calculate_billed_ar(),
                        "unbilled_ar": year_calc.calculate_unbilled_ar(),
                        "denial_resolution_rate_pct": round(year_calc.calculate_denial_resolution_rate(), 2),
                    }
                }

            logger.info(f"Yearly report generated successfully for {company_name}")
            return {
                "company": company_name,
                "generated_at": datetime.now().isoformat(),
                "years": yearly_data
            }

        except Exception as e:
            logger.error(f"Error generating yearly report: {e}", exc_info=True)
            return {
                "company": company_name,
                "generated_at": datetime.now().isoformat(),
                "error": f"Error generating yearly report: {str(e)}",
                "years": {}
            }

    def format_yearly_report_as_text(self, report: Dict[str, Any]) -> str:
        """
        Format the yearly report as human-readable text.

        Args:
            report: Report dictionary from generate_yearly_report()

        Returns:
            Formatted text yearly report
        """
        company = report["company"]

        if "error" in report:
            return f"❌ Error: {report['error']}"

        years = report.get("years", {})

        if not years:
            return "❌ No yearly data available."

        # Format output
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║          {company.upper()} - YEAR OVER YEAR KPI REPORT
╚══════════════════════════════════════════════════════════════╝

Type                        """

        # Header row with years
        sorted_years = sorted(years.keys())
        for year in sorted_years:
            output += f"{year}            "
        output += "\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        # Get KPIs for each year
        years_kpis = [years[year]["kpis"] for year in sorted_years]

        # Visits
        output += "Visits                      "
        for kpis in years_kpis:
            output += f"{kpis.get('visits', 0):,}".ljust(17)
        output += "\n"

        # Charges
        output += "Charges                     "
        for kpis in years_kpis:
            output += f"${kpis.get('charges', 0):,.2f}".ljust(17)
        output += "\n"

        # Charges Submitted %
        output += "Charges Submitted (%)       "
        for kpis in years_kpis:
            output += f"{kpis.get('charges_submitted_pct', 0):.2f}%".ljust(17)
        output += "\n"

        # Payments
        output += "Payments                    "
        for kpis in years_kpis:
            output += f"${kpis.get('payments', 0):,.2f}".ljust(17)
        output += "\n"

        # Gross Collection Rate
        output += "Gross Collection Rate (%)   "
        for kpis in years_kpis:
            output += f"{kpis.get('gross_collection_rate_pct', 0):.2f}%".ljust(17)
        output += "\n"

        # Net Collection Rate
        output += "Net Collection Rate (%)     "
        for kpis in years_kpis:
            output += f"{kpis.get('net_collection_rate_pct', 0):.2f}%".ljust(17)
        output += "\n"

        # Days in AR
        output += "Days in AR (DAR)            "
        for kpis in years_kpis:
            output += f"{kpis.get('days_in_ar', 0)} Days".ljust(17)
        output += "\n"

        # Billed AR
        output += "Billed AR                   "
        for kpis in years_kpis:
            output += f"${kpis.get('billed_ar', 0):,.2f}".ljust(17)
        output += "\n"

        # Unbilled AR
        output += "Unbilled AR                 "
        for kpis in years_kpis:
            output += f"${kpis.get('unbilled_ar', 0):,.2f}".ljust(17)
        output += "\n"

        # Denial vs Resolution
        output += "Denial vs Resolution (%)    "
        for kpis in years_kpis:
            output += f"{kpis.get('denial_resolution_rate_pct', 0):.2f}%".ljust(17)
        output += "\n"

        output += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"Generated: {report['generated_at']}\n"

        return output.strip()

    def generate_quarterly_report(self, company_name: str = "Company") -> Dict[str, Any]:
        """
        Generate a quarterly KPI report grouped by year and quarter.

        Args:
            company_name: Name of the company for the report

        Returns:
            Dictionary containing quarterly KPI metrics organized by year and quarter
        """
        logger.info(f"Generating quarterly report for {company_name}...")

        # Check if Year and Quarter columns already exist (preferred method)
        has_year_col = 'year' in self.df.columns
        has_quarter_col = 'quarter' in self.df.columns

        df_copy = self.df.copy()
        use_date_fallback = False

        if has_year_col and has_quarter_col:
            # Use pre-calculated Year and Quarter columns
            logger.info("Using pre-calculated Year and Quarter columns")
            logger.info(f"Total rows before filtering: {len(df_copy)}")

            # Check original data types
            logger.info(f"Year column dtype: {df_copy['year'].dtype}")
            logger.info(f"Quarter column dtype: {df_copy['quarter'].dtype}")
            logger.info(f"Year sample values: {df_copy['year'].head(10).tolist()}")
            logger.info(f"Quarter sample values: {df_copy['quarter'].head(10).tolist()}")

            # Try to convert to numeric - keep original if already numeric
            if not pd.api.types.is_numeric_dtype(df_copy['year']):
                df_copy['year'] = pd.to_numeric(df_copy['year'], errors='coerce')
            if not pd.api.types.is_numeric_dtype(df_copy['quarter']):
                df_copy['quarter'] = pd.to_numeric(df_copy['quarter'], errors='coerce')

            # Log data quality info
            year_nulls = df_copy['year'].isna().sum()
            quarter_nulls = df_copy['quarter'].isna().sum()
            valid_rows = ((df_copy['year'].notna()) & (df_copy['quarter'].notna())).sum()

            logger.info(f"Year column null count: {year_nulls}/{len(df_copy)}")
            logger.info(f"Quarter column null count: {quarter_nulls}/{len(df_copy)}")
            logger.info(f"Rows with valid year AND quarter: {valid_rows}")

            # If too many nulls, fall back to date parsing
            if valid_rows == 0 or valid_rows < len(df_copy) * 0.5:  # If 0 or more than 50% are null
                logger.warning(f"Too many null values in Year/Quarter columns ({valid_rows}/{len(df_copy)} valid). Falling back to date parsing.")
                use_date_fallback = True
            else:
                # Remove rows with invalid year/quarter
                df_copy = df_copy[df_copy['year'].notna() & df_copy['quarter'].notna()]
                logger.info(f"Total rows after filtering: {len(df_copy)}")
        else:
            use_date_fallback = True

        if use_date_fallback:
            # Fallback: Find date column and calculate year/quarter
            date_col = None
            date_columns = ['visit_date', 'date', 'dos', 'date_of_service', 'service_date', 'transaction_date', 'claim_date']
            for col in date_columns:
                if col in self.df.columns:
                    date_col = col
                    break

            if date_col is None:
                logger.warning("No Year/Quarter columns or date column found. Cannot generate quarterly report.")
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": "No Year/Quarter columns or date field found in data.",
                    "quarters": {}
                }

            try:
                # Parse dates and extract year/quarter
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy = df_copy[df_copy[date_col].notna()]

                if df_copy.empty:
                    return {
                        "company": company_name,
                        "generated_at": datetime.now().isoformat(),
                        "error": "No valid dates found in data.",
                        "quarters": {}
                    }

                df_copy['year'] = df_copy[date_col].dt.year
                df_copy['quarter'] = df_copy[date_col].dt.quarter

            except Exception as e:
                logger.error(f"Error parsing dates: {e}")
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": f"Error parsing dates: {str(e)}",
                    "quarters": {}
                }

        try:
            if df_copy.empty:
                return {
                    "company": company_name,
                    "generated_at": datetime.now().isoformat(),
                    "error": "No valid data after filtering.",
                    "quarters": {}
                }

            # Group by year and quarter
            quarterly_data = {}

            for (year, quarter), group_df in df_copy.groupby(['year', 'quarter']):
                # Create a temporary calculator for this quarter's data
                quarter_calc = KPICalculator(group_df)

                quarter_key = f"{int(year)}_Q{int(quarter)}"
                quarterly_data[quarter_key] = {
                    "year": int(year),
                    "quarter": int(quarter),
                    "kpis": {
                        "visits": quarter_calc.calculate_total_visits(),
                        "charges": quarter_calc.calculate_total_charges(),
                        "charges_submitted_pct": round(quarter_calc.calculate_charges_submitted_pct(), 2),
                        "payments": quarter_calc.calculate_total_payments(),
                        "gross_collection_rate_pct": round(quarter_calc.calculate_gross_collection_rate(), 2),
                        "net_collection_rate_pct": round(quarter_calc.calculate_net_collection_rate(), 2),
                        "days_in_ar": quarter_calc.calculate_days_in_ar(),
                        "billed_ar": quarter_calc.calculate_billed_ar(),
                        "billed_ar_pct": round(quarter_calc.calculate_billed_ar_percentage(), 2),
                        "unbilled_ar": quarter_calc.calculate_unbilled_ar(),
                        "unbilled_ar_pct": round(quarter_calc.calculate_unbilled_ar_percentage(), 2),
                        "denial_resolution_rate_pct": round(quarter_calc.calculate_denial_resolution_rate(), 2),
                    }
                }

            logger.info(f"Quarterly report generated successfully for {company_name}")
            return {
                "company": company_name,
                "generated_at": datetime.now().isoformat(),
                "quarters": quarterly_data
            }

        except Exception as e:
            logger.error(f"Error generating quarterly report: {e}", exc_info=True)
            return {
                "company": company_name,
                "generated_at": datetime.now().isoformat(),
                "error": f"Error generating quarterly report: {str(e)}",
                "quarters": {}
            }

    def format_quarterly_report_as_text(self, report: Dict[str, Any]) -> str:
        """
        Format the quarterly report as human-readable text.

        Args:
            report: Report dictionary from generate_quarterly_report()

        Returns:
            Formatted text quarterly report
        """
        company = report["company"]

        if "error" in report:
            return f"❌ Error: {report['error']}"

        quarters = report.get("quarters", {})

        if not quarters:
            return "❌ No quarterly data available."

        # Group quarters by year
        years = {}
        for quarter_key, quarter_data in quarters.items():
            year = quarter_data["year"]
            if year not in years:
                years[year] = {}
            years[year][quarter_data["quarter"]] = quarter_data["kpis"]

        # Format output
        output = f"""
╔══════════════════════════════════════════════════════════════╗
║          {company.upper()} - QUARTERLY KPI REPORT
╚══════════════════════════════════════════════════════════════╝

"""

        for year in sorted(years.keys()):
            output += f"📅 {year}\n"
            output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            output += "Type                        "

            # Header row with quarters
            for q in sorted(years[year].keys()):
                output += f"Q{q}              "
            output += "\n"
            output += "────────────────────────────────────────────────────────────────\n"

            # Get all quarters for this year (sorted)
            quarters_list = [years[year].get(q, {}) for q in sorted(years[year].keys())]

            # Visits
            output += f"Visits                      "
            for kpis in quarters_list:
                output += f"{kpis.get('visits', 0):,}".ljust(17)
            output += "\n"

            # Charges
            output += f"Charges                     "
            for kpis in quarters_list:
                output += f"${kpis.get('charges', 0):,.2f}".ljust(17)
            output += "\n"

            # Charges Submitted %
            output += f"Charges Submitted (%)       "
            for kpis in quarters_list:
                output += f"{kpis.get('charges_submitted_pct', 0):.2f}%".ljust(17)
            output += "\n"

            # Payments
            output += f"Payments                    "
            for kpis in quarters_list:
                output += f"${kpis.get('payments', 0):,.2f}".ljust(17)
            output += "\n"

            # Gross Collection Rate
            output += f"Gross Collection Rate (%)   "
            for kpis in quarters_list:
                output += f"{kpis.get('gross_collection_rate_pct', 0):.2f}%".ljust(17)
            output += "\n"

            # Net Collection Rate
            output += f"Net Collection Rate (%)     "
            for kpis in quarters_list:
                output += f"{kpis.get('net_collection_rate_pct', 0):.2f}%".ljust(17)
            output += "\n"

            # Days in AR
            output += f"Days in AR (DAR)            "
            for kpis in quarters_list:
                output += f"{kpis.get('days_in_ar', 0)} Days".ljust(17)
            output += "\n"

            # Billed AR
            output += f"Billed AR                   "
            for kpis in quarters_list:
                output += f"${kpis.get('billed_ar', 0):,.2f}".ljust(17)
            output += "\n"

            # Billed AR %
            output += f"Billed AR %                 "
            for kpis in quarters_list:
                output += f"{kpis.get('billed_ar_pct', 0):.2f}%".ljust(17)
            output += "\n"

            # Unbilled AR
            output += f"Unbilled AR                 "
            for kpis in quarters_list:
                output += f"${kpis.get('unbilled_ar', 0):,.2f}".ljust(17)
            output += "\n"

            # Unbilled AR %
            output += f"Unbilled AR (%)             "
            for kpis in quarters_list:
                output += f"{kpis.get('unbilled_ar_pct', 0):.2f}%".ljust(17)
            output += "\n"

            # Denial vs Resolution
            output += f"Denial vs Resolution (%)    "
            for kpis in quarters_list:
                output += f"{kpis.get('denial_resolution_rate_pct', 0):.2f}%".ljust(17)
            output += "\n"

            output += "\n"

        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"Generated: {report['generated_at']}\n"

        return output.strip()

    def format_report_as_text(self, report: Dict[str, Any]) -> str:
        """
        Format the report as human-readable text.

        Args:
            report: Report dictionary from generate_weekly_report()

        Returns:
            Formatted text report
        """
        kpis = report["kpis"]
        company = report["company"]

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║          {company.upper()} - WEEKLY KPI REPORT
╚══════════════════════════════════════════════════════════════╝

📊 KEY PERFORMANCE INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Type                              Value
────────────────────────────────────────────────────────────────
Visits                            {kpis['visits']:,}
Charges                           ${kpis['charges']:,.2f}
Charges Submitted (%)             {kpis['charges_submitted_pct']:.2f}%
Payments                          ${kpis['payments']:,.2f}
Gross Collection Rate (%)         {kpis['gross_collection_rate_pct']:.2f}%
Net Collection Rate (%)           {kpis['net_collection_rate_pct']:.2f}%
Days in AR (DAR)                  {kpis['days_in_ar']} Days
A/R (31-60 Days)                  ${kpis['ar_31_60_days']:,.2f}
A/R (60+ Days)                    ${kpis['ar_60_plus_days']:,.2f}
Denial vs Resolution (%)          {kpis['denial_resolution_rate_pct']:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: {report['generated_at']}
"""
        return output.strip()
