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

        Formula: ∑ Visits or Encounters
        """
        # Try multiple column names that might represent visits
        visit_columns = ['visits', 'encounters', 'visit', 'encounter', 'visit_count']

        for col in visit_columns:
            if col in self.df.columns:
                return int(self.df[col].sum())

        # If no specific visit column, count rows as encounters
        logger.warning("No visit/encounter column found. Using row count as visits.")
        return len(self.df)

    def calculate_total_charges(self) -> float:
        """
        Calculate total charges.

        Formula: ∑ Charges
        """
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

        Formula: (Total Payments ÷ Total Charges) × 100

        Where:
        - Total Charges = SUM(unit × cpt_amount) excluding certain CPT codes and status != 12
        - Total Payments = SUM(payment amount) excluding transaction types 9, 10
        """
        # Calculate total charges with filters
        total_charges = self._calculate_filtered_charges()

        # Calculate total payments with filters
        total_payments = self._calculate_filtered_payments()

        return self._safe_divide(total_payments, total_charges, 0.0) * 100

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
        logger.info(f"  Available columns: {list(self.df.columns)}")

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

        Formula: (Total Balance) / (Average Daily Charges)

        Where:
        - Total Balance = Total Charges - Total Payments (Outstanding AR)
        - Average Daily Charges = Total Charges / 365
        """
        # Calculate total charges (filtered)
        total_charges = self._calculate_filtered_charges()

        # Calculate total payments (filtered)
        total_payments = self._calculate_filtered_payments()

        # Calculate outstanding balance (AR)
        outstanding_balance = total_charges - total_payments

        # Calculate average daily charges (annualized)
        avg_daily_charges = self._safe_divide(total_charges, 365, 0.0)

        # Calculate Days in AR
        days_in_ar = self._safe_divide(outstanding_balance, avg_daily_charges, 0.0)

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
