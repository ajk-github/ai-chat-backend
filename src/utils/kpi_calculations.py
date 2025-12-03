"""
Pure KPI calculation functions.

These functions perform individual metric calculations on DataFrames.
They are shared across all company reports and can be used by AI agents
to answer specific metric questions.

Each function takes a DataFrame and necessary parameters, and returns
a calculated value without any formatting.
"""

import pandas as pd
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# BASIC METRICS
# ============================================================================

def calculate_total_visits(df: pd.DataFrame) -> int:
    """
    Calculate total number of visits/encounters.

    Args:
        df: DataFrame with visit data

    Returns:
        Count of visits
    """
    return len(df)


def calculate_total_charges(df: pd.DataFrame, charge_col: str = 'charge') -> float:
    """
    Calculate total charges (excluding negative values).

    Args:
        df: DataFrame with charge data
        charge_col: Name of the charge column

    Returns:
        Sum of positive charges
    """
    if charge_col not in df.columns:
        return 0.0

    return float(df[df[charge_col] > 0][charge_col].sum())


def calculate_total_payments(df: pd.DataFrame, payment_col: str = 'total_payment') -> float:
    """
    Calculate total payments.

    Args:
        df: DataFrame with payment data
        payment_col: Name of the payment column

    Returns:
        Sum of payments
    """
    if payment_col not in df.columns:
        return 0.0

    return float(df[payment_col].sum())


def calculate_balance(df: pd.DataFrame, balance_col: str = 'balance') -> float:
    """
    Calculate total outstanding balance.

    Args:
        df: DataFrame with balance data
        balance_col: Name of the balance column

    Returns:
        Sum of balances
    """
    if balance_col not in df.columns:
        return 0.0

    return float(df[balance_col].sum())


# ============================================================================
# COLLECTION RATES
# ============================================================================

def calculate_gross_collection_rate(
    total_payments: float,
    total_charges: float
) -> float:
    """
    Calculate Gross Collection Rate (GCR).

    Formula: (Total Payments / Total Charges) × 100

    Args:
        total_payments: Sum of all payments
        total_charges: Sum of all charges

    Returns:
        GCR as a percentage
    """
    if total_charges == 0:
        return 0.0

    return (total_payments / total_charges) * 100


def calculate_net_collection_rate(
    total_payments: float,
    total_expected: float
) -> float:
    """
    Calculate Net Collection Rate (NCR).

    Formula: (Total Payments / Total Expected) × 100

    Args:
        total_payments: Sum of all payments
        total_expected: Sum of expected/allowed amounts

    Returns:
        NCR as a percentage
    """
    if total_expected == 0:
        return 0.0

    return (total_payments / total_expected) * 100


# ============================================================================
# ACCOUNTS RECEIVABLE METRICS
# ============================================================================

def calculate_days_in_ar(
    ending_ar: float,
    total_charges: float,
    period_days: int
) -> int:
    """
    Calculate Days in AR (DAR) - Generic version.

    Formula: Ending AR / (Total Charges / Period Days)

    Args:
        ending_ar: Outstanding balance
        total_charges: Total charges for the period
        period_days: Number of days in the period

    Returns:
        Days in AR

    Note:
        For specific report types, use:
        - calculate_days_in_ar_weekly() for Slide 3 (Weekly KPI)
        - calculate_days_in_ar_quarterly() for Slide 4 (Quarterly)
        - calculate_days_in_ar_yearly() for Slide 5 (Yearly)
    """
    if total_charges == 0 or period_days == 0:
        return 0

    average_daily_charges = total_charges / period_days

    if average_daily_charges == 0:
        return 0

    return math.ceil(ending_ar / average_daily_charges)


def calculate_days_in_ar_weekly(
    ending_ar: float,
    total_charges: float,
    df: pd.DataFrame,
    date_col: str = 'visit_date'
) -> int:
    """
    Calculate Days in AR for Weekly KPI Metrics (Slide 3).

    Period Days Logic:
    - If data contains full year(s): 365 days (fixed, no leap year adjustment)
    - If data is current year only: Days from Jan 1 to today

    Formula: Ending AR / (Total Charges / Period Days)

    Args:
        ending_ar: Outstanding balance
        total_charges: Total charges
        df: DataFrame with date column
        date_col: Name of date column (default: 'visit_date')

    Returns:
        Days in AR as integer
    """
    period_days = _get_period_days_weekly(df, date_col)

    if total_charges == 0 or period_days == 0:
        return 0

    avg_daily_charges = total_charges / period_days
    if avg_daily_charges == 0:
        return 0

    return math.ceil(ending_ar / avg_daily_charges)


def calculate_days_in_ar_quarterly(
    ending_ar: float,
    total_charges: float,
    year: int,
    quarter: int
) -> int:
    """
    Calculate Days in AR for Quarterly Report (Slide 4).

    Period Days Logic:
    - Complete quarter: Actual days in that quarter (89, 90, or 91)
    - Current quarter: Days from quarter start to today

    Formula: Ending AR / (Total Charges / Period Days)

    Args:
        ending_ar: Outstanding balance
        total_charges: Total charges for the quarter
        year: Year (e.g., 2024)
        quarter: Quarter number (1, 2, 3, or 4)

    Returns:
        Days in AR as integer
    """
    period_days = _get_period_days_quarterly(year, quarter)

    if total_charges == 0 or period_days == 0:
        return 0

    avg_daily_charges = total_charges / period_days
    if avg_daily_charges == 0:
        return 0

    return math.ceil(ending_ar / avg_daily_charges)


def calculate_days_in_ar_yearly(
    ending_ar: float,
    total_charges: float,
    year: int
) -> int:
    """
    Calculate Days in AR for Yearly Report (Slide 5).

    Period Days Logic:
    - Complete year: 365 days (fixed, no leap year adjustment)
    - Current year: Days from Jan 1 to today

    Formula: Ending AR / (Total Charges / Period Days)

    Args:
        ending_ar: Outstanding balance
        total_charges: Total charges for the year
        year: Year (e.g., 2024)

    Returns:
        Days in AR as integer
    """
    period_days = _get_period_days_yearly(year)

    if total_charges == 0 or period_days == 0:
        return 0

    avg_daily_charges = total_charges / period_days
    if avg_daily_charges == 0:
        return 0

    return math.ceil(ending_ar / avg_daily_charges)


# ============================================================================
# HELPER FUNCTIONS FOR DAYS IN AR CALCULATIONS
# ============================================================================

def _get_period_days_weekly(df: pd.DataFrame, date_col: str) -> int:
    """
    Get period days for Weekly KPI (Slide 3).

    Logic:
    - If data spans full year(s): 365 days (fixed)
    - If data is current year only: Jan 1 to today

    Args:
        df: DataFrame with date column
        date_col: Name of date column

    Returns:
        Number of days in period
    """
    if df.empty or date_col not in df.columns:
        return 365  # Default to full year

    try:
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if len(dates) == 0:
            return 365

        min_date = dates.min()
        max_date = dates.max()
        today = pd.Timestamp.now().normalize()

        # Get years in data
        min_year = min_date.year
        max_year = max_date.year
        current_year = today.year

        # If data is only from current year
        if min_year == current_year and max_year == current_year:
            # Year-to-date: Jan 1 to today
            year_start = pd.Timestamp(year=current_year, month=1, day=1)
            period_days = (today - year_start).days + 1
            logger.info(f"Weekly KPI DAR - Current year only: {period_days} days (YTD)")
            return period_days

        # Data spans multiple years or complete past years - use 365 days
        period_days = 365
        logger.info(f"Weekly KPI DAR - Full year(s) in data: {period_days} days")
        return period_days

    except Exception as e:
        logger.warning(f"Error calculating weekly period days: {e}")
        return 365


def _get_period_days_quarterly(year: int, quarter: int) -> int:
    """
    Get period days for Quarterly Report (Slide 4).

    Logic:
    - Complete quarter: Actual days in quarter (89, 90, or 91)
    - Current quarter: Quarter start to today

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1, 2, 3, or 4)

    Returns:
        Number of days in quarter
    """
    import calendar

    # Define quarter start months
    quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}  # Jan, Apr, Jul, Oct

    start_month = quarter_starts[quarter]

    # Calculate quarter end
    if quarter == 4:
        end_month = 12
        end_day = 31
    else:
        end_month = quarter_starts[quarter + 1] - 1
        # Get last day of end month
        end_day = calendar.monthrange(year, end_month)[1]

    quarter_start = pd.Timestamp(year=year, month=start_month, day=1)
    quarter_end = pd.Timestamp(year=year, month=end_month, day=end_day)

    today = pd.Timestamp.now().normalize()

    # Check if this is the current quarter
    current_year = today.year
    current_month = today.month
    current_quarter = (current_month - 1) // 3 + 1

    if year == current_year and quarter == current_quarter:
        # Current quarter: quarter start to today
        period_days = (today - quarter_start).days + 1
        logger.info(f"Quarterly DAR - Current quarter Q{quarter} {year}: {period_days} days (QTD)")
    else:
        # Complete quarter: full quarter days
        period_days = (quarter_end - quarter_start).days + 1
        logger.info(f"Quarterly DAR - Complete quarter Q{quarter} {year}: {period_days} days")

    return period_days


def _get_period_days_yearly(year: int) -> int:
    """
    Get period days for Yearly Report (Slide 5).

    Logic:
    - Complete year: 365 days (fixed, no leap year adjustment)
    - Current year: Jan 1 to today

    Args:
        year: Year (e.g., 2024)

    Returns:
        Number of days in year
    """
    today = pd.Timestamp.now().normalize()
    current_year = today.year

    if year == current_year:
        # Current year: Jan 1 to today
        year_start = pd.Timestamp(year=year, month=1, day=1)
        period_days = (today - year_start).days + 1
        logger.info(f"Yearly DAR - Current year {year}: {period_days} days (YTD)")
    else:
        # Complete year: always 365 days
        period_days = 365
        logger.info(f"Yearly DAR - Complete year {year}: {period_days} days")

    return period_days


def calculate_ar_aging(
    df: pd.DataFrame,
    date_col: str,
    balance_col: str,
    age_ranges: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Calculate AR aging for specified age ranges.

    Args:
        df: DataFrame with balance and date data
        date_col: Name of the date column
        balance_col: Name of the balance column
        age_ranges: List of tuples (min_days, max_days) for aging buckets

    Returns:
        Dictionary with aging bucket balances
    """
    if date_col not in df.columns or balance_col not in df.columns:
        return {f"{min_days}-{max_days}": 0.0 for min_days, max_days in age_ranges}

    df_copy = df.copy()
    today = datetime.now()
    df_copy['age_days'] = (today - pd.to_datetime(df_copy[date_col])).dt.days

    aging = {}
    for min_days, max_days in age_ranges:
        mask = (df_copy['age_days'] >= min_days) & (df_copy['age_days'] <= max_days)
        aging[f"{min_days}-{max_days}"] = float(df_copy[mask][balance_col].sum())

    return aging


def calculate_billed_ar(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    billed_statuses: List[str]
) -> float:
    """
    Calculate billed AR (claims submitted to payer).

    Args:
        df: DataFrame with balance and status data
        balance_col: Name of the balance column
        status_col: Name of the status column
        billed_statuses: List of statuses considered "billed"

    Returns:
        Sum of billed AR
    """
    if balance_col not in df.columns or status_col not in df.columns:
        return 0.0

    df_copy = df.copy()
    df_copy[status_col] = df_copy[status_col].str.lower()
    billed_statuses_lower = [s.lower() for s in billed_statuses]

    mask = df_copy[status_col].isin(billed_statuses_lower)
    return float(df_copy[mask][balance_col].sum())


def calculate_unbilled_ar(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    unbilled_statuses: List[str]
) -> float:
    """
    Calculate unbilled AR (claims not yet submitted).

    Args:
        df: DataFrame with balance and status data
        balance_col: Name of the balance column
        status_col: Name of the status column
        unbilled_statuses: List of statuses considered "unbilled"

    Returns:
        Sum of unbilled AR
    """
    if balance_col not in df.columns or status_col not in df.columns:
        return 0.0

    df_copy = df.copy()
    df_copy[status_col] = df_copy[status_col].str.lower()
    unbilled_statuses_lower = [s.lower() for s in unbilled_statuses]

    mask = df_copy[status_col].isin(unbilled_statuses_lower)
    return float(df_copy[mask][balance_col].sum())


def calculate_billed_ar_quarterly(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    date_col: str,
    year: int,
    quarter: int
) -> float:
    """
    Calculate Billed AR for Quarterly Report (Slide 4).

    Billed AR = SUM(balance) WHERE visit_status = 'Claim Created'
    Date Range: Quarter start to today (Quarter-to-date)

    Args:
        df: DataFrame with balance, status, and date data
        balance_col: Name of the balance column
        status_col: Name of the status column (visit_status)
        date_col: Name of the date column (visit_date)
        year: Year (e.g., 2024)
        quarter: Quarter number (1, 2, 3, or 4)

    Returns:
        Sum of billed AR for the quarter-to-date
    """
    if balance_col not in df.columns or status_col not in df.columns or date_col not in df.columns:
        return 0.0

    try:
        df_copy = df.copy()

        # Convert date column
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # Calculate quarter date range
        quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}
        start_month = quarter_starts[quarter]
        quarter_start = pd.Timestamp(year=year, month=start_month, day=1)
        today = pd.Timestamp.now().normalize()

        # Filter by status: 'Claim Created' only
        billed_status = 'claim created'
        status_mask = df_copy[status_col].astype(str).str.lower().str.strip() == billed_status

        # Filter by date: quarter start to today
        date_mask = (df_copy[date_col] >= quarter_start) & (df_copy[date_col] <= today)

        # Combined filter
        combined_mask = status_mask & date_mask

        billed_ar = float(df_copy.loc[combined_mask, balance_col].sum())
        logger.info(f"Quarterly Billed AR Q{quarter} {year}: ${billed_ar:,.2f} (status='Claim Created', {quarter_start.date()} to {today.date()})")

        return billed_ar
    except Exception as e:
        logger.error(f"Error calculating quarterly billed AR: {e}")
        return 0.0


def calculate_unbilled_ar_quarterly(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    date_col: str,
    year: int,
    quarter: int
) -> float:
    """
    Calculate Unbilled AR for Quarterly Report (Slide 4).

    Unbilled AR = SUM(balance) WHERE visit_status != 'Claim Created'
    (All other statuses besides 'Claim Created')
    Date Range: Quarter start to today (Quarter-to-date)

    Args:
        df: DataFrame with balance, status, and date data
        balance_col: Name of the balance column
        status_col: Name of the status column (visit_status)
        date_col: Name of the date column (visit_date)
        year: Year (e.g., 2024)
        quarter: Quarter number (1, 2, 3, or 4)

    Returns:
        Sum of unbilled AR for the quarter-to-date
    """
    if balance_col not in df.columns or status_col not in df.columns or date_col not in df.columns:
        return 0.0

    try:
        df_copy = df.copy()

        # Convert date column
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # Calculate quarter date range
        quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}
        start_month = quarter_starts[quarter]
        quarter_start = pd.Timestamp(year=year, month=start_month, day=1)
        today = pd.Timestamp.now().normalize()

        # Filter by status: NOT 'Claim Created' (all other statuses)
        billed_status = 'claim created'
        status_mask = ~(df_copy[status_col].astype(str).str.lower().str.strip() == billed_status)

        # Filter by date: quarter start to today
        date_mask = (df_copy[date_col] >= quarter_start) & (df_copy[date_col] <= today)

        # Combined filter
        combined_mask = status_mask & date_mask

        unbilled_ar = float(df_copy.loc[combined_mask, balance_col].sum())
        logger.info(f"Quarterly Unbilled AR Q{quarter} {year}: ${unbilled_ar:,.2f} (status!='Claim Created', {quarter_start.date()} to {today.date()})")

        return unbilled_ar
    except Exception as e:
        logger.error(f"Error calculating quarterly unbilled AR: {e}")
        return 0.0


def calculate_billed_ar_yearly(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    date_col: str,
    year: int
) -> float:
    """
    Calculate Billed AR for Yearly Report (Slide 5).

    Billed AR = SUM(balance) WHERE visit_status = 'Claim Created'
    Date Range: Year start to today (Year-to-date)

    Args:
        df: DataFrame with balance, status, and date data
        balance_col: Name of the balance column
        status_col: Name of the status column (visit_status)
        date_col: Name of the date column (visit_date)
        year: Year (e.g., 2024)

    Returns:
        Sum of billed AR for the year-to-date
    """
    if balance_col not in df.columns or status_col not in df.columns or date_col not in df.columns:
        return 0.0

    try:
        df_copy = df.copy()

        # Convert date column
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # Calculate year date range
        year_start = pd.Timestamp(year=year, month=1, day=1)
        today = pd.Timestamp.now().normalize()
        current_year = today.year

        # Determine year end date
        if year < current_year:
            # Complete/past year: use Dec 31 of that year
            year_end = pd.Timestamp(year=year, month=12, day=31)
        else:
            # Current year: use today (Year-to-date)
            year_end = today

        # Filter by status: 'Claim Created' only
        billed_status = 'claim created'
        status_mask = df_copy[status_col].astype(str).str.lower().str.strip() == billed_status

        # Filter by date: year start to year end (Dec 31 for past years, today for current year)
        date_mask = (df_copy[date_col] >= year_start) & (df_copy[date_col] <= year_end)

        # Combined filter
        combined_mask = status_mask & date_mask

        billed_ar = float(df_copy.loc[combined_mask, balance_col].sum())
        logger.info(f"Yearly Billed AR {year}: ${billed_ar:,.2f} (status='Claim Created', {year_start.date()} to {year_end.date()})")

        return billed_ar
    except Exception as e:
        logger.error(f"Error calculating yearly billed AR: {e}")
        return 0.0


def calculate_unbilled_ar_yearly(
    df: pd.DataFrame,
    balance_col: str,
    status_col: str,
    date_col: str,
    year: int
) -> float:
    """
    Calculate Unbilled AR for Yearly Report (Slide 5).

    Unbilled AR = SUM(balance) WHERE visit_status != 'Claim Created'
    (All other statuses besides 'Claim Created')
    Date Range: Year start to today (Year-to-date)

    Args:
        df: DataFrame with balance, status, and date data
        balance_col: Name of the balance column
        status_col: Name of the status column (visit_status)
        date_col: Name of the date column (visit_date)
        year: Year (e.g., 2024)

    Returns:
        Sum of unbilled AR for the year-to-date
    """
    if balance_col not in df.columns or status_col not in df.columns or date_col not in df.columns:
        return 0.0

    try:
        df_copy = df.copy()

        # Convert date column
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # Calculate year date range
        year_start = pd.Timestamp(year=year, month=1, day=1)
        today = pd.Timestamp.now().normalize()
        current_year = today.year

        # Determine year end date
        if year < current_year:
            # Complete/past year: use Dec 31 of that year
            year_end = pd.Timestamp(year=year, month=12, day=31)
        else:
            # Current year: use today (Year-to-date)
            year_end = today

        # Filter by status: NOT 'Claim Created' (all other statuses)
        billed_status = 'claim created'
        status_mask = ~(df_copy[status_col].astype(str).str.lower().str.strip() == billed_status)

        # Filter by date: year start to year end (Dec 31 for past years, today for current year)
        date_mask = (df_copy[date_col] >= year_start) & (df_copy[date_col] <= year_end)

        # Combined filter
        combined_mask = status_mask & date_mask

        unbilled_ar = float(df_copy.loc[combined_mask, balance_col].sum())
        logger.info(f"Yearly Unbilled AR {year}: ${unbilled_ar:,.2f} (status!='Claim Created', {year_start.date()} to {year_end.date()})")

        return unbilled_ar
    except Exception as e:
        logger.error(f"Error calculating yearly unbilled AR: {e}")
        return 0.0


# ============================================================================
# CHARGES SUBMITTED PERCENTAGE
# ============================================================================

def calculate_charges_submitted_percentage(
    df: pd.DataFrame,
    charge_col: str,
    status_col: str,
    submitted_statuses: List[str]
) -> float:
    """
    Calculate percentage of charges that have been submitted.

    Formula: (Claimed Charges / Total Charges) × 100

    Args:
        df: DataFrame with charge and status data
        charge_col: Name of the charge column
        status_col: Name of the status column
        submitted_statuses: List of statuses considered "submitted"

    Returns:
        Percentage of charges submitted
    """
    if charge_col not in df.columns or status_col not in df.columns:
        return 0.0

    # Total charges
    total_charges = calculate_total_charges(df, charge_col)

    if total_charges == 0:
        return 0.0

    # Claimed charges
    df_copy = df.copy()
    df_copy[status_col] = df_copy[status_col].str.lower()
    submitted_statuses_lower = [s.lower() for s in submitted_statuses]

    claimed_mask = (df_copy[charge_col] > 0) & (df_copy[status_col].isin(submitted_statuses_lower))
    claimed_charges = float(df_copy[claimed_mask][charge_col].sum())

    return (claimed_charges / total_charges) * 100


# ============================================================================
# DENIAL METRICS
# ============================================================================

def calculate_denial_resolution_rate() -> float:
    """
    Calculate denial resolution rate.

    Currently returns a static value.

    Returns:
        Denial resolution rate as percentage
    """
    return 85.0


# ============================================================================
# FORECASTING
# ============================================================================

def calculate_forecast(
    historical_df: pd.DataFrame,
    metric_col: str,
    date_col: str,
    forecast_days: int
) -> float:
    """
    Calculate forecast based on historical average.

    Args:
        historical_df: DataFrame with historical data
        metric_col: Column to forecast (e.g., 'visits', 'collections')
        date_col: Date column for calculating period
        forecast_days: Number of days to forecast for

    Returns:
        Forecasted value
    """
    if historical_df.empty or metric_col not in historical_df.columns:
        return 0.0

    # Calculate period days
    if date_col in historical_df.columns:
        min_date = historical_df[date_col].min()
        max_date = historical_df[date_col].max()
        period_days = (max_date - min_date).days + 1
    else:
        period_days = len(historical_df)

    if period_days == 0:
        return 0.0

    # Calculate total and daily average
    if metric_col == 'visits':
        total = len(historical_df)
    else:
        # Exclude negative values for monetary columns
        total = float(historical_df[historical_df[metric_col] > 0][metric_col].sum())

    daily_average = total / period_days

    return daily_average * forecast_days


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_date_range(df: pd.DataFrame, date_col: str) -> Tuple[datetime, datetime, int]:
    """
    Get date range information from DataFrame.

    Args:
        df: DataFrame with date column
        date_col: Name of the date column

    Returns:
        Tuple of (min_date, max_date, period_days)
    """
    if date_col not in df.columns or df.empty:
        return None, None, 0

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    period_days = (max_date - min_date).days + 1

    return min_date, max_date, period_days


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default

    return numerator / denominator
