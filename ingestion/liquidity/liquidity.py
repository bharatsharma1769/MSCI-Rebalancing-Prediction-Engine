import numpy as np
import pandas as pd
from pathlib import Path


MARKET_PATH = "data/processed/market_data_fif.parquet"
MASTER_PATH = "data/processed/historical_security_master.parquet"

OUTPUT_PATH = "data/processed/liquidity_monthly.parquet"


# --------------------------------------------------
# Load
# --------------------------------------------------

market = pd.read_parquet(
    MARKET_PATH,
    columns=[
        "security_id",
        "date",
        "close",
        "volume",
        "shares_outstanding",
        "fif",
        "free_float_market_cap_inr",
    ]
)

master = pd.read_parquet(
    MASTER_PATH,
    columns=[
        "security_id",
        "nse_symbol",
        "listing_date",
        "delisted_date",
    ]
)


market["date"] = pd.to_datetime(
    market["date"]
)

master["listing_date"] = pd.to_datetime(
    master["listing_date"],
    errors="coerce"
)

master["delisted_date"] = pd.to_datetime(
    master["delisted_date"],
    errors="coerce"
)

# --------------------------------------------------
# Liquidity-only historical denominator proxy
# --------------------------------------------------
# ATVR requires historical FF market cap.
#
# For established securities where our NSE ownership
# history starts after prices begin, use the earliest
# observed shares/FIF prior to that first observation.
#
# This is ONLY for liquidity calculation and does not
# modify the PIT market-cap dataset used for index
# construction.
# --------------------------------------------------

market = market.merge(
    master[
        [
            "security_id",
            "listing_date",
        ]
    ],
    on="security_id",
    how="left"
)

market["listing_date"] = pd.to_datetime(
    market["listing_date"],
    errors="coerce"
)

market = market.sort_values(
    [
        "security_id",
        "date"
    ]
)


# Earliest available shares
first_shares = (
    market[
        market["shares_outstanding"].notna()
    ]
    .sort_values("date")
    .groupby("security_id")
    .first()[
        "shares_outstanding"
    ]
)


# Earliest available FIF
first_fif = (
    market[
        market["fif"].notna()
    ]
    .sort_values("date")
    .groupby("security_id")
    .first()[
        "fif"
    ]
)


market["liquidity_shares"] = (
    market["shares_outstanding"]
)

market["liquidity_fif"] = (
    market["fif"]
)


market["shares_proxy"] = False
market["fif_proxy"] = False


# Only fill dates when the security was already listed.
listed = (
    market["listing_date"].isna()
    |
    (
        market["date"]
        >= market["listing_date"]
    )
)


missing_shares = (
    listed
    &
    market["liquidity_shares"].isna()
)


market.loc[
    missing_shares,
    "liquidity_shares"
] = (
    market.loc[
        missing_shares,
        "security_id"
    ].map(first_shares)
)


market.loc[
    missing_shares
    &
    market["liquidity_shares"].notna(),
    "shares_proxy"
] = True


missing_fif = (
    listed
    &
    market["liquidity_fif"].isna()
)


market.loc[
    missing_fif,
    "liquidity_fif"
] = (
    market.loc[
        missing_fif,
        "security_id"
    ].map(first_fif)
)


market.loc[
    missing_fif
    &
    market["liquidity_fif"].notna(),
    "fif_proxy"
] = True


market["liquidity_ff_market_cap_inr"] = (
    market["close"]
    * market["liquidity_shares"]
    * market["liquidity_fif"]
)


print("\n--- LIQUIDITY DENOMINATOR PROXY ---")

print(
    "Rows using shares proxy:",
    market["shares_proxy"].sum()
)

print(
    "Rows using FIF proxy:",
    market["fif_proxy"].sum()
)

print(
    "Rows with liquidity FF mcap:",
    market[
        "liquidity_ff_market_cap_inr"
    ].notna().sum()
)

# --------------------------------------------------
# Exchange trading calendar
# --------------------------------------------------
# Use dates on which a meaningful number of NSE
# securities have valid prices.
#
# This removes any isolated/non-market dates that
# may appear because of source artifacts.
# --------------------------------------------------

daily_market_count = (
    market[
        market["close"].notna()
    ]
    .groupby("date")
    ["security_id"]
    .nunique()
)

exchange_dates = (
    daily_market_count[
        daily_market_count >= 100
    ]
    .index
    .sort_values()
)

exchange_calendar = pd.DataFrame({
    "date": exchange_dates
})

exchange_calendar["month"] = (
    exchange_calendar["date"]
    .dt.to_period("M")
)

exchange_days_month = (
    exchange_calendar
    .groupby("month")
    .size()
    .rename("exchange_days")
)

liquidity_month_end = (
    exchange_calendar
    .groupby("month")["date"]
    .max()
)


print(
    "Exchange dates:",
    len(exchange_dates)
)

print(
    "Calendar:",
    exchange_dates.min().date(),
    "to",
    exchange_dates.max().date()
)


# --------------------------------------------------
# Restrict daily data to NSE trading dates
# --------------------------------------------------

market = market[
    market["date"].isin(
        exchange_dates
    )
].copy()

market["month"] = (
    market["date"]
    .dt.to_period("M")
)


# --------------------------------------------------
# First observed market date
# --------------------------------------------------

first_observed = (
    market[
        market["close"].notna()
    ]
    .groupby("security_id")
    ["date"]
    .min()
    .rename("first_observed_date")
)

master = master.merge(
    first_observed,
    on="security_id",
    how="left"
)


# If listing date is absent, use first observed date.
# If listing date predates our data, that is fine.

master["effective_start"] = (
    master["listing_date"]
    .fillna(
        master["first_observed_date"]
    )
)

master["effective_end"] = (
    master["delisted_date"]
    .fillna(
        exchange_dates.max()
    )
)


# --------------------------------------------------
# Add listing information
# --------------------------------------------------

market = market.merge(
    master[
        [
            "security_id",
            "effective_start",
            "effective_end",
        ]
    ],
    on="security_id",
    how="left"
)


# Security must actually be listed / available

market["available"] = (
    (market["date"] >= market["effective_start"])
    &
    (market["date"] <= market["effective_end"])
)


# --------------------------------------------------
# Daily traded value
# --------------------------------------------------

market["traded"] = (
    market["available"]
    &
    market["close"].notna()
    &
    market["volume"].notna()
    &
    (market["volume"] > 0)
)

market["potential_trade_day"] = (
    market["available"]
    &
    market["volume"].notna()
    &
    (market["volume"] == 0)
)


market["daily_traded_value"] = np.where(
    market["traded"],
    market["close"]
    * market["volume"],
    np.nan
)


# --------------------------------------------------
# Identify IPO month + first three traded days
# --------------------------------------------------

market["listing_month"] = (
    market["listing_date"]
    .dt.to_period("M")
)

market["is_ipo_month"] = (
    market["month"]
    == market["listing_month"]
)


market = market.sort_values(
    [
        "security_id",
        "date"
    ]
)


trade_rank = (
    market[
        market["traded"]
    ]
    .groupby("security_id")
    .cumcount()
    + 1
)

market["trade_rank"] = np.nan

market.loc[
    market["traded"],
    "trade_rank"
] = trade_rank.values


market["exclude_ipo_trade"] = (
    market["is_ipo_month"]
    &
    market["traded"]
    &
    (market["trade_rank"] <= 3)
)


# ATVR median excludes first 3 IPO trading days

market["atvr_daily_value"] = (
    market["daily_traded_value"]
)

market.loc[
    market["exclude_ipo_trade"],
    "atvr_daily_value"
] = np.nan


# --------------------------------------------------
# Monthly raw statistics
# --------------------------------------------------

group = market.groupby(
    [
        "security_id",
        "month"
    ],
    sort=False
)


monthly = group.agg(

    traded_days=(
        "traded",
        "sum"
    ),

    potential_days=(
        "potential_trade_day",
        "sum"
    ),

    median_daily_traded_value=(
        "atvr_daily_value",
        "median"
    ),

    first_date=(
        "date",
        "min"
    ),

    last_date=(
        "date",
        "max"
    ),

).reset_index()


# --------------------------------------------------
# Month-end free-float market cap
# --------------------------------------------------

mcap = (
    market[
        market[
            "liquidity_ff_market_cap_inr"
        ].notna()
    ]
    .sort_values(
        [
            "security_id",
            "date"
        ]
    )
    .groupby(
        [
            "security_id",
            "month"
        ]
    )
    .tail(1)
    [
        [
            "security_id",
            "month",
            "date",
            "liquidity_ff_market_cap_inr",
        ]
    ]
    .rename(
        columns={
            "date": "mcap_date",
            "liquidity_ff_market_cap_inr":
                "free_float_market_cap_inr",
        }
    )
)

monthly = monthly.merge(
    mcap,
    on=[
        "security_id",
        "month"
    ],
    how="left"
)


# --------------------------------------------------
# Available exchange days per security/month
# --------------------------------------------------

availability_rows = []

for row in master.itertuples():

    if pd.isna(
        row.effective_start
    ):
        continue

    start = max(
        row.effective_start,
        exchange_dates.min()
    )

    end = min(
        row.effective_end,
        exchange_dates.max()
    )

    if start > end:
        continue

    dates = exchange_dates[
        (exchange_dates >= start)
        &
        (exchange_dates <= end)
    ]

    if len(dates) == 0:
        continue

    temp = pd.DataFrame({
        "date": dates
    })

    temp["month"] = (
        temp["date"]
        .dt.to_period("M")
    )

    counts = (
        temp
        .groupby("month")
        .size()
    )

    for month, count in counts.items():

        availability_rows.append({
            "security_id":
                row.security_id,

            "month":
                month,

            "possible_days":
                int(count),
        })


availability = pd.DataFrame(
    availability_rows
)


monthly = monthly.merge(
    availability,
    on=[
        "security_id",
        "month"
    ],
    how="left"
)


# --------------------------------------------------
# IPO pre-listing days
# --------------------------------------------------

listing_lookup = (
    master
    .set_index("security_id")
    ["listing_date"]
)

monthly["listing_date"] = (
    monthly["security_id"]
    .map(listing_lookup)
)

monthly["listing_month"] = (
    monthly["listing_date"]
    .dt.to_period("M")
)

monthly["is_ipo_month"] = (
    monthly["month"]
    == monthly["listing_month"]
)


def count_pre_ipo_days(row):

    if (
        not row["is_ipo_month"]
        or pd.isna(
            row["listing_date"]
        )
    ):
        return 0

    month_dates = exchange_calendar[
        exchange_calendar["month"]
        == row["month"]
    ]["date"]

    return int(
        (
            month_dates
            < row["listing_date"]
        ).sum()
    )


monthly["pre_ipo_days"] = (
    monthly.apply(
        count_pre_ipo_days,
        axis=1
    )
)


# --------------------------------------------------
# Minimum days requirement
# --------------------------------------------------
# MSCI:
#
# trade days + potential days >= 5
#
# In IPO month first three trading days are removed
# for this test.
# --------------------------------------------------

monthly["usable_atvr_days"] = (
    monthly["traded_days"]
    + monthly["potential_days"]
)

monthly.loc[
    monthly["is_ipo_month"],
    "usable_atvr_days"
] = (
    monthly.loc[
        monthly["is_ipo_month"],
        "usable_atvr_days"
    ]
    - 3
)


monthly["monthly_atvr_valid"] = (
    monthly["usable_atvr_days"]
    >= 5
)


# --------------------------------------------------
# Monthly Median Traded Value
# --------------------------------------------------
#
# Normal:
# median daily traded value × traded days
#
# IPO month:
# MSCI extrapolates by adding pre-IPO exchange
# days to the actual number of traded days.
# --------------------------------------------------

monthly["atvr_day_multiplier"] = (
    monthly["traded_days"]
    .astype(float)
)

monthly.loc[
    monthly["is_ipo_month"],
    "atvr_day_multiplier"
] = (
    monthly.loc[
        monthly["is_ipo_month"],
        "traded_days"
    ]
    +
    monthly.loc[
        monthly["is_ipo_month"],
        "pre_ipo_days"
    ]
)


monthly["monthly_median_traded_value"] = (
    monthly["median_daily_traded_value"]
    * monthly["atvr_day_multiplier"]
)


monthly.loc[
    ~monthly["monthly_atvr_valid"],
    "monthly_median_traded_value"
] = np.nan


# --------------------------------------------------
# Monthly traded value ratio / ATVR
# --------------------------------------------------

monthly["monthly_traded_value_ratio"] = (
    monthly["monthly_median_traded_value"]
    /
    monthly["free_float_market_cap_inr"]
)


monthly["atvr_1m"] = (
    monthly["monthly_traded_value_ratio"]
    * 12
)


# --------------------------------------------------
# 1-month FOT
# --------------------------------------------------

monthly["fot_1m"] = (
    monthly["traded_days"]
    /
    monthly["possible_days"]
)


# --------------------------------------------------
# Create complete monthly grid
# --------------------------------------------------
# Needed so rolling 3/12 month values cannot silently
# skip missing months.
# --------------------------------------------------

all_months = pd.period_range(
    monthly["month"].min(),
    monthly["month"].max(),
    freq="M"
)

security_ids = (
    monthly["security_id"]
    .unique()
)

grid = pd.MultiIndex.from_product(
    [
        security_ids,
        all_months
    ],
    names=[
        "security_id",
        "month"
    ]
).to_frame(
    index=False
)


monthly = grid.merge(
    monthly,
    on=[
        "security_id",
        "month"
    ],
    how="left"
)


monthly = monthly.sort_values(
    [
        "security_id",
        "month"
    ]
).reset_index(
    drop=True
)


# --------------------------------------------------
# FAST MSCI fallback liquidity windows
# --------------------------------------------------

g = monthly.groupby("security_id", group_keys=False)

# 12m ATVR candidate
atvr12 = (
    g["monthly_traded_value_ratio"]
    .rolling(12, min_periods=12)
    .mean()
    .reset_index(level=0, drop=True)
    * 12
)

# 6m fallback
atvr6 = (
    g["monthly_traded_value_ratio"]
    .rolling(6, min_periods=6)
    .mean()
    .reset_index(level=0, drop=True)
    * 12
)

# 3m fallback
atvr3_full = (
    g["monthly_traded_value_ratio"]
    .rolling(3, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
    * 12
)

# 1m fallback
atvr1 = (
    monthly["monthly_traded_value_ratio"]
    * 12
)

monthly["atvr_12m"] = (
    atvr12
    .fillna(atvr6)
    .fillna(atvr3_full)
    .fillna(atvr1)
)


# --------------------------------------------------
# 3m ATVR: 3m -> 1m fallback
# --------------------------------------------------

monthly["atvr_3m"] = (
    atvr3_full
    .fillna(atvr1)
)


# --------------------------------------------------
# FAST 3m FOT: 3m -> 1m fallback
# --------------------------------------------------

traded_3m = (
    g["traded_days"]
    .rolling(3, min_periods=3)
    .sum()
    .reset_index(level=0, drop=True)
)

possible_3m = (
    g["possible_days"]
    .rolling(3, min_periods=3)
    .sum()
    .reset_index(level=0, drop=True)
)

fot_3m_full = (
    traded_3m
    /
    possible_3m
)

fot_1m = (
    monthly["traded_days"]
    /
    monthly["possible_days"]
)

monthly["fot_3m"] = (
    fot_3m_full
    .fillna(fot_1m)
)

# --------------------------------------------------
# Attach symbols
# --------------------------------------------------

monthly = monthly.merge(
    master[
        [
            "security_id",
            "nse_symbol"
        ]
    ],
    on="security_id",
    how="left"
)


# --------------------------------------------------
# Basic sanity
# --------------------------------------------------

print("\n--- BASIC ---")

print(
    "Rows:",
    len(monthly)
)

print(
    "Securities:",
    monthly[
        "security_id"
    ].nunique()
)

print(
    "Months:",
    monthly[
        "month"
    ].min(),
    "to",
    monthly[
        "month"
    ].max()
)

print(
    "Invalid FOT:",
    (
        (monthly["fot_3m"] < 0)
        |
        (monthly["fot_3m"] > 1)
    ).sum()
)

print(
    "Negative ATVR:",
    (
        monthly["atvr_3m"] < 0
    ).sum()
)


# --------------------------------------------------
# Review liquidity cutoffs
# --------------------------------------------------

review_cutoffs = {
    "2022-02": "2021-12",
    "2022-05": "2022-03",
    "2022-08": "2022-06",
    "2022-11": "2022-09",

    "2023-02": "2022-12",
    "2023-05": "2023-03",
    "2023-08": "2023-06",
    "2023-11": "2023-09",

    "2024-02": "2023-12",
    "2024-05": "2024-03",
    "2024-08": "2024-06",
    "2024-11": "2024-09",

    "2025-02": "2024-12",
    "2025-05": "2025-03",
    "2025-08": "2025-06",
    "2025-11": "2025-09",

    "2026-02": "2025-12",
    "2026-05": "2026-03",
}


print(
    "\n--- REVIEW LIQUIDITY COVERAGE ---"
)

for review, cutoff in review_cutoffs.items():

    cutoff_period = pd.Period(
        cutoff,
        freq="M"
    )

    cutoff_date = liquidity_month_end.loc[cutoff_period]

    snap = monthly[
        monthly["month"]
        == cutoff_period
    ]

    valid = snap[
        snap["atvr_12m"].notna()
        &
        snap["atvr_3m"].notna()
        &
        snap["fot_3m"].notna()
    ]

    print(
        review,
        "cutoff",
        cutoff_date.date(),
        f"{len(valid)}/{len(snap)}",
        f"{len(valid) / len(snap):.1%}"
        if len(snap)
        else ""
    )


# --------------------------------------------------
# Sample major names
# --------------------------------------------------

print(
    "\n--- SAMPLE ---"
)

sample_symbols = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "MCX",
]

sample = monthly[
    monthly["nse_symbol"].isin(
        sample_symbols
    )
    &
    monthly["month"].isin(
        [
            pd.Period(
                "2025-03",
                freq="M"
            ),
            pd.Period(
                "2026-03",
                freq="M"
            ),
        ]
    )
][
    [
        "nse_symbol",
        "month",
        "atvr_3m",
        "atvr_12m",
        "fot_3m",
       
    ]
]

print(
    sample.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    OUTPUT_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)

# Period dtype can be awkward across some parquet
# tooling, so save month as timestamp.

monthly["month_end"] = monthly["month"].map(liquidity_month_end)

if monthly["month_end"].isna().any():
    raise RuntimeError(
        f"Missing liquidity month-end dates: {int(monthly['month_end'].isna().sum())}"
    )

monthly = monthly.drop(
    columns=[
        "month",
        "listing_month"
    ]
)


monthly.to_parquet(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_PATH
)