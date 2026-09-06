import pandas as pd
from pathlib import Path


MARKET_PATH = "data/processed/market_data.parquet"
FIF_PATH = "data/raw/fif/fif.parquet"

OUTPUT_PATH = "data/processed/market_data_fif.parquet"


market = pd.read_parquet(
    MARKET_PATH
)

fif = pd.read_parquet(
    FIF_PATH
)


# -----------------------------------------
# Dates
# -----------------------------------------

market["date"] = pd.to_datetime(
    market["date"]
).astype("datetime64[ns]")

fif["available_date"] = pd.to_datetime(
    fif["available_date"]
).astype("datetime64[ns]")

fif["report_date"] = pd.to_datetime(
    fif["report_date"]
).astype("datetime64[ns]")


# -----------------------------------------
# Keep FIF fields needed downstream
# -----------------------------------------

fif = fif[
    [
        "security_id",
        "report_date",
        "available_date",
        "raw_free_float",
        "fif",
        "source",
    ]
].copy()

fif = fif.rename(
    columns={
        "report_date": "fif_report_date",
        "available_date": "fif_available_date",
        "source": "fif_source",
    }
)


# -----------------------------------------
# Point-in-time merge
# -----------------------------------------

market = market.sort_values(
    [
        "date",
        "security_id",
    ]
)

fif = fif.sort_values(
    [
        "fif_available_date",
        "security_id",
    ]
)

market["security_id"] = (
    market["security_id"]
    .astype("string")
)

fif["security_id"] = (
    fif["security_id"]
    .astype("string")
)

out = pd.merge_asof(
    market,
    fif,
    left_on="date",
    right_on="fif_available_date",
    by="security_id",
    direction="backward",
)


# -----------------------------------------
# Free-float market cap
# -----------------------------------------

out["free_float_market_cap_inr"] = (
    out["full_market_cap_inr"]
    * out["fif"]
)


# -----------------------------------------
# Sanity checks
# -----------------------------------------

print("\n--- BASIC ---")

print(
    "Rows:",
    len(out)
)

print(
    "Securities:",
    out["security_id"].nunique()
)

print(
    "Duplicate date/security:",
    out.duplicated(
        ["date", "security_id"]
    ).sum()
)

print(
    "Rows with FIF:",
    out["fif"].notna().sum()
)

print(
    "Rows without FIF:",
    out["fif"].isna().sum()
)

print(
    "Invalid FIF:",
    (
        out["fif"].notna()
        & ~out["fif"].between(0, 1)
    ).sum()
)

print(
    "Future FIF used:",
    (
        out["fif_available_date"].notna()
        & (
            out["fif_available_date"]
            > out["date"]
        )
    ).sum()
)

print(
    "FF market cap > full market cap:",
    (
        out["free_float_market_cap_inr"]
        > out["full_market_cap_inr"] * 1.000001
    ).sum()
)


# -----------------------------------------
# Review-date coverage
# -----------------------------------------

review_dates = pd.to_datetime([
    "2021-11-30",
    "2022-02-28",
    "2022-05-31",
    "2022-08-31",
    "2022-11-30",
    "2023-02-28",
    "2023-05-31",
    "2023-08-31",
    "2023-11-30",
    "2024-02-29",
    "2024-05-31",
    "2024-08-30",
    "2024-11-29",
    "2025-02-28",
    "2025-05-30",
    "2025-08-29",
    "2025-11-28",
    "2026-02-27",
    "2026-05-29",
])

print("\n--- REVIEW DATE FIF COVERAGE ---")

for date in review_dates:

    snapshot = out[
        out["date"] == date
    ]

    # Only names for which market cap exists
    eligible = snapshot[
        snapshot["full_market_cap_inr"].notna()
    ]

    if len(eligible) == 0:
        print(
            date.date(),
            "no market data"
        )
        continue

    covered = eligible[
        eligible["fif"].notna()
    ]

    print(
        date.date(),
        f"{len(covered)}/{len(eligible)}",
        f"{len(covered) / len(eligible):.1%}"
    )


# -----------------------------------------
# Largest names missing FIF
# -----------------------------------------

latest = out[
    out["date"] <= pd.Timestamp(
        "2026-05-29"
    )
].copy()

max_mcap = (
    latest
    .groupby("security_id")
    ["full_market_cap_inr"]
    .max()
)

has_fif = (
    latest
    .groupby("security_id")
    ["fif"]
    .apply(
        lambda x: x.notna().any()
    )
)

missing_ids = has_fif[
    ~has_fif
].index

missing = (
    max_mcap
    .loc[
        max_mcap.index.intersection(
            missing_ids
        )
    ]
    .sort_values(
        ascending=False
    )
    .head(30)
)

print("\n--- LARGEST SECURITIES WITH NO FIF ---")

print(
    missing.to_string()
)


# -----------------------------------------
# Save
# -----------------------------------------

Path(
    OUTPUT_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_PATH
)