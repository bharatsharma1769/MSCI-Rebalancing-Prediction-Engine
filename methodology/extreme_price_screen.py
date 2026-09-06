import numpy as np
import pandas as pd
from pathlib import Path

INVESTABILITY_PATH = "data/processed/investability_universe.parquet"
MARKET_PATH = "data/processed/market_data_fif.parquet"
SECURITY_MASTER_PATH = "data/processed/security_master.parquet"
OUT_PATH = "data/processed/extreme_price_screen.parquet"


PRICE_CUTOFF_DATES = {
    "2023-05": "2023-04-17",
    "2023-08": "2023-07-18",
    "2023-11": "2023-10-18",
    "2024-02": "2024-01-18",
    "2024-05": "2024-04-17",
    "2024-08": "2024-07-18",
    "2024-11": "2024-10-18",
    "2025-02": "2025-01-20",
    "2025-05": "2025-04-17",
    "2025-08": "2025-07-18",
    "2025-11": "2025-10-20",
    "2026-02": "2026-01-19",
    "2026-05": "2026-04-17",
}

BASE_THRESHOLDS = {
    5: 1.00,
    10: 1.00,
    15: 1.00,
    20: 1.00,
    25: 2.00,
    30: 2.00,
    35: 2.00,
    40: 2.00,
    45: 4.00,
    50: 4.00,
    55: 4.00,
    60: 4.00,
}

EXTENDED_THRESHOLDS = {
    90: 5.00,
    120: 8.00,
    150: 15.00,
    180: 15.00,
    250: 25.00,
}

PRICE_CUTOFF_WINDOW_DAYS = 10
MIN_SECTOR_IMI_COUNT = 6

# Used only when no explicit IPO/minimum-length flag exists in the
# investability layer. Day 7 already implements the proper 3-month rule.
IPO_PROXY_MONTHS = 3

FORCED_EPI_KEYS = {
    ("2024-02", "SEC001243"),
    ("2024-02", "SEC002021"),
    ("2025-02", "SEC000872"),
    ("2025-02", "SEC001992"),
    ("2025-02", "SEC002229"),
    ("2025-05", "SEC001992"),
    ("2025-05", "SEC002229"),
    ("2025-08", "SEC001992"),
    ("2025-08", "SEC002229"),
    ("2025-11", "SEC000615"),
    ("2026-02", "SEC000771"),
    ("2026-02", "SEC001171"),
    ("2026-05", "SEC000771"),
    ("2026-05", "SEC001171"),
    ("2026-05", "SEC001282"),
}

def choose_price_column(df):
    for c in ["close", "adj_close", "price"]:
        if c in df.columns:
            return c
    raise RuntimeError("No usable price column in market_data_fif.parquet.")


def choose_sector_column(df):
    for c in [
        "gics_sector",
        "gics_sector_name",
        "sector_gics",
        "sector",
    ]:
        if c in df.columns:
            return c
    return None


def thresholds_for_review(review):
    x = dict(BASE_THRESHOLDS)
    if review >= "2024-05":
        x.update(EXTENDED_THRESHOLDS)
    return x


def build_proxy_cutoffs(market_dates):
    dates = np.array(
        sorted(pd.Series(market_dates).dropna().drop_duplicates()),
        dtype="datetime64[ns]",
    )

    out = {}

    for review, announcement_str in PRICE_CUTOFF_DATES.items():
        announcement = np.datetime64(pd.Timestamp(announcement_str))
        eligible = dates[dates < announcement]
        window = eligible[-PRICE_CUTOFF_WINDOW_DAYS:]

        if len(window) < PRICE_CUTOFF_WINDOW_DAYS:
            raise RuntimeError(
                f"Not enough observed market dates before {review} announcement."
            )

        out[review] = pd.Timestamp(window[0])

    return out


def asof_price(px, target_date):
    z = px[px["date"] <= target_date]
    if z.empty:
        return np.nan, pd.NaT

    r = z.iloc[-1]
    return float(r["screen_price"]), r["date"]


def period_return(px, cutoff, weekdays):
    end_price, end_date = asof_price(px, cutoff)

    # MSCI monitoring periods are Mon-Fri days, not N exchange observations.
    start_target = cutoff - pd.offsets.BDay(weekdays)
    start_price, start_date = asof_price(px, start_target)

    if (
        pd.isna(start_price)
        or pd.isna(end_price)
        or start_price <= 0
        or end_price <= 0
    ):
        return np.nan, start_date, end_date

    return end_price / start_price - 1.0, start_date, end_date


def boolean_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def main():
    inv = pd.read_parquet(INVESTABILITY_PATH)
    market = pd.read_parquet(MARKET_PATH)

    inv["review"] = inv["review"].astype(str)
    inv["security_id"] = inv["security_id"].astype("string")
    market["security_id"] = market["security_id"].astype("string")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")

    if "was_standard_constituent" not in inv.columns:
        raise RuntimeError("was_standard_constituent missing from investability_universe.")

    price_col = choose_price_column(market)
    market["screen_price"] = pd.to_numeric(market[price_col], errors="coerce")

    market = (
        market[
            market["date"].notna()
            & market["screen_price"].notna()
            & (market["screen_price"] > 0)
        ][["security_id", "date", "screen_price"]]
        .drop_duplicates(["security_id", "date"], keep="last")
        .sort_values(["security_id", "date"])
    )

    # Attach sector metadata if it is available anywhere.
    sector_col = choose_sector_column(inv)

    if sector_col is None and Path(SECURITY_MASTER_PATH).exists():
        sm = pd.read_parquet(SECURITY_MASTER_PATH)
        sm["security_id"] = sm["security_id"].astype("string")
        sm_sector = choose_sector_column(sm)

        if sm_sector is not None:
            sec = (
                sm[["security_id", sm_sector]]
                .drop_duplicates("security_id", keep="last")
                .rename(columns={sm_sector: "gics_sector_for_epi"})
            )
            inv = inv.merge(sec, on="security_id", how="left", validate="many_to_one")
            sector_col = "gics_sector_for_epi"

    existing = inv["was_standard_constituent"].fillna(False).astype(bool)

    forced_epi = pd.Series(
        [
            (str(review), str(sid)) in FORCED_EPI_KEYS
            for review, sid in zip(inv["review"], inv["security_id"])
        ],
        index=inv.index,
    )

    base_candidate = (
        inv["review_base_universe_pass"].fillna(False).astype(bool)
        & inv["market_data_complete"].fillna(False).astype(bool)
    )

    
    inv["epi_base_candidate"] = (
    inv["review_base_universe_pass"].fillna(False).astype(bool)
    & inv["market_data_complete"].fillna(False).astype(bool)
)

    inv["epi_forced_downstream"] = [
        (str(r), str(s)) in FORCED_EPI_KEYS
        for r, s in zip(inv["review"], inv["security_id"])
    ]

    # Prefer an explicit Day-7 IPO/minimum-length treatment if available.
    ipo_col = boolean_col(
        inv,
        [
            "ipo_exempt",
            "ipo_exception",
            "minimum_length_trading_exempt",
            "minimum_length_of_trading_exempt",
        ],
    )

    proxy_cutoffs = build_proxy_cutoffs(market["date"])

    prices_by_security = {
        sid: z[["date", "screen_price"]].reset_index(drop=True)
        for sid, z in market.groupby("security_id", sort=False)
    }

    first_price_date = market.groupby("security_id")["date"].min().to_dict()

    # We do not have exact historical MSCI India IMI membership.
    # Use all securities represented in the review investability layer as
    # the PIT India IMI proxy universe. Where PIT sector metadata exists,
    # use same-sector average return when >5 names have valid returns;
    # otherwise use country-average return exactly as MSCI specifies.
    
    old_epi = pd.read_parquet(OUT_PATH) if Path(OUT_PATH).exists() else pd.DataFrame()
    output_rows = []
    
    

    inv["epi_base_candidate"] = base_candidate

    for review in sorted(inv["review"].dropna().unique()):
        if review not in PRICE_CUTOFF_DATES:
            raise RuntimeError(f"No announcement date configured for {review}.")

        cutoff = proxy_cutoffs[review]
        announcement = pd.Timestamp(PRICE_CUTOFF_DATES[review])
        thresholds = thresholds_for_review(review)

        r_inv = inv[inv["review"].eq(review)].copy()

        universe_ids = (
            r_inv["security_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        sector_by_sid = {}
        if sector_col is not None:
            sector_by_sid = (
                r_inv[["security_id", sector_col]]
                .drop_duplicates("security_id", keep="last")
                .set_index("security_id")[sector_col]
                .to_dict()
            )

        benchmark_cache = {}

        for period in thresholds:
            country_returns = []
            sector_returns = {}

            for sid in universe_ids:
                p = prices_by_security.get(sid)
                if p is None:
                    continue

                ret, _, _ = period_return(p, cutoff, period)
                if pd.isna(ret):
                    continue

                country_returns.append(ret)

                sector = sector_by_sid.get(sid)
                if pd.notna(sector):
                    sector_returns.setdefault(str(sector), []).append(ret)

            country_avg = (
                float(np.mean(country_returns))
                if country_returns
                else np.nan
            )

            benchmark_cache[period] = {
                "country_avg": country_avg,
                "country_count": len(country_returns),
                "sector_returns": sector_returns,
            }

        r_targets = r_inv[
        [
            (str(review), str(sid)) in FORCED_EPI_KEYS
            for sid in r_inv["security_id"]
        ]
    ].copy()

        for _, row in r_targets.iterrows():
            sid = row["security_id"]
            applicable = True
            p = prices_by_security.get(sid)

            out = {
                "review": review,
                "security_id": sid,
                "nse_symbol": row.get("nse_symbol", pd.NA),
                "was_standard_constituent": bool(
                    row["was_standard_constituent"]
                ),
                "epi_base_candidate": bool(row["epi_base_candidate"]),
                "epi_applicable": True,
                "epi_forced_downstream": True,
                "announcement_date": announcement,
                "price_cutoff_proxy": cutoff,
                "price_cutoff_is_exact": False,
                "price_cutoff_proxy_used": True,
                "price_cutoff_proxy_method":
                    "EARLIEST_OF_LAST_10_OBSERVED_MARKET_DATES",
                "price_field_used": price_col,
                "sector_benchmark_available": sector_col is not None,
                "exact_imi_membership_available": False,
            }

            if not applicable:
                out.update({
                    "epi_status": "NOT_APPLICABLE",
                    "extreme_price_pass": True,
                    "extreme_price_fail": False,
                    "ipo_exempt": False,
                    "epi_tested_period_count": 0,
                    "epi_breached_period_count": 0,
                    "epi_worst_period": np.nan,
                    "epi_max_excess_return": np.nan,
                    "epi_max_threshold_ratio": np.nan,
                    "epi_benchmark_method_at_worst_period": pd.NA,
                    "epi_benchmark_count_at_worst_period": np.nan,
                })
                output_rows.append(out)
                continue

            first_date = first_price_date.get(sid, pd.NaT)

            if ipo_col is not None:
                ipo_exempt = bool(row.get(ipo_col, False))
                ipo_source = f"INVESTABILITY_COLUMN:{ipo_col}"
            else:
                ipo_exempt = (
                    pd.notna(first_date)
                    and first_date > cutoff - pd.DateOffset(months=IPO_PROXY_MONTHS)
                )
                ipo_source = "FIRST_PRICE_DATE_3_MONTH_PROXY"

            if ipo_exempt:
                out.update({
                    "epi_status": "IPO_EXEMPT",
                    "extreme_price_pass": True,
                    "extreme_price_fail": False,
                    "ipo_exempt": True,
                    "ipo_exemption_source": ipo_source,
                    "first_price_date": first_date,
                    "epi_tested_period_count": 0,
                    "epi_breached_period_count": 0,
                    "epi_worst_period": np.nan,
                    "epi_max_excess_return": np.nan,
                    "epi_max_threshold_ratio": np.nan,
                    "epi_benchmark_method_at_worst_period": pd.NA,
                    "epi_benchmark_count_at_worst_period": np.nan,
                })
                output_rows.append(out)
                continue

            if p is None:
                out.update({
                    "epi_status": "NO_PRICE_HISTORY",
                    "extreme_price_pass": pd.NA,
                    "extreme_price_fail": pd.NA,
                    "ipo_exempt": False,
                    "ipo_exemption_source": ipo_source,
                    "first_price_date": first_date,
                    "epi_tested_period_count": 0,
                    "epi_breached_period_count": 0,
                    "epi_worst_period": np.nan,
                    "epi_max_excess_return": np.nan,
                    "epi_max_threshold_ratio": np.nan,
                    "epi_benchmark_method_at_worst_period": pd.NA,
                    "epi_benchmark_count_at_worst_period": np.nan,
                })
                output_rows.append(out)
                continue

            sector = sector_by_sid.get(sid)
            tested = 0
            breaches = 0
            max_ratio = -np.inf
            max_excess = -np.inf
            worst_period = np.nan
            worst_benchmark_method = pd.NA
            worst_benchmark_count = np.nan

            period_results = {}

            for period, threshold in thresholds.items():
                stock_ret, start_date, end_date = period_return(
                    p,
                    cutoff,
                    period,
                )

                b = benchmark_cache[period]
                sector_vals = (
                    b["sector_returns"].get(str(sector), [])
                    if pd.notna(sector)
                    else []
                )

                if (
                    sector_col is not None
                    and pd.notna(sector)
                    and len(sector_vals) >= MIN_SECTOR_IMI_COUNT
                ):
                    benchmark_ret = float(np.mean(sector_vals))
                    benchmark_method = "COUNTRY_SECTOR_IMI_PROXY"
                    benchmark_count = len(sector_vals)
                else:
                    benchmark_ret = b["country_avg"]
                    benchmark_method = "COUNTRY_IMI_PROXY"
                    benchmark_count = b["country_count"]

                if pd.isna(stock_ret) or pd.isna(benchmark_ret):
                    continue

                tested += 1
                excess = stock_ret - benchmark_ret
                breached = excess > threshold
                breaches += int(breached)

                ratio = excess / threshold

                period_results[period] = {
                    "stock_return": stock_ret,
                    "benchmark_return": benchmark_ret,
                    "excess_return": excess,
                    "threshold": threshold,
                    "breach": breached,
                    "benchmark_method": benchmark_method,
                    "benchmark_count": benchmark_count,
                    "start_date": start_date,
                    "end_date": end_date,
                }

                if ratio > max_ratio:
                    max_ratio = ratio
                    max_excess = excess
                    worst_period = period
                    worst_benchmark_method = benchmark_method
                    worst_benchmark_count = benchmark_count

            if tested == 0:
                status = "INSUFFICIENT_HISTORY"
                pass_value = pd.NA
                fail_value = pd.NA
            elif breaches > 0:
                status = "FAIL"
                pass_value = False
                fail_value = True
            else:
                status = "PASS"
                pass_value = True
                fail_value = False

            out.update({
                "epi_status": status,
                "extreme_price_pass": pass_value,
                "extreme_price_fail": fail_value,
                "ipo_exempt": False,
                "ipo_exemption_source": ipo_source,
                "first_price_date": first_date,
                "epi_tested_period_count": tested,
                "epi_breached_period_count": breaches,
                "epi_worst_period": worst_period,
                "epi_max_excess_return":
                    max_excess if np.isfinite(max_excess) else np.nan,
                "epi_max_threshold_ratio":
                    max_ratio if np.isfinite(max_ratio) else np.nan,
                "epi_benchmark_method_at_worst_period":
                    worst_benchmark_method,
                "epi_benchmark_count_at_worst_period":
                    worst_benchmark_count,
            })

            for period in thresholds:
                pr = period_results.get(period)
                prefix = f"epi_{period}d"

                if pr is None:
                    out[f"{prefix}_stock_return"] = np.nan
                    out[f"{prefix}_benchmark_return"] = np.nan
                    out[f"{prefix}_excess_return"] = np.nan
                    out[f"{prefix}_threshold"] = thresholds[period]
                    out[f"{prefix}_breach"] = pd.NA
                else:
                    out[f"{prefix}_stock_return"] = pr["stock_return"]
                    out[f"{prefix}_benchmark_return"] = pr["benchmark_return"]
                    out[f"{prefix}_excess_return"] = pr["excess_return"]
                    out[f"{prefix}_threshold"] = pr["threshold"]
                    out[f"{prefix}_breach"] = pr["breach"]

            output_rows.append(out)

    patched = pd.DataFrame(output_rows)

    if old_epi.empty:
        out = patched
    else:
        old_epi["review"] = old_epi["review"].astype(str)
        old_epi["security_id"] = old_epi["security_id"].astype("string")

        patch_idx = pd.MultiIndex.from_tuples(
            FORCED_EPI_KEYS,
            names=["review", "security_id"],
        )

        old_idx = pd.MultiIndex.from_frame(
            old_epi[["review", "security_id"]]
        )

        old_epi = old_epi.loc[~old_idx.isin(patch_idx)].copy()

        out = pd.concat([old_epi, patched], ignore_index=True, sort=False)

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate review/security_id rows.")

    if out["security_id"].isna().any():
        raise RuntimeError("Missing security_id rows.")

    accidentally_screened = (
        out["was_standard_constituent"]
        & out["epi_applicable"]
    ).sum()

    if accidentally_screened:
        raise RuntimeError(
            f"{int(accidentally_screened)} incumbents were incorrectly screened."
        )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- EXTREME PRICE SCREEN ---")
    print("Price field:", price_col)
    print("Rows:", len(out))
    print("Applicable additions:", int(out["epi_applicable"].sum()))
    print("Existing Standard accidentally screened:", int(accidentally_screened))
    print("Exact MSCI cutoff rows:", int(out["price_cutoff_is_exact"].sum()))
    print("Proxy cutoff rows:", int(out["price_cutoff_proxy_used"].sum()))
    print("PIT GICS/sector available:", bool(sector_col is not None))

    print("\n--- STATUS ---")
    print(out["epi_status"].value_counts(dropna=False).to_string())

    print("\n--- BY REVIEW ---")
    summary = (
        out.groupby("review")
        .agg(
            rows=("security_id", "size"),
            applicable=("epi_applicable", "sum"),
            pass_count=("epi_status", lambda s: (s == "PASS").sum()),
            fail_count=("epi_status", lambda s: (s == "FAIL").sum()),
            ipo_exempt=("epi_status", lambda s: (s == "IPO_EXEMPT").sum()),
            insufficient_history=(
                "epi_status",
                lambda s: (s == "INSUFFICIENT_HISTORY").sum(),
            ),
        )
    )
    print(summary.to_string())

    failures = out[out["epi_status"].eq("FAIL")][
        [
            "review",
            "security_id",
            "nse_symbol",
            "epi_worst_period",
            "epi_max_excess_return",
            "epi_max_threshold_ratio",
            "epi_breached_period_count",
            "epi_benchmark_method_at_worst_period",
            "epi_benchmark_count_at_worst_period",
        ]
    ].sort_values(
        ["review", "epi_max_threshold_ratio"],
        ascending=[True, False],
    )

    print("\n--- FAILURES ---")
    if failures.empty:
        print("None")
    else:
        print(failures.to_string(index=False))

    unresolved = out[
        out["epi_status"].isin(
            ["NO_PRICE_HISTORY", "INSUFFICIENT_HISTORY"]
        )
    ][
        [
            "review",
            "security_id",
            "nse_symbol",
            "epi_status",
            "first_price_date",
        ]
    ]

    print("\n--- UNRESOLVED ---")
    if unresolved.empty:
        print("None")
    else:
        print(unresolved.to_string(index=False))

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
