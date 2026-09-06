from pathlib import Path

import pandas as pd
import yfinance as yf


OUT_PATH = Path("data/processed/gmsr_estimates.parquet")
CACHE_PATH = Path("data/raw/market/urth_gmsr_proxy.parquet")
BENCHMARK = "URTH"

REVIEW_ORDER = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05",
    "2024-08", "2024-11", "2025-02", "2025-05", "2025-08",
    "2025-11", "2026-02", "2026-05",
]

PRIOR_REVIEW_GMSR = {
    "2023-05": ("2023-02", "2023-01-18", 4024.0),
    "2023-08": ("2023-05", "2023-04-18", 4133.0),
    "2023-11": ("2023-08", "2023-07-18", 4503.0),
    "2024-02": ("2023-11", "2023-10-18", 4352.0),
    "2024-05": ("2024-02", "2024-01-18", 4838.0),
    "2024-08": ("2024-05", "2024-04-17", 5096.0),
    "2024-11": ("2024-08", "2024-07-18", 5519.0),
    "2025-02": ("2024-11", "2024-10-16", 5934.0),
    "2025-05": ("2025-02", "2025-01-17", 6175.0),
    "2025-08": ("2025-05", "2025-04-17", 5928.0),
    "2025-11": ("2025-08", "2025-07-16", 6676.0),
    "2026-02": ("2025-11", "2025-10-15", 7153.0),
    "2026-05": ("2026-02", "2026-01-16", 7599.0),
}

PRICE_CUTOFF_PROXY = {
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


def download_benchmark():
    start = min(
        pd.Timestamp(v[1]) for v in PRIOR_REVIEW_GMSR.values()
    ) - pd.Timedelta(days=10)

    end = max(
        pd.Timestamp(v) for v in PRICE_CUTOFF_PROXY.values()
    ) + pd.Timedelta(days=10)

    try:
        x = yf.download(
            BENCHMARK,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )

        if x.empty:
            raise RuntimeError("Empty download.")

        if isinstance(x.columns, pd.MultiIndex):
            close = x["Close"][BENCHMARK]
        else:
            close = x["Close"]

        out = (
            close.dropna()
            .rename("close")
            .reset_index()
            .rename(columns={"Date": "date"})
        )

        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        out = out.sort_values("date").drop_duplicates("date")

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(CACHE_PATH, index=False)

        return out

    except Exception:
        if not CACHE_PATH.exists():
            raise

        out = pd.read_parquet(CACHE_PATH)
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")


def latest_close(px, date):
    date = pd.Timestamp(date)
    z = px[px["date"] <= date]

    if z.empty:
        raise RuntimeError(f"No {BENCHMARK} price <= {date.date()}")

    r = z.iloc[-1]
    return pd.Timestamp(r["date"]), float(r["close"])


def main():
    px = download_benchmark()
    rows = []

    for review in REVIEW_ORDER:
        source_review, anchor_date, prior_gmsr = PRIOR_REVIEW_GMSR[review]
        cutoff = pd.Timestamp(PRICE_CUTOFF_PROXY[review])

        start_date, p0 = latest_close(px, anchor_date)
        end_date, p1 = latest_close(px, cutoff)

        if start_date > pd.Timestamp(anchor_date):
            raise RuntimeError(f"{review}: benchmark anchor lookahead.")

        if end_date > cutoff:
            raise RuntimeError(f"{review}: benchmark cutoff lookahead.")

        scaled = prior_gmsr * p1 / p0

        rows.append({
            "review": review,
            "est_em_standard_ref_usd_m": scaled,
            "prior_official_em_standard_ref_usd_m": prior_gmsr,
            "gmsr_proxy_method": "PRIOR_OFFICIAL_GMSR_SCALED_BY_URTH",
            "gmsr_source_review": source_review,
            "gmsr_source_data_date": pd.Timestamp(anchor_date),
            "gmsr_proxy_cutoff_date": cutoff,
            "benchmark": BENCHMARK,
            "benchmark_anchor_date": start_date,
            "benchmark_cutoff_date": end_date,
            "benchmark_anchor_close": p0,
            "benchmark_cutoff_close": p1,
            "benchmark_return": p1 / p0 - 1,
            "gmsr_lookahead": False,
        })

    out = pd.DataFrame(rows)

    if out["review"].duplicated().any():
        raise RuntimeError("Duplicate GMSR reviews.")

    if out["est_em_standard_ref_usd_m"].isna().any():
        raise RuntimeError("Missing GMSR estimate.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp = OUT_PATH.with_name(OUT_PATH.stem + "_tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)

    print("\n--- PIT MARKET-SCALED GMSR ---")
    print(
        out[[
            "review",
            "gmsr_source_review",
            "prior_official_em_standard_ref_usd_m",
            "benchmark_return",
            "est_em_standard_ref_usd_m",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()