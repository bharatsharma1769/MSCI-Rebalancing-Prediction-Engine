from pathlib import Path
import re

import numpy as np
import pandas as pd
import yfinance as yf


MARKET_PATH = Path("data/processed/market_data_fif.parquet")
MASTER_PATH = Path("data/processed/security_master.parquet")
GMSR_PATH = Path("data/processed/gmsr_estimates.parquet")

OUT_PATH = Path("data/processed/india_size_cutoffs.parquet")
SNAPSHOT_PATH = Path("data/processed/india_price_cutoff_snapshot.parquet")
FX_CACHE_PATH = Path("data/raw/market/inr_usd_fx.parquet")

REVIEW_ORDER = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05",
    "2024-08", "2024-11", "2025-02", "2025-05", "2025-08",
    "2025-11", "2026-02", "2026-05",
]

UNIVERSE_DATE = {
    "2023-05": "2023-02-28", "2023-08": "2023-05-31", "2023-11": "2023-08-31",
    "2024-02": "2023-11-30", "2024-05": "2024-02-29", "2024-08": "2024-05-31",
    "2024-11": "2024-08-30", "2025-02": "2024-11-29", "2025-05": "2025-02-28",
    "2025-08": "2025-05-30", "2025-11": "2025-08-29", "2026-02": "2025-11-28",
    "2026-05": "2026-02-27",
}

LIQUIDITY_CUTOFF = {
    "2023-05": "2023-03-31", "2023-08": "2023-06-30", "2023-11": "2023-09-29",
    "2024-02": "2023-12-29", "2024-05": "2024-03-28", "2024-08": "2024-06-28",
    "2024-11": "2024-09-30", "2025-02": "2024-12-31", "2025-05": "2025-03-31",
    "2025-08": "2025-06-30", "2025-11": "2025-09-30", "2026-02": "2025-12-31",
    "2026-05": "2026-03-31",
}

PRICE_CUTOFF = {
    "2023-05": "2023-04-17", "2023-08": "2023-07-18", "2023-11": "2023-10-18",
    "2024-02": "2024-01-18", "2024-05": "2024-04-17", "2024-08": "2024-07-18",
    "2024-11": "2024-10-18", "2025-02": "2025-01-20", "2025-05": "2025-04-17",
    "2025-08": "2025-07-18", "2025-11": "2025-10-20", "2026-02": "2026-01-19",
    "2026-05": "2026-04-17",
}

PIT_FIF_OVERRIDES = {
    ("2023-05", "SEC000044"): 0.10,  # ADANIENSOL
    ("2023-05", "SEC000206"): 0.14,  # ATGL
    ("2023-08", "SEC000034"): 0.35,  # ACC
    ("2023-08", "SEC000045"): 0.15,  # ADANIENT
    ("2023-08", "SEC000044"): 0.10,  # ADANIENSOL
    ("2023-08", "SEC000206"): 0.14,  # ATGL
}

MAX_PRICE_STALENESS_DAYS = 15


def normalize_company_name(x):
    if pd.isna(x): return None
    s = re.sub(r"[^A-Z0-9]+", " ", str(x).upper().strip())
    s = re.sub(r"\bLIMITED\b", "LTD", s)
    s = re.sub(r"\bPRIVATE\b", "PVT", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def load_fx():
    start = pd.Timestamp(min(PRICE_CUTOFF.values())) - pd.Timedelta(days=20)
    end = pd.Timestamp(max(PRICE_CUTOFF.values())) + pd.Timedelta(days=10)

    try:
        x = yf.download("INR=X", start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False, progress=False)
        if x.empty: raise RuntimeError("Empty INR=X download.")
        close = x["Close"]["INR=X"] if isinstance(x.columns, pd.MultiIndex) else x["Close"]
        out = close.dropna().rename("usd_inr").reset_index()
        out.columns = ["date", "usd_inr"]
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        FX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(FX_CACHE_PATH, index=False)
        return out.sort_values("date")
    except Exception:
        if not FX_CACHE_PATH.exists(): raise
        out = pd.read_parquet(FX_CACHE_PATH)
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")


def latest_fx(fx, cutoff):
    z = fx[fx["date"] <= cutoff]
    if z.empty: raise RuntimeError(f"No USD/INR observation <= {cutoff.date()}")
    row = z.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["usd_inr"])


def load_gmsr():
    g = pd.read_parquet(GMSR_PATH)
    if not {"review", "est_em_standard_ref_usd_m"}.issubset(g.columns): raise RuntimeError("gmsr_estimates.parquet missing required columns.")
    g["review"] = g["review"].astype(str)
    g["standard_gmsr_usd"] = pd.to_numeric(g["est_em_standard_ref_usd_m"], errors="coerce") * 1_000_000
    if g["standard_gmsr_usd"].isna().any(): raise RuntimeError("Missing GMSR estimate.")
    return g[["review", "standard_gmsr_usd"]].drop_duplicates("review", keep="last")


def apply_fif_overrides(x, review):
    x = x.copy()
    x["pit_fif_override_applied"] = False

    for (rev, sid), fif in PIT_FIF_OVERRIDES.items():
        if rev != review: continue
        m = x["security_id"].eq(sid)
        if m.sum() != 1: raise RuntimeError(f"{review} {sid}: expected 1 Price-Cutoff row, found {int(m.sum())}")
        x.loc[m, "fif"] = fif
        x.loc[m, "pit_fif_override_applied"] = True

    return x


def build_company_key(x):
    key = x["company_name"].map(normalize_company_name).astype("string")
    x["company_key"] = key.where(key.notna(), "SECURITY::" + x["security_id"].astype("string"))
    x.loc[x["security_id"].eq("HIST_TATAMTRDVR"), "company_key"] = "TATA MOTORS LTD"
    return x


def build_price_snapshot(market, master, fx, review):
    cutoff = pd.Timestamp(PRICE_CUTOFF[review])
    fx_date, usd_inr = latest_fx(fx, cutoff)

    x = market[market["date"] <= cutoff].sort_values(["security_id", "date"]).groupby("security_id", as_index=False).tail(1).copy()
    x = x.merge(master, on="security_id", how="left", validate="many_to_one")
    x = build_company_key(x)
    x = apply_fif_overrides(x, review)

    x["price_cutoff_proxy"] = cutoff
    x["price_cutoff_market_date"] = x["date"]
    x["price_cutoff_data_age_days"] = (cutoff - x["date"]).dt.days
    x["price_cutoff_refresh_valid"] = x["price_cutoff_data_age_days"].between(0, MAX_PRICE_STALENESS_DAYS, inclusive="both")
    x["full_market_cap_usd"] = pd.to_numeric(x["full_market_cap_inr"], errors="coerce") / usd_inr
    x["fif"] = pd.to_numeric(x["fif"], errors="coerce")
    x["ff_market_cap_usd"] = x["full_market_cap_usd"] * x["fif"]
    x["price_cutoff_refresh_valid"] &= x["full_market_cap_usd"].notna() & x["ff_market_cap_usd"].notna() & x["fif"].notna()
    x["review"] = review
    x["universe_date"] = pd.Timestamp(UNIVERSE_DATE[review])
    x["liquidity_cutoff_date"] = pd.Timestamp(LIQUIDITY_CUTOFF[review])
    x["fx_observation_date"] = fx_date
    x["usd_inr"] = usd_inr

    keep = [
        "review", "security_id", "nse_symbol", "company_name", "company_key",
        "listing_date", "universe_date", "liquidity_cutoff_date",
        "price_cutoff_proxy", "price_cutoff_market_date", "price_cutoff_data_age_days",
        "price_cutoff_refresh_valid", "fx_observation_date", "usd_inr",
        "full_market_cap_usd", "ff_market_cap_usd", "fif",
        "pit_fif_override_applied",
    ]
    return x[[c for c in keep if c in x.columns]]


def aggregate_companies(snapshot, eligible_company_keys=None):
    x = snapshot[snapshot["price_cutoff_refresh_valid"]].copy()
    if eligible_company_keys is not None: x = x[x["company_key"].isin(set(eligible_company_keys))]

    c = x.groupby("company_key", as_index=False, dropna=False).agg(
        company_full_mcap_usd=("full_market_cap_usd", "sum"),
        company_ff_mcap_usd=("ff_market_cap_usd", "sum"),
        security_count=("security_id", "nunique"),
    )
    c = c[c["company_key"].notna() & c["company_full_mcap_usd"].gt(0) & c["company_ff_mcap_usd"].ge(0)].copy()
    c = c.sort_values(["company_full_mcap_usd", "company_ff_mcap_usd", "company_key"], ascending=[False, False, True]).reset_index(drop=True)
    c["company_rank"] = np.arange(1, len(c) + 1)
    return c


def derive_interim_cutoff(snapshot, prior_mieu_company_keys, prior_segment_number, eumsr_usd):
    """
    QIR Interim Standard cutoff.

    prior_mieu_company_keys:
        Post-review MIEU company state entering this review.

    prior_segment_number:
        Standard Segment Number entering this review.

    The QIR Interim cutoff is NOT clamped to the current GMSR range.
    It is floored only at the EUMSR.
    """
    companies = aggregate_companies(snapshot, prior_mieu_company_keys)
    if companies.empty: raise RuntimeError("Prior MIEU produced no valid Price-Cutoff companies.")

    n = int(prior_segment_number)
    if n < 1: raise RuntimeError("Prior Segment Number must be positive.")
    if n > len(companies): raise RuntimeError(f"Prior Segment Number {n} exceeds prior-MIEU company count {len(companies)}.")

    row = companies.iloc[n - 1]
    raw = float(row["company_full_mcap_usd"])
    interim = max(raw, float(eumsr_usd))

    return {
        "prior_segment_number": n,
        "prior_mieu_company_count": len(companies),
        "raw_interim_standard_cutoff_usd": raw,
        "interim_standard_cutoff_usd": interim,
        "interim_eumsr_floor_applied": raw < float(eumsr_usd),
        "interim_cutoff_rank_company_key": row["company_key"],
    }


def main():
    market = pd.read_parquet(MARKET_PATH)
    master = pd.read_parquet(MASTER_PATH)
    gmsr = load_gmsr()
    fx = load_fx()

    required_market = {"security_id", "date", "full_market_cap_inr", "fif"}
    required_master = {"security_id", "nse_symbol", "company_name"}

    if not required_market.issubset(market.columns): raise RuntimeError(f"market_data_fif.parquet missing: {sorted(required_market - set(market.columns))}")
    if not required_master.issubset(master.columns): raise RuntimeError(f"security_master.parquet missing: {sorted(required_master - set(master.columns))}")

    market["security_id"] = market["security_id"].astype("string")
    master["security_id"] = master["security_id"].astype("string")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    if "listing_date" in master.columns: master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")

    master_cols = ["security_id", "nse_symbol", "company_name"]
    if "listing_date" in master.columns: master_cols.append("listing_date")
    master = master[master_cols].drop_duplicates("security_id", keep="last")

    snapshots, summary = [], []

    for review in REVIEW_ORDER:
        s = build_price_snapshot(market, master, fx, review)
        snapshots.append(s)

        valid = s[s["price_cutoff_refresh_valid"]]
        companies = aggregate_companies(s)
        g = gmsr[gmsr["review"].eq(review)]

        if g.empty: raise RuntimeError(f"{review}: missing GMSR.")

        summary.append({
            "review": review,
            "universe_date": pd.Timestamp(UNIVERSE_DATE[review]),
            "liquidity_cutoff_date": pd.Timestamp(LIQUIDITY_CUTOFF[review]),
            "price_cutoff_proxy": pd.Timestamp(PRICE_CUTOFF[review]),
            "fx_observation_date": s["fx_observation_date"].iloc[0],
            "usd_inr": s["usd_inr"].iloc[0],
            "standard_gmsr_usd": float(g["standard_gmsr_usd"].iloc[0]),
            "valid_security_rows": len(valid),
            "valid_company_count": len(companies),
            "stale_or_missing_rows": int((~s["price_cutoff_refresh_valid"]).sum()),
            "pit_fif_overrides": int(s["pit_fif_override_applied"].sum()),
        })

    snapshot = pd.concat(snapshots, ignore_index=True)
    out = pd.DataFrame(summary)

    if snapshot.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate review/security_id rows in Price-Cutoff snapshot.")
    if out["review"].duplicated().any(): raise RuntimeError("Duplicate review rows in cutoff metadata.")
    if len(out) != len(REVIEW_ORDER): raise RuntimeError("Missing review metadata rows.")

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(SNAPSHOT_PATH, index=False)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- INDIA PRICE-CUTOFF INPUTS ---")
    print(out[["review", "price_cutoff_proxy", "valid_security_rows", "valid_company_count", "stale_or_missing_rows", "pit_fif_overrides"]].to_string(index=False))
    print("\nPrice snapshot rows:", len(snapshot))
    print("Duplicates:", int(snapshot.duplicated(["review", "security_id"]).sum()))
    print("Saved:", SNAPSHOT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()