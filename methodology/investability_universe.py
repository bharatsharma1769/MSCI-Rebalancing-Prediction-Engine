from pathlib import Path
import pandas as pd


PRICE_PATH = Path("data/processed/india_price_cutoff_snapshot.parquet")
LIQUIDITY_PATH = Path("data/processed/liquidity_monthly.parquet")
MEMBERSHIP_PATH = Path("data/processed/msci_standard_membership.parquet")
OUT_PATH = Path("data/processed/investability_universe.parquet")

REVIEW_ORDER = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05", "2024-08", "2024-11",
    "2025-02", "2025-05", "2025-08", "2025-11", "2026-02", "2026-05",
]

MEMBERSHIP_TRUE = {("2023-05", "SEC000820")}

LIQUIDITY_OVERRIDES = {
    ("2023-05", "HIST_HDFC"): (0.406133, 0.364341, 0.953846),
    ("2023-05", "SEC001391"): (0.331547, 0.283622, 0.953846),
    ("2023-08", "SEC001391"): (0.332251, 0.382093, 0.937500),
    ("2024-05", "HIST_TATAMTRDVR"): (1.043898, 0.784227, 0.937500),
    ("2024-08", "HIST_TATAMTRDVR"): (0.980558, 0.798734, 0.923077),
}


def implementation_date_proxy(review):
    return pd.Timestamp(review + "-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)


def latest_liquidity(liq, date_col, cutoff):
    cols = ["security_id", date_col, "atvr_12m", "atvr_3m", "fot_3m"]
    x = liq.loc[liq[date_col].le(cutoff), cols].copy()
    x = x.sort_values(["security_id", date_col]).groupby("security_id", as_index=False).tail(1)
    return x.rename(columns={date_col: "liquidity_observation_date"})


def quarter_history(liq, date_col, cutoff, listings):
    cols = ["security_id", date_col, "atvr_3m", "fot_3m"]
    x = liq.loc[liq[date_col].le(cutoff), cols].copy()
    x = x.merge(listings, on="security_id", how="inner", validate="many_to_one")
    x = x[x["listing_date"].notna() & x[date_col].ge(x["listing_date"])].copy()

    x["quarter"] = x[date_col].dt.to_period("Q")
    x = x.sort_values(["security_id", "quarter", date_col]).groupby(["security_id", "quarter"], as_index=False).tail(1)
    x = x.sort_values(["security_id", date_col], ascending=[True, False])
    x["quarter_rank"] = x.groupby("security_id").cumcount() + 1
    x = x[x["quarter_rank"].le(4)].copy()

    agg = x.groupby("security_id", as_index=False).agg(
        liquidity_available_quarters=("quarter", "nunique"),
        liquidity_4q_min_atvr_3m=("atvr_3m", "min"),
        liquidity_4q_min_fot_3m=("fot_3m", "min"),
    )

    for q in range(1, 5):
        z = x[x["quarter_rank"].eq(q)][["security_id", date_col, "atvr_3m", "fot_3m"]].rename(columns={
            date_col: f"liq_q{q}_date",
            "atvr_3m": f"liq_q{q}_atvr_3m",
            "fot_3m": f"liq_q{q}_fot_3m",
        })
        agg = agg.merge(z, on="security_id", how="left", validate="one_to_one")

    agg["liquidity_4q_complete"] = agg["liquidity_available_quarters"].eq(4)
    return agg


def apply_membership_override(x):
    x["membership_override_applied"] = False

    for review, sid in MEMBERSHIP_TRUE:
        m = x["review"].eq(review) & x["security_id"].eq(sid)
        if m.sum() != 1: raise RuntimeError(f"{review} {sid}: expected 1 membership row, found {int(m.sum())}")
        x.loc[m, "was_standard_constituent"] = True
        x.loc[m, "membership_override_applied"] = True

    return x


def apply_liquidity_override(x, review):
    x["liquidity_override_applied"] = False

    for (rev, sid), values in LIQUIDITY_OVERRIDES.items():
        if rev != review: continue

        m = x["security_id"].eq(sid)
        if m.sum() != 1: raise RuntimeError(f"{review} {sid}: expected 1 liquidity row, found {int(m.sum())}")

        x.loc[m, ["atvr_12m", "atvr_3m", "fot_3m"]] = values
        x.loc[m, "liquidity_override_applied"] = True

    return x


def main():
    price = pd.read_parquet(PRICE_PATH)
    liquidity = pd.read_parquet(LIQUIDITY_PATH)
    membership = pd.read_parquet(MEMBERSHIP_PATH)

    required_price = {
        "review", "security_id", "nse_symbol", "company_name", "company_key", "listing_date",
        "universe_date", "liquidity_cutoff_date", "price_cutoff_proxy",
        "price_cutoff_refresh_valid", "full_market_cap_usd", "ff_market_cap_usd", "fif",
    }
    required_membership = {"review", "security_id", "was_standard_constituent"}

    if not required_price.issubset(price.columns): raise RuntimeError("Price snapshot missing: " + str(sorted(required_price - set(price.columns))))
    if not required_membership.issubset(membership.columns): raise RuntimeError("Membership missing: " + str(sorted(required_membership - set(membership.columns))))

    liq_date_col = next((c for c in ["month_end", "month", "date"] if c in liquidity.columns), None)
    if liq_date_col is None: raise RuntimeError("Cannot find liquidity date column.")

    required_liq = {"security_id", "atvr_12m", "atvr_3m", "fot_3m"}
    if not required_liq.issubset(liquidity.columns): raise RuntimeError("Liquidity missing: " + str(sorted(required_liq - set(liquidity.columns))))

    for df in [price, membership, liquidity]: df["security_id"] = df["security_id"].astype("string")

    price["review"] = price["review"].astype(str)
    membership["review"] = membership["review"].astype(str)
    liquidity[liq_date_col] = pd.to_datetime(liquidity[liq_date_col], errors="coerce")

    for c in ["listing_date", "universe_date", "liquidity_cutoff_date", "price_cutoff_proxy"]:
        price[c] = pd.to_datetime(price[c], errors="coerce")

    if membership.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate membership keys.")

    membership = membership[["review", "security_id", "was_standard_constituent"]].copy()
    all_rows = []

    for review in REVIEW_ORDER:
        x = price[price["review"].eq(review)].copy()
        if x.empty: raise RuntimeError(f"{review}: no Price-Cutoff snapshot.")

        universe_date = x["universe_date"].dropna().iloc[0]
        liquidity_cutoff = x["liquidity_cutoff_date"].dropna().iloc[0]

        m = membership[membership["review"].eq(review)]
        x = x.merge(m, on=["review", "security_id"], how="left", validate="one_to_one")
        x["was_standard_constituent"] = x["was_standard_constituent"].fillna(False).astype(bool)

        x["implementation_date_proxy"] = implementation_date_proxy(review)
        x["equity_universe_date_pass"] = x["listing_date"].notna() & x["listing_date"].le(universe_date)
        x["market_data_complete"] = x["price_cutoff_refresh_valid"].fillna(False).astype(bool)
        x["review_base_universe_pass"] = x["market_data_complete"] & (x["equity_universe_date_pass"] | x["was_standard_constituent"])
        x = x[x["review_base_universe_pass"]].copy()

        company = x.groupby("company_key", as_index=False, dropna=False).agg(
            company_full_mcap_usd=("full_market_cap_usd", "sum"),
            company_ff_mcap_usd=("ff_market_cap_usd", "sum"),
            company_security_count=("security_id", "nunique"),
        )
        x = x.merge(company, on="company_key", how="left", validate="many_to_one")

        listings = x[["security_id", "listing_date"]].drop_duplicates("security_id")
        latest = latest_liquidity(liquidity, liq_date_col, liquidity_cutoff)
        qhist = quarter_history(liquidity, liq_date_col, liquidity_cutoff, listings)

        x = x.merge(latest, on="security_id", how="left", validate="one_to_one")
        x = x.merge(qhist, on="security_id", how="left", validate="one_to_one")
        x = apply_liquidity_override(x, review)

        x["liquidity_available_quarters"] = x["liquidity_available_quarters"].fillna(0).astype(int)
        x["liquidity_4q_complete"] = x["liquidity_4q_complete"].fillna(False).astype(bool)
        x["liquidity_age_days"] = (liquidity_cutoff - x["listing_date"]).dt.days

        x["short_history_liquidity_rule_used"] = (
            x["liquidity_age_days"].between(0, 364, inclusive="both")
            & x["liquidity_available_quarters"].between(1, 3, inclusive="both")
        )

        x["liquidity_new_regular_pass"] = (
            x["atvr_12m"].ge(0.15)
            & x["liquidity_4q_complete"]
            & x["liquidity_4q_min_atvr_3m"].ge(0.15)
            & x["liquidity_4q_min_fot_3m"].ge(0.80)
        )

        x["liquidity_new_short_history_pass"] = (
            x["short_history_liquidity_rule_used"]
            & x["atvr_12m"].ge(0.15)
            & x["liquidity_4q_min_atvr_3m"].ge(0.15)
            & x["liquidity_4q_min_fot_3m"].ge(0.80)
        )

        x["liquidity_new_pass"] = x["liquidity_new_regular_pass"] | x["liquidity_new_short_history_pass"]
        x["liquidity_existing_pass"] = x["atvr_12m"].ge(0.10) & x["atvr_3m"].ge(0.05) & x["fot_3m"].ge(0.70)

        x["fif_regular_pass"] = pd.to_numeric(x["fif"], errors="coerce").ge(0.15)
        x["low_fif_exception_pending"] = ~x["fif_regular_pass"]

        three_month_date = x["implementation_date_proxy"] - pd.DateOffset(months=3)
        x["trading_length_regular_pass"] = x["listing_date"].notna() & x["listing_date"].le(three_month_date)
        x["short_trading_exception_pending"] = ~x["trading_length_regular_pass"]

        all_rows.append(x)

    out = pd.concat(all_rows, ignore_index=True)

    out["membership_override_applied"] = False
    for review, sid in MEMBERSHIP_TRUE:
        m = out["review"].eq(review) & out["security_id"].eq(sid)
        if m.sum() != 1: raise RuntimeError(f"{review} {sid}: expected 1 membership row, found {int(m.sum())}")
        out.loc[m, "was_standard_constituent"] = True
        out.loc[m, "membership_override_applied"] = True

    out["was_imi_constituent"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["mieu_liquidity_pass"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["mieu_fif_pass"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["mieu_trading_pass"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["mieu_foreign_room_pass"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["mieu_pass"] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    if out.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate review/security_id rows.")

    diag = out.groupby("review").agg(
        rows=("security_id", "size"),
        standard_incumbents=("was_standard_constituent", "sum"),
        low_fif=("fif_regular_pass", lambda s: int((~s).sum())),
        trading_regular_fail=("trading_length_regular_pass", lambda s: int((~s).sum())),
        new_liq_pass=("liquidity_new_pass", "sum"),
        existing_liq_pass=("liquidity_existing_pass", "sum"),
        short_liq_used=("short_history_liquidity_rule_used", "sum"),
        short_liq_pass=("liquidity_new_short_history_pass", "sum"),
        liquidity_missing=("atvr_12m", lambda s: int(s.isna().sum())),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- INVESTABILITY BASE UNIVERSE ---")
    print(diag.to_string())
    print("\nMembership overrides:", int(out["membership_override_applied"].sum()))
    print("Liquidity overrides:", int(out["liquidity_override_applied"].sum()))
    print("Short-history rows:", int(out["short_history_liquidity_rule_used"].sum()))
    print("Short-history passes:", int(out["liquidity_new_short_history_pass"].sum()))
    print("Duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("Rows:", len(out))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()