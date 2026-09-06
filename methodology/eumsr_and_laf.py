from pathlib import Path
import pandas as pd


IN_PATH = Path("data/processed/investability_universe.parquet")
GMSR_PATH = Path("data/processed/gmsr_estimates.parquet")
OUT_PATH = Path("data/processed/investability_eumsr.parquet")

REVIEW_ORDER = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05", "2024-08", "2024-11",
    "2025-02", "2025-05", "2025-08", "2025-11", "2026-02", "2026-05",
]

EUMSR_USD_M = {
    "2023-05": 292.0,
    "2023-08": 291.0,
    "2023-11": 313.0,
    "2024-02": 323.0,
    "2024-05": 371.0,
    "2024-08": 383.0,
    "2024-11": 422.0,
    "2025-02": 443.0,
    "2025-05": 430.0,
    "2025-08": 448.0,
    "2025-11": 505.0,
    "2026-02": 507.0,
    "2026-05": 537.0,
}


def load_gmsr():
    g = pd.read_parquet(GMSR_PATH)
    required = {"review", "est_em_standard_ref_usd_m"}
    if not required.issubset(g.columns): raise RuntimeError("gmsr_estimates.parquet missing: " + str(sorted(required - set(g.columns))))

    g["review"] = g["review"].astype(str)
    g["standard_gmsr_usd"] = pd.to_numeric(g["est_em_standard_ref_usd_m"], errors="coerce") * 1_000_000
    g = g[["review", "standard_gmsr_usd"]].drop_duplicates("review", keep="last")

    if g["standard_gmsr_usd"].isna().any(): raise RuntimeError("Missing GMSR values.")
    return g


def main():
    x = pd.read_parquet(IN_PATH)
    g = load_gmsr()

    required = {
        "review", "security_id", "nse_symbol", "company_key",
        "company_full_mcap_usd", "ff_market_cap_usd",
        "was_standard_constituent", "liquidity_existing_pass",
    }

    if not required.issubset(x.columns):
        raise RuntimeError("investability_universe.parquet missing: " + str(sorted(required - set(x.columns))))

    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")
    x["was_standard_constituent"] = x["was_standard_constituent"].fillna(False).astype(bool)
    x["liquidity_existing_pass"] = x["liquidity_existing_pass"].fillna(False).astype(bool)

    if x.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate review/security_id rows.")

    x["equity_universe_min_size_usd"] = x["review"].map(EUMSR_USD_M) * 1_000_000
    if x["equity_universe_min_size_usd"].isna().any():
        bad = x.loc[x["equity_universe_min_size_usd"].isna(), "review"].drop_duplicates().tolist()
        raise RuntimeError(f"Missing EUMSR for reviews: {bad}")

    x["equity_universe_min_ff_mcap_usd"] = 0.5 * x["equity_universe_min_size_usd"]

    x["eumsr_new_full_mcap_pass"] = x["company_full_mcap_usd"].ge(x["equity_universe_min_size_usd"])
    x["eumsr_new_ff_mcap_pass"] = x["ff_market_cap_usd"].ge(x["equity_universe_min_ff_mcap_usd"])
    x["eumsr_new_pass"] = x["eumsr_new_full_mcap_pass"] & x["eumsr_new_ff_mcap_pass"]

    # Existing IMI constituents are exempt. Final selection between
    # existing/new treatment is done later from recursive IMI state.
    x["eumsr_existing_pass"] = True
    x["eumsr_pass"] = pd.Series(pd.NA, index=x.index, dtype="boolean")
    x["eumsr_state_pending"] = True

    x = x.merge(g, on="review", how="left", validate="many_to_one")
    if x["standard_gmsr_usd"].isna().any(): raise RuntimeError("Missing GMSR after merge.")

    # Initial LAF screen for known Standard incumbents.
    # Exact country-index weight may later be affected by foreign-room
    # adjustment state, so any trigger remains pending rather than guessed.
    std_ff = x["ff_market_cap_usd"].where(x["was_standard_constituent"], 0.0)
    x["existing_standard_ff_total_usd"] = std_ff.groupby(x["review"]).transform("sum")

    x["existing_standard_weight_proxy"] = pd.NA
    m = x["was_standard_constituent"] & x["existing_standard_ff_total_usd"].gt(0)
    x.loc[m, "existing_standard_weight_proxy"] = (
        x.loc[m, "ff_market_cap_usd"] / x.loc[m, "existing_standard_ff_total_usd"]
    )
    x["existing_standard_weight_proxy"] = pd.to_numeric(x["existing_standard_weight_proxy"], errors="coerce")

    x["laf_initial_trigger_proxy"] = (
        x["was_standard_constituent"]
        & ~x["liquidity_existing_pass"]
        & x["existing_standard_weight_proxy"].gt(0.10)
        & x["ff_market_cap_usd"].gt(0.5 * x["standard_gmsr_usd"])
    )

    x["liquidity_adjustment_factor_pending"] = x["laf_initial_trigger_proxy"]
    x["liquidity_adjustment_factor"] = 1.0
    x.loc[x["laf_initial_trigger_proxy"], "liquidity_adjustment_factor"] = pd.NA

    if x.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate keys after EUMSR/GMSR merge.")

    diag = x.groupby("review").agg(
        rows=("security_id", "size"),
        eumsr_usd_m=("equity_universe_min_size_usd", lambda s: float(s.iloc[0]) / 1_000_000),
        new_full_fail=("eumsr_new_full_mcap_pass", lambda s: int((~s).sum())),
        new_ff_fail=("eumsr_new_ff_mcap_pass", lambda s: int((~s).sum())),
        new_eumsr_pass=("eumsr_new_pass", "sum"),
        standard_incumbents=("was_standard_constituent", "sum"),
        std_incumbent_new_eumsr_fail=("eumsr_new_pass", lambda s: 0),
        laf_proxy_triggers=("laf_initial_trigger_proxy", "sum"),
    )

    std_fail = (
        x[x["was_standard_constituent"]]
        .groupby("review")["eumsr_new_pass"]
        .apply(lambda s: int((~s).sum()))
    )
    diag["std_incumbent_new_eumsr_fail"] = std_fail.reindex(diag.index, fill_value=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(OUT_PATH, index=False)

    print("\n--- EUMSR + LAF BASE ---")
    print(diag.to_string())
    print("\nEUMSR state pending:", int(x["eumsr_state_pending"].sum()))
    print("LAF pending:", int(x["liquidity_adjustment_factor_pending"].sum()))
    print("Duplicates:", int(x.duplicated(["review", "security_id"]).sum()))
    print("Rows:", len(x))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()