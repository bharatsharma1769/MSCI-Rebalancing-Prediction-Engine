from pathlib import Path
import numpy as np
import pandas as pd


IN_PATH = Path("data/processed/investability_eumsr.parquet")
FOL_PATH = Path("data/processed/fol_master.parquet")
OUT_PATH = Path("data/processed/foreign_room_candidates.parquet")


def first_existing(cols, names):
    return next((c for c in names if c in cols), None)


def current_fol_reference(fol):
    x = fol.copy()
    x["security_id"] = x["security_id"].astype("string")

    if x["security_id"].duplicated().any():
        date_col = first_existing(x.columns, ["date", "as_of_date", "effective_date", "available_date", "updated_at"])
        if date_col is None: raise RuntimeError("fol_master has duplicate security_id rows and no date column.")
        x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
        x = x.sort_values(["security_id", date_col]).groupby("security_id", as_index=False).tail(1)

    fol_col = first_existing(x.columns, ["fol", "foreign_ownership_limit", "foreign_ownership_limit_pct", "fol_pct", "current_fol", "current_fol_reference"])
    source_col = first_existing(x.columns, ["source", "fol_source", "current_fol_source", "source_name", "provenance"])
    date_col = first_existing(x.columns, ["date", "as_of_date", "effective_date", "available_date", "updated_at"])

    out = x[["security_id"]].copy()

    if fol_col is None:
        out["current_fol_reference"] = np.nan
        out["current_fol_reference_raw"] = np.nan
        out["current_fol_reference_field"] = pd.NA
    else:
        raw = pd.to_numeric(x[fol_col], errors="coerce")
        out["current_fol_reference"] = np.where(raw.gt(1) & raw.le(100), raw / 100, raw)
        out["current_fol_reference_raw"] = raw
        out["current_fol_reference_field"] = fol_col

    out["current_fol_source"] = x[source_col].astype("string") if source_col else "fol_master.parquet"
    out["current_fol_asof_date"] = pd.to_datetime(x[date_col], errors="coerce") if date_col else pd.NaT
    out["current_fol_is_reference_only"] = True

    return out


def main():
    x = pd.read_parquet(IN_PATH)
    fol = pd.read_parquet(FOL_PATH)

    required = {
        "review", "security_id", "nse_symbol", "company_name", "company_key",
        "was_standard_constituent", "company_full_mcap_usd", "ff_market_cap_usd",
        "eumsr_new_full_mcap_pass", "eumsr_new_pass", "liquidity_new_pass",
        "equity_universe_min_size_usd",
    }

    if not required.issubset(x.columns):
        raise RuntimeError("investability_eumsr.parquet missing: " + str(sorted(required - set(x.columns))))

    if "security_id" not in fol.columns: raise RuntimeError("fol_master.parquet missing security_id.")

    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")
    x["was_standard_constituent"] = x["was_standard_constituent"].fillna(False).astype(bool)
    x["eumsr_new_full_mcap_pass"] = x["eumsr_new_full_mcap_pass"].fillna(False).astype(bool)
    x["eumsr_new_pass"] = x["eumsr_new_pass"].fillna(False).astype(bool)
    x["liquidity_new_pass"] = x["liquidity_new_pass"].fillna(False).astype(bool)

    if x.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate input review/security_id rows.")

    x = x.merge(current_fol_reference(fol), on="security_id", how="left", validate="many_to_one")
    x["current_fol_is_reference_only"] = x["current_fol_is_reference_only"].fillna(True).astype(bool)

    # New/non-IMI candidate before FIF, trading-length exceptions and foreign-room screen.
    x["preliminary_pass"] = x["eumsr_new_pass"] & x["liquidity_new_pass"]

    # Additional conservative watch set. A restricted present-day FOL is only
    # a research signal; it is never substituted for historical FOL.
    x["restricted_fol_reference"] = x["current_fol_reference"].notna() & x["current_fol_reference"].lt(0.999)
    x["restricted_fol_watch"] = x["restricted_fol_reference"] & x["eumsr_new_full_mcap_pass"]

    x["selection_existing_standard"] = x["was_standard_constituent"]
    x["selection_preliminary_pass"] = x["preliminary_pass"]
    x["selection_restricted_fol_watch"] = x["restricted_fol_watch"]

    x["needs_foreign_room_check"] = (
        x["selection_existing_standard"]
        | x["selection_preliminary_pass"]
        | x["selection_restricted_fol_watch"]
    )

    x = x[x["needs_foreign_room_check"]].copy()

    x["selection_reason"] = np.select(
        [
            x["selection_existing_standard"] & x["selection_preliminary_pass"],
            x["selection_existing_standard"],
            x["selection_preliminary_pass"] & x["selection_restricted_fol_watch"],
            x["selection_preliminary_pass"],
            x["selection_restricted_fol_watch"],
        ],
        [
            "EXISTING_STANDARD+NEW_SCREEN_PASS",
            "EXISTING_STANDARD",
            "NEW_SCREEN_PASS+RESTRICTED_FOL_WATCH",
            "NEW_SCREEN_PASS",
            "RESTRICTED_FOL_WATCH",
        ],
        default="SELECTED",
    )

    # Compatibility fields for the historical FO/FOL layer. No Standard
    # boundary exists yet, so boundary information is intentionally unknown.
    x["boundary_ratio"] = np.nan
    x["near_standard_boundary"] = False
    x["selection_near_boundary"] = False

    x["needs_historical_foreign_ownership"] = True
    x["needs_historical_fol"] = True
    x["needs_india_red_flag_breach_check"] = True

    x["full_mcap_usd"] = pd.to_numeric(x["company_full_mcap_usd"], errors="coerce")
    x["ff_mcap_usd"] = pd.to_numeric(x["ff_market_cap_usd"], errors="coerce")

    keep = [
        "review", "security_id", "nse_symbol", "company_name", "company_key",
        "was_standard_constituent", "preliminary_pass",
        "eumsr_new_pass", "liquidity_new_pass",
        "full_mcap_usd", "ff_mcap_usd", "equity_universe_min_size_usd",
        "restricted_fol_reference", "restricted_fol_watch",
        "selection_existing_standard", "selection_preliminary_pass",
        "selection_restricted_fol_watch", "selection_near_boundary",
        "near_standard_boundary", "boundary_ratio", "selection_reason",
        "needs_historical_foreign_ownership", "needs_historical_fol",
        "needs_foreign_room_check", "needs_india_red_flag_breach_check",
        "current_fol_reference", "current_fol_reference_raw",
        "current_fol_reference_field", "current_fol_source",
        "current_fol_asof_date", "current_fol_is_reference_only",
    ]

    out = x[[c for c in keep if c in x.columns]].copy()
    out = out.sort_values(["review", "selection_existing_standard", "selection_preliminary_pass", "security_id"], ascending=[True, False, False, True]).reset_index(drop=True)

    if out.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate output review/security_id rows.")
    if out["security_id"].isna().any(): raise RuntimeError("Missing security_id.")

    expected_existing = int(pd.read_parquet(IN_PATH)["was_standard_constituent"].fillna(False).sum())
    selected_existing = int(out["selection_existing_standard"].sum())
    if expected_existing != selected_existing: raise RuntimeError(f"Existing Standard coverage mismatch: {selected_existing}/{expected_existing}")

    diag = out.groupby("review").agg(
        rows=("security_id", "size"),
        existing_standard=("selection_existing_standard", "sum"),
        new_screen_pass=("selection_preliminary_pass", "sum"),
        restricted_fol_watch=("selection_restricted_fol_watch", "sum"),
        current_fol_available=("current_fol_reference", lambda s: int(s.notna().sum())),
        restricted_current_fol=("restricted_fol_reference", "sum"),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- FOREIGN ROOM CANDIDATES ---")
    print(diag.to_string())
    print("\nRows:", len(out))
    print("Unique securities:", out["security_id"].nunique())
    print("Existing Standard coverage:", f"{selected_existing}/{expected_existing}")
    print("Duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("Current FOL used as historical:", int((~out["current_fol_is_reference_only"]).sum()))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()