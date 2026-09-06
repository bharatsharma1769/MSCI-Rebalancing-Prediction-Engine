from pathlib import Path
import numpy as np
import pandas as pd


STAGE_PATH = Path("data/processed/historical_fol_policy_stage.parquet")
OUT_PATH = Path("data/processed/historical_fol.parquet")

AGARWALEYE_ID = "SEC000075"
AGARWALEYE_DATE = pd.Timestamp("2024-09-30")


def prepare_stage(x):
    x = x.copy()
    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")

    for c in [
        "foreign_ownership_cutoff_date",
        "historical_fol_effective_date",
        "historical_fol_available_date",
    ]:
        x[c] = pd.to_datetime(x[c], errors="coerce")

    x["historical_fol"] = pd.to_numeric(x["historical_fol"], errors="coerce")
    x["historical_fol_lower_bound"] = pd.to_numeric(
        x["historical_fol_lower_bound"], errors="coerce"
    )

    for c in [
        "historical_fol_source",
        "historical_fol_provenance",
        "historical_fol_status",
        "historical_fol_reason",
        "evidence_action",
        "historical_fol_lower_bound_basis",
    ]:
        if c not in x.columns:
            x[c] = pd.NA

    # Exact company evidence previously established.
    special = (
        x["security_id"].eq(AGARWALEYE_ID)
        & x["foreign_ownership_cutoff_date"].ge(AGARWALEYE_DATE)
    )

    x.loc[special, "historical_fol"] = 1.00
    x.loc[special, "historical_fol_effective_date"] = AGARWALEYE_DATE
    x.loc[special, "historical_fol_available_date"] = AGARWALEYE_DATE
    x.loc[special, "historical_fol_source"] = "COMPANY_SHAREHOLDING_PATTERN"
    x.loc[special, "historical_fol_provenance"] = "EXACT_PIT"
    x.loc[special, "historical_fol_status"] = "RESOLVED_SPECIAL_CASE"
    x.loc[special, "historical_fol_reason"] = "AGARWALEYE_APPROVED_LIMIT_100"
    x.loc[special, "evidence_action"] = "USE_AGARWALEYE_EXACT_PIT"
    x.loc[special, "historical_fol_lower_bound"] = 1.00
    x.loc[special, "historical_fol_lower_bound_basis"] = "EXACT_HISTORICAL_FOL"

    resolved = x["historical_fol"].notna()

    x["historical_fol_lower_bound_available_date"] = pd.NaT
    x["historical_fol_lower_bound_source"] = pd.Series(pd.NA, index=x.index, dtype="string")

    x.loc[resolved, "historical_fol_lower_bound"] = x.loc[resolved, "historical_fol"]
    x.loc[resolved, "historical_fol_lower_bound_basis"] = "RESOLVED_HISTORICAL_FOL"
    x.loc[resolved, "historical_fol_lower_bound_available_date"] = x.loc[
        resolved, "historical_fol_available_date"
    ]
    x.loc[resolved, "historical_fol_lower_bound_source"] = x.loc[
        resolved, "historical_fol_source"
    ].astype("string")

    floor24 = (
        ~resolved
        & x["historical_fol_lower_bound"].eq(0.24)
        & x["historical_fol_lower_bound_basis"].astype("string").str.contains(
            "NDI_2020", na=False
        )
    )

    x.loc[floor24, "historical_fol_lower_bound_available_date"] = pd.Timestamp("2020-04-01")
    x.loc[floor24, "historical_fol_lower_bound_source"] = "FEMA_NDI_SCHEDULE_II"

    source = x["historical_fol_source"].astype("string").fillna("")
    prov = x["historical_fol_provenance"].astype("string").fillna("")
    status = x["historical_fol_status"].astype("string").fillna("")

    x["historical_fol_resolution_type"] = "UNRESOLVED_NO_BOUND"

    bound_only = ~resolved & x["historical_fol_lower_bound"].notna()
    x.loc[bound_only, "historical_fol_resolution_type"] = "POLICY_LOWER_BOUND_ONLY"

    x.loc[resolved, "historical_fol_resolution_type"] = "RESOLVED_OTHER"

    policy_recon = resolved & (
        prov.str.contains("POLICY_RECONSTRUCTED", case=False)
        | source.str.contains("DPIIT_POLICY_BASELINE", case=False)
    )
    x.loc[policy_recon, "historical_fol_resolution_type"] = "POLICY_RECONSTRUCTED"

    policy_exact = resolved & prov.str.contains("POLICY_EXACT", case=False)
    x.loc[policy_exact, "historical_fol_resolution_type"] = "POLICY_EXACT"

    event = resolved & (
        source.str.contains("NSE_ANNOUNCEMENT", case=False)
        | status.str.contains("EVENT", case=False)
    )
    x.loc[event, "historical_fol_resolution_type"] = "EVENT_PIT"

    exact = resolved & (
        prov.str.contains("EXACT_PIT", case=False)
        | source.str.contains("COMPANY_SHAREHOLDING_PATTERN", case=False)
    )
    x.loc[exact, "historical_fol_resolution_type"] = "EXACT_PIT"

    x["historical_fol_resolved"] = resolved
    x["historical_fol_is_exact_evidence"] = x[
        "historical_fol_resolution_type"
    ].isin(["EXACT_PIT", "EVENT_PIT", "POLICY_EXACT"])

    return x, special


def prepare_legacy_extras(old, candidate_keys):
    if old.empty:
        return old

    old = old.copy()
    old["review"] = old["review"].astype(str)
    old["security_id"] = old["security_id"].astype("string")

    idx = pd.MultiIndex.from_frame(old[["review", "security_id"]])
    extras = old.loc[~idx.isin(candidate_keys)].copy()

    if extras.empty:
        return extras

    if "historical_fol" in extras.columns:
        extras["historical_fol"] = pd.to_numeric(extras["historical_fol"], errors="coerce")

    for c in ["historical_fol_effective_date", "historical_fol_available_date"]:
        if c in extras.columns:
            extras[c] = pd.to_datetime(extras[c], errors="coerce")

    if "historical_fol_lower_bound" not in extras.columns:
        extras["historical_fol_lower_bound"] = extras["historical_fol"]

    if "historical_fol_lower_bound_basis" not in extras.columns:
        extras["historical_fol_lower_bound_basis"] = np.where(
            extras["historical_fol"].notna(),
            "PRESERVED_LEGACY_RESOLVED",
            pd.NA,
        )

    if "historical_fol_resolution_type" not in extras.columns:
        extras["historical_fol_resolution_type"] = np.where(
            extras["historical_fol"].notna(),
            "PRESERVED_LEGACY",
            "UNRESOLVED_NO_BOUND",
        )

    extras["historical_fol_resolved"] = extras["historical_fol"].notna()

    if "historical_fol_is_exact_evidence" not in extras.columns:
        extras["historical_fol_is_exact_evidence"] = False

    return extras


def main():
    if not STAGE_PATH.exists():
        raise FileNotFoundError(STAGE_PATH)

    stage = pd.read_parquet(STAGE_PATH)

    if stage.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate policy-stage keys.")

    old = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    stage, special = prepare_stage(stage)

    candidate_keys = pd.MultiIndex.from_frame(stage[["review", "security_id"]])
    extras = prepare_legacy_extras(old, candidate_keys)

    keep = [
        "review", "security_id", "nse_symbol", "company_name",
        "foreign_ownership_cutoff_date",
        "historical_fol",
        "historical_fol_effective_date",
        "historical_fol_available_date",
        "historical_fol_source",
        "historical_fol_provenance",
        "historical_fol_status",
        "historical_fol_reason",
        "evidence_action",
        "historical_fol_resolution_type",
        "historical_fol_resolved",
        "historical_fol_is_exact_evidence",
        "historical_fol_lower_bound",
        "historical_fol_lower_bound_basis",
        "historical_fol_lower_bound_available_date",
        "historical_fol_lower_bound_source",
    ]

    current = stage[[c for c in keep if c in stage.columns]].copy()

    if not extras.empty:
        for c in keep:
            if c not in extras.columns:
                extras[c] = pd.NA
        extras = extras[keep]
        out = pd.concat([current, extras], ignore_index=True, sort=False)
    else:
        out = current

    out["security_id"] = out["security_id"].astype("string")
    out["historical_fol"] = pd.to_numeric(out["historical_fol"], errors="coerce")
    out["historical_fol_lower_bound"] = pd.to_numeric(
        out["historical_fol_lower_bound"], errors="coerce"
    )

    for c in [
        "foreign_ownership_cutoff_date",
        "historical_fol_effective_date",
        "historical_fol_available_date",
        "historical_fol_lower_bound_available_date",
    ]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    resolved = out["historical_fol"].notna()
    invalid = resolved & ~out["historical_fol"].between(0, 1)
    invalid_lb = out["historical_fol_lower_bound"].notna() & ~out[
        "historical_fol_lower_bound"
    ].between(0, 1)

    if invalid.any():
        raise RuntimeError(f"Invalid historical FOL rows: {int(invalid.sum())}")

    if invalid_lb.any():
        raise RuntimeError(f"Invalid historical FOL lower bounds: {int(invalid_lb.sum())}")

    candidate = out.set_index(["review", "security_id"]).loc[candidate_keys].reset_index()

    missing_date = (
        candidate["historical_fol"].notna()
        & candidate["historical_fol_available_date"].isna()
    )

    lookahead = (
        candidate["historical_fol_available_date"].notna()
        & candidate["foreign_ownership_cutoff_date"].notna()
        & candidate["historical_fol_available_date"].gt(
            candidate["foreign_ownership_cutoff_date"]
        )
    )

    lb_lookahead = (
        candidate["historical_fol_lower_bound_available_date"].notna()
        & candidate["foreign_ownership_cutoff_date"].notna()
        & candidate["historical_fol_lower_bound_available_date"].gt(
            candidate["foreign_ownership_cutoff_date"]
        )
    )

    if missing_date.any():
        raise RuntimeError(
            f"Resolved candidate FOL rows missing availability date: {int(missing_date.sum())}"
        )

    if lookahead.any() or lb_lookahead.any():
        raise RuntimeError("Historical FOL PIT lookahead detected.")

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate canonical historical FOL keys.")

    out = out.sort_values(["review", "security_id"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    candidate = out.merge(
        stage[["review", "security_id"]],
        on=["review", "security_id"],
        how="inner",
        validate="one_to_one",
    )

    print("\n--- CANONICAL HISTORICAL FOL REBUILD ---")
    print("Candidate keys:", len(stage))
    print("Candidate key coverage:", len(candidate))
    print("Candidate resolved FOL rows:", int(candidate["historical_fol"].notna().sum()))
    print("Candidate resolved securities:", candidate.loc[
        candidate["historical_fol"].notna(), "security_id"
    ].nunique())
    print("Candidate exact-evidence rows:", int(
        candidate["historical_fol_is_exact_evidence"].fillna(False).sum()
    ))
    print("Candidate bound-only rows:", int(
        (
            candidate["historical_fol"].isna()
            & candidate["historical_fol_lower_bound"].notna()
        ).sum()
    ))
    print("Candidate rows with no FOL bound:", int(
        (
            candidate["historical_fol"].isna()
            & candidate["historical_fol_lower_bound"].isna()
        ).sum()
    ))
    print("AGARWALEYE exact rows applied:", int(special.sum()))
    print("Legacy extra rows preserved:", len(extras))

    print("\nResolution type:")
    print(
        candidate["historical_fol_resolution_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nResolved provenance:")
    print(
        candidate.loc[
            candidate["historical_fol"].notna(),
            "historical_fol_provenance",
        ]
        .astype("string")
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nResolved rows missing availability date:", int(missing_date.sum()))
    print("FOL lookahead:", int(lookahead.sum()))
    print("Lower-bound lookahead:", int(lb_lookahead.sum()))
    print("Canonical duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("Canonical rows:", len(out))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()