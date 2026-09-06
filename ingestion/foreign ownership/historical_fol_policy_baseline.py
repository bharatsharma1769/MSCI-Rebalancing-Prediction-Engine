from pathlib import Path
import numpy as np
import pandas as pd


TARGET_PATH = Path("data/processed/historical_fol_targets.parquet")
OLD_FOL_PATH = Path("data/processed/historical_fol.parquet")
ANN_PATH = Path("data/processed/historical_fol_announcement_stage.parquet")
FOL_PATH = Path("data/processed/fol_master.parquet")
OUT_PATH = Path("data/processed/historical_fol_policy_stage.parquet")

ROOM_PASS = 0.15
ROOM_NO_ADJUST = 0.25
TOL = 1e-9

PSB_20 = {
    "SEC000272",  # BANKBARODA
    "SEC000273",  # BANKINDIA
    "SEC000373",  # CANBK
    "SEC000400",  # CENTRALBK
    "SEC000912",  # INDIANB
    "SEC000953",  # IOB
    "SEC001217",  # MAHABANK
    "SEC001546",  # PNB
    "SEC001595",  # PSB
    "SEC001773",  # SBIN
    "SEC002120",  # UCOBANK
    "SEC002138",  # UNIONBANK
}

LIC_ID = "SEC001180"


def first_existing(df, names):
    return next((c for c in names if c in df.columns), None)


def prepare_old(old):
    fol = first_existing(old, ["historical_fol"])
    avail = first_existing(old, ["historical_fol_available_date", "available_date"])
    effective = first_existing(old, ["historical_fol_effective_date", "snapshot_date", "announcement_date"])
    source = first_existing(old, ["historical_fol_source", "source"])
    provenance = first_existing(old, ["historical_fol_provenance", "provenance"])
    status = first_existing(old, ["historical_fol_status", "status"])

    if fol is None:
        raise RuntimeError("Old historical_fol.parquet missing historical_fol.")

    x = old[["review", "security_id"]].copy()
    x["historical_fol"] = pd.to_numeric(old[fol], errors="coerce")
    x["historical_fol_available_date"] = pd.to_datetime(old[avail], errors="coerce") if avail else pd.NaT
    x["historical_fol_effective_date"] = pd.to_datetime(old[effective], errors="coerce") if effective else pd.NaT
    x["historical_fol_source"] = old[source] if source else pd.NA
    x["historical_fol_provenance"] = old[provenance] if provenance else pd.NA
    x["historical_fol_status"] = old[status] if status else pd.NA
    x["evidence_action"] = "PRESERVE_OLD_EXACT"
    x["evidence_priority"] = 1
    return x[x["historical_fol"].notna()]


def prepare_ann(ann):
    cols = [
        "review", "security_id", "historical_fol",
        "historical_fol_available_date", "historical_fol_effective_date",
        "historical_fol_source", "historical_fol_provenance",
        "historical_fol_status", "stage_action",
    ]

    x = ann[[c for c in cols if c in ann.columns]].copy()
    x["historical_fol"] = pd.to_numeric(x["historical_fol"], errors="coerce")

    for c in ["historical_fol_available_date", "historical_fol_effective_date"]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")

    x = x[x["historical_fol"].notna()].copy()
    x["evidence_action"] = np.where(
        x.get("stage_action", "").astype(str).eq("USE_ANNOUNCEMENT_EVENT"),
        "USE_ANNOUNCEMENT_EVENT",
        "PRESERVE_ANNOUNCEMENT_STAGE_EXACT",
    )
    x["evidence_priority"] = 2
    return x.drop(columns="stage_action", errors="ignore")


def main():
    targets = pd.read_parquet(TARGET_PATH)
    old = pd.read_parquet(OLD_FOL_PATH)
    ann = pd.read_parquet(ANN_PATH)
    fol = pd.read_parquet(FOL_PATH)

    for df in [targets, old, ann, fol]:
        df["security_id"] = df["security_id"].astype("string")

    for df in [targets, old, ann]:
        df["review"] = df["review"].astype(str)

    for name, df in [("targets", targets), ("old FOL", old), ("announcement stage", ann)]:
        if df.duplicated(["review", "security_id"]).any():
            raise RuntimeError(f"Duplicate {name} keys.")

    if fol["security_id"].duplicated().any():
        raise RuntimeError("fol_master has duplicate security_id rows.")

    targets["foreign_ownership"] = pd.to_numeric(targets["foreign_ownership"], errors="coerce")
    targets["foreign_ownership_cutoff_date"] = pd.to_datetime(
        targets["foreign_ownership_cutoff_date"], errors="coerce"
    )

    old_exact = prepare_old(old)
    ann_exact = prepare_ann(ann)

    evidence = pd.concat([old_exact, ann_exact], ignore_index=True, sort=False)
    evidence = (
        evidence.sort_values(["review", "security_id", "evidence_priority"])
        .drop_duplicates(["review", "security_id"], keep="last")
        .drop(columns="evidence_priority")
    )

    x = targets.merge(
        evidence,
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    fcols = [
        "security_id", "fol", "fpi_limit", "sectoral_cap",
        "govt_approved_limit", "fol_source", "fol_snapshot_date",
    ]
    f = fol[[c for c in fcols if c in fol.columns]].copy()

    rename = {
        "fol": "current_fol_snapshot",
        "fpi_limit": "current_fpi_limit",
        "sectoral_cap": "current_sectoral_cap",
        "govt_approved_limit": "current_govt_approved_limit",
        "fol_source": "current_fol_source",
        "fol_snapshot_date": "current_fol_snapshot_date",
    }
    f = f.rename(columns=rename)

    for c in [
        "current_fol_snapshot", "current_fpi_limit",
        "current_sectoral_cap", "current_govt_approved_limit",
    ]:
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")

    x = x.merge(f, on="security_id", how="left", validate="many_to_one")

    for c in [
        "historical_fol_available_date",
        "historical_fol_effective_date",
        "current_fol_snapshot_date",
    ]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")

    # Exact policy cases: public-sector banks and LIC.
    psb = (
        x["historical_fol"].isna()
        & x["security_id"].isin(PSB_20)
    )

    x.loc[psb, "historical_fol"] = 0.20
    x.loc[psb, "historical_fol_effective_date"] = pd.Timestamp("2020-04-01")
    x.loc[psb, "historical_fol_available_date"] = pd.Timestamp("2020-04-01")
    x.loc[psb, "historical_fol_source"] = "INDIA_STATUTORY_POLICY"
    x.loc[psb, "historical_fol_provenance"] = "POLICY_EXACT"
    x.loc[psb, "historical_fol_status"] = "PUBLIC_SECTOR_BANK_20"
    x.loc[psb, "evidence_action"] = "USE_EXACT_POLICY_20"

    lic = (
        x["historical_fol"].isna()
        & x["security_id"].eq(LIC_ID)
        & x["foreign_ownership_cutoff_date"].ge(pd.Timestamp("2022-03-14"))
    )

    x.loc[lic, "historical_fol"] = 0.20
    x.loc[lic, "historical_fol_effective_date"] = pd.Timestamp("2022-03-14")
    x.loc[lic, "historical_fol_available_date"] = pd.Timestamp("2022-03-14")
    x.loc[lic, "historical_fol_source"] = "INDIA_STATUTORY_POLICY"
    x.loc[lic, "historical_fol_provenance"] = "POLICY_EXACT"
    x.loc[lic, "historical_fol_status"] = "LIC_20"
    x.loc[lic, "evidence_action"] = "USE_EXACT_POLICY_20"

    exact = x["historical_fol"].notna()

    x["historical_fol_lower_bound"] = np.nan
    x["historical_fol_lower_bound_basis"] = pd.Series(pd.NA, index=x.index, dtype="string")

    x.loc[exact, "historical_fol_lower_bound"] = x.loc[exact, "historical_fol"]
    x.loc[exact, "historical_fol_lower_bound_basis"] = "EXACT_HISTORICAL_FOL"

    # Do NOT write 24% into historical_fol.
    # It is only a conservative post-April-2020 screening floor.
    floor24 = (
        ~exact
        & x["current_fpi_limit"].notna()
        & x["current_fpi_limit"].ge(0.24 - TOL)
    )

    x.loc[floor24, "historical_fol_lower_bound"] = 0.24
    x.loc[floor24, "historical_fol_lower_bound_basis"] = "NDI_2020_CONSERVATIVE_FPI_FLOOR_24"

    fo = x["foreign_ownership"]
    lb = x["historical_fol_lower_bound"]

    valid = fo.notna() & lb.notna() & lb.gt(0)

    x["foreign_room_lower_bound"] = np.where(
        valid,
        (lb - fo) / lb,
        np.nan,
    )

    x["foreign_room_exact"] = np.where(
        exact & fo.notna() & x["historical_fol"].gt(0),
        (x["historical_fol"] - fo) / x["historical_fol"],
        np.nan,
    )

    x["room_ge_15_proven"] = (
        x["foreign_room_lower_bound"].notna()
        & x["foreign_room_lower_bound"].ge(ROOM_PASS - TOL)
    )

    x["room_ge_25_proven"] = (
        x["foreign_room_lower_bound"].notna()
        & x["foreign_room_lower_bound"].ge(ROOM_NO_ADJUST - TOL)
    )

    x["policy_screen_status"] = "NO_POLICY_LOWER_BOUND"

    x.loc[
        ~exact & lb.notna() & ~x["room_ge_15_proven"],
        "policy_screen_status",
    ] = "LOWER_BOUND_LT_15_NEEDS_EXACT"

    x.loc[
        ~exact & x["room_ge_15_proven"] & ~x["room_ge_25_proven"],
        "policy_screen_status",
    ] = "PROVEN_GE_15_NOT_GE_25"

    x.loc[
        ~exact & x["room_ge_25_proven"],
        "policy_screen_status",
    ] = "PROVEN_GE_25"

    x.loc[exact, "policy_screen_status"] = "EXACT_FOL"

    x.loc[
        fo.isna(),
        "policy_screen_status",
    ] = "FO_UNRESOLVED"

    x["needs_exact_fol_for_new_room_test"] = (
        fo.notna()
        & ~exact
        & ~x["room_ge_15_proven"]
    )

    x["needs_exact_fol_for_full_room_band"] = (
        fo.notna()
        & ~exact
        & ~x["room_ge_25_proven"]
    )

    lookahead = (
        exact
        & x["historical_fol_available_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["historical_fol_available_date"].gt(
            x["foreign_ownership_cutoff_date"]
        )
    )

    if lookahead.any():
        raise RuntimeError(
            f"Historical FOL lookahead detected: {int(lookahead.sum())}"
        )

    if x.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate policy-stage keys.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(OUT_PATH, index=False)

    diag = x.groupby("review").agg(
        rows=("security_id", "size"),
        fo_resolved=("foreign_ownership", lambda s: int(s.notna().sum())),
        exact_fol=("historical_fol", lambda s: int(s.notna().sum())),
        proven_ge15=("room_ge_15_proven", "sum"),
        proven_ge25=("room_ge_25_proven", "sum"),
        need_exact_new=("needs_exact_fol_for_new_room_test", "sum"),
        need_exact_band=("needs_exact_fol_for_full_room_band", "sum"),
    )

    original_research = x[
        x["fol_research_priority"].isin(["CRITICAL", "HIGH"])
    ]

    print("\n--- HISTORICAL FOL POLICY SCREEN ---")
    print(diag.to_string())

    print("\nRows:", len(x))
    print("FO resolved:", int(fo.notna().sum()))
    print("Exact historical FOL rows:", int(exact.sum()))
    print("Exact historical FOL securities:", x.loc[exact, "security_id"].nunique())
    print("Exact 20% policy rows added:", int((psb | lic).sum()))
    print("24% conservative-floor rows:", int(floor24.sum()))
    print("Rows proven room >=15%:", int(x["room_ge_15_proven"].sum()))
    print("Rows proven room >=25%:", int(x["room_ge_25_proven"].sum()))
    print("Rows proven 15%-25% only:", int((x["room_ge_15_proven"] & ~x["room_ge_25_proven"]).sum()))
    print("Rows needing exact FOL for new-room test:", int(x["needs_exact_fol_for_new_room_test"].sum()))
    print("Securities needing exact FOL for new-room test:", x.loc[x["needs_exact_fol_for_new_room_test"], "security_id"].nunique())
    print("Rows needing exact FOL for full room band:", int(x["needs_exact_fol_for_full_room_band"].sum()))
    print("Securities needing exact FOL for full room band:", x.loc[x["needs_exact_fol_for_full_room_band"], "security_id"].nunique())
    print("Rows with no policy lower bound:", int((x["policy_screen_status"] == "NO_POLICY_LOWER_BOUND").sum()))

    print("\nOriginal HIGH/CRITICAL research rows:", len(original_research))
    print(
        "Original research rows still needing exact FOL for 15% test:",
        int(original_research["needs_exact_fol_for_new_room_test"].sum()),
    )
    print(
        "Original research securities still needing exact FOL for 15% test:",
        original_research.loc[
            original_research["needs_exact_fol_for_new_room_test"],
            "security_id",
        ].nunique(),
    )

    print("\nPolicy screen status:")
    print(x["policy_screen_status"].value_counts(dropna=False).to_string())

    print("\nPIT lookahead:", int(lookahead.sum()))
    print("Duplicates:", int(x.duplicated(["review", "security_id"]).sum()))
    print("Saved:", OUT_PATH)
    print("Canonical historical_fol.parquet was NOT modified.")


if __name__ == "__main__":
    main()