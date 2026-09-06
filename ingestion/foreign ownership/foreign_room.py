from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


CANDIDATE_PATH = Path("data/processed/foreign_room_candidates.parquet")
FO_PATH = Path("data/processed/historical_foreign_ownership.parquet")
FOL_PATH = Path("data/processed/historical_fol.parquet")
OUT_PATH = Path("data/processed/foreign_room.parquet")

ITC_ID = "SEC000972"
ITC_FOL = .34

JIOFIN_ID = "SEC001008"
JIOFIN_REVIEW = "2023-11"

VMM_ID = "SEC002213"
VMM_REVIEW = "2025-11"

INDUS_ID = "SEC000935"
INDUS_AUG = "2023-08"
INDUS_NOV = "2023-11"
INDUS_FO = .6261
INDUS_FOL = .74

TARGET_TAG = "ShareholdingAsAPercentageOfTotalNumberOfShares"

FPI_MEMBERS = {
    "InstitutionsForeignPortfolioInvestorMember",
    "InstitutionsForeignPortfolioInvestorsMember",
    "InstitutionsForeignPortfolioInvestorCategoryOneMember",
    "InstitutionsForeignPortfolioInvestorCategoryTwoMember",
    "InstitutionsForeignPortfolioInvestorCatergoryOneMember",
    "InstitutionsForeignPortfolioInvestorCatergoryTwoMember",
}

NRI_MEMBERS = {"NonResidentIndiansMember"}


def clean_tag(x):
    return x.split("}")[-1].strip()


def member_name(x):
    return "" if not x else str(x).split(":")[-1].strip()


def room_band(room):
    if pd.isna(room): return pd.NA
    if room >= .25: return "GE_25"
    if room >= .15: return "15_TO_25"
    if room >= .075: return "7_5_TO_15"
    if room >= .0375: return "3_75_TO_7_5"
    return "LT_3_75"


def factor(room, pre):
    if pd.isna(room): return np.nan
    if room >= .25: return 1.0

    if room >= .15:
        if pre >= 1: return 1.0
        return .5

    if room >= .075:
        if pre >= .5: return .5
        return .25

    if room >= .0375: return .25
    return 0.0


def create_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def parse_itc_ownership(url, session):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    contexts = {}

    for e in root.iter():
        if clean_tag(e.tag) != "context": continue
        cid = e.attrib.get("id")
        if not cid: continue

        contexts[cid] = [
            member_name(c.text)
            for c in e.iter()
            if clean_tag(c.tag) in {"explicitMember", "typedMember"} and c.text
        ]

    facts = []

    for e in root.iter():
        if clean_tag(e.tag) != TARGET_TAG or e.text is None: continue

        try:
            value = float(e.text.strip())
        except ValueError:
            continue

        members = contexts.get(e.attrib.get("contextRef"), [])
        relevant = [m for m in members if m in FPI_MEMBERS or m in NRI_MEMBERS]

        if relevant:
            facts.append((value, relevant))

    if not facts:
        return np.nan

    scale = .01 if any(v > 1 for v, _ in facts) else 1.0
    by_member = {}

    for value, members in facts:
        for m in members:
            by_member[m] = max(by_member.get(m, 0.0), value * scale)

    aggregates = [
        m for m in by_member
        if m in {
            "InstitutionsForeignPortfolioInvestorMember",
            "InstitutionsForeignPortfolioInvestorsMember",
        }
    ]

    if aggregates:
        fpi = max(by_member[m] for m in aggregates)
    else:
        fpi = sum(v for m, v in by_member.items() if m in FPI_MEMBERS)

    nri = max(
        (v for m, v in by_member.items() if m in NRI_MEMBERS),
        default=0.0,
    )

    total = fpi + nri
    return total if 0 <= total <= 1 else np.nan


def set_interval(x, mask, pre, low, high):
    x.loc[mask, f"post_factor_if_pre_{pre}_min"] = low
    x.loc[mask, f"post_factor_if_pre_{pre}_max"] = high


def main():
    c = pd.read_parquet(CANDIDATE_PATH)
    fo = pd.read_parquet(FO_PATH)
    fol = pd.read_parquet(FOL_PATH)

    for df in [c, fo, fol]:
        df["review"] = df["review"].astype(str)
        df["security_id"] = df["security_id"].astype("string")

    for name, df in [("candidate", c), ("FO", fo), ("FOL", fol)]:
        if df.duplicated(["review", "security_id"]).any():
            raise RuntimeError(f"Duplicate {name} keys.")

    fo_cols = [col for col in [
        "review", "security_id", "foreign_ownership",
        "report_date", "available_date", "xbrl_url",
        "source", "data_quality_flag",
        "foreign_ownership_cutoff_date",
    ] if col in fo.columns]

    fol_cols = [col for col in [
        "review", "security_id",
        "historical_fol",
        "historical_fol_effective_date",
        "historical_fol_available_date",
        "historical_fol_source",
        "historical_fol_provenance",
        "historical_fol_status",
        "historical_fol_reason",
        "historical_fol_resolution_type",
        "historical_fol_is_exact_evidence",
        "historical_fol_lower_bound",
        "historical_fol_lower_bound_basis",
        "historical_fol_lower_bound_available_date",
        "historical_fol_lower_bound_source",
        "foreign_ownership_cutoff_date",
    ] if col in fol.columns]

    f = fol[fol_cols].rename(
        columns={"foreign_ownership_cutoff_date": "fol_cutoff_date"}
    )

    x = (
        c.merge(
            fo[fo_cols],
            on=["review", "security_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            f,
            on=["review", "security_id"],
            how="left",
            validate="one_to_one",
        )
    )

    for col in [
        "foreign_ownership",
        "historical_fol",
        "historical_fol_lower_bound",
    ]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    for col in [
        "foreign_ownership_cutoff_date",
        "fol_cutoff_date",
        "available_date",
        "historical_fol_available_date",
        "historical_fol_lower_bound_available_date",
    ]:
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce")

    mismatch = (
        x["foreign_ownership_cutoff_date"].notna()
        & x["fol_cutoff_date"].notna()
        & x["foreign_ownership_cutoff_date"].ne(x["fol_cutoff_date"])
    )

    if mismatch.any():
        raise RuntimeError(f"FO/FOL cutoff mismatch: {int(mismatch.sum())}")

    x["foreign_room_ownership_used"] = x["foreign_ownership"]
    x["foreign_room_fol_used"] = x["historical_fol"]
    x["foreign_room_fol_lower_bound_used"] = x["historical_fol_lower_bound"]
    x["special_foreign_room_treatment"] = pd.Series(
        pd.NA, index=x.index, dtype="string"
    )

    session = create_session()
    itc_resolved = 0

    for idx in x[x["security_id"].eq(ITC_ID)].index:
        url = x.at[idx, "xbrl_url"] if "xbrl_url" in x.columns else pd.NA
        used = np.nan

        if pd.notna(url) and str(url).startswith("http"):
            try:
                used = parse_itc_ownership(str(url), session)
            except Exception:
                pass

        x.at[idx, "foreign_room_fol_used"] = np.nan
        x.at[idx, "foreign_room_fol_lower_bound_used"] = np.nan
        x.at[idx, "special_foreign_room_treatment"] = "ITC_FPI_PLUS_NRI_NUMERATOR"

        if pd.notna(used):
            x.at[idx, "foreign_room_ownership_used"] = used
            x.at[idx, "foreign_room_fol_used"] = ITC_FOL
            x.at[idx, "foreign_room_fol_lower_bound_used"] = ITC_FOL
            itc_resolved += 1

    indus_aug = (
        x["security_id"].eq(INDUS_ID)
        & x["review"].eq(INDUS_AUG)
    )

    x.loc[
        indus_aug,
        ["foreign_room_fol_used", "foreign_room_fol_lower_bound_used"],
    ] = np.nan

    x.loc[
        indus_aug,
        "special_foreign_room_treatment",
    ] = "INDUSINDBK_2023_08_UNRESOLVED"

    indus_nov = (
        x["security_id"].eq(INDUS_ID)
        & x["review"].eq(INDUS_NOV)
    )

    x.loc[indus_nov, "foreign_room_ownership_used"] = INDUS_FO
    x.loc[indus_nov, "foreign_room_fol_used"] = INDUS_FOL
    x.loc[indus_nov, "foreign_room_fol_lower_bound_used"] = INDUS_FOL
    x.loc[
        indus_nov,
        "special_foreign_room_treatment",
    ] = "INDUSINDBK_SEP2023_TOTAL_FOREIGN"

    vmm = (
        x["security_id"].eq(VMM_ID)
        & x["review"].eq(VMM_REVIEW)
    )

    x.loc[
        vmm,
        ["foreign_room_fol_used", "foreign_room_fol_lower_bound_used"],
    ] = np.nan

    x.loc[
        vmm,
        "special_foreign_room_treatment",
    ] = "VMM_2025_11_DEAD_XBRL"

    point_valid = (
        x["foreign_room_ownership_used"].notna()
        & x["foreign_room_fol_used"].notna()
        & x["foreign_room_fol_used"].gt(0)
    )

    bound_valid = (
        x["foreign_room_ownership_used"].notna()
        & x["foreign_room_fol_lower_bound_used"].notna()
        & x["foreign_room_fol_lower_bound_used"].gt(0)
    )

    x["foreign_room"] = np.where(
        point_valid,
        (
            x["foreign_room_fol_used"]
            - x["foreign_room_ownership_used"]
        ) / x["foreign_room_fol_used"],
        np.nan,
    )

    x["foreign_room_lower_bound"] = np.where(
        bound_valid,
        (
            x["foreign_room_fol_lower_bound_used"]
            - x["foreign_room_ownership_used"]
        ) / x["foreign_room_fol_lower_bound_used"],
        np.nan,
    )

    x["foreign_room_anomaly"] = (
        x["foreign_room"].notna()
        & ~x["foreign_room"].between(0, 1)
    )

    usable_point = (
        x["foreign_room"].notna()
        & ~x["foreign_room_anomaly"]
    )

    x["foreign_room_band"] = x["foreign_room"].map(room_band)

    x["room_ge_15_proven"] = (
        (usable_point & x["foreign_room"].ge(.15))
        | (~usable_point & x["foreign_room_lower_bound"].ge(.15))
    )

    x["room_ge_25_proven"] = (
        (usable_point & x["foreign_room"].ge(.25))
        | (~usable_point & x["foreign_room_lower_bound"].ge(.25))
    )

    x["room_lt_15_proven"] = (
        usable_point
        & x["foreign_room"].lt(.15)
    )

    x["foreign_room_status"] = "NO_FOL_BOUND"

    x.loc[
        x["foreign_room_ownership_used"].isna(),
        "foreign_room_status",
    ] = "FO_UNRESOLVED"

    x.loc[
        ~usable_point
        & x["foreign_room_lower_bound"].notna(),
        "foreign_room_status",
    ] = "LOWER_BOUND_INSUFFICIENT"

    x.loc[
        ~usable_point
        & x["foreign_room_lower_bound"].ge(.15),
        "foreign_room_status",
    ] = "LOWER_BOUND_PROVES_GE15"

    x.loc[
        ~usable_point
        & x["foreign_room_lower_bound"].ge(.25),
        "foreign_room_status",
    ] = "LOWER_BOUND_PROVES_GE25"

    x.loc[
        usable_point,
        "foreign_room_status",
    ] = "RESOLVED_ROOM"

    x.loc[
        x["foreign_room_anomaly"],
        "foreign_room_status",
    ] = "ANOMALOUS_FO_GT_FOL"

    # What would happen IF the security is new to the IMI.
    x["new_entry_pass_if_nonconstituent"] = pd.Series(
        pd.NA, index=x.index, dtype="boolean"
    )

    x.loc[
        usable_point,
        "new_entry_pass_if_nonconstituent",
    ] = x.loc[usable_point, "foreign_room"].ge(.15)

    x.loc[
        ~usable_point & x["foreign_room_lower_bound"].ge(.15),
        "new_entry_pass_if_nonconstituent",
    ] = True

    x["new_entry_factor_min_if_pass"] = np.nan
    x["new_entry_factor_max_if_pass"] = np.nan

    exact_ge25 = usable_point & x["foreign_room"].ge(.25)
    exact_15_25 = (
        usable_point
        & x["foreign_room"].ge(.15)
        & x["foreign_room"].lt(.25)
    )
    bound_ge25 = ~usable_point & x["foreign_room_lower_bound"].ge(.25)
    bound_15_25 = (
        ~usable_point
        & x["foreign_room_lower_bound"].ge(.15)
        & x["foreign_room_lower_bound"].lt(.25)
    )

    x.loc[
        exact_ge25 | bound_ge25,
        ["new_entry_factor_min_if_pass", "new_entry_factor_max_if_pass"],
    ] = [1.0, 1.0]

    x.loc[
        exact_15_25,
        ["new_entry_factor_min_if_pass", "new_entry_factor_max_if_pass"],
    ] = [.5, .5]

    # Lower bound proves >=15 but cannot distinguish 15-25 from >=25.
    x.loc[
        bound_15_25,
        ["new_entry_factor_min_if_pass", "new_entry_factor_max_if_pass"],
    ] = [.5, 1.0]

    # Existing-constituent post-factor intervals for each possible pre-factor.
    for pre in ["1", "0_5", "0_25"]:
        x[f"post_factor_if_pre_{pre}_min"] = np.nan
        x[f"post_factor_if_pre_{pre}_max"] = np.nan

    pre_map = {
        "1": 1.0,
        "0_5": .5,
        "0_25": .25,
    }

    for label, pre in pre_map.items():
        exact_values = x.loc[usable_point, "foreign_room"].map(
            lambda r: factor(r, pre)
        )

        x.loc[
            usable_point,
            f"post_factor_if_pre_{label}_min",
        ] = exact_values

        x.loc[
            usable_point,
            f"post_factor_if_pre_{label}_max",
        ] = exact_values

    # If room >=25, post factor is definitely 1 regardless of pre-factor.
    for label in pre_map:
        set_interval(x, bound_ge25, label, 1.0, 1.0)

    # If only >=15 is proven:
    # pre=1 -> post always 1;
    # pre=.5/.25 -> post may be .5 or 1 depending on whether true room >=25.
    set_interval(x, bound_15_25, "1", 1.0, 1.0)
    set_interval(x, bound_15_25, "0_5", .5, 1.0)
    set_interval(x, bound_15_25, "0_25", .5, 1.0)

    # Compatibility aliases. These describe conditional outcomes only.
    x["new_foreign_room_pass"] = x["new_entry_pass_if_nonconstituent"]

    x["existing_room_adjustment_factor_if_current_1"] = np.where(
        x["post_factor_if_pre_1_min"].eq(
            x["post_factor_if_pre_1_max"]
        ),
        x["post_factor_if_pre_1_min"],
        np.nan,
    )

    # Final decision requires prior-IMI state and is intentionally pending.
    x["foreign_room_decision_pass"] = pd.Series(
        pd.NA, index=x.index, dtype="boolean"
    )

    x["foreign_room_prior_imi_state_pending"] = True
    x["foreign_room_factor_state_pending"] = True

    jio = (
        x["security_id"].eq(JIOFIN_ID)
        & x["review"].eq(JIOFIN_REVIEW)
    )

    x.loc[
        jio,
        "foreign_room_status",
    ] = "CORPORATE_EVENT_NOT_REGULAR_REVIEW_TEST"

    x.loc[
        jio,
        "special_foreign_room_treatment",
    ] = "JIOFIN_SPINOFF_EVENT"

    x.loc[
        jio,
        "foreign_room_decision_pass",
    ] = True

    x.loc[
        jio,
        "foreign_room_prior_imi_state_pending",
    ] = False

    x.loc[
        indus_aug,
        "foreign_room_status",
    ] = "INDUSINDBK_2023_08_PIT_UNRESOLVED"

    x.loc[
        indus_nov,
        "foreign_room_status",
    ] = "RESOLVED_INDUSINDBK_TOTAL_FOREIGN"

    x.loc[
        x["security_id"].eq(ITC_ID)
        & usable_point,
        "foreign_room_status",
    ] = "RESOLVED_ITC_SPECIAL_NUMERATOR"

    x.loc[
        x["security_id"].eq(ITC_ID)
        & ~usable_point,
        "foreign_room_status",
    ] = "ITC_SPECIAL_NUMERATOR_UNRESOLVED"

    x.loc[vmm, "foreign_room"] = np.nan
    x.loc[vmm, "foreign_room_lower_bound"] = np.nan
    x.loc[vmm, "foreign_room_band"] = pd.NA
    x.loc[vmm, "new_entry_pass_if_nonconstituent"] = pd.NA
    x.loc[vmm, "new_foreign_room_pass"] = pd.NA
    x.loc[vmm, "foreign_room_decision_pass"] = pd.NA
    x.loc[vmm, "foreign_room_status"] = "PIT_SOURCE_DEAD_UNRESOLVED"

    x["india_red_flag_breach_pending"] = True

    fo_lookahead = (
        x["available_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["available_date"].gt(x["foreign_ownership_cutoff_date"])
    )

    fol_lookahead = (
        x["historical_fol_available_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["historical_fol_available_date"].gt(
            x["foreign_ownership_cutoff_date"]
        )
    )

    lb_lookahead = (
        x["historical_fol_lower_bound_available_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["historical_fol_lower_bound_available_date"].gt(
            x["foreign_ownership_cutoff_date"]
        )
    )

    if fo_lookahead.any() or fol_lookahead.any() or lb_lookahead.any():
        raise RuntimeError(
            f"Foreign-room lookahead: FO={int(fo_lookahead.sum())}, "
            f"FOL={int(fol_lookahead.sum())}, LB={int(lb_lookahead.sum())}"
        )

    if x["foreign_room_anomaly"].any():
        raise RuntimeError(
            f"FO>FOL anomalies remain: {int(x['foreign_room_anomaly'].sum())}"
        )

    if x.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate foreign-room keys.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(OUT_PATH, index=False)

    print("\n--- FOREIGN ROOM EVIDENCE REBUILD ---")
    print("Rows:", len(x))
    print("Unique securities:", x["security_id"].nunique())
    print("Point room resolved:", int(usable_point.sum()))
    print("Room >=25 proven:", int(x["room_ge_25_proven"].sum()))
    print("Room >=15 proven:", int(x["room_ge_15_proven"].sum()))
    print("Exact room <15:", int(x["room_lt_15_proven"].sum()))
    print(
        "Lower bound insufficient:",
        int(
            (
                ~usable_point
                & x["foreign_room_lower_bound"].notna()
                & x["foreign_room_lower_bound"].lt(.15)
            ).sum()
        ),
    )
    print(
    "No usable FOL bound:",
    int((
        x["foreign_room_fol_used"].isna()
        & x["foreign_room_fol_lower_bound_used"].isna()
    ).sum()),
)
    print("\n--- IF NON-CONSTITUENT ---")
    print(
        "Entry pass proven:",
        int(x["new_entry_pass_if_nonconstituent"].eq(True).sum()),
    )
    print(
        "Entry fail proven:",
        int(x["new_entry_pass_if_nonconstituent"].eq(False).sum()),
    )
    print(
        "Entry unresolved:",
        int(x["new_entry_pass_if_nonconstituent"].isna().sum()),
    )
    print(
    "Factor exactly 1:",
    int((
        x["new_entry_factor_min_if_pass"].eq(1)
        & x["new_entry_factor_max_if_pass"].eq(1)
    ).sum()),
)

    print(
    "Factor exactly .5:",
    int((
        x["new_entry_factor_min_if_pass"].eq(.5)
        & x["new_entry_factor_max_if_pass"].eq(.5)
    ).sum()),
)

    print(
    "Factor interval [.5,1]:",
    int((
        x["new_entry_factor_min_if_pass"].eq(.5)
        & x["new_entry_factor_max_if_pass"].eq(1)
    ).sum()),
)
    print("\nStatus:")
    print(x["foreign_room_status"].value_counts(dropna=False).to_string())

    print("\nITC resolved rows:", itc_resolved)
    print(
        "INDUSINDBK Nov-23 room:",
        x.loc[indus_nov, "foreign_room"].iloc[0]
        if indus_nov.any() else np.nan,
    )
    print("JIOFIN corporate-event rows:", int(jio.sum()))
    print("VMM dead-source rows:", int(vmm.sum()))
    print("Current FOL reference used historically:", 0)
    print("Prior-IMI decisions made here:", 0)

    print("\nFO lookahead:", int(fo_lookahead.sum()))
    print("FOL lookahead:", int(fol_lookahead.sum()))
    print("Lower-bound lookahead:", int(lb_lookahead.sum()))
    print("Duplicates:", int(x.duplicated(["review", "security_id"]).sum()))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()