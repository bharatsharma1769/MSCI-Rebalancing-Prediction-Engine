from pathlib import Path
import numpy as np
import pandas as pd


CANDIDATE_PATH = Path("data/processed/foreign_room_candidates.parquet")
FO_PATH = Path("data/processed/historical_foreign_ownership.parquet")
OUT_PATH = Path("data/processed/historical_fol_targets.parquet")


def main():
    c = pd.read_parquet(CANDIDATE_PATH)
    fo = pd.read_parquet(FO_PATH)

    for df in [c, fo]:
        df["review"] = df["review"].astype(str)
        df["security_id"] = df["security_id"].astype("string")

    if c.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate candidate keys.")
    if fo.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate FO keys.")

    fo_cols = [c for c in [
        "review", "security_id", "foreign_ownership",
        "report_date", "available_date",
        "foreign_ownership_cutoff_date",
        "data_quality_flag", "canonical_row_action",
    ] if c in fo.columns]

    x = c.merge(
        fo[fo_cols],
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    x["foreign_ownership"] = pd.to_numeric(
        x["foreign_ownership"], errors="coerce"
    )

    x["current_fol_reference"] = pd.to_numeric(
        x["current_fol_reference"], errors="coerce"
    )

    bad = (
        x["current_fol_reference"].notna()
        & ~x["current_fol_reference"].between(0, 1)
    )
    if bad.any():
        raise RuntimeError(
            f"Invalid current FOL references: {int(bad.sum())}"
        )

    x["was_standard_constituent"] = (
        x["was_standard_constituent"]
        .fillna(False)
        .astype(bool)
    )

    x["preliminary_pass"] = (
        x["preliminary_pass"]
        .fillna(False)
        .astype(bool)
    )

    if "restricted_fol_watch" in x.columns:
        x["restricted_fol_watch"] = (
            x["restricted_fol_watch"]
            .fillna(False)
            .astype(bool)
        )
    else:
        x["restricted_fol_watch"] = False

    x["fol_required_for_15pct_room"] = (
        x["foreign_ownership"] / 0.85
    )

    valid = (
        x["foreign_ownership"].notna()
        & x["current_fol_reference"].notna()
        & x["current_fol_reference"].gt(0)
    )

    x["room_using_current_fol_reference"] = np.where(
        valid,
        (
            x["current_fol_reference"]
            - x["foreign_ownership"]
        )
        / x["current_fol_reference"],
        np.nan,
    )

    x["current_fol_reference_below_ownership"] = (
        valid
        & x["current_fol_reference"].lt(
            x["foreign_ownership"]
        )
    )

    relevant = (
        x["was_standard_constituent"]
        | x["preliminary_pass"]
        | x["restricted_fol_watch"]
    )

    x["fol_research_priority"] = "LOW"
    x["fol_research_reason"] = "SCREENABLE_LOW_RISK"
    x["needs_historical_fol_research"] = False

    fo_missing = x["foreign_ownership"].isna()

    x.loc[
        fo_missing,
        ["fol_research_priority", "fol_research_reason"],
    ] = [
        "FO_UNRESOLVED",
        "FOREIGN_OWNERSHIP_UNRESOLVED",
    ]

    critical = (
        ~fo_missing
        & relevant
        & x["current_fol_reference_below_ownership"]
    )

    x.loc[
        critical,
        [
            "fol_research_priority",
            "fol_research_reason",
            "needs_historical_fol_research",
        ],
    ] = [
        "CRITICAL",
        "CURRENT_FOL_BELOW_HISTORICAL_OWNERSHIP",
        True,
    ]

    missing_ref = (
        ~fo_missing
        & relevant
        & x["current_fol_reference"].isna()
        & ~critical
    )

    x.loc[
        missing_ref,
        [
            "fol_research_priority",
            "fol_research_reason",
            "needs_historical_fol_research",
        ],
    ] = [
        "HIGH",
        "MISSING_CURRENT_FOL_REFERENCE",
        True,
    ]

    restricted = (
        ~fo_missing
        & relevant
        & x["current_fol_reference"].notna()
        & x["current_fol_reference"].lt(0.999)
        & ~critical
    )

    x.loc[
        restricted,
        [
            "fol_research_priority",
            "fol_research_reason",
            "needs_historical_fol_research",
        ],
    ] = [
        "HIGH",
        "CURRENT_FOL_RESTRICTED",
        True,
    ]

    sensitive = (
        ~fo_missing
        & relevant
        & x["room_using_current_fol_reference"].notna()
        & x["room_using_current_fol_reference"].lt(0.30)
        & ~critical
    )

    already = sensitive & x["needs_historical_fol_research"]

    x.loc[
        already,
        "fol_research_reason",
    ] = (
        x.loc[already, "fol_research_reason"]
        + ";CURRENT_REFERENCE_ROOM_LT_30PCT"
    )

    sensitive_only = (
        sensitive
        & ~x["needs_historical_fol_research"]
    )

    x.loc[
        sensitive_only,
        [
            "fol_research_priority",
            "fol_research_reason",
            "needs_historical_fol_research",
        ],
    ] = [
        "HIGH",
        "CURRENT_REFERENCE_ROOM_LT_30PCT",
        True,
    ]

    x["current_fol_is_reference_only"] = True

    order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "FO_UNRESOLVED": 2,
        "LOW": 3,
    }

    x["_priority"] = (
        x["fol_research_priority"]
        .map(order)
        .fillna(9)
    )

    cols = [
        "review", "security_id", "nse_symbol",
        "company_name",
        "was_standard_constituent",
        "preliminary_pass",
        "restricted_fol_watch",
        "selection_reason",
        "foreign_ownership",
        "report_date", "available_date",
        "foreign_ownership_cutoff_date",
        "data_quality_flag",
        "current_fol_reference",
        "current_fol_is_reference_only",
        "room_using_current_fol_reference",
        "fol_required_for_15pct_room",
        "current_fol_reference_below_ownership",
        "fol_research_priority",
        "fol_research_reason",
        "needs_historical_fol_research",
    ]

    out = (
        x.sort_values(
            ["_priority", "review", "security_id"]
        )[
            [c for c in cols if c in x.columns]
        ]
        .reset_index(drop=True)
    )

    if out.duplicated(
        ["review", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate historical FOL target keys."
        )

    diag = out.groupby("review").agg(
        rows=("security_id", "size"),
        fo_resolved=(
            "foreign_ownership",
            lambda s: int(s.notna().sum()),
        ),
        critical=(
            "fol_research_priority",
            lambda s: int(s.eq("CRITICAL").sum()),
        ),
        high=(
            "fol_research_priority",
            lambda s: int(s.eq("HIGH").sum()),
        ),
        fo_unresolved=(
            "fol_research_priority",
            lambda s: int(
                s.eq("FO_UNRESOLVED").sum()
            ),
        ),
        low=(
            "fol_research_priority",
            lambda s: int(s.eq("LOW").sum()),
        ),
    )

    research = out[
        out["needs_historical_fol_research"]
    ]

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.to_parquet(
        OUT_PATH,
        index=False,
    )

    print(
        "\n--- HISTORICAL FOL TARGETS ---"
    )
    print(diag.to_string())

    print("\nRows:", len(out))
    print(
        "Unique securities:",
        out["security_id"].nunique(),
    )
    print("Research rows:", len(research))
    print(
        "Research securities:",
        research["security_id"].nunique(),
    )
    print(
        "Critical rows:",
        int(
            out["fol_research_priority"]
            .eq("CRITICAL")
            .sum()
        ),
    )
    print(
        "High rows:",
        int(
            out["fol_research_priority"]
            .eq("HIGH")
            .sum()
        ),
    )
    print(
        "FO-unresolved rows:",
        int(
            out["fol_research_priority"]
            .eq("FO_UNRESOLVED")
            .sum()
        ),
    )
    print(
        "Current FOL used as historical:",
        0,
    )
    print(
        "Duplicates:",
        int(
            out.duplicated(
                ["review", "security_id"]
            ).sum()
        ),
    )
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()