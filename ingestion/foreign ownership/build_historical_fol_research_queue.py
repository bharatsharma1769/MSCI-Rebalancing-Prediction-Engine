from pathlib import Path
import pandas as pd


IN_PATH = Path(
    "data/processed/historical_fol_targets.parquet"
)
OUT_PATH = Path(
    "data/processed/historical_fol_research_queue.parquet"
)


def join_unique(s):
    values = set()

    for x in s.dropna():
        for part in str(x).split(";"):
            part = part.strip()
            if part:
                values.add(part)

    return ";".join(sorted(values))


def join_reviews(s):
    return ";".join(
        sorted(set(s.dropna().astype(str)))
    )


def main():
    x = pd.read_parquet(IN_PATH)

    required = {
        "review",
        "security_id",
        "nse_symbol",
        "fol_research_priority",
        "fol_research_reason",
        "needs_historical_fol_research",
    }

    if not required.issubset(x.columns):
        raise RuntimeError(
            "Missing: "
            + str(
                sorted(
                    required - set(x.columns)
                )
            )
        )

    x["review"] = x["review"].astype(str)
    x["security_id"] = (
        x["security_id"].astype("string")
    )

    x["needs_historical_fol_research"] = (
        x["needs_historical_fol_research"]
        .fillna(False)
        .astype(bool)
    )

    if x.duplicated(
        ["review", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate FOL target keys."
        )

    r = x[
        x["needs_historical_fol_research"]
        & x["fol_research_priority"].isin(
            ["CRITICAL", "HIGH"]
        )
    ].copy()

    if r.empty:
        raise RuntimeError(
            "No historical-FOL research rows."
        )

    agg = {
        "first_relevant_review":
            ("review", "min"),
        "last_relevant_review":
            ("review", "max"),
        "relevant_review_rows":
            ("review", "size"),
        "relevant_reviews":
            ("review", join_reviews),
        "critical_rows": (
            "fol_research_priority",
            lambda s: int(
                s.eq("CRITICAL").sum()
            ),
        ),
        "high_rows": (
            "fol_research_priority",
            lambda s: int(
                s.eq("HIGH").sum()
            ),
        ),
        "research_reasons": (
            "fol_research_reason",
            join_unique,
        ),
    }

    optional = {
        "company_name":
            ("company_name", "first"),
        "was_standard_constituent_any":
            (
                "was_standard_constituent",
                "max",
            ),
        "preliminary_pass_any":
            (
                "preliminary_pass",
                "max",
            ),
        "restricted_fol_watch_any":
            (
                "restricted_fol_watch",
                "max",
            ),
        "min_current_fol_reference":
            (
                "current_fol_reference",
                "min",
            ),
        "max_foreign_ownership":
            (
                "foreign_ownership",
                "max",
            ),
        "min_current_reference_room":
            (
                "room_using_current_fol_reference",
                "min",
            ),
        "max_required_fol_for_15pct_room":
            (
                "fol_required_for_15pct_room",
                "max",
            ),
        "current_fol_below_ownership_any":
            (
                "current_fol_reference_below_ownership",
                "max",
            ),
    }

    for name, spec in optional.items():
        if spec[0] in r.columns:
            agg[name] = spec

    q = (
        r.groupby(
            ["security_id", "nse_symbol"],
            dropna=False,
        )
        .agg(**agg)
        .reset_index()
    )

    q["research_priority"] = "HIGH"

    q.loc[
        q["critical_rows"].gt(0),
        "research_priority",
    ] = "CRITICAL"

    q["needs_exact_historical_fol"] = True

    q["_priority"] = (
        q["research_priority"]
        .map(
            {
                "CRITICAL": 0,
                "HIGH": 1,
            }
        )
    )

    q = (
        q.sort_values(
            [
                "_priority",
                "critical_rows",
                "relevant_review_rows",
                "security_id",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    if q["security_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate securities in FOL queue."
        )

    summary = q.groupby(
        "research_priority"
    ).agg(
        securities=("security_id", "size"),
        review_rows=(
            "relevant_review_rows",
            "sum",
        ),
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    q.to_parquet(
        OUT_PATH,
        index=False,
    )

    print(
        "\n--- HISTORICAL FOL RESEARCH QUEUE ---"
    )
    print(summary.to_string())

    print("\nResearch securities:", len(q))
    print(
        "Critical securities:",
        int(
            q["research_priority"]
            .eq("CRITICAL")
            .sum()
        ),
    )
    print(
        "High securities:",
        int(
            q["research_priority"]
            .eq("HIGH")
            .sum()
        ),
    )
    print(
        "Underlying research rows:",
        int(
            q["relevant_review_rows"].sum()
        ),
    )
    print(
        "Duplicate securities:",
        int(
            q["security_id"]
            .duplicated()
            .sum()
        ),
    )
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()