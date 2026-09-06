from pathlib import Path

import numpy as np
import pandas as pd


PRED_PATH = Path("data/processed/predicted_changes.parquet")
STD_PATH = Path("data/processed/standard_review_state.parquet")
RANK_PATH = Path("data/processed/candidate_rankings.parquet")
OUT_PATH = Path("data/processed/final_decisions.parquet")


BURN_IN = "2023-05"


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def main():
    p = pd.read_parquet(PRED_PATH)
    r = pd.read_parquet(RANK_PATH)
    s = pd.read_parquet(STD_PATH)

    for x in [p, r, s]:

        x["review"] = x["review"].astype(str)
        x["security_id"] = x["security_id"].astype("string")

    scenario_add_counts = (
    s[
        ~s["was_standard_constituent"]
        .fillna(False)
        .astype(bool)
    ]
    .assign(
        scenario_add=lambda z:
            z["final_standard_security_state"].eq("PASS")
    )
    .groupby(["review", "scenario_id"])["scenario_add"]
    .sum()
    .reset_index()
)

    scenario_count_dist = (
        scenario_add_counts
        .groupby("review")["scenario_add"]
        .agg(
            scenario_add_min="min",
            scenario_add_median="median",
            scenario_add_max="max",
        )
    )

    require(
        p,
        [
            "review", "security_id", "nse_symbol",
            "company_name", "predicted_action",
        ],
        "predicted_changes",
    )

    require(
        r,
        [
            "review", "security_id", "nse_symbol", "company_name",
            "candidate_side", "ranking_method", "ranking_score",
            "event_probability", "model_candidate_rank",
            "methodology_support",
        ],
        "candidate_rankings",
    )

    reviews = sorted(
        set(r.loc[r["candidate_side"].eq("ADD"), "review"])
        - {BURN_IN}
    )

    outputs = []
    stats = []

    for review in reviews:
        pr = p[p["review"].eq(review)].copy()

        methodology_add_ids = set(
            pr.loc[
                pr["predicted_action"].eq("ADD"),
                "security_id",
            ]
        )

        methodology_delete = pr[
            pr["predicted_action"].eq("DELETE")
        ].copy()

        methodology_k = len(methodology_add_ids)

        add = (
            r[
                r["review"].eq(review)
                & r["candidate_side"].eq("ADD")
            ]
            .sort_values(
                ["model_candidate_rank", "security_id"]
            )
            .copy()
        )

        warmup = add["ranking_method"].str.contains(
            "WARMUP", na=False
        ).all()

        if warmup:
            selected = add[
                add["security_id"].isin(methodology_add_ids)
            ].copy()

            if len(selected) != methodology_k:
                raise RuntimeError(
                    f"{review}: warmup methodology calls missing "
                    f"from ranking universe."
                )

            final_k = methodology_k
            expected_adds = np.nan
            source = "METHODOLOGY_WARMUP"

        else:
            if review not in scenario_count_dist.index:
                raise RuntimeError(
                    f"{review}: missing scenario ADD-count distribution."
                )

            d = scenario_count_dist.loc[review]

            expected_adds = float(d["scenario_add_median"])

            # Round .5 upward, not Python banker's rounding.
            final_k = int(np.floor(expected_adds + 0.5))
            final_k = max(0, min(final_k, len(add)))

            selected = add.head(final_k).copy()
            source = "SCENARIO_MEDIAN_COUNT"

        selected["final_action"] = "ADD"
        selected["decision_source"] = source
        selected["methodology_event_count"] = methodology_k
        selected["estimated_event_count"] = final_k
        selected["expected_adds"] = expected_adds
        selected["was_methodology_definite_call"] = (
            selected["security_id"].isin(methodology_add_ids)
        )

        keep = [
            "review", "security_id", "nse_symbol", "company_name",
            "final_action", "decision_source",
            "ranking_method", "ranking_score",
            "event_probability", "model_candidate_rank",
            "methodology_support",
            "methodology_event_count", "estimated_event_count",
            "expected_adds", "was_methodology_definite_call",
        ]

        outputs.append(selected[keep])

        methodology_delete["final_action"] = "DELETE"
        methodology_delete["decision_source"] = "METHODOLOGY_DELETE"
        methodology_delete["ranking_method"] = pd.NA
        methodology_delete["ranking_score"] = np.nan
        methodology_delete["event_probability"] = np.nan
        methodology_delete["model_candidate_rank"] = np.nan
        methodology_delete["methodology_support"] = (
            pd.to_numeric(
                methodology_delete["false_scenarios"],
                errors="coerce",
            )
            / pd.to_numeric(
                methodology_delete["scenario_count"],
                errors="coerce",
            ).replace(0, np.nan)
        )
        methodology_delete["methodology_event_count"] = len(methodology_delete)
        methodology_delete["estimated_event_count"] = len(methodology_delete)
        methodology_delete["expected_adds"] = np.nan
        methodology_delete["was_methodology_definite_call"] = True

        outputs.append(methodology_delete[keep])

        stats.append({
            "review": review,
            "methodology_add_count": methodology_k,
            "expected_adds": expected_adds,
            "final_add_count": final_k,
            "delete_count": len(methodology_delete),
            "count_source": source,
        })

    out = pd.concat(outputs, ignore_index=True)

    if out.duplicated(
        ["review", "security_id", "final_action"]
    ).any():
        raise RuntimeError("Duplicate final decision keys.")

    clash = (
        out.groupby(["review", "security_id"])["final_action"]
        .nunique()
        .gt(1)
    )

    if clash.any():
        raise RuntimeError(
            "Security simultaneously selected ADD and DELETE."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    stats = pd.DataFrame(stats)

    print("\n=== WALK-FORWARD COUNT + FINAL DECISIONS ===")
    print(
        stats.to_string(
            index=False,
            formatters={
                "expected_adds": lambda x: (
                    "NA" if pd.isna(x) else f"{x:.2f}"
                )
            },
        )
    )

    print("\nRows:", len(out))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()