from pathlib import Path

import pandas as pd


IN_PATH = Path("data/processed/standard_review_state.parquet")
OUT_PATH = Path("data/processed/predicted_changes.parquet")

EXPECTED_SCENARIOS = 8
BURN_IN_REVIEW = "2023-05"


def require(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"predicted_changes missing: {missing}")


def main():
    x = pd.read_parquet(IN_PATH)

    require(
        x,
        [
            "review",
            "scenario_id",
            "security_id",
            "nse_symbol",
            "company_key",
            "was_standard_constituent",
            "predicted_standard",
        ],
    )

    x["review"] = x["review"].astype(str)
    x["scenario_id"] = x["scenario_id"].astype(str)
    x["security_id"] = x["security_id"].astype("string")
    x["company_key"] = x["company_key"].astype("string")
    x["was_standard_constituent"] = x["was_standard_constituent"].fillna(False).astype(bool)
    x["predicted_standard"] = x["predicted_standard"].astype("boolean")

    if x.duplicated(["review", "scenario_id", "security_id"]).any():
        raise RuntimeError("Duplicate review/scenario/security rows.")

    scenario_counts = x.groupby("review")["scenario_id"].nunique()

    if not scenario_counts.eq(EXPECTED_SCENARIOS).all():
        raise RuntimeError(
            "Expected 8 scenarios per review:\n"
            + scenario_counts.to_string()
        )

    security_counts = x.groupby(["review", "security_id"]).size()

    if not security_counts.eq(EXPECTED_SCENARIOS).all():
        bad = security_counts[~security_counts.eq(EXPECTED_SCENARIOS)]

        raise RuntimeError(
            "Security missing scenario rows:\n"
            + bad.head(20).to_string()
        )

    # §2.4 continuity guard.
    continuity = (
        x.assign(pred=x["predicted_standard"].fillna(False))
        .groupby(["review", "scenario_id"])["pred"]
        .sum()
        .astype(int)
    )

    if continuity.lt(3).any():
        raise RuntimeError(
            "MSCI EM Standard continuity triggered. "
            "Implement §2.4 before proceeding:\n"
            + continuity[continuity.lt(3)].to_string()
        )

    membership_consistency = (
        x.groupby(["review", "security_id"])["was_standard_constituent"]
        .nunique(dropna=False)
    )

    if membership_consistency.gt(1).any():
        raise RuntimeError(
            "Pre-review Standard membership varies across scenarios."
        )

    x["scenario_true"] = x["predicted_standard"].eq(True)
    x["scenario_false"] = x["predicted_standard"].eq(False)
    x["scenario_unresolved"] = x["predicted_standard"].isna()

    agg = {
        "nse_symbol": ("nse_symbol", "first"),
        "company_key": ("company_key", "first"),
        "was_standard_constituent": ("was_standard_constituent", "first"),
        "scenario_count": ("scenario_id", "nunique"),
        "true_scenarios": ("scenario_true", "sum"),
        "false_scenarios": ("scenario_false", "sum"),
        "unresolved_scenarios": ("scenario_unresolved", "sum"),
    }

    if "company_name" in x.columns:
        agg["company_name"] = ("company_name", "first")

    out = (
        x.groupby(["review", "security_id"], as_index=False)
        .agg(**agg)
    )

    for c in [
        "scenario_count",
        "true_scenarios",
        "false_scenarios",
        "unresolved_scenarios",
    ]:
        out[c] = out[c].astype(int)

    n = out["scenario_count"]
    current = out["was_standard_constituent"]
    t = out["true_scenarios"]
    f = out["false_scenarios"]

    out["predicted_action"] = "NONE"

    # Existing Standard
    out.loc[current & t.eq(n), "predicted_action"] = "KEEP"
    out.loc[current & f.eq(n), "predicted_action"] = "DELETE"
    out.loc[
        current & ~t.eq(n) & ~f.eq(n),
        "predicted_action",
    ] = "KEEP_OR_DELETE"

    # Non-Standard
    out.loc[~current & t.eq(n), "predicted_action"] = "ADD"
    out.loc[~current & f.eq(n), "predicted_action"] = "NONE"
    out.loc[
        ~current & ~t.eq(n) & ~f.eq(n),
        "predicted_action",
    ] = "ADD_OR_NONE"

    out["prediction_resolved"] = out["predicted_action"].isin(
        ["ADD", "DELETE", "KEEP", "NONE"]
    )

    out["predicted_standard_definite"] = t.eq(n)
    out["predicted_standard_possible"] = (t + out["unresolved_scenarios"]).gt(0)

    out["scenario_support_fraction"] = (
        pd.concat([t, f], axis=1).max(axis=1) / n
    )

    out["backtest_scored_review"] = ~out["review"].eq(BURN_IN_REVIEW)

    # Retain every evaluated security. Explicit NONE rows are required
    # for complete historical backtesting and candidate ranking.

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate final review/security rows.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    summary = (
        out.groupby("review", as_index=False)
        .agg(
            definite_add=("predicted_action", lambda s: s.eq("ADD").sum()),
            possible_add=("predicted_action", lambda s: s.eq("ADD_OR_NONE").sum()),
            definite_delete=("predicted_action", lambda s: s.eq("DELETE").sum()),
            possible_delete=("predicted_action", lambda s: s.eq("KEEP_OR_DELETE").sum()),
            definite_keep=("predicted_action", lambda s: s.eq("KEEP").sum()),
            unresolved_names=("unresolved_scenarios", lambda s: s.gt(0).sum()),
        )
    )

    print("\n--- PREDICTED STANDARD CHANGES ---")
    print(summary.to_string(index=False))

    unresolved = out[out["unresolved_scenarios"].gt(0)]

    print("\nScenario-unresolved review/security keys:", len(unresolved))

    if not unresolved.empty:
        print(
            unresolved[
                [
                    "review",
                    "security_id",
                    "nse_symbol",
                    "predicted_action",
                    "true_scenarios",
                    "false_scenarios",
                    "unresolved_scenarios",
                ]
            ].to_string(index=False)
        )

    print("\nRows:", len(out))
    print("Continuity triggers:", int(continuity.lt(3).sum()))
    print("May-2023 burn-in excluded from scoring:", True)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()