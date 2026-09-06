from pathlib import Path

import numpy as np
import pandas as pd


RANK_PATH = Path("data/processed/candidate_rankings.parquet")
LABEL_PATH = Path("data/processed/review_labels.parquet")
BASE_PATH = Path("data/processed/backtest_results.parquet")

BURN_IN = "2023-05"


def pct(v):
    return "NA" if pd.isna(v) else f"{100 * v:.1f}%"


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def side_stats(ranked, labels, side):
    reviews = sorted(
        labels.loc[
            labels["review"].ne(BURN_IN),
            "review",
        ].unique()
    )

    rows = []

    for review in reviews:
        actual = labels[
            labels["review"].eq(review)
            & labels["actual_action"].eq(side)
        ]

        candidates = ranked[
            ranked["review"].eq(review)
            & ranked["candidate_side"].eq(side)
        ].copy()

        actual_ids = set(actual["security_id"])
        n = len(actual_ids)

        candidates["actual_event"] = candidates["security_id"].isin(actual_ids)

        topn_hits = int(
            candidates[
                candidates["model_candidate_rank"].le(n)
            ]["actual_event"].sum()
        ) if n else 0

        hits3 = int(
            candidates[candidates["model_candidate_rank"].le(3)]
            ["actual_event"].sum()
        )
        hits5 = int(
            candidates[candidates["model_candidate_rank"].le(5)]
            ["actual_event"].sum()
        )
        hits10 = int(
            candidates[candidates["model_candidate_rank"].le(10)]
            ["actual_event"].sum()
        )

        captured = candidates[candidates["actual_event"]]
        captured_count = len(captured)

        rows.append({
            "review": review,
            "side": side,
            "actual_events": n,
            "candidate_coverage": captured_count,
            "top_n_hits": topn_hits,
            "top3_hits": hits3,
            "top5_hits": hits5,
            "top10_hits": hits10,
            "mean_actual_rank": (
                captured["model_candidate_rank"].mean()
                if captured_count else np.nan
            ),
            "median_actual_rank": (
                captured["model_candidate_rank"].median()
                if captured_count else np.nan
            ),
        })

    return pd.DataFrame(rows)


def baseline_topn(base, side):
    action = "ADD" if side == "ADD" else "DELETE"
    rank_col = "add_candidate_rank" if side == "ADD" else "delete_candidate_rank"

    rows = []

    for review, g in base[
        base["review"].ne(BURN_IN)
    ].groupby("review", sort=True):
        actual_ids = set(
            g.loc[g["actual_action"].eq(action), "security_id"]
        )
        n = len(actual_ids)

        if not n:
            hits = 0
        else:
            hits = int(
                g[
                    g[rank_col].le(n)
                    & g["security_id"].isin(actual_ids)
                ].shape[0]
            )

        rows.append((review, n, hits))

    x = pd.DataFrame(
        rows,
        columns=["review", "actual_events", "baseline_hits"],
    )

    total = x["actual_events"].sum()
    return x, x["baseline_hits"].sum() / total if total else np.nan


def print_side(stats, base_rate, side):
    actual = int(stats["actual_events"].sum())
    coverage = int(stats["candidate_coverage"].sum())

    topn = int(stats["top_n_hits"].sum())
    h3 = int(stats["top3_hits"].sum())
    h5 = int(stats["top5_hits"].sum())
    h10 = int(stats["top10_hits"].sum())

    ranks = stats.loc[
        stats["candidate_coverage"].gt(0),
        ["mean_actual_rank", "median_actual_rank"],
    ]

    print(f"\n{side} RANKING")
    print(
        f"Actual events: {actual}"
        f" | Candidate coverage: {coverage}/{actual}"
        f" ({pct(coverage / actual if actual else np.nan)})"
    )
    print(
        f"Top-N oracle hit rate: {pct(topn / actual if actual else np.nan)}"
        f" | Methodology baseline: {pct(base_rate)}"
    )
    print(
        f"Recall@3: {pct(h3 / actual if actual else np.nan)}"
        f" | Recall@5: {pct(h5 / actual if actual else np.nan)}"
        f" | Recall@10: {pct(h10 / actual if actual else np.nan)}"
    )

    if not ranks.empty:
        print(
            f"Mean actual rank: {ranks['mean_actual_rank'].mean():.1f}"
            f" | Median actual rank: {ranks['median_actual_rank'].median():.1f}"
        )


def main():
    r = pd.read_parquet(RANK_PATH)
    l = pd.read_parquet(LABEL_PATH)
    b = pd.read_parquet(BASE_PATH)

    for x in [r, l, b]:
        x["review"] = x["review"].astype(str)
        x["security_id"] = x["security_id"].astype("string")

    require(
        r,
        [
            "review", "security_id", "candidate_side",
            "ranking_score", "model_candidate_rank",
            "ranking_method",
        ],
        "candidate_rankings",
    )

    require(
        l,
        ["review", "security_id", "actual_action"],
        "review_labels",
    )

    require(
        b,
        [
            "review", "security_id", "actual_action",
            "add_candidate_rank", "delete_candidate_rank",
        ],
        "backtest_results",
    )

    l = l[
        ["review", "security_id", "actual_action"]
    ].drop_duplicates(["review", "security_id"])

    add = side_stats(r, l, "ADD")
    delete = side_stats(r, l, "DELETE")

    _, add_base = baseline_topn(b, "ADD")
    _, delete_base = baseline_topn(b, "DELETE")

    print("\n=== WALK-FORWARD RANKING BACKTEST | 2023-08 TO 2026-05 ===")

    print_side(add, add_base, "ADDITIONS")
    print_side(delete, delete_base, "DELETIONS")

    both = pd.concat([add, delete], ignore_index=True)

    display = both.copy()
    display["Top-N"] = (
        display["top_n_hits"].astype(int).astype(str)
        + "/"
        + display["actual_events"].astype(int).astype(str)
    )
    display["Coverage"] = (
        display["candidate_coverage"].astype(int).astype(str)
        + "/"
        + display["actual_events"].astype(int).astype(str)
    )

    print("\nBY REVIEW")
    print(
        display[
            [
                "review", "side", "Top-N", "Coverage",
                "top3_hits", "top5_hits", "top10_hits",
                "median_actual_rank",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()