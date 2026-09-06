from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PRED_PATH = Path("data/processed/predicted_changes.parquet")
MIEU_PATH = Path("data/processed/mieu_review_state.parquet")
STD_PATH = Path("data/processed/standard_review_state.parquet")
LABEL_PATH = Path("data/processed/review_labels.parquet")
OUT_PATH = Path("data/processed/candidate_rankings.parquet")

BURN_IN = "2023-05"

# All are methodology-motivated cross-sectional signals.
SIGNALS = [
    "sig_ff_gap",
    "sig_support",
    "sig_mcap_gap"
]


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def bool_num(s):
    return s.astype("boolean").astype("Float64").astype(float)


def pct_rank(s, higher_better=True):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, method="average")
    return r if higher_better else 1 - r


def build_features():
    p = pd.read_parquet(PRED_PATH)
    m = pd.read_parquet(MIEU_PATH)
    s = pd.read_parquet(STD_PATH)

    for df in [p, m, s]:
        df["review"] = df["review"].astype(str)
        df["security_id"] = df["security_id"].astype("string")

    require(
        p,
        [
            "review", "security_id", "nse_symbol", "company_name",
            "was_standard_constituent", "scenario_count",
            "true_scenarios", "false_scenarios", "unresolved_scenarios",
        ],
        "predicted_changes",
    )

    require(
        m,
        [
            "review", "security_id", "mieu_state",
            "company_full_mcap_usd", "fif",
            "atvr_12m", "atvr_3m", "fot_3m",
            "foreign_room", "foreign_room_lower_bound",
            "prior_imi", "prior_imi_uncertain",
        ],
        "mieu_review_state",
    )

    require(
        s,
        [
            "review", "security_id", "scenario_id",
            "market_standard_cutoff_usd",
            "provisional_standard",
            "assignment_priority",
            "final_required_ff_mcap_usd",
            "final_ff_mcap_evaluation_min_usd",
            "final_ff_mcap_evaluation_max_usd",
        ],
        "standard_review_state",
    )

    if p.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate prediction keys.")
    if m.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate MIEU keys.")
    if s.duplicated(["review", "scenario_id", "security_id"]).any():
        raise RuntimeError("Duplicate Standard scenario keys.")

    mc = m[
        ["review", "security_id", "company_full_mcap_usd"]
    ].copy()

    mc["company_rank_from_top"] = (
        mc.groupby("review")["company_full_mcap_usd"]
        .rank(method="dense", ascending=False)
    )

    z = s.merge(
        mc,
        on=["review", "security_id"],
        how="left",
        validate="many_to_one",
    )

    z["company_scenario_rank"] = (
        z.groupby(["review", "scenario_id"])["company_full_mcap_usd"]
        .rank(method="dense", ascending=False)
    )

    cutoff = pd.to_numeric(
        z["market_standard_cutoff_usd"], errors="coerce"
    ).replace(0, np.nan)

    mcap = pd.to_numeric(
        z["company_full_mcap_usd"], errors="coerce"
    )

    z["_above_cutoff_rank"] = z["company_scenario_rank"].where(
        mcap.ge(cutoff)
    )

    z["cutoff_rank"] = (
        z.groupby(["review", "scenario_id"])["_above_cutoff_rank"]
        .transform("max")
    )

    z["rank_gap"] = (
        z["cutoff_rank"] - z["company_scenario_rank"]
    )

    z["mcap_gap"] = mcap / cutoff - 1

    required = pd.to_numeric(
        z["final_required_ff_mcap_usd"], errors="coerce"
    ).replace(0, np.nan)

    ff_min = pd.to_numeric(
        z["final_ff_mcap_evaluation_min_usd"], errors="coerce"
    )
    ff_max = pd.to_numeric(
        z["final_ff_mcap_evaluation_max_usd"], errors="coerce"
    )

    z["ff_gap"] = ((ff_min + ff_max) / 2) / required - 1

    z["provisional_standard"] = (
        z["provisional_standard"]
        .astype("boolean")
        .astype("Float64")
    )

    z["assignment_priority"] = pd.to_numeric(
        z["assignment_priority"], errors="coerce"
    )

    a = (
        z.groupby(["review", "security_id"], as_index=False)
        .agg(
            rank_gap_med=("rank_gap", "median"),
            mcap_gap_med=("mcap_gap", "median"),
            mcap_gap_max=("mcap_gap", "max"),
            ff_gap_med=("ff_gap", "median"),
            ff_gap_max=("ff_gap", "max"),
            provisional_fraction=("provisional_standard", "mean"),
            best_assignment_priority=("assignment_priority", "min"),
        )
    )

    mcols = [
        "review", "security_id", "mieu_state",
        "company_full_mcap_usd", "fif",
        "atvr_12m", "atvr_3m", "fot_3m",
        "foreign_room", "foreign_room_lower_bound",
        "prior_imi", "prior_imi_uncertain",
    ]

    x = (
        p.merge(
            m[mcols],
            on=["review", "security_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            mc[
                ["review", "security_id", "company_rank_from_top"]
            ],
            on=["review", "security_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            a,
            on=["review", "security_id"],
            how="left",
            validate="one_to_one",
        )
    )

    n = pd.to_numeric(
        x["scenario_count"], errors="coerce"
    ).replace(0, np.nan)

    x["add_support"] = (
        pd.to_numeric(x["true_scenarios"], errors="coerce") / n
    )

    x["delete_support"] = (
        pd.to_numeric(x["false_scenarios"], errors="coerce") / n
    )

    x["unresolved_fraction"] = (
        pd.to_numeric(x["unresolved_scenarios"], errors="coerce") / n
    )

    x["mcap_to_cutoff"] = 1 + x["mcap_gap_med"]
    x["ff_to_requirement"] = 1 + x["ff_gap_med"]

    x["assignment_priority_score"] = (
        6 - pd.to_numeric(
            x["best_assignment_priority"], errors="coerce"
        )
    ).clip(0, 5).fillna(0)

    room = pd.to_numeric(x["foreign_room"], errors="coerce")
    room_lb = pd.to_numeric(
        x["foreign_room_lower_bound"], errors="coerce"
    )

    x["foreign_room_effective"] = room.fillna(room_lb)
    x["prior_imi"] = bool_num(x["prior_imi"])
    x["prior_imi_uncertain"] = bool_num(x["prior_imi_uncertain"])

    return x


def add_trajectory(x):
    x = x.copy()

    reviews = sorted(x["review"].unique())
    review_num = {r: i for i, r in enumerate(reviews)}

    x["_review_num"] = x["review"].map(review_num)
    x = x.sort_values(["security_id", "_review_num"])

    trajectory = [
        "company_rank_from_top",
        "rank_gap_med",
        "mcap_gap_med",
        "ff_gap_med",
    ]

    prev_num = x.groupby("security_id")["_review_num"].shift(1)
    immediate = x["_review_num"].sub(prev_num).eq(1)

    for c in trajectory:
        lag = x.groupby("security_id")[c].shift(1)
        x[f"delta_{c}"] = (x[c] - lag).where(immediate)

    return x.drop(columns="_review_num")


def add_cross_sectional_signals(x):
    x = x.copy()

    specs = {
        "sig_company_size": ("company_rank_from_top", False),
        "sig_rank_gap": ("rank_gap_med", True),
        "sig_mcap_gap": ("mcap_gap_med", True),
        "sig_ff_gap": ("ff_gap_med", True),
        "sig_support": ("add_support", True),
        "sig_delta_company_rank": (
            "delta_company_rank_from_top", False
        ),
        "sig_delta_rank_gap": ("delta_rank_gap_med", True),
        "sig_delta_mcap_gap": ("delta_mcap_gap_med", True),
        "sig_delta_ff_gap": ("delta_ff_gap_med", True),
    }

    for out, (src, higher) in specs.items():
        x[out] = (
            x.groupby("review")[src]
            .transform(lambda s: pct_rank(s, higher))
        )

    return x


def heuristic_score(x):
    return x[SIGNALS].mean(axis=1, skipna=True).fillna(0)


def fit_score(train, current):
    if train.empty or train["target_add"].nunique() < 2:
        return (
            heuristic_score(current),
            pd.Series(np.nan, index=current.index),
            "SIGNAL_HEURISTIC_WARMUP",
        )

    prep = ColumnTransformer(
        [
            (
                "signal",
                SimpleImputer(strategy="median", add_indicator=True),
                SIGNALS,
            )
        ],
        remainder="drop",
    )

    rank_model = Pipeline([
        ("prep", prep),
        (
            "clf",
            LogisticRegression(
                C=0.20,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=0,
            ),
        ),
    ])

    count_model = Pipeline([
    (
        "prep",
        ColumnTransformer(
            [
                (
                    "signal",
                    SimpleImputer(strategy="median", add_indicator=True),
                    SIGNALS,
                )
            ],
            remainder="drop",
        ),
    ),
    (
        "clf",
        LogisticRegression(
            C=0.20,
            class_weight=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        ),
    ),
])

    review_n = train.groupby("review")["security_id"].transform("size")
    weights = 1 / review_n
    weights *= len(weights) / weights.sum()

    rank_model.fit(
        train,
        train["target_add"].astype(int),
        clf__sample_weight=weights,
    )

    count_model.fit(
    train,
    train["target_add"].astype(int),
)
    rank_score = rank_model.predict_proba(current)[:, 1]
    event_prob = count_model.predict_proba(current)[:, 1]

    return (
        pd.Series(rank_score, index=current.index),
        pd.Series(event_prob, index=current.index),
        "FOCUSED_SIGNAL_LOGIT_WF",
    )


def main():
    x = build_features()

    labels = pd.read_parquet(LABEL_PATH)
    labels["review"] = labels["review"].astype(str)
    labels["security_id"] = labels["security_id"].astype("string")

    require(
        labels,
        ["review", "security_id", "actual_action"],
        "review_labels",
    )

    labels = labels[
        ["review", "security_id", "actual_action"]
    ].drop_duplicates(["review", "security_id"])

    reviews = sorted(x["review"].unique())
    scored = [r for r in reviews if r != BURN_IN]

    # --------------------------------------------------------------
    # ADD: methodology-defined competitive universe.
    # Trajectory is calculated BEFORE competitive filtering.
    # --------------------------------------------------------------

    add = x[
        ~x["was_standard_constituent"].fillna(False).astype(bool)
        & x["mieu_state"].isin(["PASS", "UNRESOLVED"])
    ].copy()

    add = add_trajectory(add)

    add["competitive"] = (
        add["add_support"].gt(0)
        | add["unresolved_fraction"].gt(0)
        | add["provisional_fraction"].gt(0)
        | add["mcap_gap_max"].ge(-1 / 3)
        | add["ff_gap_max"].ge(-1 / 3)
    )

    add = add[add["competitive"]].copy()
    add = add_cross_sectional_signals(add)
    add["methodology_support"] = add["add_support"]

    outputs = []
    stats = []

    for review in scored:
        current = add[add["review"].eq(review)].copy()

        if current.empty:
            continue

        prior = [r for r in scored if r < review]

        train = add[
            add["review"].isin(prior)
        ].merge(
            labels[labels["review"].isin(prior)],
            on=["review", "security_id"],
            how="inner",
            validate="one_to_one",
        )

        if not train.empty:
            train["target_add"] = (
                train["actual_action"].eq("ADD").astype(int)
            )

        score, event_prob, method = fit_score(train, current)

        current["ranking_score"] = score
        current["event_probability"] = event_prob
        current["ranking_method"] = method
        current["candidate_side"] = "ADD"

        current = current.sort_values(
            [
                "ranking_score",
                "methodology_support",
                "ff_gap_med",
                "security_id",
            ],
            ascending=[False, False, False, True],
        )

        current["model_candidate_rank"] = np.arange(
            1, len(current) + 1
        )

        outputs.append(current[[
            "review", "security_id", "nse_symbol", "company_name",
            "candidate_side", "ranking_method", "ranking_score","event_probability",
            "model_candidate_rank", "methodology_support",
            "unresolved_fraction", "provisional_fraction",
            "mcap_to_cutoff", "ff_to_requirement",
            "assignment_priority_score", "fif",
            "atvr_12m", "atvr_3m", "fot_3m",
            "foreign_room_effective", "prior_imi",
            "prior_imi_uncertain",
        ]])

        stats.append({
            "review": review,
            "candidate_side": "ADD",
            "candidates": len(current),
            "method": method,
        })

    # --------------------------------------------------------------
    # DELETE: methodology only.
    # --------------------------------------------------------------

    delete = x[
        x["was_standard_constituent"].fillna(False).astype(bool)
    ].copy()

    delete["methodology_support"] = delete["delete_support"]

    for review in scored:
        current = delete[delete["review"].eq(review)].copy()

        if current.empty:
            continue

        current = current.sort_values(
            [
                "methodology_support",
                "unresolved_fraction",
                "security_id",
            ],
            ascending=[False, False, True],
        )

        current["candidate_side"] = "DELETE"
        current["ranking_method"] = "METHODOLOGY_ONLY"

        current["ranking_score"] = (
            100 * current["methodology_support"].fillna(0)
            + current["unresolved_fraction"].fillna(0)
        )

        current["model_candidate_rank"] = np.arange(
            1, len(current) + 1
        )

        current["event_probability"] = np.nan

        outputs.append(current[[
            "review", "security_id", "nse_symbol", "company_name",
            "candidate_side", "ranking_method", "ranking_score","event_probability",
            "model_candidate_rank", "methodology_support",
            "unresolved_fraction", "provisional_fraction",
            "mcap_to_cutoff", "ff_to_requirement",
            "assignment_priority_score", "fif",
            "atvr_12m", "atvr_3m", "fot_3m",
            "foreign_room_effective", "prior_imi",
            "prior_imi_uncertain",
        ]])

        stats.append({
            "review": review,
            "candidate_side": "DELETE",
            "candidates": len(current),
            "method": "METHODOLOGY_ONLY",
        })

    out = pd.concat(outputs, ignore_index=True)

    if out.duplicated(
        ["review", "candidate_side", "security_id"]
    ).any():
        raise RuntimeError("Duplicate ranking keys.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n=== FOCUSED WALK-FORWARD RANKING ===")
    print(pd.DataFrame(stats).to_string(index=False))
    print("\nRows:", len(out))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()