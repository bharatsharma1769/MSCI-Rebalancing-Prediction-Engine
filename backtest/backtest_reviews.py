from pathlib import Path
import numpy as np
import pandas as pd

PRED_PATH = Path("data/processed/predicted_changes.parquet")
LABEL_PATH = Path("data/processed/review_labels.parquet")
RESULT_PATH = Path("data/processed/backtest_results.parquet")
SUMMARY_PATH = Path("data/outputs/backtests/backtest_summary.csv")

BURN_IN = "2023-05"
FINAL_ACTIONS = {"ADD", "DELETE", "KEEP", "NONE"}


def div(a, b):
    return a / b if b else np.nan


def prf(pred, actual):
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f1 = 2 * p * r / (p + r) if pd.notna(p) and pd.notna(r) and p + r else np.nan
    return tp, fp, fn, p, r, f1


def candidate_ranks(x, mask, support_col, out_col):
    x[out_col] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    for _, idx in x[mask].groupby("review").groups.items():
        order = x.loc[idx].sort_values(
            [support_col, "unresolved_scenarios", "security_id"],
            ascending=[False, False, True],
        ).index
        x.loc[order, out_col] = range(1, len(order) + 1)
    return x


def prediction_status(x):
    no_pred = x["predicted_action"].eq("NO_PREDICTION")
    unresolved = x["unresolved_scenarios"].fillna(0).gt(0)

    status = pd.Series("FINAL", index=x.index, dtype="string")
    status.loc[no_pred] = "NO_PREDICTION"

    fr_col = "foreign_room_status" if "foreign_room_status" in x.columns else None
    epi_col = next((c for c in ["extreme_price_status", "epi_status"] if c in x.columns), None)

    fr = (
        x[fr_col].astype("string").str.upper().str.contains("UNRESOLVED|PENDING", na=False)
        if fr_col else pd.Series(False, index=x.index)
    )
    epi = (
        x[epi_col].astype("string").str.upper().str.contains("UNRESOLVED|PENDING", na=False)
        if epi_col else pd.Series(False, index=x.index)
    )

    status.loc[unresolved & fr & ~epi] = "PROVISIONAL_FOREIGN_ROOM"
    status.loc[unresolved & epi & ~fr] = "PROVISIONAL_EXTREME_PRICE"
    status.loc[unresolved & fr & epi] = "PROVISIONAL_MULTIPLE"
    status.loc[unresolved & ~fr & ~epi] = "PROVISIONAL_SCENARIO_UNRESOLVED"
    return status


def summarize(g, review, topn=True):
    scored = review != BURN_IN
    s = g[g["label_available"]].copy()

    actual_add = s["actual_action"].eq("ADD")
    actual_del = s["actual_action"].eq("DELETE")

    add_universe = ~s["was_standard_constituent"].fillna(False) | actual_add
    del_universe = s["was_standard_constituent"].fillna(False) | actual_del

    da = s["definite_add"] & add_universe
    dd = s["definite_delete"] & del_universe
    sa = s["supported_add"] & add_universe
    sd = s["supported_delete"] & del_universe

    add = prf(da, actual_add)
    delete = prf(dd, actual_del)
    add_sup = prf(sa, actual_add)
    del_sup = prf(sd, actual_del)

    n_add, n_del = int(actual_add.sum()), int(actual_del.sum())
    add_top = int((actual_add & s["add_candidate_rank"].le(n_add)).sum()) if topn and n_add else 0
    del_top = int((actual_del & s["delete_candidate_rank"].le(n_del)).sum()) if topn and n_del else 0

    inc = s["was_standard_constituent"].fillna(False)
    retained = inc & ~actual_del
    inc_resolved = inc & s["predicted_action"].isin(["KEEP", "DELETE"])

    row = {
        "review": review,
        "scored": scored,
        "actual_adds": n_add,
        "predicted_adds": int(da.sum()),
        "correct_adds": add[0],
        "add_fp": add[1],
        "add_fn": add[2],
        "add_precision": add[3],
        "add_recall": add[4],
        "add_f1": add[5],
        "supported_add_candidates": int(sa.sum()),
        "supported_add_precision": add_sup[3],
        "supported_add_recall": add_sup[4],
        "add_envelope_coverage": div(int((actual_add & s["add_uncertainty_envelope"]).sum()), n_add),
        "actual_deletes": n_del,
        "predicted_deletes": int(dd.sum()),
        "correct_deletes": delete[0],
        "delete_fp": delete[1],
        "delete_fn": delete[2],
        "delete_precision": delete[3],
        "delete_recall": delete[4],
        "delete_f1": delete[5],
        "supported_delete_candidates": int(sd.sum()),
        "supported_delete_precision": del_sup[3],
        "supported_delete_recall": del_sup[4],
        "delete_envelope_coverage": div(int((actual_del & s["delete_uncertainty_envelope"]).sum()), n_del),
        "top_n_add_hits": add_top if topn else np.nan,
        "top_n_add_hit_rate": div(add_top, n_add) if topn else np.nan,
        "top_n_delete_hits": del_top if topn else np.nan,
        "top_n_delete_hit_rate": div(del_top, n_del) if topn else np.nan,
        "incumbents": int(inc.sum()),
        "incumbent_resolution_rate": div(int(inc_resolved.sum()), int(inc.sum())),
        "incumbent_retention_accuracy": div(
            int((retained & s["predicted_action"].eq("KEEP")).sum()),
            int(retained.sum()),
        ),
        "provisional_rows": int(s["prediction_status"].str.startswith("PROVISIONAL").sum()),
    }

    if not scored:
        for c in row:
            if c not in {
                "review", "scored", "actual_adds", "predicted_adds",
                "actual_deletes", "predicted_deletes", "incumbents",
                "provisional_rows",
            }:
                row[c] = np.nan
    return row

def pct(v):
    return "NA" if pd.isna(v) else f"{100 * v:.1f}%"


def main():
    p = pd.read_parquet(PRED_PATH)
    y = pd.read_parquet(LABEL_PATH)

    need_p = {
        "review", "security_id", "was_standard_constituent",
        "predicted_action", "true_scenarios", "false_scenarios",
        "unresolved_scenarios",
    }
    need_y = {"review", "security_id", "actual_action"}

    if need_p - set(p.columns):
        raise RuntimeError(f"predicted_changes missing: {sorted(need_p - set(p.columns))}")
    if need_y - set(y.columns):
        raise RuntimeError(f"review_labels missing: {sorted(need_y - set(y.columns))}")

    for x in [p, y]:
        x["review"] = x["review"].astype(str)
        x["security_id"] = x["security_id"].astype("string")

    if p.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate prediction keys.")
    if y.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate label keys.")

    label_extra = [c for c in y.columns if c not in {"review", "security_id", "actual_action"}]
    y = y.rename(columns={c: f"label_{c}" for c in label_extra})

    x = p.merge(y, on=["review", "security_id"], how="outer", validate="one_to_one", indicator=True)

    x["prediction_available"] = x["_merge"].ne("right_only")
    x["label_available"] = x["_merge"].ne("left_only")
    x["coverage_status"] = "MATCHED"

    x.loc[
        x["prediction_available"] & ~x["label_available"],
        "coverage_status",
    ] = "MISSING_LABEL"

    x.loc[
        ~x["prediction_available"]
        & x["label_available"]
        & x["actual_action"].eq("UNCHANGED"),
        "coverage_status",
    ] = "OUTSIDE_DAY9_CANDIDATE_SET"

    x.loc[
        ~x["prediction_available"]
        & x["label_available"]
        & x["actual_action"].isin(["ADD", "DELETE"]),
        "coverage_status",
    ] = "OUTSIDE_MODEL_UNIVERSE"

    x["predicted_action"] = x["predicted_action"].astype("string").str.upper().fillna("NO_PREDICTION")
    x["actual_action"] = x["actual_action"].astype("string").str.upper()
    x["actual_action"] = x["actual_action"].replace({
        "NONE": "UNCHANGED", "KEEP": "UNCHANGED",
        "RETAIN": "UNCHANGED", "NO_CHANGE": "UNCHANGED",
    }).fillna("NO_LABEL")

    x["was_standard_constituent"] = x["was_standard_constituent"].astype("boolean")
    x.loc[x["was_standard_constituent"].isna() & x["actual_action"].eq("ADD"),
          "was_standard_constituent"] = False
    x.loc[x["was_standard_constituent"].isna() & x["actual_action"].eq("DELETE"),
          "was_standard_constituent"] = True

    for c in ["true_scenarios", "false_scenarios", "unresolved_scenarios"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(int)

    total = x["true_scenarios"] + x["false_scenarios"] + x["unresolved_scenarios"]
    x["event_support_fraction"] = np.where(
        x["was_standard_constituent"].fillna(False),
        x["false_scenarios"] / total.replace(0, np.nan),
        x["true_scenarios"] / total.replace(0, np.nan),
    )

    incumbent = x["was_standard_constituent"].fillna(False)

    x["definite_add"] = x["predicted_action"].eq("ADD")
    x["definite_delete"] = x["predicted_action"].eq("DELETE")

    x["supported_add"] = ~incumbent & x["true_scenarios"].gt(0)
    x["supported_delete"] = incumbent & x["false_scenarios"].gt(0)

    x["add_uncertainty_envelope"] = (
        ~incumbent
        & (x["true_scenarios"].gt(0) | x["unresolved_scenarios"].gt(0))
    )
    x["delete_uncertainty_envelope"] = (
        incumbent
        & (x["false_scenarios"].gt(0) | x["unresolved_scenarios"].gt(0))
    )

    x["prediction_status"] = prediction_status(x)
    x["scored"] = x["review"].ne(BURN_IN) & x["label_available"]

    x = candidate_ranks(
        x,
        ~incumbent & x["prediction_available"],
        "event_support_fraction",
        "add_candidate_rank",
    )
    x = candidate_ranks(
        x,
        incumbent & x["prediction_available"],
        "event_support_fraction",
        "delete_candidate_rank",
    )

    x["predicted_rank"] = x["add_candidate_rank"].where(
        ~incumbent, x["delete_candidate_rank"]
    )

    x["correct"] = pd.Series(pd.NA, index=x.index, dtype="boolean")
    resolved = x["predicted_action"].isin(FINAL_ACTIONS) & x["label_available"]

    correct = (
        (x["predicted_action"].eq("ADD") & x["actual_action"].eq("ADD"))
        | (x["predicted_action"].eq("DELETE") & x["actual_action"].eq("DELETE"))
        | (x["predicted_action"].isin(["KEEP", "NONE"]) & x["actual_action"].eq("UNCHANGED"))
    )
    x.loc[resolved, "correct"] = correct[resolved]

    event_without_prediction = (
        x["predicted_action"].eq("NO_PREDICTION")
        & x["actual_action"].isin(["ADD", "DELETE"])
    )
    x.loc[event_without_prediction, "correct"] = False

    x["add_false_positive"] = x["definite_add"] & ~x["actual_action"].eq("ADD")
    x["add_false_negative"] = x["actual_action"].eq("ADD") & ~x["definite_add"]
    x["delete_false_positive"] = x["definite_delete"] & ~x["actual_action"].eq("DELETE")
    x["delete_false_negative"] = x["actual_action"].eq("DELETE") & ~x["definite_delete"]

    summary = pd.DataFrame([
        summarize(g, review)
        for review, g in x.groupby("review", sort=True)
    ])

    scored = x[x["review"].ne(BURN_IN)]
    pooled = summarize(scored, "POOLED", topn=False)
    pooled["scored"] = True
    summary = pd.concat([summary, pd.DataFrame([pooled])], ignore_index=True)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(RESULT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    
    pooled = summary[summary["review"].eq("POOLED")].iloc[0]
    reviews = summary[
        summary["scored"]
        & ~summary["review"].eq("POOLED")
    ].copy()

    top_add_hits = int(reviews["top_n_add_hits"].sum())
    top_del_hits = int(reviews["top_n_delete_hits"].sum())
    actual_adds = int(reviews["actual_adds"].sum())
    actual_dels = int(reviews["actual_deletes"].sum())

    outside = int(
        (
            x["coverage_status"].eq("OUTSIDE_MODEL_UNIVERSE")
            & x["actual_action"].isin(["ADD", "DELETE"])
            & x["review"].ne(BURN_IN)
        ).sum()
    )

    provisional = int(
        x["prediction_status"]
        .astype("string")
        .str.startswith("PROVISIONAL", na=False)
        .sum()
    )

    print("\n=== HISTORICAL BACKTEST | 2023-08 TO 2026-05 ===")

    print("\nADDITIONS")
    print(
        f"Definite calls: {int(pooled['predicted_adds'])}"
        f" | Correct: {int(pooled['correct_adds'])}"
        f" | Actual additions: {int(pooled['actual_adds'])}"
    )
    print(
        f"Precision: {pct(pooled['add_precision'])}"
        f" | Recall: {pct(pooled['add_recall'])}"
        f" | F1: {pct(pooled['add_f1'])}"
    )
    print(
        f"Scenario-supported recall: {pct(pooled['supported_add_recall'])}"
        f" | Top-N hit rate: {pct(top_add_hits / actual_adds)}"
    )

    print("\nDELETIONS")
    print(
        f"Definite calls: {int(pooled['predicted_deletes'])}"
        f" | Correct: {int(pooled['correct_deletes'])}"
        f" | Actual deletions: {int(pooled['actual_deletes'])}"
    )
    print(
        f"Precision: {pct(pooled['delete_precision'])}"
        f" | Recall: {pct(pooled['delete_recall'])}"
        f" | F1: {pct(pooled['delete_f1'])}"
    )
    print(
        f"Scenario-supported recall: {pct(pooled['supported_delete_recall'])}"
        f" | Top-N hit rate: {pct(top_del_hits / actual_dels)}"
    )

    print("\nROBUSTNESS")
    print(
        f"Incumbent retention accuracy: {pct(pooled['incumbent_retention_accuracy'])}"
        f" | Provisional rows: {provisional}"
        f" | Scored events outside model: {outside}"
    )

    compact = reviews[[
        "review",
        "actual_adds", "predicted_adds", "correct_adds",
        "add_precision", "add_recall",
        "actual_deletes", "predicted_deletes", "correct_deletes",
        "delete_precision", "delete_recall",
    ]].copy()

    compact["Adds"] = (
        compact["correct_adds"].fillna(0).astype(int).astype(str)
        + "/"
        + compact["predicted_adds"].fillna(0).astype(int).astype(str)
        + " correct"
    )

    compact["Add actual"] = compact["actual_adds"].astype(int)
    compact["Add P"] = compact["add_precision"].map(pct)
    compact["Add R"] = compact["add_recall"].map(pct)

    compact["Dels"] = (
        compact["correct_deletes"].fillna(0).astype(int).astype(str)
        + "/"
        + compact["predicted_deletes"].fillna(0).astype(int).astype(str)
        + " correct"
    )

    compact["Del actual"] = compact["actual_deletes"].astype(int)
    compact["Del P"] = compact["delete_precision"].map(pct)
    compact["Del R"] = compact["delete_recall"].map(pct)

    print("\nBY REVIEW")
    print(
        compact[
            [
                "review",
                "Adds", "Add actual", "Add P", "Add R",
                "Dels", "Del actual", "Del P", "Del R",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", RESULT_PATH)
    print("Saved:", SUMMARY_PATH)


if __name__ == "__main__":
    main()