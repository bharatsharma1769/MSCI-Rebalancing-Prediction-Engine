from pathlib import Path

import numpy as np
import pandas as pd


DECISION_PATH = Path("data/processed/final_decisions.parquet")
LABEL_PATH = Path("data/processed/review_labels.parquet")
BASE_PATH = Path("data/processed/backtest_results.parquet")
OUT_DIR = Path("data/backtest")

BURN_IN = "2023-05"


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def pct(v):
    return "NA" if pd.isna(v) else f"{100 * v:.1f}%"


def metrics(predicted, actual):
    hits = len(predicted & actual)
    p = hits / len(predicted) if predicted else np.nan
    r = hits / len(actual) if actual else np.nan

    if pd.isna(p) or pd.isna(r) or p + r == 0:
        f1 = np.nan if pd.isna(p) or pd.isna(r) else 0.0
    else:
        f1 = 2 * p * r / (p + r)

    return {
        "predicted": len(predicted),
        "actual": len(actual),
        "hits": hits,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def pooled(rows, side, prefix):
    predicted = sum(r[f"{prefix}_predicted"] for r in rows)
    actual = sum(r[f"{prefix}_actual"] for r in rows)
    hits = sum(r[f"{prefix}_hits"] for r in rows)

    p = hits / predicted if predicted else np.nan
    r = hits / actual if actual else np.nan
    f1 = (
        2 * p * r / (p + r)
        if pd.notna(p) and pd.notna(r) and p + r
        else np.nan
    )

    return predicted, actual, hits, p, r, f1


def main():
    d = pd.read_parquet(DECISION_PATH)
    l = pd.read_parquet(LABEL_PATH)
    b = pd.read_parquet(BASE_PATH)

    for x in [d, l, b]:
        x["review"] = x["review"].astype(str)
        x["security_id"] = x["security_id"].astype("string")

    require(
        d,
        ["review", "security_id", "final_action"],
        "final_decisions",
    )

    require(
        l,
        ["review", "security_id", "actual_action"],
        "review_labels",
    )

    require(
        b,
        ["review", "security_id", "predicted_action", "actual_action"],
        "backtest_results",
    )

    l = l[
        ["review", "security_id", "actual_action"]
    ].drop_duplicates(["review", "security_id"])

    reviews = sorted(
        set(d["review"])
        & set(l["review"])
        - {BURN_IN}
    )

    rows = []

    for review in reviews:
        dr = d[d["review"].eq(review)]
        lr = l[l["review"].eq(review)]
        br = b[b["review"].eq(review)]

        actual_add = set(
            lr.loc[lr["actual_action"].eq("ADD"), "security_id"]
        )
        actual_del = set(
            lr.loc[lr["actual_action"].eq("DELETE"), "security_id"]
        )

        final_add = set(
            dr.loc[dr["final_action"].eq("ADD"), "security_id"]
        )
        final_del = set(
            dr.loc[dr["final_action"].eq("DELETE"), "security_id"]
        )

        base_add = set(
            br.loc[br["predicted_action"].eq("ADD"), "security_id"]
        )
        base_del = set(
            br.loc[br["predicted_action"].eq("DELETE"), "security_id"]
        )

        fa = metrics(final_add, actual_add)
        fd = metrics(final_del, actual_del)
        ba = metrics(base_add, actual_add)
        bd = metrics(base_del, actual_del)

        rows.append({
            "review": review,

            "final_add_predicted": fa["predicted"],
            "final_add_actual": fa["actual"],
            "final_add_hits": fa["hits"],
            "final_add_precision": fa["precision"],
            "final_add_recall": fa["recall"],
            "final_add_f1": fa["f1"],

            "base_add_predicted": ba["predicted"],
            "base_add_actual": ba["actual"],
            "base_add_hits": ba["hits"],
            "base_add_precision": ba["precision"],
            "base_add_recall": ba["recall"],
            "base_add_f1": ba["f1"],

            "final_del_predicted": fd["predicted"],
            "final_del_actual": fd["actual"],
            "final_del_hits": fd["hits"],
            "final_del_precision": fd["precision"],
            "final_del_recall": fd["recall"],
            "final_del_f1": fd["f1"],

            "base_del_predicted": bd["predicted"],
            "base_del_actual": bd["actual"],
            "base_del_hits": bd["hits"],
            "base_del_precision": bd["precision"],
            "base_del_recall": bd["recall"],
            "base_del_f1": bd["f1"],
        })

    result = pd.DataFrame(rows)

    fp, fa, fh, fpp, fpr, fpf = pooled(rows, "ADD", "final_add")
    bp, ba, bh, bpp, bpr, bpf = pooled(rows, "ADD", "base_add")

    dp, da, dh, dpp, dpr, dpf = pooled(rows, "DELETE", "final_del")
    dbp, dba, dbh, dbpp, dbpr, dbpf = pooled(rows, "DELETE", "base_del")

    summary = pd.DataFrame([
        {"side":"ADD","final_calls":fp,"correct":fh,"actual":fa,"precision":fpp,"recall":fpr,"f1":fpf,"baseline_precision":bpp,"baseline_recall":bpr,"baseline_f1":bpf,"f1_improvement_pp":100*(fpf-bpf)},
        {"side":"DELETE","final_calls":dp,"correct":dh,"actual":da,"precision":dpp,"recall":dpr,"f1":dpf,"baseline_precision":dbpp,"baseline_recall":dbpr,"baseline_f1":dbpf,"f1_improvement_pp":100*(dpf-dbpf)},
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "final_backtest_summary.csv", index=False)
    result.to_csv(OUT_DIR / "final_backtest_by_review.csv", index=False)

    print("\n=== FINAL NON-ORACLE BACKTEST | 2023-08 TO 2026-05 ===")

    print("\nADDITIONS")
    print(
        f"Final calls: {fp} | Correct: {fh} | Actual additions: {fa}"
    )
    print(
        f"Precision: {pct(fpp)} | Recall: {pct(fpr)} | F1: {pct(fpf)}"
    )
    print(
        f"Methodology baseline: P {pct(bpp)}"
        f" | R {pct(bpr)} | F1 {pct(bpf)}"
    )
    print(
        f"F1 improvement: "
        f"{100 * (fpf - bpf):+.1f} pp"
        if pd.notna(fpf) and pd.notna(bpf)
        else "F1 improvement: NA"
    )

    print("\nDELETIONS")
    print(
        f"Final calls: {dp} | Correct: {dh} | Actual deletions: {da}"
    )
    print(
        f"Precision: {pct(dpp)} | Recall: {pct(dpr)} | F1: {pct(dpf)}"
    )
    print(
        "(Deletion layer remains methodology-only.)"
    )

    display = result.copy()

    display["Final Adds"] = (
        display["final_add_hits"].astype(int).astype(str)
        + "/"
        + display["final_add_predicted"].astype(int).astype(str)
    )

    display["Base Adds"] = (
        display["base_add_hits"].astype(int).astype(str)
        + "/"
        + display["base_add_predicted"].astype(int).astype(str)
    )

    display["Actual Adds"] = display["final_add_actual"].astype(int)

    display["Final P"] = display["final_add_precision"].map(pct)
    display["Final R"] = display["final_add_recall"].map(pct)
    display["Final F1"] = display["final_add_f1"].map(pct)

    display["Base F1"] = display["base_add_f1"].map(pct)

    print("\nADDITIONS BY REVIEW")
    print(
        display[
            [
                "review", "Final Adds", "Base Adds", "Actual Adds",
                "Final P", "Final R", "Final F1", "Base F1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUT_DIR / "final_backtest_summary.csv")
    print("Saved:", OUT_DIR / "final_backtest_by_review.csv")


if __name__ == "__main__":
    main()