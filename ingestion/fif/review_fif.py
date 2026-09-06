from pathlib import Path

import numpy as np
import pandas as pd


FIF_PATH = Path("data/raw/fif/fif.parquet")
INV_PATH = Path("data/processed/investability_universe.parquet")
OUT_PATH = Path("data/processed/review_fif.parquet")

PRICE_CUTOFF = {
    "2023-05": "2023-04-17",
    "2023-08": "2023-07-18",
    "2023-11": "2023-10-18",
    "2024-02": "2024-01-18",
    "2024-05": "2024-04-17",
    "2024-08": "2024-07-18",
    "2024-11": "2024-10-18",
    "2025-02": "2025-01-20",
    "2025-05": "2025-04-17",
    "2025-08": "2025-07-18",
    "2025-11": "2025-10-20",
    "2026-02": "2026-01-19",
    "2026-05": "2026-04-17",
}

PRE_FEB26_CUTOFF = pd.Timestamp("2025-10-20")


def latest(f, cutoff):
    z = f[f["available_date"].le(pd.Timestamp(cutoff))].copy()
    return (
        z.sort_values(["security_id", "available_date", "report_date"])
        .groupby("security_id", as_index=False)
        .tail(1)
    )


def enhanced_round(x):
    x = float(x)
    step = .025 if x > .25 else .005 if x >= .05 else .001
    return min(1.0, np.floor(x / step + .5) * step), step


def main():
    f = pd.read_parquet(FIF_PATH)
    inv = pd.read_parquet(INV_PATH)

    f["security_id"] = f["security_id"].astype("string")
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f["available_date"] = pd.to_datetime(f["available_date"], errors="coerce")

    inv["review"] = inv["review"].astype(str)
    inv["security_id"] = inv["security_id"].astype("string")

    required = {
        "security_id", "report_date", "available_date",
        "raw_free_float", "fif",
    }
    missing = required - set(f.columns)
    if missing:
        raise RuntimeError(f"fif.parquet missing: {sorted(missing)}")

    keys = inv[
        ["review", "security_id"]
        + [c for c in ["nse_symbol", "company_name"] if c in inv.columns]
    ].drop_duplicates(["review", "security_id"])

    prior_nov = latest(f, PRE_FEB26_CUTOFF)[
        ["security_id", "fif"]
    ].rename(columns={"fif": "prior_index_fif_proxy"})

    outputs = []
    feb_state = None

    for review, cutoff in PRICE_CUTOFF.items():
        cur = latest(f, cutoff)[
            [
                "security_id",
                "report_date",
                "available_date",
                "raw_free_float",
                "fif",
                "source",
            ]
        ].copy()

        cur = cur.rename(columns={
            "report_date": "fif_report_date",
            "available_date": "fif_available_date",
            "fif": "legacy_proforma_fif",
            "source": "fif_source",
        })

        r = keys[keys["review"].eq(review)].merge(
            cur,
            on="security_id",
            how="left",
            validate="one_to_one",
        )

        r["price_cutoff"] = pd.Timestamp(cutoff)
        r["fif_existing_min"] = np.nan
        r["fif_existing_max"] = np.nan
        r["fif_new_min"] = np.nan
        r["fif_new_max"] = np.nan
        r["fif_regime"] = "LEGACY_QCIR"

        legacy = pd.to_numeric(r["legacy_proforma_fif"], errors="coerce")

        if review < "2026-02":
            r["fif_existing_min"] = legacy
            r["fif_existing_max"] = legacy
            r["fif_new_min"] = legacy
            r["fif_new_max"] = legacy

        elif review == "2026-02":
            r["fif_regime"] = "FEB2026_TRANSITION"

            r = r.merge(
                prior_nov,
                on="security_id",
                how="left",
                validate="one_to_one",
            )

            prior = pd.to_numeric(
                r["prior_index_fif_proxy"], errors="coerce"
            )

            significant = (legacy - prior).abs().ge(.15)
            low_decrease = legacy.lt(.15) & legacy.lt(prior)
            update = significant | low_decrease

            existing = prior.where(~update, legacy)
            existing = existing.where(prior.notna())

            r["fif_existing_min"] = existing
            r["fif_existing_max"] = existing

            # New additions still used the then-current legacy methodology.
            r["fif_new_min"] = legacy
            r["fif_new_max"] = legacy

            feb_state = r[
                [
                    "security_id",
                    "fif_existing_min",
                    "fif_existing_max",
                    "fif_new_min",
                    "fif_new_max",
                ]
            ].copy()

        else:
            r["fif_regime"] = "MAY2026_ENHANCED"

            if feb_state is None:
                raise RuntimeError("Missing February-2026 FIF state.")

            r = r.merge(
                feb_state,
                on="security_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "_feb"),
            )

            rows = []

            for _, row in r.iterrows():
                raw = pd.to_numeric(
                    pd.Series([row["raw_free_float"]]),
                    errors="coerce",
                ).iloc[0]

                if pd.isna(raw):
                    rows.append((np.nan, np.nan, np.nan, np.nan, np.nan))
                    continue

                enhanced, step = enhanced_round(raw)

                priors = {
                    float(v)
                    for v in [
                        row["fif_existing_min_feb"],
                        row["fif_existing_max_feb"],
                        row["fif_new_min_feb"],
                        row["fif_new_max_feb"],
                    ]
                    if pd.notna(v)
                }

                # Migration/addition/deletion exception means the unbuffered
                # enhanced value must remain a valid branch.
                existing_values = {enhanced}

                for p in priors:
                    buffered = (
                        enhanced
                        if abs(raw - p) >= step - 1e-12
                        else p
                    )
                    existing_values.add(buffered)

                rows.append((
                    min(existing_values),
                    max(existing_values),
                    enhanced,
                    enhanced,
                    step,
                ))

            vals = pd.DataFrame(
                rows,
                index=r.index,
                columns=[
                    "_existing_min",
                    "_existing_max",
                    "_new_min",
                    "_new_max",
                    "fif_buffer_threshold",
                ],
            )

            r["fif_existing_min"] = vals["_existing_min"]
            r["fif_existing_max"] = vals["_existing_max"]
            r["fif_new_min"] = vals["_new_min"]
            r["fif_new_max"] = vals["_new_max"]
            r["fif_buffer_threshold"] = vals["fif_buffer_threshold"]

            r = r.drop(columns=[
                c for c in r.columns
                if c.endswith("_feb")
            ])

        r["fif_branch_uncertain"] = (
            pd.to_numeric(r["fif_existing_min"], errors="coerce")
            .ne(pd.to_numeric(r["fif_existing_max"], errors="coerce"))
            | pd.to_numeric(r["fif_existing_min"], errors="coerce")
            .ne(pd.to_numeric(r["fif_new_min"], errors="coerce"))
            | pd.to_numeric(r["fif_existing_max"], errors="coerce")
            .ne(pd.to_numeric(r["fif_new_max"], errors="coerce"))
        )

        outputs.append(r)

    out = pd.concat(outputs, ignore_index=True)

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate review FIF keys.")

    lookahead = (
        out["fif_available_date"].notna()
        & out["fif_available_date"].gt(out["price_cutoff"])
    )

    if lookahead.any():
        raise RuntimeError("Review FIF contains look-ahead.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    feb = out[out["review"].eq("2026-02")]
    may = out[out["review"].eq("2026-05")]
    hyu = may[may["security_id"].eq("SEC000872")]

    print("\n=== REVIEW-SPECIFIC FIF ===")
    print("Rows:", len(out))
    print("Duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("Lookahead:", int(lookahead.sum()))
    print(
        "Feb-2026 existing/new branch differences:",
        int(feb["fif_branch_uncertain"].sum()),
    )
    print(
        "May-2026 branch differences:",
        int(may["fif_branch_uncertain"].sum()),
    )
    print(
        "May-2026 raw FF missing:",
        int(may["raw_free_float"].isna().sum()),
    )

    print("\nHYUNDAI MAY-2026")
    print(
        hyu[
            [
                "fif_report_date",
                "fif_available_date",
                "raw_free_float",
                "legacy_proforma_fif",
                "fif_existing_min",
                "fif_existing_max",
                "fif_new_min",
                "fif_new_max",
                "fif_buffer_threshold",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()