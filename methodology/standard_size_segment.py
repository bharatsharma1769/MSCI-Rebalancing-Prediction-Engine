from pathlib import Path
import math

import numpy as np
import pandas as pd


IN_PATH = Path("data/processed/mieu_review_state.parquet")
SIZE_PATH = Path("data/processed/india_size_cutoffs.parquet")
OUT_PATH = Path("data/processed/standard_size_segment.parquet")

REVIEWS = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05",
    "2024-08", "2024-11", "2025-02", "2025-05", "2025-08",
    "2025-11", "2026-02", "2026-05",
]

COVERAGE_LOWER = 0.80
COVERAGE_UPPER = 0.90
GMSR_LOWER = 0.50
LOWER_PROX_UPPER = 0.575
UPPER_PROX_LOWER = 1.00
GMSR_UPPER = 1.15


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def company_table(x, factor_col):
    z = x.copy()

    for c in ["company_full_mcap_usd", factor_col]:
        z[c] = pd.to_numeric(z[c], errors="coerce")

    if z["company_key"].isna().any():
        raise RuntimeError("MIEU security missing company_key.")

    if z["company_full_mcap_usd"].isna().any():
        raise RuntimeError("Included MIEU company missing full market cap.")

    if z[factor_col].isna().any():
        bad = z.loc[
            z[factor_col].isna(),
            ["review", "security_id", "nse_symbol", "mieu_state"],
        ]
        raise RuntimeError(
            f"Included MIEU security missing {factor_col}:\n"
            + bad.head(30).to_string(index=False)
        )

    inconsistent = (
        z.groupby("company_key")["company_full_mcap_usd"]
        .nunique(dropna=True)
        .gt(1)
    )
    if inconsistent.any():
        raise RuntimeError(
            "Inconsistent company_full_mcap_usd within company."
        )

    companies = (
        z.groupby("company_key", as_index=False)
        .agg(
            segment_company_full_mcap_usd=(
                "company_full_mcap_usd", "first"
            ),
            segment_company_ff_mcap_usd=(
                factor_col, "sum"
            ),
            company_security_count=(
                "security_id", "nunique"
            ),
            company_has_existing_standard=(
                "company_was_standard_constituent", "max"
            ),
        )
    )

    companies = companies[
        companies["segment_company_full_mcap_usd"].gt(0)
        & companies["segment_company_ff_mcap_usd"].ge(0)
    ].copy()

    companies = companies.sort_values(
        [
            "segment_company_full_mcap_usd",
            "segment_company_ff_mcap_usd",
            "company_key",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    companies["company_rank"] = np.arange(1, len(companies) + 1)

    total_ff = companies["segment_company_ff_mcap_usd"].sum()
    if not np.isfinite(total_ff) or total_ff <= 0:
        raise RuntimeError("Invalid MIEU FF market-cap denominator.")

    companies["cumulative_ff_mcap_usd"] = (
        companies["segment_company_ff_mcap_usd"].cumsum()
    )
    companies["cumulative_ff_coverage"] = (
        companies["cumulative_ff_mcap_usd"] / total_ff
    )
    companies["market_total_ff_mcap_usd"] = total_ff

    return companies


def company_row(companies, n):
    if n < 1 or n > len(companies):
        return None
    return companies.iloc[n - 1]


def size_coverage_ok(mcap, coverage, gmsr):
    return (
        GMSR_LOWER * gmsr <= mcap <= GMSR_UPPER * gmsr
        and COVERAGE_LOWER <= coverage <= COVERAGE_UPPER
    )


def proximity_ok(mcap, coverage, gmsr):
    lower = (
        GMSR_LOWER * gmsr <= mcap <= LOWER_PROX_UPPER * gmsr
        and coverage <= COVERAGE_UPPER
    )
    upper = (
        UPPER_PROX_LOWER * gmsr <= mcap <= GMSR_UPPER * gmsr
        and coverage >= COVERAGE_LOWER
    )
    return lower or upper


def initial_segment_number(companies, interim, gmsr):
    lower = GMSR_LOWER * gmsr

    if interim >= lower:
        n = int(
            companies["segment_company_full_mcap_usd"]
            .ge(interim)
            .sum()
        )
        method = "UPDATED_MIEU_ABOVE_INTERIM"

    else:
        above_lower = (
            companies["segment_company_full_mcap_usd"] >= lower
        )

        prior_standard_between = (
            companies["company_has_existing_standard"].astype(bool)
            & companies["segment_company_full_mcap_usd"].lt(lower)
            & companies["segment_company_full_mcap_usd"].ge(interim)
        )

        n = int(above_lower.sum() + prior_standard_between.sum())
        method = "GMSR_LOWER_PLUS_PRIOR_STANDARD_BETWEEN"

    n = max(1, min(n, len(companies)))
    return n, method


def increase_segment_number(companies, initial_n, gmsr):
    n = initial_n
    upper = GMSR_UPPER * gmsr
    lower_prox = LOWER_PROX_UPPER * gmsr

    above_upper = companies[
        companies["segment_company_full_mcap_usd"] > upper
    ]

    if not above_upper.empty:
        n = max(n, int(above_upper["company_rank"].max()))

    while n < len(companies):
        current = company_row(companies, n)

        if current["cumulative_ff_coverage"] >= COVERAGE_LOWER:
            break

        nxt = companies.iloc[n]

        if nxt["segment_company_full_mcap_usd"] <= lower_prox:
            break

        n += 1

    row = company_row(companies, n)
    cutoff = min(float(row["segment_company_full_mcap_usd"]), upper)

    return n, cutoff


def reduce_segment_number(companies, initial_n, gmsr):
    n = initial_n
    lower = GMSR_LOWER * gmsr
    upper_prox = UPPER_PROX_LOWER * gmsr

    original = company_row(companies, initial_n)
    original_mcap = float(original["segment_company_full_mcap_usd"])

    max_first = max(1, math.floor(initial_n * 0.05))
    removed_first = 0

    while n > 1 and removed_first < max_first:
        current = company_row(companies, n)
        mcap = float(current["segment_company_full_mcap_usd"])
        coverage = float(current["cumulative_ff_coverage"])

        if size_coverage_ok(mcap, coverage, gmsr):
            break

        if mcap >= upper_prox:
            break

        previous = company_row(companies, n - 1)
        previous_cov = float(previous["cumulative_ff_coverage"])

        if (
            lower <= mcap <= GMSR_UPPER * gmsr
            and coverage > COVERAGE_UPPER
            and previous_cov < COVERAGE_LOWER
        ):
            break

        n -= 1
        removed_first += 1

    current = company_row(companies, n)

    if current["segment_company_full_mcap_usd"] >= lower:
        return n, float(current["segment_company_full_mcap_usd"]), False

    lo, hi = sorted([original_mcap, lower])

    tail_ff = companies.loc[
        companies["segment_company_full_mcap_usd"]
        .between(lo, hi, inclusive="both"),
        "segment_company_ff_mcap_usd",
    ].sum()

    removed_ff = companies.loc[
        companies["company_rank"].gt(n)
        & companies["company_rank"].le(initial_n),
        "segment_company_ff_mcap_usd",
    ].sum()

    if tail_ff > 0 and removed_ff >= 0.5 * tail_ff:
        return n, lower, True

    max_total = max(max_first, math.floor(initial_n * 0.20))
    minimum_n = max(1, initial_n - max_total)

    while n > minimum_n:
        candidate_n = n - 1

        candidate_removed_ff = companies.loc[
            companies["company_rank"].gt(candidate_n)
            & companies["company_rank"].le(initial_n),
            "segment_company_ff_mcap_usd",
        ].sum()

        if tail_ff > 0 and candidate_removed_ff > 0.5 * tail_ff:
            break

        n = candidate_n

    row = company_row(companies, n)
    cutoff = max(float(row["segment_company_full_mcap_usd"]), lower)

    return n, cutoff, True


def adjust_segment_number(companies, initial_n, gmsr):
    row = company_row(companies, initial_n)
    mcap = float(row["segment_company_full_mcap_usd"])
    coverage = float(row["cumulative_ff_coverage"])

    if size_coverage_ok(mcap, coverage, gmsr) or proximity_ok(
        mcap, coverage, gmsr
    ):
        return initial_n, mcap, coverage, "NO_CHANGE", False

    upper = GMSR_UPPER * gmsr

    if mcap > upper:
        between = companies[
            companies["segment_company_full_mcap_usd"].lt(mcap)
            & companies["segment_company_full_mcap_usd"].gt(upper)
        ]

        if between.empty:
            return (
                initial_n,
                mcap,
                coverage,
                "NO_CHANGE_ABOVE_GMSR_NO_COMPANY_GAP",
                False,
            )

    if mcap > upper or coverage < COVERAGE_LOWER:
        n, cutoff = increase_segment_number(
            companies, initial_n, gmsr
        )
        r = company_row(companies, n)

        return (
            n,
            cutoff,
            float(r["cumulative_ff_coverage"]),
            "INCREASE_SEGMENT_NUMBER" if n > initial_n else "NO_CHANGE",
            False,
        )

    if mcap < GMSR_LOWER * gmsr or coverage > COVERAGE_UPPER:
        n, cutoff, limited = reduce_segment_number(
            companies, initial_n, gmsr
        )
        r = company_row(companies, n)

        return (
            n,
            cutoff,
            float(r["cumulative_ff_coverage"]),
            "REDUCE_SEGMENT_NUMBER" if n < initial_n else "NO_CHANGE",
            limited,
        )

    return initial_n, mcap, coverage, "NO_CHANGE", False


def main():
    x = pd.read_parquet(IN_PATH)
    size = pd.read_parquet(SIZE_PATH)

    require(
        x,
        [
            "review",
            "security_id",
            "company_key",
            "full_market_cap_usd",
            "company_full_mcap_usd",
            "was_standard_constituent",
            "mieu_state",
            "post_adjusted_ff_mcap_usd_min",
            "post_adjusted_ff_mcap_usd_max",
            "interim_standard_cutoff_min_usd",
            "interim_standard_cutoff_max_usd",
        ],
        "mieu_review_state",
    )

    require(
        size,
        ["review", "standard_gmsr_usd"],
        "india_size_cutoffs",
    )

    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")

    if x.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate MIEU state keys.")

    gmsr_map = (
        size.assign(review=size["review"].astype(str))
        .drop_duplicates("review")
        .set_index("review")["standard_gmsr_usd"]
        .to_dict()
    )

    outputs = []

    for review in REVIEWS:
        r = x[x["review"].eq(review)].copy()

        r["company_was_standard_constituent"] = (
        r.groupby("company_key")["was_standard_constituent"]
        .transform(lambda s: s.fillna(False).astype(bool).any())
        )

        if r.empty:
            raise RuntimeError(f"{review}: no MIEU rows.")
        if review not in gmsr_map:
            raise RuntimeError(f"{review}: missing GMSR.")

        gmsr = float(gmsr_map[review])

        interim_min = float(
            r["interim_standard_cutoff_min_usd"].dropna().iloc[0]
        )
        interim_max = float(
            r["interim_standard_cutoff_max_usd"].dropna().iloc[0]
        )

        if r["interim_standard_cutoff_min_usd"].nunique(dropna=True) != 1:
            raise RuntimeError(f"{review}: inconsistent Interim min.")
        if r["interim_standard_cutoff_max_usd"].nunique(dropna=True) != 1:
            raise RuntimeError(f"{review}: inconsistent Interim max.")

        scenarios = [
            ("DEFINITE", r["mieu_state"].eq("PASS")),
            (
                "POSSIBLE",
                r["mieu_state"].isin(["PASS", "UNRESOLVED"]),
            ),
        ]

        for membership_name, membership_mask in scenarios:
            sec = r[membership_mask].copy()

            if sec.empty:
                raise RuntimeError(
                    f"{review} {membership_name}: empty MIEU."
                )

            for factor_name, factor_col in [
                ("FACTOR_MIN", "post_adjusted_ff_mcap_usd_min"),
                ("FACTOR_MAX", "post_adjusted_ff_mcap_usd_max"),
            ]:
                companies = company_table(sec, factor_col)

                for interim_name, interim in [
                    ("INTERIM_MIN", interim_min),
                    ("INTERIM_MAX", interim_max),
                ]:
                    initial_n, initial_method = initial_segment_number(
                        companies,
                        interim,
                        gmsr,
                    )

                    final_n, cutoff, coverage, action, limited = (
                        adjust_segment_number(
                            companies,
                            initial_n,
                            gmsr,
                        )
                    )

                    outputs.append({
                        "review": review,
                        "scenario_id": (
                            f"{membership_name}_"
                            f"{factor_name}_"
                            f"{interim_name}"
                        ),
                        "membership_case": membership_name,
                        "factor_case": factor_name,
                        "interim_case": interim_name,
                        "mieu_security_count": len(sec),
                        "mieu_company_count": len(companies),
                        "standard_gmsr_usd": gmsr,
                        "interim_standard_cutoff_usd": interim,
                        "initial_standard_segment_number": initial_n,
                        "initial_segment_number_method": initial_method,
                        "final_standard_segment_number": final_n,
                        "market_standard_cutoff_usd": cutoff,
                        "final_standard_coverage": coverage,
                        "segment_number_action": action,
                        "reduction_limit_applied": limited,
                    })

    out = pd.DataFrame(outputs)

    if out.duplicated(["review", "scenario_id"]).any():
        raise RuntimeError("Duplicate review/scenario rows.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    summary = (
        out.groupby("review", as_index=False)
        .agg(
            scenarios=("scenario_id", "size"),
            mieu_companies_min=("mieu_company_count", "min"),
            mieu_companies_max=("mieu_company_count", "max"),
            initial_n_min=("initial_standard_segment_number", "min"),
            initial_n_max=("initial_standard_segment_number", "max"),
            final_n_min=("final_standard_segment_number", "min"),
            final_n_max=("final_standard_segment_number", "max"),
            cutoff_min_bn=(
                "market_standard_cutoff_usd",
                lambda s: s.min() / 1e9,
            ),
            cutoff_max_bn=(
                "market_standard_cutoff_usd",
                lambda s: s.max() / 1e9,
            ),
            coverage_min=("final_standard_coverage", "min"),
            coverage_max=("final_standard_coverage", "max"),
        )
    )

    summary["segment_state_unresolved"] = (
        summary["final_n_min"].ne(summary["final_n_max"])
        | ~np.isclose(
            summary["cutoff_min_bn"],
            summary["cutoff_max_bn"],
            rtol=0,
            atol=1e-9,
        )
    )

    print("\n--- STANDARD SIZE SEGMENT ---")
    print(summary.to_string(index=False))

    print("\nRows:", len(out))
    print(
        "Duplicates:",
        int(out.duplicated(["review", "scenario_id"]).sum()),
    )
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()