from pathlib import Path

import numpy as np
import pandas as pd

from standard_assignment import prepare_assignment_state, assign_standard
from standard_final_requirements import evaluate_standard_final_requirements


MIEU_PATH = Path("data/processed/mieu_review_state.parquet")
SIZE_PATH = Path("data/processed/standard_size_segment.parquet")
EPI_PATH = Path("data/processed/extreme_price_screen.parquet")
OUT_PATH = Path("data/processed/standard_review_state.parquet")

REVIEWS = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05",
    "2024-08", "2024-11", "2025-02", "2025-05", "2025-08",
    "2025-11", "2026-02", "2026-05",
]


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def normalize(df):
    df = df.copy()

    if "review" in df.columns:
        df["review"] = df["review"].astype(str)

    if "security_id" in df.columns:
        df["security_id"] = df["security_id"].astype("string")

    if "company_key" in df.columns:
        df["company_key"] = df["company_key"].astype("string")

    return df


def company_keys(df, mask):
    return set(df.loc[mask, "company_key"].dropna())


def build_company_table(sec):
    x = sec.copy()

    x["company_full_mcap_usd"] = pd.to_numeric(
        x["company_full_mcap_usd"], errors="coerce"
    )

    if x["company_key"].isna().any():
        raise RuntimeError("Scenario MIEU security missing company_key.")

    if x["company_full_mcap_usd"].isna().any():
        raise RuntimeError("Scenario MIEU company missing full market cap.")

    inconsistent = (
        x.groupby("company_key")["company_full_mcap_usd"]
        .nunique(dropna=True)
        .gt(1)
    )

    if inconsistent.any():
        raise RuntimeError(
            "Inconsistent company_full_mcap_usd within company."
        )

    c = (
        x.groupby("company_key", as_index=False)
        .agg(
            segment_company_full_mcap_usd=(
                "company_full_mcap_usd", "first"
            ),
            company_security_count=("security_id", "nunique"),
        )
    )

    return c


def membership_mask(r, case):
    if case == "DEFINITE":
        return r["mieu_state"].eq("PASS")

    if case == "POSSIBLE":
        return r["mieu_state"].isin(["PASS", "UNRESOLVED"])

    raise RuntimeError(f"Unknown membership case: {case}")


def factor_column(case):
    if case == "FACTOR_MIN":
        return "post_adjusted_ff_mcap_usd_min"

    if case == "FACTOR_MAX":
        return "post_adjusted_ff_mcap_usd_max"

    raise RuntimeError(f"Unknown factor case: {case}")


def merge_epi(sec, epi, review):
    e = epi[epi["review"].eq(review)].copy()

    keep = [
        "security_id",
        "epi_status",
        "extreme_price_pass",
        "extreme_price_fail",
    ]

    for c in [
        "epi_applicable",
        "ipo_exempt",
        "epi_tested_period_count",
        "epi_breached_period_count",
    ]:
        if c in e.columns:
            keep.append(c)

    e = e[keep].drop_duplicates("security_id")

    if e["security_id"].duplicated().any():
        raise RuntimeError(f"{review}: duplicate EPI security rows.")

    return sec.merge(
        e,
        on="security_id",
        how="left",
        validate="one_to_one",
    )


def company_post_state(assigned, evaluated, prior_lower_keys, cutoff):
    c = assigned.copy()

    final = (
        evaluated[
            ["company_key", "final_standard_company_state"]
        ]
        .drop_duplicates()
    )

    if final["company_key"].duplicated().any():
        raise RuntimeError("Inconsistent company final Standard state.")

    c = c.merge(
        final,
        on="company_key",
        how="left",
        validate="one_to_one",
    )

    c["was_prior_lower"] = c["company_key"].isin(prior_lower_keys)

    lower_buffer_current_standard = (
        c["assignment_current_standard"]
        & c["provisional_standard"]
        & c["segment_company_full_mcap_usd"].lt(cutoff)
    )

    c["move_to_lower_if_final_fail"] = (
        c["was_prior_lower"]
        | lower_buffer_current_standard
    )

    c["post_review_assignment_state"] = pd.Series(
        pd.NA,
        index=c.index,
        dtype="string",
    )

    c.loc[
        c["provisional_lower_segment"],
        "post_review_assignment_state",
    ] = "LOWER_SEGMENT"

    c.loc[
        c["provisional_standard"]
        & c["final_standard_company_state"].eq("PASS"),
        "post_review_assignment_state",
    ] = "STANDARD"

    fail = (
        c["provisional_standard"]
        & c["final_standard_company_state"].eq("FAIL")
    )

    c.loc[
        fail & c["move_to_lower_if_final_fail"],
        "post_review_assignment_state",
    ] = "LOWER_SEGMENT"

    c.loc[
        fail & ~c["move_to_lower_if_final_fail"],
        "post_review_assignment_state",
    ] = "ASSIGNED_STANDARD_FAILED"

    c.loc[
        c["provisional_standard"]
        & c["final_standard_company_state"].eq("UNRESOLVED"),
        "post_review_assignment_state",
    ] = "UNRESOLVED"

    if c["post_review_assignment_state"].isna().any():
        bad = c.loc[
            c["post_review_assignment_state"].isna(),
            [
                "company_key",
                "provisional_standard",
                "final_standard_company_state",
            ],
        ]

        raise RuntimeError(
            "Could not classify post-review company state:\n"
            + bad.head(20).to_string(index=False)
        )

    return c


def next_state_sets(company_state):
    key = company_state["company_key"]

    lower_definite = set(
        key[
            company_state["post_review_assignment_state"]
            .eq("LOWER_SEGMENT")
        ]
    )

    failed_definite = set(
        key[
            company_state["post_review_assignment_state"]
            .eq("ASSIGNED_STANDARD_FAILED")
        ]
    )

    unresolved = company_state[
        company_state["post_review_assignment_state"]
        .eq("UNRESOLVED")
    ]

    unresolved_lower = set(
        unresolved.loc[
            unresolved["move_to_lower_if_final_fail"],
            "company_key",
        ]
    )

    unresolved_failed = set(
        unresolved.loc[
            ~unresolved["move_to_lower_if_final_fail"],
            "company_key",
        ]
    )

    return {
        "lower_definite": lower_definite,
        "lower_possible": lower_definite | unresolved_lower,
        "failed_definite": failed_definite,
        "failed_possible": failed_definite | unresolved_failed,
        "unassigned_definite": set(),
        "unassigned_possible": set(),
    }


def main():
    mieu = normalize(pd.read_parquet(MIEU_PATH))
    size = normalize(pd.read_parquet(SIZE_PATH))
    epi = normalize(pd.read_parquet(EPI_PATH))

    require(
        mieu,
        [
            "review",
            "security_id",
            "nse_symbol",
            "company_key",
            "company_full_mcap_usd",
            "full_market_cap_usd",
            "ff_market_cap_usd",
            "fif",
            "was_standard_constituent",
            "prior_imi",
            "prior_imi_uncertain",
            "mieu_state",
            "pre_adjusted_fif_min",
            "pre_adjusted_fif_max",
            "pre_adjusted_ff_mcap_usd_min",
            "pre_adjusted_ff_mcap_usd_max",
            "post_adjusted_ff_mcap_usd_min",
            "post_adjusted_ff_mcap_usd_max",
        ],
        "mieu_review_state",
    )

    require(
        size,
        [
            "review",
            "scenario_id",
            "membership_case",
            "factor_case",
            "interim_case",
            "mieu_company_count",
            "final_standard_segment_number",
            "market_standard_cutoff_usd",
            "standard_gmsr_usd",
        ],
        "standard_size_segment",
    )

    require(
        epi,
        [
            "review",
            "security_id",
            "epi_status",
            "extreme_price_pass",
            "extreme_price_fail",
        ],
        "extreme_price_screen",
    )

    if mieu.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate MIEU review/security keys.")

    if size.duplicated(["review", "scenario_id"]).any():
        raise RuntimeError("Duplicate size review/scenario keys.")

    if epi.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate EPI review/security keys.")

    state = {}
    outputs = []
    stats = []

    for review in REVIEWS:
        r = mieu[mieu["review"].eq(review)].copy()
        sz = size[size["review"].eq(review)].copy()

        if r.empty or sz.empty:
            raise RuntimeError(f"{review}: missing review inputs.")

        if len(sz) != 8:
            raise RuntimeError(
                f"{review}: expected 8 size scenarios, got {len(sz)}."
            )

        current_standard_keys = company_keys(
            r,
            r["was_standard_constituent"]
            .fillna(False)
            .astype(bool),
        )

        if review == "2023-05":
            bootstrap_prior_mieu = company_keys(
                r,
                r["prior_imi"].fillna(False).astype(bool),
            )

            bootstrap_prior_lower = (
                bootstrap_prior_mieu - current_standard_keys
            )

        for _, srow in sz.sort_values("scenario_id").iterrows():
            scenario_id = str(srow["scenario_id"])
            membership_case = str(srow["membership_case"])
            factor_case = str(srow["factor_case"])

            mask = membership_mask(r, membership_case)
            sec = r[mask].copy()

            if sec.empty:
                raise RuntimeError(
                    f"{review} {scenario_id}: empty scenario MIEU."
                )

            companies = build_company_table(sec)

            expected_companies = int(srow["mieu_company_count"])

            if len(companies) != expected_companies:
                raise RuntimeError(
                    f"{review} {scenario_id}: company count "
                    f"{len(companies)} != size-stage "
                    f"{expected_companies}."
                )

            current_mieu_keys = set(companies["company_key"])

            if review == "2023-05":
                prior_mieu_keys = bootstrap_prior_mieu
                prior_lower_keys = bootstrap_prior_lower
                prior_failed_keys = set()
                prior_unassigned_keys = set()
                prior_state_source = "MAY2023_BOOTSTRAP"

            else:
                if scenario_id not in state:
                    raise RuntimeError(
                        f"{review} {scenario_id}: previous state missing."
                    )

                prev = state[scenario_id]
                prior_mieu_keys = prev["mieu_keys"]

                possible = membership_case == "POSSIBLE"

                prior_lower_keys = prev[
                    "lower_possible"
                    if possible
                    else "lower_definite"
                ]

                prior_failed_keys = prev[
                    "failed_possible"
                    if possible
                    else "failed_definite"
                ]

                prior_unassigned_keys = prev[
                    "unassigned_possible"
                    if possible
                    else "unassigned_definite"
                ]

                prior_state_source = (
                    "PREVIOUS_REVIEW_MODEL_STATE"
                )

            newly_investable_keys = (
                current_mieu_keys
                - prior_mieu_keys
                - current_standard_keys
                - prior_failed_keys
            )

            prepared = prepare_assignment_state(
                companies=companies,
                current_standard_keys=current_standard_keys,
                prior_standard_failed_keys=prior_failed_keys,
                newly_investable_keys=newly_investable_keys,
                prior_lower_keys=prior_lower_keys,
                prior_lower_failed_keys=set(),
                prior_unassigned_keys=prior_unassigned_keys,
            )

            segment_number = int(
                srow["final_standard_segment_number"]
            )

            cutoff = float(
                srow["market_standard_cutoff_usd"]
            )

            assigned = assign_standard(
                prepared,
                segment_number,
                cutoff,
            )

            if int(assigned["provisional_standard"].sum()) != segment_number:
                raise RuntimeError(
                    f"{review} {scenario_id}: assignment count mismatch."
                )

            assign_cols = [
                "company_key",
                "segment_company_full_mcap_usd",
                "assignment_current_standard",
                "assignment_prior_standard_failed",
                "assignment_newly_investable",
                "assignment_prior_lower",
                "assignment_prior_lower_failed",
                "assignment_prior_unassigned",
                "assignment_standard_state",
                "assignment_lower_or_unassigned_state",
                "standard_lower_buffer_usd",
                "lower_segment_upper_buffer_usd",
                "assignment_priority",
                "assignment_reason",
                "provisional_standard",
                "provisional_lower_segment",
                "assignment_selected_rank",
                "assignment_continuity_retained",
                "assignment_new_standard_candidate",
            ]

            sec = sec.merge(
                assigned[assign_cols],
                on="company_key",
                how="left",
                validate="many_to_one",
            )

            if sec["provisional_standard"].isna().any():
                raise RuntimeError(
                    f"{review} {scenario_id}: assignment mapping failed."
                )

            factor_col = factor_column(factor_case)
            suffix = "min" if factor_case == "FACTOR_MIN" else "max"

            selected_ff = pd.to_numeric(
                sec[factor_col],
                errors="coerce",
            )
            selected_pre_ff = pd.to_numeric(
                sec[f"pre_adjusted_ff_mcap_usd_{suffix}"],
                errors="coerce",
            )
            selected_pre_fif = pd.to_numeric(
                sec[f"pre_adjusted_fif_{suffix}"],
                errors="coerce",
            )

            missing_selected = (
                selected_ff.isna()
                | selected_pre_ff.isna()
                | selected_pre_fif.isna()
            )

            if missing_selected.any():
                bad = sec.loc[
                    missing_selected,
                    ["security_id", "nse_symbol", "mieu_state"],
                ]

                raise RuntimeError(
                    f"{review} {scenario_id}: selected FIF/FF branch missing:\n"
                    + bad.head(20).to_string(index=False)
                )

            # Fix both the review-FIF and foreign-room uncertainty to this scenario.
            sec["fif"] = selected_pre_fif
            sec["ff_market_cap_usd"] = selected_pre_ff
            sec["post_adjusted_ff_mcap_usd_min"] = selected_ff
            sec["post_adjusted_ff_mcap_usd_max"] = selected_ff

            sec = merge_epi(sec, epi, review)

            sec["extreme_price_pass"] = (
                sec["extreme_price_pass"].astype("boolean")
            )

            sec["extreme_price_fail"] = (
                sec["extreme_price_fail"].astype("boolean")
            )

            predicted_addition = (
                sec["provisional_standard"].fillna(False).astype(bool)
                & ~sec["was_standard_constituent"]
                .fillna(False)
                .astype(bool)
            )

            # An old NOT_APPLICABLE row cannot be treated as a pass
            # when our recursive model now makes it a Standard addition.
            epi_not_tested = (
                predicted_addition
                & (
                    sec["epi_status"].isna()
                    | sec["epi_status"].eq("NOT_APPLICABLE")
                )
            )

            sec.loc[
                epi_not_tested,
                "extreme_price_pass",
            ] = pd.NA

            sec.loc[
                epi_not_tested,
                "extreme_price_fail",
            ] = pd.NA

            evaluated = evaluate_standard_final_requirements(
                securities=sec,
                market_standard_cutoff_usd=cutoff,
                standard_gmsr_usd=float(
                    srow["standard_gmsr_usd"]
                ),
            )

            company_state = company_post_state(
                assigned=assigned,
                evaluated=evaluated,
                prior_lower_keys=prior_lower_keys,
                cutoff=cutoff,
            )

            next_sets = next_state_sets(company_state)

            state[scenario_id] = {
                "mieu_keys": current_mieu_keys,
                **next_sets,
            }

            company_output = company_state[
                [
                    "company_key",
                    "post_review_assignment_state",
                    "move_to_lower_if_final_fail",
                ]
            ]

            full_cols = [
                "review",
                "security_id",
                "nse_symbol",
                "company_key",
                "was_standard_constituent",
                "mieu_state",
                "prior_imi",
                "prior_imi_uncertain",
            ]

            if "company_name" in r.columns:
                full_cols.insert(3, "company_name")

            full = r[full_cols].copy()

            included_ids = set(sec["security_id"])

            full["scenario_id"] = scenario_id
            full["membership_case"] = membership_case
            full["factor_case"] = factor_case
            full["interim_case"] = str(srow["interim_case"])
            full["scenario_mieu_included"] = (
                full["security_id"].isin(included_ids)
            )
            full["prior_assignment_state_source"] = prior_state_source
            full["final_standard_segment_number"] = segment_number
            full["market_standard_cutoff_usd"] = cutoff
            full["standard_gmsr_usd"] = float(
                srow["standard_gmsr_usd"]
            )

            eval_keep = [
                "security_id",
                "provisional_standard",
                "provisional_lower_segment",
                "assignment_priority",
                "assignment_reason",
                "final_ff_threshold_basis_usd",
                "final_min_ff_mcap_usd",
                "final_required_ff_mcap_usd",
                "final_ff_mcap_basis",
                "final_ff_mcap_evaluation_min_usd",
                "final_ff_mcap_evaluation_max_usd",
                "final_ff_requirement_state",
                "final_epi_applicable",
                "final_epi_state",
                "final_standard_security_state",
                "final_standard_security_pass",
                "final_standard_reason",
                "final_standard_company_state",
                "final_standard_company_pass",
                "epi_blocks_standard_migration",
            ]

            full = full.merge(
                evaluated[eval_keep],
                on="security_id",
                how="left",
                validate="one_to_one",
            )

            full = full.merge(
                company_output,
                on="company_key",
                how="left",
                validate="many_to_one",
            )

            full["review_standard_security_state"] = pd.Series(
                "OUT_OF_MIEU",
                index=full.index,
                dtype="string",
            )

            in_mieu = full["scenario_mieu_included"]

            full.loc[
                in_mieu
                & full["provisional_standard"].eq(False).fillna(False),
                "review_standard_security_state",
            ] = "LOWER_SEGMENT"

            for final_state in ["PASS", "FAIL", "UNRESOLVED"]:
                full.loc[
                    in_mieu
                    & full["final_standard_security_state"].eq(
                        final_state
                    ).fillna(False),
                    "review_standard_security_state",
                ] = final_state

            full["predicted_standard"] = pd.array(
                np.where(
                    full["review_standard_security_state"].eq("PASS"),
                    True,
                    np.where(
                        full["review_standard_security_state"].eq(
                            "UNRESOLVED"
                        ),
                        pd.NA,
                        False,
                    ),
                ),
                dtype="boolean",
            )

            full["continuity_applied"] = False
            full["continuity_status"] = (
                "PENDING_SEPARATE_STANDARD_CONTINUITY_STAGE"
            )

            outputs.append(full)

            stats.append(
                {
                    "review": review,
                    "scenario_id": scenario_id,
                    "standard_companies": int(
                        company_state[
                            "post_review_assignment_state"
                        ].eq("STANDARD").sum()
                    ),
                    "lower_companies": int(
                        company_state[
                            "post_review_assignment_state"
                        ].eq("LOWER_SEGMENT").sum()
                    ),
                    "assigned_failed_companies": int(
                        company_state[
                            "post_review_assignment_state"
                        ].eq("ASSIGNED_STANDARD_FAILED").sum()
                    ),
                    "unresolved_companies": int(
                        company_state[
                            "post_review_assignment_state"
                        ].eq("UNRESOLVED").sum()
                    ),
                    "standard_securities": int(
                        evaluated[
                            "final_standard_security_state"
                        ].eq("PASS").sum()
                    ),
                    "unresolved_securities": int(
                        evaluated[
                            "final_standard_security_state"
                        ].eq("UNRESOLVED").sum()
                    ),
                    "epi_not_tested_additions": int(
                        epi_not_tested.sum()
                    ),
                }
            )

    out = pd.concat(outputs, ignore_index=True)

    if out.duplicated(
        ["review", "scenario_id", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate review/scenario/security output rows."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    scenario_stats = []

    for (review, scenario_id), g in out.groupby(
        ["review", "scenario_id"],
        sort=True,
    ):
        companies = (
            g[
                [
                    "company_key",
                    "post_review_assignment_state",
                ]
            ]
            .dropna(subset=["post_review_assignment_state"])
            .drop_duplicates("company_key")
        )

        scenario_stats.append(
            {
                "review": review,
                "scenario_id": scenario_id,
                "standard_companies": int(
                    companies["post_review_assignment_state"]
                    .eq("STANDARD")
                    .sum()
                ),
                "lower_companies": int(
                    companies["post_review_assignment_state"]
                    .eq("LOWER_SEGMENT")
                    .sum()
                ),
                "assigned_failed_companies": int(
                    companies["post_review_assignment_state"]
                    .eq("ASSIGNED_STANDARD_FAILED")
                    .sum()
                ),
                "unresolved_companies": int(
                    companies["post_review_assignment_state"]
                    .eq("UNRESOLVED")
                    .sum()
                ),
                "standard_securities": int(
                    g["predicted_standard"].eq(True).sum()
                ),
                "unresolved_securities": int(
                    g["predicted_standard"].isna().sum()
                ),
            }
        )

    stats = pd.DataFrame(scenario_stats)

    summary = (
    stats.groupby("review", as_index=False)
    .agg(
        scenarios=("scenario_id", "size"),
        standard_companies_min=("standard_companies", "min"),
        standard_companies_max=("standard_companies", "max"),
        standard_securities_min=("standard_securities", "min"),
        standard_securities_max=("standard_securities", "max"),
        unresolved_companies_max=("unresolved_companies", "max"),
        unresolved_securities_max=("unresolved_securities", "max"),
        assigned_failed_max=("assigned_failed_companies", "max"),
    )
)
    print("\n--- STANDARD REVIEW STATE ---")
    print(summary.to_string(index=False))

    print("\nRows:", len(out))
    print(
        "Duplicates:",
        int(
            out.duplicated(
                ["review", "scenario_id", "security_id"]
            ).sum()
        ),
    )
    print("Continuity stage: PENDING")
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()