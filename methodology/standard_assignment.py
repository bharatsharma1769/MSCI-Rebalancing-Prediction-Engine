import numpy as np
import pandas as pd


STANDARD_LOWER_BUFFER = 2.0 / 3.0
LOWER_SEGMENT_UPPER_BUFFER = 1.5


def require(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"standard_assignment missing: {missing}")


def prepare_assignment_state(
    companies,
    current_standard_keys,
    prior_standard_failed_keys,
    newly_investable_keys,
    prior_lower_keys,
    prior_lower_failed_keys,
    prior_unassigned_keys,
):
    c = companies.copy()
    keys = c["company_key"]

    current_standard_keys = set(current_standard_keys)
    prior_standard_failed_keys = set(prior_standard_failed_keys)
    newly_investable_keys = set(newly_investable_keys)
    prior_lower_keys = set(prior_lower_keys)
    prior_lower_failed_keys = set(prior_lower_failed_keys)
    prior_unassigned_keys = set(prior_unassigned_keys)

    c["assignment_current_standard"] = keys.isin(current_standard_keys)
    c["assignment_prior_standard_failed"] = keys.isin(prior_standard_failed_keys)
    c["assignment_newly_investable"] = keys.isin(newly_investable_keys)
    c["assignment_prior_lower"] = keys.isin(prior_lower_keys)
    c["assignment_prior_lower_failed"] = keys.isin(prior_lower_failed_keys)
    c["assignment_prior_unassigned"] = keys.isin(prior_unassigned_keys)

    c["assignment_standard_state"] = (
        c["assignment_current_standard"]
        | c["assignment_prior_standard_failed"]
    )

    c["assignment_lower_or_unassigned_state"] = (
        c["assignment_prior_lower"]
        | c["assignment_prior_lower_failed"]
        | c["assignment_prior_unassigned"]
    )

    return c


def assign_standard(companies, segment_number, cutoff_usd):
    require(
        companies,
        [
            "company_key",
            "segment_company_full_mcap_usd",
            "assignment_standard_state",
            "assignment_newly_investable",
            "assignment_lower_or_unassigned_state",
            "assignment_prior_lower",
        ],
    )

    c = companies.copy()

    c["segment_company_full_mcap_usd"] = pd.to_numeric(
        c["segment_company_full_mcap_usd"], errors="coerce"
    )

    if c["company_key"].isna().any():
        raise RuntimeError("Assignment company missing company_key.")

    if c["segment_company_full_mcap_usd"].isna().any():
        raise RuntimeError("Assignment company missing full market cap.")

    segment_number = int(segment_number)
    cutoff_usd = float(cutoff_usd)

    if segment_number < 1:
        raise RuntimeError("Invalid Standard Segment Number.")

    if segment_number > len(c):
        raise RuntimeError(
            f"Standard Segment Number {segment_number} exceeds "
            f"current MIEU companies {len(c)}."
        )

    lower_buffer = STANDARD_LOWER_BUFFER * cutoff_usd
    upper_buffer = LOWER_SEGMENT_UPPER_BUFFER * cutoff_usd
    mcap = c["segment_company_full_mcap_usd"]

    c["standard_lower_buffer_usd"] = lower_buffer
    c["lower_segment_upper_buffer_usd"] = upper_buffer
    c["assignment_priority"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    c["assignment_reason"] = pd.Series(pd.NA, index=c.index, dtype="string")

    rules = [
        (
            c["assignment_standard_state"] & mcap.ge(cutoff_usd),
            1,
            "CURRENT_STANDARD_OR_ASSIGNED_FAILED_ABOVE_CUTOFF",
        ),
        (
            c["assignment_newly_investable"] & mcap.ge(cutoff_usd),
            2,
            "NEWLY_INVESTABLE_ABOVE_CUTOFF",
        ),
        (
            c["assignment_lower_or_unassigned_state"] & mcap.gt(upper_buffer),
            3,
            "PRIOR_LOWER_OR_UNASSIGNED_ABOVE_UPPER_BUFFER",
        ),
        (
            c["assignment_standard_state"]
            & mcap.lt(cutoff_usd)
            & mcap.ge(lower_buffer),
            4,
            "CURRENT_STANDARD_OR_ASSIGNED_FAILED_IN_LOWER_BUFFER",
        ),
        (
            c["assignment_prior_lower"]
            & mcap.ge(cutoff_usd)
            & mcap.le(upper_buffer),
            5,
            "PRIOR_LOWER_IN_UPPER_BUFFER",
        ),
    ]

    for mask, priority, reason in rules:
        use = c["assignment_priority"].isna() & mask
        c.loc[use, "assignment_priority"] = priority
        c.loc[use, "assignment_reason"] = reason

    ordered = c[c["assignment_priority"].notna()].sort_values(
        [
            "assignment_priority",
            "segment_company_full_mcap_usd",
            "company_key",
        ],
        ascending=[True, False, True],
    )

    if len(ordered) < segment_number:
        raise RuntimeError(
            f"Assignment cannot fill Standard Segment Number: "
            f"need {segment_number}, eligible={len(ordered)}, "
            f"shortfall={segment_number - len(ordered)}. "
            "Do not fill with arbitrary companies."
        )

    selected = set(ordered.head(segment_number)["company_key"])

    c["provisional_standard"] = c["company_key"].isin(selected)
    c["provisional_lower_segment"] = ~c["provisional_standard"]
    c["assignment_selected_rank"] = np.nan

    selected_order = ordered.head(segment_number)["company_key"].tolist()
    rank_map = {key: i + 1 for i, key in enumerate(selected_order)}

    c.loc[c["provisional_standard"], "assignment_selected_rank"] = (
        c.loc[c["provisional_standard"], "company_key"].map(rank_map)
    )

    c["assignment_continuity_retained"] = (
        c["provisional_standard"]
        & c["assignment_standard_state"]
        & mcap.lt(cutoff_usd)
    )

    c["assignment_new_standard_candidate"] = (
        c["provisional_standard"]
        & ~c["assignment_current_standard"]
    )

    c["assignment_fallback_used"] = False

    if int(c["provisional_standard"].sum()) != segment_number:
        raise RuntimeError("Provisional Standard company count mismatch.")

    return c