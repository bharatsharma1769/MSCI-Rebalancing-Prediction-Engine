import numpy as np
import pandas as pd


GMSR_LOWER = 0.50
GMSR_UPPER = 1.15
STANDARD_MIN_FF_MULTIPLIER = 0.50
EXISTING_STANDARD_RETENTION = 2.0 / 3.0
LOW_FIF_MULTIPLIER = 1.80


def require(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"standard_final_requirements missing: {missing}"
        )


def bool_series(s):
    return s.astype("boolean")


def ff_state(min_ff, max_ff, required):
    out = pd.Series(
        "UNRESOLVED",
        index=min_ff.index,
        dtype="string",
    )

    valid = (
        min_ff.notna()
        & max_ff.notna()
        & required.notna()
    )

    out.loc[
        valid & min_ff.ge(required)
    ] = "PASS"

    out.loc[
        valid & max_ff.lt(required)
    ] = "FAIL"

    return out


def evaluate_standard_final_requirements(
    securities,
    market_standard_cutoff_usd,
    standard_gmsr_usd,
):
    require(
        securities,
        [
            "company_key",
            "security_id",
            "provisional_standard",
            "was_standard_constituent",
            "prior_imi",
            "prior_imi_uncertain",
            "fif",
            "ff_market_cap_usd",
            "post_adjusted_ff_mcap_usd_min",
            "post_adjusted_ff_mcap_usd_max",
            "extreme_price_pass",
        ],
    )

    x = securities.copy()

    cutoff = float(market_standard_cutoff_usd)
    gmsr = float(standard_gmsr_usd)

    if not np.isfinite(cutoff) or cutoff <= 0:
        raise RuntimeError("Invalid market Standard cutoff.")

    if not np.isfinite(gmsr) or gmsr <= 0:
        raise RuntimeError("Invalid Standard GMSR.")

    x["provisional_standard"] = (
        x["provisional_standard"]
        .fillna(False)
        .astype(bool)
    )

    x["was_standard_constituent"] = (
        x["was_standard_constituent"]
        .fillna(False)
        .astype(bool)
    )

    x["prior_imi"] = (
        x["prior_imi"]
        .fillna(False)
        .astype(bool)
    )

    x["prior_imi_uncertain"] = (
        x["prior_imi_uncertain"]
        .fillna(False)
        .astype(bool)
    )

    if (
        x["prior_imi"]
        & x["prior_imi_uncertain"]
    ).any():
        raise RuntimeError(
            "Security cannot be both definite and uncertain prior IMI."
        )

    for c in [
        "fif",
        "ff_market_cap_usd",
        "post_adjusted_ff_mcap_usd_min",
        "post_adjusted_ff_mcap_usd_max",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    basis = np.clip(
        cutoff,
        GMSR_LOWER * gmsr,
        GMSR_UPPER * gmsr,
    )

    minimum_ff = (
        STANDARD_MIN_FF_MULTIPLIER * basis
    )

    x["final_standard_gmsr_lower_usd"] = (
        GMSR_LOWER * gmsr
    )

    x["final_standard_gmsr_upper_usd"] = (
        GMSR_UPPER * gmsr
    )

    x["final_ff_threshold_basis_usd"] = basis
    x["final_min_ff_mcap_usd"] = minimum_ff

    current_standard = x["was_standard_constituent"]
    low_fif = x["fif"].lt(0.15)

    multiplier = pd.Series(
        1.0,
        index=x.index,
        dtype=float,
    )

    multiplier.loc[low_fif] = LOW_FIF_MULTIPLIER

    multiplier.loc[
        current_standard & ~low_fif
    ] = EXISTING_STANDARD_RETENTION

    multiplier.loc[
        current_standard & low_fif
    ] = (
        EXISTING_STANDARD_RETENTION
        * LOW_FIF_MULTIPLIER
    )

    x["final_ff_requirement_multiplier"] = multiplier

    x["final_required_ff_mcap_usd"] = (
        minimum_ff * multiplier
    )

    # -------------------------------------------------
    # FF-cap basis.
    #
    # Definite prior IMI:
    #   pre-adjustment FF mcap.
    #
    # Definite non-prior IMI:
    #   post-adjustment FF mcap interval.
    #
    # Uncertain prior IMI:
    #   union of both possibilities.
    # -------------------------------------------------

    pre = x["ff_market_cap_usd"]
    post_min = x["post_adjusted_ff_mcap_usd_min"]
    post_max = x["post_adjusted_ff_mcap_usd_max"]

    x["final_ff_mcap_basis"] = pd.Series(
        "POST_ADJUSTMENT",
        index=x.index,
        dtype="string",
    )

    x.loc[
        x["prior_imi"],
        "final_ff_mcap_basis",
    ] = "PRE_ADJUSTMENT"

    x.loc[
        x["prior_imi_uncertain"],
        "final_ff_mcap_basis",
    ] = "PRE_OR_POST_ADJUSTMENT"

    eval_min = post_min.copy()
    eval_max = post_max.copy()

    eval_min.loc[x["prior_imi"]] = pre.loc[
        x["prior_imi"]
    ]

    eval_max.loc[x["prior_imi"]] = pre.loc[
        x["prior_imi"]
    ]

    uncertain = x["prior_imi_uncertain"]

    eval_min.loc[uncertain] = pd.concat(
        [
            pre.loc[uncertain],
            post_min.loc[uncertain],
        ],
        axis=1,
    ).min(axis=1, skipna=False)

    eval_max.loc[uncertain] = pd.concat(
        [
            pre.loc[uncertain],
            post_max.loc[uncertain],
        ],
        axis=1,
    ).max(axis=1, skipna=False)

    x["final_ff_mcap_evaluation_min_usd"] = eval_min
    x["final_ff_mcap_evaluation_max_usd"] = eval_max

    x["final_ff_requirement_state"] = ff_state(
        eval_min,
        eval_max,
        x["final_required_ff_mcap_usd"],
    )

    # Missing FIF means the required threshold itself is unknown.
    x.loc[
        x["fif"].isna(),
        "final_ff_requirement_state",
    ] = "UNRESOLVED"

    # -------------------------------------------------
    # Extreme Price Increase.
    #
    # EPI is a Standard-addition restriction.
    # Current Standard constituents are not additions.
    # -------------------------------------------------

    epi = bool_series(x["extreme_price_pass"])

    addition = (
        x["provisional_standard"]
        & ~x["was_standard_constituent"]
    )

    x["final_epi_applicable"] = addition

    x["final_epi_state"] = pd.Series(
        "NOT_APPLICABLE",
        index=x.index,
        dtype="string",
    )

    x.loc[
        addition & epi.eq(True).fillna(False),
        "final_epi_state",
    ] = "PASS"

    x.loc[
        addition & epi.eq(False).fillna(False),
        "final_epi_state",
    ] = "FAIL"

    x.loc[
        addition & epi.isna(),
        "final_epi_state",
    ] = "UNRESOLVED"

    # -------------------------------------------------
    # Final security-level Standard conformity.
    # -------------------------------------------------

    x["final_standard_security_state"] = pd.Series(
        "NOT_ASSIGNED",
        index=x.index,
        dtype="string",
    )

    assigned = x["provisional_standard"]

    ff_fail = x["final_ff_requirement_state"].eq("FAIL")
    ff_unresolved = x[
        "final_ff_requirement_state"
    ].eq("UNRESOLVED")

    epi_fail = x["final_epi_state"].eq("FAIL")
    epi_unresolved = x[
        "final_epi_state"
    ].eq("UNRESOLVED")

    fail = assigned & (ff_fail | epi_fail)

    unresolved = (
        assigned
        & ~fail
        & (ff_unresolved | epi_unresolved)
    )

    passed = assigned & ~fail & ~unresolved

    x.loc[
        fail,
        "final_standard_security_state",
    ] = "FAIL"

    x.loc[
        unresolved,
        "final_standard_security_state",
    ] = "UNRESOLVED"

    x.loc[
        passed,
        "final_standard_security_state",
    ] = "PASS"

    x["final_standard_security_pass"] = pd.array(
        np.where(
            x["final_standard_security_state"].eq("PASS"),
            True,
            np.where(
                x["final_standard_security_state"].eq("FAIL"),
                False,
                pd.NA,
            ),
        ),
        dtype="boolean",
    )

    # -------------------------------------------------
    # Failure / uncertainty reason.
    # -------------------------------------------------

    x["final_standard_reason"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x.loc[
        ff_fail & epi_fail,
        "final_standard_reason",
    ] = "FF_MCAP_AND_EPI_FAIL"

    x.loc[
        ff_fail & ~epi_fail,
        "final_standard_reason",
    ] = "FF_MCAP_FAIL"

    x.loc[
        ~ff_fail & epi_fail,
        "final_standard_reason",
    ] = "EPI_FAIL"

    x.loc[
        unresolved & ff_unresolved & epi_unresolved,
        "final_standard_reason",
    ] = "FF_MCAP_AND_EPI_UNRESOLVED"

    x.loc[
        unresolved & ff_unresolved & ~epi_unresolved,
        "final_standard_reason",
    ] = "FF_MCAP_UNRESOLVED"

    x.loc[
        unresolved & ~ff_unresolved & epi_unresolved,
        "final_standard_reason",
    ] = "EPI_UNRESOLVED"

    x.loc[
        passed,
        "final_standard_reason",
    ] = "PASS"

    # EPI failures by existing IMI additions remain lower-segment
    # candidates rather than being treated as MIEU failures.
    x["epi_blocks_standard_migration"] = (
        addition
        & (
            x["prior_imi"]
            | x["prior_imi_uncertain"]
        )
        & epi_fail
    )

    # -------------------------------------------------
    # Company-level Standard result.
    #
    # At least one passing security retains the company
    # in the Standard size-segment.
    # -------------------------------------------------

    def company_state(g):
        if not g["provisional_standard"].any():
            return "NOT_ASSIGNED"

        states = set(
            g["final_standard_security_state"]
        )

        if "PASS" in states:
            return "PASS"

        if "UNRESOLVED" in states:
            return "UNRESOLVED"

        return "FAIL"

    company_states = (
        x.groupby("company_key", dropna=False)
        .apply(company_state, include_groups=False)
        .rename("final_standard_company_state")
    )

    x = x.merge(
        company_states,
        left_on="company_key",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    x["final_standard_company_pass"] = pd.array(
        np.where(
            x["final_standard_company_state"].eq("PASS"),
            True,
            np.where(
                x["final_standard_company_state"].eq("FAIL"),
                False,
                pd.NA,
            ),
        ),
        dtype="boolean",
    )

    return x