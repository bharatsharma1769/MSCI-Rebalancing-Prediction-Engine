from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_PATH = Path("data/processed/backtest_results.parquet")
MIEU_PATH = Path("data/processed/mieu_review_state.parquet")
STANDARD_PATH = Path("data/processed/standard_review_state.parquet")

ERROR_PATH = Path("data/backtest/backtest_error_analysis.csv")
DIAG_PATH = Path("data/processed/review_diagnostics.parquet")

BURN_IN = "2023-05"


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def join_states(s):
    vals = sorted({str(v) for v in s.dropna()})
    return "|".join(vals)


def closest_zero(*vals):
    vals = [float(v) for v in vals if pd.notna(v) and np.isfinite(v)]
    return min(vals, key=abs) if vals else np.nan


def bound_nearest(lo, hi):
    if pd.isna(lo) and pd.isna(hi):
        return np.nan
    if pd.isna(lo):
        return hi
    if pd.isna(hi):
        return lo
    if lo <= 0 <= hi:
        return 0.0
    return lo if abs(lo) <= abs(hi) else hi


def crosses_zero(lo, hi):
    return pd.notna(lo) and pd.notna(hi) and lo <= 0 <= hi


def is_true(v):
    return pd.notna(v) and v == True


def is_false(v):
    return pd.notna(v) and v == False

def to_pct(s):
    return 100 * pd.to_numeric(s, errors="coerce")


def choose_liquidity_margin(r):
    existing = r["liquidity_existing_margin_pct"]
    new = r["liquidity_new_margin_pct"]

    if not is_true(r["prior_imi_uncertain"]):
        if is_true(r["prior_imi"]):
            return existing
        if is_false(r["prior_imi"]):
            return new

    return closest_zero(existing, new)


def classify_event(r):
    actual = r["actual_action"]

    if actual == "ADD":
        if is_true(r["definite_add"]):
            return "DEFINITE_HIT"
        if is_true(r["supported_add"]):
            return "SCENARIO_SUPPORTED_ONLY"
        if is_true(r["add_uncertainty_envelope"]):
            return "UNCERTAINTY_ONLY"
        return "HARD_MISS"

    if actual == "DELETE":
        if is_true(r["definite_delete"]):
            return "DEFINITE_HIT"
        if is_true(r["supported_delete"]):
            return "SCENARIO_SUPPORTED_ONLY"
        if is_true(r["delete_uncertainty_envelope"]):
            return "UNCERTAINTY_ONLY"
        return "HARD_MISS"

    if is_true(r["add_false_positive"]):
        return "DEFINITE_ADD_FALSE_POSITIVE"

    if is_true(r["delete_false_positive"]):
        return "DEFINITE_DELETE_FALSE_POSITIVE"

    return "NO_EVENT"


def attribution(r):
    actual = r["actual_action"]
    pred = r["predicted_action"]

    mcap_gap = r["mcap_gap_to_cutoff_nearest_pct"]
    ff_gap = r["ff_gap_to_requirement_nearest_pct"]
    liq_gap = r["liquidity_margin_pct"]
    room_gap = r["foreign_room_margin_pct"]

    mcap5 = pd.notna(mcap_gap) and abs(mcap_gap) <= 5
    mcap10 = pd.notna(mcap_gap) and abs(mcap_gap) <= 10
    ff5 = pd.notna(ff_gap) and abs(ff_gap) <= 5
    ff10 = pd.notna(ff_gap) and abs(ff_gap) <= 10
    liq5 = pd.notna(liq_gap) and abs(liq_gap) <= 5
    room5 = pd.notna(room_gap) and abs(room_gap) <= 5

    cutoff_cross = crosses_zero(
        r["mcap_gap_to_cutoff_min_pct"],
        r["mcap_gap_to_cutoff_max_pct"],
    )
    ff_cross = crosses_zero(
        r["ff_gap_to_requirement_min_pct"],
        r["ff_gap_to_requirement_max_pct"],
    )

    room_unresolved = (
        is_true(r["foreign_room_factor_unresolved"])
        or "UNRESOLVED" in str(r["foreign_room_status"])
        or "PENDING" in str(r["foreign_room_status"])
    )

    epi_unresolved = r["epi_unresolved_scenarios"] > 0
    epi_fail = r["epi_fail_scenarios"] > 0
    ff_fail = r["ff_fail_scenarios"] > 0

    supported_event = (
        actual == "ADD" and is_true(r["supported_add"])
    ) or (
        actual == "DELETE" and is_true(r["supported_delete"])
    )

    definite_event = (
        actual == "ADD" and is_true(r["definite_add"])
    ) or (
        actual == "DELETE" and is_true(r["definite_delete"])
    )

    # Actual event, but methodology scenarios disagree.
    if actual in {"ADD", "DELETE"} and supported_event and not definite_event:
        if room_unresolved:
            return "Scenario uncertainty", "Foreign room", "HIGH"
        if epi_unresolved:
            return "Scenario uncertainty", "Extreme-price exclusion", "HIGH"
        if is_true(r["prior_imi_uncertain"]):
            return "Scenario uncertainty", "Membership / continuity", "HIGH"
        if cutoff_cross or mcap5:
            return "Scenario uncertainty", "GMSR / Standard cutoff approximation", "MEDIUM"
        if ff_cross or ff5:
            return "Scenario uncertainty", "FIF / FF-mcap approximation", "MEDIUM"

        return "Scenario uncertainty", "Market coverage / segment count", "LOW"

    # Actual ADD: true hard miss.
    if actual == "ADD":
        if not is_true(r["prediction_available"]):
            return "Data / reconstruction", "Data unavailable", "HIGH"

        if str(r["mieu_state"]) != "PASS":
            if is_false(r["market_data_complete"]) or is_false(r["review_base_universe_pass"]):
                return "MIEU / investability", "Data unavailable", "HIGH"

            if is_false(r["mieu_liquidity_pass"]):
                return "MIEU / investability", "Liquidity", "HIGH"

            if is_false(r["mieu_fif_pass"]):
                return "MIEU / investability", "FIF approximation", "HIGH"

            if is_false(r["mieu_foreign_room_pass"]) or room_unresolved:
                return "MIEU / investability", "Foreign room", "HIGH"

            if is_false(r["mieu_eumsr_pass"]):
                return "MIEU / investability", "EUMSR / MIEU size threshold", "HIGH"

            if is_false(r["mieu_trading_pass"]):
                return "MIEU / investability", "Data unavailable", "MEDIUM"

            return "MIEU / investability", "Data unavailable", "LOW"

        if r["provisional_standard_scenarios"] == 0:
            if cutoff_cross or mcap5:
                return "Standard assignment", "GMSR / Standard cutoff approximation", "MEDIUM"
            if mcap10:
                return "Standard assignment", "GMSR / Standard cutoff approximation", "LOW"

            return "Standard assignment", "Market coverage / segment count", "MEDIUM"

        if ff_fail:
            if ff_cross or ff5:
                return "Final FF requirement", "FIF / FF-mcap approximation", "MEDIUM"

            if is_true(r["pit_fif_override_applied"]):
                return "Final FF requirement", "FIF approximation", "MEDIUM"

            if pd.notna(r["price_cutoff_data_age_days"]) and r["price_cutoff_data_age_days"] > 3:
                return "Final FF requirement", "Price cutoff approximation", "MEDIUM"

            if ff10:
                return "Final FF requirement", "FIF / FF-mcap approximation", "LOW"

            return "Final FF requirement", "FIF / FF-mcap approximation", "LOW"

        if epi_fail:
            return "Extreme-price screen", "Extreme-price exclusion", "HIGH"

        return "Ranking / boundary selection", "True MSCI discretion / unexplained", "UNEXPLAINED"

    # Actual DELETE that model did not definitely delete.
    if actual == "DELETE":
        if not is_true(r["prediction_available"]):
            return "Data / reconstruction", "Data unavailable", "HIGH"

        if cutoff_cross or mcap5:
            return "Ranking / boundary selection", "GMSR / Standard cutoff approximation", "MEDIUM"

        if ff_cross or ff5:
            return "Ranking / boundary selection", "FIF / FF-mcap approximation", "MEDIUM"

        if is_true(r["prior_imi_uncertain"]):
            return "Membership / continuity", "Membership / continuity", "MEDIUM"

        return "Membership / continuity", "True MSCI discretion / unexplained", "UNEXPLAINED"

    # Definite ADD false positive.
    if pred == "ADD":
        if cutoff_cross or mcap5:
            return "Ranking / boundary selection", "GMSR / Standard cutoff approximation", "MEDIUM"

        if ff_cross or ff5:
            return "Ranking / boundary selection", "FIF / FF-mcap approximation", "MEDIUM"

        if liq5:
            return "Ranking / boundary selection", "Liquidity", "LOW"

        if room5:
            return "Ranking / boundary selection", "Foreign room", "LOW"

        return "Ranking / boundary selection", "Market coverage / segment count", "LOW"

    # Definite DELETE false positive.
    if pred == "DELETE":
        if r["provisional_standard_scenarios"] == 0:
            if mcap10:
                return "Standard assignment", "GMSR / Standard cutoff approximation", "MEDIUM"

            return "Standard assignment", "Market coverage / segment count", "LOW"

        if ff_fail:
            if ff10:
                return "Final FF requirement", "FIF / FF-mcap approximation", "MEDIUM"

            return "Final FF requirement", "FIF / FF-mcap approximation", "LOW"

        if epi_fail:
            return "Extreme-price screen", "Extreme-price exclusion", "HIGH"

        return "Ranking / boundary selection", "True MSCI discretion / unexplained", "UNEXPLAINED"

    return "Unexplained", "True MSCI discretion / unexplained", "UNEXPLAINED"


def main():
    b = pd.read_parquet(BACKTEST_PATH)
    m = pd.read_parquet(MIEU_PATH)
    s = pd.read_parquet(STANDARD_PATH)

    require(
        b,
        [
            "review", "security_id", "nse_symbol", "company_name",
            "was_standard_constituent", "predicted_action", "actual_action",
            "prediction_available", "true_scenarios", "false_scenarios",
            "unresolved_scenarios", "event_support_fraction",
            "definite_add", "definite_delete", "supported_add",
            "supported_delete", "add_uncertainty_envelope",
            "delete_uncertainty_envelope", "add_candidate_rank",
            "delete_candidate_rank", "add_false_positive",
            "add_false_negative", "delete_false_positive",
            "delete_false_negative",
        ],
        "backtest_results",
    )

    require(
        m,
        [
            "review", "security_id", "full_market_cap_usd",
            "ff_market_cap_usd", "company_full_mcap_usd",
            "company_ff_mcap_usd", "company_security_count", "fif",
            "pit_fif_override_applied", "price_cutoff_data_age_days",
            "price_cutoff_refresh_valid", "market_data_complete",
            "review_base_universe_pass", "atvr_12m", "atvr_3m",
            "fot_3m", "liquidity_4q_min_atvr_3m",
            "liquidity_4q_min_fot_3m", "liquidity_new_pass",
            "liquidity_existing_pass", "prior_imi", "prior_imi_uncertain",
            "mieu_eumsr_pass", "mieu_liquidity_pass", "mieu_fif_pass",
            "mieu_trading_pass", "mieu_foreign_room_pass",
            "foreign_room", "foreign_room_lower_bound",
            "foreign_room_status", "foreign_room_factor_unresolved",
            "mieu_state",
        ],
        "mieu_review_state",
    )

    require(
        s,
        [
            "review", "security_id", "scenario_id",
            "final_standard_segment_number", "market_standard_cutoff_usd",
            "standard_gmsr_usd", "provisional_standard",
            "assignment_priority", "assignment_reason",
            "final_required_ff_mcap_usd",
            "final_ff_mcap_evaluation_min_usd",
            "final_ff_mcap_evaluation_max_usd",
            "final_ff_requirement_state", "final_epi_state",
            "final_standard_security_state", "final_standard_reason",
            "predicted_standard",
        ],
        "standard_review_state",
    )

    for x in [b, m, s]:
        x["review"] = x["review"].astype(str)
        x["security_id"] = x["security_id"].astype("string")

    if m.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate MIEU review/security rows.")

    if s.duplicated(["review", "scenario_id", "security_id"]).any():
        raise RuntimeError("Duplicate Standard review/scenario/security rows.")

    mcols = [
        "review", "security_id", "full_market_cap_usd",
        "ff_market_cap_usd", "company_full_mcap_usd",
        "company_ff_mcap_usd", "company_security_count", "fif",
        "pit_fif_override_applied", "price_cutoff_data_age_days",
        "price_cutoff_refresh_valid", "market_data_complete",
        "review_base_universe_pass", "atvr_12m", "atvr_3m",
        "fot_3m", "liquidity_4q_min_atvr_3m",
        "liquidity_4q_min_fot_3m", "liquidity_new_pass",
        "liquidity_existing_pass", "prior_imi", "prior_imi_uncertain",
        "mieu_eumsr_pass", "mieu_liquidity_pass", "mieu_fif_pass",
        "mieu_trading_pass", "mieu_foreign_room_pass",
        "foreign_room", "foreign_room_lower_bound",
        "foreign_room_status", "foreign_room_factor_unresolved",
        "mieu_state",
    ]

    z = b.merge(
        m[mcols],
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    s["ff_gap_min_pct"] = 100 * (
        pd.to_numeric(s["final_ff_mcap_evaluation_min_usd"], errors="coerce")
        / pd.to_numeric(s["final_required_ff_mcap_usd"], errors="coerce").replace(0, np.nan)
        - 1
    )

    s["ff_gap_max_pct"] = 100 * (
        pd.to_numeric(s["final_ff_mcap_evaluation_max_usd"], errors="coerce")
        / pd.to_numeric(s["final_required_ff_mcap_usd"], errors="coerce").replace(0, np.nan)
        - 1
    )

    s["provisional_standard_true"] = s["provisional_standard"].eq(True)
    s["ff_fail"] = s["final_ff_requirement_state"].eq("FAIL")
    s["ff_unresolved"] = s["final_ff_requirement_state"].eq("UNRESOLVED")
    s["epi_fail"] = s["final_epi_state"].eq("FAIL")
    s["epi_unresolved"] = s["final_epi_state"].eq("UNRESOLVED")

    sa = (
        s.groupby(["review", "security_id"], as_index=False)
        .agg(
            standard_cutoff_min_usd=("market_standard_cutoff_usd", "min"),
            standard_cutoff_max_usd=("market_standard_cutoff_usd", "max"),
            standard_gmsr_min_usd=("standard_gmsr_usd", "min"),
            standard_gmsr_max_usd=("standard_gmsr_usd", "max"),
            segment_number_min=("final_standard_segment_number", "min"),
            segment_number_max=("final_standard_segment_number", "max"),
            assignment_priority_min=("assignment_priority", "min"),
            assignment_priority_max=("assignment_priority", "max"),
            assignment_reasons=("assignment_reason", join_states),
            provisional_standard_scenarios=("provisional_standard_true", "sum"),
            final_required_ff_mcap_min_usd=("final_required_ff_mcap_usd", "min"),
            final_required_ff_mcap_max_usd=("final_required_ff_mcap_usd", "max"),
            final_ff_eval_min_usd=("final_ff_mcap_evaluation_min_usd", "min"),
            final_ff_eval_max_usd=("final_ff_mcap_evaluation_max_usd", "max"),
            ff_gap_to_requirement_min_pct=("ff_gap_min_pct", "min"),
            ff_gap_to_requirement_max_pct=("ff_gap_max_pct", "max"),
            ff_fail_scenarios=("ff_fail", "sum"),
            ff_unresolved_scenarios=("ff_unresolved", "sum"),
            epi_fail_scenarios=("epi_fail", "sum"),
            epi_unresolved_scenarios=("epi_unresolved", "sum"),
            final_ff_states=("final_ff_requirement_state", join_states),
            final_epi_states=("final_epi_state", join_states),
            final_standard_states=("final_standard_security_state", join_states),
            final_standard_reasons=("final_standard_reason", join_states),
        )
    )

    z = z.merge(
        sa,
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    cutoff_min = pd.to_numeric(z["standard_cutoff_min_usd"], errors="coerce").replace(0, np.nan)
    cutoff_max = pd.to_numeric(z["standard_cutoff_max_usd"], errors="coerce").replace(0, np.nan)
    company_mcap = pd.to_numeric(z["company_full_mcap_usd"], errors="coerce")

    z["mcap_gap_to_cutoff_min_pct"] = 100 * (company_mcap / cutoff_max - 1)
    z["mcap_gap_to_cutoff_max_pct"] = 100 * (company_mcap / cutoff_min - 1)

    z["mcap_gap_to_cutoff_nearest_pct"] = z.apply(
        lambda r: bound_nearest(
            r["mcap_gap_to_cutoff_min_pct"],
            r["mcap_gap_to_cutoff_max_pct"],
        ),
        axis=1,
    )

    z["ff_gap_to_requirement_nearest_pct"] = z.apply(
        lambda r: bound_nearest(
            r["ff_gap_to_requirement_min_pct"],
            r["ff_gap_to_requirement_max_pct"],
        ),
        axis=1,
    )

    atvr12 = to_pct(z["atvr_12m"])
    atvr3 = to_pct(z["atvr_3m"])
    fot3 = to_pct(z["fot_3m"])

    min_atvr3 = to_pct(z["liquidity_4q_min_atvr_3m"]).fillna(atvr3)
    min_fot3 = to_pct(z["liquidity_4q_min_fot_3m"]).fillna(fot3)

    z["liquidity_existing_margin_pct"] = pd.concat(
        [atvr12 - 10, atvr3 - 5, fot3 - 70],
        axis=1,
    ).min(axis=1)

    z["liquidity_new_margin_pct"] = pd.concat(
        [atvr12 - 15, min_atvr3 - 15, min_fot3 - 80],
        axis=1,
    ).min(axis=1)

    z["liquidity_margin_pct"] = z.apply(choose_liquidity_margin, axis=1)

    room = to_pct(z["foreign_room"])
    room_lb = to_pct(z["foreign_room_lower_bound"])
    room_effective = room.fillna(room_lb)

    z["foreign_room_margin_pct"] = room_effective - 15

    z["boundary_gap_pct"] = z.apply(
        lambda r: closest_zero(
            r["mcap_gap_to_cutoff_nearest_pct"],
            r["ff_gap_to_requirement_nearest_pct"],
        ),
        axis=1,
    )

    z["near_boundary_2pct"] = z["boundary_gap_pct"].abs().le(2)
    z["near_boundary_5pct"] = z["boundary_gap_pct"].abs().le(5)
    z["near_boundary_10pct"] = z["boundary_gap_pct"].abs().le(10)

    z["candidate_rank"] = z["predicted_rank"]

    add_side = (
        z["actual_action"].eq("ADD")
        | z["predicted_action"].isin(["ADD", "ADD_OR_NONE"])
    )
    del_side = (
        z["actual_action"].eq("DELETE")
        | z["predicted_action"].isin(["DELETE", "KEEP_OR_DELETE"])
    )

    z.loc[add_side, "candidate_rank"] = z.loc[add_side, "add_candidate_rank"]
    z.loc[del_side, "candidate_rank"] = z.loc[del_side, "delete_candidate_rank"]

    z["event_class"] = z.apply(classify_event, axis=1)

    error_mask = (
        z["review"].ne(BURN_IN)
        & (
            z["add_false_positive"].fillna(False)
            | z["add_false_negative"].fillna(False)
            | z["delete_false_positive"].fillna(False)
            | z["delete_false_negative"].fillna(False)
        )
    )

    attr = z.loc[error_mask].apply(attribution, axis=1)
    attr = pd.DataFrame(
        attr.tolist(),
        index=z.loc[error_mask].index,
        columns=[
            "failure_stage",
            "estimated_error_source",
            "attribution_confidence",
        ],
    )

    z.loc[attr.index, attr.columns] = attr

    error_cols = [
        "review", "security_id", "nse_symbol", "company_name",
        "actual_action", "predicted_action", "was_standard_constituent",
        "event_class", "true_scenarios", "false_scenarios",
        "unresolved_scenarios", "event_support_fraction", "candidate_rank",
        "full_market_cap_usd", "ff_market_cap_usd",
        "company_full_mcap_usd", "company_ff_mcap_usd", "fif",
        "standard_cutoff_min_usd", "standard_cutoff_max_usd",
        "mcap_gap_to_cutoff_min_pct", "mcap_gap_to_cutoff_max_pct",
        "mcap_gap_to_cutoff_nearest_pct",
        "final_required_ff_mcap_min_usd",
        "final_required_ff_mcap_max_usd",
        "final_ff_eval_min_usd", "final_ff_eval_max_usd",
        "ff_gap_to_requirement_min_pct",
        "ff_gap_to_requirement_max_pct",
        "ff_gap_to_requirement_nearest_pct",
        "liquidity_margin_pct", "foreign_room",
        "foreign_room_lower_bound", "foreign_room_margin_pct",
        "mieu_state", "assignment_reasons",
        "final_ff_states", "final_epi_states",
        "final_standard_states", "final_standard_reasons",
        "boundary_gap_pct", "near_boundary_2pct",
        "near_boundary_5pct", "near_boundary_10pct",
        "failure_stage", "estimated_error_source",
        "attribution_confidence",
        "add_false_positive", "add_false_negative",
        "delete_false_positive", "delete_false_negative",
    ]

    errors = z.loc[error_mask, error_cols].copy()

    diag_mask = (
        z["actual_action"].isin(["ADD", "DELETE"])
        | z["predicted_action"].isin(["ADD", "DELETE"])
        | z["unresolved_scenarios"].gt(0)
        | z["near_boundary_10pct"]
    )

    diag_cols = [
        "review", "security_id", "nse_symbol", "company_name",
        "actual_action", "predicted_action", "was_standard_constituent",
        "event_class", "true_scenarios", "false_scenarios",
        "unresolved_scenarios", "event_support_fraction",
        "add_candidate_rank", "delete_candidate_rank", "candidate_rank",
        "full_market_cap_usd", "ff_market_cap_usd",
        "company_full_mcap_usd", "company_ff_mcap_usd", "fif",
        "standard_cutoff_min_usd", "standard_cutoff_max_usd",
        "standard_gmsr_min_usd", "standard_gmsr_max_usd",
        "segment_number_min", "segment_number_max",
        "assignment_priority_min", "assignment_priority_max",
        "assignment_reasons", "provisional_standard_scenarios",
        "mcap_gap_to_cutoff_min_pct", "mcap_gap_to_cutoff_max_pct",
        "mcap_gap_to_cutoff_nearest_pct",
        "final_required_ff_mcap_min_usd",
        "final_required_ff_mcap_max_usd",
        "ff_gap_to_requirement_min_pct",
        "ff_gap_to_requirement_max_pct",
        "ff_gap_to_requirement_nearest_pct",
        "liquidity_margin_pct", "foreign_room",
        "foreign_room_lower_bound", "foreign_room_margin_pct",
        "mieu_state", "final_ff_states", "final_epi_states",
        "final_standard_states", "boundary_gap_pct",
        "near_boundary_2pct", "near_boundary_5pct",
        "near_boundary_10pct",
    ]

    diagnostics = z.loc[diag_mask, diag_cols].copy()

    ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERROR_PATH, index=False)
    diagnostics.to_parquet(DIAG_PATH, index=False)

    scored_events = z[
        z["review"].ne(BURN_IN)
        & z["actual_action"].isin(["ADD", "DELETE"])
    ]

    add_classes = scored_events[
        scored_events["actual_action"].eq("ADD")
    ]["event_class"].value_counts()

    del_classes = scored_events[
        scored_events["actual_action"].eq("DELETE")
    ]["event_class"].value_counts()

    def count(vc, key):
        return int(vc.get(key, 0))

    print("\n=== ERROR ATTRIBUTION | 2023-08 TO 2026-05 ===")

    print("\nADDITION EVENTS")
    print(
        f"Definite hits: {count(add_classes, 'DEFINITE_HIT')}"
        f" | Scenario-supported only: {count(add_classes, 'SCENARIO_SUPPORTED_ONLY')}"
        f" | Uncertainty-only: {count(add_classes, 'UNCERTAINTY_ONLY')}"
        f" | Hard misses: {count(add_classes, 'HARD_MISS')}"
    )

    print("\nDELETION EVENTS")
    print(
        f"Definite hits: {count(del_classes, 'DEFINITE_HIT')}"
        f" | Scenario-supported only: {count(del_classes, 'SCENARIO_SUPPORTED_ONLY')}"
        f" | Uncertainty-only: {count(del_classes, 'UNCERTAINTY_ONLY')}"
        f" | Hard misses: {count(del_classes, 'HARD_MISS')}"
    )

    print("\nSTRICT ERRORS")
    print(
        f"Add FP: {int(errors['add_false_positive'].fillna(False).sum())}"
        f" | Add FN: {int(errors['add_false_negative'].fillna(False).sum())}"
        f" | Delete FP: {int(errors['delete_false_positive'].fillna(False).sum())}"
        f" | Delete FN: {int(errors['delete_false_negative'].fillna(False).sum())}"
    )

    fn = errors[
        errors["add_false_negative"].fillna(False)
        | errors["delete_false_negative"].fillna(False)
    ]

    print("\nFALSE NEGATIVES NEAR A DECISION BOUNDARY")
    print(
        f"<=2%: {int(fn['near_boundary_2pct'].sum())}"
        f" | <=5%: {int(fn['near_boundary_5pct'].sum())}"
        f" | <=10%: {int(fn['near_boundary_10pct'].sum())}"
        f" | Total FN: {len(fn)}"
    )

    print("\nPRIMARY FAILURE STAGE")
    print(
        errors["failure_stage"]
        .fillna("UNATTRIBUTED")
        .value_counts()
        .to_string()
    )

    print("\nESTIMATED ERROR SOURCE")
    print(
        errors["estimated_error_source"]
        .fillna("UNATTRIBUTED")
        .value_counts()
        .to_string()
    )

    print("\nSaved:", ERROR_PATH)
    print("Saved:", DIAG_PATH)


if __name__ == "__main__":
    main()