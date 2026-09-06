from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


INV_PATH = Path("data/processed/investability_universe.parquet")
FR_PATH = Path("data/processed/foreign_room.parquet")
SNAP_PATH = Path("data/processed/india_price_cutoff_snapshot.parquet")
SIZE_PATH = Path("data/processed/india_size_cutoffs.parquet")
MASTER_PATH = Path("data/processed/historical_security_master.parquet")
REVIEW_FIF_PATH = Path("data/processed/review_fif.parquet")
OUT_PATH = Path("data/processed/mieu_review_state.parquet")

SMIN_XML = "https://www.sec.gov/Archives/edgar/data/1100663/000175272423012260/primary_doc.xml"
BOOTSTRAP_METHOD = "SMIN_2022_11_30_PLUS_FEB2023_STANDARD_EQ_SCOPE"
HDFCBANK = "SEC000820"

REVIEWS = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05",
    "2024-08", "2024-11", "2025-02", "2025-05", "2025-08",
    "2025-11", "2026-02", "2026-05",
]

EUMSR_USD_M = {
    "2023-05": 292, "2023-08": 291, "2023-11": 313, "2024-02": 323,
    "2024-05": 371, "2024-08": 383, "2024-11": 422, "2025-02": 443,
    "2025-05": 430, "2025-08": 448, "2025-11": 505, "2026-02": 507,
    "2026-05": 537,
}

PRICE_CUTOFF = {
    "2023-05": "2023-04-17", "2023-08": "2023-07-18",
    "2023-11": "2023-10-18", "2024-02": "2024-01-18",
    "2024-05": "2024-04-17", "2024-08": "2024-07-18",
    "2024-11": "2024-10-18", "2025-02": "2025-01-20",
    "2025-05": "2025-04-17", "2025-08": "2025-07-18",
    "2025-11": "2025-10-20", "2026-02": "2026-01-19",
    "2026-05": "2026-04-17",
}

FACTOR_COLS = {
    1.00: ("post_factor_if_pre_1_min", "post_factor_if_pre_1_max"),
    0.50: ("post_factor_if_pre_0_5_min", "post_factor_if_pre_0_5_max"),
    0.25: ("post_factor_if_pre_0_25_min", "post_factor_if_pre_0_25_max"),
}


def num(x):
    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]


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
    return df


def load_smin_bootstrap(inv, master):
    r = requests.get(
        SMIN_XML,
        headers={"User-Agent": "Mozilla/5.0 MSCI-Rebalancing-Research bharat14@connect.hku.hk"},
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)

    isins = set()
    for e in root.iter():
        if e.tag.rsplit("}", 1)[-1].lower() != "isin":
            continue
        for v in [e.text, *e.attrib.values()]:
            if v:
                v = str(v).strip().upper()
                if re.fullmatch(r"IN[A-Z0-9]{10}", v):
                    isins.add(v)

    if len(isins) != 358:
        raise RuntimeError(f"SMIN ISIN count changed: {len(isins)} != 358")

    require(master, ["security_id", "isin"], "historical_security_master")

    m = master[["security_id", "isin"]].dropna().copy()
    m["security_id"] = m["security_id"].astype("string")
    m["isin"] = m["isin"].astype(str).str.strip().str.upper()
    m = m[m["isin"].isin(isins)].drop_duplicates()

    counts = m.groupby("isin")["security_id"].nunique()
    mapped = set(m[m["isin"].isin(counts[counts.eq(1)].index)]["security_id"])

    if len(mapped) != 318:
        raise RuntimeError(f"SMIN mapped count changed: {len(mapped)} != 318")

    may = inv[inv["review"].eq("2023-05")]
    may_ids = set(may["security_id"])
    smin_may = mapped & may_ids

    standard_may = set(
        may.loc[
            may["was_standard_constituent"].fillna(False).astype(bool),
            "security_id",
        ]
    )

    bootstrap = standard_may | smin_may

    if (len(standard_may), len(smin_may), len(bootstrap)) != (115, 315, 426):
        raise RuntimeError(
            f"May bootstrap changed: Standard={len(standard_may)}, "
            f"SMIN={len(smin_may)}, combined={len(bootstrap)}"
        )

    return bootstrap


def prior_standard_n(r):
    z = r[r["was_standard_constituent"].fillna(False).astype(bool)]
    if z["company_key"].isna().any():
        raise RuntimeError("Standard constituent missing company_key.")
    return int(z["company_key"].nunique())


def ranked_cutoff(snapshot, keys, n, eumsr):
    z = snapshot[
        snapshot["company_key"].isin(keys)
        & snapshot["price_cutoff_refresh_valid"].fillna(False).astype(bool)
    ].copy()

    z["full_market_cap_usd"] = pd.to_numeric(
        z["full_market_cap_usd"], errors="coerce"
    )

    z = (
        z.groupby("company_key", as_index=False)
        .agg(
            full_market_cap_usd=(
                "full_market_cap_usd",
                lambda s: s.sum(min_count=1),
            )
        )
        .dropna(subset=["full_market_cap_usd"])
        .sort_values(
            ["full_market_cap_usd", "company_key"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    if len(z) < n:
        raise RuntimeError(
            f"Only {len(z)} prior-MIEU companies valued; need {n}."
        )

    raw = float(z.iloc[n - 1]["full_market_cap_usd"])
    return raw, max(raw, eumsr), len(z)


def combine_bool(prior_true, prior_uncertain, existing, new):
    existing = existing.astype("boolean")
    new = new.astype("boolean")

    out = new.copy()
    out.loc[prior_true] = existing.loc[prior_true]

    same = existing.eq(new).fillna(False)
    out.loc[prior_uncertain & same] = existing.loc[prior_uncertain & same]
    out.loc[prior_uncertain & ~same] = pd.NA
    return out


def interval_ge(lo, hi, threshold):
    lo = pd.to_numeric(lo, errors="coerce")
    hi = pd.to_numeric(hi, errors="coerce")
    out = pd.Series(pd.NA, index=lo.index, dtype="boolean")
    out.loc[lo.ge(threshold)] = True
    out.loc[hi.lt(threshold)] = False
    return out


def branch_state(base, checks):
    z = pd.concat([c.astype("boolean") for c in checks], axis=1)
    fail = ~base | z.eq(False).any(axis=1)
    unresolved = ~fail & z.isna().any(axis=1)

    return pd.Series(
        np.select(
            [fail, unresolved],
            ["FAIL", "UNRESOLVED"],
            default="PASS",
        ),
        index=base.index,
    )


def final_state(prior_true, prior_uncertain, existing_state, new_state):
    out = new_state.copy()
    out.loc[prior_true] = existing_state.loc[prior_true]

    same = existing_state.eq(new_state)
    out.loc[prior_uncertain & same] = existing_state.loc[
        prior_uncertain & same
    ]
    out.loc[prior_uncertain & ~same] = "UNRESOLVED"
    return out


def hdfc_factor(review):
    if REVIEWS.index(review) < REVIEWS.index("2024-08"):
        return 0.50
    if review == "2024-08":
        return 0.75
    return 1.00


def existing_factor(row, review, prior_factor_state):
    sid = str(row["security_id"])
    prior = prior_factor_state.get(sid)

    if prior:
        pmin, pmax = prior["min"], prior["max"]
        last_reduction = prior["last_reduction"]
        previous_fol = prior["exact_fol"]
        history_uncertain = prior.get("history_uncertain", False)
    else:
        pmin = pmax = 0.50 if sid == HDFCBANK else 1.00
        last_reduction = None
        previous_fol = np.nan
        history_uncertain = False

    if sid == HDFCBANK:
        qmin = qmax = hdfc_factor(review)
        return (
            pmin, pmax, qmin, qmax,
            last_reduction, previous_fol,
            False, "KNOWN_HDFCBANK_FACTOR_HISTORY",
        )

    vals = []
    for p in (0.25, 0.50, 1.00):
        if pmin - 1e-9 <= p <= pmax + 1e-9:
            lo_col, hi_col = FACTOR_COLS[p]
            lo, hi = num(row.get(lo_col)), num(row.get(hi_col))
            if pd.notna(lo) and pd.notna(hi):
                vals.append((float(lo), float(hi)))

    if not vals:
        return (
            pmin, pmax, pmin, pmax,
            last_reduction, previous_fol,
            True, "EXISTING_FACTOR_EVIDENCE_UNRESOLVED",
        )

    qmin = min(v[0] for v in vals)
    qmax = max(v[1] for v in vals)
    unresolved = not np.isclose(qmin, qmax)
    source = "RECURSIVE_FOREIGN_ROOM_FACTOR"

    idx = REVIEWS.index(review)
    current_cutoff = pd.Timestamp(PRICE_CUTOFF[review])
    previous_cutoff = (
        pd.Timestamp(PRICE_CUTOFF[REVIEWS[idx - 1]]) if idx else None
    )

    upward = qmax > pmax + 1e-9

    if upward and history_uncertain:
        qmin = min(qmin, pmin)
        qmax = max(qmax, pmax)
        unresolved = True
        source = "UPWARD_WAIT_HISTORY_UNRESOLVED"

    elif (
        upward
        and last_reduction is not None
        and (current_cutoff - last_reduction).days < 365
    ):
        current_fol = num(row.get("historical_fol"))
        effective = pd.to_datetime(
            row.get("historical_fol_effective_date"),
            errors="coerce",
        )

        fol_exception = (
            pd.notna(current_fol)
            and pd.notna(previous_fol)
            and current_fol > previous_fol
            and pd.notna(effective)
            and previous_cutoff is not None
            and effective <= previous_cutoff
        )

        if not fol_exception:
            qmax = min(qmax, pmax)
            qmin = min(qmin, qmax)
            unresolved = not np.isclose(qmin, qmax)
            source = "12M_UPWARD_WAIT"

    if qmin < pmin - 1e-9 or qmax < pmax - 1e-9:
        last_reduction = current_cutoff

    return (
        pmin, pmax, qmin, qmax,
        last_reduction, previous_fol,
        unresolved, source,
    )


def apply_factors(r, review, prior_factor_state):
    records = []

    for i, row in r.iterrows():
        (
            pmin, pmax, emin, emax,
            last_reduction, previous_fol,
            existing_unresolved, existing_source,
        ) = existing_factor(row, review, prior_factor_state)

        entry = row.get("new_entry_pass_if_nonconstituent")
        nmin = num(row.get("new_entry_factor_min_if_pass"))
        nmax = num(row.get("new_entry_factor_max_if_pass"))

        if pd.notna(entry) and not bool(entry):
            nmin = nmax = np.nan
        elif pd.notna(entry) and bool(entry):
            if pd.isna(nmin) or pd.isna(nmax):
                raise RuntimeError(
                    f"{review} {row['security_id']}: proven new-entry pass but factor interval missing."
                )
        elif pd.isna(nmin) or pd.isna(nmax):
            nmin, nmax = 0.50, 1.00

        existing_state = row["mieu_state_if_existing"]
        new_state = row["mieu_state_if_new"]

        if pd.notna(emax) and np.isclose(emax, 0):
            existing_state = "FAIL"
            r.at[i, "mieu_state_if_existing"] = "FAIL"
            r.at[i, "existing_factor_forces_fail"] = True

        viable_existing = (
            existing_state != "FAIL"
            and pd.notna(emin)
            and pd.notna(emax)
        )
        viable_new = (
            new_state != "FAIL"
            and pd.notna(nmin)
            and pd.notna(nmax)
        )

        factor_intervals = []
        branch_intervals = []

        def add_existing():
            if not viable_existing:
                return
            factor_intervals.append((emin, emax))
            branch_intervals.append((
                num(row.get("review_fif_existing_min")),
                num(row.get("review_fif_existing_max")),
                num(row.get("review_ff_mcap_existing_min_usd")),
                num(row.get("review_ff_mcap_existing_max_usd")),
                emin,
                emax,
            ))

        def add_new():
            if not viable_new:
                return
            factor_intervals.append((nmin, nmax))
            branch_intervals.append((
                num(row.get("review_fif_new_min")),
                num(row.get("review_fif_new_max")),
                num(row.get("review_ff_mcap_new_min_usd")),
                num(row.get("review_ff_mcap_new_max_usd")),
                nmin,
                nmax,
            ))

        if bool(row["prior_imi"]):
            add_existing()
            source = existing_source
        elif bool(row["prior_imi_uncertain"]):
            add_existing()
            add_new()
            source = "PRIOR_IMI_BRANCH_UNION"
        else:
            add_new()
            source = "NEW_ENTRY_FOREIGN_ROOM_FACTOR"

        qmin = min((x[0] for x in factor_intervals), default=np.nan)
        qmax = max((x[1] for x in factor_intervals), default=np.nan)

        if bool(row["prior_imi"]):
            history_uncertain = existing_unresolved
        elif bool(row["prior_imi_uncertain"]):
            history_uncertain = (
                (viable_existing and existing_unresolved)
                or (viable_existing and viable_new)
            )
        else:
            history_uncertain = False

        value_unresolved = (
            pd.notna(qmin)
            and pd.notna(qmax)
            and not np.isclose(qmin, qmax)
        )
        factor_unresolved = history_uncertain or value_unresolved

        valid_branches = [
            x for x in branch_intervals
            if all(pd.notna(v) for v in x)
        ]

        pre_fif_min = min((x[0] for x in valid_branches), default=np.nan)
        pre_fif_max = max((x[1] for x in valid_branches), default=np.nan)
        pre_ff_min = min((x[2] for x in valid_branches), default=np.nan)
        pre_ff_max = max((x[3] for x in valid_branches), default=np.nan)

        adj_fif_min = min((x[0] * x[4] for x in valid_branches), default=np.nan)
        adj_fif_max = max((x[1] * x[5] for x in valid_branches), default=np.nan)
        adj_ff_min = min((x[2] * x[4] for x in valid_branches), default=np.nan)
        adj_ff_max = max((x[3] * x[5] for x in valid_branches), default=np.nan)

        current_fol = num(row.get("historical_fol"))
        exact_fol = current_fol if pd.notna(current_fol) else previous_fol

        records.append({
            "prior_min": pmin,
            "prior_max": pmax,
            "existing_min": emin,
            "existing_max": emax,
            "new_min": nmin,
            "new_max": nmax,
            "post_min": qmin,
            "post_max": qmax,
            "unresolved": factor_unresolved,
            "source": source,
            "last_reduction": last_reduction,
            "exact_fol": exact_fol,
            "history_uncertain": history_uncertain,
            "value_unresolved": value_unresolved,
            "pre_fif_min": pre_fif_min,
            "pre_fif_max": pre_fif_max,
            "pre_ff_min": pre_ff_min,
            "pre_ff_max": pre_ff_max,
            "adj_fif_min": adj_fif_min,
            "adj_fif_max": adj_fif_max,
            "adj_ff_min": adj_ff_min,
            "adj_ff_max": adj_ff_max,
        })

    f = pd.DataFrame(records, index=r.index)

    r["prior_foreign_room_factor_min"] = f["prior_min"]
    r["prior_foreign_room_factor_max"] = f["prior_max"]
    r["post_factor_if_existing_min"] = f["existing_min"]
    r["post_factor_if_existing_max"] = f["existing_max"]
    r["post_factor_if_new_min"] = f["new_min"]
    r["post_factor_if_new_max"] = f["new_max"]
    r["post_foreign_room_factor_min"] = f["post_min"]
    r["post_foreign_room_factor_max"] = f["post_max"]
    r["foreign_room_factor_value_unresolved"] = f["value_unresolved"].astype(bool)
    r["foreign_room_factor_history_unresolved"] = f["history_uncertain"].astype(bool)
    r["foreign_room_factor_source"] = f["source"].astype("string")

    r["mieu_state"] = final_state(
        r["prior_imi"],
        r["prior_imi_uncertain"],
        r["mieu_state_if_existing"],
        r["mieu_state_if_new"],
    )

    r["foreign_room_factor_unresolved"] = (
        (
            r["foreign_room_factor_value_unresolved"]
            | r["foreign_room_factor_history_unresolved"]
        )
        & ~r["mieu_state"].eq("FAIL")
    )

    r["post_mieu"] = pd.array(
        np.where(
            r["mieu_state"].eq("PASS"),
            True,
            np.where(r["mieu_state"].eq("FAIL"), False, pd.NA),
        ),
        dtype="boolean",
    )

    r["pre_adjusted_fif_min"] = f["pre_fif_min"]
    r["pre_adjusted_fif_max"] = f["pre_fif_max"]
    r["pre_adjusted_ff_mcap_usd_min"] = f["pre_ff_min"]
    r["pre_adjusted_ff_mcap_usd_max"] = f["pre_ff_max"]
    r["post_adjusted_fif_min"] = f["adj_fif_min"]
    r["post_adjusted_fif_max"] = f["adj_fif_max"]
    r["post_adjusted_ff_mcap_usd_min"] = f["adj_ff_min"]
    r["post_adjusted_ff_mcap_usd_max"] = f["adj_ff_max"]

    r["review_fif_value_unresolved"] = (
        r["pre_adjusted_fif_min"].notna()
        & r["pre_adjusted_fif_max"].notna()
        & ~np.isclose(
            r["pre_adjusted_fif_min"],
            r["pre_adjusted_fif_max"],
        )
        & ~r["mieu_state"].eq("FAIL")
    )

    new_factor_state = {}

    for i, row in r.iterrows():
        if row["mieu_state"] not in {"PASS", "UNRESOLVED"}:
            continue

        qmin = f.at[i, "post_min"]
        qmax = f.at[i, "post_max"]

        if pd.isna(qmin) or pd.isna(qmax):
            continue

        new_factor_state[str(row["security_id"])] = {
            "min": float(qmin),
            "max": float(qmax),
            "last_reduction": f.at[i, "last_reduction"],
            "exact_fol": f.at[i, "exact_fol"],
            "history_uncertain": bool(f.at[i, "history_uncertain"]),
        }

    return r, new_factor_state

def main():
    inv = normalize(pd.read_parquet(INV_PATH))
    fr = normalize(pd.read_parquet(FR_PATH))
    snap = normalize(pd.read_parquet(SNAP_PATH))
    size = normalize(pd.read_parquet(SIZE_PATH))
    review_fif = normalize(pd.read_parquet(REVIEW_FIF_PATH))
    master = pd.read_parquet(MASTER_PATH)

    require(inv, [
        "review", "security_id", "company_key", "fif",
        "full_market_cap_usd", "ff_market_cap_usd", "company_full_mcap_usd",
        "review_base_universe_pass", "market_data_complete",
        "liquidity_new_pass", "liquidity_existing_pass",
        "fif_regular_pass", "trading_length_regular_pass",
        "was_standard_constituent", "price_cutoff_proxy",
        "price_cutoff_market_date", "liquidity_cutoff_date",
        "liquidity_observation_date",
    ], "investability_universe")

    require(fr, [
        "review", "security_id",
        "new_entry_pass_if_nonconstituent",
        "new_entry_factor_min_if_pass",
        "new_entry_factor_max_if_pass",
        "post_factor_if_pre_1_min", "post_factor_if_pre_1_max",
        "post_factor_if_pre_0_5_min", "post_factor_if_pre_0_5_max",
        "post_factor_if_pre_0_25_min", "post_factor_if_pre_0_25_max",
        "historical_fol", "historical_fol_effective_date",
        "india_new_entry_alert_pass_if_nonconstituent",
        "india_existing_alert_pass_if_constituent",
    ], "foreign_room")

    require(
        snap,
        [
            "review", "security_id", "company_key",
            "full_market_cap_usd", "price_cutoff_refresh_valid",
        ],
        "india_price_cutoff_snapshot",
    )

    require(size, ["review", "standard_gmsr_usd"], "india_size_cutoffs")

    require(review_fif, [
        "review", "security_id",
        "fif_existing_min", "fif_existing_max",
        "fif_new_min", "fif_new_max",
        "fif_regime",
    ], "review_fif")


    if inv.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate investability keys.")
    if fr.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate foreign-room keys.")
    if review_fif.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate review-FIF keys.")

    inv = inv.merge(
        review_fif[
            [
                "review", "security_id",
                "fif_existing_min", "fif_existing_max",
                "fif_new_min", "fif_new_max",
                "fif_regime",
            ]
        ],
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    

    for c in [
        "price_cutoff_proxy",
        "price_cutoff_market_date",
        "liquidity_cutoff_date",
        "liquidity_observation_date",
    ]:
        inv[c] = pd.to_datetime(inv[c], errors="coerce")

    fr["historical_fol_effective_date"] = pd.to_datetime(
        fr["historical_fol_effective_date"],
        errors="coerce",
    )

    old_state_cols = [
        "was_imi_constituent",
        "mieu_liquidity_pass",
        "mieu_fif_pass",
        "mieu_trading_pass",
        "mieu_foreign_room_pass",
        "mieu_pass",
    ]

    inv = inv.rename(
        columns={
            c: f"base_{c}"
            for c in old_state_cols
            if c in inv.columns
        }
    )

    fr_keep = [
        "review", "security_id",
        "foreign_room", "foreign_room_lower_bound",
        "foreign_room_status",
        "historical_fol", "historical_fol_effective_date",
        "new_entry_pass_if_nonconstituent",
        "new_entry_factor_min_if_pass",
        "new_entry_factor_max_if_pass",
        "post_factor_if_pre_1_min", "post_factor_if_pre_1_max",
        "post_factor_if_pre_0_5_min", "post_factor_if_pre_0_5_max",
        "post_factor_if_pre_0_25_min", "post_factor_if_pre_0_25_max",
        "india_red_flag", "india_breach", "india_alert_is_exact",
        "india_new_entry_alert_pass_if_nonconstituent",
        "india_existing_alert_pass_if_constituent",
        "india_red_flag_breach_status",
    ]

    inv = inv.merge(
        fr[fr_keep],
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    gmsr = (
        size.drop_duplicates("review")
        .set_index("review")["standard_gmsr_usd"]
        .to_dict()
    )

    bootstrap_ids = load_smin_bootstrap(inv, master)

    outputs = []
    summaries = []

    prev_pass_ids = set()
    prev_unresolved_ids = set()
    prev_pass_companies = set()
    prev_unresolved_companies = set()
    factor_state = {}

    for review in REVIEWS:
        r = inv[inv["review"].eq(review)].copy()
        s = snap[snap["review"].eq(review)].copy()

        if r.empty or review not in gmsr:
            raise RuntimeError(f"{review}: missing inputs.")

        cutoff = pd.Timestamp(PRICE_CUTOFF[review])

        bad_cutoff = (
            r["price_cutoff_proxy"].notna()
            & r["price_cutoff_proxy"].ne(cutoff)
        )

        if bad_cutoff.any():
            raise RuntimeError(f"{review}: Price-Cutoff mismatch.")

        standard = (
            r["was_standard_constituent"]
            .fillna(False)
            .astype(bool)
        )

        published_std_ids = set(
            r.loc[standard, "security_id"]
        )

        published_std_companies = set(
            r.loc[standard, "company_key"].dropna()
        )

        if review == "2023-05":
            definite_ids = bootstrap_ids
            uncertain_ids = set()

            definite_companies = set(
                r.loc[
                    r["security_id"].isin(definite_ids),
                    "company_key",
                ].dropna()
            )

            uncertain_companies = set()
            prior_source = BOOTSTRAP_METHOD

        else:
            # Published pre-review Standard is a known subset of prior MIEU.
            definite_ids = prev_pass_ids | published_std_ids
            uncertain_ids = prev_unresolved_ids - definite_ids

            definite_companies = (
                prev_pass_companies | published_std_companies
            )

            uncertain_companies = (
                prev_unresolved_companies - definite_companies
            )

            prior_source = (
                "PREVIOUS_REVIEW_RECURSION_PLUS_"
                "PUBLISHED_STANDARD_SUBSET"
            )

        r["prior_imi"] = r["security_id"].isin(definite_ids)
        r["prior_imi_uncertain"] = (
            r["security_id"].isin(uncertain_ids)
        )

        r["prior_imi_source"] = prior_source
        r["prior_imi_bootstrap_method"] = (
            BOOTSTRAP_METHOD if review == "2023-05" else pd.NA
        )

        n = prior_standard_n(r)
        eumsr = EUMSR_USD_M[review] * 1_000_000

        raw_lo, interim_lo, valued_lo = ranked_cutoff(
            s,
            definite_companies,
            n,
            eumsr,
        )

        raw_hi, interim_hi, valued_hi = ranked_cutoff(
            s,
            definite_companies | uncertain_companies,
            n,
            eumsr,
        )

        if interim_lo > interim_hi + 1e-6:
            raise RuntimeError(
                f"{review}: invalid Interim cutoff bounds."
            )

        r["prior_standard_segment_number"] = n
        r["prior_standard_segment_number_source"] = (
            "PUBLISHED_PRE_REVIEW_STANDARD_COMPANY_COUNT_PROXY"
        )

        r["prior_mieu_company_count_min"] = len(definite_companies)
        r["prior_mieu_company_count_max"] = len(
            definite_companies | uncertain_companies
        )

        r["prior_mieu_companies_valued_min"] = valued_lo
        r["prior_mieu_companies_valued_max"] = valued_hi

        r["interim_standard_cutoff_raw_min_usd"] = raw_lo
        r["interim_standard_cutoff_raw_max_usd"] = raw_hi
        r["interim_standard_cutoff_min_usd"] = interim_lo
        r["interim_standard_cutoff_max_usd"] = interim_hi
        r["equity_universe_min_size_usd"] = eumsr

        base = (
            r["review_base_universe_pass"]
            .fillna(False)
            .astype(bool)
            & r["market_data_complete"]
            .fillna(False)
            .astype(bool)
        )

        existing_needed = base & (r["prior_imi"] | r["prior_imi_uncertain"])
        new_needed = base & ~r["prior_imi"]

        ex_min = pd.to_numeric(r["fif_existing_min"], errors="coerce")
        ex_max = pd.to_numeric(r["fif_existing_max"], errors="coerce")
        nw_min = pd.to_numeric(r["fif_new_min"], errors="coerce")
        nw_max = pd.to_numeric(r["fif_new_max"], errors="coerce")

        bad_existing = existing_needed & ex_min.isna().ne(ex_max.isna())
        bad_new = new_needed & nw_min.isna().ne(nw_max.isna())

        if bad_existing.any() or bad_new.any():
            raise RuntimeError(f"{review}: malformed review-FIF interval.")

        missing_existing = existing_needed & ex_min.isna() & ex_max.isna()
        missing_new = new_needed & nw_min.isna() & nw_max.isna()

        # No PIT FIF evidence: preserve uncertainty rather than use later filings.
        r.loc[missing_existing, ["fif_existing_min", "fif_existing_max"]] = [0.0, 1.0]
        r.loc[missing_new, ["fif_new_min", "fif_new_max"]] = [0.0, 1.0]

        r["review_fif_missing_existing"] = missing_existing
        r["review_fif_missing_new"] = missing_new

        full = pd.to_numeric(
            r["company_full_mcap_usd"],
            errors="coerce",
        )
        security_full = pd.to_numeric(
            r["full_market_cap_usd"],
            errors="coerce",
        )

        r["review_fif_existing_min"] = pd.to_numeric(
            r["fif_existing_min"], errors="coerce"
        )
        r["review_fif_existing_max"] = pd.to_numeric(
            r["fif_existing_max"], errors="coerce"
        )
        r["review_fif_new_min"] = pd.to_numeric(
            r["fif_new_min"], errors="coerce"
        )
        r["review_fif_new_max"] = pd.to_numeric(
            r["fif_new_max"], errors="coerce"
        )

        r["review_ff_mcap_existing_min_usd"] = (
            security_full * r["review_fif_existing_min"]
        )
        r["review_ff_mcap_existing_max_usd"] = (
            security_full * r["review_fif_existing_max"]
        )
        r["review_ff_mcap_new_min_usd"] = (
            security_full * r["review_fif_new_min"]
        )
        r["review_ff_mcap_new_max_usd"] = (
            security_full * r["review_fif_new_max"]
        )

        ff_new_min = r["review_ff_mcap_new_min_usd"]
        ff_new_max = r["review_ff_mcap_new_max_usd"]

        new_eumsr = interval_ge(
            ff_new_min,
            ff_new_max,
            0.5 * eumsr,
        )
        new_eumsr.loc[full.lt(eumsr)] = False

        existing_eumsr = pd.Series(
            True,
            index=r.index,
            dtype="boolean",
        )

        new_liq = r["liquidity_new_pass"].astype("boolean")
        existing_liq = (
            r["liquidity_existing_pass"].astype("boolean")
        )

        fif_new_min = r["review_fif_new_min"]
        fif_new_max = r["review_fif_new_max"]

        regular_fif_state = interval_ge(
            fif_new_min,
            fif_new_max,
            0.15,
        )

        low_candidate = regular_fif_state.eq(False).fillna(False)
        low_pass = (
            low_candidate
            & ff_new_min.ge(0.9 * interim_hi)
        )
        low_fail = (
            low_candidate
            & ff_new_max.lt(0.9 * interim_lo)
        )
        low_uncertain = (
            low_candidate & ~low_pass & ~low_fail
        )

        new_fif = regular_fif_state.copy()
        new_fif.loc[low_pass] = True
        new_fif.loc[low_fail] = False
        new_fif.loc[low_uncertain] = pd.NA

        straddle = regular_fif_state.isna()
        straddle_pass = (
            straddle
            & fif_new_min.notna()
            & fif_new_max.notna()
            & ff_new_min.ge(0.9 * interim_hi)
        )
        new_fif.loc[straddle_pass] = True

        existing_fif = pd.Series(
            True,
            index=r.index,
            dtype="boolean",
        )

        r["low_fif_exception_applied"] = low_pass
        r["low_fif_exception_uncertain"] = (
            low_uncertain | (straddle & ~straddle_pass)
        )

        regular_trading = (
            r["trading_length_regular_pass"]
            .fillna(False)
            .astype(bool)
        )

        short_candidate = ~regular_trading

        short_pass = (
            short_candidate
            & full.ge(1.8 * interim_hi)
            & ff_new_min.ge(0.9 * interim_hi)
        )

        short_fail = (
            short_candidate
            & (
                full.lt(1.8 * interim_lo)
                | ff_new_max.lt(0.9 * interim_lo)
            )
        )

        short_uncertain = (
            short_candidate & ~short_pass & ~short_fail
        )

        new_trading = pd.Series(
            pd.array(regular_trading, dtype="boolean"),
            index=r.index,
        )

        new_trading.loc[short_uncertain] = pd.NA
        new_trading.loc[short_pass] = True
        new_trading.loc[short_fail] = False

        existing_trading = pd.Series(
            True,
            index=r.index,
            dtype="boolean",
        )

        r["short_trading_exception_applied"] = short_pass
        r["short_trading_exception_uncertain"] = (
            short_uncertain
        )

        new_fr = (
            r["new_entry_pass_if_nonconstituent"]
            .astype("boolean")
        )

        existing_fr = pd.Series(
            True,
            index=r.index,
            dtype="boolean",
        )

        new_alert = (
            r[
                "india_new_entry_alert_pass_if_nonconstituent"
            ].astype("boolean")
        )

        existing_alert = (
            r[
                "india_existing_alert_pass_if_constituent"
            ].astype("boolean")
        )

        r["mieu_eumsr_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_eumsr,
            new_eumsr,
        )

        r["mieu_liquidity_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_liq,
            new_liq,
        )

        r["mieu_fif_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_fif,
            new_fif,
        )

        r["mieu_trading_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_trading,
            new_trading,
        )

        r["mieu_foreign_room_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_fr,
            new_fr,
        )

        r["mieu_alert_pass"] = combine_bool(
            r["prior_imi"],
            r["prior_imi_uncertain"],
            existing_alert,
            new_alert,
        )

        # LAF uses pre-adjustment FF market cap of existing Standard names.
        # Review-specific FIF can be an interval, so bound both the country
        # weight and the >0.5*GMSR test rather than falling back to legacy FF.
        std_ff_min = r["review_ff_mcap_existing_min_usd"].where(standard, 0.0)
        std_ff_max = r["review_ff_mcap_existing_max_usd"].where(standard, 0.0)
        total_min = float(std_ff_min.sum())
        total_max = float(std_ff_max.sum())

        own_min = r["review_ff_mcap_existing_min_usd"]
        own_max = r["review_ff_mcap_existing_max_usd"]
        other_min = (total_min - std_ff_min).clip(lower=0.0)
        other_max = (total_max - std_ff_max).clip(lower=0.0)

        weight_min_denom = own_min + other_max
        weight_max_denom = own_max + other_min

        r["existing_standard_weight_min"] = np.where(
            standard & weight_min_denom.gt(0),
            own_min / weight_min_denom,
            np.nan,
        )
        r["existing_standard_weight_max"] = np.where(
            standard & weight_max_denom.gt(0),
            own_max / weight_max_denom,
            np.nan,
        )
        r["existing_standard_weight_proxy"] = (
            r["existing_standard_weight_min"]
            + r["existing_standard_weight_max"]
        ) / 2.0

        laf_base = (
            standard
            & r["prior_imi"]
            & existing_liq.eq(False).fillna(False)
        )
        laf_threshold = 0.5 * float(gmsr[review])

        r["laf_initial_trigger"] = (
            laf_base
            & r["existing_standard_weight_min"].gt(0.10)
            & own_min.gt(laf_threshold)
        )
        r["laf_initial_trigger_unresolved"] = (
            laf_base
            & r["existing_standard_weight_max"].gt(0.10)
            & own_max.gt(laf_threshold)
            & ~r["laf_initial_trigger"]
        )

        laf_any = (
            r["laf_initial_trigger"]
            | r["laf_initial_trigger_unresolved"]
        )
        if laf_any.any():
            bad = r.loc[
                laf_any,
                [
                    "security_id",
                    "nse_symbol",
                    "existing_standard_weight_min",
                    "existing_standard_weight_max",
                    "review_ff_mcap_existing_min_usd",
                    "review_ff_mcap_existing_max_usd",
                    "laf_initial_trigger",
                    "laf_initial_trigger_unresolved",
                ],
            ]
            raise RuntimeError(
                f"{review}: LAF trigger/uncertainty found:\n"
                + bad.to_string(index=False)
            )

        r["liquidity_adjustment_factor"] = 1.0

        r["mieu_state_if_existing"] = branch_state(
            base,
            [
                existing_eumsr,
                existing_liq,
                existing_fif,
                existing_trading,
                existing_fr,
                existing_alert,
            ],
        )

        r["mieu_state_if_new"] = branch_state(
            base,
            [
                new_eumsr,
                new_liq,
                new_fif,
                new_trading,
                new_fr,
                new_alert,
            ],
        )

        r["existing_factor_forces_fail"] = False

        r, factor_state = apply_factors(
            r,
            review,
            factor_state,
        )

        prev_pass_ids = set(
            r.loc[
                r["mieu_state"].eq("PASS"),
                "security_id",
            ]
        )

        prev_unresolved_ids = set(
            r.loc[
                r["mieu_state"].eq("UNRESOLVED"),
                "security_id",
            ]
        )

        prev_pass_companies = set(
            r.loc[
                r["mieu_state"].eq("PASS"),
                "company_key",
            ].dropna()
        )

        prev_unresolved_companies = set(
            r.loc[
                r["mieu_state"].eq("UNRESOLVED"),
                "company_key",
            ].dropna()
        )

        summaries.append({
            "review": review,
            "prior_definite": int(r["prior_imi"].sum()),
            "prior_uncertain": int(
                r["prior_imi_uncertain"].sum()
            ),
            "prior_companies_min": len(definite_companies),
            "prior_companies_max": len(
                definite_companies | uncertain_companies
            ),
            "prior_standard_n": n,
            "interim_min_bn": interim_lo / 1e9,
            "interim_max_bn": interim_hi / 1e9,
            "post_pass": int(
                r["mieu_state"].eq("PASS").sum()
            ),
            "post_fail": int(
                r["mieu_state"].eq("FAIL").sum()
            ),
            "post_unresolved": int(
                r["mieu_state"].eq("UNRESOLVED").sum()
            ),
            "low_fif_uncertain": int(
                r["low_fif_exception_uncertain"].sum()
            ),
            "short_trade_uncertain": int(
                r["short_trading_exception_uncertain"].sum()
            ),
            "factor_unresolved": int(
                r["foreign_room_factor_unresolved"].sum()
            ),
            "fif_value_unresolved": int(
                r["review_fif_value_unresolved"].sum()
            ),
            "pit_fif_missing": int(
    (r["review_fif_missing_existing"] | r["review_fif_missing_new"]).sum()
)
        })

        outputs.append(r)

    out = pd.concat(outputs, ignore_index=True)

    price_lookahead = (
        out["price_cutoff_market_date"].notna()
        & out["price_cutoff_market_date"].gt(
            out["price_cutoff_proxy"]
        )
    )

    liquidity_lookahead = (
        out["liquidity_observation_date"].notna()
        & out["liquidity_observation_date"].gt(
            out["liquidity_cutoff_date"]
        )
    )

    if price_lookahead.any() or liquidity_lookahead.any():
        raise RuntimeError("MIEU state contains PIT lookahead.")

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate MIEU state keys.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- MIEU REVIEW STATE ---")
    print(pd.DataFrame(summaries).to_string(index=False))

    print("\nRows:", len(out))
    print(
        "Duplicates:",
        int(
            out.duplicated(
                ["review", "security_id"]
            ).sum()
        ),
    )
    print("Price lookahead:", int(price_lookahead.sum()))
    print(
        "Liquidity lookahead:",
        int(liquidity_lookahead.sum()),
    )
    print(
        "May bootstrap securities:",
        int(
            out.loc[
                out["review"].eq("2023-05"),
                "prior_imi",
            ].sum()
        ),
    )
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()