from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


CANDIDATE_PATH = Path("data/processed/foreign_room_candidates.parquet")
INV_PATH = Path("data/processed/investability_universe.parquet")
MASTER_PATH = Path("data/processed/historical_security_master.parquet")
OUT_PATH = Path("data/processed/historical_foreign_ownership.parquet")

CACHE_PATH = Path("data/raw/foreign_ownership/historical_foreign_ownership_filings.parquet")
FAILED_PATH = Path("data/raw/foreign_ownership/historical_foreign_ownership_failed.csv")

NSE_BASE = "https://www.nseindia.com"
SHAREHOLDING_URL = "https://www.nseindia.com/api/corporate-share-holdings-master"

SYMBOL_OVERRIDES = {
    "HIST_HDFC": "HDFC",
    "HIST_TATAMTRDVR": "TATAMTRDVR",
}

# Known rows where prior targeted work must take precedence over the generic parser.
PROTECTED_KEYS = {
    ("2023-11", "SEC001008"),  # JIOFIN corporate-event treatment
    ("2023-11", "SEC000935"),  # INDUSINDBK total-foreign correction
    ("2024-08", "SEC000269"),  # BANDHANBNK targeted FO repair
    ("2025-11", "SEC002213"),  # VMM dead historical source
}

PROTECTED_SECURITIES = {
    "SEC000972",  # ITC foreign-ownership numerator special case
}

PROTECTED_TEXT = r"MANUAL|PATCH|REPAIR|SPECIAL|OVERRIDE|CORPORATE_EVENT|SOURCE_DEAD|DEAD_XBRL"
FOREIGN_TERMS = [
    "foreignportfolio", "foreigninstitutional", "foreigncompanies", "foreigncompany",
    "foreigncorporate", "foreignbodycorporate", "foreignventurecapital",
    "nonresident", "nri", "overseas",
]


def clean_tag(tag):
    return tag.split("}")[-1].strip()


def norm_symbol(x):
    if pd.isna(x): return pd.NA
    x = str(x).strip().upper()
    return x or pd.NA


def parse_date(x):
    return pd.to_datetime(x, errors="coerce", dayfirst=True)


def create_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern",
    })
    s.get(NSE_BASE, timeout=20)
    return s


def get_filings(symbol, session):
    r = session.get(SHAREHOLDING_URL, params={"index": "equities", "symbol": symbol}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list): return data
    if isinstance(data, dict): return data.get("data", [])
    return []


def is_foreign_member(text):
    if not text: return False
    x = str(text).lower().replace("_", "").replace("-", "").replace(" ", "")
    return any(term in x for term in FOREIGN_TERMS)


def extract_foreign_ownership(url, session):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    contexts = {}

    for e in root.iter():
        if clean_tag(e.tag) != "context": continue
        cid = e.attrib.get("id")
        if not cid: continue
        members = [c.text.strip() for c in e.iter() if clean_tag(c.tag) in {"explicitMember", "typedMember"} and c.text]
        foreign = [m for m in members if is_foreign_member(m)]
        if foreign: contexts[cid] = foreign

    if not contexts: return np.nan, 0, None

    values = []
    for e in root.iter():
        if clean_tag(e.tag) != "ShareholdingAsAPercentageOfTotalNumberOfShares": continue
        cid = e.attrib.get("contextRef")
        if cid not in contexts or e.text is None: continue
        try: value = float(e.text.strip())
        except ValueError: continue
        if value < 0 or value > 1: continue
        values.append({"context": cid, "member": " | ".join(contexts[cid]), "value": value})

    if not values: return np.nan, 0, None

    z = pd.DataFrame(values).drop_duplicates(["context", "member"])
    total = z["value"].sum()
    members = " || ".join(sorted(z["member"].unique()))

    if total < 0 or total > 1.0001: return np.nan, len(z), members
    return float(total), len(z), members


def prepare_inputs():
    candidates = pd.read_parquet(CANDIDATE_PATH)
    inv = pd.read_parquet(INV_PATH)
    master = pd.read_parquet(MASTER_PATH)
    old = pd.read_parquet(OUT_PATH)

    for df in [candidates, inv, master, old]: df["security_id"] = df["security_id"].astype("string")
    candidates["review"] = candidates["review"].astype(str)
    inv["review"] = inv["review"].astype(str)
    old["review"] = old["review"].astype(str)

    if candidates.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate candidate keys.")
    if old.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate canonical FO keys.")

    dates = inv[["review", "security_id", "price_cutoff_proxy"]].drop_duplicates(["review", "security_id"])
    dates["price_cutoff_proxy"] = pd.to_datetime(dates["price_cutoff_proxy"], errors="coerce")
    candidates = candidates.merge(dates, on=["review", "security_id"], how="left", validate="one_to_one")

    if candidates["price_cutoff_proxy"].isna().any(): raise RuntimeError("Candidate row missing Price Cutoff.")

    master_cols = [c for c in ["security_id", "nse_symbol", "isin"] if c in master.columns]
    m = master[master_cols].copy()
    if "nse_symbol" in m.columns: m["nse_symbol"] = m["nse_symbol"].map(norm_symbol)
    m = m.drop_duplicates("security_id", keep="last")

    securities = candidates[["security_id", "nse_symbol"]].drop_duplicates("security_id", keep="last")
    securities["nse_symbol"] = securities["nse_symbol"].map(norm_symbol)
    securities = securities.merge(m.rename(columns={"nse_symbol": "master_symbol", "isin": "master_isin"}), on="security_id", how="left", validate="one_to_one")

    securities["fetch_symbol"] = securities["nse_symbol"].fillna(securities.get("master_symbol"))
    for sid, symbol in SYMBOL_OVERRIDES.items(): securities.loc[securities["security_id"].eq(sid), "fetch_symbol"] = symbol
    securities["fetch_symbol"] = securities["fetch_symbol"].map(norm_symbol)

    return candidates, securities, old


def load_cache():
    if not CACHE_PATH.exists(): return pd.DataFrame()
    x = pd.read_parquet(CACHE_PATH)
    x["security_id"] = x["security_id"].astype("string")
    for c in ["report_date", "submission_date", "available_date"]: x[c] = pd.to_datetime(x[c], errors="coerce")
    return x


def save_cache(rows):
    if not rows: return
    x = pd.DataFrame(rows)
    x["security_id"] = x["security_id"].astype("string")
    for c in ["report_date", "submission_date", "available_date"]: x[c] = pd.to_datetime(x[c], errors="coerce")

    x = x.sort_values(["security_id", "available_date", "xbrl_url"], na_position="last")
    x = x.drop_duplicates(["security_id", "report_date", "available_date", "xbrl_url", "fetch_status"], keep="last")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(CACHE_PATH, index=False)


def fetch_cache(securities):
    cached = load_cache()
    rows = cached.to_dict("records") if not cached.empty else []

    completed = set()
    if not cached.empty:
        good = cached[cached["fetch_status"].isin(["OK", "NO_FILINGS"])]
        completed = set(good["security_id"].dropna().astype(str))

    todo = securities[~securities["security_id"].astype(str).isin(completed)].copy()
    failures = []
    session = create_session()

    print("\n--- FOREIGN OWNERSHIP FETCH ---")
    print("Candidate securities:", len(securities))
    print("Already cached:", len(completed))
    print("To fetch:", len(todo))

    for i, (_, sec) in enumerate(todo.iterrows(), 1):
        sid, symbol = sec["security_id"], sec["fetch_symbol"]
        isin = sec.get("master_isin", pd.NA)

        if pd.isna(symbol):
            rows.append({
                "security_id": sid, "nse_symbol": pd.NA, "source_isin": isin,
                "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                "foreign_ownership": np.nan, "foreign_context_count": np.nan,
                "foreign_members": pd.NA, "xbrl_url": pd.NA, "source": pd.NA,
                "mapping_method": pd.NA, "fetch_status": "NO_SYMBOL",
            })
            failures.append({"security_id": sid, "nse_symbol": None, "error": "NO_SYMBOL"})
            continue

        try:
            filings = get_filings(symbol, session)
            xbrls = [f for f in filings if isinstance(f.get("xbrl"), str) and f["xbrl"].startswith("http")]

            if not xbrls:
                rows.append({
                    "security_id": sid, "nse_symbol": symbol, "source_isin": isin,
                    "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                    "foreign_ownership": np.nan, "foreign_context_count": np.nan,
                    "foreign_members": pd.NA, "xbrl_url": pd.NA, "source": "NSE",
                    "mapping_method": "security_id_to_nse_symbol", "fetch_status": "NO_FILINGS",
                })

            for filing in xbrls:
                url = filing["xbrl"]
                report_date = parse_date(filing.get("date"))
                submission_date = parse_date(filing.get("submissionDate"))

                try:
                    fo, count, members = extract_foreign_ownership(url, session)
                    rows.append({
                        "security_id": sid, "nse_symbol": symbol, "source_isin": isin,
                        "report_date": report_date, "submission_date": submission_date,
                        "available_date": submission_date, "foreign_ownership": fo,
                        "foreign_context_count": count, "foreign_members": members,
                        "xbrl_url": url, "source": "NSE_XBRL",
                        "mapping_method": "security_id_symbol_override" if sid in SYMBOL_OVERRIDES else "security_id_to_nse_symbol",
                        "fetch_status": "OK",
                    })
                except Exception as e:
                    rows.append({
                        "security_id": sid, "nse_symbol": symbol, "source_isin": isin,
                        "report_date": report_date, "submission_date": submission_date,
                        "available_date": submission_date, "foreign_ownership": np.nan,
                        "foreign_context_count": np.nan, "foreign_members": pd.NA,
                        "xbrl_url": url, "source": "NSE_XBRL",
                        "mapping_method": "security_id_to_nse_symbol",
                        "fetch_status": "OK", "parse_error": str(e)[:300],
                    })

        except Exception as e:
            failures.append({"security_id": sid, "nse_symbol": symbol, "error": str(e)[:300]})
            rows.append({
                "security_id": sid, "nse_symbol": symbol, "source_isin": isin,
                "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                "foreign_ownership": np.nan, "foreign_context_count": np.nan,
                "foreign_members": pd.NA, "xbrl_url": pd.NA, "source": pd.NA,
                "mapping_method": "security_id_to_nse_symbol", "fetch_status": "FETCH_FAILED",
            })

        if i % 25 == 0:
            save_cache(rows)
            print(f"Processed: {i}/{len(todo)} | failures: {len(failures)}")

        time.sleep(0.05)

    save_cache(rows)

    if failures:
        pd.DataFrame(failures).to_csv(FAILED_PATH, index=False)

    return load_cache(), failures


def select_pit(candidates, cache):
    valid_cache = cache[cache["fetch_status"].eq("OK")].copy()
    grouped = {sid: g.copy() for sid, g in valid_cache.groupby("security_id", sort=False)}
    rows = []

    for _, r in candidates.iterrows():
        sid, cutoff = r["security_id"], r["price_cutoff_proxy"]
        g = grouped.get(sid)

        base = {
            "review": r["review"],
            "security_id": sid,
            "nse_symbol": r.get("nse_symbol", pd.NA),
            "foreign_ownership_cutoff_date": cutoff,
            "foreign_ownership_cutoff_is_proxy": True,
            "foreign_ownership_cutoff_method": "DETERMINISTIC_PRICE_CUTOFF_PROXY",
        }

        if g is None:
            rows.append({**base, "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                         "foreign_ownership": np.nan, "foreign_context_count": np.nan, "foreign_members": pd.NA,
                         "source": pd.NA, "xbrl_url": pd.NA, "mapping_method": pd.NA,
                         "data_quality_flag": "NO_FILINGS_FOUND"})
            continue

        eligible = g[g["available_date"].notna() & g["available_date"].le(cutoff)].copy()

        if eligible.empty:
            rows.append({**base, "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                         "foreign_ownership": np.nan, "foreign_context_count": np.nan, "foreign_members": pd.NA,
                         "source": pd.NA, "xbrl_url": pd.NA, "mapping_method": pd.NA,
                         "data_quality_flag": "NO_ELIGIBLE_FILING"})
            continue

        chosen = eligible.sort_values(["report_date", "available_date"]).iloc[-1]
        fo = chosen["foreign_ownership"]

        if pd.isna(fo): quality = "PIT_FILING_PARSE_MISSING"
        elif fo < 0 or fo > 1: quality = "PIT_FILING_INVALID_VALUE"
        else: quality = "OK"

        rows.append({
            **base,
            "report_date": chosen["report_date"],
            "submission_date": chosen["submission_date"],
            "available_date": chosen["available_date"],
            "foreign_ownership": fo,
            "foreign_context_count": chosen["foreign_context_count"],
            "foreign_members": chosen["foreign_members"],
            "source": chosen["source"],
            "xbrl_url": chosen["xbrl_url"],
            "mapping_method": chosen["mapping_method"],
            "data_quality_flag": quality,
        })

    return pd.DataFrame(rows)


def protected_old_rows(old):
    protected = pd.Series(False, index=old.index)

    key_strings = old["review"].astype(str) + "|" + old["security_id"].astype(str)
    protected_keys = {review + "|" + sid for review, sid in PROTECTED_KEYS}
    protected |= key_strings.isin(protected_keys)
    protected |= old["security_id"].isin(PROTECTED_SECURITIES)

    for c in ["source", "mapping_method", "data_quality_flag", "provenance", "status"]:
        if c in old.columns:
            protected |= old[c].astype("string").str.contains(PROTECTED_TEXT, case=False, regex=True, na=False)

    return protected


def merge_canonical(old, fresh, candidates):
    old = old.copy()
    fresh = fresh.copy()

    for df in [old, fresh]:
        df["review"] = df["review"].astype(str)
        df["security_id"] = df["security_id"].astype("string")
        for c in ["report_date", "submission_date", "available_date", "foreign_ownership_cutoff_date"]:
            if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")

    old["protected_row"] = protected_old_rows(old)

    old_map = {
        (r["review"], r["security_id"]): r
        for _, r in old.iterrows()
    }
    fresh_map = {
        (r["review"], r["security_id"]): r
        for _, r in fresh.iterrows()
    }
    cutoff_map = {
        (r["review"], r["security_id"]): r["price_cutoff_proxy"]
        for _, r in candidates.iterrows()
    }

    candidate_keys = set(cutoff_map)
    selected = []
    actions = []

    for key in candidate_keys:
        o = old_map.get(key)
        n = fresh_map.get(key)

        if n is None:
            raise RuntimeError(f"Fresh FO reconstruction missing candidate key: {key}")

        if o is None:
            chosen = n.copy()
            action = "ADD_NEW_KEY"

        elif bool(o["protected_row"]):
            chosen = o.copy()
            action = "PRESERVE_PROTECTED"

        else:
            old_fo = pd.to_numeric(pd.Series([o.get("foreign_ownership")]), errors="coerce").iloc[0]
            new_fo = pd.to_numeric(pd.Series([n.get("foreign_ownership")]), errors="coerce").iloc[0]
            old_date = pd.to_datetime(o.get("available_date"), errors="coerce")
            new_date = pd.to_datetime(n.get("available_date"), errors="coerce")

            if pd.notna(new_fo) and (
                pd.isna(old_fo)
                or (
                    pd.notna(new_date)
                    and (
                        pd.isna(old_date)
                        or new_date > old_date
                    )
                )
            ):
                chosen = n.copy()
                action = "USE_NEWER_PIT"

            elif pd.notna(old_fo):
                chosen = o.copy()
                action = "PRESERVE_OLD_RESOLVED"

            else:
                chosen = n.copy()
                action = "USE_FRESH_UNRESOLVED"

        chosen = chosen.to_dict()
        chosen["foreign_ownership_cutoff_date"] = cutoff_map[key]
        chosen["foreign_ownership_cutoff_is_proxy"] = True
        chosen["foreign_ownership_cutoff_method"] = "DETERMINISTIC_PRICE_CUTOFF_PROXY"
        chosen["canonical_row_action"] = action
        chosen.pop("protected_row", None)

        selected.append(chosen)
        actions.append(action)

    current = pd.DataFrame(selected)

    old_keys = pd.MultiIndex.from_frame(old[["review", "security_id"]])
    required_keys = pd.MultiIndex.from_tuples(candidate_keys, names=["review", "security_id"])
    extras = old.loc[~old_keys.isin(required_keys)].drop(columns=["protected_row"], errors="ignore").copy()
    extras["canonical_row_action"] = "PRESERVE_OLD_NOT_CURRENTLY_REQUIRED"

    out = pd.concat([extras, current], ignore_index=True, sort=False)
    out["security_id"] = out["security_id"].astype("string")
    out = out.sort_values(["review", "security_id"]).drop_duplicates(["review", "security_id"], keep="last").reset_index(drop=True)

    return out, pd.Series(actions, dtype="string")


def main():
    candidates, securities, old = prepare_inputs()
    cache, failures = fetch_cache(securities)
    fresh = select_pit(candidates, cache)

    lookahead = fresh["available_date"].notna() & fresh["available_date"].gt(fresh["foreign_ownership_cutoff_date"])
    if lookahead.any(): raise RuntimeError("Price-Cutoff lookahead detected.")

    out, actions = merge_canonical(old, fresh, candidates)

    if out.duplicated(["review", "security_id"]).any(): raise RuntimeError("Duplicate canonical FO keys.")

    candidate_result = out.merge(
        candidates[["review", "security_id"]].assign(_candidate=True),
        on=["review", "security_id"], how="inner", validate="one_to_one"
    )

    expected = len(candidates)
    if len(candidate_result) != expected: raise RuntimeError(f"Candidate FO coverage mismatch: {len(candidate_result)}/{expected}")

    summary = candidate_result.groupby("review").agg(
        rows=("security_id", "size"),
        resolved=("foreign_ownership", lambda s: int(s.notna().sum())),
        unresolved=("foreign_ownership", lambda s: int(s.isna().sum())),
        newer_pit=("canonical_row_action", lambda s: int(s.eq("USE_NEWER_PIT").sum())),
        new_keys=("canonical_row_action", lambda s: int(s.eq("ADD_NEW_KEY").sum())),
        protected=("canonical_row_action", lambda s: int(s.eq("PRESERVE_PROTECTED").sum())),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\n--- HISTORICAL FOREIGN OWNERSHIP REBUILD ---")
    print(summary.to_string())

    print("\nCandidate keys:", expected)
    print("Candidate FO key coverage:", len(candidate_result))
    print("Candidate FO resolved:", int(candidate_result["foreign_ownership"].notna().sum()))
    print("New candidate keys added:", int(actions.eq("ADD_NEW_KEY").sum()))
    print("Existing rows upgraded to later PIT filing:", int(actions.eq("USE_NEWER_PIT").sum()))
    print("Protected repaired rows preserved:", int(actions.eq("PRESERVE_PROTECTED").sum()))
    print("Old resolved rows preserved:", int(actions.eq("PRESERVE_OLD_RESOLVED").sum()))
    print("Network fetch failures:", len(failures))
    print("Price-Cutoff lookahead:", int(lookahead.sum()))
    print("Canonical duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("Canonical rows:", len(out))
    print("Raw filing cache:", CACHE_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()