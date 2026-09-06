from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

CAND_PATH = Path("data/processed/foreign_room_candidates.parquet")
FO_PATH = Path("data/processed/historical_foreign_ownership.parquet")
FOL_PATH = Path("data/processed/historical_fol.parquet")
MASTER_PATH = Path("data/processed/historical_security_master.parquet")
CORE_PATH = Path("ingestion/foreign ownership/historical_foreign_ownership.py")

TARGETS = [
    ("2023-11", "SEC002116", "TVSSCS", "2023-10-18"),
    ("2024-02", "SEC000963", "IREDA", "2024-01-18"),
    ("2024-02", "SEC002021", "TATATECH", "2024-01-18"),
    ("2024-02", "SEC002114", "TVSHLTD", "2024-01-18"),
    ("2024-05", "SEC001040", "JUNIPER", "2024-04-17"),
    ("2024-08", "SEC000231", "AWFIS", "2024-07-18"),
    ("2024-08", "SEC000737", "GODIGIT", "2024-07-18"),
]

BOUND = 0.24
BOUND_BASIS = "NDI_2020_CONSERVATIVE_FPI_FLOOR_24"


def load_core():
    spec = importlib.util.spec_from_file_location("fo_core", CORE_PATH)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    needed = ["create_session", "get_filings", "extract_foreign_ownership", "parse_date"]
    missing = [x for x in needed if not hasattr(core, x)]
    if missing: raise RuntimeError(f"FO parser missing functions: {missing}")
    return core


def normalize(df):
    df = df.copy()
    df["review"] = df["review"].astype(str)
    df["security_id"] = df["security_id"].astype("string")
    return df


def atomic_save(df, path):
    tmp = path.with_name(path.stem + "_tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def append_aligned(old, new):
    for c in old.columns:
        if c not in new.columns: new[c] = pd.NA
    for c in new.columns:
        if c not in old.columns: old[c] = pd.NA
    return pd.concat([old, new[old.columns]], ignore_index=True)


def build_fo_rows(core, missing, master):
    session = core.create_session()
    master = master.copy()
    master["security_id"] = master["security_id"].astype("string")

    if "isin" not in master.columns: raise RuntimeError("Historical master missing isin.")

    master = master[["security_id", "isin"]].drop_duplicates("security_id", keep="last").set_index("security_id")
    rows = []

    for _, r in missing.iterrows():
        sid, symbol, cutoff = r["security_id"], r["nse_symbol"], r["foreign_ownership_cutoff_date"]
        isin = master.at[sid, "isin"] if sid in master.index else pd.NA

        base = {
            "review": r["review"],
            "security_id": sid,
            "nse_symbol": symbol,
            "source_isin": isin,
            "foreign_ownership_cutoff_date": cutoff,
            "foreign_ownership_cutoff_is_proxy": True,
            "foreign_ownership_cutoff_method": "DETERMINISTIC_PRICE_CUTOFF_PROXY",
        }

        try:
            filings = core.get_filings(symbol, session)
        except Exception as e:
            print(sid, symbol, "filing lookup failed:", str(e)[:150])
            filings = []

        all_xbrls, eligible = [], []

        for f in filings:
            url = f.get("xbrl")
            if not isinstance(url, str) or not url.startswith("http"): continue

            all_xbrls.append(url)
            report_date = core.parse_date(f.get("date"))
            submission_date = core.parse_date(f.get("submissionDate"))

            if pd.notna(submission_date) and submission_date <= cutoff:
                eligible.append({
                    "report_date": report_date,
                    "submission_date": submission_date,
                    "available_date": submission_date,
                    "xbrl_url": url,
                })

        if not all_xbrls:
            rows.append({
                **base, "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                "foreign_ownership": np.nan, "foreign_context_count": np.nan,
                "foreign_members": pd.NA, "xbrl_url": pd.NA, "source": pd.NA,
                "mapping_method": "security_id_to_nse_symbol",
                "data_quality_flag": "NO_FILINGS_FOUND",
            })
            continue

        if not eligible:
            rows.append({
                **base, "report_date": pd.NaT, "submission_date": pd.NaT, "available_date": pd.NaT,
                "foreign_ownership": np.nan, "foreign_context_count": np.nan,
                "foreign_members": pd.NA, "xbrl_url": pd.NA, "source": pd.NA,
                "mapping_method": "security_id_to_nse_symbol",
                "data_quality_flag": "NO_ELIGIBLE_FILING",
            })
            continue

        chosen = pd.DataFrame(eligible).sort_values(["report_date", "available_date"]).iloc[-1]

        try:
            fo, count, members = core.extract_foreign_ownership(chosen["xbrl_url"], session)
        except Exception as e:
            print(sid, symbol, "XBRL failed:", str(e)[:150])
            fo, count, members = np.nan, np.nan, pd.NA

        if pd.isna(fo): quality = "PIT_FILING_PARSE_MISSING"
        elif 0 <= fo <= 1: quality = "OK"
        else: quality = "PIT_FILING_INVALID_VALUE"

        rows.append({
            **base,
            "report_date": chosen["report_date"],
            "submission_date": chosen["submission_date"],
            "available_date": chosen["available_date"],
            "foreign_ownership": fo,
            "foreign_context_count": count,
            "foreign_members": members,
            "xbrl_url": chosen["xbrl_url"],
            "source": "NSE_XBRL",
            "mapping_method": "security_id_to_nse_symbol",
            "data_quality_flag": quality,
        })

    return pd.DataFrame(rows)


def patch_fol(fol, missing):
    if missing.empty: return fol, pd.DataFrame()

    required = {"historical_fol", "historical_fol_lower_bound", "historical_fol_lower_bound_basis"}
    absent = required - set(fol.columns)
    if absent: raise RuntimeError(f"Missing FOL columns: {sorted(absent)}")

    template = fol[
        fol["historical_fol"].isna()
        & fol["historical_fol_lower_bound"].eq(BOUND)
        & fol["historical_fol_lower_bound_basis"].eq(BOUND_BASIS)
    ]

    if template.empty: raise RuntimeError("No canonical 24% lower-bound template found.")

    template = template.iloc[0]
    evidence_cols = [c for c in fol.columns if c == "historical_fol" or c.startswith("historical_fol_")]
    alias_cols = [c for c in ["resolution_type", "status", "provenance", "source", "effective_date", "available_date"] if c in fol.columns]

    rows = []

    for _, r in missing.iterrows():
        row = {c: pd.NA for c in fol.columns}

        for c in evidence_cols + alias_cols:
            row[c] = template[c]

        row["review"] = r["review"]
        row["security_id"] = r["security_id"]

        if "nse_symbol" in fol.columns: row["nse_symbol"] = r["nse_symbol"]
        if "foreign_ownership_cutoff_date" in fol.columns:
            row["foreign_ownership_cutoff_date"] = r["foreign_ownership_cutoff_date"]

        row["historical_fol"] = np.nan
        row["historical_fol_lower_bound"] = BOUND
        row["historical_fol_lower_bound_basis"] = BOUND_BASIS
        rows.append(row)

    new = pd.DataFrame(rows, columns=fol.columns)
    return pd.concat([fol, new], ignore_index=True), new


def main():
    targets = pd.DataFrame(TARGETS, columns=[
        "review", "security_id", "nse_symbol", "foreign_ownership_cutoff_date"
    ])
    targets["security_id"] = targets["security_id"].astype("string")
    targets["foreign_ownership_cutoff_date"] = pd.to_datetime(targets["foreign_ownership_cutoff_date"])

    cand = normalize(pd.read_parquet(CAND_PATH))
    fo = normalize(pd.read_parquet(FO_PATH))
    fol = normalize(pd.read_parquet(FOL_PATH))
    master = pd.read_parquet(MASTER_PATH)

    keys = targets[["review", "security_id"]]
    current = cand[["review", "security_id"]].drop_duplicates()

    if not keys.merge(current, on=["review", "security_id"], how="left", indicator=True)["_merge"].eq("both").all():
        raise RuntimeError("A hard-coded target is no longer a current candidate.")

    fo_keys = fo[["review", "security_id"]].drop_duplicates()
    fol_keys = fol[["review", "security_id"]].drop_duplicates()

    missing_fo = targets.merge(fo_keys, on=["review", "security_id"], how="left", indicator=True)
    missing_fo = missing_fo[missing_fo["_merge"].eq("left_only")].drop(columns="_merge")

    missing_fol = targets.merge(fol_keys, on=["review", "security_id"], how="left", indicator=True)
    missing_fol = missing_fol[missing_fol["_merge"].eq("left_only")].drop(columns="_merge")

    print("\n--- TARGETED FOREIGN-ROOM PATCH ---")
    print("Target keys:", len(targets))
    print("Missing FO keys:", len(missing_fo))
    print("Missing FOL keys:", len(missing_fol))

    if not missing_fo.empty:
        new_fo = build_fo_rows(load_core(), missing_fo, master)

        if len(new_fo) != len(missing_fo):
            raise RuntimeError(f"Expected {len(missing_fo)} FO rows, built {len(new_fo)}.")

        lookahead = (
            new_fo["available_date"].notna()
            & new_fo["available_date"].gt(new_fo["foreign_ownership_cutoff_date"])
        )
        if lookahead.any(): raise RuntimeError("New FO patch contains look-ahead.")

        fo = append_aligned(fo, new_fo)

        print("\n--- NEW FO ---")
        cols = [c for c in [
            "review", "security_id", "nse_symbol", "report_date", "available_date",
            "foreign_ownership", "data_quality_flag"
        ] if c in new_fo.columns]
        print(new_fo[cols].to_string(index=False))
    else:
        new_fo = pd.DataFrame()
        print("\nNo FO rows needed.")

    fol, new_fol = patch_fol(fol, missing_fol)

    if not new_fol.empty:
        print("\n--- NEW FOL ---")
        cols = [c for c in [
            "review", "security_id", "nse_symbol", "historical_fol",
            "historical_fol_lower_bound", "historical_fol_lower_bound_basis"
        ] if c in new_fol.columns]
        print(new_fol[cols].to_string(index=False))

    if fo.duplicated(["review", "security_id"]).any(): raise RuntimeError("FO duplicates after patch.")
    if fol.duplicated(["review", "security_id"]).any(): raise RuntimeError("FOL duplicates after patch.")

    atomic_save(fo, FO_PATH)
    atomic_save(fol, FOL_PATH)

    print("\n--- SAVED ---")
    print("FO rows:", len(fo), "| new:", len(new_fo))
    print("FOL rows:", len(fol), "| new:", len(new_fol))


if __name__ == "__main__":
    main()