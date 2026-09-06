from pathlib import Path

import numpy as np
import pandas as pd


CHANGES_PATH = Path("data/raw/msci/msci_india_standard_changes.csv")
INV_PATH = Path("data/processed/investability_universe.parquet")
SECURITY_MASTER_PATH = Path("data/processed/security_master.parquet")
HISTORICAL_MASTER_PATH = Path("data/processed/historical_security_master.parquet")
OUT_PATH = Path("data/processed/review_labels.parquet")

BURN_IN_REVIEW = "2023-05"

MSCI_NAME_TO_TICKER = {
    "HINDUSTAN AERONAUTICS": "HAL",
    "MAX HEALTHCARE INSTITUTE": "MAXHEALTH",
    "SONA BLW PRECISION": "SONACOMS",
    "ADANI TOTAL GAS": "ATGL",
    "ADANI TRANSMISSION": "ADANIENSOL",
    "INDUS TOWERS": "INDUSTOWER",

    "ASHOK LEYLAND": "ASHOKLEY",
    "ASTRAL": "ASTRAL",
    "CUMMINS INDIA KIRLOSKAR": "CUMMINSIND",
    "HDFC ASSET MANAGEMENT": "HDFCAMC",
    "IDFC FIRST BANK": "IDFCFIRSTB",
    "POWER FINANCE CORP": "PFC",
    "REC": "RECLTD",
    "SUPREME INDUSTRIES": "SUPREMEIND",
    "ACC": "ACC",

    "APL APOLLO TUBES": "APLAPOLLO",
    "INDUSIND BANK": "INDUSINDBK",
    "MACROTECH DEVELOPERS": "LODHA",
    "ONE 97 COMMUNICATIONS": "PAYTM",
    "PERSISTENT SYSTEMS": "PERSISTENT",
    "POLYCAB INDIA": "POLYCAB",
    "SUZLON ENERGY LIMITED": "SUZLON",
    "TATA COMMUNICATIONS": "TATACOMM",
    "TATA MOTORS A": "TATAMTRDVR",

    "BHARAT HEAVY ELECTRICALS": "BHEL",
    "GMR AIRPORTS INFRA": "GMRAIRPORT",
    "NMDC": "NMDC",
    "PUNJAB NATL BANK": "PNB",
    "UNION BANK OF INDIA": "UNIONBANK",

    "BOSCH": "BOSCHLTD",
    "CANARA BANK": "CANBK",
    "JINDAL STAINLESS": "JSL",
    "JSW ENERGY": "JSWENERGY",
    "MANKIND PHARMA": "MANKIND",
    "NHPC": "NHPC",
    "PB FINTECH": "POLICYBZR",
    "PHOENIX MILLS": "PHOENIXLTD",
    "SOLAR INDUSTRIES INDIA": "SOLARINDS",
    "SUNDARAM FINANCE": "SUNDARMFIN",
    "THERMAX": "THERMAX",
    "TORRENT POWER": "TORNTPOWER",
    "BERGER PAINTS INDIA": "BERGEPAINT",
    "INDRAPRASTHA GAS": "IGL",

    "DIXON TECHNOLOGIES INDIA": "DIXON",
    "OIL INDIA": "OIL",
    "ORACLE FINL SVCS SOFTW": "OFSS",
    "PRESTIGE ESTATES PROJECT": "PRESTIGE",
    "RAIL VIKAS NIGAM": "RVNL",
    "VODAFONE IDEA": "IDEA",
    "ZYDUS LIFESCIENCES": "ZYDUSLIFE",
    "BANDHAN BANK": "BANDHANBNK",

    "ALKEM LABORATORIES": "ALKEM",
    "BOMBAY STOCK EXCHANGE": "BSE",
    "KALYAN JEWELLERS INDIA": "KALYANKJIL",
    "OBEROI REALTY": "OBEROIRLTY",
    "VOLTAS": "VOLTAS",

    "HYUNDAI MOTOR INDIA": "HYUNDAI",
    "ADANI GREEN ENERGY": "ADANIGREEN",

    "COROMANDEL INTERNATIONAL": "COROMANDEL",
    "FSN ECOMMERCE VENTURES": "NYKAA",

    "HITACHI ENERGY INDIA": "POWERINDIA",
    "SWIGGY": "SWIGGY",
    "VISHAL MEGA MART": "VMM",
    "WAAREE ENERGIES": "WAAREEENER",

    "FORTIS HEALTHCARE": "FORTIS",
    "GE VERNOVA T&D INDIA": "GVT&D",
    "SIEMENS ENERGY INDIA": "ENRIN",
    "CONTAINER CORP OF INDIA": "CONCOR",
    "TATA ELXSI": "TATAELXSI",

    "ADITYA BIRLA CAPITAL": "ABCAPITAL",
    "L AND T FINANCE": "LTF",
    "INDIAN RAIL CATER & TOUR": "IRCTC",

    "ADANI ENERGY SOLUTIONS": "ADANIENSOL",
    "FEDERAL BANK": "FEDERALBNK",
    "INDIAN BANK": "INDIANB",
    "MULTI CMDTY EXCH INDIA": "MCX",
    "NATIONAL ALUMINIUM CO": "NATIONALUM",
    "JUBILANT FOODWORKS": "JUBLFOOD",
}


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing: {missing}")


def load_master(inv):
    frames = [inv[["security_id", "nse_symbol"]]]

    for path in [SECURITY_MASTER_PATH, HISTORICAL_MASTER_PATH]:
        if path.exists():
            x = pd.read_parquet(path)

            if {"security_id", "nse_symbol"}.issubset(x.columns):
                frames.append(x[["security_id", "nse_symbol"]])

    x = pd.concat(frames, ignore_index=True).dropna()
    x["security_id"] = x["security_id"].astype("string")
    x["ticker"] = x["nse_symbol"].astype("string").str.upper().str.strip()

    return x.drop_duplicates()


def map_change(row, inv, master):
    ticker = row["mapped_ticker"]
    review = row["review"]

    # Prefer exact security identity present in this review.
    ids = (
        inv.loc[
            inv["review"].eq(review)
            & inv["nse_symbol"].astype("string").str.upper().str.strip().eq(ticker),
            "security_id",
        ]
        .dropna()
        .unique()
    )

    if len(ids) == 1:
        return ids[0]

    if len(ids) > 1:
        raise RuntimeError(
            f"{review} {row['msci_name']}: ambiguous review ticker {ticker}: {list(ids)}"
        )

    # Fallback only if ticker has one permanent ID across known masters.
    ids = master.loc[master["ticker"].eq(ticker), "security_id"].dropna().unique()

    if len(ids) == 1:
        return ids[0]

    return pd.NA


def main():
    changes = pd.read_csv(CHANGES_PATH)
    inv = pd.read_parquet(INV_PATH)

    require(changes, ["review", "action", "msci_name"], "changes")
    require(
        inv,
        ["review", "security_id", "nse_symbol", "company_name", "company_key"],
        "investability_universe",
    )

    changes["review"] = changes["review"].astype(str).str.strip()
    changes["action"] = changes["action"].astype(str).str.upper().str.strip()
    changes["msci_name"] = changes["msci_name"].astype(str).str.strip()

    inv["review"] = inv["review"].astype(str)
    inv["security_id"] = inv["security_id"].astype("string")
    inv["nse_symbol"] = inv["nse_symbol"].astype("string")

    bad_actions = set(changes["action"]) - {"ADD", "DELETE"}

    if bad_actions:
        raise RuntimeError(f"Invalid actions: {bad_actions}")

    missing_names = sorted(set(changes["msci_name"]) - set(MSCI_NAME_TO_TICKER))

    if missing_names:
        raise RuntimeError(f"Missing explicit MSCI mappings: {missing_names}")

    changes["mapped_ticker"] = changes["msci_name"].map(MSCI_NAME_TO_TICKER)

    master = load_master(inv)
    changes["security_id"] = changes.apply(
        lambda r: map_change(r, inv, master), axis=1
    ).astype("string")

    unmapped = changes[changes["security_id"].isna()]

    if not unmapped.empty:
        print("\n--- UNMAPPED ACTUAL CHANGES ---")
        print(
            unmapped[
                ["review", "action", "msci_name", "mapped_ticker"]
            ].to_string(index=False)
        )
        raise RuntimeError("Resolve actual-change mappings before continuing.")

    if changes.duplicated(["review", "security_id"]).any():
        bad = changes[
            changes.duplicated(["review", "security_id"], keep=False)
        ]
        print(bad.to_string(index=False))
        raise RuntimeError("Multiple actual actions for one review/security.")

    base = (
        inv[
            [
                "review",
                "security_id",
                "nse_symbol",
                "company_name",
                "company_key",
            ]
        ]
        .drop_duplicates(["review", "security_id"])
        .copy()
    )

    if base.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate investability review/security keys.")

    base_keys = pd.MultiIndex.from_frame(base[["review", "security_id"]])
    change_keys = pd.MultiIndex.from_frame(changes[["review", "security_id"]])

    missing_from_inv = changes.loc[
        ~change_keys.isin(base_keys)
    ].copy()

    if not missing_from_inv.empty:
        extra = missing_from_inv[
            ["review", "security_id", "mapped_ticker", "msci_name"]
        ].rename(
            columns={
                "mapped_ticker": "nse_symbol",
                "msci_name": "company_name",
            }
        )

        extra["company_key"] = pd.NA
        base = pd.concat([base, extra[base.columns]], ignore_index=True)

    change_labels = changes[
        ["review", "security_id", "action", "msci_name", "mapped_ticker"]
    ].rename(columns={"msci_name": "actual_change_name"})

    out = base.merge(
        change_labels,
        on=["review", "security_id"],
        how="left",
        validate="one_to_one",
    )

    out["actual_add"] = out["action"].eq("ADD")
    out["actual_delete"] = out["action"].eq("DELETE")
    out["actual_unchanged"] = ~(out["actual_add"] | out["actual_delete"])

    out["actual_action"] = "UNCHANGED"
    out.loc[out["actual_add"], "actual_action"] = "ADD"
    out.loc[out["actual_delete"], "actual_action"] = "DELETE"

    actual_only_keys = pd.MultiIndex.from_frame(
        missing_from_inv[["review", "security_id"]]
    )

    out_keys = pd.MultiIndex.from_frame(out[["review", "security_id"]])

    out["label_universe_source"] = np.where(
        out_keys.isin(actual_only_keys),
        "ACTUAL_CHANGE_ONLY",
        "INVESTABILITY_UNIVERSE",
    )

    out["label_source"] = np.where(
        out["actual_unchanged"],
        "MSCI_ST_PUBLIC_LIST_ABSENCE",
        "MSCI_ST_PUBLIC_LIST_CHANGE",
    )

    out["backtest_scored_review"] = ~out["review"].eq(BURN_IN_REVIEW)

    out = out[
        [
            "review",
            "security_id",
            "nse_symbol",
            "company_name",
            "company_key",
            "actual_add",
            "actual_delete",
            "actual_unchanged",
            "actual_action",
            "actual_change_name",
            "mapped_ticker",
            "label_source",
            "label_universe_source",
            "backtest_scored_review",
        ]
    ].sort_values(["review", "security_id"])

    if out.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate review-label keys.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    summary = (
        out.groupby("review", as_index=False)
        .agg(
            rows=("security_id", "size"),
            actual_add=("actual_add", "sum"),
            actual_delete=("actual_delete", "sum"),
            actual_unchanged=("actual_unchanged", "sum"),
        )
    )

    print("\n--- REVIEW LABELS ---")
    print(summary.to_string(index=False))
    print("\nRaw announced changes:", len(changes))
    print("Unmapped changes:", int(changes["security_id"].isna().sum()))
    print("Actual changes outside investability universe:", len(missing_from_inv))
    print("Duplicates:", int(out.duplicated(["review", "security_id"]).sum()))
    print("May-2023 scored:", False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()