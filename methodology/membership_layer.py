from pathlib import Path
import pandas as pd

BASE_PATH = Path("data/raw/msci/msci_india_standard_base_feb2023.csv")
CHANGES_PATH = Path("data/raw/msci/msci_india_standard_changes.csv")
MASTER_PATH = Path("data/processed/security_master.parquet")
HIST_PATH = Path("data/processed/historical_security_master.parquet")
OUT_PATH = Path("data/processed/msci_standard_membership.parquet")

REVIEWS = [
    "2023-05", "2023-08", "2023-11", "2024-02", "2024-05", "2024-08",
    "2024-11", "2025-02", "2025-05", "2025-08", "2025-11", "2026-02", "2026-05"
]

BASE_OVERRIDES = {
    "ADANITRANS": "SEC000044",
    "HDFC": "HIST_HDFC",
    "LTIM": "SEC001200",
    "TATAMOTORS": "SEC002077",
    "ZOMATO": "SEC000628",
    "TATAMTRDVR": "HIST_TATAMTRDVR"
}

ALIASES = {
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
    "JUBILANT FOODWORKS": "JUBLFOOD"
}

# Only persistent membership changes between reviews.
# Temporary detached spin-off securities are deliberately excluded.
CORPORATE_EVENTS = {
    "2023-08": [
        ("DELETE", "HDFC"),
        
    ],
    "2023-11": [
        ("ADD", "JIOFIN"),
    ],
    "2024-11": [
        ("DELETE", "TATAMTRDVR"),
    ],
    "2026-02": [
        ("ADD", "TMCV"),
    ],
}


def load_ticker_map():
    sm = pd.concat([pd.read_parquet(MASTER_PATH), pd.read_parquet(HIST_PATH)], ignore_index=True)
    x = sm[["security_id", "nse_symbol"]].dropna().drop_duplicates()
    x["ticker"] = x["nse_symbol"].astype(str).str.upper().str.strip()

    counts = x.groupby("ticker")["security_id"].nunique()
    ambiguous = set(counts[counts > 1].index)
    x = x[~x["ticker"].isin(ambiguous)]

    return dict(zip(x["ticker"], x["security_id"]))


def ticker_to_sid(ticker, ticker_map):
    ticker = str(ticker).upper().strip()
    return BASE_OVERRIDES.get(ticker, ticker_map.get(ticker))


def main():
    ticker_map = load_ticker_map()
    base = pd.read_csv(BASE_PATH)
    changes = pd.read_csv(CHANGES_PATH)

    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()

    if len(base) != 114:
        raise ValueError(f"Feb-2023 base should have 114 rows, got {len(base)}")

    required = {"ACC", "SHRIRAMFIN", "YESBANK"}
    missing = required - set(base["ticker"])
    if missing:
        raise ValueError(f"Feb-2023 base missing verified constituents: {missing}")

    base["security_id"] = base["ticker"].apply(lambda x: ticker_to_sid(x, ticker_map))

    unmapped_base = base[base["security_id"].isna()]
    if len(unmapped_base):
        print("\nUNMAPPED BASE:")
        print(unmapped_base[["ticker", "company_name"]].to_string(index=False))
        raise ValueError("Resolve base mappings")

    if base["security_id"].duplicated().any():
        print(base[base["security_id"].duplicated(False)][["ticker", "company_name", "security_id"]])
        raise ValueError("Duplicate security_id in base")

    changes["review"] = changes["review"].astype(str).str.strip()
    changes["action"] = changes["action"].astype(str).str.upper().str.strip()
    changes["msci_name"] = changes["msci_name"].astype(str).str.strip()

    if set(changes["action"]) - {"ADD", "DELETE"}:
        raise ValueError("Invalid action in changes file")

    bad_reviews = set(changes["review"]) - set(REVIEWS)
    if bad_reviews:
        raise ValueError(f"Unexpected reviews: {bad_reviews}")

    changes["ticker"] = changes["msci_name"].map(ALIASES)
    unmapped_names = changes[changes["ticker"].isna()]

    if len(unmapped_names):
        print("\nMSCI NAMES WITHOUT ALIAS:")
        print(unmapped_names[["review", "action", "msci_name"]].to_string(index=False))
        raise ValueError("Add explicit aliases above")

    changes["security_id"] = changes["ticker"].apply(lambda x: ticker_to_sid(x, ticker_map))
    unmapped_changes = changes[changes["security_id"].isna()]

    if len(unmapped_changes):
        print("\nALIASES NOT FOUND IN SECURITY MASTER:")
        print(unmapped_changes[["review", "action", "msci_name", "ticker"]].to_string(index=False))
        raise ValueError("Resolve change ticker mappings")

    membership = set(base["security_id"])
    rows = []
    event_audit = []

    for review in REVIEWS:
        for action, ticker in CORPORATE_EVENTS.get(review, []):
            sid = ticker_to_sid(ticker, ticker_map)

            if sid is None:
                raise ValueError(f"{review}: corporate event ticker {ticker} not mapped")

            if action == "ADD":
                if sid in membership:
                    raise ValueError(f"{review}: corporate ADD {ticker} already present")
                membership.add(sid)
            else:
                if sid not in membership:
                    raise ValueError(f"{review}: corporate DELETE {ticker} not present")
                membership.remove(sid)

            event_audit.append({"review": review, "action": action, "ticker": ticker, "security_id": sid})

        pre_count = len(membership)

        rows.extend({
            "review": review,
            "security_id": sid,
            "was_standard_constituent": True
        } for sid in membership)

        r = changes[changes["review"] == review]
        adds = set(r.loc[r["action"] == "ADD", "security_id"])
        deletes = set(r.loc[r["action"] == "DELETE", "security_id"])

        existing_adds = adds & membership
        missing_deletes = deletes - membership

        if existing_adds or missing_deletes:
            if existing_adds:
                print(f"\n{review} ADD already present:")
                print(r[r["security_id"].isin(existing_adds)][["msci_name", "security_id"]].to_string(index=False))
            if missing_deletes:
                print(f"\n{review} DELETE not present:")
                print(r[r["security_id"].isin(missing_deletes)][["msci_name", "security_id"]].to_string(index=False))
            raise ValueError(f"Membership chain inconsistent at {review}")

        membership -= deletes
        membership |= adds

        print(f"{review}: pre={pre_count}, adds={len(adds)}, deletes={len(deletes)}, post={len(membership)}")

    out = pd.DataFrame(rows)

    if out.duplicated(["review", "security_id"]).any():
        raise ValueError("Duplicate review/security_id rows")

    if out["review"].nunique() != len(REVIEWS):
        raise ValueError("Missing review snapshots")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    if event_audit:
        pd.DataFrame(event_audit).to_csv(
            "data/processed/msci_membership_corporate_events.csv", index=False
        )

    print("\nSaved:", OUT_PATH)
    print("\nPre-review counts:")
    print(out.groupby("review")["security_id"].nunique().reindex(REVIEWS).to_string())


if __name__ == "__main__":
    main()