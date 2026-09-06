import pandas as pd

def load_raw_nse_universe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def clean_security_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "symbol": "nse_symbol",
        "name_of_company": "company_name",
        "isin_number": "isin",
        "date_of_listing": "listing_date",
    }

    df = df.rename(columns=rename_map)

    required = [
        "nse_symbol",
        "company_name",
        "isin",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["nse_symbol"] = df["nse_symbol"].astype(str).str.strip()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip()

    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(
            df["listing_date"],
            format="%d-%b-%y",
            errors="coerce"
        )
    else:
        df["listing_date"] = pd.NaT

    return df


def filter_ordinary_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "series" in df.columns:
        df = df[df["series"].eq("EQ")]

    return df


def build_security_master(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df = clean_security_master(df)
    df = filter_ordinary_shares(df)

    df = df.sort_values("nse_symbol").reset_index(drop=True)

    df["security_id"] = [
        f"SEC{i:06d}"
        for i in range(1, len(df) + 1)
    ]

    df["company_id"] = df["security_id"]
    df["exchange"] = "NSE"
    df["security_type"] = "Ordinary Share"
    df["country"] = "India"
    df["currency"] = "INR"
    df["is_active"] = True
    df["is_eligible_equity"] = True
    df["delisting_date"] = pd.NaT

    columns = [
        "security_id",
        "company_id",
        "company_name",
        "nse_symbol",
        "isin",
        "exchange",
        "security_type",
        "listing_date",
        "delisting_date",
        "country",
        "currency",
        "is_active",
        "is_eligible_equity",
    ]

    df = df[columns]
    df = df.drop_duplicates(subset="security_id")
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw = load_raw_nse_universe(
        "data/raw/universe/nse_equity_list.csv"
    )

    master = build_security_master(raw)

    master.to_parquet(
        "data/processed/security_master.parquet",
        index=False
    )
    print(master.head())
    print(f"Saved {len(master)} securities")