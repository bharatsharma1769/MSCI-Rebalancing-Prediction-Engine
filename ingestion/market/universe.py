import pandas as pd

def load_security_master(path: str = "data/processed/security_master.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def get_equity_universe(as_of_date: str,master: pd.DataFrame | None = None) -> pd.DataFrame:

    if master is None:
        master = load_security_master()

    df = master.copy()

    as_of_date = pd.Timestamp(as_of_date)

    df = df[df["is_eligible_equity"]]

    df = df[
        df["listing_date"].isna()
        | (df["listing_date"] <= as_of_date)
    ]

    df = df[
        df["delisting_date"].isna()
        | (df["delisting_date"] > as_of_date)
    ]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    universe = get_equity_universe("2026-05-29")

    print(universe.head())
    print(f"\nUniverse size: {len(universe)}")