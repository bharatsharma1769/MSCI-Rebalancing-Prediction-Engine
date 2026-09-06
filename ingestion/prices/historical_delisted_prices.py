import io
import pickle
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


COVERAGE_PATH = Path(
    "data/raw/prices/"
    "historical_price_coverage.csv"
)

MASTER_PATH = Path(
    "data/processed/"
    "historical_security_master.parquet"
)

OUTPUT_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_prices.parquet"
)

CHECKPOINT_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_prices_checkpoint.pkl"
)

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2026-06-01")
CHECKPOINT_EVERY = 100


def get_missing_symbols():
    coverage = pd.read_csv(
        COVERAGE_PATH
    )

    missing = coverage[
        coverage["price_rows"] == 0
    ].copy()

    return set(
        missing["nse_symbol"]
        .dropna()
        .astype(str)
    )


def create_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        )
    })

    return session


def download_bhavcopy(
    session,
    date,
):
    date = pd.Timestamp(date)

    year = date.strftime("%Y")
    month = date.strftime("%b").upper()
    day = date.strftime("%d")
    mon = date.strftime("%b").upper()

    filename = (
        f"cm{day}{mon}{year}"
        f"bhav.csv.zip"
    )

    url = (
        "https://nsearchives.nseindia.com/"
        "content/historical/EQUITIES/"
        f"{year}/{month}/{filename}"
    )

    for attempt in range(3):
        try:
            r = session.get(
                url,
                timeout=20,
            )

            if r.status_code == 404:
                return None

            if r.status_code != 200:
                time.sleep(
                    attempt + 1
                )
                continue

            with zipfile.ZipFile(
                io.BytesIO(r.content)
            ) as z:
                csv_name = z.namelist()[0]

                with z.open(
                    csv_name
                ) as f:
                    return pd.read_csv(f)

        except Exception:
            time.sleep(
                attempt + 1
            )

    return None


def clean_bhavcopy(
    df,
    date,
    targets,
    symbol_to_id,
):
    if df is None:
        return pd.DataFrame()

    df.columns = [
        str(c)
        .strip()
        .upper()
        for c in df.columns
    ]

    if "SYMBOL" not in df.columns:
        return pd.DataFrame()

    df["SYMBOL"] = (
        df["SYMBOL"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["SYMBOL"].isin(
            targets
        )
    ].copy()

    if "SERIES" in df.columns:
        df = df[
            df["SERIES"].eq("EQ")
        ].copy()

    if df.empty:
        return df

    df = df.rename(
        columns={
            "SYMBOL":
                "ticker_symbol",
            "OPEN":
                "open",
            "HIGH":
                "high",
            "LOW":
                "low",
            "CLOSE":
                "close",
            "TOTTRDQTY":
                "volume",
        }
    )

    df["date"] = pd.Timestamp(
        date
    )

    df["ticker"] = (
        df["ticker_symbol"]
        + ".NS"
    )

    df["security_id"] = (
        df["ticker_symbol"]
        .map(symbol_to_id)
    )

    df["adj_close"] = pd.NA
    df["price_source"] = (
        "NSE_BHAVCOPY"
    )

    keep = [
        "date",
        "ticker",
        "adj_close",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "security_id",
        "price_source",
    ]

    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep]


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return [], None

    with open(
        CHECKPOINT_PATH,
        "rb",
    ) as f:
        state = pickle.load(f)

    print(
        "Checkpoint loaded through:",
        state.get(
            "processed_through"
        ),
    )

    return (
        state.get(
            "results",
            [],
        ),
        state.get(
            "processed_through"
        ),
    )


def save_checkpoint(
    results,
    processed_through,
):
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHECKPOINT_PATH,
        "wb",
    ) as f:
        pickle.dump(
            {
                "results":
                    results,
                "processed_through":
                    processed_through,
            },
            f,
        )

    print(
        "Checkpoint saved through:",
        processed_through,
    )


def main():
    targets = get_missing_symbols()

    master = pd.read_parquet(
        MASTER_PATH
    )

    symbol_to_id = dict(
        zip(
            master["nse_symbol"],
            master["security_id"],
        )
    )

    results, processed_through = (
        load_checkpoint()
    )

    session = create_session()

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="B",
    )

    if processed_through is not None:
        dates = dates[
            dates
            > pd.Timestamp(
                processed_through
            )
        ]

    print(
        "Missing delisted symbols:",
        len(targets),
    )

    for i, date in enumerate(
        dates,
        start=1,
    ):
        df = download_bhavcopy(
            session,
            date,
        )

        found = clean_bhavcopy(
            df,
            date,
            targets,
            symbol_to_id,
        )

        if not found.empty:
            results.append(found)

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                results,
                date,
            )

            print(
                f"Processed {i} dates"
            )

        time.sleep(0.03)

    if not results:
        raise RuntimeError(
            "No historical prices found."
        )

    prices = pd.concat(
        results,
        ignore_index=True,
    )

    prices["date"] = pd.to_datetime(
        prices["date"]
    ).astype("datetime64[ns]")

    prices = (
        prices.drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .sort_values(
            ["date", "security_id"]
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    coverage = (
        prices.groupby(
            "security_id"
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            rows=("date", "size"),
        )
        .reset_index()
    )

    coverage = coverage.merge(
        master[
            [
                "security_id",
                "nse_symbol",
                "delisted_date",
            ]
        ],
        on="security_id",
        how="left",
    )

    print(
        "\nRecovered securities:",
        len(coverage),
    )

    print(
        coverage[
            [
                "nse_symbol",
                "first_date",
                "last_date",
                "rows",
                "delisted_date",
            ]
        ].to_string(
            index=False
        )
    )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        "\nCheckpoint removed."
    )


if __name__ == "__main__":
    main()
