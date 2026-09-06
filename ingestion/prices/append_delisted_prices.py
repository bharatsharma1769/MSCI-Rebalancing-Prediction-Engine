import time
from pathlib import Path

import pandas as pd
import yfinance as yf


COVERAGE_PATH = Path(
    "data/raw/prices/"
    "historical_price_coverage.csv"
)

MASTER_PATH = Path(
    "data/processed/"
    "historical_security_master.parquet"
)

NSE_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_prices.parquet"
)

YAHOO_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_yahoo.parquet"
)

ALL_HISTORICAL_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_all.parquet"
)

PRICES_PATH = Path(
    "data/raw/prices/prices.parquet"
)

CHECKPOINT_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_yahoo_checkpoint.parquet"
)

CHECKPOINT_EVERY = 5


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame(), set()

    x = pd.read_parquet(
        CHECKPOINT_PATH
    )

    processed = set(
        x["security_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        "Yahoo checkpoint loaded:",
        len(processed),
    )

    return x, processed


def fetch_yahoo(
    ticker,
    security_id,
):
    df = yf.download(
        ticker,
        start="2021-01-01",
        end="2026-06-01",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    df.columns = [
        str(col)
        .lower()
        .replace(" ", "_")
        for col in df.columns
    ]

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.tz_localize(None)

    df["ticker"] = ticker
    df["security_id"] = (
        security_id
    )
    df["price_source"] = "YAHOO"

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


def save_checkpoint(parts):
    if not parts:
        return

    x = pd.concat(
        parts,
        ignore_index=True,
    )

    x = (
        x.drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .sort_values(
            ["date", "security_id"]
        )
        .reset_index(drop=True)
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x.to_parquet(
        CHECKPOINT_PATH,
        index=False,
    )


def download_yahoo_prices():
    coverage = pd.read_csv(
        COVERAGE_PATH
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    yahoo_names = coverage[
        coverage["price_rows"] > 0
    ].copy()

    expected_ids = set(
        yahoo_names[
            "security_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if YAHOO_PATH.exists():
        existing = pd.read_parquet(
            YAHOO_PATH
        )

        existing_ids = set(
            existing[
                "security_id"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if expected_ids.issubset(
            existing_ids
        ):
            print(
                "Existing Yahoo historical file "
                "already complete:",
                len(existing_ids),
            )

            if CHECKPOINT_PATH.exists():
                CHECKPOINT_PATH.unlink()

            return existing

    yahoo_names = yahoo_names.merge(
        master[
            [
                "security_id",
                "nse_symbol",
            ]
        ],
        on=[
            "security_id",
            "nse_symbol",
        ],
        how="left",
    )

    checkpoint, processed = (
        load_checkpoint()
    )

    parts = []

    if not checkpoint.empty:
        parts.append(checkpoint)

    todo = yahoo_names[
        ~yahoo_names[
            "security_id"
        ].astype(str).isin(
            processed
        )
    ].copy()

    print(
        "Downloading Yahoo histories:",
        len(todo),
    )

    for i, row in enumerate(
        todo.itertuples(
            index=False
        ),
        start=1,
    ):
        ticker = (
            f"{row.nse_symbol}.NS"
        )

        print(
            "Fetching",
            ticker,
        )

        x = pd.DataFrame()
        last_error = None

        for attempt in range(3):
            try:
                x = fetch_yahoo(
                    ticker,
                    row.security_id,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(
                    f"{ticker} attempt "
                    f"{attempt + 1} failed:",
                    str(e)[:300],
                )
                time.sleep(
                    2 * (attempt + 1)
                )

        if last_error is not None:
            print(
                "Failed after retries:",
                ticker,
            )

        if not x.empty:
            parts.append(x)

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                parts
            )

        time.sleep(0.2)

    if not parts:
        raise RuntimeError(
            "No Yahoo historical "
            "prices downloaded."
        )

    yahoo = pd.concat(
        parts,
        ignore_index=True,
    )

    yahoo = (
        yahoo.drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .sort_values(
            ["date", "security_id"]
        )
        .reset_index(drop=True)
    )

    yahoo.to_parquet(
        YAHOO_PATH,
        index=False,
    )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        "Yahoo checkpoint removed."
    )

    return yahoo


def main():
    YAHOO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yahoo = download_yahoo_prices()

    nse = pd.read_parquet(
        NSE_PATH
    )

    for x in [
        yahoo,
        nse,
    ]:
        x["date"] = pd.to_datetime(
            x["date"]
        ).astype("datetime64[ns]")

    historical = pd.concat(
        [
            yahoo,
            nse,
        ],
        ignore_index=True,
    )

    historical = (
        historical.drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .sort_values(
            ["date", "security_id"]
        )
        .reset_index(drop=True)
    )

    coverage = pd.read_csv(
        COVERAGE_PATH
    )

    yahoo_expected_ids = set(
        coverage.loc[
            coverage[
                "price_rows"
            ] > 0,
            "security_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    nse_recovered_ids = set(
        nse[
            "security_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected_ids = (
        yahoo_expected_ids
        | nse_recovered_ids
    )

    found_ids = set(
        historical[
            "security_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_ids = sorted(
        expected_ids
        - found_ids
    )

    extra_ids = sorted(
        found_ids
        - expected_ids
    )

    print(
        "Combined historical securities:",
        len(found_ids),
        "/",
        len(expected_ids),
    )

    print(
        "Yahoo expected:",
        len(yahoo_expected_ids),
    )

    print(
        "NSE recovered:",
        len(nse_recovered_ids),
    )

    if missing_ids:
        print(
            "Missing expected historical securities:",
            len(missing_ids),
        )

        print(
            missing_ids[:100]
        )

        raise RuntimeError(
            "Historical price coverage is incomplete. "
            "Main prices.parquet was not modified."
        )

    if extra_ids:
        print(
            "Extra historical securities:",
            len(extra_ids),
        )


    historical.to_parquet(
        ALL_HISTORICAL_PATH,
        index=False,
    )

    prices = pd.read_parquet(
        PRICES_PATH
    )

    prices["date"] = pd.to_datetime(
        prices["date"]
    ).astype("datetime64[ns]")

    all_columns = list(
        dict.fromkeys(
            list(prices.columns)
            + list(
                historical.columns
            )
        )
    )

    for col in all_columns:
        if col not in prices.columns:
            prices[col] = pd.NA

        if col not in historical.columns:
            historical[col] = pd.NA

    prices = prices[
        all_columns
    ]

    historical = historical[
        all_columns
    ]

    combined = pd.concat(
        [
            prices,
            historical,
        ],
        ignore_index=True,
    )

    combined = (
        combined.drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .sort_values(
            ["date", "security_id"]
        )
        .reset_index(drop=True)
    )

    combined.to_parquet(
        PRICES_PATH,
        index=False,
    )

    print(
        "\nHistorical rows added:",
        len(historical),
    )

    print(
        "Final price rows:",
        len(combined),
    )

    print(
        "Final securities:",
        combined[
            "security_id"
        ].nunique(),
    )


if __name__ == "__main__":
    main()
