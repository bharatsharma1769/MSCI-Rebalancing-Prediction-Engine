import time
from pathlib import Path

import pandas as pd
import yfinance as yf


MASTER_PATH = Path("data/processed/security_master.parquet")
OUTPUT_PATH = Path("data/raw/prices/prices.parquet")
CHECKPOINT_PATH = Path("data/raw/prices/prices_checkpoint.parquet")

START_DATE = "2021-01-01"
BATCH_SIZE = 100


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame(), set()

    x = pd.read_parquet(CHECKPOINT_PATH)

    processed = set(
        x["security_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"Checkpoint loaded: "
        f"{len(processed)} securities"
    )

    return x, processed


def extract_ticker_frame(raw, ticker):
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)

        if ticker in level0:
            x = raw[ticker].copy()
        elif ticker in level1:
            x = raw.xs(
                ticker,
                axis=1,
                level=1,
            ).copy()
        else:
            return pd.DataFrame()
    else:
        x = raw.copy()

    x = x.reset_index()

    x.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        for c in x.columns
    ]

    if "date" not in x.columns:
        return pd.DataFrame()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.tz_localize(None)

    return x


def download_batch(batch):
    tickers = [
        f"{symbol}.NS"
        for symbol in batch["nse_symbol"]
    ]

    raw = yf.download(
        tickers,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    rows = []

    ticker_to_id = dict(
        zip(
            tickers,
            batch["security_id"],
        )
    )

    for ticker in tickers:
        x = extract_ticker_frame(
            raw,
            ticker,
        )

        if x.empty:
            continue

        x["ticker"] = ticker
        x["security_id"] = (
            ticker_to_id[ticker]
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
        ]

        for col in keep:
            if col not in x.columns:
                x[col] = pd.NA

        x["price_source"] = "YAHOO"

        rows.append(
            x[
                keep
                + ["price_source"]
            ]
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


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


def main():
    master = pd.read_parquet(
        MASTER_PATH
    )[
        [
            "security_id",
            "nse_symbol",
        ]
    ].copy()

    master = master[
        master["nse_symbol"].notna()
    ].copy()

    master["security_id"] = (
        master["security_id"]
        .astype(str)
    )

    checkpoint, processed = (
        load_checkpoint()
    )

    parts = []

    if not checkpoint.empty:
        parts.append(checkpoint)

    todo = master[
        ~master["security_id"].isin(
            processed
        )
    ].copy()

    print(
        "Securities to download:",
        len(todo),
    )

    for start in range(
        0,
        len(todo),
        BATCH_SIZE,
    ):
        batch = todo.iloc[
            start:start + BATCH_SIZE
        ].copy()

        print(
            f"Batch "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, len(todo))}"
        )

        x = pd.DataFrame()
        last_error = None

        for attempt in range(3):
            try:
                x = download_batch(batch)
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(
                    f"Batch attempt {attempt + 1} failed:",
                    str(e)[:300],
                )
                time.sleep(
                    2 * (attempt + 1)
                )

        if last_error is not None:
            print(
                "Batch failed after retries."
            )

        if not x.empty:
            parts.append(x)

        save_checkpoint(parts)
        time.sleep(0.5)

    if not parts:
        raise RuntimeError(
            "No prices downloaded."
        )

    prices = pd.concat(
        parts,
        ignore_index=True,
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
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

    print(
        "\nSaved rows:",
        len(prices),
    )

    print(
        "Securities:",
        prices["security_id"].nunique(),
    )

    missing = sorted(
        set(master["security_id"])
        - set(
            prices["security_id"]
            .dropna()
            .astype(str)
        )
    )

    print(
        "Missing securities:",
        len(missing),
    )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        "Checkpoint removed."
    )


if __name__ == "__main__":
    main()
