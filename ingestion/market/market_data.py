from pathlib import Path

import numpy as np
import pandas as pd


PRICES_PATH = Path(
    "data/raw/prices/prices.parquet"
)

SHARES_PATH = Path(
    "data/raw/shares/shares_nse.parquet"
)

OUT_PATH = Path(
    "data/processed/market_data.parquet"
)


def prepare_prices(prices):
    x = prices.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    x["security_id"] = (
        x["security_id"]
        .astype("string")
    )

    x["close"] = pd.to_numeric(
        x["close"],
        errors="coerce",
    )

    if "price_source" not in x.columns:
        x["price_source"] = "YAHOO"

    return (
        x.sort_values(
            ["date", "security_id"]
        )
        .drop_duplicates(
            ["date", "security_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def prepare_shares(shares):
    x = shares.copy()

    required = {
        "security_id",
        "available_date",
        "shares_outstanding",
        "future_ca_factor",
    }

    missing = required - set(
        x.columns
    )

    if missing:
        raise RuntimeError(
            "shares_nse.parquet missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    x["security_id"] = (
        x["security_id"]
        .astype("string")
    )

    x["available_date"] = (
        pd.to_datetime(
            x["available_date"],
            errors="coerce",
        )
        .astype("datetime64[ns]")
    )

    x["shares_outstanding"] = (
        pd.to_numeric(
            x["shares_outstanding"],
            errors="coerce",
        )
    )

    x["future_ca_factor"] = (
        pd.to_numeric(
            x["future_ca_factor"],
            errors="coerce",
        )
        .fillna(1.0)
    )

    keep = [
        "security_id",
        "available_date",
        "shares_outstanding",
        "future_ca_factor",
    ]

    # Do not carry the shares-side generic "date" column into
    # merge_asof. Prices also use "date", so keeping both would make
    # pandas suffix them to date_x/date_y and remove the market-date
    # column that the rest of the pipeline expects.
    for col in [
        "report_date",
        "reported_shares",
        "ca_multiplier_since_report",
        "source",
    ]:
        if col in x.columns:
            keep.append(col)

    x = x[keep]

    return (
        x.sort_values(
            [
                "available_date",
                "security_id",
            ]
        )
        .drop_duplicates(
            [
                "security_id",
                "available_date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def build_market_data(
    prices,
    shares,
):
    prices = prepare_prices(
        prices
    )

    shares = prepare_shares(
        shares
    )

    market = pd.merge_asof(
        prices.sort_values(
            ["date", "security_id"]
        ),
        shares.sort_values(
            [
                "available_date",
                "security_id",
            ]
        ),
        left_on="date",
        right_on="available_date",
        by="security_id",
        direction="backward",
    )

    is_yahoo = (
        market["price_source"]
        .astype(str)
        .str.upper()
        .str.startswith("YAHOO")
    )

    market[
        "price_basis_factor"
    ] = np.where(
        is_yahoo,
        market[
            "future_ca_factor"
        ].fillna(1.0),
        1.0,
    )

    market[
        "mcap_price"
    ] = (
        pd.to_numeric(
            market["close"],
            errors="coerce",
        )
        * market[
            "price_basis_factor"
        ]
    )

    market[
        "full_market_cap_inr"
    ] = (
        market[
            "mcap_price"
        ]
        * market[
            "shares_outstanding"
        ]
    )

    return market


def diagnostics(market):
    covered = (
        market[
            "shares_outstanding"
        ].notna()
    )

    print(
        "\nShare coverage:",
        round(
            covered.mean(),
            4,
        ),
    )

    print(
        "Rows:",
        len(market),
    )

    print(
        "Securities:",
        market[
            "security_id"
        ].nunique(),
    )

    print(
        "\nPrice source:"
    )

    print(
        market[
            "price_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    suspicious = market[
        covered
        & (
            (
                market[
                    "price_basis_factor"
                ] < 0.05
            )
            | (
                market[
                    "price_basis_factor"
                ] > 50
            )
        )
    ]

    print(
        "\nExtreme price-basis "
        "factor rows:",
        len(suspicious),
    )

    audit = market.sort_values(
        [
            "security_id",
            "date",
        ]
    ).copy()

    audit["previous_mcap"] = (
        audit.groupby(
            "security_id"
        )[
            "full_market_cap_inr"
        ].shift(1)
    )

    audit["mcap_ratio"] = (
        audit[
            "full_market_cap_inr"
        ]
        / audit[
            "previous_mcap"
        ]
    )

    jumps = audit[
        audit["mcap_ratio"].notna()
        & (
            (
                audit[
                    "mcap_ratio"
                ] >= 2.0
            )
            | (
                audit[
                    "mcap_ratio"
                ] <= 0.50
            )
        )
    ]

    print(
        ">=2x or <=0.5x one-day "
        "market-cap jumps:",
        len(jumps),
    )

    if not jumps.empty:
        cols = [
            "security_id",
            "date",
            "price_source",
            "close",
            "price_basis_factor",
            "mcap_price",
            "shares_outstanding",
            "previous_mcap",
            "full_market_cap_inr",
            "mcap_ratio",
        ]

        print(
            jumps[
                cols
            ]
            .head(100)
            .to_string(index=False)
        )


def main():
    prices = pd.read_parquet(
        PRICES_PATH
    )

    shares = pd.read_parquet(
        SHARES_PATH
    )

    market = build_market_data(
        prices,
        shares,
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    market.to_parquet(
        OUT_PATH,
        index=False,
    )

    diagnostics(
        market
    )

    print(
        "\nSaved:",
        OUT_PATH,
    )


if __name__ == "__main__":
    main()