from pathlib import Path

import pandas as pd


MAIN_PATH = Path(
    "data/raw/shares/"
    "shares_nse.parquet"
)

HIST_PATH = Path(
    "data/raw/shares/"
    "historical_delisted_shares.parquet"
)


def main():
    main = pd.read_parquet(
        MAIN_PATH
    )

    hist = pd.read_parquet(
        HIST_PATH
    )

    all_columns = list(
        dict.fromkeys(
            list(main.columns)
            + list(hist.columns)
        )
    )

    for col in all_columns:
        if col not in main.columns:
            main[col] = pd.NA

        if col not in hist.columns:
            hist[col] = pd.NA

    main = main[
        all_columns
    ]

    hist = hist[
        all_columns
    ]

    for col in [
        "date",
        "report_date",
        "available_date",
    ]:
        if col in all_columns:
            main[col] = pd.to_datetime(
                main[col],
                errors="coerce",
            ).astype("datetime64[ns]")

            hist[col] = pd.to_datetime(
                hist[col],
                errors="coerce",
            ).astype("datetime64[ns]")

    combined = pd.concat(
        [
            main,
            hist,
        ],
        ignore_index=True,
    )

    combined = (
        combined.drop_duplicates(
            [
                "security_id",
                "available_date",
            ],
            keep="last",
        )
        .sort_values(
            [
                "available_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    combined.to_parquet(
        MAIN_PATH,
        index=False,
    )

    print(
        "Historical securities added:",
        hist[
            "security_id"
        ].nunique(),
    )

    print(
        "Historical rows added:",
        len(hist),
    )

    print(
        "Final share securities:",
        combined[
            "security_id"
        ].nunique(),
    )

    print(
        "Final share rows:",
        len(combined),
    )


if __name__ == "__main__":
    main()
