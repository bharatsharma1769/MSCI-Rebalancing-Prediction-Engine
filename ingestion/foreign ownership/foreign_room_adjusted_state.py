import numpy as np
import pandas as pd


FOREIGN_ROOM_PATH = "data/processed/foreign_room.parquet"


REVIEW_ORDER = [
    "2023-05",
    "2023-08",
    "2023-11",
    "2024-02",
    "2024-05",
    "2024-08",
    "2024-11",
    "2025-02",
    "2025-05",
    "2025-08",
    "2025-11",
    "2026-02",
    "2026-05",
]


# Official exceptional treatment for HDFC BANK.
# MSCI stated in Aug-2024 that HDFC BANK was coming from 0.50,
# moved it to 0.75 in Aug-2024 and then to 1.00 in Nov-2024.
HDFCBANK_SYMBOL = "HDFCBANK"

HDFCBANK_OFFICIAL_FACTOR = {
    "2024-08": 0.75,
    "2024-11": 1.00,
}


def factor_from_room(current_factor, room):
    if pd.isna(room):
        return np.nan

    if room >= 0.25:
        return 1.00

    if room >= 0.15:
        if current_factor >= 1.00:
            return 1.00
        if current_factor >= 0.50:
            return 0.50
        return 0.50

    if room >= 0.075:
        if current_factor >= 1.00:
            return 0.50
        if current_factor >= 0.50:
            return 0.50
        return 0.25

    if room >= 0.0375:
        return 0.25

    return 0.00


def review_distance(a, b):
    ia = REVIEW_ORDER.index(a)
    ib = REVIEW_ORDER.index(b)
    return ib - ia


def main():
    x = pd.read_parquet(
        FOREIGN_ROOM_PATH
    )

    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")

    order_map = {
        r: i
        for i, r in enumerate(
            REVIEW_ORDER
        )
    }

    x["_review_order"] = (
        x["review"]
        .map(order_map)
    )

    x["foreign_room_adjustment_factor_pre"] = np.nan
    x["foreign_room_adjustment_factor_post"] = np.nan
    x["foreign_room_factor_change"] = pd.NA
    x["foreign_room_12m_hold_applied"] = False
    x["foreign_room_last_reduction_review"] = pd.NA
    x["foreign_room_factor_source"] = pd.NA

    for sid, idxs in (
        x.groupby(
            "security_id",
            sort=False,
        ).groups.items()
    ):
        idxs = sorted(
            idxs,
            key=lambda i: x.at[
                i,
                "_review_order",
            ],
        )

        current_factor = 1.0
        last_reduction_review = None

        for idx in idxs:
            review = x.at[
                idx,
                "review",
            ]

            existing = bool(
                x.at[
                    idx,
                    "was_standard_constituent",
                ]
            )

            room = pd.to_numeric(
                pd.Series(
                    [
                        x.at[
                            idx,
                            "foreign_room",
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            symbol = str(
                x.at[
                    idx,
                    "nse_symbol",
                ]
            )

            if not existing:
                x.at[
                    idx,
                    "foreign_room_factor_source",
                ] = "NOT_EXISTING"
                continue

            if symbol == HDFCBANK_SYMBOL and review == "2024-08":
                    current_factor = 0.50
            x.at[
                idx,
                "foreign_room_adjustment_factor_pre",
            ] = current_factor

            proposed = factor_from_room(
                current_factor,
                room,
            )

            source = "ROOM_BAND"

            # Red Flag/Breach deletion overrides ordinary factor state.
            if (
                "india_red_flag_breach_fail"
                in x.columns
                and bool(
                    x.at[
                        idx,
                        "india_red_flag_breach_fail",
                    ]
                )
            ):
                proposed = 0.0
                source = "INDIA_RED_FLAG_BREACH"

            # Explicit official HDFC BANK treatment.
            if (
                symbol == HDFCBANK_SYMBOL
                and review
                in HDFCBANK_OFFICIAL_FACTOR
            ):
                proposed = (
                    HDFCBANK_OFFICIAL_FACTOR[
                        review
                    ]
                )
                source = (
                    "OFFICIAL_HDFCBANK_OVERRIDE"
                )

            if pd.isna(proposed):
                x.at[
                    idx,
                    "foreign_room_factor_source",
                ] = "ROOM_UNRESOLVED"
                continue

            # General 12-month rule:
            # after a reduction, upward movement is normally delayed
            # until four quarterly reviews have elapsed.
            #
            # We do not invoke the FOL-increase exception unless an
            # explicit official override is provided.
            if (
                proposed > current_factor
                and last_reduction_review
                is not None
                and source
                != "OFFICIAL_HDFCBANK_OVERRIDE"
            ):
                elapsed = review_distance(
                    last_reduction_review,
                    review,
                )

                if elapsed < 4:
                    proposed = current_factor
                    x.at[
                        idx,
                        "foreign_room_12m_hold_applied",
                    ] = True
                    source = "12M_RESTORATION_HOLD"

            if proposed < current_factor:
                last_reduction_review = review

            if proposed > current_factor:
                change = "UP"
            elif proposed < current_factor:
                change = "DOWN"
            else:
                change = "UNCHANGED"

            x.at[
                idx,
                "foreign_room_adjustment_factor_post",
            ] = proposed

            x.at[
                idx,
                "foreign_room_factor_change",
            ] = change

            x.at[
                idx,
                "foreign_room_last_reduction_review",
            ] = last_reduction_review

            x.at[
                idx,
                "foreign_room_factor_source",
            ] = source

            current_factor = proposed

    x = x.drop(
        columns=[
            "_review_order"
        ]
    )

    x.to_parquet(
        FOREIGN_ROOM_PATH,
        index=False,
    )

    print(
        "\n--- FOREIGN ROOM ADJUSTMENT FACTOR STATE ---"
    )

    existing = (
        x[
            "was_standard_constituent"
        ].fillna(False).astype(bool)
    )

    print(
        "Existing rows:",
        int(existing.sum()),
    )

    print(
        "Reduced-factor rows:",
        int(
            (
                existing
                & x[
                    "foreign_room_adjustment_factor_post"
                ].notna()
                & (
                    x[
                        "foreign_room_adjustment_factor_post"
                    ] < 1
                )
            ).sum()
        ),
    )

    print(
        "12m restoration holds:",
        int(
            x[
                "foreign_room_12m_hold_applied"
            ].sum()
        ),
    )

    print(
        "Zero-factor rows:",
        int(
            x[
                "foreign_room_adjustment_factor_post"
            ].eq(0)
            .sum()
        ),
    )

    print(
        "\nNon-1 factor rows:"
    )

    z = x[
        existing
        & x[
            "foreign_room_adjustment_factor_post"
        ].notna()
        & (
            x[
                "foreign_room_adjustment_factor_post"
            ] < 1
        )
    ][
        [
            "review",
            "security_id",
            "nse_symbol",
            "foreign_room",
            "foreign_room_adjustment_factor_pre",
            "foreign_room_adjustment_factor_post",
            "foreign_room_factor_change",
            "foreign_room_12m_hold_applied",
            "foreign_room_factor_source",
        ]
    ]

    if z.empty:
        print("None")
    else:
        print(
            z.to_string(
                index=False
            )
        )

    print(
        "\nSaved:",
        FOREIGN_ROOM_PATH,
    )


if __name__ == "__main__":
    main()
