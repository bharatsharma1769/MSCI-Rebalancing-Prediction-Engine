from pathlib import Path

import numpy as np
import pandas as pd


SHARES_PATH = Path(
    "data/raw/shares/shares_nse.parquet"
)

TARGET_IDS = {
    "SEC000562": "DRREDDY",
    "SEC001842": "SHRIRAMFIN",
    "SEC001369": "NAUKRI",
    "SEC000189": "ASHOKLEY",
    "SEC000894": "IGL",
    "SEC000848": "HINDPETRO",
    "SEC001391": "NESTLEIND",
}

RECENT_CA_DAYS = 500
MIN_RAW_ERROR = 0.25
MAX_FIXED_ERROR = 0.12
MIN_IMPROVEMENT = 0.10


def rel_error(a, b):
    if (
        not np.isfinite(a)
        or not np.isfinite(b)
        or b == 0
    ):
        return np.inf

    return abs(a - b) / abs(b)


def event_factor_from_future(
    previous_future,
    current_future,
):
    if (
        not np.isfinite(previous_future)
        or not np.isfinite(current_future)
        or previous_future <= 0
        or current_future <= 0
    ):
        return 1.0

    factor = (
        previous_future
        / current_future
    )

    if np.isclose(
        factor,
        1.0,
        rtol=1e-6,
        atol=1e-9,
    ):
        return 1.0

    if (
        factor < 0.05
        or factor > 50
    ):
        return 1.0

    return float(factor)


def candidate_basis_factors(
    recent_events,
):
    factors = {1.0}

    product = 1.0

    for _, factor in reversed(
        recent_events
    ):
        product *= factor

        if (
            np.isfinite(product)
            and 0.02
            <= product
            <= 100
        ):
            factors.add(
                float(product)
            )

    return sorted(factors)


def repair_security(g):
    g = (
        g.sort_values(
            "available_date"
        )
        .copy()
        .reset_index(drop=True)
    )

    for col in [
        "reported_shares",
        "shares_outstanding",
        "ca_multiplier_since_report",
        "future_ca_factor",
    ]:
        g[col] = pd.to_numeric(
            g[col],
            errors="coerce",
        )

    g[
        "ca_multiplier_since_report"
    ] = g[
        "ca_multiplier_since_report"
    ].fillna(1.0)

    g[
        "future_ca_factor"
    ] = g[
        "future_ca_factor"
    ].fillna(1.0)

    g["filing_basis_factor"] = 1.0

    g[
        "economic_reported_shares"
    ] = g[
        "reported_shares"
    ]

    previous_state = np.nan
    previous_future = np.nan
    recent_events = []

    for i, row in g.iterrows():
        event_date = row[
            "available_date"
        ]

        raw = row[
            "reported_shares"
        ]

        ca_mult = row[
            "ca_multiplier_since_report"
        ]

        current_future = row[
            "future_ca_factor"
        ]

        if not np.isfinite(raw):
            previous_future = (
                current_future
            )
            continue

        if (
            not np.isfinite(ca_mult)
            or ca_mult <= 0
        ):
            ca_mult = 1.0

        new_event_factor = (
            event_factor_from_future(
                previous_future,
                current_future,
            )
        )

        if not np.isclose(
            new_event_factor,
            1.0,
        ):
            recent_events.append(
                (
                    event_date,
                    new_event_factor,
                )
            )

        if pd.notna(event_date):
            recent_events = [
                (d, f)
                for d, f
                in recent_events
                if (
                    pd.notna(d)
                    and (
                        event_date - d
                    ).days
                    <= RECENT_CA_DAYS
                )
            ]

        raw_state = (
            float(raw)
            * float(ca_mult)
        )

        if (
            np.isfinite(previous_state)
            and not np.isclose(
                new_event_factor,
                1.0,
            )
        ):
            expected_state = (
                float(previous_state)
                * float(
                    new_event_factor
                )
            )
        else:
            expected_state = (
                float(previous_state)
                if np.isfinite(
                    previous_state
                )
                else raw_state
            )

        best_factor = 1.0
        best_error = rel_error(
            raw_state,
            expected_state,
        )

        for factor in (
            candidate_basis_factors(
                recent_events
            )
        ):
            corrected_state = (
                raw_state
                * factor
            )

            error = rel_error(
                corrected_state,
                expected_state,
            )

            if error < best_error:
                best_error = error
                best_factor = factor

        raw_error = rel_error(
            raw_state,
            expected_state,
        )

        use_factor = 1.0

        if (
            not np.isclose(
                best_factor,
                1.0,
            )
            and raw_error
            >= MIN_RAW_ERROR
            and best_error
            <= MAX_FIXED_ERROR
            and best_error
            <= (
                raw_error
                - MIN_IMPROVEMENT
            )
        ):
            use_factor = float(
                best_factor
            )

        economic_reported = int(
            round(
                float(raw)
                * use_factor
            )
        )

        economic_state = int(
            round(
                float(raw_state)
                * use_factor
            )
        )

        g.loc[
            i,
            "filing_basis_factor",
        ] = use_factor

        g.loc[
            i,
            "economic_reported_shares",
        ] = economic_reported

        g.loc[
            i,
            "shares_outstanding",
        ] = economic_state

        previous_state = float(
            economic_state
        )

        previous_future = (
            current_future
        )

    return g


def main():
    shares = pd.read_parquet(
        SHARES_PATH
    )

    required = {
        "security_id",
        "available_date",
        "reported_shares",
        "shares_outstanding",
        "ca_multiplier_since_report",
        "future_ca_factor",
    }

    missing = (
        required
        - set(
            shares.columns
        )
    )

    if missing:
        raise RuntimeError(
            "shares_nse.parquet missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    shares[
        "available_date"
    ] = pd.to_datetime(
        shares[
            "available_date"
        ],
        errors="coerce",
    )

    if "report_date" in shares.columns:
        shares[
            "report_date"
        ] = pd.to_datetime(
            shares[
                "report_date"
            ],
            errors="coerce",
        )

    before = pd.to_numeric(
        shares[
            "shares_outstanding"
        ],
        errors="coerce",
    ).copy()

    repaired = pd.concat(
        [
            repair_security(
                g.copy()
            )
            for _, g
            in shares.groupby(
                "security_id",
                sort=False,
            )
        ],
        ignore_index=True,
    )

    repaired = (
        repaired.sort_values(
            [
                "available_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    # Compare by stable event key, not row position.
    before_df = shares[
        [
            "security_id",
            "available_date",
        ]
    ].copy()

    before_df[
        "shares_before"
    ] = before

    check = repaired[
        [
            "security_id",
            "available_date",
            "shares_outstanding",
        ]
    ].merge(
        before_df,
        on=[
            "security_id",
            "available_date",
        ],
        how="left",
    )

    after = pd.to_numeric(
        check[
            "shares_outstanding"
        ],
        errors="coerce",
    )

    old = pd.to_numeric(
        check[
            "shares_before"
        ],
        errors="coerce",
    )

    changed = ~np.isclose(
        after,
        old,
        rtol=1e-10,
        atol=0,
        equal_nan=True,
    )

    basis_fixed = (
        ~np.isclose(
            pd.to_numeric(
                repaired[
                    "filing_basis_factor"
                ],
                errors="coerce",
            ).fillna(1.0),
            1.0,
        )
    )

    print(
        "Rows changed:",
        int(
            changed.sum()
        ),
    )

    print(
        "Securities changed:",
        check.loc[
            changed,
            "security_id",
        ].nunique(),
    )

    print(
        "Rows using filing-basis correction:",
        int(
            basis_fixed.sum()
        ),
    )

    print(
        "Securities using filing-basis correction:",
        repaired.loc[
            basis_fixed,
            "security_id",
        ].nunique(),
    )

    print(
        "\n--- TARGET SHARE STATES ---"
    )

    target = repaired[
        repaired[
            "security_id"
        ].isin(
            TARGET_IDS
        )
    ].copy()

    target["symbol"] = (
        target[
            "security_id"
        ].map(
            TARGET_IDS
        )
    )

    cols = [
        "security_id",
        "symbol",
        "report_date",
        "available_date",
        "reported_shares",
        "filing_basis_factor",
        "economic_reported_shares",
        "ca_multiplier_since_report",
        "shares_outstanding",
        "future_ca_factor",
        "source",
    ]

    cols = [
        c
        for c in cols
        if c in target.columns
    ]

    print(
        target[
            cols
        ]
        .sort_values(
            [
                "security_id",
                "available_date",
            ]
        )
        .to_string(
            index=False
        )
    )

    temp_path = (
        SHARES_PATH.with_name(
            SHARES_PATH.stem
            + "_repair_tmp.parquet"
        )
    )

    repaired.to_parquet(
        temp_path,
        index=False,
    )

    temp_path.replace(
        SHARES_PATH
    )

    print(
        "\nUpdated:",
        SHARES_PATH,
    )


if __name__ == "__main__":
    main()
