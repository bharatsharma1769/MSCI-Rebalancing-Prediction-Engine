import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests


MASTER_PATH = Path(
    "data/processed/"
    "historical_security_master.parquet"
)

HISTORICAL_PRICES_PATH = Path(
    "data/raw/prices/"
    "historical_delisted_all.parquet"
)

OUTPUT_PATH = Path(
    "data/raw/shares/"
    "historical_delisted_shares.parquet"
)

FAILED_PATH = Path(
    "data/raw/shares/"
    "historical_delisted_shares_failed.csv"
)

CHECKPOINT_PATH = Path(
    "data/raw/shares/"
    "historical_delisted_shares_checkpoint.pkl"
)

SHAREHOLDING_URL = (
    "https://www.nseindia.com/api/"
    "corporate-share-holdings-master"
)

CORPORATE_ACTIONS_URL = (
    "https://www.nseindia.com/api/"
    "corporates-corporateActions"
)

START_DATE = pd.Timestamp("2021-01-01")
CHECKPOINT_EVERY = 10

TATA_DVR_ID = "HIST_TATAMTRDVR"
TATA_DVR_SHARES = 508_502_896
TATA_DVR_KNOWN_DATE = pd.Timestamp(
    "2023-06-30"
)
TATA_DVR_END = pd.Timestamp(
    "2024-09-01"
)


def create_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    })

    session.get(
        "https://www.nseindia.com/",
        timeout=20,
    )

    return session


def get_json(session, url, params):
    r = session.get(
        url,
        params=params,
        timeout=25,
    )

    r.raise_for_status()

    data = r.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get(
            "data",
            [],
        )

    return []


def get_total_shares(
    session,
    xbrl_url,
):
    r = session.get(
        xbrl_url,
        timeout=25,
    )

    r.raise_for_status()

    root = ET.fromstring(
        r.content
    )

    values = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag != "NumberOfShares":
            continue

        if elem.text is None:
            continue

        try:
            values.append(
                int(
                    float(
                        elem.text
                        .replace(",", "")
                        .strip()
                    )
                )
            )
        except ValueError:
            pass

    if not values:
        return None

    return max(values)


def parse_date(x):
    return pd.to_datetime(
        x,
        dayfirst=True,
        errors="coerce",
    )


def parse_ca_multiplier(subject):
    s = str(subject).lower()

    if "bonus" in s:
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*[:/]\s*"
            r"(\d+(?:\.\d+)?)",
            s,
        )

        if m:
            a = float(m.group(1))
            b = float(m.group(2))

            if b > 0:
                return 1.0 + a / b

    if any(
        word in s
        for word in [
            "split",
            "sub-division",
            "sub division",
            "subdivision",
            "consolidation",
        ]
    ):
        m = re.search(
            r"from[^0-9]*"
            r"(\d+(?:\.\d+)?)"
            r".*?to[^0-9]*"
            r"(\d+(?:\.\d+)?)",
            s,
        )

        if m:
            old_fv = float(
                m.group(1)
            )
            new_fv = float(
                m.group(2)
            )

            if new_fv > 0:
                return old_fv / new_fv

        m = re.search(
            r"(\d+(?:\.\d+)?)\s*:\s*"
            r"(\d+(?:\.\d+)?)",
            s,
        )

        if m:
            a = float(m.group(1))
            b = float(m.group(2))

            if a > 0 and b > 0:
                if "consolidation" in s:
                    return min(a, b) / max(a, b)

                return max(a, b) / min(a, b)

    return np.nan


def get_filings(
    session,
    symbol,
    security_id,
):
    rows = []

    filings = get_json(
        session,
        SHAREHOLDING_URL,
        {
            "index": "equities",
            "symbol": symbol,
        },
    )

    for filing in filings:
        report_date = parse_date(
            filing.get("date")
        )

        available_date = parse_date(
            filing.get(
                "submissionDate"
            )
        )

        xbrl_url = filing.get(
            "xbrl"
        )

        if (
            pd.isna(report_date)
            or pd.isna(available_date)
            or not xbrl_url
        ):
            continue

        try:
            shares = get_total_shares(
                session,
                xbrl_url,
            )
        except Exception:
            continue

        if shares is None:
            continue

        rows.append({
            "security_id":
                security_id,
            "report_date":
                report_date.normalize(),
            "available_date":
                available_date.normalize(),
            "reported_shares":
                int(shares),
        })

        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "report_date",
                "available_date",
            ]
        )
        .drop_duplicates(
            "report_date",
            keep="last",
        )
        .reset_index(drop=True)
    )


def get_actions(
    session,
    symbol,
):
    end_date = (
        pd.Timestamp.today()
        .normalize()
        + pd.Timedelta(days=1)
    )

    actions = get_json(
        session,
        CORPORATE_ACTIONS_URL,
        {
            "index": "equities",
            "symbol": symbol,
            "from_date": (
                START_DATE.strftime(
                    "%d-%m-%Y"
                )
            ),
            "to_date": (
                end_date.strftime(
                    "%d-%m-%Y"
                )
            ),
        },
    )

    rows = []

    for action in actions:
        subject = action.get(
            "subject",
            "",
        )

        s = str(subject).lower()

        if not any(
            word in s
            for word in [
                "bonus",
                "split",
                "sub-division",
                "sub division",
                "subdivision",
                "consolidation",
            ]
        ):
            continue

        ex_date = parse_date(
            action.get("exDate")
        )

        if pd.isna(ex_date):
            continue

        rows.append({
            "ex_date":
                ex_date.normalize(),
            "subject":
                subject,
            "multiplier":
                parse_ca_multiplier(
                    subject
                ),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "ex_date",
                "subject",
                "multiplier",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("ex_date")
        .drop_duplicates(
            [
                "ex_date",
                "subject",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def snap_factor(x):
    candidates = np.array([
        0.10,
        0.20,
        0.25,
        1 / 3,
        0.50,
        2 / 3,
        0.80,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
        4.00,
        5.00,
        10.00,
        20.00,
    ])

    if not np.isfinite(x):
        return np.nan

    idx = np.argmin(
        np.abs(
            candidates - x
        )
        / candidates
    )

    candidate = float(
        candidates[idx]
    )

    if abs(candidate - x) / candidate <= 0.08:
        return candidate

    return np.nan


def infer_missing(
    filings,
    actions,
):
    x = actions.copy()

    for i, row in x[
        x["multiplier"].isna()
    ].iterrows():
        ex_date = row["ex_date"]

        pre = filings[
            filings["report_date"]
            < ex_date
        ]

        post = filings[
            filings["report_date"]
            >= ex_date
        ]

        if pre.empty or post.empty:
            continue

        a = pre.iloc[-1]
        b = post.iloc[0]

        between = x[
            (
                x["ex_date"]
                > a["report_date"]
            )
            & (
                x["ex_date"]
                <= b["report_date"]
            )
        ]

        if len(between) != 1:
            continue

        raw = (
            b["reported_shares"]
            / a["reported_shares"]
        )

        inferred = snap_factor(
            raw
        )

        if pd.notna(inferred):
            x.loc[
                i,
                "multiplier",
            ] = inferred

    return x


def build_timeline(
    filings,
    actions,
    security_id,
):
    actions = infer_missing(
        filings,
        actions,
    )

    unresolved = actions[
        actions["multiplier"].isna()
    ]

    if not unresolved.empty:
        raise RuntimeError(
            "Unparsed split/bonus action: "
            + " | ".join(
                unresolved["subject"]
                .astype(str)
                .head(5)
            )
        )

    event_dates = set(
        filings[
            "available_date"
        ].tolist()
    )

    event_dates.update(
        actions[
            "ex_date"
        ].tolist()
    )

    rows = []

    for event_date in sorted(
        event_dates
    ):
        known = filings[
            filings["available_date"]
            <= event_date
        ]

        if known.empty:
            continue

        base = (
            known.sort_values(
                [
                    "report_date",
                    "available_date",
                ]
            )
            .iloc[-1]
        )

        active_ca = actions[
            (
                actions["ex_date"]
                > base["report_date"]
            )
            & (
                actions["ex_date"]
                <= event_date
            )
        ]

        ca_since_report = (
            active_ca["multiplier"]
            .prod()
            if not active_ca.empty
            else 1.0
        )

        future_ca = actions[
            actions["ex_date"]
            > event_date
        ]

        future_factor = (
            future_ca["multiplier"]
            .prod()
            if not future_ca.empty
            else 1.0
        )

        shares = int(
            round(
                float(
                    base["reported_shares"]
                )
                * float(
                    ca_since_report
                )
            )
        )

        rows.append({
            "date":
                base["report_date"],
            "report_date":
                base["report_date"],
            "available_date":
                event_date,
            "security_id":
                security_id,
            "shares_outstanding":
                shares,
            "reported_shares":
                int(
                    base[
                        "reported_shares"
                    ]
                ),
            "ca_multiplier_since_report":
                float(
                    ca_since_report
                ),
            "future_ca_factor":
                float(
                    future_factor
                ),
            "source":
                (
                    "NSE_XBRL_HISTORICAL_CA_STATE"
                    if not active_ca.empty
                    else "NSE_XBRL_HISTORICAL"
                ),
        })

    return pd.DataFrame(rows)


def tata_dvr_special():
    return pd.DataFrame([
        {
            "date":
                TATA_DVR_KNOWN_DATE,
            "report_date":
                TATA_DVR_KNOWN_DATE,
            "available_date":
                TATA_DVR_KNOWN_DATE,
            "security_id":
                TATA_DVR_ID,
            "shares_outstanding":
                TATA_DVR_SHARES,
            "reported_shares":
                TATA_DVR_SHARES,
            "ca_multiplier_since_report":
                1.0,
            "future_ca_factor":
                1.0,
            "source":
                "OFFICIAL_TATA_DVR_SHARE_COUNT",
        }
    ])


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return [], set(), []

    state = pd.read_pickle(
        CHECKPOINT_PATH
    )

    return (
        state.get(
            "results",
            [],
        ),
        set(
            state.get(
                "processed_ids",
                [],
            )
        ),
        state.get(
            "failed",
            [],
        ),
    )


def save_checkpoint(
    results,
    processed,
    failed,
):
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.to_pickle(
        {
            "results":
                results,
            "processed_ids":
                sorted(processed),
            "failed":
                failed,
        },
        CHECKPOINT_PATH,
    )

    print(
        "Checkpoint saved:",
        len(processed),
    )


def main():
    master = pd.read_parquet(
        MASTER_PATH
    )

    prices = pd.read_parquet(
        HISTORICAL_PRICES_PATH
    )

    target_ids = set(
        prices["security_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    targets = master[
        master["security_id"]
        .astype(str)
        .isin(target_ids)
    ].copy()

    results, processed, failed = (
        load_checkpoint()
    )

    session = create_session()

    count = 0

    for _, row in targets.iterrows():
        security_id = str(
            row["security_id"]
        )

        symbol = str(
            row["nse_symbol"]
        )

        if security_id in processed:
            continue

        print(
            "Fetching",
            symbol,
        )

        x = pd.DataFrame()
        last_error = None

        for attempt in range(3):
            try:
                if security_id == TATA_DVR_ID:
                    x = tata_dvr_special()
                else:
                    filings = get_filings(
                        session,
                        symbol,
                        security_id,
                    )

                    if filings.empty:
                        raise RuntimeError(
                            "NO_USABLE_FILINGS"
                        )

                    actions = get_actions(
                        session,
                        symbol,
                    )

                    x = build_timeline(
                        filings,
                        actions,
                        security_id,
                    )

                if x.empty:
                    raise RuntimeError(
                        "EMPTY_TIMELINE"
                    )

                last_error = None
                break

            except Exception as e:
                last_error = e

                print(
                    f"Attempt {attempt + 1} failed:",
                    str(e)[:300],
                )

                time.sleep(
                    2 * (attempt + 1)
                )

                if attempt == 1:
                    try:
                        session = create_session()
                    except Exception:
                        pass

        if last_error is not None:
            failed.append({
                "security_id":
                    security_id,
                "nse_symbol":
                    symbol,
                "reason":
                    str(last_error)[:300],
            })
        else:
            results.append(x)

        processed.add(
            security_id
        )

        count += 1

        if count % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                results,
                processed,
                failed,
            )

        time.sleep(0.2)

    if not results:
        raise RuntimeError(
            "No historical shares recovered."
        )

    shares = pd.concat(
        results,
        ignore_index=True,
    )

    for col in [
        "date",
        "report_date",
        "available_date",
    ]:
        shares[col] = pd.to_datetime(
            shares[col],
            errors="coerce",
        ).astype("datetime64[ns]")

    shares = (
        shares.drop_duplicates(
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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shares.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    failed_df = pd.DataFrame(
        failed
    )

    if not failed_df.empty:
        failed_df.to_csv(
            FAILED_PATH,
            index=False,
        )
    elif FAILED_PATH.exists():
        FAILED_PATH.unlink()

    print(
        "\nHistorical share securities:",
        shares["security_id"].nunique(),
    )

    print(
        "Rows:",
        len(shares),
    )

    print(
        "Failures:",
        len(failed_df),
    )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(
        "Checkpoint removed."
    )


if __name__ == "__main__":
    main()
