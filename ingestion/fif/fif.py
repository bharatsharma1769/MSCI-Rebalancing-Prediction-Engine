import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests


MASTER_PATH = "data/processed/historical_security_master.parquet"
PRICES_PATH = "data/raw/prices/prices.parquet"

OUTPUT_PATH = Path("data/raw/fif/fif.parquet")
CHECKPOINT_PATH = Path("data/raw/fif/fif_checkpoint.parquet")
FAILED_PATH = Path("data/raw/fif/fif_failed.csv")

SHAREHOLDING_URL = (
    "https://www.nseindia.com/api/"
    "corporate-share-holdings-master"
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

START_DATE = pd.Timestamp("2021-01-01")


# Public holdings treated as strategic / non-free-float
STRATEGIC_CONTEXTS = [
    [
        "Governments_ContextI",
        "GovermentsI",
    ],
    [
        "AssociateCompaniesOrSubsidiaries_ContextI",
        "AssociateCompaniesOrSubsidiariesI",
    ],
    [
        "DirectorsAndDirectorsRelatives_ContextI",
        "DirectorsAndDirectorsRelativesI",
    ],
    [
        "KeyManagerialPersonnel_ContextI",
        "KeyManagerialPersonnelI",
    ],
    [
        "RelativesOfPromotersOtherThanPromoterGroup_ContextI",
        "RelativesOfPromotersOtherThanPromoterGroupI",
    ],
    [
        "InvestorEducationAndProtectionFund_ContextI",
        "InvestorEducationAndProtectionFundI",
    ],
    [
    "EmployeeBenefitsTrusts_ContextI",
    "EmployeeBenefitsTrustsI",
]
]

PROMOTER_CONTEXTS = [
    "ShareholdingOfPromoterAndPromoterGroup_ContextI",
    "ShareholdingOfPromoterAndPromoterGroupI",
]

PUBLIC_CONTEXTS = [
    "PublicShareholding_ContextI",
    "PublicShareholdingI",
]

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
        timeout=20
    )

    return session


def get_filings(session, symbol, retries=4):

    for attempt in range(retries):

        try:

            r = session.get(
                SHAREHOLDING_URL,
                params={
                    "index": "equities",
                    "symbol": symbol,
                },
                timeout=15,
            )

            r.raise_for_status()

            return r.json(), session

        except Exception as e:

            print(
                f"{symbol} metadata attempt "
                f"{attempt + 1} failed: {e}"
            )

            time.sleep(
                2 * (attempt + 1)
            )

            # Refresh the session after a failure
            try:
                session.close()
            except:
                pass

            session = create_session()

    return None, session

def get_context_pct(root, contexts):

    if isinstance(contexts, str):
        contexts = [contexts]

    for elem in root.iter():

        if (
            elem.tag.split("}")[-1]
            != "ShareholdingAsAPercentageOfTotalNumberOfShares"
        ):
            continue

        context = elem.attrib.get(
            "contextRef",
            ""
        )

        if context not in contexts:
            continue

        try:
            value = float(elem.text)
        except (TypeError, ValueError):
            continue

        return value

    return None


def parse_fif(xml_bytes):

    root = ET.fromstring(xml_bytes)

    promoter = get_context_pct(
        root,
        PROMOTER_CONTEXTS
    )

    public = get_context_pct(
        root,
        PUBLIC_CONTEXTS
    )

    # Cannot calculate FIF without these
    if promoter is None or public is None:
        return None

    strategic = 0.0

    for aliases in STRATEGIC_CONTEXTS:

        value = get_context_pct(
            root,
            aliases
        )

        if value is not None:
            strategic += value

    # Normalize scale first
    ownership_total = promoter + public

    if ownership_total > 2:
        promoter /= 100
        public /= 100
        strategic /= 100

    raw_free_float = public - strategic

    raw_free_float = max(
        0.0,
        min(raw_free_float, 1.0)
    )

    return {
        "promoter_pct": promoter,
        "public_pct": public,
        "strategic_public_pct": strategic,
        "raw_free_float": raw_free_float,
        "fif": round_fif(raw_free_float),
    }


def round_fif(free_float):

    if pd.isna(free_float):
        return None

    free_float = float(free_float)

    if free_float >= 0.15:

        # Round UP to nearest 5%
        fif = (
            int(
                free_float * 20
                + 0.999999999
            )
            / 20
        )

    else:

        # Nearest 1%
        fif = round(
            free_float,
            2
        )

    return min(
        max(fif, 0.0),
        1.0
    )


def load_checkpoint():

    if CHECKPOINT_PATH.exists():

        df = pd.read_parquet(
            CHECKPOINT_PATH
        )

        processed = set(
            df["security_id"]
            .unique()
        )

        print(
            "Loaded checkpoint:",
            len(processed),
            "securities"
        )

        return df, processed

    return pd.DataFrame(), set()


def save_checkpoint(results, failed):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if results:

        df = pd.concat(
            results,
            ignore_index=True
        )

        df = (
            df
            .sort_values(
                [
                    "security_id",
                    "available_date",
                    "report_date",
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

        df.to_parquet(
            CHECKPOINT_PATH,
            index=False
        )

    pd.DataFrame(
        failed
    ).to_csv(
        FAILED_PATH,
        index=False
    )


def main():

    master = pd.read_parquet(
        MASTER_PATH
    )

    prices = pd.read_parquet(
        PRICES_PATH,
        columns=["security_id"]
    )

    # Only securities for which we actually have market data
    price_ids = set(
        prices["security_id"]
        .dropna()
        .unique()
    )

    master = master[
        master["security_id"].isin(
            price_ids
        )
    ].copy()

    checkpoint, processed_ids = (
        load_checkpoint()
    )

    results = []

    if not checkpoint.empty:
        results.append(
            checkpoint
        )

    failed = []

    session = create_session()

    total = len(master)
    processed_this_run = 0

    for _, row in master.iterrows():

        security_id = row["security_id"]
        symbol = row["nse_symbol"]

        if security_id in processed_ids:
            continue

        print(
            f"\n{symbol} "
            f"({processed_this_run + 1}/{total})"
        )

        filings, session = get_filings(
        session,
        symbol
    )

        if filings is None:

            failed.append({
                "security_id": security_id,
                "nse_symbol": symbol,
                "reason": "metadata_failed",
            })

            processed_this_run += 1
            continue

        security_rows = []

        for filing in filings:

            report_date = pd.to_datetime(
                filing.get("date"),
                format="%d-%b-%Y",
                errors="coerce",
            )

            available_date = pd.to_datetime(
                filing.get("submissionDate"),
                format="%d-%b-%Y",
                errors="coerce",
            )

            xbrl_url = filing.get(
                "xbrl"
            )

            if not xbrl_url:
                continue

            try:
                xr = session.get(
                    xbrl_url,
                    timeout=30
                )

                xr.raise_for_status()

                parsed = parse_fif(
                    xr.content
                )

            except Exception as e:

                print(
                    "XBRL failed:",
                    e
                )

                continue

            if parsed is None:
                continue

            fif = round_fif(
                parsed["raw_free_float"]
            )

            security_rows.append({
                "security_id": security_id,
                "report_date": report_date,
                "available_date": available_date,
                "promoter_pct":
                    parsed["promoter_pct"],
                "public_pct":
                    parsed["public_pct"],
                "strategic_public_pct":
                    parsed["strategic_public_pct"],
                "raw_free_float":
                    parsed["raw_free_float"],
                "fif": fif,
                "source": "NSE_XBRL_ESTIMATE",
            })

            time.sleep(0.04)

        if security_rows:

            security_df = pd.DataFrame(
                security_rows
            )

            # If several reports became public on the
            # same date, retain latest report period
            security_df = (
                security_df
                .sort_values(
                    [
                        "available_date",
                        "report_date",
                    ]
                )
                .drop_duplicates(
                    ["available_date"],
                    keep="last",
                )
            )

            results.append(
                security_df
            )

            print(
                "Filings:",
                len(security_df)
            )

        else:

            failed.append({
                "security_id": security_id,
                "nse_symbol": symbol,
                "reason": "no_usable_fif_filings",
            })

            print(
                "No usable FIF filings"
            )

        processed_ids.add(
            security_id
        )

        processed_this_run += 1

        if processed_this_run % 25 == 0:

            save_checkpoint(
                results,
                failed
            )

            print(
                "\nCheckpoint saved."
            )

        time.sleep(0.12)

    if not results:
        raise ValueError(
            "No FIF data recovered."
        )

    fif = pd.concat(
        results,
        ignore_index=True
    )

    fif["report_date"] = pd.to_datetime(
        fif["report_date"]
    ).astype("datetime64[ns]")

    fif["available_date"] = pd.to_datetime(
        fif["available_date"]
    ).astype("datetime64[ns]")

    fif = (
        fif
        .sort_values(
            [
                "security_id",
                "available_date",
                "report_date",
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

    # Sanity checks
    bad_dates = (
        fif["available_date"]
        < fif["report_date"]
    )

    if bad_dates.any():
        raise ValueError(
            "Found available_date before report_date."
        )

    bad_ff = ~fif[
        "raw_free_float"
    ].between(
        0,
        1
    )

    if bad_ff.any():
        raise ValueError(
            "Invalid raw free-float values."
        )

    bad_fif = ~fif[
        "fif"
    ].between(
        0,
        1
    )

    if bad_fif.any():
        raise ValueError(
            "Invalid FIF values."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fif.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    pd.DataFrame(
        failed
    ).to_csv(
        FAILED_PATH,
        index=False
    )

    print(
        "\nSaved rows:",
        len(fif)
    )

    print(
        "Covered securities:",
        fif["security_id"].nunique()
    )

    print(
        "Failed securities:",
        len(failed)
    )

    print(
        "\nFIF distribution:"
    )

    print(
        fif["fif"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nRaw free-float summary:"
    )

    print(
        fif["raw_free_float"]
        .describe()
    )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

        print(
            "\nFinished. Checkpoint removed."
        )


if __name__ == "__main__":
    main()