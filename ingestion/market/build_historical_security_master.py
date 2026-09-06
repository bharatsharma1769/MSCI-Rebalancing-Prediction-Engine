import pandas as pd
import requests
from pathlib import Path


MASTER_PATH = "data/processed/security_master.parquet"
OUTPUT_PATH = "data/processed/historical_security_master.parquet"

DELISTED_URL = (
    "https://www.nseindia.com/"
    "static/list/list-of-companies-proposed-to-be-delisted"
)


from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from pathlib import Path
import requests


def download_delisted_file():

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    })

    page_url = (
        "https://www.nseindia.com/static/"
        "list/list-of-companies-proposed-to-be-delisted"
    )

    page = session.get(
        page_url,
        timeout=30
    )

    page.raise_for_status()

    soup = BeautifulSoup(
        page.text,
        "html.parser"
    )

    link = None

    for a in soup.find_all("a"):

        text = a.get_text(
            " ",
            strip=True
        ).lower()

        href = a.get("href", "")

        if (
            "companies delisted from nse" in text
            and ".xlsx" in href.lower()
        ):
            link = href
            break

    if link is None:
        raise ValueError(
            "Could not find delisted-company Excel file."
        )

    # NSE sometimes wraps Excel files in Microsoft's viewer
    if "view.officeapps.live.com" in link:

        parsed = urlparse(link)

        params = parse_qs(
            parsed.query
        )

        if "src" not in params:
            raise ValueError(
                "Could not extract Excel URL from Office viewer."
            )

        link = unquote(
            params["src"][0]
        )

    if link.startswith("/"):
        link = (
            "https://www.nseindia.com"
            + link
        )

    print("Actual Excel URL:")
    print(link)

    r = session.get(
        link,
        timeout=30
    )

    r.raise_for_status()

    path = Path(
        "data/raw/universe/"
        "nse_delisted_companies.xlsx"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_bytes(
        r.content
    )

    print(
        "Downloaded bytes:",
        len(r.content)
    )

    return path


def load_delisted(path):

    df = pd.read_excel(path)

    print("\nRaw columns:")
    print(df.columns.tolist())

    df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        for col in df.columns
    ]

    return df


def build_historical_master():

    current = pd.read_parquet(
        MASTER_PATH
    )

    path = download_delisted_file()

    delisted = load_delisted(path)

    print("\nFirst rows:")
    print(delisted.head())

    print(
        "\nCurrent securities:",
        len(current)
    )

    print(
        "Delisted rows:",
        len(delisted)
    )

    delisted.to_parquet(
        "data/raw/security_master/"
        "nse_delisted_companies.parquet",
        index=False
    )

def build_historical_master():

    current = pd.read_parquet(
        MASTER_PATH
    )

    path = download_delisted_file()

    delisted = load_delisted(path)

    delisted["delisted_date"] = pd.to_datetime(
        delisted["delisted_date"]
    )

    # We only care about names that could affect our backtest
    delisted = delisted[
        delisted["delisted_date"] >= pd.Timestamp("2021-01-01")
    ].copy()

    print(
        "\nRelevant delisted rows:",
        len(delisted)
    )

    # Remove anything already in current master
    existing_isins = set(
        current["isin"]
        .dropna()
        .astype(str)
    )

    missing = delisted[
        ~delisted["isin"].isin(existing_isins)
    ].copy()

    print(
        "Missing historical securities:",
        len(missing)
    )

    # Assign new internal IDs after current max
    current_ids = (
        current["security_id"]
        .str.replace("SEC", "", regex=False)
        .astype(int)
    )

    next_id = current_ids.max() + 1

    missing = missing.reset_index(drop=True)

    missing["security_id"] = [
        f"SEC{i:06d}"
        for i in range(
            next_id,
            next_id + len(missing)
        )
    ]

    missing["company_id"] = missing["security_id"]

    missing["nse_symbol"] = missing["symbol"]
    missing["listing_date"] = pd.NaT

    missing["exchange"] = "NSE"
    missing["country"] = "India"
    missing["currency"] = "INR"
    missing["security_type"] = "Ordinary Share"

    missing["active"] = False
    missing["eligible_equity"] = True

    missing = missing[
        [
            "security_id",
            "company_id",
            "nse_symbol",
            "company_name",
            "isin",
            "listing_date",
            "delisted_date",
            "exchange",
            "country",
            "currency",
            "security_type",
            "active",
            "eligible_equity"
        ]
    ]

    # Ensure current master has delisted_date
    if "delisted_date" not in current.columns:
        current["delisted_date"] = pd.NaT

    historical = pd.concat(
        [current, missing],
        ignore_index=True
    )

    historical.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nCurrent securities:",
        len(current)
    )

    print(
        "Added historical securities:",
        len(missing)
    )

    print(
        "Historical master total:",
        len(historical)
    )

    print(
        "\nSaved to:",
        OUTPUT_PATH
    )

    if not missing.empty:
        print(
            "\nAdded names:"
        )

        print(
            missing[
                [
                    "nse_symbol",
                    "company_name",
                    "isin",
                    "delisted_date"
                ]
            ].to_string(index=False)
        )
        
if __name__ == "__main__":
    build_historical_master()