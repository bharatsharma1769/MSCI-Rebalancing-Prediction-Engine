import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


PDF_PATH = (
    "data/raw/foreign_ownership/"
    "csdl_fol.pdf"
)

MASTER_PATH = (
    "data/processed/"
    "historical_security_master.parquet"
)

OUT_PATH = (
    "data/raw/foreign_ownership/"
    "fol_cdsl.parquet"
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

ISIN_RE = re.compile(
    r"IN[A-Z0-9]{10}"
)


def pct(x):
    if x is None:
        return np.nan

    x = str(x).strip().upper()

    if x in {
        "",
        "NA",
        "N.A.",
        "NAN",
        "-",
    }:
        return np.nan

    try:
        v = float(x)
    except ValueError:
        return np.nan

    if not 0 <= v <= 100:
        return np.nan

    return v / 100


def is_pct_token(x):
    x = str(x).strip().upper()

    if x in {
        "NA",
        "N.A.",
        "-",
    }:
        return True

    try:
        v = float(x)
        return 0 <= v <= 100
    except ValueError:
        return False


def clean_int_token(x):
    return re.sub(
        r"[,\s]",
        "",
        str(x)
    )


def is_share_count(x):
    x = clean_int_token(x)

    if not x.isdigit():
        return False

    # paid-up share counts in this file
    # are much larger than percentage values
    return int(x) > 100


# --------------------------------------------------
# Extract text
# --------------------------------------------------

reader = PdfReader(
    PDF_PATH
)

lines = []

for page in reader.pages:

    text = page.extract_text() or ""

    for line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if line:
            lines.append(line)


print(
    "PDF pages:",
    len(reader.pages)
)


# --------------------------------------------------
# Build logical company records
# --------------------------------------------------
#
# Usually:
# 123 INE... COMPANY ... 10 100 100 ...
#
# But wrapped cases can be:
#
# 948
# INE040A01034 HDFC BANK LIMITED ...
# --------------------------------------------------

records = []

current = None
pending_serial = None


for line in lines:

    # stand-alone serial number
    if re.fullmatch(
        r"\d+",
        line
    ):
        pending_serial = line
        continue


    # normal: serial + ISIN on same line
    m = re.match(
        r"^(\d+)\s+"
        r"(IN[A-Z0-9]{10})\s+"
        r"(.*)$",
        line
    )

    if m:

        if current:
            records.append(current)

        current = (
            f"{m.group(1)} "
            f"{m.group(2)} "
            f"{m.group(3)}"
        )

        pending_serial = None
        continue


    # wrapped: serial was previous line,
    # current line starts with ISIN
    m = re.match(
        r"^(IN[A-Z0-9]{10})\s+"
        r"(.*)$",
        line
    )

    if (
        m
        and
        pending_serial is not None
    ):

        if current:
            records.append(current)

        current = (
            f"{pending_serial} "
            f"{m.group(1)} "
            f"{m.group(2)}"
        )

        pending_serial = None
        continue


    # continuation of existing record
    if current is not None:

        # ignore obvious page/footer headers
        lower = line.lower()

        if (
            "classification - public"
            in lower
            or
            "aggregate permissible foreign"
            in lower
            or
            "foreign portfolio investor"
            in lower
            or
            "non-resident indian"
            in lower
            or
            "paid up equity capital"
            in lower
        ):
            continue

        current += " " + line


if current:
    records.append(current)


print(
    "Logical records:",
    len(records)
)


# --------------------------------------------------
# Parse logical records
# --------------------------------------------------

rows = []


for rec in records:

    m = re.match(
        r"^(\d+)\s+"
        r"(IN[A-Z0-9]{10})\s+"
        r"(.+)$",
        rec
    )

    if not m:
        continue


    serial = int(
        m.group(1)
    )

    isin = m.group(2)

    rest = m.group(3)

    tokens = rest.split()


    # Find the paid-up share-count token.
    #
    # Immediately before it should normally be:
    #
    # NRI, FPI, sector cap
    #
    # or:
    #
    # NRI, FPI, sector cap, govt approval
    #
    share_idx = None


    for i in range(
        3,
        len(tokens)
    ):

        if not is_share_count(
            tokens[i]
        ):
            continue


        # Must have at least 3 pct-like tokens
        # immediately before share count.
        if (
            i >= 3
            and
            is_pct_token(tokens[i - 1])
            and
            is_pct_token(tokens[i - 2])
            and
            is_pct_token(tokens[i - 3])
        ):

            share_idx = i
            break


    if share_idx is None:
        continue


    # Determine whether there are 3 or 4
    # percentage fields preceding paid-up shares.
    #
    # Govt approval is optional.

    numeric_start = share_idx - 3

    govt_raw = None


    if (
        share_idx >= 4
        and
        is_pct_token(
            tokens[
                share_idx - 4
            ]
        )
    ):
        # Avoid accidentally consuming a number
        # belonging to the issuer name.
        #
        # Treat as govt approval only when the
        # preceding three are also valid pct fields.
        candidate = tokens[
            share_idx - 4
        ]

        issuer_candidate = " ".join(
            tokens[
                :share_idx - 4
            ]
        )

        if issuer_candidate:
            numeric_start = (
                share_idx - 4
            )
            govt_raw = tokens[
                share_idx - 1
            ]


    nums = tokens[
        numeric_start:
        share_idx
    ]


    if len(nums) == 3:

        nri_raw = nums[0]
        fpi_raw = nums[1]
        sector_raw = nums[2]
        govt_raw = None

    elif len(nums) == 4:

        nri_raw = nums[0]
        fpi_raw = nums[1]
        sector_raw = nums[2]
        govt_raw = nums[3]

    else:
        continue


    issuer = " ".join(
        tokens[
            :numeric_start
        ]
    )


    paid_up_raw = tokens[
        share_idx
    ]


    remarks = " ".join(
        tokens[
            share_idx + 1:
        ]
    )


    rows.append(
        {
            "serial": serial,
            "isin": isin,
            "issuer_name": issuer,
            "nri_limit_raw": nri_raw,
            "fpi_limit_raw": fpi_raw,
            "sectoral_cap_raw": sector_raw,
            "govt_approved_limit_raw": govt_raw,
            "paid_up_capital_raw": paid_up_raw,
            "remarks": remarks,
        }
    )


out = pd.DataFrame(
    rows
)


print(
    "Parsed rows:",
    len(out)
)

print(
    "Unique ISINs:",
    out["isin"].nunique()
)


# --------------------------------------------------
# Numeric fields
# --------------------------------------------------

out["nri_limit"] = (
    out["nri_limit_raw"]
    .map(pct)
)

out["fpi_limit"] = (
    out["fpi_limit_raw"]
    .map(pct)
)

out["sectoral_cap"] = (
    out["sectoral_cap_raw"]
    .map(pct)
)

out["govt_approved_limit"] = (
    out["govt_approved_limit_raw"]
    .map(pct)
)


# --------------------------------------------------
# Effective FOL proxy
# --------------------------------------------------

ceilings = pd.concat(
    [
        out["fpi_limit"],
        out["sectoral_cap"],
        out[
            "govt_approved_limit"
        ],
    ],
    axis=1
)


out["fol"] = (
    ceilings.min(
        axis=1,
        skipna=True
    )
)


out.loc[
    ceilings.notna()
    .sum(axis=1)
    == 0,
    "fol"
] = np.nan


out["source"] = "CDSL"


# --------------------------------------------------
# Critical checks BEFORE master merge
# --------------------------------------------------

targets = [
    "INE040A01034",  # HDFCBANK
    "INE090A01021",  # ICICIBANK
    "INE238A01034",  # AXISBANK
    "INE062A01020",  # SBIN
]


print(
    "\n--- IMPORTANT CHECKS ---"
)

print(
    out[
        out["isin"]
        .isin(targets)
    ][
        [
            "isin",
            "issuer_name",
            "fpi_limit",
            "sectoral_cap",
            "govt_approved_limit",
            "fol",
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# Map to master
# --------------------------------------------------

master = pd.read_parquet(
    MASTER_PATH,
    columns=[
        "security_id",
        "nse_symbol",
        "isin",
    ]
)


master["isin"] = (
    master["isin"]
    .astype("string")
    .str.strip()
)


out = out.merge(
    master,
    on="isin",
    how="left"
)


out = (
    out
    .sort_values("serial")
    .drop_duplicates(
        "isin",
        keep="last"
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print(
    "\n--- BASIC ---"
)

print(
    "Rows:",
    len(out)
)

print(
    "Mapped securities:",
    out[
        "security_id"
    ].nunique()
)

print(
    "Missing security_id:",
    out[
        "security_id"
    ].isna().sum()
)

print(
    "With FOL:",
    out[
        "fol"
    ].notna().sum()
)

print(
    "Invalid FOL:",
    (
        (out["fol"] <= 0)
        |
        (out["fol"] > 1)
    ).sum()
)


print(
    "\n--- FOL DISTRIBUTION ---"
)

print(
    out["fol"]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# --------------------------------------------------
# Clean parquet dtypes
# --------------------------------------------------

text_cols = [
    "isin",
    "issuer_name",
    "nri_limit_raw",
    "fpi_limit_raw",
    "sectoral_cap_raw",
    "govt_approved_limit_raw",
    "paid_up_capital_raw",
    "remarks",
    "source",
    "security_id",
    "nse_symbol",
]


for col in text_cols:

    if col in out.columns:
        out[col] = (
            out[col]
            .astype("string")
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(
    OUT_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)


out.to_parquet(
    OUT_PATH,
    index=False
)


print(
    "\nSaved:",
    OUT_PATH
)