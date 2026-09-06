import re
from pathlib import Path

import pandas as pd
import os

JAVA_PATH = r"C:\Program Files\Common Files\Oracle\Java\javapath"

os.environ["PATH"] = (
    JAVA_PATH
    + os.pathsep
    + os.environ.get("PATH", "")
)

import tabula

PDF_PATH = (
    "data/raw/foreign_ownership/"
    "nsdl_fol.pdf"
)

MASTER_PATH = (
    "data/processed/"
    "historical_security_master.parquet"
)

OUT_PATH = (
    "data/raw/foreign_ownership/"
    "fol_nsdl.parquet"
)



tables = tabula.read_pdf(
    PDF_PATH,
    pages="all",
    multiple_tables=True,
    force_subprocess=True,
    pandas_options={
        "header": None
    }
)




print("Tables:", len(tables))

rows = []

for t in tables:

    for _, r in t.iterrows():

        isin = str(r.get(1, "")).strip()

        if not re.fullmatch(
            r"INE[A-Z0-9]{8}[0-9]",
            isin
        ):
            continue

        rows.append(
            {
                "isin": isin,
                "issuer_name": r.get(2),
                "nri_limit_raw": r.get(3),
                "fpi_limit_raw": r.get(4),
                "sectoral_cap_raw": r.get(5),
                "govt_approved_limit_raw": r.get(6),
                "paid_up_capital_raw": r.get(7),
                "remarks": r.get(8),
            }
        )

out = pd.DataFrame(rows)

print("Parsed rows:", len(out))

if out.empty:
    raise RuntimeError(
        "No NSDL FOL rows parsed"
    )


# --------------------------------------------------
# Percentage parser
# --------------------------------------------------

def pct(x):

    if pd.isna(x):
        return None

    x = str(x).strip()

    m = re.search(
        r"\d+(?:\.\d+)?",
        x
    )

    if not m:
        return None

    value = float(
        m.group()
    )

    if value < 0 or value > 100:
        return None

    return value / 100


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
# Effective overall FOL
# --------------------------------------------------

out["fol"] = (
    out["govt_approved_limit"]
    .fillna(
        out["sectoral_cap"]
    )
)

out["source"] = (
    "NSDL_STATIC_REPORT"
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
    .astype(str)
    .str.strip()
)

out["isin"] = (
    out["isin"]
    .astype(str)
    .str.strip()
)


out = out.merge(
    master,
    on="isin",
    how="left"
)


out = (
    out
    .drop_duplicates(
        "isin",
        keep="last"
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# Sanity
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


print(
    "\n--- LOWEST FOL ---"
)

print(
    out[
        out["fol"].notna()
    ][
        [
            "nse_symbol",
            "issuer_name",
            "fol",
            "fpi_limit",
            "nri_limit",
            "sectoral_cap",
            "govt_approved_limit",
        ]
    ]
    .sort_values(
        "fol"
    )
    .head(30)
    .to_string(index=False)
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