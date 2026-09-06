import numpy as np
import pandas as pd
from pathlib import Path


NSDL_PATH = (
    "data/raw/foreign_ownership/"
    "fol_nsdl.parquet"
)

CDSL_PATH = (
    "data/raw/foreign_ownership/"
    "fol_cdsl.parquet"
)

MASTER_PATH = (
    "data/processed/"
    "security_master.parquet"
)

OUT_PATH = (
    "data/processed/"
    "fol_master.parquet"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

nsdl = pd.read_parquet(
    NSDL_PATH
)

cdsl = pd.read_parquet(
    CDSL_PATH
)

master = pd.read_parquet(
    MASTER_PATH,
    columns=[
        "security_id",
        "nse_symbol",
        "company_name",
        "isin",
    ]
)


for df in [
    nsdl,
    cdsl,
    master,
]:

    df["isin"] = (
        df["isin"]
        .astype("string")
        .str.strip()
    )


# --------------------------------------------------
# Effective India FOL proxy
# --------------------------------------------------
#
# For our MSCI replication:
#
# 1. Company-specific FPI limit is the primary
#    accessible foreign ownership limit.
#
# 2. It cannot exceed the sectoral cap.
#
# 3. If an explicit government-approved aggregate
#    limit exists, it also acts as a ceiling.
#
# NRI limit is kept separately and is NOT used as
# the institutional foreign-investor FOL proxy.
# --------------------------------------------------

def build_fol(df):

    x = df.copy()

    for col in [
        "fpi_limit",
        "sectoral_cap",
        "govt_approved_limit",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce"
        )

    ceilings = pd.concat(
        [
            x["fpi_limit"],
            x["sectoral_cap"],
            x["govt_approved_limit"],
        ],
        axis=1
    )

    x["fol"] = (
        ceilings.min(
            axis=1,
            skipna=True
        )
    )

    # no usable foreign-limit information
    no_limit = (
        ceilings.notna()
        .sum(axis=1)
        == 0
    )

    x.loc[
        no_limit,
        "fol"
    ] = np.nan

    return x


nsdl = build_fol(
    nsdl
)

cdsl = build_fol(
    cdsl
)


nsdl["fol_source"] = "NSDL"
cdsl["fol_source"] = "CDSL"


# --------------------------------------------------
# Keep common schema
# --------------------------------------------------

cols = [
    "isin",
    "issuer_name",
    "nri_limit",
    "fpi_limit",
    "sectoral_cap",
    "govt_approved_limit",
    "fol",
    "fol_source",
]


nsdl = nsdl[
    [
        c
        for c in cols
        if c in nsdl.columns
    ]
].copy()


cdsl = cdsl[
    [
        c
        for c in cols
        if c in cdsl.columns
    ]
].copy()


# --------------------------------------------------
# Combine designated depositories
# --------------------------------------------------

combined = pd.concat(
    [
        nsdl,
        cdsl,
    ],
    ignore_index=True
)


# --------------------------------------------------
# Check overlaps
# --------------------------------------------------

dup = combined[
    combined.duplicated(
        "isin",
        keep=False
    )
].copy()


print(
    "\n--- NSDL/CDSL OVERLAPS ---"
)

print(
    "Overlapping ISIN rows:",
    len(dup)
)

print(
    "Overlapping ISINs:",
    dup["isin"].nunique()
)


if not dup.empty:

    conflict = (
        dup.groupby("isin")["fol"]
        .nunique(
            dropna=True
        )
    )

    conflict = conflict[
        conflict > 1
    ]

    print(
        "FOL conflicts:",
        len(conflict)
    )

    if len(conflict):

        print(
            dup[
                dup["isin"].isin(
                    conflict.index
                )
            ][
                [
                    "isin",
                    "issuer_name",
                    "fpi_limit",
                    "sectoral_cap",
                    "govt_approved_limit",
                    "fol",
                    "fol_source",
                ]
            ]
            .sort_values(
                [
                    "isin",
                    "fol_source",
                ]
            )
            .head(50)
            .to_string(
                index=False
            )
        )


# --------------------------------------------------
# Dedupe
# --------------------------------------------------
#
# They should normally be designated to one
# depository. If duplicated and identical, either
# source is fine.
# --------------------------------------------------

combined = (
    combined
    .sort_values(
        [
            "isin",
            "fol_source",
        ]
    )
    .drop_duplicates(
        "isin",
        keep="first"
    )
)


# --------------------------------------------------
# Map onto NSE universe
# --------------------------------------------------

out = master.merge(
    combined,
    on="isin",
    how="left"
)

import re


def norm_name(x):

    if pd.isna(x):
        return ""

    x = str(x).upper()

    x = re.sub(
        r"\bLIMITED\b|\bLTD\b|\bPRIVATE\b|\bPVT\b",
        " ",
        x,
    )

    x = re.sub(
        r"\([^)]*\)",
        " ",
        x,
    )

    x = re.sub(
        r"[^A-Z0-9]+",
        " ",
        x,
    )

    return " ".join(
        x.split()
    )


# --------------------------------------------------
# Exact issuer-name fallback for changed ISINs
# --------------------------------------------------


combined["name_norm"] = (
    combined["issuer_name"]
    .map(norm_name)
)

out["name_norm"] = (
    out["company_name"]
    .map(norm_name)
)

out["fol_is_snapshot"] = True
out["fol_snapshot_date"] = pd.Timestamp("2026-08-29")

# Only allow names that identify exactly one
# depository record.
name_counts = (
    combined[
        combined["name_norm"] != ""
    ]
    .groupby("name_norm")
    .size()
)


unique_names = set(
    name_counts[
        name_counts == 1
    ].index
)


name_lookup = (
    combined[
        combined["name_norm"]
        .isin(unique_names)
    ]
    .set_index("name_norm")
)


missing_mask = (
    out["fol_source"].isna()
    &
    out["name_norm"].isin(
        unique_names
    )
)


fallback_cols = [
    "issuer_name",
    "nri_limit",
    "fpi_limit",
    "sectoral_cap",
    "govt_approved_limit",
    "fol",
    "fol_source",
]


for col in fallback_cols:

    out.loc[
        missing_mask,
        col,
    ] = (
        out.loc[
            missing_mask,
            "name_norm",
        ]
        .map(
            name_lookup[col]
        )
        .values
    )


out.loc[
    missing_mask,
    "fol_source",
] = (
    out.loc[
        missing_mask,
        "fol_source",
    ]
    + "_NAME_MATCH"
)


print(
    "\n--- ISIN CHANGE FALLBACK ---"
)

print(
    "Recovered:",
    missing_mask.sum()
)


print(
    out.loc[
        missing_mask,
        [
            "nse_symbol",
            "isin",
            "issuer_name",
            "fol",
            "fol_source",
        ],
    ]
    .head(100)
    .to_string(index=False)
)

# --------------------------------------------------
# Sanity
# --------------------------------------------------

print(
    "\n--- COMBINED FOL COVERAGE ---"
)

print(
    "NSE securities:",
    len(out)
)

print(
    "With source:",
    out[
        "fol_source"
    ].notna().sum()
)

print(
    "With FOL:",
    out[
        "fol"
    ].notna().sum()
)

print(
    "FOL coverage:",
    round(
        out[
            "fol"
        ].notna().mean()
        * 100,
        1
    ),
    "%"
)

print(
    "\n--- SOURCE ---"
)

print(
    out[
        "fol_source"
    ]
    .value_counts(
        dropna=False
    )
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
    "\n--- LOW FOL ---"
)

print(
    out[
        out["fol"].notna()
    ][
        [
            "nse_symbol",
            "isin",
            "fpi_limit",
            "sectoral_cap",
            "govt_approved_limit",
            "fol",
            "fol_source",
        ]
    ]
    .sort_values(
        "fol"
    )
    .head(50)
    .to_string(
        index=False
    )
)


print(
    "\n--- MISSING FOL ---"
)

print(
    out[
        out["fol"].isna()
    ][
        [
            "security_id",
            "nse_symbol",
            "isin",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
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