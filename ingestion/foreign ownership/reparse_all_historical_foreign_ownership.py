from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


FO_PATH = Path("data/processed/historical_foreign_ownership.parquet")
CHECKPOINT_PATH = Path(
    "data/raw/foreign_ownership/"
    "historical_foreign_ownership_reparse_v2_checkpoint.parquet"
)

TARGET_TAG = "ShareholdingAsAPercentageOfTotalNumberOfShares"

PROTECTED_KEYS = {
    ("2023-11", "SEC001008"),  # JIOFIN
    ("2023-11", "SEC000935"),  # INDUSINDBK
    ("2024-08", "SEC000269"),  # BANDHANBNK
    ("2025-11", "SEC002213"),  # VMM
}

PROTECTED_SECURITIES = {"SEC000972"}  # ITC

PROTECTED_TEXT = (
    r"MANUAL|PATCH|REPAIR|SPECIAL|OVERRIDE|"
    r"CORPORATE_EVENT|SOURCE_DEAD|DEAD_XBRL"
)

AGGREGATE_FOREIGN_MEMBERS = {
    "InstitutionsForeignMember",
    "NonResidentIndiansMember",
    "ForeignNationalsMember",
    "ForeignCompaniesMember",
    "NonResidentIndividualsOrForeignIndividualsMember",
}

INSTITUTION_CHILD_MEMBERS = {
    "ForeignDirectInvestmentMember",
    "ForeignVentureCapitalInvestorsMember",
    "SovereignWealthFundsForeignMember",
    "InstitutionsForeignPortfolioInvestorCatergoryOneMember",
    "InstitutionsForeignPortfolioInvestorCatergoryTwoMember",
    "InstitutionsForeignPortfolioInvestorCategoryOneMember",
    "InstitutionsForeignPortfolioInvestorCategoryTwoMember",
    "OverseasDepositoriesMember",
    "OtherInstitutionsForeignMember",
    "ForeignInstitutionsMember",
}


def clean_tag(tag):
    return tag.split("}")[-1].strip()


def member_name(x):
    return "" if not x else str(x).split(":")[-1].strip()


def create_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def percentage_scale(values):
    vals = pd.Series(values, dtype="float64").dropna()
    if vals.empty:
        return np.nan
    return .01 if vals.gt(1).any() else 1.0


def extract_foreign_ownership(url, session):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    contexts = {}

    for e in root.iter():
        if clean_tag(e.tag) != "context":
            continue

        cid = e.attrib.get("id")
        if not cid:
            continue

        contexts[cid] = [
            member_name(c.text)
            for c in e.iter()
            if clean_tag(c.tag) in {"explicitMember", "typedMember"} and c.text
        ]

    facts = []
    all_pct = []

    for e in root.iter():
        if clean_tag(e.tag) != TARGET_TAG or e.text is None:
            continue

        try:
            value = float(e.text.strip())
        except ValueError:
            continue

        if value < 0:
            continue

        all_pct.append(value)
        members = contexts.get(e.attrib.get("contextRef"), [])

        if members:
            facts.append({
                "context": e.attrib.get("contextRef"),
                "members": members,
                "value": value,
            })

    if not facts:
        return np.nan, 0, None, "NO_PERCENTAGE_FACTS"

    scale = percentage_scale(all_pct)

    if pd.isna(scale):
        return np.nan, 0, None, "NO_PERCENTAGE_SCALE"

    rows = []

    for fact in facts:
        members = [
            m for m in fact["members"]
            if m in AGGREGATE_FOREIGN_MEMBERS
            or m in INSTITUTION_CHILD_MEMBERS
        ]

        if members:
            rows.append({
                "context": fact["context"],
                "value": fact["value"] * scale,
                "members": members,
            })

    if not rows:
        return np.nan, 0, None, "NO_FOREIGN_CONTEXTS"

    rows = list({r["context"]: r for r in rows}.values())

    aggregate = [
        r for r in rows
        if "InstitutionsForeignMember" in r["members"]
    ]

    if aggregate:
        institutional = max(r["value"] for r in aggregate)
        institutional_members = {"InstitutionsForeignMember"}
        method = "AGGREGATE_INSTITUTION_USED"

    else:
        by_member = {}

        for r in rows:
            for m in r["members"]:
                if m not in INSTITUTION_CHILD_MEMBERS:
                    continue
                by_member[m] = max(by_member.get(m, 0.0), r["value"])

        institutional = sum(by_member.values())
        institutional_members = set(by_member)
        method = "CHILD_INSTITUTIONS_SUMMED"

    other = {}

    for r in rows:
        for m in r["members"]:
            if m not in AGGREGATE_FOREIGN_MEMBERS:
                continue
            if m == "InstitutionsForeignMember":
                continue
            other[m] = max(other.get(m, 0.0), r["value"])

    if (
        "NonResidentIndividualsOrForeignIndividualsMember" in other
        and (
            "NonResidentIndiansMember" in other
            or "ForeignNationalsMember" in other
        )
    ):
        other.pop("NonResidentIndividualsOrForeignIndividualsMember", None)

    foreign_ownership = institutional + sum(other.values())
    used = institutional_members | set(other)

    if foreign_ownership < -1e-9 or foreign_ownership > 1.0001:
        return np.nan, len(rows), " || ".join(sorted(used)), "TOTAL_OUT_OF_RANGE"

    return foreign_ownership, len(rows), " || ".join(sorted(used)), method


def protected_mask(x):
    explicit = [
        (r, s) in PROTECTED_KEYS or s in PROTECTED_SECURITIES
        for r, s in zip(x["review"], x["security_id"])
    ]

    text_cols = [
        c for c in [
            "source", "data_quality_flag", "canonical_row_action",
            "parser_repair_status", "foreign_ownership_method",
        ]
        if c in x.columns
    ]

    if text_cols:
        text = x[text_cols].astype("string").fillna("").agg(" ".join, axis=1)
        textual = text.str.contains(PROTECTED_TEXT, case=False, regex=True)
    else:
        textual = pd.Series(False, index=x.index)

    return pd.Series(explicit, index=x.index) | textual


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame()

    cp = pd.read_parquet(CHECKPOINT_PATH)
    cp["xbrl_url"] = cp["xbrl_url"].astype("string")
    return cp


def save_checkpoint(rows):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("xbrl_url", keep="last").to_parquet(
        CHECKPOINT_PATH, index=False
    )


def main():
    x = pd.read_parquet(FO_PATH)
    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")

    if x.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate historical FO keys.")

    x["protected_parser_row"] = protected_mask(x)

    valid_url = (
        x["xbrl_url"].notna()
        & x["xbrl_url"].astype(str).str.startswith("http")
    )

    reparsable = x[valid_url & ~x["protected_parser_row"]].copy()
    urls = reparsable["xbrl_url"].drop_duplicates().astype("string")

    cp = load_checkpoint()

    completed = set()
    rows = []

    if not cp.empty:
        rows = cp.to_dict("records")
        completed = set(
            cp.loc[
                ~cp["parser_status_v2"].eq("FETCH_OR_XML_ERROR"),
                "xbrl_url",
            ].dropna().astype(str)
        )

    todo = [u for u in urls if str(u) not in completed]

    print("\n--- HISTORICAL FO PARSER V2 ---")
    print("Canonical rows:", len(x))
    print("Rows with reparsable XBRL:", len(reparsable))
    print("Unique XBRLs:", len(urls))
    print("Protected rows skipped:", int(x["protected_parser_row"].sum()))
    print("Already parsed:", len(completed))
    print("To fetch:", len(todo))

    session = create_session()

    for i, url in enumerate(todo, 1):
        try:
            fo, nctx, members, status = extract_foreign_ownership(str(url), session)
            error = pd.NA
        except Exception as e:
            fo, nctx, members = np.nan, np.nan, None
            status = "FETCH_OR_XML_ERROR"
            error = str(e)[:500]

        rows.append({
            "xbrl_url": str(url),
            "foreign_ownership_v2": fo,
            "foreign_context_count_v2": nctx,
            "foreign_members_v2": members,
            "parser_status_v2": status,
            "parser_error_v2": error,
        })

        if i % 50 == 0:
            save_checkpoint(rows)
            print(f"Processed: {i}/{len(todo)}")

        time.sleep(.03)

    save_checkpoint(rows)
    cp = load_checkpoint()

    parsed = cp.drop_duplicates("xbrl_url", keep="last").set_index("xbrl_url")
    old_fo = pd.to_numeric(x["foreign_ownership"], errors="coerce").copy()

    x["fo_parser_v2_status"] = pd.NA
    x["fo_parser_v2_action"] = "NO_REPARSE"

    x.loc[x["protected_parser_row"], "fo_parser_v2_action"] = "PRESERVE_PROTECTED"

    changed = pd.Series(False, index=x.index)
    reparsed = pd.Series(False, index=x.index)

    for idx in x[valid_url & ~x["protected_parser_row"]].index:
        url = str(x.at[idx, "xbrl_url"])

        if url not in parsed.index:
            x.at[idx, "fo_parser_v2_action"] = "PRESERVE_NO_PARSE_RESULT"
            continue

        p = parsed.loc[url]
        status = p["parser_status_v2"]
        new_fo = pd.to_numeric(
            pd.Series([p["foreign_ownership_v2"]]),
            errors="coerce",
        ).iloc[0]

        x.at[idx, "fo_parser_v2_status"] = status

        if pd.isna(new_fo) or status in {
            "FETCH_OR_XML_ERROR",
            "NO_PERCENTAGE_FACTS",
            "NO_PERCENTAGE_SCALE",
            "NO_FOREIGN_CONTEXTS",
            "TOTAL_OUT_OF_RANGE",
        }:
            x.at[idx, "fo_parser_v2_action"] = "PRESERVE_OLD_PARSE_UNRESOLVED"
            continue

        old = pd.to_numeric(
            pd.Series([x.at[idx, "foreign_ownership"]]),
            errors="coerce",
        ).iloc[0]

        x.at[idx, "foreign_ownership"] = new_fo
        x.at[idx, "foreign_context_count"] = p["foreign_context_count_v2"]
        x.at[idx, "foreign_members"] = p["foreign_members_v2"]
        x.at[idx, "data_quality_flag"] = "OK"
        x.at[idx, "parser_repair_status"] = status
        x.at[idx, "fo_parser_v2_action"] = "REPARSED_V2"

        reparsed.at[idx] = True
        changed.at[idx] = pd.isna(old) or abs(new_fo - old) > 1e-10

    x["foreign_ownership"] = pd.to_numeric(x["foreign_ownership"], errors="coerce")

    invalid = (
        x["foreign_ownership"].notna()
        & ~x["foreign_ownership"].between(0, 1)
    )

    if invalid.any():
        raise RuntimeError(
            f"Invalid foreign ownership after reparse: {int(invalid.sum())}"
        )

    x["available_date"] = pd.to_datetime(x["available_date"], errors="coerce")
    x["foreign_ownership_cutoff_date"] = pd.to_datetime(
        x["foreign_ownership_cutoff_date"], errors="coerce"
    )

    lookahead = (
        x["available_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["available_date"].gt(x["foreign_ownership_cutoff_date"])
    )

    if lookahead.any():
        raise RuntimeError(
            f"Available-date lookahead after reparse: {int(lookahead.sum())}"
        )

    if x.duplicated(["review", "security_id"]).any():
        raise RuntimeError("Duplicate historical FO keys after reparse.")

    new_fo = x["foreign_ownership"]
    delta = new_fo - old_fo

    x = x.drop(columns="protected_parser_row")

    x.to_parquet(FO_PATH, index=False)

    status_counts = cp["parser_status_v2"].value_counts(dropna=False)
    fetch_failures = int(cp["parser_status_v2"].eq("FETCH_OR_XML_ERROR").sum())

    print("\n--- HISTORICAL FO PARSER V2 RESULT ---")
    print("Rows reparsed successfully:", int(reparsed.sum()))
    print("Rows whose FO changed:", int(changed.sum()))
    print("Rows newly resolved:", int((old_fo.isna() & new_fo.notna()).sum()))
    print("Rows changed >5pp:", int((delta.abs() > .05).sum()))
    print("Rows changed >20pp:", int((delta.abs() > .20).sum()))
    print("Protected rows preserved:", int(x["fo_parser_v2_action"].eq("PRESERVE_PROTECTED").sum()))
    print("Rows preserving old value due unresolved v2 parse:", int(
        x["fo_parser_v2_action"].eq("PRESERVE_OLD_PARSE_UNRESOLVED").sum()
    ))
    print("Fetch/XML failures:", fetch_failures)

    print("\nParser status:")
    print(status_counts.to_string())

    print("\nFO before resolved:", int(old_fo.notna().sum()))
    print("FO after resolved:", int(new_fo.notna().sum()))
    print("Available-date lookahead:", int(lookahead.sum()))
    print("Duplicates:", int(x.duplicated(["review", "security_id"]).sum()))
    print("Saved:", FO_PATH)

    if fetch_failures == 0 and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Temporary checkpoint removed.")
    else:
        print("Checkpoint retained for retry:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()