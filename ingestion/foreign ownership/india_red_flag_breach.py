from pathlib import Path
import csv
import io
import re

import pandas as pd


FOREIGN_ROOM_PATH = Path("data/processed/foreign_room.parquet")
MASTER_PATH = Path("data/processed/historical_security_master.parquet")
RAW_DIR = Path("data/raw/foreign_ownership")

REVIEW_ORDER = [
    "2023-05", "2023-08", "2023-11",
    "2024-02", "2024-05", "2024-08", "2024-11",
    "2025-02", "2025-05", "2025-08", "2025-11",
    "2026-02", "2026-05",
]

ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")


def read_text(path):
    data = path.read_bytes()

    for enc in ["utf-8-sig", "cp1252", "latin1"]:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass

    raise RuntimeError(f"Could not decode {path.name}")


def file_date(path):
    m = re.search(r"List_(\d{8})", path.name, flags=re.I)

    if not m:
        raise RuntimeError(
            f"Could not extract date from {path.name}"
        )

    return pd.to_datetime(
        m.group(1),
        format="%d%m%Y",
    )


def review_from_date(d):
    return (
        d + pd.offsets.MonthBegin(1)
    ).strftime("%Y-%m")


def parse_dissemination_date(text):
    m = re.search(
        r"Dissemination Date:\s*(\d{1,2}-[A-Za-z]{3}-\d{4})",
        text,
        flags=re.I,
    )

    if not m:
        return pd.NaT

    return pd.to_datetime(
        m.group(1),
        format="%d-%b-%Y",
        errors="coerce",
    )


def parse_num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return pd.NA


def parse_alert_file(path, kind):
    text = read_text(path)
    rows = csv.reader(io.StringIO(text))

    section = None
    out = []

    for raw in rows:
        row = [str(x).strip() for x in raw]

        if not row:
            continue

        first = row[0].lower()

        if first.startswith("overall / sectoral limit"):
            section = "OVERALL_SECTORAL"
            continue

        if first.startswith("aggregate fpi investment limit"):
            section = "FPI"
            continue

        if first.startswith("aggregate nri"):
            section = "NRI"
            continue

        if not row[0].isdigit() or len(row) < 3:
            continue

        isin = row[2].strip().upper()

        if not ISIN_RE.fullmatch(isin):
            continue

        rec = {
            "isin": isin,
            "issuer_name": row[1] if len(row) > 1 else pd.NA,
            "section": section,
        }

        if kind == "BREACH":
            rec["activation_date"] = pd.to_datetime(
                row[5] if len(row) > 5 else pd.NA,
                dayfirst=True,
                errors="coerce",
            )

        else:
            rec["aggregate_investment_pct"] = parse_num(
                row[5] if len(row) > 5 else pd.NA
            )
            rec["permissible_limit_pct"] = parse_num(
                row[6] if len(row) > 6 else pd.NA
            )
            rec["available_headroom_qty"] = parse_num(
                row[7] if len(row) > 7 else pd.NA
            )
            rec["activation_date"] = pd.to_datetime(
                row[8] if len(row) > 8 else pd.NA,
                dayfirst=True,
                errors="coerce",
            )

        out.append(rec)

    return pd.DataFrame(out), parse_dissemination_date(text)


def load_alert_files():
    files = [
        p for p in RAW_DIR.glob("*.csv")
        if p.name.lower().startswith(
            ("breach_list_", "red_flag_list_")
        )
    ]

    rows = []

    for p in files:
        kind = (
            "BREACH"
            if p.name.lower().startswith("breach_list_")
            else "RED_FLAG"
        )

        snapshot = file_date(p)
        review = review_from_date(snapshot)

        if review not in REVIEW_ORDER:
            continue

        entries, dissemination = parse_alert_file(
            p,
            kind,
        )

        rows.append({
            "review": review,
            "kind": kind,
            "snapshot_date": snapshot,
            "dissemination_date": dissemination,
            "filename": p.name,
            "entries": entries,
        })

    meta = pd.DataFrame(rows)

    if len(meta) != 26:
        raise RuntimeError(
            f"Expected 26 review alert files, found {len(meta)}."
        )

    counts = (
        meta.groupby(["review", "kind"])
        .size()
    )

    missing = []

    for review in REVIEW_ORDER:
        for kind in ["RED_FLAG", "BREACH"]:
            if counts.get((review, kind), 0) != 1:
                missing.append(
                    f"{review}/{kind}: "
                    f"{counts.get((review, kind), 0)} files"
                )

    if missing:
        raise RuntimeError(
            "Alert file coverage problem:\n"
            + "\n".join(missing)
        )

    out = {}

    for review in REVIEW_ORDER:
        red = meta[
            meta["review"].eq(review)
            & meta["kind"].eq("RED_FLAG")
        ].iloc[0]

        breach = meta[
            meta["review"].eq(review)
            & meta["kind"].eq("BREACH")
        ].iloc[0]

        if red["snapshot_date"] != breach["snapshot_date"]:
            raise RuntimeError(
                f"{review}: Red Flag/Breach snapshot dates differ."
            )

        red_df = red["entries"]
        breach_df = breach["entries"]

        red_sections = (
            red_df.groupby("isin")["section"]
            .agg(
                lambda s: ";".join(
                    sorted(
                        set(
                            s.dropna().astype(str)
                        )
                    )
                )
            )
            .to_dict()
            if not red_df.empty
            else {}
        )

        breach_sections = (
            breach_df.groupby("isin")["section"]
            .agg(
                lambda s: ";".join(
                    sorted(
                        set(
                            s.dropna().astype(str)
                        )
                    )
                )
            )
            .to_dict()
            if not breach_df.empty
            else {}
        )

        out[review] = {
            "snapshot_date": red["snapshot_date"],
            "red_file": red["filename"],
            "breach_file": breach["filename"],
            "red_isins": set(red_df["isin"]) if not red_df.empty else set(),
            "breach_isins": set(breach_df["isin"]) if not breach_df.empty else set(),
            "red_sections": red_sections,
            "breach_sections": breach_sections,
            "red_rows": len(red_df),
            "breach_rows": len(breach_df),
        }

    return out


def load_security_isins():
    m = pd.read_parquet(MASTER_PATH)

    required = {"security_id", "isin"}
    missing = required - set(m.columns)

    if missing:
        raise RuntimeError(
            "Historical security master missing: "
            + ", ".join(sorted(missing))
        )

    m["security_id"] = m["security_id"].astype("string")

    m["isin"] = (
        m["isin"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    m = m[
        m["isin"].notna()
        & m["isin"].str.match(ISIN_RE, na=False)
    ]

    return {
        str(sid): tuple(
            dict.fromkeys(
                g["isin"].astype(str)
            )
        )
        for sid, g in m.groupby(
            "security_id",
            sort=False,
        )
    }


def main():
    x = pd.read_parquet(FOREIGN_ROOM_PATH)

    x["review"] = x["review"].astype(str)
    x["security_id"] = x["security_id"].astype("string")

    x["foreign_ownership_cutoff_date"] = pd.to_datetime(
        x["foreign_ownership_cutoff_date"],
        errors="coerce",
    )

    if x.duplicated(
        ["review", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate foreign-room keys."
        )

    alerts = load_alert_files()
    isin_map = load_security_isins()

    for review in REVIEW_ORDER:
        dates = (
            x.loc[
                x["review"].eq(review),
                "foreign_ownership_cutoff_date",
            ]
            .dropna()
            .drop_duplicates()
        )

        if len(dates) != 1:
            raise RuntimeError(
                f"{review}: expected one Price Cutoff, found {len(dates)}."
            )

        if alerts[review]["snapshot_date"] > dates.iloc[0]:
            raise RuntimeError(
                f"{review}: alert snapshot "
                f"{alerts[review]['snapshot_date'].date()} "
                f"is after Price Cutoff {dates.iloc[0].date()}."
            )

    x["india_red_flag"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    x["india_breach"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    x["india_alert_is_exact"] = False
    x["india_alert_snapshot_date"] = pd.NaT
    x["india_alert_source"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_red_flag_file"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_breach_file"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_red_flag_limit_type"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_breach_limit_type"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_alert_matched_isin"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    x["india_alert_has_isin_mapping"] = False

    for review in REVIEW_ORDER:
        info = alerts[review]

        for idx in x.index[
            x["review"].eq(review)
        ]:
            sid = str(x.at[idx, "security_id"])
            isins = isin_map.get(sid, ())

            if not isins:
                continue

            x.at[
                idx,
                "india_alert_has_isin_mapping",
            ] = True

            red_hits = [
                isin for isin in isins
                if isin in info["red_isins"]
            ]

            breach_hits = [
                isin for isin in isins
                if isin in info["breach_isins"]
            ]

            x.at[idx, "india_red_flag"] = bool(red_hits)
            x.at[idx, "india_breach"] = bool(breach_hits)
            x.at[idx, "india_alert_is_exact"] = True

            x.at[
                idx,
                "india_alert_snapshot_date",
            ] = info["snapshot_date"]

            x.at[
                idx,
                "india_alert_source",
            ] = "NSDL_CDSL_OFFICIAL_LIST_SNAPSHOT"

            x.at[
                idx,
                "india_red_flag_file",
            ] = info["red_file"]

            x.at[
                idx,
                "india_breach_file",
            ] = info["breach_file"]

            if red_hits:
                x.at[
                    idx,
                    "india_red_flag_limit_type",
                ] = ";".join(
                    sorted({
                        info["red_sections"][i]
                        for i in red_hits
                    })
                )

            if breach_hits:
                x.at[
                    idx,
                    "india_breach_limit_type",
                ] = ";".join(
                    sorted({
                        info["breach_sections"][i]
                        for i in breach_hits
                    })
                )

            hits = list(
                dict.fromkeys(
                    red_hits + breach_hits
                )
            )

            if hits:
                x.at[
                    idx,
                    "india_alert_matched_isin",
                ] = ";".join(hits)

    exact = x["india_alert_is_exact"]

    red = (
        x["india_red_flag"]
        .fillna(False)
        .astype(bool)
    )

    breach = (
        x["india_breach"]
        .fillna(False)
        .astype(bool)
    )

    room = pd.to_numeric(
        x["foreign_room"],
        errors="coerce",
    )

    lower = pd.to_numeric(
        x["foreign_room_lower_bound"],
        errors="coerce",
    )

    # Conditional result if NOT already in MIEU.
    x[
        "india_new_entry_alert_pass_if_nonconstituent"
    ] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    x.loc[
        exact,
        "india_new_entry_alert_pass_if_nonconstituent",
    ] = ~(red | breach)

    # Conditional result if already an MIEU constituent.
    x[
        "india_existing_alert_pass_if_constituent"
    ] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    no_alert = exact & ~red & ~breach

    x.loc[
        no_alert,
        "india_existing_alert_pass_if_constituent",
    ] = True

    x.loc[
        exact & breach,
        "india_existing_alert_pass_if_constituent",
    ] = False

    red_only = (
        exact
        & red
        & ~breach
    )

    red_exact_low = (
        red_only
        & room.notna()
        & room.lt(.0375)
    )

    red_exact_safe = (
        red_only
        & room.notna()
        & room.ge(.0375)
    )

    red_bound_safe = (
        red_only
        & room.isna()
        & lower.notna()
        & lower.ge(.0375)
    )

    x.loc[
        red_exact_low,
        "india_existing_alert_pass_if_constituent",
    ] = False

    x.loc[
        red_exact_safe | red_bound_safe,
        "india_existing_alert_pass_if_constituent",
    ] = True

    red_room_unresolved = (
        red_only
        & x[
            "india_existing_alert_pass_if_constituent"
        ].isna()
    )

    x["india_red_flag_breach_status"] = (
        "NO_ISIN_FOR_ALERT_MATCH"
    )

    x.loc[
        exact,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_NO_ALERT"

    x.loc[
        exact & breach,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_BREACH"

    x.loc[
        red_exact_low,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_RED_FLAG_ROOM_LT_3_75"

    x.loc[
        red_exact_safe,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_RED_FLAG_ROOM_GE_3_75"

    x.loc[
        red_bound_safe,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_RED_FLAG_ROOM_GE_3_75_PROVEN_BOUND"

    x.loc[
        red_room_unresolved,
        "india_red_flag_breach_status",
    ] = "EXACT_PIT_RED_FLAG_ROOM_UNRESOLVED"

    # Actual result remains pending until prior-MIEU state is known.
    x["india_red_flag_breach_pass"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    x["india_red_flag_breach_fail"] = pd.Series(
        pd.NA,
        index=x.index,
        dtype="boolean",
    )

    x[
        "india_red_flag_breach_prior_imi_state_pending"
    ] = True

    x["india_red_flag_breach_pending"] = ~exact

    x["india_existing_alert_room_state_pending"] = (
        red_room_unresolved
    )

    x["india_red_flag_breach_methodology_note"] = (
        "Exact NSDL/CDSL snapshot. New MIEU security fails "
        "if present on Red Flag or Breach list. Existing IMI "
        "constituent fails on Breach, or on Red Flag when "
        "foreign room is below 3.75%."
    )

    alert_lookahead = (
        x["india_alert_snapshot_date"].notna()
        & x["foreign_ownership_cutoff_date"].notna()
        & x["india_alert_snapshot_date"].gt(
            x["foreign_ownership_cutoff_date"]
        )
    )

    if alert_lookahead.any():
        raise RuntimeError(
            f"India alert lookahead: "
            f"{int(alert_lookahead.sum())}"
        )

    if x.duplicated(
        ["review", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate keys after India alert stage."
        )

    x.to_parquet(
        FOREIGN_ROOM_PATH,
        index=False,
    )

    print(
        "\n--- INDIA RED FLAG / BREACH EXACT PIT ---"
    )

    file_diag = []

    for review in REVIEW_ORDER:
        info = alerts[review]

        file_diag.append({
            "review": review,
            "snapshot": info["snapshot_date"].date(),
            "red_flag_list_rows": info["red_rows"],
            "breach_list_rows": info["breach_rows"],
        })

    print(
        pd.DataFrame(file_diag)
        .set_index("review")
        .to_string()
    )

    print("\nRows:", len(x))
    print(
        "Exact alert-status rows:",
        int(exact.sum()),
    )
    print(
        "Rows missing ISIN mapping:",
        int(
            (~x["india_alert_has_isin_mapping"]).sum()
        ),
    )
    print(
        "Securities missing ISIN mapping:",
        x.loc[
            ~x["india_alert_has_isin_mapping"],
            "security_id",
        ].nunique(),
    )

    print(
        "\nCandidate Red Flag rows:",
        int((exact & red).sum()),
    )
    print(
        "Candidate Breach rows:",
        int((exact & breach).sum()),
    )
    print(
        "Candidate both Red Flag + Breach:",
        int((exact & red & breach).sum()),
    )

    print("\n--- IF NON-CONSTITUENT ---")
    print(
        "Alert pass:",
        int(
            x[
                "india_new_entry_alert_pass_if_nonconstituent"
            ].eq(True).sum()
        ),
    )
    print(
        "Alert fail:",
        int(
            x[
                "india_new_entry_alert_pass_if_nonconstituent"
            ].eq(False).sum()
        ),
    )
    print(
        "Alert unresolved:",
        int(
            x[
                "india_new_entry_alert_pass_if_nonconstituent"
            ].isna().sum()
        ),
    )

    print("\n--- IF EXISTING IMI ---")
    print(
        "Alert pass:",
        int(
            x[
                "india_existing_alert_pass_if_constituent"
            ].eq(True).sum()
        ),
    )
    print(
        "Alert fail:",
        int(
            x[
                "india_existing_alert_pass_if_constituent"
            ].eq(False).sum()
        ),
    )
    print(
        "Alert unresolved:",
        int(
            x[
                "india_existing_alert_pass_if_constituent"
            ].isna().sum()
        ),
    )

    print(
        "Red Flag rows with room still unresolved:",
        int(red_room_unresolved.sum()),
    )

    print("\nStatus:")
    print(
        x[
            "india_red_flag_breach_status"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nFinal prior-IMI decisions made here:", 0)
    print("Room<3.75 used as fake alert:", 0)
    print(
        "Alert lookahead:",
        int(alert_lookahead.sum()),
    )
    print(
        "Duplicates:",
        int(
            x.duplicated(
                ["review", "security_id"]
            ).sum()
        ),
    )
    print("Saved:", FOREIGN_ROOM_PATH)


if __name__ == "__main__":
    main()