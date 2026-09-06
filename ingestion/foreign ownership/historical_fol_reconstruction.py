from pathlib import Path
import io
import re
import time

import numpy as np
import pandas as pd
import requests


TARGET_PATH = Path("data/processed/historical_fol_targets.parquet")
QUEUE_PATH = Path("data/processed/historical_fol_research_queue.parquet")
OLD_FOL_PATH = Path("data/processed/historical_fol.parquet")
EVENTS_PATH = Path("data/processed/historical_fol_events.parquet")

CHECKPOINT_PATH = Path(
    "data/raw/foreign_ownership/"
    "historical_fol_announcement_checkpoint.parquet"
)

STAGE_PATH = Path(
    "data/processed/"
    "historical_fol_announcement_stage.parquet"
)

NSE_HOME = "https://www.nseindia.com/"
NSE_PAGE = (
    "https://www.nseindia.com/"
    "companies-listing/corporate-filings-announcements"
)
NSE_API = (
    "https://www.nseindia.com/api/"
    "corporate-announcements"
)

START_DATE = pd.Timestamp("2021-01-01")

KEYWORDS = [
    "foreign investment",
    "foreign shareholding",
    "foreign ownership",
    "foreign institutional",
    "foreign portfolio",
    "fpi limit",
    "fii limit",
    "sectoral cap",
    "sectoral limit",
    "board approved limit",
    "permissible limit",
    "aggregate foreign",
    "foreign investment limit",
    "foreign holding limit",
    "nri limit",
    "rbi approval",
    "government approval",
]

LIMIT_PATTERNS = {
    "fpi_limit": [
        r"(?:fpi|foreign portfolio(?: investor)?s?).{0,100}?(?:limit|cap).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(?:limit|cap).{0,50}?(?:fpi|foreign portfolio(?: investor)?s?).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
    ],
    "foreign_investment_limit": [
        r"(?:foreign investment|foreign shareholding|foreign ownership|aggregate foreign).{0,100}?(?:limit|cap|ceiling).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(?:limit|cap|ceiling).{0,50}?(?:foreign investment|foreign shareholding|foreign ownership|aggregate foreign).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
    ],
    "sectoral_cap": [
        r"sectoral\s+(?:cap|limit).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%.{0,50}?sectoral\s+(?:cap|limit)",
    ],
    "board_approved_limit": [
        r"board\s+approved\s+(?:limit|cap).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%.{0,50}?board\s+approved\s+(?:limit|cap)",
    ],
    "nri_limit": [
        r"nri\s+(?:limit|cap).{0,50}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%.{0,50}?nri\s+(?:limit|cap)",
    ],
}


def first_existing(df, names):
    return next((c for c in names if c in df.columns), None)


def norm_symbol(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().upper()
    return x or pd.NA


def parse_date(x):
    return pd.to_datetime(x, errors="coerce", dayfirst=True)


def create_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": NSE_PAGE,
    })
    s.get(NSE_HOME, timeout=20)
    s.get(NSE_PAGE, timeout=20)
    return s


def fetch_announcements(session, symbol, end_date):
    params = {
        "index": "equities",
        "symbol": symbol,
        "from_date": START_DATE.strftime("%d-%m-%Y"),
        "to_date": end_date.strftime("%d-%m-%Y"),
    }

    last_error = None

    for attempt in range(1, 5):
        try:
            r = session.get(NSE_API, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            last_error = e

            if attempt < 4:
                time.sleep(3 * attempt)
                try:
                    session.get(NSE_PAGE, timeout=20)
                except Exception:
                    pass

    raise RuntimeError(
        f"NSE announcement fetch failed for {symbol}: {last_error}"
    )


def announcement_text(row):
    return " ".join(
        str(row.get(c, ""))
        for c in ["desc", "attchmntText", "sm_name"]
        if pd.notna(row.get(c))
    )


def relevant_announcement(row):
    text = announcement_text(row).lower()
    return any(k in text for k in KEYWORDS)


def extract_pdf_text(content):
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(
                page.extract_text() or ""
                for page in pdf.pages
            )
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    except Exception:
        return ""


def fetch_attachment_text(session, url):
    if pd.isna(url) or not str(url).startswith("http"):
        return ""

    try:
        r = session.get(str(url), timeout=60)
        r.raise_for_status()
    except Exception:
        return ""

    ctype = r.headers.get("Content-Type", "").lower()
    url_lower = str(url).lower()

    if "pdf" in ctype or url_lower.endswith(".pdf"):
        return extract_pdf_text(r.content)

    try:
        return r.text
    except Exception:
        return ""


def pct(v):
    if pd.isna(v):
        return np.nan

    v = float(v)
    if v < 0 or v > 100:
        return np.nan

    return v / 100.0


def first_match(text, patterns):
    text = str(text).replace("\n", " ")

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m:
            try:
                return pct(float(m.group(1)))
            except Exception:
                pass

    return np.nan


def extract_limits(text):
    out = {
        name: first_match(text, patterns)
        for name, patterns in LIMIT_PATTERNS.items()
    }

    usable = [
        out["fpi_limit"],
        out["foreign_investment_limit"],
        out["sectoral_cap"],
        out["board_approved_limit"],
    ]

    usable = [
        x for x in usable
        if pd.notna(x) and x > 0
    ]

    out["historical_fol"] = (
        min(usable)
        if usable
        else np.nan
    )

    return out


def event_provenance(limits):
    if pd.isna(limits["historical_fol"]):
        return "UNRESOLVED"

    if pd.notna(limits["board_approved_limit"]):
        return "EVENT_RECONSTRUCTED"

    return "POLICY_RECONSTRUCTED"


def load_old_fol():
    old = pd.read_parquet(OLD_FOL_PATH)
    old["review"] = old["review"].astype(str)
    old["security_id"] = old["security_id"].astype("string")

    fol_col = first_existing(
        old,
        ["historical_fol"],
    )

    avail_col = first_existing(
        old,
        [
            "historical_fol_available_date",
            "available_date",
        ],
    )

    effective_col = first_existing(
        old,
        [
            "historical_fol_effective_date",
            "snapshot_date",
            "announcement_date",
        ],
    )

    source_col = first_existing(
        old,
        ["historical_fol_source", "source"],
    )

    provenance_col = first_existing(
        old,
        ["historical_fol_provenance", "provenance"],
    )

    status_col = first_existing(
        old,
        ["historical_fol_status", "status"],
    )

    if fol_col is None:
        raise RuntimeError(
            "Old historical_fol.parquet has no historical_fol."
        )

    out = old[["review", "security_id"]].copy()

    out["historical_fol"] = pd.to_numeric(
        old[fol_col],
        errors="coerce",
    )

    out["historical_fol_available_date"] = (
        pd.to_datetime(
            old[avail_col],
            errors="coerce",
        )
        if avail_col
        else pd.NaT
    )

    out["historical_fol_effective_date"] = (
        pd.to_datetime(
            old[effective_col],
            errors="coerce",
        )
        if effective_col
        else pd.NaT
    )

    out["historical_fol_source"] = (
        old[source_col]
        if source_col
        else pd.NA
    )

    out["historical_fol_provenance"] = (
        old[provenance_col]
        if provenance_col
        else pd.NA
    )

    out["historical_fol_status"] = (
        old[status_col]
        if status_col
        else pd.NA
    )

    return out


def load_checkpoint(end_date):
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame()

    x = pd.read_parquet(CHECKPOINT_PATH)
    x["security_id"] = x["security_id"].astype("string")

    for c in [
        "announcement_date",
        "available_date",
        "fetch_end_date",
    ]:
        if c in x.columns:
            x[c] = pd.to_datetime(
                x[c],
                errors="coerce",
            )

    return x[
        x["fetch_end_date"].eq(end_date)
    ].copy()


def save_checkpoint(x):
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x.to_parquet(
        CHECKPOINT_PATH,
        index=False,
    )


def fetch_new_events(securities, end_date):
    checkpoint = load_checkpoint(end_date)

    completed = set()

    if not checkpoint.empty:
        completed = set(
            checkpoint.loc[
                checkpoint["security_fetch_complete"]
                .fillna(False)
                .astype(bool),
                "security_id",
            ]
            .dropna()
            .astype(str)
        )

    todo = securities[
        ~securities["security_id"]
        .astype(str)
        .isin(completed)
    ].copy()

    rows = (
        checkpoint.to_dict("records")
        if not checkpoint.empty
        else []
    )

    session = create_session()

    print("\n--- HISTORICAL FOL ANNOUNCEMENT FETCH ---")
    print("Research securities:", len(securities))
    print("Already completed:", len(completed))
    print("To fetch:", len(todo))

    for i, (_, sec) in enumerate(
        todo.iterrows(),
        1,
    ):
        sid = sec["security_id"]
        symbol = sec["nse_symbol"]

        if pd.isna(symbol):
            rows.append({
                "security_id": sid,
                "nse_symbol": pd.NA,
                "announcement_date": pd.NaT,
                "available_date": pd.NaT,
                "subject": pd.NA,
                "announcement_text": pd.NA,
                "attachment_url": pd.NA,
                "fpi_limit": np.nan,
                "foreign_investment_limit": np.nan,
                "sectoral_cap": np.nan,
                "board_approved_limit": np.nan,
                "nri_limit": np.nan,
                "historical_fol": np.nan,
                "source": "NSE_ANNOUNCEMENT",
                "provenance": "UNRESOLVED",
                "status": "NO_CURRENT_NSE_SYMBOL",
                "error": pd.NA,
                "fetch_end_date": end_date,
                "security_fetch_complete": True,
            })
            continue

        try:
            anns = fetch_announcements(
                session,
                symbol,
                end_date,
            )

            relevant = [
                a for a in anns
                if relevant_announcement(a)
            ]

            if not relevant:
                rows.append({
                    "security_id": sid,
                    "nse_symbol": symbol,
                    "announcement_date": pd.NaT,
                    "available_date": pd.NaT,
                    "subject": pd.NA,
                    "announcement_text": pd.NA,
                    "attachment_url": pd.NA,
                    "fpi_limit": np.nan,
                    "foreign_investment_limit": np.nan,
                    "sectoral_cap": np.nan,
                    "board_approved_limit": np.nan,
                    "nri_limit": np.nan,
                    "historical_fol": np.nan,
                    "source": "NSE_ANNOUNCEMENT",
                    "provenance": "UNRESOLVED",
                    "status": "NO_RELEVANT_ANNOUNCEMENT",
                    "error": pd.NA,
                    "fetch_end_date": end_date,
                    "security_fetch_complete": True,
                })

            for j, a in enumerate(relevant):
                metadata = announcement_text(a)
                url = a.get(
                    "attchmntFile",
                    pd.NA,
                )

                attachment = fetch_attachment_text(
                    session,
                    url,
                )

                limits = extract_limits(
                    metadata
                    + "\n"
                    + attachment
                )

                an_date = parse_date(
                    a.get("an_dt")
                )

                if pd.isna(an_date):
                    an_date = parse_date(
                        a.get("sort_date")
                    )

                rows.append({
                    "security_id": sid,
                    "nse_symbol": symbol,
                    "announcement_date": an_date,
                    "available_date": an_date,
                    "subject": a.get(
                        "desc",
                        pd.NA,
                    ),
                    "announcement_text": a.get(
                        "attchmntText",
                        pd.NA,
                    ),
                    "attachment_url": url,
                    **limits,
                    "source": "NSE_ANNOUNCEMENT",
                    "provenance": event_provenance(
                        limits
                    ),
                    "status": (
                        "LIMIT_EXTRACTED"
                        if pd.notna(
                            limits["historical_fol"]
                        )
                        else
                        "RELEVANT_ANNOUNCEMENT_LIMIT_UNRESOLVED"
                    ),
                    "error": pd.NA,
                    "fetch_end_date": end_date,
                    "security_fetch_complete": (
                        j == len(relevant) - 1
                    ),
                })

                time.sleep(0.05)

        except Exception as e:
            rows.append({
                "security_id": sid,
                "nse_symbol": symbol,
                "announcement_date": pd.NaT,
                "available_date": pd.NaT,
                "subject": pd.NA,
                "announcement_text": pd.NA,
                "attachment_url": pd.NA,
                "fpi_limit": np.nan,
                "foreign_investment_limit": np.nan,
                "sectoral_cap": np.nan,
                "board_approved_limit": np.nan,
                "nri_limit": np.nan,
                "historical_fol": np.nan,
                "source": "NSE_ANNOUNCEMENT",
                "provenance": "UNRESOLVED",
                "status": "ANNOUNCEMENT_FETCH_FAILED",
                "error": str(e)[:500],
                "fetch_end_date": end_date,
                "security_fetch_complete": False,
            })

        if i % 25 == 0:
            save_checkpoint(
                pd.DataFrame(rows)
            )
            print(
                f"Processed: {i}/{len(todo)}"
            )

        time.sleep(0.10)

    out = pd.DataFrame(rows)

    for c in [
        "announcement_date",
        "available_date",
        "fetch_end_date",
    ]:
        out[c] = pd.to_datetime(
            out[c],
            errors="coerce",
        )

    out["security_id"] = (
        out["security_id"]
        .astype("string")
    )

    save_checkpoint(out)

    return out


def merge_event_cache(new_events):
    if EVENTS_PATH.exists():
        old = pd.read_parquet(
            EVENTS_PATH
        )

        old["security_id"] = (
            old["security_id"]
            .astype("string")
        )

        for c in [
            "announcement_date",
            "available_date",
        ]:
            if c in old.columns:
                old[c] = pd.to_datetime(
                    old[c],
                    errors="coerce",
                )

        old["event_origin"] = (
            "OLD_EVENT_CACHE"
        )
    else:
        old = pd.DataFrame()

    new = new_events.copy()
    new["event_origin"] = (
        "NEW_ANNOUNCEMENT_RESEARCH"
    )

    common = [
        "security_id",
        "nse_symbol",
        "announcement_date",
        "available_date",
        "subject",
        "announcement_text",
        "attachment_url",
        "fpi_limit",
        "foreign_investment_limit",
        "sectoral_cap",
        "board_approved_limit",
        "nri_limit",
        "historical_fol",
        "source",
        "provenance",
        "status",
        "error",
        "event_origin",
    ]

    for df in [old, new]:
        for c in common:
            if c not in df.columns:
                df[c] = pd.NA

    events = pd.concat(
        [
            old[common],
            new[common],
        ],
        ignore_index=True,
    )

    events["historical_fol"] = (
        pd.to_numeric(
            events["historical_fol"],
            errors="coerce",
        )
    )

    events = (
        events.sort_values(
            [
                "security_id",
                "available_date",
                "event_origin",
            ],
            na_position="last",
        )
        .drop_duplicates(
            [
                "security_id",
                "available_date",
                "attachment_url",
                "historical_fol",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    EVENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
        EVENTS_PATH,
        index=False,
    )

    return events


def protected_old(row):
    text = " ".join(
        str(row.get(c, ""))
        for c in [
            "historical_fol_source",
            "historical_fol_provenance",
            "historical_fol_status",
        ]
        if pd.notna(row.get(c))
    )

    return bool(
        re.search(
            r"MANUAL|SPECIAL|TARGETED|OVERRIDE",
            text,
            flags=re.IGNORECASE,
        )
    )


def build_stage(research, old_fol, events):
    old_map = {
        (r["review"], r["security_id"]): r
        for _, r in old_fol.iterrows()
        if pd.notna(
            r["historical_fol"]
        )
    }

    usable_events = events[
        events["historical_fol"].notna()
        & events["available_date"].notna()
    ].copy()

    grouped = {
        sid: g.sort_values(
            "available_date"
        )
        for sid, g in usable_events.groupby(
            "security_id",
            sort=False,
        )
    }

    rows = []

    for _, target in research.iterrows():
        key = (
            target["review"],
            target["security_id"],
        )

        sid = target["security_id"]
        cutoff = target[
            "foreign_ownership_cutoff_date"
        ]

        old = old_map.get(key)
        old_usable = (
            old is not None
            and (
                pd.isna(
                    old[
                        "historical_fol_available_date"
                    ]
                )
                or old[
                    "historical_fol_available_date"
                ] <= cutoff
            )
        )

        g = grouped.get(sid)
        chosen_event = None

        if g is not None:
            eligible = g[
                g["available_date"].le(
                    cutoff
                )
            ]

            if not eligible.empty:
                chosen_event = (
                    eligible.iloc[-1]
                )

        use_old = False
        use_event = False

        if old_usable:
            if protected_old(old):
                use_old = True

            elif chosen_event is None:
                use_old = True

            else:
                old_date = old[
                    "historical_fol_available_date"
                ]

                event_date = chosen_event[
                    "available_date"
                ]

                if (
                    pd.isna(old_date)
                    or event_date > old_date
                ):
                    use_event = True
                else:
                    use_old = True

        elif chosen_event is not None:
            use_event = True

        base = {
            "review": target["review"],
            "security_id": sid,
            "nse_symbol": target[
                "nse_symbol"
            ],
            "foreign_ownership_cutoff_date": cutoff,
        }

        if use_old:
            rows.append({
                **base,
                "historical_fol":
                    old["historical_fol"],
                "historical_fol_effective_date":
                    old["historical_fol_effective_date"],
                "historical_fol_available_date":
                    old["historical_fol_available_date"],
                "historical_fol_source":
                    old["historical_fol_source"],
                "historical_fol_provenance":
                    old["historical_fol_provenance"],
                "historical_fol_status":
                    old["historical_fol_status"],
                "stage_action":
                    "PRESERVE_OLD_RESOLVED",
            })

        elif use_event:
            rows.append({
                **base,
                "historical_fol":
                    chosen_event[
                        "historical_fol"
                    ],
                "historical_fol_effective_date":
                    chosen_event[
                        "announcement_date"
                    ],
                "historical_fol_available_date":
                    chosen_event[
                        "available_date"
                    ],
                "historical_fol_source":
                    chosen_event[
                        "source"
                    ],
                "historical_fol_provenance":
                    chosen_event[
                        "provenance"
                    ],
                "historical_fol_status":
                    "RESOLVED_EVENT",
                "stage_action":
                    "USE_ANNOUNCEMENT_EVENT",
            })

        else:
            rows.append({
                **base,
                "historical_fol":
                    np.nan,
                "historical_fol_effective_date":
                    pd.NaT,
                "historical_fol_available_date":
                    pd.NaT,
                "historical_fol_source":
                    pd.NA,
                "historical_fol_provenance":
                    "UNRESOLVED",
                "historical_fol_status":
                    "NO_PIT_FOL_EVIDENCE",
                "stage_action":
                    "UNRESOLVED",
            })

    out = pd.DataFrame(rows)

    lookahead = (
        out[
            "historical_fol_available_date"
        ].notna()
        & out[
            "historical_fol_available_date"
        ].gt(
            out[
                "foreign_ownership_cutoff_date"
            ]
        )
    )

    if lookahead.any():
        raise RuntimeError(
            "Historical FOL lookahead detected."
        )

    if out.duplicated(
        ["review", "security_id"]
    ).any():
        raise RuntimeError(
            "Duplicate FOL stage keys."
        )

    return out, lookahead


def main():
    targets = pd.read_parquet(
        TARGET_PATH
    )

    queue = pd.read_parquet(
        QUEUE_PATH
    )

    for df in [targets, queue]:
        df["security_id"] = (
            df["security_id"]
            .astype("string")
        )

    targets["review"] = (
        targets["review"]
        .astype(str)
    )

    targets[
        "foreign_ownership_cutoff_date"
    ] = pd.to_datetime(
        targets[
            "foreign_ownership_cutoff_date"
        ],
        errors="coerce",
    )

    research = targets[
        targets[
            "needs_historical_fol_research"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if research.empty:
        raise RuntimeError(
            "No FOL research rows."
        )

    old_fol = load_old_fol()

    old_resolved_keys = set(
        map(
            tuple,
            old_fol.loc[
                old_fol[
                    "historical_fol"
                ].notna(),
                [
                    "review",
                    "security_id",
                ],
            ]
            .itertuples(
                index=False,
                name=None,
            ),
        )
    )

    research["_old_resolved"] = [
        (
            review,
            sid,
        )
        in old_resolved_keys
        for review, sid
        in zip(
            research["review"],
            research["security_id"],
        )
    ]

    missing = research[
        ~research["_old_resolved"]
    ].copy()

    missing_ids = set(
        missing[
            "security_id"
        ].dropna()
    )

    securities = (
        queue[
            queue["security_id"]
            .isin(missing_ids)
        ][
            [
                "security_id",
                "nse_symbol",
            ]
        ]
        .drop_duplicates(
            "security_id"
        )
        .copy()
    )

    securities["nse_symbol"] = (
        securities["nse_symbol"]
        .map(norm_symbol)
    )

    expected_ids = set(
        research.loc[
            ~research[
                "_old_resolved"
            ],
            "security_id",
        ]
    )

    if set(
        securities["security_id"]
    ) != expected_ids:
        raise RuntimeError(
            "Missing-security set and queue "
            "security set do not match."
        )

    end_date = (
        research[
            "foreign_ownership_cutoff_date"
        ]
        .max()
        .normalize()
    )

    new_events = fetch_new_events(
        securities,
        end_date,
    )

    events = merge_event_cache(
        new_events,
    )

    stage, lookahead = build_stage(
        research,
        old_fol,
        events,
    )

    STAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage.to_parquet(
        STAGE_PATH,
        index=False,
    )

    fetch_failed = (
        new_events["status"]
        .eq(
            "ANNOUNCEMENT_FETCH_FAILED"
        )
    )

    completed = (
        new_events[
            "security_fetch_complete"
        ]
        .fillna(False)
        .astype(bool)
    )

    extracted = events[
        events[
            "historical_fol"
        ].notna()
    ]

    print(
        "\n--- HISTORICAL FOL ANNOUNCEMENT STAGE ---"
    )

    print(
        "Research rows:",
        len(research),
    )

    print(
        "Research securities:",
        research[
            "security_id"
        ].nunique(),
    )

    print(
        "Old-resolved research rows:",
        int(
            research[
                "_old_resolved"
            ].sum()
        ),
    )

    print(
        "Rows requiring new evidence:",
        len(missing),
    )

    print(
        "Securities requiring new research:",
        len(securities),
    )

    print(
        "Completed announcement fetches:",
        new_events.loc[
            completed,
            "security_id",
        ].nunique(),
    )

    print(
        "Announcement fetch failures:",
        new_events.loc[
            fetch_failed,
            "security_id",
        ].nunique(),
    )

    print(
        "Total extracted limit events:",
        len(extracted),
    )

    print(
        "Securities with extracted limit event:",
        extracted[
            "security_id"
        ].nunique(),
    )

    print(
        "\nStage action:"
    )

    print(
        stage[
            "stage_action"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nStage resolved rows:",
        int(
            stage[
                "historical_fol"
            ].notna()
            .sum()
        ),
    )

    print(
        "Stage unresolved rows:",
        int(
            stage[
                "historical_fol"
            ].isna()
            .sum()
        ),
    )

    print(
        "Stage resolved securities:",
        stage.loc[
            stage[
                "historical_fol"
            ].notna(),
            "security_id",
        ].nunique(),
    )

    print(
        "PIT lookahead:",
        int(lookahead.sum()),
    )

    print(
        "Stage duplicates:",
        int(
            stage.duplicated(
                [
                    "review",
                    "security_id",
                ]
            ).sum()
        ),
    )

    print(
        "\nSaved events:",
        EVENTS_PATH,
    )

    print(
        "Saved stage:",
        STAGE_PATH,
    )

    print(
        "Canonical historical_fol.parquet "
        "was NOT modified."
    )


if __name__ == "__main__":
    main()