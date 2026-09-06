# MSCI India Standard Index Rebalancing Prediction Engine

A public-data, point-in-time research engine for predicting **MSCI India Standard Index additions and deletions** ahead of quarterly review announcements.

The project reconstructs the main MSCI Global Investable Market Indexes (GIMI) review logic for India, including investability, size segmentation, liquidity, free float, foreign ownership and review-state maintenance. A small walk-forward model is then used to rank plausible **addition** candidates where the deterministic methodology alone is too conservative.

The historical study covers quarterly reviews from **May 2023 to May 2026**. May 2023 is used only to initialize historical state; scored performance begins in August 2023.

## Results

| Side | Final calls | Correct | Actual events | Precision | Recall | F1 | Methodology-only F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additions | 65 | 39 | 65 | 60.0% | 60.0% | **60.0%** | 45.1% |
| Deletions | 17 | 7 | 15 | 41.2% | 46.7% | **43.7%** | 43.7% |

The addition layer improves F1 by **14.9 percentage points** over the methodology-only baseline. Deletions remain methodology-only because learned deletion ranking and count models did not improve historical performance.

The addition result is a strict walk-forward historical backtest, but not a pristine untouched holdout: the final feature specification was selected during model development. Future reviews provide the cleanest out-of-sample validation.

---

## Project objective

The engine asks a simple question:

> Given only information that would have been available before an MSCI quarterly review announcement, which Indian securities were most likely to enter or leave the MSCI India Standard Index?

This is harder than ranking stocks by market capitalization. MSCI Standard membership depends on a sequence of interacting rules involving:

- the eligible equity universe;
- company and security market capitalization;
- free float and FIF;
- liquidity;
- foreign ownership limits and foreign room;
- minimum trading history;
- global and local size thresholds;
- existing-constituent buffers;
- prior index state;
- review-specific methodology changes.

The project therefore treats index rebalancing as a **state-reconstruction problem first** and a prediction problem second.

---

## Scope

The current implementation focuses on:

- **India / Emerging Markets**
- **MSCI India Standard**
- February, May, August and November reviews
- NSE **EQ-series** ordinary shares
- additions and deletions
- historical reviews from 2023-05 through 2026-05
- public and free data only

The project does not attempt to reproduce every MSCI rule globally. BSE-only securities, foreign listings, detailed GICS construction, Micro Cap methodology and exhaustive corporate-event handling are outside the current scope.

---

## How the engine works

```text
Historical NSE universe
        │
        ▼
Prices + shares + FX
        │
        ▼
Market capitalization
        │
        ├───────────────┐
        ▼               ▼
Review FIF          Liquidity
        │               │
        └───────┬───────┘
                ▼
      Foreign ownership / FOL
                │
                ▼
           Foreign room
                │
                ▼
         Investability state
                │
                ▼
        Recursive MIEU state
                │
                ▼
    Standard size-segment scenarios
                │
                ▼
     Assignment + final requirements
                │
                ▼
       Predicted methodology changes
                │
        ┌───────┴────────┐
        ▼                ▼
  ADD candidate      methodology
     ranking           DELETEs
        │                │
        └───────┬────────┘
                ▼
          Final decisions
```

### Historical security universe

The engine reconstructs the NSE equity universe historically rather than starting from today's listed securities.

A permanent internal `security_id` is used as the security identity; ticker symbols are treated as display fields. This matters for delisted, merged, renamed or replaced securities and avoids survivorship bias.

Historical identities such as `HIST_HDFC` and `HIST_TATAMTRDVR` are preserved so that past index membership and market data remain attached to the correct economic security.

### Point-in-time market data

Prices, shares outstanding and FX are combined to construct full market capitalization and free-float-adjusted market capitalization.

The project distinguishes three review dates:

- **Equity Universe Cutoff**
- **Liquidity Cutoff**
- **Price Cutoff**

These dates serve different methodology functions and are resolved separately for each historical review. Review-specific dates are recorded in `references/reviews.yaml`.

### Free float and FIF

Free float is reconstructed from public shareholding information where possible and converted into review-specific FIF.

The FIF implementation is **methodology-vintage aware**. In particular:

- reviews before February 2026 use the applicable legacy review treatment;
- February 2026 is treated as a transition review;
- from May 2026, enhanced FIF rounding is applied:
  - above 25% free float: nearest 2.5 percentage points;
  - 5–25%: nearest 0.5 percentage points;
  - below 5%: nearest 0.1 percentage points.

Later methodology rules are not retroactively applied to earlier reviews.

### Liquidity

The liquidity layer reconstructs MSCI-style:

- 12-month ATVR;
- 3-month ATVR;
- 3-month frequency of trading;
- available-history fallback windows.

Monthly observations use the **actual final NSE trading day of the month**, which matters when the calendar month-end is not a trading day.

### Foreign ownership, FOL and foreign room

India requires additional treatment because foreign ownership restrictions can affect investability and FIF.

The engine separates:

- historical foreign ownership;
- foreign ownership limits (FOL);
- foreign-room evidence.

Historical FOL evidence is not always an exact point value. The system therefore distinguishes between:

- exact/reconstructed FOL;
- a policy lower bound;
- unresolved FOL.

A lower bound is used only to establish threshold conditions; it is not converted into a false exact foreign-room estimate.

The project also incorporates historical NSDL/CDSL information and India Red Flag/Breach evidence where available.

### Recursive MIEU reconstruction

MSCI review logic is stateful: treatment depends on whether a security was already investable in the previous review.

The engine reconstructs the **Market Investable Equity Universe (MIEU)** review by review. The previous review's reconstructed post-review state becomes the starting point for the next review.

May 2023 is a special burn-in review because exact February 2023 MIEU/Small Cap membership was not fully recoverable. A point-in-time proxy is used only for this initialization. From August 2023 onward, MIEU state is fully recursive within the engine.

Each security can remain:

```text
PASS
FAIL
UNRESOLVED
```

when historical evidence is insufficient to make a defensible point-in-time decision.

### Standard size segmentation

The updated MIEU is used to reconstruct the Standard size segment using:

- company full market capitalization;
- free-float-adjusted market coverage;
- global size reference estimates;
- Standard coverage targets;
- prior segment state;
- MSCI size buffers.

Where exact historical size-reference inputs cannot be recovered, the project evaluates **eight methodology scenarios per review** rather than forcing a single retrospective estimate.

The scenario engine represents uncertainty in historical inputs, not alternative outcomes chosen after seeing index changes.

### Assignment and final Standard requirements

After the Standard cutoff is determined, companies are assigned using MSCI-style priority and buffer logic.

Important boundaries include:

- Standard lower buffer: approximately `2/3 × Standard cutoff`;
- Small Cap upper buffer: approximately `1.5 × Standard cutoff`.

Final Standard-specific checks then incorporate items such as:

- minimum free-float-adjusted market capitalization;
- existing-constituent retention treatment;
- low-FIF treatment;
- foreign-room adjustment;
- Red Flag/Breach restrictions;
- extreme price increase rules.

The output is a scenario-level estimate of post-review Standard membership.

---

## Prediction layer

The methodology reconstruction itself produces definite and possible actions such as:

```text
ADD
DELETE
KEEP
ADD_OR_NONE
KEEP_OR_DELETE
```

The methodology-only layer is relatively conservative for additions: it achieves high precision on definite calls but misses many actual additions that sit close to the reconstructed Standard boundary.

### Addition ranker

A small statistical model is therefore used to rank plausible addition candidates.

The final model is a walk-forward logistic regression using three cross-sectional signals:

```text
sig_ff_gap
sig_support
sig_mcap_gap
```

Conceptually these measure:

- **FF gap** — strength relative to the required free-float-adjusted market capitalization;
- **scenario support** — how consistently the security qualifies across legitimate methodology scenarios;
- **market-cap gap** — distance from the reconstructed Standard market-cap cutoff.

The model is trained only on **prior scored reviews**. Same-review outcomes never enter the prediction for that review.

The final number of additions is also determined without using the realized event count. From November 2023 onward, it is based on the median number of additions implied across the methodology scenarios.

### Deletions

A learned deletion layer was tested but rejected.

The scored window contains only 15 actual deletion events, and both learned deletion ranking and learned event-count estimation underperformed the methodology-only approach. The final engine therefore uses definite methodology DELETE signals directly.

---

## Backtest interpretation

The methodology-only addition baseline produces:

| Calls | Correct | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 37 | 23 | 62.2% | 35.4% | 45.1% |

The final addition layer produces:

| Calls | Correct | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 65 | 39 | 60.0% | 60.0% | **60.0%** |

The improvement comes from converting the broader methodology candidate/scenario information into a ranked set of live-style calls, rather than relying only on unanimous methodology outcomes.

Performance varies across reviews. The model performs strongly in several 2023–2024 reviews and more weakly in parts of late 2024 and 2025, suggesting that the relationship between size, free float and realized MSCI decisions is not perfectly stable through time.

---

## Point-in-time integrity

Avoiding historical leakage is central to the project.

Key design choices include:

- historical rather than current security universes;
- permanent security identities;
- review-specific cutoffs;
- event-dated share counts;
- no future ownership/FOL backfill;
- recursive prior MIEU state;
- no same-review labels in model features;
- no realized addition/deletion count in final decisions;
- explicit unresolved states when historical evidence is incomplete.

This is also why the project sometimes accepts uncertainty rather than retrospectively forcing a perfect reconstruction.

---

## Public data sources

The project uses no Bloomberg or paid historical MSCI constituent feed.

Main sources include:

- NSE security master and historical listing data;
- NSE corporate shareholding and corporate-action data;
- Yahoo Finance price history;
- NSE archives for selected historical/delisted names;
- NSE/public company shareholding disclosures;
- NSDL and CDSL foreign-ownership/FOL information;
- India Red Flag and Breach lists;
- public MSCI Standard review PDFs;
- public market data used for FX and global-size-reference estimation.

Raw source files are retained under `data/raw/`, while reconstructed methodology states and model outputs are stored under `data/processed/`.

---

## Important limitations

### Public-data reconstruction

MSCI has access to internal data, classifications and case-by-case judgments that are not fully observable from public sources. This project is therefore a **public-data reconstruction of the methodology**, not an exact replica of MSCI's internal production system.

### NSE EQ-only universe

The universe intentionally excludes BSE-only securities, non-EQ NSE series, REIT/RR securities and foreign listings.

### Foreign-room reconstruction

India's full FPI/FDI dual-headroom structure is not completely recoverable historically from the available public evidence. Exact values, threshold-proving bounds and unresolved states are therefore kept distinct.

### Historical FOL

For many observations, only a lower bound on FOL can be established. These bounds are useful for threshold tests but cannot support an exact foreign-room calculation.

### Global size references

Exact historical MSCI global size reference inputs are not always publicly available. The engine uses public estimates and scenario ranges where required.

### Corporate events

Targeted corporate-event repairs are included where necessary for historically material names, but the project does not implement the complete MSCI Corporate Events Methodology.

### February 2026 FIF transition

The February 2026 transition is reconstructed from available public evidence but is not exhaustive for every historical FOL or adjustment-event decrease MSCI may have reflected.

### Backtest selection

The final addition model is walk-forward at prediction time, but its feature specification was finalized during historical development. The 60.0% F1 is therefore best interpreted as the final historical research result rather than a fully untouched holdout estimate.

---

## Repository overview

```text
backtest/      historical labels, evaluation and error analysis
ingestion/     market, prices, shares, FIF, liquidity and ownership reconstruction
methodology/   investability, MIEU and Standard review methodology
model/         predicted changes, candidate ranking and final decisions
data/raw/      original public data and historical source documents
data/processed/ reconstructed states and model outputs
data/backtest/ persisted backtest summaries
references/    methodology and review-specific reference configuration
```

Important final outputs include:

- `data/processed/mieu_review_state.parquet`
- `data/processed/standard_size_segment.parquet`
- `data/processed/standard_review_state.parquet`
- `data/processed/predicted_changes.parquet`
- `data/processed/candidate_rankings.parquet`
- `data/processed/final_decisions.parquet`
- `data/backtest/final_backtest_summary.csv`
- `data/backtest/final_backtest_by_review.csv`

`references/methodology.yaml` documents the implemented methodology and known approximations, while `references/reviews.yaml` records the review-specific historical cutoffs and methodology vintages.

---

## Summary

The project combines three layers:

1. **historical point-in-time data reconstruction**;
2. **an interpretable MSCI methodology engine**;
3. **a small walk-forward ranking overlay for additions**.

The result is a research framework that can explain *why* a security is close to MSCI Standard inclusion, not just assign it a probability.

Across the August 2023–May 2026 scored window, the final engine identifies **39 of 65 additions** with 60.0% precision and recall, compared with a 45.1% F1 methodology-only addition baseline. Deletions remain methodology-driven, reflecting both the smaller sample and the stronger performance of the deterministic rules on that side.
