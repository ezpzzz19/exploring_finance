# Look-Ahead Bias Audit

## Summary

A look-ahead bias was identified in the WRDS feature merge step of `download_wrds_ratios.py`.
The bug inflated OOS R² from a realistic ~0.1–0.5% to **10–19%**, and Sharpe ratios from
a realistic 0.5–2.0 to **10+**. The fix is a 2-month lag on all WRDS features before merging.

---

## Root Cause

### JKP Dataset Convention (`mma_sample_v2.csv`)

The Jensen, Kelly & Pedersen (2022) dataset uses this timing:

| Row fields | Meaning |
|---|---|
| `year=2010, month=1, date=2010-01-29` | Portfolio formation at end of Jan 2010 |
| Characteristics (147 `stock_vars`) | Measured at **end of Dec 2009** (lagged 1 month) |
| `stock_exret` | Realized excess return **during** Jan 2010 |

So when we predict `stock_exret` for `(year=2010, month=1)`, all characteristics in that
row are from **before** Jan 2010. This is correct — no look-ahead.

### The Bug: WRDS Features Merged Without Lag

In `download_wrds_ratios.py`, WRDS data was merged on `(permno, year, month)` directly:

```python
# BUG: no lag applied
merged = merged.merge(ratios, on=["permno", "year", "month"], how="left")
```

This means:
- Financial ratio with `public_date = 2010-01-15` → `year=2010, month=1`
- Merged into JKP row `(year=2010, month=1)` where `stock_exret` = Jan 2010 return
- **Problem**: the ratio uses price data from Jan 2010, which already embeds the return

| WRDS Source | Timestamp | Available when? | Was matched to | Bias? |
|---|---|---|---|---|
| Financial Ratios | `public_date` Jan 15 | Mid-Jan | Jan row (predicts Jan return) | ✅ **Yes** |
| IBES Consensus | `statpers` Jan 2010 | During Jan | Jan row | ✅ **Yes** |
| Inst. Ownership | Q4 2019 13F filing | ~Feb 15, 2020 | Dec 2019 row | ✅ **Yes** |
| Short Interest | Mid-Jan report | ~2-week delay | Jan row | ✅ **Yes** |

### Evidence

| Metric | With bug | Realistic range |
|---|---|---|
| OOS R² (ensemble) | 17.3% | 0.1–0.5% |
| OOS R² (autoencoder) | 19.1% | 0.1–0.5% |
| Annualized Sharpe (L/S) | 10.4 | 0.5–2.0 |
| Monthly decile spread | 16.7% | 0.5–1.5% |
| Correlation(pred, realized) | 0.42 | 0.02–0.05 |

Top XGBoost feature = `io_num_holders` (a WRDS feature), confirming the model
was exploiting the contemporaneous information.

Cross-check with Ridge regression on a single test year:
- **JKP features only**: R² = -3.4%, corr = 0.02  ← realistic
- **JKP + WRDS (no lag)**: R² = -1.1%, corr = 0.14  ← WRDS adds suspicious signal

### Why `ret_1_0` Is NOT the Problem

`ret_1_0` (1-month lagged return) has 0.9999 correlation with `stock_exret` from the
**previous** row — i.e. it is last month's return, correctly lagged. The JKP dataset
already handles this correctly.

---

## Fix Applied

In `download_wrds_ratios.py`, all WRDS features are now lagged by **2 months** before merging:

```python
def lag_wrds(wrds_df, lag_months=2):
    """Shift WRDS data forward by lag_months so it aligns with JKP timing.
    E.g. WRDS data from Jan 2010 → assigned to Mar 2010 row."""
    d = wrds_df.copy()
    d["_dt"] = pd.to_datetime(d["year"].astype(str) + "-" + d["month"].astype(str) + "-01")
    d["_dt"] = d["_dt"] + pd.DateOffset(months=lag_months)
    d["year"] = d["_dt"].dt.year
    d["month"] = d["_dt"].dt.month
    d.drop(columns=["_dt"], inplace=True)
    return d
```

Why 2 months:
1. **+1 month**: JKP convention (chars at end of month t-1 used for month t predictions)
2. **+1 month**: publication delay (financial ratios, IBES, 13F filings all have
   reporting lags of 2–6 weeks)

This is conservative and standard in the empirical asset pricing literature.

---

## Steps to Reproduce the Fix

```bash
# 1. Re-run the WRDS download (now with 2-month lag)
cd individual_assignment
python download_wrds_ratios.py

# 2. Re-run the ML pipeline
python main.py

# 3. Re-run portfolio construction
python build_portfolio.py
```

After the fix, expect:
- OOS R² ≈ 0.1–0.5%  (maybe slightly negative for some models)
- Sharpe ≈ 0.5–2.0
- Monthly decile spread ≈ 0.5–1.5%

These are consistent with published results in Gu, Kelly & Xiu (2020) and
Jensen, Kelly & Pedersen (2022).

---

## Lesson

> Always verify that the **timestamp of every predictor** is strictly **before** the
> start of the return period being predicted. When merging external datasets, account
> for both the dataset's own timing convention AND the target dataset's convention.
> A simple same-key merge (`on=["permno","year","month"]`) is almost always wrong
> when combining datasets from different sources.
