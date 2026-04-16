"""
Download WRDS data and merge with mma_sample_v2.csv
=====================================================
Three data sources:
  1. Financial Ratios Suite  (van Binsbergen et al. 2023) - 39 ratios
  2. IBES Analyst Consensus  - earnings forecasts, dispersion, revisions
  3. Institutional Ownership  - quarterly 13F aggregates via Compustat
  4. Short Interest            - monthly short interest via Compustat
"""

import pandas as pd
import numpy as np
import wrds
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_we")
OUTPUT_PATH = os.path.join(DATA_DIR, "mma_sample_enhanced.csv")

# ── 1. Load existing data ────────────────────────────────────────────────────
print("=" * 60)
print("Loading mma_sample_v2.csv ...")
df = pd.read_csv(os.path.join(DATA_DIR, "mma_sample_v2.csv"), low_memory=False)
print(f"  Shape: {df.shape}")
existing_cols = set(df.columns.str.lower())
df["permno"] = df["permno"].astype(int)
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

print("Connecting to WRDS ...")
db = wrds.Connection(wrds_username='etremblay')

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 1: Financial Ratios Suite
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[1/4] Downloading Financial Ratios Suite ...")

WRDS_COLS = [
    "permno", "public_date",
    "capei", "evm", "pe_op_dil", "pe_exi", "ps", "pcf", "ptb",
    "peg_trailing", "divyield",
    "npm", "opmbd", "opmad", "gpm", "cfm", "roa", "roe", "roce",
    "efftax", "aftret_eq", "aftret_invcapx", "pretret_noa",
    "de_ratio", "debt_ebitda", "debt_capital", "debt_at",
    "intcov_ratio", "curr_ratio", "quick_ratio", "cash_ratio",
    "cash_conversion", "inv_turn", "rect_turn", "pay_turn", "sale_invcap",
    "accrual", "fcf_ocf", "cash_debt", "short_debt",
    "gsector",
]

ratios = db.raw_sql(f"""
    SELECT {', '.join(WRDS_COLS)}
    FROM wrdsapps_finratio.firm_ratio
    WHERE public_date >= '2000-01-01' AND public_date <= '2023-12-31'
      AND permno IS NOT NULL
    ORDER BY permno, public_date
""")
print(f"  Downloaded: {ratios.shape}")

ratios["public_date"] = pd.to_datetime(ratios["public_date"])
ratios["year"] = ratios["public_date"].dt.year
ratios["month"] = ratios["public_date"].dt.month
ratios["permno"] = ratios["permno"].astype(int)
ratios.drop(columns=["public_date"], inplace=True)

# Deduplicate: keep last row per (permno, year, month)
before_dedup = len(ratios)
ratios = ratios.groupby(["permno", "year", "month"]).last().reset_index()
print(f"  Deduped ratios: {before_dedup} → {len(ratios)} rows")

# Drop overlapping columns
ratio_feat = [c for c in ratios.columns if c not in ("permno", "year", "month")]
drop_cols = [c for c in ratio_feat if c.lower() in existing_cols]
if drop_cols:
    print(f"  Dropping {len(drop_cols)} overlapping: {drop_cols}")
    ratios.drop(columns=drop_cols, inplace=True)
new_ratio_cols = [c for c in ratios.columns if c not in ("permno", "year", "month")]
print(f"  New ratio features: {len(new_ratio_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 2: IBES Analyst Consensus EPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[2/4] Downloading IBES Analyst Consensus ...")

ibes_raw = db.raw_sql("""
    SELECT l.permno,
           s.statpers,
           s.numest   AS ibes_numest,
           s.numup    AS ibes_numup,
           s.numdown  AS ibes_numdown,
           s.medest   AS ibes_medest,
           s.meanest  AS ibes_meanest,
           s.stdev    AS ibes_stdev,
           s.highest  AS ibes_highest,
           s.lowest   AS ibes_lowest,
           s.actual   AS ibes_actual
    FROM ibes.statsum_epsus s
    INNER JOIN wrdsapps.ibcrsphist l
      ON s.ticker = l.ticker
      AND s.statpers BETWEEN l.sdate AND l.edate
    WHERE s.statpers >= '2000-01-01' AND s.statpers <= '2023-12-31'
      AND s.fpi = '1'
      AND s.measure = 'EPS'
      AND l.permno IS NOT NULL
    ORDER BY l.permno, s.statpers
""")
print(f"  Downloaded: {ibes_raw.shape}")

ibes_raw["statpers"] = pd.to_datetime(ibes_raw["statpers"])
ibes_raw["year"] = ibes_raw["statpers"].dt.year
ibes_raw["month"] = ibes_raw["statpers"].dt.month
ibes_raw["permno"] = ibes_raw["permno"].astype(int)

# Construct derived features
ibes_raw["ibes_disp"] = ibes_raw["ibes_stdev"] / ibes_raw["ibes_meanest"].abs().clip(lower=0.01)
ibes_raw["ibes_range"] = ibes_raw["ibes_highest"] - ibes_raw["ibes_lowest"]
# NOTE: ibes_surprise uses ibes_actual which is FORWARD-LOOKING (known only after
# earnings announcement). We compute it LAGGED: merge actual from the PREVIOUS
# period's realization, not the current forecast's actual.
# For safety, we drop ibes_surprise and ibes_actual entirely — they are not
# available at the time of the forecast.
ibes_raw["ibes_revision"] = (ibes_raw["ibes_numup"] - ibes_raw["ibes_numdown"]) / \
                              ibes_raw["ibes_numest"].clip(lower=1)

# Keep last stat period per permno-month (most recent consensus)
ibes_raw.sort_values("statpers", inplace=True)
ibes = ibes_raw.groupby(["permno", "year", "month"]).last().reset_index()

ibes_keep = ["permno", "year", "month",
             "ibes_numest", "ibes_meanest", "ibes_medest", "ibes_stdev",
             "ibes_disp", "ibes_range", "ibes_revision",
             "ibes_numup", "ibes_numdown"]
ibes = ibes[ibes_keep]
new_ibes_cols = [c for c in ibes_keep if c not in ("permno", "year", "month")]
print(f"  New IBES features: {len(new_ibes_cols)} → {new_ibes_cols}")

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 3: Institutional Ownership (Compustat quarterly, forward-filled)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[3/4] Downloading Institutional Ownership ...")

io_raw = db.raw_sql("""
    SELECT ccm.lpermno AS permno,
           io.datadate,
           io.iotlshr  AS io_total_shares,
           io.iotlhldr AS io_num_holders,
           io.ionwhldr AS io_new_holders,
           io.ioshrchg AS io_share_change_pct,
           io.iotlbys  AS io_num_buyers,
           io.iotlsls  AS io_num_sellers
    FROM comp.io_qaggregate io
    INNER JOIN crsp.ccmxpf_lnkhist ccm
      ON io.gvkey = ccm.gvkey
      AND ccm.linktype IN ('LC','LU')
      AND ccm.linkprim IN ('P','C')
      AND io.datadate BETWEEN ccm.linkdt AND COALESCE(ccm.linkenddt, '2099-12-31')
    WHERE io.datadate >= '2000-01-01' AND io.datadate <= '2023-12-31'
    ORDER BY ccm.lpermno, io.datadate
""")
print(f"  Downloaded: {io_raw.shape}")

io_raw["datadate"] = pd.to_datetime(io_raw["datadate"])
io_raw["year"] = io_raw["datadate"].dt.year
io_raw["month"] = io_raw["datadate"].dt.month
io_raw["permno"] = io_raw["permno"].astype(int)

# Buy-sell imbalance
io_raw["io_buy_sell_ratio"] = (io_raw["io_num_buyers"] - io_raw["io_num_sellers"]) / \
                                io_raw["io_num_holders"].clip(lower=1)

io_raw.sort_values("datadate", inplace=True)
io = io_raw.groupby(["permno", "year", "month"]).last().reset_index()

io_keep = ["permno", "year", "month",
           "io_num_holders", "io_new_holders", "io_share_change_pct",
           "io_buy_sell_ratio"]
io = io[io_keep]

# Forward-fill quarterly IO data to monthly
# Create full monthly grid per permno and ffill
all_permnos = io["permno"].unique()
date_grid = pd.DataFrame(
    [(p, y, m) for p in all_permnos
     for y in range(2000, 2024) for m in range(1, 13)],
    columns=["permno", "year", "month"]
)
io = date_grid.merge(io, on=["permno", "year", "month"], how="left")
io.sort_values(["permno", "year", "month"], inplace=True)
io_feat_cols = [c for c in io.columns if c.startswith("io_")]
io[io_feat_cols] = io.groupby("permno")[io_feat_cols].ffill()
io.dropna(subset=io_feat_cols, how="all", inplace=True)

new_io_cols = io_feat_cols
print(f"  New IO features: {len(new_io_cols)} → {list(new_io_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE 4: Short Interest (Compustat, semi-monthly → monthly)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[4/4] Downloading Short Interest ...")

si_raw = db.raw_sql("""
    SELECT ccm.lpermno AS permno,
           si.datadate,
           si.shortintadj AS si_shares_short
    FROM comp.sec_shortint si
    INNER JOIN crsp.ccmxpf_lnkhist ccm
      ON si.gvkey = ccm.gvkey
      AND ccm.linktype IN ('LC','LU')
      AND ccm.linkprim IN ('P','C')
      AND si.datadate BETWEEN ccm.linkdt AND COALESCE(ccm.linkenddt, '2099-12-31')
    WHERE si.datadate >= '2000-01-01' AND si.datadate <= '2023-12-31'
      AND si.iid = '01'
    ORDER BY ccm.lpermno, si.datadate
""")
print(f"  Downloaded: {si_raw.shape}")

si_raw["datadate"] = pd.to_datetime(si_raw["datadate"])
si_raw["year"] = si_raw["datadate"].dt.year
si_raw["month"] = si_raw["datadate"].dt.month
si_raw["permno"] = si_raw["permno"].astype(int)

# Take last report per month, compute MoM change
si_raw.sort_values("datadate", inplace=True)
si = si_raw.groupby(["permno", "year", "month"]).last().reset_index()
si.sort_values(["permno", "year", "month"], inplace=True)
si["si_shares_short_chg"] = si.groupby("permno")["si_shares_short"].pct_change()

si_keep = ["permno", "year", "month", "si_shares_short", "si_shares_short_chg"]
si = si[si_keep]
new_si_cols = ["si_shares_short", "si_shares_short_chg"]
print(f"  New SI features: {len(new_si_cols)} → {new_si_cols}")

db.close()

# ══════════════════════════════════════════════════════════════════════════════
#  MERGE ALL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Merging all sources ...")

merged = df.copy()
for name, data, cols in [
    ("Financial Ratios", ratios, new_ratio_cols),
    ("IBES Analyst",     ibes,   new_ibes_cols),
    ("Inst. Ownership",  io,     new_io_cols),
    ("Short Interest",   si,     new_si_cols),
]:
    before = merged.shape[1]
    merged = merged.merge(data, on=["permno", "year", "month"], how="left")
    cov = merged[cols[0]].notna().mean() if cols else 0
    print(f"  + {name}: {len(cols)} cols, {cov:.1%} coverage → {merged.shape}")

all_new = new_ratio_cols + new_ibes_cols + list(new_io_cols) + new_si_cols
print(f"\nTotal new features: {len(all_new)}")
print(f"Final shape: {merged.shape}")

# ── Duplicate check ──────────────────────────────────────────────────────────
dup_count = merged.duplicated(subset=["permno", "year", "month"]).sum()
print(f"\nDuplicate (permno, year, month) rows: {dup_count}")
if dup_count > 0:
    print("  ⚠ Dropping duplicates — keeping last occurrence ...")
    merged = merged.drop_duplicates(subset=["permno", "year", "month"], keep="last")
    print(f"  Shape after dedup: {merged.shape}")
else:
    print("  ✅ No duplicates found")

assert merged.duplicated(subset=["permno", "year", "month"]).sum() == 0, \
    "FATAL: duplicates remain after dedup!"

# ── Save ─────────────────────────────────────────────────────────────────────
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
print("Done!")
