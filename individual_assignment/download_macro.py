"""
download_macro.py - Enhance the professor-provided macro file with GWZ 2024
============================================================================
The professor-provided `macro_data_goyal_initial.csv` contains the 8 classic
Goyal-Welch predictors (dp, ep, bm, ntis, tbl, tms, dfy, svar) from 1995-01
to 2019-12. Our test sample runs into 2023, so roughly 4 years of OOS months
have no real macro values at all.

This script reads the INITIAL file, downloads the public GWZ 2024 archive of
monthly predictors, and writes an ENHANCED file that
  (a) preserves every original value unchanged, and
  (b) adds GWZ monthly predictors covering 1978-01 to 2023-11, providing
      real macro observations for the 2020-2023 window.

PIPELINE
--------
0. READ     data_we/macro_data_goyal_initial.csv  (8 classic vars, 1995-2019)

1. FETCH    the public GWZ 2024 predictor archive from Google Drive (handles
            the >100 MB virus-scan confirmation page).
                id = 1TfU9fMXP8DMI5m3FvivDYFq9ZSHlDz9F
                ≈ 28 MB zip, 47 per-series CSVs (monthly/quarterly/annual)

2. EXTRACT  unzip into data_we/_gwz_raw/.  Keep only monthly files (stems
            ending `_m`); mixing frequencies would require forward-fill of
            quarterly/annual values into month-end, introducing timing
            ambiguity we'd rather avoid.

3. TRANSFORM each GWZ CSV is an expanding-window OOS forecast MATRIX, not a
             simple time series:
                 rows    = training-cutoff yyyymm
                 cols    = target yyyymm
                 cell[s,t] = forecast for month t using data through s
             For a time-series value usable at month t with NO look-ahead we
             take the DIAGONAL cell [t, t] — the value fit with data
             through t.

4. ALIGN    outer-join GWZ diagonals onto the initial file on yyyymm.  Drop
            GWZ predictors covering <70% of 1978-present.  Original
            classic-8 columns are kept as-is; classic values stay NaN
            outside 1995-2019 (we do NOT fabricate values).

5. TIMESTAMP set `date1` = first day of the month AFTER `yyyymm`.  That's
             the earliest real-time observation date, enabling
             `pd.merge_asof(direction="backward")` in main.py to attach
             month-t macro values only to CRSP rows dated t+1 or later.

6. VALIDATE  (a) Coherence  : every initial cell == its enhanced counterpart.
             (b) No look-ahead: `date1` equals first-of-month-after(yyyymm).
             (c) Monotonic yyyymm with no duplicates.
             (d) Classic columns are NaN outside the initial date range
                 (we did not invent data).

OUTPUT
------
    data_we/macro_data_goyal_enhanced.csv
        yyyymm, date1, <8 classic vars>, <N monthly GWZ predictors>
        1978-01 → 2023-11

Source:
    Goyal, Welch & Zafirov (2024) "A Comprehensive 2022 Look at the
    Empirical Performance of Equity Premium Prediction"
    https://drive.google.com/file/d/1TfU9fMXP8DMI5m3FvivDYFq9ZSHlDz9F/view

Run:
    uv run python individual_assignment/download_macro.py
"""

import os
import re
import sys
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import requests

DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_we")
INITIAL_PATH  = os.path.join(DATA_DIR, "macro_data_goyal_initial.csv")
ENHANCED_PATH = os.path.join(DATA_DIR, "macro_data_goyal_enhanced.csv")
EXTRACT_DIR   = os.path.join(DATA_DIR, "_gwz_raw")

GDRIVE_FILE_ID = "1TfU9fMXP8DMI5m3FvivDYFq9ZSHlDz9F"
FREQ_SUFFIX    = "_m"   # monthly-only predictors


# ── Download helpers ────────────────────────────────────────────────────────
def download_from_gdrive(file_id: str) -> bytes:
    """Handle the Google Drive virus-scan confirm page for large files."""
    session = requests.Session()
    r = session.get(
        "https://docs.google.com/uc",
        params={"id": file_id, "export": "download"},
        stream=True, timeout=60,
    )
    r.raise_for_status()

    if r.content[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return r.content

    html = r.content.decode("utf-8", errors="ignore")
    token_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    uuid_match  = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    params = {
        "id": file_id, "export": "download",
        "confirm": token_match.group(1) if token_match else "t",
    }
    if uuid_match:
        params["uuid"] = uuid_match.group(1)
    r2 = session.get(
        "https://drive.usercontent.google.com/download",
        params=params, stream=True, timeout=300,
    )
    r2.raise_for_status()
    buf = BytesIO()
    for chunk in r2.iter_content(chunk_size=65536):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def unzip_to_dir(raw: bytes, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        zf.extractall(out_dir)
        return zf.namelist()


# ── GWZ CSV → pd.Series (diagonal extraction) ───────────────────────────────
def load_series(path: str, name: str) -> pd.Series | None:
    """Load a GWZ predictor CSV and return it as a yyyymm-indexed Series.

    GWZ 2024 files are OOS forecast matrices: rows are training-cutoff
    yyyymm, cols are target yyyymm, cell[s,t] is the forecast for t using
    data through s.  The diagonal [t,t] is the value available at t with no
    look-ahead.
    """
    try:
        df = pd.read_csv(path, header=0, index_col=0)
    except Exception:
        return None
    if df.empty:
        return None

    cols_num = pd.to_numeric(df.columns, errors="coerce")
    rows_num = pd.to_numeric(df.index,   errors="coerce")
    is_matrix = (
        cols_num.notna().sum() > 10
        and pd.Series(rows_num).notna().sum() > 10
    )

    if is_matrix:
        df = df.copy()
        df.columns = cols_num
        df.index   = rows_num
        df = df.loc[df.index.notna(), df.columns.notna()]
        df.index   = df.index.astype(int)
        df.columns = df.columns.astype(int)
        common = sorted(set(df.index) & set(df.columns))
        if not common:
            return None
        s = pd.Series([df.at[t, t] for t in common], index=common,
                      name=name, dtype="float64")
        return s.dropna().sort_index()

    # Fallback: simple 2-col (yyyymm, value) layout
    if df.shape[1] < 1:
        return None
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.index = pd.to_numeric(df.index, errors="coerce")
    s = s[s.index.notna()]
    s.index = s.index.astype(int)
    s.name  = name
    return s.dropna().sort_index()


def load_gwz_monthly(extract_dir: str) -> pd.DataFrame:
    """Walk the extracted archive, load every *_m.csv, return a yyyymm-wide frame."""
    monthly_paths: dict[str, str] = {}
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            stem = os.path.splitext(fn)[0].lower()
            if stem.endswith(FREQ_SUFFIX):
                monthly_paths[stem] = os.path.join(root, fn)
    if not monthly_paths:
        raise RuntimeError(f"No monthly (*_m.csv) files under {extract_dir}")
    print(f"  Found {len(monthly_paths)} monthly predictor files.")

    series_list = []
    for stem, path in sorted(monthly_paths.items()):
        colname = stem[: -len(FREQ_SUFFIX)]   # drop trailing "_m"
        s = load_series(path, colname)
        if s is not None:
            series_list.append(s)
    wide = pd.concat(series_list, axis=1)
    wide.index.name = "yyyymm"
    wide = wide.reset_index().sort_values("yyyymm").reset_index(drop=True)
    wide["yyyymm"] = wide["yyyymm"].astype(int)

    # Coverage filter over 1960+ window
    in_win = wide[wide["yyyymm"] >= 196000]
    value_cols = [c for c in wide.columns if c != "yyyymm"]
    coverage = in_win[value_cols].notna().mean()
    kept = coverage[coverage >= 0.70].index.tolist()
    dropped = sorted(set(value_cols) - set(kept))
    if dropped:
        print(f"  Dropped {len(dropped)} low-coverage GWZ predictors: {dropped}")
    return wide[["yyyymm"] + kept]


# ── Enhance = initial ⨝ GWZ monthly ─────────────────────────────────────────
def build_enhanced(initial: pd.DataFrame, gwz: pd.DataFrame) -> pd.DataFrame:
    initial = initial.copy()
    initial["yyyymm"] = initial["yyyymm"].astype(int)
    init_cols = [c for c in initial.columns if c not in ("yyyymm", "date1")]
    gwz_cols  = [c for c in gwz.columns     if c != "yyyymm"]

    # Avoid collisions: if any GWZ stem equals a classic name, rename GWZ
    # side (we trust the professor's initial values for the overlap period).
    clash = sorted(set(init_cols) & set(gwz_cols))
    if clash:
        gwz = gwz.rename(columns={c: f"{c}_gwz" for c in clash})
        gwz_cols = [c for c in gwz.columns if c != "yyyymm"]
        print(f"  Renamed GWZ columns clashing with initial: {clash}")

    # Outer-join on yyyymm keeping initial's classic cols + GWZ cols
    merged = pd.merge(
        initial[["yyyymm"] + init_cols],
        gwz[["yyyymm"] + gwz_cols],
        on="yyyymm", how="outer",
    ).sort_values("yyyymm").reset_index(drop=True)

    # Recompute date1 consistently from yyyymm = first-of-month-AFTER yyyymm
    ym = merged["yyyymm"].astype(int).astype(str).str.zfill(6)
    base = pd.to_datetime(ym, format="%Y%m")
    merged["date1"] = (base + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m-%d")

    return merged[["yyyymm", "date1"] + init_cols + gwz_cols]


# ── Validation ─────────────────────────────────────────────────────────────
def validate(initial: pd.DataFrame, enhanced: pd.DataFrame) -> None:
    problems: list[str] = []
    init_cols = [c for c in initial.columns if c not in ("yyyymm", "date1")]

    # (a) Coherence: every cell in initial matches enhanced on shared (yyyymm, col)
    joined = pd.merge(
        initial[["yyyymm"] + init_cols],
        enhanced[["yyyymm"] + init_cols],
        on="yyyymm", how="inner", suffixes=("_i", "_e"),
    )
    max_abs = 0.0
    for c in init_cols:
        diff = (joined[f"{c}_i"] - joined[f"{c}_e"]).abs()
        m = float(diff.max()) if len(diff) else 0.0
        if np.isnan(m) or m > 1e-10:
            problems.append(f"(a) {c}: max|Δ| = {m}")
        max_abs = max(max_abs, 0.0 if np.isnan(m) else m)
    # Also confirm every initial yyyymm appears in enhanced
    missing = set(initial["yyyymm"]) - set(enhanced["yyyymm"])
    if missing:
        problems.append(f"(a) {len(missing)} initial yyyymm missing from enhanced")

    # (b) No look-ahead: date1 = first-of-month-after(yyyymm) for every row
    ym   = enhanced["yyyymm"].astype(int).astype(str).str.zfill(6)
    exp  = (pd.to_datetime(ym, format="%Y%m") + pd.offsets.MonthBegin(1))
    got  = pd.to_datetime(enhanced["date1"])
    bad_dates = int((exp != got).sum())
    if bad_dates:
        problems.append(f"(b) {bad_dates} rows with date1 != first-of-next-month")
    # Stronger: date1 strictly after last day of yyyymm
    year  = enhanced["yyyymm"] // 100
    month = enhanced["yyyymm"] % 100
    month_end = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1})
    ) + pd.offsets.MonthEnd(0)
    if int((got <= month_end).sum()) > 0:
        problems.append("(b) some date1 falls on/before yyyymm month-end")

    # (c) Monotonic yyyymm + no duplicates
    if enhanced["yyyymm"].duplicated().any():
        problems.append("(c) duplicate yyyymm in enhanced")
    if not enhanced["yyyymm"].is_monotonic_increasing:
        problems.append("(c) yyyymm not sorted ascending in enhanced")

    # (d) Classic cols NaN outside initial's yyyymm range
    lo, hi = int(initial["yyyymm"].min()), int(initial["yyyymm"].max())
    outside = enhanced[(enhanced["yyyymm"] < lo) | (enhanced["yyyymm"] > hi)]
    for c in init_cols:
        n_leaked = int(outside[c].notna().sum())
        if n_leaked:
            problems.append(f"(d) classic '{c}' has {n_leaked} non-NaN values "
                            f"outside initial range [{lo}, {hi}]")

    print("\nValidation")
    print(f"  (a) coherence          : max|initial-enhanced| = {max_abs:.2e} "
          f"over {len(joined)} shared months × {len(init_cols)} cols")
    print(f"  (b) no look-ahead      : date1 matches first-of-next-month for "
          f"{len(enhanced) - bad_dates}/{len(enhanced)} rows")
    print(f"  (c) monotonic / unique : "
          f"{'OK' if enhanced['yyyymm'].is_monotonic_increasing and not enhanced['yyyymm'].duplicated().any() else 'FAIL'}")
    print(f"  (d) classic vars only in [{lo}, {hi}]: "
          f"{'OK' if not any(p.startswith('(d)') for p in problems) else 'FAIL'}")

    if problems:
        print("\n  FAILURES:")
        for p in problems:
            print(f"    - {p}")
        raise RuntimeError("Validation failed; enhanced file NOT written.")
    print("  All checks passed.")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    if not os.path.exists(INITIAL_PATH):
        raise FileNotFoundError(f"Missing initial file: {INITIAL_PATH}")

    print(f"Reading initial file: {INITIAL_PATH}")
    initial = pd.read_csv(INITIAL_PATH)
    init_cols = [c for c in initial.columns if c not in ("yyyymm", "date1")]
    print(f"  shape {initial.shape}, classic vars: {init_cols}")
    print(f"  range {int(initial['yyyymm'].min())} → {int(initial['yyyymm'].max())}")

    print(f"\nDownloading GWZ zip from Google Drive (id={GDRIVE_FILE_ID}) ...")
    raw = download_from_gdrive(GDRIVE_FILE_ID)
    print(f"  received {len(raw):,} bytes")
    if raw[:4] not in (b"PK\x03\x04", b"PK\x05\x06"):
        raise RuntimeError(f"Not a ZIP archive (first bytes = {raw[:16]!r})")

    print(f"Extracting to {EXTRACT_DIR} ...")
    names = unzip_to_dir(raw, EXTRACT_DIR)
    print(f"  extracted {len(names)} files")

    print("\nLoading GWZ monthly predictors (diagonals of OOS matrices) ...")
    gwz = load_gwz_monthly(EXTRACT_DIR)
    print(f"  GWZ wide frame: {gwz.shape}")

    print("\nBuilding enhanced frame = initial ⨝ GWZ ...")
    enhanced = build_enhanced(initial, gwz)
    print(f"  enhanced shape : {enhanced.shape}")
    print(f"  enhanced range : {enhanced['date1'].min()} → {enhanced['date1'].max()}")
    print(f"  columns        : {list(enhanced.columns)}")

    validate(initial, enhanced)

    enhanced.to_csv(ENHANCED_PATH, index=False)
    print(f"\nSaved: {ENHANCED_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
