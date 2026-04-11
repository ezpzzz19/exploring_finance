"""Quick end-to-end test of the 3-model ensemble (NN3 + AE + IPCA)."""
import os, sys, time, traceback
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import (load_and_preprocess, run_ipca_stage,
                      run_ensemble_stage, evaluate_portfolio)

    t0 = time.time()
    data, stock_vars, mkt = load_and_preprocess('data')
    print(f'Data loaded: {len(data)} rows, {time.time()-t0:.1f}s', flush=True)

    # Small slice: through mid-2012 (3 test windows: 2010, 2011, 2012)
    data_small = data[data['date'] < pd.to_datetime('2012-07-01')].copy()
    print(f'Small slice: {len(data_small)} rows', flush=True)

    # IPCA (K=3 for speed)
    ipca_pred = run_ipca_stage(data_small, stock_vars, K=3,
                               oos_min_periods=96)
    print(f'IPCA done: {len(ipca_pred)} rows, {time.time()-t0:.1f}s',
          flush=True)

    # Ensemble (NN3 + AE + IPCA)
    pred_out = run_ensemble_stage(data_small, stock_vars, ipca_pred)
    print(f'Ensemble done: {len(pred_out)} rows, {time.time()-t0:.1f}s',
          flush=True)

    # Portfolio
    if len(pred_out) > 0:
        strat, deciles = evaluate_portfolio(pred_out, mkt,
                                            model='ensemble',
                                            n_long=60, n_short=40)
        print(f'Portfolio done, {time.time()-t0:.1f}s', flush=True)
    else:
        print('No predictions produced!', flush=True)

    print(f'\nTotal time: {time.time()-t0:.0f}s', flush=True)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
