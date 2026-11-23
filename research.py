"""Reproducible daily equity volatility research. Run: python research.py"""
from pathlib import Path
import hashlib, json, time, platform
import numpy as np
import pandas as pd

from models import HistGradientBoostingRegressor, Ridge, mean_absolute_error, mean_squared_error





ROOT = Path(__file__).resolve().parent
URL = 'https://finance.yahoo.com/'
VALIDATION_START = '2024-01-01'
TEST_START = '2025-01-01'

def prepare(raw):
    d = raw.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values(['Name', 'date']).drop_duplicates(['Name', 'date'])
    for col in ['open','high','low','close','volume']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d[(d[['open','high','low','close']] > 0).all(axis=1) & (d.volume > 0)].copy()
    g = d.groupby('Name', sort=False)
    d['return_1'] = g.close.pct_change(fill_method=None)
    d['abs_return_1'] = d.return_1.abs()
    d['range'] = (d.high-d.low)/d.close
    d['intraday'] = d.close/d.open-1
    d['momentum_5'] = g.close.pct_change(5, fill_method=None)
    for w in [5,20]:
        d[f'abs_mean_{w}'] = d.groupby('Name').abs_return_1.transform(lambda x:x.rolling(w).mean())
    d['volume_ratio'] = d.volume / g.volume.transform(lambda x:x.rolling(20).mean())
    d['target'] = d.groupby('Name').return_1.shift(-1).abs()
    d['target_date'] = d.groupby('Name').date.shift(-1)
    features = ['abs_return_1','return_1','range','intraday','momentum_5','abs_mean_5','abs_mean_20','volume_ratio']
    # Retain tail events; no filtering based on the future target.
    d = d.replace([np.inf,-np.inf],np.nan).dropna(subset=features+['target','target_date'])
    d = d.copy()
    return d, features

def main():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    data = ROOT/'data'; data.mkdir(exist_ok=True)
    path = data/'recent_stocks.csv'
    if not path.exists():
        from download_recent import download
        download()
    manifest = json.loads((data/'recent_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256'], 'Data differs from download manifest'
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    raw=pd.read_csv(path)
    start=time.perf_counter(); d, features=prepare(raw)
    prep_seconds=time.perf_counter()-start
    train=d[d.target_date < VALIDATION_START]
    val=d[(d.date >= VALIDATION_START) & (d.target_date < TEST_START)]
    test=d[d.date >= TEST_START]
    assert train.target_date.max() < val.date.min()
    assert val.target_date.max() < test.date.min()
    assert not d.duplicated(['Name','date']).any()
    assert all(len(x)>0 for x in [train,val,test])
    rows=[]; candidates=[]
    def evaluate(name, split, actual, pred, seconds):
        return dict(model=name,split=split,n=len(actual),mae=float(mean_absolute_error(actual,pred)),rmse=float(np.sqrt(mean_squared_error(actual,pred))),fit_seconds=seconds)
    for depth in [7,15]:
        m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=depth,learning_rate=.08,l2_regularization=10,random_state=42)
        t=time.perf_counter();m.fit(train[features],train.target);elapsed=time.perf_counter()-t
        pred=np.maximum(m.predict(val[features]),0)
        row=evaluate(f'boosting_bins_{depth}','validation',val.target,pred,elapsed)
        rows.append(row); candidates.append((row['mae'],depth))
    best=min(candidates)[1]
    predictions=pd.DataFrame({'date':test.date,'ticker':test.Name,'actual':test.target})
    # Hyperparameters chosen using validation only; refit using all pre-test observations.
    fit=d[d.target_date < TEST_START]
    models={'ridge':Ridge(alpha=100),
            'gradient_boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=best,learning_rate=.08,l2_regularization=10,random_state=42)}
    preds={'constant':np.full(len(test),fit.target.mean()),'rolling_20':test.abs_mean_20.to_numpy()}
    for name,m in models.items():
        t=time.perf_counter();m.fit(fit[features],fit.target);elapsed=time.perf_counter()-t
        preds[name]=np.maximum(m.predict(test[features]),0)
        rows.append(evaluate(name,'test',test.target,preds[name],elapsed))
    for name in ['constant','rolling_20']:
        rows.append(evaluate(name,'test',test.target,preds[name],0))
    for name,p in preds.items(): predictions[name]=p
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    monthly=[]
    for month,part in predictions.groupby(predictions.date.dt.to_period('M')):
        for name in preds:
            monthly.append({'month':str(month),'model':name,'mae':float(mean_absolute_error(part.actual,part[name]))})
    pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    meta={'source':URL,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_first_date':str(raw.date.min()),'raw_last_date':str(raw.date.max()),'as_of_exclusive':manifest['as_of_exclusive'],'validation_start':VALIDATION_START,'test_start':TEST_START,'failed_tickers':manifest['failed_tickers'],'raw_rows':len(raw),'usable_rows':len(d),'stocks':int(d.Name.nunique()),'first_date':str(d.date.min().date()),'last_date':str(d.date.max().date()),'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'refit_rows':len(fit),'selected_bins':best,'feature_memory_mb':float(d[features].memory_usage(deep=True).sum()/1024**2),'preparation_seconds':prep_seconds,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'model_implementation':'NumPy ridge and custom quantile-binned boosted stumps'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    table=metrics[metrics.split=='test'].sort_values('mae')
    baseline=float(table[table.model=='rolling_20'].mae.iloc[0]); score=float(table[table.model=='gradient_boosting'].mae.iloc[0])
    lines=['# Equity volatility research: measured results','',f"Source: {URL}",'',f"Dataset: {len(raw):,} original rows; {len(d):,} usable rows across {meta['stocks']} stocks.",f"Raw prices: {meta['raw_first_date']} through {meta['raw_last_date']}. Usable feature dates: {meta['first_date']} through {meta['last_date']}; final labels use the next observed close.",'', '## Research question','Can price and volume information available after a daily close predict the magnitude of the next observed close-to-close return? This predicts volatility, not direction or profit.','', '## Evaluation','Train: 2020-2023. Validation: 2024. Test: 2025 onward. Labels crossing split boundaries are excluded. Two quantile-bin configurations for a custom NumPy gradient-boosted decision-stump model are compared on validation MAE; the selected configuration and ridge model are refit on pre-2025 data. Test data is reserved for the final comparison. Features include the current close, so predictions are only available after that close.','', '| Model | Test MAE (return units) | Test RMSE | Fit seconds |','|---|---:|---:|---:|']
    for r in table.itertuples(): lines.append(f'| {r.model} | {r.mae:.6f} | {r.rmse:.6f} | {r.fit_seconds:.2f} |')
    lines += ['',f'Gradient boosting improvement over rolling-20 baseline in test MAE: {(baseline-score)/baseline*100:.2f}%. Negative means worse.', '', '## Scale and limitations',f"Feature table memory: {meta['feature_memory_mb']:.1f} MiB (not peak process memory). This is a modest-sized educational dataset, not evidence of distributed computing or institutional-scale research.",'The convenience sample contains selected currently traded companies, introducing selection and survivorship bias; it is not a point-in-time index universe. OHLC values use Yahoo adjusted-close ratios; volume is unchanged. Adjustments are retrospective and vendor data is not independently verified. Tail events are retained; vendor errors may create spurious extremes. No transaction costs, execution assumptions, or tradable strategy are tested. No claim of profitability is supported. A single historical holdout cannot establish robustness across regimes.','', '## Next experiments','Investigate tail observations; obtain verified adjusted point-in-time data; add rolling walk-forward folds, daily block-bootstrap uncertainty, and feature ablations. Measure runtime and peak memory at increasing sample sizes. Do this before making stronger research claims.','', '## Data snapshot','Provider: Yahoo Finance. The download manifest records the as-of cutoff, retrieval time, ticker coverage, exclusions and adjustment convention. Latest raw close is reported in run_metadata.json; the last feature date is one observation earlier because evaluation needs a known next close. Raw data is not bundled for redistribution.', '', '## Ownership', 'Prepared with AI assistance. Review, reproduce, explain, and extend the work before presenting it as your own independent research. Do not describe this as professional experience or a proven track record.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(meta,indent=2));print(table.to_string(index=False))

if __name__=='__main__':main()




