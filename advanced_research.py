"""Exploratory multi-period modeling evidence. Run after research.py."""
import hashlib, html, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from research import ROOT, prepare
from models import Ridge, HistGradientBoostingRegressor, NeuralRegressor

OUT=ROOT/'results'/'advanced'
NAMES=['ridge','boosting','neural_network']

def create_models():
    return {'ridge':Ridge(100),'boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15), 'neural_network':NeuralRegressor()}

def weights_grid():
    return [np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a)]

def bootstrap(daily,block):
    rng=np.random.default_rng(2026+block);x=np.asarray(daily);n=len(x);means=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        means.append(x[idx].mean())
    return np.quantile(means,[.025,.975]).tolist()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data'/'recent_stocks.csv';raw=pd.read_csv(path)
    manifest=json.loads((ROOT/'data'/'recent_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    plan=json.loads((ROOT/'EXPERIMENT_PLAN.json').read_text())
    d,features=prepare(raw)
    assert np.isfinite(d[features+['target']].to_numpy()).all()
    audit={'raw_rows':len(raw),'duplicate_stock_dates':int(raw.duplicated(['Name','date']).sum()),'missing_cells':raw.isna().sum().to_dict(),'nonpositive_price_rows':int((raw[['open','high','low','close']]<=0).any(axis=1).sum()),'invalid_ohlc_rows':int(((raw.high<raw[['open','close','low']].max(axis=1))|(raw.low>raw[['open','close','high']].min(axis=1))).sum()),'prepared_rows':len(d),'absolute_targets_over_20pct':int((d.target>.2).sum()),'next_observation_gaps_over_4_calendar_days':int(((d.target_date-d.date).dt.days>4).sum())}
    raw.groupby('Name').agg(rows=('date','size'),first_date=('date','min'),last_date=('date','max')).to_csv(OUT/'ticker_coverage.csv')
    d.nlargest(25,'target')[['Name','date','target_date','target','return_1']].to_csv(OUT/'tail_observations.csv',index=False)
    scores=[];all_predictions=[];choices=[];timings=[]
    for year in plan['test_years']:
        vs=pd.Timestamp(f'{year-1}-01-01');ts=pd.Timestamp(f'{year}-01-01');end=pd.Timestamp(f'{year+1}-01-01')
        train=d[d.target_date<vs];val=d[(d.date>=vs)&(d.target_date<ts)]
        fit=d[d.target_date<ts];test=d[(d.date>=ts)&(d.target_date<end)]
        assert train.target_date.max()<val.date.min() and fit.target_date.max()<test.date.min()
        assert len(train)>0 and len(val)>0 and len(test)>0
        vp=[]
        for name,m in create_models().items():
            begin=time.perf_counter();m.fit(train[features],train.target)
            timings.append({'test_year':year,'stage':'validation_fit','model':name,'rows':len(train),'seconds':time.perf_counter()-begin})
            vp.append(np.maximum(m.predict(val[features]),0))
        matrix=np.column_stack(vp)
        best=min(weights_grid(),key=lambda w:np.abs(matrix@w-val.target.to_numpy()).mean())
        choices.append({'test_year':year,'ridge_weight':best[0],'boosting_weight':best[1],'neural_weight':best[2],'validation_mae':float(np.abs(matrix@best-val.target.to_numpy()).mean()),'train_rows':len(train),'validation_rows':len(val),'refit_rows':len(fit),'test_rows':len(test)})
        p=test[['date','target_date','Name','target','abs_mean_20']].copy().rename(columns={'Name':'ticker','target':'actual','abs_mean_20':'rolling_20'})
        p['test_year']=year;p['constant']=fit.target.mean()
        for name,m in create_models().items():
            begin=time.perf_counter();m.fit(fit[features],fit.target)
            timings.append({'test_year':year,'stage':'test_refit','model':name,'rows':len(fit),'seconds':time.perf_counter()-begin})
            p[name]=np.maximum(m.predict(test[features]),0)
        p['ensemble']=p[NAMES].to_numpy()@best
        reduced=['abs_return_1','abs_mean_5','abs_mean_20']
        ablation=Ridge(100).fit(fit[reduced],fit.target)
        p['ridge_volatility_only']=np.maximum(ablation.predict(test[reduced]),0)
        for name in ['constant','rolling_20']+NAMES+['ensemble','ridge_volatility_only']:
            err=p[name]-p.actual
            scores.append({'test_year':year,'model':name,'n':len(p),'mae':float(err.abs().mean()),'rmse':float(np.sqrt((err**2).mean()))})
        all_predictions.append(p)
        print(f'Completed fold {year}: {len(p):,} observations; ensemble weights {best.tolist()}',flush=True)
    pred=pd.concat(all_predictions,ignore_index=True)
    assert not pred.duplicated(['ticker','date']).any()
    modelnames=['constant','rolling_20']+NAMES+['ensemble','ridge_volatility_only']
    uncertainty=[];summary=[];monthly=[];slices=[]
    for name in modelnames:
        err=(pred[name]-pred.actual).abs()
        improvement=(pred.rolling_20-pred.actual).abs()-err
        daily=improvement.groupby(pred.date).mean()
        summary.append({'model':name,'mae':float(err.mean()),'rmse':float(np.sqrt(((pred[name]-pred.actual)**2).mean())),'improvement_vs_rolling_pct':float(100*(1-err.mean()/(pred.rolling_20-pred.actual).abs().mean()))})
        for block in [5,10,20]:
            lo,hi=bootstrap(daily,block)
            uncertainty.append({'model':name,'block_sessions':block,'daily_mean_mae_gain':float(daily.mean()),'ci95_low':lo,'ci95_high':hi,'days':len(daily)})
        for period,idx in pred.groupby(pred.date.dt.to_period('M')).groups.items():monthly.append({'month':str(period),'model':name,'mae':float(err.loc[idx].mean())})
        for label,mask in [('target_above_5pct',pred.actual>.05),('target_at_most_5pct',pred.actual<=.05)]:slices.append({'slice':label,'model':name,'n':int(mask.sum()),'mae':float(err[mask].mean())})
    summary=pd.DataFrame(summary).sort_values('mae');folds=pd.DataFrame(scores)
    files={'summary':summary,'fold_metrics':folds,'bootstrap_intervals':pd.DataFrame(uncertainty),'monthly_metrics':pd.DataFrame(monthly),'error_slices':pd.DataFrame(slices),'ensemble_selection':pd.DataFrame(choices),'runtime':pd.DataFrame(timings)}
    for name,table in files.items():table.to_csv(OUT/f'{name}.csv',index=False)
    pred.to_csv(OUT/'out_of_time_predictions.csv',index=False)
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    metadata={'data_sha256':manifest['sha256'],'plan_sha256':hashlib.sha256((ROOT/'EXPERIMENT_PLAN.json').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'observations':len(pred),'test_feature_start':str(pred.date.min().date()),'test_feature_end':str(pred.date.max().date()),'last_label_date':str(pred.target_date.max().date()),'scope':'exploratory; previously inspected dataset; no untouched confirmatory holdout'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    ci=pd.DataFrame(uncertainty);ci=ci[ci.block_sessions==10]
    pivot=folds.pivot(index='test_year',columns='model',values='mae')
    sections=[('Research question','Do range, volume and momentum features add predictive information beyond trailing absolute returns? We compare standardized ridge, quantile-binned boosted stumps, a 16-unit ReLU network, and a validation-selected convex ensemble.'),('Design','Expanding-window folds test 2023, 2024, 2025, and partial 2026. The previous year selects ensemble weights; earlier data trains candidate models. Models then refit on all pre-test labels. Feature/target crossing split boundaries is purged. Fixed model settings and the experiment plan are saved. Each stock-day receives equal weight in aggregate MAE.'),('Evidence limits','This is exploratory retrospective research. The earlier project already exposed 2025-2026 outcomes. These years cannot be described as a fresh untouched confirmatory holdout. More model complexity does not guarantee improvement. Intervals are paired circular moving-block bootstrap intervals of daily average MAE gains, not individual-row resampling. They are not adjusted for model comparisons or the broader research process.'),('Data quality','102 convenience-selected current stocks; selection and survivorship bias remain. Retrospectively adjusted Yahoo OHLC, unchanged provider volume, no independently verified point-in-time corporate actions. Invalid/missing bars are excluded. The target is next observed return and may cross a data gap. Tail observations are retained and listed for investigation; no manual correction or tail exclusion was performed.'),('Interpretation','Positive bootstrap gain favors the model over rolling-20; an interval spanning zero does not establish a reliable gain. Daily-average bootstrap estimates and stock-day-weighted summary metrics have different weighting. Volatility-only ridge is an ablation, not a causal proof. Test-time tail slices are diagnostics, not information available when predicting. No trading strategy or profit is tested.'),('Next validation','Freeze a model before collecting genuinely unseen future outcomes; test an independently sourced point-in-time universe; investigate listed extremes and gaps; compare with established model libraries when the runtime permits. Maintain a dated experiment log and repeat evidence across independently motivated projects.'),('Application wording','Describe this as an AI-assisted research project that you reproduced and extended. Do not describe one generated study as professional experience, proven trading performance, or a sustained track record.')]
    content='<html><head><meta charset="utf-8"><title>Equity volatility research</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#17243a}h1,h2{color:#153e64}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ccd7df;text-align:right}th:first-child,td:first-child{text-align:left}.note{background:#eef4fa;padding:18px}a{color:#1464a0}</style></head><body><h1>Equity volatility: multi-period modeling evidence</h1><p class="note">Real measured results, 2020–November 2026 data. Exploratory retrospective portfolio research; no claim of trading profitability or a sustained track record.</p>'
    for title,body in sections[:3]:content+=f'<h2>{title}</h2><p>{html.escape(body)}</p>'
    for title,table in [('Aggregate out-of-time errors',summary),('MAE by test year',pivot),('Paired bootstrap gains: 10-session blocks',ci),('Validation-selected ensemble weights',pd.DataFrame(choices)),('Tail error diagnostics',pd.DataFrame(slices))]:content+=f'<h2>{title}</h2>'+table.to_html(index=title=='MAE by test year',float_format=lambda x:f'{x:.6f}')
    content+='<h2>Data audit</h2><pre>'+html.escape(json.dumps(audit,indent=2))+'</pre>'
    for title,body in sections[3:]:content+=f'<h2>{title}</h2><p>{html.escape(body)}</p>'
    content+='<p>See companion CSV files for monthly errors, timing, data coverage, uncertainty sensitivity and full predictions.</p></body></html>'
    (OUT/'RESEARCH_REPORT.html').write_text(content,encoding='utf-8')
    bullet=f"Developed and evaluated ridge, gradient-boosted, neural-network and validation-selected ensemble volatility models on {len(raw):,} daily observations across 102 equities (2020–2026), using four expanding-window evaluations, feature ablation and paired block-bootstrap uncertainty analysis."
    (OUT/'APPLICATION_WORDING.txt').write_text('Use only after reviewing, reproducing and being able to explain the work. Disclose AI assistance as appropriate.\n\n'+bullet+'\n\nOne project does not establish a sustained proven track record.\n',encoding='utf-8')
    print(summary.to_string(index=False));print(ci.to_string(index=False))

if __name__=='__main__':main()
