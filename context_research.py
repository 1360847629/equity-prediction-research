"""Market/sector context experiment. Optional mature XGBoost, no silent substitution."""
import hashlib,html,json,time,platform
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from research import ROOT
from download_recent import fetch,START,AS_OF
from direction_models import LogisticClassifier,NeuralClassifier
from direction_research import prepare_direction,metrics,loss
from advanced_research import bootstrap

OUT=ROOT/'results'/'context'

def load_context(symbols):
    path=ROOT/'data'/'context_etfs.csv'
    if not path.exists():
        cache=ROOT/'data'/'yahoo_raw';cache.mkdir(parents=True,exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            frames=list(pool.map(lambda symbol:fetch(symbol,START,AS_OF,cache),symbols))
        pd.concat(frames,ignore_index=True).sort_values(['Name','date']).to_csv(path,index=False)
    raw=pd.read_csv(path)
    assert set(raw.Name)==set(symbols)
    return raw

def features(raw,context):
    d,base=prepare_direction(raw)
    g=d.groupby('Name')
    # Compute long momentum on raw rows, before target/warmup row filtering.
    stock=raw.sort_values(['Name','date']).copy();stock['date']=pd.to_datetime(stock.date)
    for w in [20,60]:stock[f'stock_momentum_{w}']=stock.groupby('Name').close.pct_change(w,fill_method=None)
    d=d.merge(stock[['Name','date','stock_momentum_20','stock_momentum_60']],on=['Name','date'],validate='one_to_one')
    extra=['stock_momentum_20','stock_momentum_60'];parts=[]
    for ticker,part in context.groupby('Name'):
        part=part.sort_values('date').copy();part['date']=pd.to_datetime(part.date)
        for w in [1,5,20]:part[f'{ticker}_return_{w}']=part.close.pct_change(w,fill_method=None)
        part[f'{ticker}_volatility_20']=part.close.pct_change(fill_method=None).abs().rolling(20).mean()
        names=[f'{ticker}_return_{w}' for w in [1,5,20]]+[f'{ticker}_volatility_20']
        parts.append(part[['date']+names].set_index('date'));extra.extend(names)
    market=pd.concat(parts,axis=1).reset_index()
    d=d.merge(market,on='date',how='left',validate='many_to_one')
    d['relative_momentum_5']=d.momentum_5-d.SPY_return_5;extra.append('relative_momentum_5')
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=base+extra)
    return d,base,base+extra

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'CONTEXT_PLAN.json').read_text());raw=pd.read_csv(ROOT/'data'/'recent_stocks.csv')
    manifest=json.loads((ROOT/'data'/'recent_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'data'/'recent_stocks.csv').read_bytes()).hexdigest()==manifest['sha256']
    context=load_context(plan['context_symbols']);d,base,full=features(raw,context)
    status={'xgboost':'not loaded'}
    try:
        import xgboost as xgb
        # Force library initialization so DLL restrictions are recorded before fitting.
        xgb.DMatrix(np.array([[0.],[1.]]),label=np.array([0,1]))
        status={'xgboost':'available','version':xgb.__version__}
    except Exception as exc:
        xgb=None;status={'xgboost':'unavailable; not run, no substitute tree model used','error':str(exc)}
    (OUT/'runtime_status.json').write_text(json.dumps(status,indent=2))
    print(status,flush=True)
    rows=[];predictions=[];selections=[];timings=[]
    factories={'logistic_original':(LogisticClassifier,base),'logistic_context':(LogisticClassifier,full),'neural_context':(NeuralClassifier,full)}
    for year in plan['folds']:
        vs=pd.Timestamp(f'{year-1}-01-01');ts=pd.Timestamp(f'{year}-01-01');end=pd.Timestamp(f'{year+1}-01-01')
        train=d[d.target_date<vs];val=d[(d.date>=vs)&(d.target_date<ts)]
        fit=d[d.target_date<ts];test=d[(d.date>=ts)&(d.target_date<end)]
        assert train.target_date.max()<val.date.min() and fit.target_date.max()<test.date.min()
        p=test[['Name','date','target_date','next_return','up']].copy().rename(columns={'Name':'ticker'});p['year']=year
        p['historical_rate']=fit.up.mean();p['always_up']=1.
        validation={}
        for name,(factory,columns) in factories.items():
            begin=time.perf_counter();m=factory().fit(train[columns],train.up)
            validation[name]=float(loss(val.up,m.predict(val[columns])).mean())
            m=factory().fit(fit[columns],fit.up);p[name]=m.predict(test[columns])
            timings.append({'year':year,'model':name,'validation_and_refit_seconds':time.perf_counter()-begin})
        chosen_rounds=None
        if xgb is not None:
            begin=time.perf_counter();dt=xgb.DMatrix(train[full],label=train.up);dv=xgb.DMatrix(val[full]);df=xgb.DMatrix(fit[full],label=fit.up)
            candidates=[]
            for rounds in plan['boosting_rounds']:
                m=xgb.train(plan['xgboost'],dt,num_boost_round=rounds)
                candidates.append((float(loss(val.up,m.predict(dv)).mean()),rounds))
            validation['xgboost_context'],chosen_rounds=min(candidates)
            m=xgb.train(plan['xgboost'],df,num_boost_round=chosen_rounds)
            p['xgboost_context']=m.predict(xgb.DMatrix(test[full]))
            timings.append({'year':year,'model':'xgboost_context','validation_and_refit_seconds':time.perf_counter()-begin})
            m.save_model(OUT/f'xgboost_{year}.json')
        eligible=[name for name in validation if name!='logistic_original']
        selected=min(eligible,key=validation.get);p['validation_selected']=p[selected]
        selections.append({'year':year,'selected':selected,'xgboost_rounds':chosen_rounds,'validation_log_losses':json.dumps(validation),'train_rows':len(train),'validation_rows':len(val),'refit_rows':len(fit),'test_rows':len(test)})
        for name in ['historical_rate','always_up']+list(validation)+['validation_selected']:rows.append({'year':year,'model':name,**metrics(p.up,p[name])})
        predictions.append(p);print(f'Completed context fold {year}, selected {selected}',flush=True)
    p=pd.concat(predictions,ignore_index=True);names=[c for c in p.columns if c not in ['ticker','date','target_date','next_return','up','year']]
    summary=pd.DataFrame([{'model':name,**metrics(p.up,p[name])} for name in names]).sort_values('log_loss')
    verdict = '本轮所有拟合模型的总体对数损失均未优于历史上涨频率基准；未建立可靠的方向预测优势。' if summary.iloc[0]['model']=='historical_rate' else '请根据基准、逐年结果和不确定性区间判断改善；单次较好结果不能证明稳定优势。'
    intervals=[]
    for name in names:
        if name=='always_up':continue
        daily=pd.Series(loss(p.up,p.historical_rate)-loss(p.up,p[name])).groupby(p.date).mean()
        for block in [5,10,20]:
            lo,hi=bootstrap(daily,block);intervals.append({'model':name,'block_sessions':block,'mean_log_loss_gain':float(daily.mean()),'ci95_low':lo,'ci95_high':hi})
    tables={'summary':summary,'fold_metrics':pd.DataFrame(rows),'selection':pd.DataFrame(selections),'runtime':pd.DataFrame(timings),'bootstrap_intervals':pd.DataFrame(intervals)}
    for name,table in tables.items():table.to_csv(OUT/f'{name}.csv',index=False)
    p.to_csv(OUT/'predictions.csv',index=False)
    metadata={'stock_sha256':manifest['sha256'],'context_sha256':hashlib.sha256((ROOT/'data'/'context_etfs.csv').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'CONTEXT_PLAN.json').read_bytes()).hexdigest(),'rows':len(p),'features':full,'feature_count':len(full),'context_symbols':plan['context_symbols'],'context_latest_date':context.date.max(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scope':plan['scope']}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    content='<html lang="zh"><head><meta charset="utf-8"><title>市场与行业特征实验</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}</style></head><body><h1>方向预测：市场、行业 ETF 与成熟树模型实验</h1>'
    content+='<p><strong>'+verdict+'</strong></p>'
    content+='<p>加入市场/行业 ETF 的 1、5、20 日收益率与波动信息，以及股票 20、60 日动量和相对市场动量。所有模型使用相同的可用日期和样本；原始特征逻辑回归作为对照。主要指标为对数损失，越低越好。固定 0.5 分类阈值。</p>'
    content+='<p>使用测试期之前一年的验证结果选择模型，然后在测试年前的全部数据上重训。XGBoost 的轮数仅在验证期从 50/150 中选择。模型与参数在查看本轮结果前写入 CONTEXT_PLAN.json。样本和年份此前已被探索，不属于新的独立验证。</p>'
    content+='<h2>树模型运行状态</h2><pre>'+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>'
    for title,table in [('总体表现',summary),('逐年结果',tables['fold_metrics']),('验证期选择记录',tables['selection']),('相对历史频率基准的改善区间（10 日区块）',tables['bootstrap_intervals'].query('block_sessions==10'))]:content+=f'<h2>{title}</h2>'+table.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    content+='<h2>限制</h2><p>当前便利股票样本、ETF 代理与事后复权快照仍有选择/幸存者偏差；ETF 特征不表示历史行业成员身份。统计区间为探索性、未校正多次比较。加入特征或成熟模型不保证改善。不根据测试结果继续调参。没有交易策略、实盘执行或盈利验证。</p><h2>数据与特征</h2><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></body></html>'
    (OUT/'CONTEXT_REPORT.html').write_text(content,encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
