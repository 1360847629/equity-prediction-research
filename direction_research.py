"""Predict next-observation up probability, not volatility. python direction_research.py"""
import hashlib, html, json, time
import numpy as np
import pandas as pd
from research import ROOT,prepare
from direction_models import LogisticClassifier,NeuralClassifier
from advanced_research import bootstrap

OUT=ROOT/'results'/'direction'
def loss(y,p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8);y=np.asarray(y)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);decision=p>=.5;pos=y==1;neg=~pos
    ranks=pd.Series(p).rank(method='average').to_numpy()
    auc=(ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()) if pos.any() and neg.any() else np.nan
    return {'n':len(y),'accuracy':float((decision==y).mean()),'balanced_accuracy':float(((decision[pos]==1).mean()+(decision[neg]==0).mean())/2),'roc_auc':float(auc),'log_loss':float(loss(y,p).mean()),'brier':float(((y-p)**2).mean())}

def prepare_direction(raw):
    d,f=prepare(raw)
    ordered=raw.sort_values(['Name','date']).drop_duplicates(['Name','date']).copy()
    ordered['next_return']=ordered.groupby('Name').close.shift(-1)/ordered.close-1
    ordered['date']=pd.to_datetime(ordered.date)
    d=d.merge(ordered[['Name','date','next_return']],on=['Name','date'],validate='one_to_one')
    assert np.isfinite(d.next_return).all()
    d['up']=(d.next_return>0).astype(int)
    return d,f

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data'/'recent_stocks.csv'
    if not path.exists():
        from download_recent import download
        download()
    manifest=json.loads((ROOT/'data'/'recent_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    raw=pd.read_csv(path);d,f=prepare_direction(raw)
    plan=json.loads((ROOT/'DIRECTION_PLAN.json').read_text())
    predictions=[];choices=[];folds=[];runtime=[]
    factories={'logistic':LogisticClassifier,'neural_network':NeuralClassifier}
    for year in plan['folds']:
        vs=pd.Timestamp(f'{year-1}-01-01');ts=pd.Timestamp(f'{year}-01-01');end=pd.Timestamp(f'{year+1}-01-01')
        train=d[d.target_date<vs];val=d[(d.date>=vs)&(d.target_date<ts)]
        fit=d[d.target_date<ts];test=d[(d.date>=ts)&(d.target_date<end)]
        assert train.target_date.max()<val.date.min() and fit.target_date.max()<test.date.min()
        vp={}
        for name,factory in factories.items():
            begin=time.perf_counter();m=factory().fit(train[f],train.up);vp[name]=m.predict(val[f])
            runtime.append({'year':year,'model':name,'stage':'validation','rows':len(train),'seconds':time.perf_counter()-begin})
        weight=min([0,.25,.5,.75,1],key=lambda w:loss(val.up,w*vp['logistic']+(1-w)*vp['neural_network']).mean())
        choices.append({'year':year,'logistic_weight':weight,'neural_weight':1-weight,'validation_log_loss':float(loss(val.up,weight*vp['logistic']+(1-weight)*vp['neural_network']).mean()),'train_rows':len(train),'validation_rows':len(val),'refit_rows':len(fit),'test_rows':len(test)})
        p=test[['Name','date','target_date','next_return','up']].copy().rename(columns={'Name':'ticker'})
        p['year']=year;p['historical_rate']=fit.up.mean();p['coin_flip']=.5;p['always_up']=1.
        momentum=(fit.return_1>0)
        conditional={flag:float((fit.loc[momentum==flag,'up'].sum()+1)/(int((momentum==flag).sum())+2)) for flag in [False,True]}
        p['momentum']=(test.return_1>0).map(conditional)
        for name,factory in factories.items():
            begin=time.perf_counter();m=factory().fit(fit[f],fit.up);p[name]=m.predict(test[f])
            runtime.append({'year':year,'model':name,'stage':'refit','rows':len(fit),'seconds':time.perf_counter()-begin})
        p['ensemble']=weight*p.logistic+(1-weight)*p.neural_network
        for name in ['historical_rate','coin_flip','always_up','momentum','logistic','neural_network','ensemble']:
            folds.append({'year':year,'model':name,**metrics(p.up,p[name])})
        predictions.append(p);print(f'Completed direction fold {year}; logistic ensemble weight {weight}',flush=True)
    p=pd.concat(predictions,ignore_index=True);assert not p.duplicated(['ticker','date']).any()
    names=['historical_rate','coin_flip','always_up','momentum','logistic','neural_network','ensemble']
    summary=pd.DataFrame([{'model':name,**metrics(p.up,p[name])} for name in names]).sort_values('log_loss')
    intervals=[];calibration=[];monthly=[]
    for name in names:
        if name!='always_up':
            daily=pd.Series(loss(p.up,p.historical_rate)-loss(p.up,p[name])).groupby(p.date).mean()
            for block in [5,10,20]:
                lo,hi=bootstrap(daily,block)
                intervals.append({'model':name,'block_sessions':block,'mean_log_loss_gain':float(daily.mean()),'ci95_low':lo,'ci95_high':hi})
        for bucket in range(10):
            mask=(np.minimum((p[name]*10).astype(int),9)==bucket)
            if mask.any():calibration.append({'model':name,'bin_low':bucket/10,'n':int(mask.sum()),'mean_probability':float(p.loc[mask,name].mean()),'observed_up_rate':float(p.loc[mask,'up'].mean())})
        for month,part in p.groupby(p.date.dt.to_period('M')):monthly.append({'month':str(month),'model':name,**metrics(part.up,part[name])})
    tables={'summary':summary,'fold_metrics':pd.DataFrame(folds),'ensemble_selection':pd.DataFrame(choices),'bootstrap_intervals':pd.DataFrame(intervals),'calibration':pd.DataFrame(calibration),'monthly_metrics':pd.DataFrame(monthly),'runtime':pd.DataFrame(runtime)}
    for name,table in tables.items():table.to_csv(OUT/f'{name}.csv',index=False)
    p.to_csv(OUT/'direction_predictions.csv',index=False)
    metadata={'task':plan['task'],'data_sha256':manifest['sha256'],'plan_sha256':hashlib.sha256((ROOT/'DIRECTION_PLAN.json').read_bytes()).hexdigest(),'raw_first_date':raw.date.min(),'raw_last_date':raw.date.max(),'raw_rows':len(raw),'stocks':int(raw.Name.nunique()),'evaluated_rows':len(p),'up_rate':float(p.up.mean()),'zero_returns':int((p.next_return==0).sum()),'test_dates':[str(p.date.min().date()),str(p.date.max().date())],'last_label_date':str(p.target_date.max().date()),'scope':plan['scope']}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    content='<html lang="zh"><head><meta charset="utf-8"><title>股票涨跌方向预测</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#17243a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}h1,h2{color:#153e64}</style></head><body><h1>股票下一交易日上涨概率预测</h1>'
    paragraphs=[('目标','预测下一次观测的复权收盘收益率是否大于零。上涨标记为 1；下跌或持平标记为 0。输出是上涨概率，不是波动大小。所有特征只能在当日收盘后获得，因此这不是可在同一收盘价执行的交易策略。'),('方法','使用 2020 年起的数据，以 2023、2024、2025 和 2026 年部分时段进行扩展窗口评估。前一年用于选择逻辑回归与神经网络的组合权重，然后使用全部测试年前的数据重新训练。神经网络采用 16 个 ReLU 隐藏单元、sigmoid 输出、二元交叉熵和 Adam 优化。分类阈值固定为 0.5。'),('如何比较','历史上涨频率基准只使用测试期之前的数据。例如上涨天数较多时，一直猜上涨也可能超过 50% 准确率，所以不能仅与 50% 比较。主要指标是概率对数损失，越低越好；同时报告准确率、平衡准确率、AUC 和 Brier 分数。always_up 的概率固定为 1，其对数损失仅用于演示，计算前会截断为 1−1e−8。'),('证据边界','同一份历史数据此前已用于探索性研究，这不是全新的独立验证。当前股票样本有选择和幸存者偏差，复权数据为事后快照。区块自助法区间未进行多重比较校正。无实盘测试，无收益承诺，也没有将方向准确率转换成交易盈利。')]
    for title,body in paragraphs[:3]:content+=f'<h2>{title}</h2><p>{body}</p>'
    for title,table in [('总体结果',summary),('逐年结果',tables['fold_metrics']),('相对历史上涨频率基准的对数损失改善区间（10 日区块）',tables['bootstrap_intervals'].query('block_sessions==10')),('验证期选择的组合权重',tables['ensemble_selection']),('概率校准',tables['calibration'])]:content+=f'<h2>{title}</h2>'+table.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    for title,body in paragraphs[3:]:content+=f'<h2>{title}</h2><p>{body}</p>'
    content+='<h2>数据覆盖</h2><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre><p>AI 协助开发。完整逐条预测、逐月指标、模型配置与代码随项目提供。</p></body></html>'
    (OUT/'DIRECTION_REPORT.html').write_text(content,encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
