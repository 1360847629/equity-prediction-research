"""Target, causality, classifier and saved-output checks for the direction study."""
import numpy as np
import pandas as pd
from research import ROOT
from direction_research import prepare_direction,metrics
from direction_models import LogisticClassifier,NeuralClassifier

raw=pd.read_csv(ROOT/'data'/'recent_stocks.csv')
sample=raw[raw.Name.isin(raw.Name.unique()[:3])].copy()
d,f=prepare_direction(sample)
for name,part in sample.groupby('Name'):
    ordered=part.sort_values('date').set_index('date')
    expected=(ordered.close.shift(-1)/ordered.close-1)
    rows=d[d.Name==name]
    np.testing.assert_allclose(rows.next_return,expected.loc[rows.date.dt.strftime('%Y-%m-%d')])
    np.testing.assert_array_equal(rows.up,(rows.next_return>0).astype(int))
changed=sample.copy();changed.loc[changed.date>='2024-01-01',['open','high','low','close']]*=2
other,_=prepare_direction(changed)
a=d[d.date<'2024-01-01'].set_index(['Name','date']);b=other.set_index(['Name','date'])
np.testing.assert_allclose(a[f],b.loc[a.index,f])
print('PASS: signed next-return labels align; future price changes do not alter past features.')
assert metrics([0,0,1,1],[.1,.2,.8,.9])['roc_auc']==1
assert metrics([0,0,1,1],[.5,.5,.5,.5])['roc_auc']==.5
rng=np.random.default_rng(2026);x=rng.normal(size=(6000,3));y=(x[:,0]+.5*x[:,1]>0).astype(int)
for name,m in [('logistic',LogisticClassifier()),('neural',NeuralClassifier(epochs=70))]:
    m.fit(x[:4500],y[:4500]);p=m.predict(x[4500:])
    assert ((p>=0)&(p<=1)).all()
    acc=((p>=.5)==y[4500:]).mean();assert acc>.9,(name,acc)
    print(f'PASS: {name} synthetic holdout accuracy {acc:.4f}.')
folder=ROOT/'results'/'direction';p=pd.read_csv(folder/'direction_predictions.csv')
assert not p.duplicated(['ticker','date']).any()
assert (pd.to_datetime(p.target_date)>pd.to_datetime(p.date)).all()
np.testing.assert_array_equal(p.up,(p.next_return>0).astype(int))
for row in pd.read_csv(folder/'summary.csv').itertuples():
    assert np.isfinite(p[row.model]).all() and p[row.model].between(0,1).all()
    actual=metrics(p.up,p[row.model])
    for key in ['accuracy','balanced_accuracy','log_loss','brier','roc_auc']:
        np.testing.assert_allclose(actual[key],getattr(row,key),atol=1e-10)
for row in pd.read_csv(folder/'ensemble_selection.csv').itertuples():
    part=p[p.year==row.year]
    np.testing.assert_allclose(part.ensemble,row.logistic_weight*part.logistic+row.neural_weight*part.neural_network,atol=1e-12)
print('PASS: probabilities, ensemble weights and all aggregate metrics match saved predictions.')
