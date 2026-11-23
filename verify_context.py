"""Feature-causality, common-sample, target and saved-result integrity checks."""
import numpy as np
import pandas as pd
from research import ROOT
from context_research import features
from direction_research import metrics

raw=pd.read_csv(ROOT/'data'/'recent_stocks.csv')
context=pd.read_csv(ROOT/'data'/'context_etfs.csv')
small=raw[raw.Name.isin(raw.Name.unique()[:3])].copy()
d,base,full=features(small,context)
future=context.copy();future.loc[future.date>='2024-01-01',['open','high','low','close']]*=2
changed,_,_=features(small,future)
a=d[d.date<'2024-01-01'].set_index(['Name','date']);b=changed.set_index(['Name','date'])
np.testing.assert_allclose(a[full],b.loc[a.index,full])
assert len(full)==len(set(full)) and np.isfinite(d[full].to_numpy()).all()
np.testing.assert_array_equal(d.up,(d.next_return>0).astype(int))
print('PASS: future ETF changes leave past features unchanged; all features finite and labels aligned.')
folder=ROOT/'results'/'context';p=pd.read_csv(folder/'predictions.csv')
assert not p.duplicated(['ticker','date']).any()
assert (pd.to_datetime(p.target_date)>pd.to_datetime(p.date)).all()
summary=pd.read_csv(folder/'summary.csv')
for row in summary.itertuples():
    assert p[row.model].between(0,1).all()
    assert row.n==len(p)
    for key,value in metrics(p.up,p[row.model]).items():
        np.testing.assert_allclose(value,getattr(row,key),atol=1e-10)
for row in pd.read_csv(folder/'selection.csv').itertuples():
    part=p[p.year==row.year]
    np.testing.assert_allclose(part.validation_selected,part[row.selected])
print('PASS: models use identical evaluated rows; probabilities, selected predictions and metrics agree.')
