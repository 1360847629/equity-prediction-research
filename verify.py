"""Check temporal feature causality and target alignment on real observations."""
import numpy as np
import pandas as pd
from research import ROOT, prepare

raw=pd.read_csv(ROOT/'data'/'recent_stocks.csv')
sample=raw[raw.Name.isin(raw.Name.unique()[:3])].copy()
full,features=prepare(sample)
cutoff='2024-01-01'
changed=sample.copy()
changed.loc[changed.date >= cutoff,['open','high','low','close']]*=2
changed.loc[changed.date >= cutoff,'volume']*=3
altered,_=prepare(changed)
a=full[full.date < cutoff].set_index(['Name','date'])
b=altered[altered.date < cutoff].set_index(['Name','date'])
np.testing.assert_allclose(a[features],b.loc[a.index,features])
for ticker,part in sample.groupby('Name'):
    ordered=part.sort_values('date').set_index('date')
    expected=(ordered.close.shift(-1)/ordered.close-1).abs()
    observed=full[full.Name==ticker]
    np.testing.assert_allclose(observed.target,expected.loc[observed.date.dt.strftime('%Y-%m-%d')])
print('PASS: changing future prices/volumes does not alter past features.')
print('PASS: targets equal next-observation absolute close-to-close returns.')
