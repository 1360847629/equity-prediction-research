"""Independent numerical sanity checks and persisted-result integrity checks."""
import numpy as np
import pandas as pd
from models import Ridge, NeuralRegressor
from research import ROOT

rng=np.random.default_rng(4)
x=rng.normal(size=(3000,3));y=.03+.01*np.maximum(x[:,0],0)+.005*x[:,1]
network=NeuralRegressor(epochs=150).fit(x[:2400],y[:2400])
mse=np.mean((network.predict(x[2400:])-y[2400:])**2)
baseline=np.mean((y[:2400].mean()-y[2400:])**2)
assert mse < baseline*.2, (mse,baseline)
print(f'PASS: nonlinear synthetic holdout MSE/baseline = {mse/baseline:.4f} (<0.2).')
ridge=Ridge(alpha=0).fit(x,y)
z=(x-x.mean(0))/x.std(0);design=np.column_stack([np.ones(len(x)),z])
reference=np.linalg.lstsq(design,y,rcond=None)[0]
np.testing.assert_allclose(ridge.predict(x),design@reference,atol=1e-10)
print('PASS: ridge matches independent least-squares reference at alpha=0.')
folder=ROOT/'results'/'advanced'
p=pd.read_csv(folder/'out_of_time_predictions.csv')
s=pd.read_csv(folder/'summary.csv')
selection=pd.read_csv(folder/'ensemble_selection.csv').set_index('test_year')
assert not p.duplicated(['ticker','date']).any()
assert np.isfinite(p.select_dtypes('number').to_numpy()).all()
for row in s.itertuples():
    np.testing.assert_allclose(np.abs(p[row.model]-p.actual).mean(),row.mae,atol=1e-12)
for year,part in p.groupby('test_year'):
    weights=selection.loc[year,['ridge_weight','boosting_weight','neural_weight']].to_numpy(dtype=float)
    np.testing.assert_allclose(weights.sum(),1)
    assert (weights>=0).all()
    np.testing.assert_allclose(part[['ridge','boosting','neural_network']].to_numpy()@weights,part.ensemble,atol=1e-12)
print('PASS: saved MAE values recomputed from predictions; ensemble weights and predictions agree.')
