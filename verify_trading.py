"""Check execution timing, accounting, limits, drawdowns and cost sensitivity."""
import json
import numpy as np
import pandas as pd
from research import ROOT
from trading_backtest import position_pnl,max_drawdown

# Independent hand calculations for flat and winning round trips.
n,e,fees,pnl=position_pnl(1000,100,100,.001)
np.testing.assert_allclose(pnl,1000*(.999/1.001-1))
assert pnl<0
n,e,fees,pnl=position_pnl(1000,100,110,0)
np.testing.assert_allclose(pnl,100)
np.testing.assert_allclose(max_drawdown([90,95,80],100),-.2)
print('PASS: round-trip costs, positive-return arithmetic and initial-capital drawdown match hand calculations.')
out=ROOT/'results'/'trading'
daily=pd.read_csv(out/'daily_portfolios.csv',parse_dates=['date'])
ledger=pd.read_csv(out/'trade_ledger.csv',parse_dates=['signal_date','execution_date'])
summary=pd.read_csv(out/'summary.csv')
assert (ledger.execution_date>ledger.signal_date).all()
assert ledger.weight.le(.05+1e-12).all()
assert ledger.entry_notional.gt(0).all()
np.testing.assert_allclose(ledger.gross_pnl-ledger.cost_usd,ledger.net_pnl,atol=1e-9)
for (name,bps),frame in daily.groupby(['strategy','cost_bps']):
    frame=frame.sort_values('date')
    np.testing.assert_allclose(frame.equity,frame.start_equity*(1+frame.net_return),atol=1e-7)
    np.testing.assert_allclose(frame.start_equity.iloc[1:],frame.equity.iloc[:-1],atol=1e-7)
    np.testing.assert_allclose(frame.gross_return-frame.cost_usd/frame.start_equity,frame.net_return,atol=1e-10)
    assert frame.gross_exposure.between(0,1+1e-10).all()
    row=summary[(summary.strategy==name)&(summary.cost_bps==bps)].iloc[0]
    np.testing.assert_allclose(row.max_drawdown,max_drawdown(frame.equity,100000),atol=1e-10)
    np.testing.assert_allclose(row.total_return,frame.equity.iloc[-1]/100000-1,atol=1e-10)
    if name in ['primary','top20_unfiltered','equal_weight_intraday']:
        trades=ledger[(ledger.strategy==name)&(ledger.cost_bps==bps)]
        amounts=trades.groupby('execution_date').net_pnl.sum().reindex(frame.date,fill_value=0)
        np.testing.assert_allclose(amounts,frame.equity-frame.start_equity,atol=1e-7)
        c=bps/10000
        np.testing.assert_allclose(trades.cost_usd,c*(trades.entry_notional+trades.exit_notional),atol=1e-8)
        assert trades.groupby('execution_date').weight.sum().le(1+1e-10).all()
    if name in ['primary','top20_unfiltered']:
        assert frame.round_trips.le(20).all()
for name,part in summary.groupby('strategy'):
    assert np.all(np.diff(part.sort_values('cost_bps').end_equity)<=1e-6)
metadata=json.loads((out/'metadata.json').read_text())
for date in metadata['sessions_without_predictions_cash_for_stock_strategies']:
    part=daily[(daily.date==date)&daily.strategy.isin(['primary','top20_unfiltered','equal_weight_intraday'])]
    assert (part.net_return==0).all()
print('PASS: delayed execution, position limits, ledger reconciliation, fee charges, cost monotonicity and cash-gap behavior.')
