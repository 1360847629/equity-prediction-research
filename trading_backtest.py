"""Auditable delayed-execution, cost-aware backtest of saved direction predictions."""
import hashlib,html,json
import numpy as np
import pandas as pd
from research import ROOT

OUT=ROOT/'results'/'trading'

def position_pnl(budget,entry,exit,cost):
    notional=budget/(1+cost);exit_notional=notional*exit/entry
    fees=cost*(notional+exit_notional)
    return notional,exit_notional,fees,exit_notional-notional-fees

def max_drawdown(equity,initial):
    a=np.r_[initial,np.asarray(equity,dtype=float)]
    return float((a/np.maximum.accumulate(a)-1).min())

def summarize(frame,ledger,initial):
    r=frame.net_return.to_numpy();sd=r.std(ddof=1)
    active=frame.gross_exposure>0
    trades=ledger if len(ledger) else pd.DataFrame({'net_pnl':[]})
    wins=trades.loc[trades.net_pnl>0,'net_pnl'].sum();losses=-trades.loc[trades.net_pnl<0,'net_pnl'].sum()
    return {'sessions':len(r),'end_equity':float(frame.equity.iloc[-1]),'total_return':float(frame.equity.iloc[-1]/initial-1),'annualized_return':float((frame.equity.iloc[-1]/initial)**(252/len(r))-1),'annualized_volatility':float(sd*np.sqrt(252)),'sharpe_rf_zero':float(r.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,'max_drawdown':max_drawdown(frame.equity,initial),'worst_day':float(r.min()),'active_fraction':float(active.mean()),'average_gross_exposure':float(frame.gross_exposure.mean()),'average_two_way_turnover':float(frame.turnover.mean()),'total_cost_usd':float(frame.cost_usd.sum()),'stock_round_trips':len(trades),'net_trade_win_rate':float((trades.net_pnl>0).mean()) if len(trades) else np.nan,'net_profit_factor':float(wins/losses) if losses>0 else np.nan}

def simulate(signals,prices,sessions,strategy,bps,initial=100000):
    cost=bps/10000;equity=float(initial);rows=[];trades=[]
    for date in sessions:
        start=equity;day=signals[signals.target_date==date].sort_values(['validation_selected','ticker'],ascending=[False,True])
        if strategy=='primary':
            selected=day[day.validation_selected>=.55].head(20);weight=.05
        elif strategy=='top20_unfiltered':selected=day.head(20);weight=.05
        elif strategy=='equal_weight_intraday':selected=day;weight=min(.05,1/len(day)) if len(day) else 0
        else:raise ValueError(strategy)
        exposure=len(selected)*weight
        assert exposure<=1+1e-10
        pnl=0.;fees=0.;turnover=0.;gross=0.
        for trade in selected.itertuples():
            # Selection is made before accessing future execution prices; no silently skipped fills.
            entry,exit=prices[(trade.ticker,date)]
            assert entry>0 and exit>0
            notional,exit_notional,charged,net=position_pnl(start*weight,entry,exit,cost)
            pnl+=net;fees+=charged;turnover+=notional+exit_notional;gross+=exit_notional-notional
            trades.append({'strategy':strategy,'cost_bps':bps,'signal_date':trade.date,'execution_date':date,'ticker':trade.ticker,'up_probability':trade.validation_selected,'weight':weight,'entry_adjusted_open':entry,'exit_adjusted_close':exit,'entry_notional':notional,'exit_notional':exit_notional,'cost_usd':charged,'gross_pnl':exit_notional-notional,'net_pnl':net})
        equity=start+pnl
        assert equity>0
        rows.append({'date':date,'strategy':strategy,'cost_bps':bps,'start_equity':start,'gross_return':gross/start,'net_return':pnl/start,'equity':equity,'cost_usd':fees,'turnover':turnover/start,'gross_exposure':exposure/(1+cost),'round_trips':len(selected)})
    return pd.DataFrame(rows),pd.DataFrame(trades)

def spy_benchmark(spy,sessions,bps,initial,buy_hold=False):
    bars=spy.set_index('date').loc[sessions];cost=bps/10000;eq=initial;rows=[]
    if buy_hold:
        units=initial/(1+cost)/float(bars.open.iloc[0]);prior_close=None
    for i,bar in enumerate(bars.itertuples()):
        start=eq
        if buy_hold:
            eq=units*bar.close;fees=initial/(1+cost)*cost if i==0 else 0.;turnover=initial/(1+cost)/start if i==0 else 0.
            if i==len(bars)-1:
                fees+=eq*cost;turnover+=eq/start;eq*=1-cost
            gross=(units*bar.close-(initial/(1+cost) if i==0 else units*prior_close))/start
            prior_close=bar.close
        else:
            n,e,fees,pnl=position_pnl(start,float(bar.open),float(bar.close),cost);eq=start+pnl;gross=(e-n)/start;turnover=(n+e)/start
        exposure=(1/(1+cost) if i==0 else 1.) if buy_hold else 1/(1+cost)
        rows.append({'date':bar.Index,'strategy':'SPY_buy_hold' if buy_hold else 'SPY_intraday','cost_bps':bps,'start_equity':start,'gross_return':gross,'net_return':eq/start-1,'equity':eq,'cost_usd':fees,'turnover':turnover,'gross_exposure':exposure,'round_trips':0})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'TRADING_PLAN.json').read_text());initial=plan['initial_capital_usd']
    predpath=ROOT/'results'/'context'/'predictions.csv';signals=pd.read_csv(predpath,parse_dates=['date','target_date'])
    rawpath=ROOT/'data'/'recent_stocks.csv';raw=pd.read_csv(rawpath,parse_dates=['date'])
    contextpath=ROOT/'data'/'context_etfs.csv';context=pd.read_csv(contextpath,parse_dates=['date'])
    provenance=json.loads((ROOT/'results'/'context'/'metadata.json').read_text())
    assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==provenance['stock_sha256']
    assert hashlib.sha256(contextpath.read_bytes()).hexdigest()==provenance['context_sha256']
    assert not raw.duplicated(['Name','date']).any()
    spy=context[context.Name=='SPY'].sort_values('date')
    calendar=spy.date.to_list();nextsession=dict(zip(calendar[:-1],calendar[1:]))
    assert (signals.date.map(nextsession)==signals.target_date).all(), 'Stale or gapped signal'
    observed=pd.DatetimeIndex(sorted(signals.target_date.unique()))
    sessions=pd.DatetimeIndex([x for x in calendar if observed.min()<=x<=observed.max()])
    no_signal_sessions=sessions.difference(observed)
    prices={(row.Name,row.date):(float(row.open),float(row.close)) for row in raw.itertuples()};summaries=[];daily=[];ledgers=[];yearly=[]
    for bps in plan['one_way_total_cost_sensitivity_bps']:
        for strategy in ['primary','top20_unfiltered','equal_weight_intraday','SPY_intraday','SPY_buy_hold','cash']:
            if strategy.startswith('SPY'):
                frame=spy_benchmark(spy,sessions,bps,initial,buy_hold=strategy=='SPY_buy_hold');ledger=pd.DataFrame()
            elif strategy=='cash':
                frame=pd.DataFrame({'date':sessions,'strategy':strategy,'cost_bps':bps,'start_equity':initial,'gross_return':0.,'net_return':0.,'equity':initial,'cost_usd':0.,'turnover':0.,'gross_exposure':0.,'round_trips':0});ledger=pd.DataFrame()
            else:frame,ledger=simulate(signals,prices,sessions,strategy,bps,initial)
            summaries.append({'strategy':strategy,'cost_bps':bps,**summarize(frame,ledger,initial)})
            daily.append(frame)
            if len(ledger):ledgers.append(ledger)
            for year,part in frame.groupby(frame.date.dt.year):yearly.append({'year':year,'strategy':strategy,'cost_bps':bps,'return':float(part.equity.iloc[-1]/part.start_equity.iloc[0]-1),'max_drawdown':max_drawdown(part.equity,part.start_equity.iloc[0])})
            print(f'Completed {strategy} / {bps} bps',flush=True)
    summary=pd.DataFrame(summaries);pd.concat(daily).to_csv(OUT/'daily_portfolios.csv',index=False)
    pd.concat(ledgers).to_csv(OUT/'trade_ledger.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    years=pd.DataFrame(yearly);years.to_csv(OUT/'annual_metrics.csv',index=False)
    metadata={'signals_sha256':hashlib.sha256(predpath.read_bytes()).hexdigest(),'stock_sha256':provenance['stock_sha256'],'context_sha256':provenance['context_sha256'],'plan_sha256':hashlib.sha256((ROOT/'TRADING_PLAN.json').read_bytes()).hexdigest(),'first_signal_date':str(signals.date.min().date()),'first_execution':str(sessions.min().date()),'last_execution':str(sessions.max().date()),'sessions':len(sessions),'sessions_without_predictions_cash_for_stock_strategies':[str(x.date()) for x in no_signal_sessions],'costs_are_assumptions_not_broker_quotes':True,'scope':plan['status']}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    base=summary[summary.cost_bps==plan['primary_cost_scenario_bps']]
    columns=['strategy','total_return','annualized_return','max_drawdown','sharpe_rf_zero','average_gross_exposure','total_cost_usd','net_trade_win_rate','net_profit_factor']
    text='<html><head><meta charset="utf-8"><title>Cost-aware trading simulation</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body><h1>Delayed-execution trading simulation</h1>'
    primary=base[base.strategy=='primary'].iloc[0]
    text+=f'<p><strong>Base-cost result: primary cumulative return {primary.total_return:.2%}; maximum daily-close drawdown {primary.max_drawdown:.2%}. This experiment does not establish a profitable trading strategy.</strong></p>'
    text+='<p>Historical simulation, not live orders. Starting capital $100,000. Primary rule: after-close probability ≥55%, top 20, at most 5% of equity per stock; remaining capital stays in cash. Enter next-session open, exit that session close. Long-only, no leverage, no parameter changes after results.</p>'
    text+='<p>Base assumptions per side: 1 basis point commission + 2 basis points half-spread + 3 basis points slippage = 6 bps; sensitivity at 0 and 15 bps. These are hypothetical inputs, not measured execution costs. Both sides are charged and entry costs are funded from the allocation. Fractional notionals assumed.</p>'
    text+='<h2>Base scenario (6 bps each side)</h2>'+base[columns].to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text+='<p>Return/drawdown figures are fractions: 0.10 means 10%. Sharpe assumes zero cash interest/risk-free rate. Trade win rate is net of costs; profit factor is aggregate winning trade dollars divided by losing trade dollars. Drawdown includes starting capital and is measured at daily closes, not intraday.</p>'
    text+='<h2>All cost scenarios</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text+='<h2>Annual returns and drawdowns</h2>'+years.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text+='<h2>Important interpretation</h2><p>The model predicts next close-to-close direction, but a signal computed after the close cannot earn that day-to-next-open gap at the earlier close. The executable proxy therefore earns only next open-to-close returns. The equal-weight intraday comparator matches that exposure window. SPY buy-and-hold includes overnight returns and is contextual, not exposure matched. Daily OHLC prices do not demonstrate actual auction or broker fills. No liquidity capacity, market-impact estimation, taxes, share rounding, minimum broker fees, or intraday stop fills are modeled. No shorting or borrow costs apply. Maximum drawdown is an observed historical result, not a guaranteed loss limit.</p>'
    text+='<p>Convenience-universe selection, survivorship bias, retrospective adjusted prices and previously inspected test periods remain. The 55% rule was fixed for this experiment, not independently validated. Cost scenarios are sensitivity checks, not an optimization over test profits. Do not interpret favorable returns as proven trading skill or readiness for real capital.</p>'
    text+='<p>Year-boundary sessions without saved predictions are retained in the daily calendar. Stock strategies hold cash on those sessions; SPY benchmarks retain their specified exposure. Missing-signal dates are listed below.</p>'
    text+='<h2>Sources and artifacts</h2><p><a href="https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/types-orders">SEC Investor.gov: order types and execution-price uncertainty</a>. Full daily curves and round-trip records are in the companion CSV files.</p><pre>'+html.escape(json.dumps(metadata,indent=2))+'</pre></body></html>'
    (OUT/'TRADING_REPORT.html').write_text(text,encoding='utf-8')
    print(base[columns].to_string(index=False))

if __name__=='__main__':main()
