"""Fetch and cache daily Yahoo Finance bars; exclude the as-of date itself."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
START='2020-01-01'
AS_OF='2026-11-23'
# Explicit convenience sample, not point-in-time index membership or a random sample.
TICKERS='AAPL MSFT AMZN GOOGL META NVDA TSLA AVGO ORCL CRM ADBE AMD INTC QCOM CSCO TXN IBM AMAT MU NOW JPM BAC WFC GS MS C V MA AXP BLK SCHW UNH JNJ LLY MRK ABBV PFE TMO ABT AMGN GILD XOM CVX COP SLB CAT DE HON GE RTX LMT BA UPS UNP WMT COST HD LOW TGT PG KO PEP MDLZ MCD SBUX NKE DIS NFLX CMCSA VZ T TMUS CHTR NEE DUK SO AEP EXC D DLR AMT PLD SPG EQIX O PSA MET AFL CB TRV PGR ALL F GM FDX DAL LUV UBER PYPL INTU ISRG BKNG'.split()

def fetch(ticker,start,as_of,cache):
    path=cache/f'{ticker}_{start}_{as_of}.json'
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    if path.exists(): payload=json.loads(path.read_text())
    else:
        params={'period1':int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()),'period2':int(datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc).timestamp()),'interval':'1d','events':'div,splits'}
        for attempt in range(3):
            try:
                r=requests.get(url,params=params,headers={'User-Agent':'quant-research-educational/1.0'},timeout=45)
                r.raise_for_status();payload=r.json()
                if payload['chart'].get('error'):raise ValueError(str(payload['chart']['error']))
                break
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
        path.write_text(json.dumps(payload))
    result=payload['chart']['result'][0]
    q=result['indicators']['quote'][0]
    d=pd.DataFrame({key:q[key] for key in ['open','high','low','close','volume']})
    d['date']=pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d')
    adj=result['indicators'].get('adjclose',[{}])[0].get('adjclose')
    if adj is None:raise ValueError('Missing adjusted close')
    factor=np.asarray(adj,dtype=float)/d.close.to_numpy(dtype=float)
    for col in ['open','high','low','close']:d[col]=d[col]*factor
    d['Name']=ticker
    d=d[(d.date>=start)&(d.date<as_of)].dropna(subset=['open','high','low','close','volume'])
    if len(d)<250:raise ValueError(f'Insufficient observations: {len(d)}')
    return d[['date','open','high','low','close','volume','Name']]

def download(start=START,as_of=AS_OF):
    folder=ROOT/'data';cache=folder/'yahoo_raw';cache.mkdir(parents=True,exist_ok=True)
    frames=[];errors={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,t,start,as_of,cache):t for t in TICKERS}
        for f in as_completed(futures):
            ticker=futures[f]
            try:frames.append(f.result())
            except Exception as exc:errors[ticker]=str(exc)
    if not frames:raise RuntimeError(f'No data downloaded: {errors}')
    d=pd.concat(frames,ignore_index=True).sort_values(['Name','date'])
    path=folder/'recent_stocks.csv';d.to_csv(path,index=False)
    manifest={'provider':'Yahoo Finance chart endpoint','source':'https://finance.yahoo.com/','start':start,'as_of_exclusive':as_of,'requested_tickers':TICKERS,'downloaded_tickers':sorted(d.Name.unique()),'failed_tickers':errors,'rows':len(d),'first_date':d.date.min(),'last_date':d.date.max(),'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'adjustment':'OHLC multiplied by Yahoo adjusted-close/close ratio; provider volume unchanged. Retrospectively adjusted snapshot; not point-in-time data.'}
    (folder/'recent_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['requested_tickers','downloaded_tickers']},indent=2),flush=True)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',default=START);p.add_argument('--as-of',default=AS_OF)
    args=p.parse_args();download(args.start,args.as_of)
