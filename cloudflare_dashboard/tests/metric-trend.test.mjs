import assert from 'node:assert/strict';
import test from 'node:test';
import {movingAverage,orderRounds,selectedRounds,trendSummary,trendImproved,scopeLabel} from '../app/metric-trend-rules.ts';

const row=(i,total,n,extra={})=>({event_id:'e1',round:1,date:`2026-01-${String(i).padStart(2,'0')}`,metrics:{m:n===null?undefined:{total,n_shots:n}},...extra});
test('orders by date, event then round regardless of input order',()=>{
  const rows=[{date:'2026-02-01',event_id:'b',round:2},{date:'2026-01-01',event_id:'z',round:1},{date:'2026-02-01',event_id:'b',round:1},{date:'2026-02-01',event_id:'a',round:4}];
  assert.deepEqual(orderRounds(rows).map(r=>`${r.event_id}${r.round}`),['z1','a4','b1','b2']);
});
test('moving average is shot weighted and skips rounds without data',()=>{
  const rows=[row(1,10,10),row(2,0,null),row(3,1,1),row(4,5,5)];
  const p=movingAverage(rows,'m',2);
  assert.equal(p.length,3);assert.deepEqual(p.map(x=>x.index),[1,2,3]);
  assert.equal(p[0].sma,1);assert.equal(p[1].sma,11/11);
  assert.equal(p[2].sma,6/6);assert.equal(p[2].windowRounds,2);
  const q=movingAverage([row(1,10,10),row(2,0,1),row(3,0,1)],'m',2);
  assert.equal(q[2].sma,0);assert.equal(q[1].sma,10/11);
});
test('window drops the oldest round and summary compares with window rounds earlier',()=>{
  const rows=Array.from({length:6},(_,i)=>row(i+1,i+1,1));
  const p=movingAverage(rows,'m',3);
  assert.deepEqual(p.map(x=>x.sma),[1,1.5,2,3,4,5]);
  const s=trendSummary(p,3);
  assert.equal(s.current,5);assert.equal(s.previous,2);assert.equal(s.change,3);assert.equal(s.fullWindow,true);assert.equal(s.sparse,true);
  const short=trendSummary(movingAverage(rows.slice(0,3),'m',3),3);
  assert.equal(short.change,null);assert.equal(short.previous,null);
});
test('empty or invalid data gives an empty series',()=>{
  assert.deepEqual(movingAverage([],'m',50),[]);
  assert.deepEqual(movingAverage([row(1,Number.NaN,5),row(2,1,0)],'m',50),[]);
  assert.equal(trendSummary([],50).current,null);
});
test('improvement respects lower-is-better and is silent for descriptive measures',()=>{
  assert.equal(trendImproved(-1,true,false),true);assert.equal(trendImproved(-1,false,false),false);
  assert.equal(trendImproved(1,true,true),null);assert.equal(trendImproved(null,false,false),null);
});
test('selection keeps only chosen rounds and labels radar scope',()=>{
  const rows=[row(1,1,1),{...row(2,1,1),event_id:'e2',round:3}];
  assert.equal(selectedRounds(rows,new Set(['e2|3'])).length,1);assert.equal(selectedRounds(rows,null).length,2);
  assert.match(scopeLabel(7,true,true),/7 radar rounds .*selected/);assert.match(scopeLabel(7,false,false),/7 rounds .*last three years/);
});
