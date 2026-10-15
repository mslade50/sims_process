import assert from 'node:assert/strict';
import test from 'node:test';
import {parseSeries,seriesValues,seriesShotCount,shotMovingAverage,shotWindows,defaultShotWindow,shotTrendSummary,luckBand,nearestByTime,indexTicks,tickDate} from '../app/shot-series-rules.ts';

const series={schema_version:'player_shot_series.v1',rounds:[['2026-01-01','pga:2026:1',1],['2026-01-02','pga:2026:1',2],['2026-02-01','pga:2026:2',1]],
  metrics:{tee_sg:{n:6,rc:[0,2,1,1,2,3],v:[1,2,3,4,5,6]},tee_accuracy:{n:4,rc:[0,1,2,3],b:'1011'}}};
test('expands run-lengths into play-ordered shots with their round and date',()=>{
  const s=seriesValues(series,'tee_sg');
  assert.deepEqual(s.map(p=>p.value),[1,2,3,4,5,6]);
  assert.deepEqual(s.map(p=>p.round),[1,1,2,1,1,1]);assert.equal(s[2].date,'2026-01-02');assert.equal(s[5].event_id,'pga:2026:2');
  assert.ok(s.every((p,i)=>i===0||p.time>=s[i-1].time));
});
test('binary metrics decode from the 0/1 string',()=>{assert.deepEqual(seriesValues(series,'tee_accuracy').map(p=>p.value),[1,0,1,1]);});
test('the selected-rounds filter keeps only matching event|round shots and keeps alignment',()=>{
  const sel=new Set(['pga:2026:1|2','pga:2026:2|1']);
  assert.deepEqual(seriesValues(series,'tee_sg',sel).map(p=>p.value),[3,4,5,6]);
  assert.equal(seriesShotCount(series,'tee_sg',sel),4);assert.equal(seriesShotCount(series,'tee_sg'),6);
  assert.deepEqual(seriesValues(series,'tee_accuracy',new Set(['pga:2026:2|1'])).map(p=>p.value),[0,1,1]);
  assert.equal(seriesShotCount(series,'nope'),0);assert.deepEqual(seriesValues(series,'nope'),[]);
});
test('shot moving average is the plain trailing mean and has no partial-window values',()=>{
  const p=shotMovingAverage(seriesValues(series,'tee_sg'),3);
  assert.deepEqual(p.map(x=>x.sma),[null,null,2,3,4,5]);assert.deepEqual(p.map(x=>x.index),[1,2,3,4,5,6]);
  assert.equal(shotMovingAverage([],3).length,0);
  const s=shotTrendSummary(p,2);assert.equal(s.current,5);assert.equal(s.previous,3);assert.equal(s.change,2);
  assert.equal(shotTrendSummary(p,6).change,null);
});
test('defaults: longer windows only when there are two windows of shots; hole-based and rare metrics get their own sizes',()=>{
  assert.deepEqual(shotWindows('tee_sg'),[50,100,200]);assert.deepEqual(shotWindows('app_gir'),[50,100,200]);assert.deepEqual(shotWindows('tee_penalty'),[250,500,1000]);
  assert.equal(defaultShotWindow(5000,'tee_sg'),100);assert.equal(defaultShotWindow(150,'tee_sg'),50);assert.equal(defaultShotWindow(250,'tee_sg'),100);assert.equal(defaultShotWindow(40,'tee_sg'),50);
  assert.equal(defaultShotWindow(1000,'app_gir'),100);
});
test('nearestByTime snaps to the closest dated point',()=>{
  const pts=[{time:10},{time:20},{time:20},{time:50}];
  assert.equal(nearestByTime(pts,0),0);assert.equal(nearestByTime(pts,34),2);assert.equal(nearestByTime(pts,36),3);assert.equal(nearestByTime(pts,99),3);assert.equal(nearestByTime([],5),-1);
});
test('malformed series are rejected so the round view stays the fallback',()=>{
  assert.equal(parseSeries(null),null);assert.equal(parseSeries({rounds:[]}),null);assert.ok(parseSeries(series));
});
test('indexTicks spaces markers evenly by position, first and last included',()=>{
  assert.deepEqual(indexTicks(0),[]);assert.deepEqual(indexTicks(1),[0]);assert.deepEqual(indexTicks(2,4),[0,1]);
  assert.deepEqual(indexTicks(10,4),[0,3,6,9]);assert.deepEqual(indexTicks(3,4),[0,1,2]);
});
test('tickDate picks month+year for long spans and month/day for short ones',()=>{
  const a=Date.UTC(2023,0,15),b=Date.UTC(2026,5,1);
  assert.equal(tickDate(a,a,b),'Jan 2023');assert.equal(tickDate(a,a,a+30*864e5),'Jan 15, 23');
});
test('luck band is the overall mean ± 2 standard errors of a window mean, and narrows with longer windows',()=>{
  const shots=Array.from({length:400},(_,i)=>({value:i%2}));
  const b50=luckBand(shots,50),b200=luckBand(shots,200);
  assert.ok(Math.abs(b50.mean-.5)<1e-9);
  const se=Math.sqrt(.25*400/399/50);assert.ok(Math.abs(b50.hi-(.5+2*se))<1e-9&&Math.abs(b50.lo-(.5-2*se))<1e-9);
  assert.ok(b200.hi-b200.lo<b50.hi-b50.lo);assert.equal(luckBand([{value:1}],50),null);
});
