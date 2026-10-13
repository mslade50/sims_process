import assert from 'node:assert/strict';
import test from 'node:test';
import {styleOf,narrative,ranked,grouped,shareBeyond,typicalSwing} from '../app/player-style-rules.ts';
import {roundShape,readShape} from '../app/round-shape-rules.ts';

const d=(key,percentile,level,direction,extra={})=>({key,group:extra.group??'Pace across a week',title:key,label:key.toUpperCase(),value:.5,percentile,level,direction,sentence:`${key} sentence`,clause:direction==='mid'?null:`does ${key}`,...extra});
const style={rounds:300,descriptors:[d('r1',12,'headline','low'),d('swing',91,'headline','high',{group:'Consistency',typical_value:2.6}),d('carry',60,'typical','mid',{group:'Momentum'}),d('r4',20,'notable','low')]};

test('styleOf falls back to null for unpublished or malformed style blocks',()=>{
  assert.equal(styleOf(null),null);assert.equal(styleOf({label:'x'}),null);assert.equal(styleOf({style:{descriptors:[]}}),null);
  assert.equal(styleOf({style:{descriptors:[{key:'a'}]}}),null);
  assert.equal(styleOf({style}).descriptors.length,4);
});
test('ranking, grouping and share wording read the percentile in the right direction',()=>{
  assert.deepEqual(ranked(style.descriptors).map(x=>x.key),['swing','r1','r4','carry']);
  assert.equal(shareBeyond(style.descriptors[0]),88);assert.equal(shareBeyond(style.descriptors[1]),91);
  assert.deepEqual(grouped(style).map(g=>g.group),['Consistency','Pace across a week','Momentum']);
  assert.equal(typicalSwing({style}),2.6);assert.equal(typicalSwing(null),null);
});
test('narrative uses headline clauses, hedges leanings, and admits when nothing stands out',()=>{
  const n=narrative(style,'Ludvig Aberg');
  assert.match(n,/^Ludvig Aberg does swing; and does r1\./);assert.match(n,/300 PGA Tour rounds/);assert.doesNotMatch(n,/leanings/);
  const weak=narrative({descriptors:[d('r4',20,'notable','low')]});
  assert.match(weak,/^This player does r4\./);assert.match(weak,/leanings rather than firm traits/);
  assert.match(narrative({descriptors:[d('carry',55,'typical','mid')]}),/no round-pattern that clearly separates/);
});

const spread=(sd,n=200,seed=1)=>{let s=seed;const r=()=>{s=(s*16807)%2147483647;return s/2147483647;};return Array.from({length:n},()=>{const u=(r()+r()+r()+r()-2)*Math.sqrt(3)*sd;return u;});};
test('roundShape needs enough rounds and measures tails from the median',()=>{
  assert.equal(roundShape([1,2,3]),null);
  const s=roundShape(spread(2.5));assert.ok(Math.abs(s.sd-2.5)<.3);assert.ok(Math.abs(s.skew)<.4);assert.ok(s.hi>0&&s.lo>0);
});
test('readShape compares spread and tails with the PGA slice and prefers the within-player reference',()=>{
  const peer=roundShape(spread(2.9,2000,7)),tight=roundShape(spread(2.2,150,3));
  const r=readShape(tight,peer);assert.match(r.headline,/Steadier/);assert.match(r.spread,/vs PGA 2\.9 · tighter than typical/);assert.match(r.caveat,/pools many players/);
  const w=readShape(tight,peer,2.2);assert.match(w.spread,/typical PGA regular 2\.2 · about typical/);
  const lean=roundShape([...spread(2,100,5),...Array(30).fill(4.5)]);
  assert.match(readShape(lean,peer).shape,/Leans to big good rounds|Long tail of good rounds|Big rounds both ways/);
  assert.equal(readShape(null,peer),null);assert.match(readShape(tight,null).spread,/Spread 2\./);
  assert.equal(readShape(roundShape(spread(2,15)),peer).small,true);
});
