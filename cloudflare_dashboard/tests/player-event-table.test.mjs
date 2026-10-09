import assert from 'node:assert/strict';
import test from 'node:test';
import {SHOT_COLUMNS,aggregateShots,buildRows,divergingBackground,finishRank,formatShot,shotReferenceOf,sortRows} from '../app/player-event-table-rules.ts';
import {FINGERPRINT_GROUPS,AXIS_LABEL,AXIS_HELP,labelPlacement} from '../app/player-fingerprint-rules.ts';
import {histogram} from '../app/player-deep-rules.ts';
import {stepPath} from '../app/player-histogram-rules.ts';

const ev=(id,date,patch={})=>({id,tour:'pga',year:2026,name:`E${id}`,date,sg_basis:'source_adjusted',sg_total:2,sg_ott:.5,sg_app:.4,sg_arg:.1,sg_putt:.2,round_data:[{round:1},{round:2}],...patch});
test('rows: total is vs the PGA anchor, T2G is the sum of three categories and blank when one is missing',()=>{
  const [a,b]=buildRows([ev('1','2026-01-01'),ev('2','2026-01-02',{sg_arg:null,sg_basis:'field_relative_only'})],.5,undefined,{});
  assert.equal(a.sg.sg_total,1.5);assert.ok(Math.abs(a.sg.t2g-1)<1e-9);
  assert.equal(b.sg.t2g,null);assert.equal(b.sg.sg_total,null);
});
test('shot aggregation sums only the selected rounds, needs enough shots, and orients z by direction',()=>{
  const rows=[{event_id:'1',round:1,metrics:{tee_distance:{total:3000,n_shots:10},prox_fairway:{total:100,n_shots:2}}},{event_id:'1',round:3,metrics:{tee_distance:{total:9999,n_shots:10}}},{event_id:'2',round:1,metrics:{tee_distance:{total:1,n_shots:1}}}];
  const ref={tee_distance:{mean:290,sd:10,higher:true},prox_fairway:{mean:40,sd:5,higher:false}};
  const out=aggregateShots('1',[1,2],rows,ref);
  assert.equal(out.tee_distance.value,300);assert.equal(out.tee_distance.n,10);assert.equal(out.tee_distance.z,1);
  assert.equal(out.prox_fairway.value,null);// 2 shots < 5
  assert.equal(aggregateShots('9',[1],rows,ref).tee_distance.value,null);
  assert.equal(aggregateShots('1',[1],undefined,ref).tee_distance.value,null);
});
test('shotReferenceOf maps published reference fields',()=>{
  assert.deepEqual(shotReferenceOf([{key:'a',reference_mean:1,reference_sd:2,higher_is_better:false}]),{a:{mean:1,sd:2,higher:false}});
});
test('diverging scale: green positive, red negative, centred on 0, saturating, null-safe',()=>{
  assert.equal(divergingBackground(0,1),undefined);assert.equal(divergingBackground(null,1),undefined);
  assert.match(divergingBackground(1,1),/--positive\) 40%/);assert.match(divergingBackground(-5,1),/--negative\) 40%/);
  assert.match(divergingBackground(.5,1),/--positive\) 20%/);
});
test('finish ranks parse ties and put cut/withdrawn last',()=>{
  assert.equal(finishRank('1'),1);assert.equal(finishRank('T12'),12);assert.equal(finishRank('CUT'),10000);assert.equal(finishRank('MC'),10000);assert.equal(finishRank(''),null);assert.equal(finishRank('x'),null);
});
test('sorting: date desc default, numeric desc with blanks last, finish asc',()=>{
  const rows=buildRows([ev('1','2026-01-01',{finish:'T5',sg_ott:1}),ev('2','2026-03-01',{finish:'CUT',sg_ott:null}),ev('3','2026-02-01',{finish:'1',sg_ott:3})],0,undefined,{});
  assert.deepEqual(sortRows(rows,'date','desc').map(r=>r.id),['2','3','1']);
  assert.deepEqual(sortRows(rows,'sg_ott','desc').map(r=>r.id),['3','1','2']);
  assert.deepEqual(sortRows(rows,'sg_ott','asc').map(r=>r.id),['1','3','2']);
  assert.deepEqual(sortRows(rows,'finish','asc').map(r=>r.id),['3','1','2']);
});
test('formatters',()=>{assert.equal(formatShot('pct',.684),'68%');assert.equal(formatShot('yd',299.6),'300');assert.equal(formatShot('ft',34.4),'34′');assert.equal(formatShot('ft',null),'—');assert.ok(SHOT_COLUMNS.length>=5);});
test('fingerprint groups: every axis has a label and hover text, none repeated across wrong categories',()=>{
  for(const [g,keys] of Object.entries(FINGERPRINT_GROUPS)){assert.ok(keys.length>=3,g);for(const k of keys){assert.ok(AXIS_LABEL[k],k);assert.ok(AXIS_HELP[k],k);}}
  assert.ok(!FINGERPRINT_GROUPS['Short game'].some(k=>k.startsWith('prox_')||k.startsWith('app_')));
  assert.ok(FINGERPRINT_GROUPS['Approach'].includes('app_poor'));assert.ok(FINGERPRINT_GROUPS['Approach'].includes('prox_rough'));
  assert.ok(!('All-round detail' in FINGERPRINT_GROUPS));
});
test('axis labels sit to the outside of their spoke',()=>{
  assert.equal(labelPlacement(0,6,0,0,10).anchor,'middle');assert.equal(labelPlacement(1,6,0,0,10).anchor,'start');assert.equal(labelPlacement(4,6,0,0,10).anchor,'end');
});
test('histogram bins are adjacent: the step outline is one connected path across every bin',()=>{
  const bins=histogram([0,.2,.6,-1],-8,8,.5),x=v=>v*10,y=p=>100-p*100;
  assert.equal(bins.length,32);assert.equal(bins.reduce((s,b)=>s+b.count,0),4);
  const d=stepPath(bins,x,y,-8,.5);
  assert.equal((d.match(/M/g)??[]).length,1);assert.equal((d.match(/L/g)??[]).length,64+1);assert.ok(d.endsWith('Z'));
});
