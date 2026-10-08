import assert from 'node:assert/strict';
import test from 'node:test';
import {DEFAULT_FILTERS, commonEventSlices, centeredValues, evidenceSlice, eventMetricValue, exploratoryStandouts, openingContext, distributionSummary, filterEvents, formatMetric, histogram, metricValue,
  parseDeep, roundValues, shapePoints, sortEvents, humanCheckpoint} from '../app/player-deep-rules.ts';

const event=(id,patch={})=>({id,tour:'pga',year:2026,name:`Event ${id}`,date:'2026-09-01',course:'North Course',major:false,
  difficulty:2,field_strength:1,sg_total:1,sg_basis:"source_adjusted",round_data:[{round:1,sg_total:2,sg_basis:"source_adjusted",difficulty:patch.difficulty===undefined?2:patch.difficulty,field_strength:patch.field_strength===undefined?1:patch.field_strength},{round:2,sg_total:-1,sg_basis:"source_adjusted",difficulty:patch.difficulty===undefined?2:patch.difficulty,field_strength:patch.field_strength===undefined?1:patch.field_strength},{round:3,sg_total:5,sg_basis:"source_adjusted"}],...patch});

test('deep batches require exact stable identity and supported schema',()=>{
  const deep={schema_version:'player_deep.v1',profile_id:'17639'};
  assert.equal(parseDeep(deep,'9771'),null);
  assert.equal(parseDeep({...deep,schema_version:'wrong'},'17639'),null);
  assert.equal(parseDeep({schema_version:'player_deep.batch.v1',profiles:{17639:deep}},'17639'),deep);
});
test('course difficulty filters use explicit -1/+1 thresholds and exclude missing context',()=>{
  const es=[event('hard'),event('moderate',{difficulty:0}),event('easy',{difficulty:-2}),event('missing',{difficulty:null})];
  for(const bucket of ['hard','moderate','easy'])assert.deepEqual(filterEvents(es,{...DEFAULT_FILTERS,difficulty:bucket}).map(e=>e.id),[bucket]);
  assert.equal(filterEvents(es,DEFAULT_FILTERS).length,4);
});
test('tour season major strength and course search combine without imputing unknown values',()=>{
  const es=[event('major',{major:true}),event('weak',{field_strength:-1}),event('missing',{field_strength:null}),event('feeder',{tour:'kft'})];
  assert.deepEqual(filterEvents(es,{...DEFAULT_FILTERS,tour:'pga',season:'2026',major:true,search:'north',strength:'strong'}).map(e=>e.id),['major']);
  assert.equal(filterEvents(es,{...DEFAULT_FILTERS,strength:'weak'})[0].id,'weak');
  assert.equal(filterEvents(es,{...DEFAULT_FILTERS,season:'2025'}).length,0);
});
test('round distributions retain first two rounds, never unknown rounds or field relative only data',()=>{
  const es=[event('a'),event('b',{round_data:[{sg_total:9},{round:0,sg_total:8},{round:1,sg_total:null},{round:2,sg_total:7,sg_basis:'field_relative_only'}]})];
  assert.deepEqual(roundValues(es),[2,-1]);
  assert.deepEqual(roundValues([es[0]],true),[2,-1,5]);
  assert.deepEqual(distributionSummary([]),{n:0,p10:null,median:null,p90:null,plus3:null,minus3:null});
  const s=distributionSummary([3,-3,0,null,NaN]);assert.equal(s.n,3);assert.equal(s.plus3,1/3);assert.equal(s.minus3,1/3);
});
test('histogram open tails preserve denominator and shares sum to one',()=>{
  const h=histogram([-99,-8,0,7,99]);
  assert.equal(h.reduce((s,b)=>s+b.count,0),5);
  assert.equal(h[0].count,2);assert.equal(h.at(-1).count,2);
  assert.ok(Math.abs(h.reduce((s,b)=>s+b.share,0)-1)<1e-10);
});
test('unsupported or missing shot axes stay empty and zero is a real observed value',()=>{
  const metrics=[{key:'a',status:'available',value:0,reference_z:0},{key:'b',status:'insufficient_data',value:4,reference_z:2}];
  assert.equal(metricValue(metrics[0]),0);assert.equal(metricValue(metrics[1]),null);
  assert.deepEqual(shapePoints(metrics,['a','b','absent']),{complete:false,radii:[.5,null,null]});
  assert.equal(formatMetric(null,'fraction'),'—');assert.equal(formatMetric(0,'fraction'),'0.0%');
});
test('metric sorting puts missing values last and preserves original event order',()=>{
  const es=[event('a'),event('missing',{sg_total:null}),event('high',{sg_total:3})];
  assert.deepEqual(sortEvents(es,'sg_total','desc').map(e=>e.id),['high','a','missing']);
  assert.deepEqual(es.map(e=>e.id),['a','missing','high']);
});
test('checkpoint text is human readable and never exposes raw run identifier',()=>{
  assert.match(humanCheckpoint('2026-10-08T12:21:00Z','live'),/^Live update · Oct 8, /);
  assert.equal(humanCheckpoint('R1_20261008T122100Z'),'Saved time unavailable');
});
test('adjusted summary and event numbers require positive source-adjusted basis',()=>{
  const unknown=event('unknown',{sg_basis:undefined,sg_total:99,round_data:[{round:1,sg_total:99},{round:2,sg_total:88,sg_basis:'unknown'}]});
  assert.deepEqual(roundValues([unknown]),[]);
  assert.equal(eventMetricValue(unknown,'sg_total'),null);
  assert.equal(eventMetricValue({...unknown,sg_total_field:5},'sg_total_field'),5);
  assert.deepEqual(evidenceSlice([unknown]),{n_events:0,start:null,end:null,seasons:[],tours:[]});
});
test('default recent window anchors to published date and all-history is explicit',()=>{
  const es=[event('old',{date:'2024-10-07'}),event('boundary',{date:'2024-10-08'}),event('recent',{date:'2026-10-07'}),event('future',{date:'2026-10-09'})];
  const f={...DEFAULT_FILTERS,asOf:'2026-10-08T15:00:00Z'};
  assert.equal(f.window,'recent2y');
  assert.deepEqual(filterEvents(es,f).map(e=>e.id),['boundary','recent']);
  assert.equal(filterEvents(es,{...f,window:'all'}).length,4);
});
test('opening-round context filters cannot depend on played weekend rounds',()=>{
  const base=event('a',{round_data:[{round:1,difficulty:0,field_strength:1},{round:2,difficulty:0,field_strength:1}]});
  const weekend={...base,round_data:[...base.round_data,{round:3,difficulty:10,field_strength:-10},{round:4,difficulty:10,field_strength:-10}]};
  assert.equal(openingContext(base,'difficulty'),openingContext(weekend,'difficulty'));
  assert.equal(openingContext(weekend,'field_strength'),1);
  assert.equal(filterEvents([weekend],{...DEFAULT_FILTERS,difficulty:'moderate',strength:'strong'}).length,1);
  assert.equal(filterEvents([event('unknown',{round_data:[{round:3,difficulty:0}]})],{...DEFAULT_FILTERS,difficulty:'moderate'}).length,0);
});
test('distribution evidence counts only contributing events and centers on own median',()=>{
  const es=[event('a',{date:'2026-01-01'}),event('b',{date:'2026-02-01'}),event('noadjust',{round_data:[{round:1,sg_total:5,sg_basis:'field_relative_only'}]})];
  assert.deepEqual(evidenceSlice(es),{n_events:2,start:'2026-01-01',end:'2026-02-01',seasons:[2026],tours:['pga']});
  assert.deepEqual(centeredValues([1,2,4],true),[-1,0,2]);
  assert.deepEqual(centeredValues([1,2,4],false),[1,2,4]);
  assert.deepEqual(centeredValues([],true),[]);
});
test('exploratory strengths respect visible axes and substantial support',()=>{
  const m=(key,patch={})=>({key,status:'available',value:1,reference_z:1,n_shots:100,n_rounds:10,...patch});
  const out=exploratoryStandouts([m('visible'),m('hidden',{reference_z:4}),m('thin',{n_shots:99}),m('fewrounds',{n_rounds:9})],['visible','thin','fewrounds']);
  assert.deepEqual(out.map(m=>m.key),['visible']);
});
test('common-event comparisons are symmetric and exclude unadjusted overlap',()=>{
  const a=[event('shared'),event('aonly'),event('unadjusted',{round_data:[]})],b=[event('shared'),event('bonly'),event('unadjusted')];
  const matched=commonEventSlices(a,b),reverse=commonEventSlices(b,a);
  assert.equal(matched.n_common,1);
  assert.deepEqual(matched.a.map(e=>e.id),['shared']);
  assert.deepEqual(matched.a.map(e=>e.id),reverse.b.map(e=>e.id));
  assert.equal(commonEventSlices([event('a')],[event('b')]).n_common,0);
});
