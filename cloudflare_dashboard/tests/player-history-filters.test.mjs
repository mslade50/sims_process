import test from 'node:test';
import assert from 'node:assert/strict';
import {HISTORY_DEFAULTS,hasSituationFilters,playerBaselineFilters,adjustedValues,decodeReference,filteredShotMetrics,filteredSkillProfile,matchedPgaEvents,matchesRound,sliceHistory,summaryWithMean} from '../app/player-history-filters.ts';
const round=(r,gap=null)=>({round:r,date:`2026-09-0${r}`,course:'North',sg_total:r,sg_basis:'source_adjusted',score:72-r,par:72,strokes_behind_leader_before:gap,sg_ott:r/10});
const event=(id='pga:2026:1',patch={})=>({id,tour:'pga',year:2026,name:'Test',course:'North / South',courses:['North','South'],date:'2026-09-01',major:false,round_data:[round(1),round(2),round(3,2),round(4,3)],...patch});
const f=(patch={})=>({...HISTORY_DEFAULTS,asOf:'2026-10-08T16:27:09Z',...patch});
test('default includes every known played round; exact selections recompute event means and exclude unknowns',()=>{
 assert.deepEqual(HISTORY_DEFAULTS.rounds,[1,2,3,4]);
 const es=[event('pga:2026:1',{round_data:[round(1),round(2),round(3),round(4),{round:0,sg_total:99},{sg_total:99}]})];
 const all=sliceHistory(es,f());assert.equal(all[0].round_data.length,4);assert.equal(all[0].sg_total,2.5);
 const r3=sliceHistory(es,f({rounds:[3]}));assert.equal(r3[0].sg_total,3);assert.equal(r3[0].score_to_par,-3);assert.deepEqual(adjustedValues(r3,-.1),[3.1]);
 assert.deepEqual(sliceHistory(es,f({rounds:[]})),[]);
});
test('contention is inclusive at two before R3/4, rejects missing, negative and opening rounds',()=>{
 for(const n of [3,4]){assert.equal(matchesRound(round(n,0),f({contention:'near'})),true);assert.equal(matchesRound(round(n,2),f({contention:'near'})),true);assert.equal(matchesRound(round(n,2.01),f({contention:'near'})),false);assert.equal(matchesRound(round(n,null),f({contention:'outside'})),false);assert.equal(matchesRound(round(n,-1),f({contention:'near'})),false);}
 assert.equal(matchesRound(round(1,0),f({contention:'near'})),false);
 assert.deepEqual(sliceHistory([event()],f({contention:'near'}))[0].round_data.map(r=>r.round),[3]);
});
test('exact course facet rejects missing round course and other venues; event facet uses stable ID',()=>{
 const e=event('pga:2026:1',{round_data:[round(1),{...round(2),course:'South'},{...round(3),course:null}]});
 assert.deepEqual(sliceHistory([e],f({course:'North'}))[0].round_data.map(r=>r.round),[1]);
 assert.equal(sliceHistory([e],f({event:'pga:2025:1'})).length,0);
});
test('PGA comparator uses matching PGA events and exact round/position/course selection, never feeder counterparts',()=>{
 const selected=sliceHistory([event(),event('kft:2026:1',{tour:'kft'})],f({rounds:[3],contention:'near'}));
 const ref=[event(),event('pga:2026:other')];
 const out=matchedPgaEvents(ref,selected,f({rounds:[3],contention:'near'}));assert.equal(out.length,1);assert.equal(out[0].id,'pga:2026:1');assert.deepEqual(adjustedValues(out,-.2),[3.2]);
 assert.deepEqual(matchedPgaEvents(ref,selected.filter(e=>e.tour==='kft'),f()),[]);
 assert.deepEqual(adjustedValues(out,null),[]);
 assert.deepEqual(adjustedValues([{...out[0],round_data:[{...round(3),sg_basis:'field_relative_only'}]}],0),[]);
});
test('reference decoder enforces publication identity and selectively decodes matching events',()=>{
 const raw={schema_version:'player_round_reference.v1',as_of:'2026-10-08T12:00:00Z',columns:['round','sg_total','sg_basis'],events:[{...event(),round_data:[[3,2,'source_adjusted']]},event('other',{round_data:[]})]};
 assert.deepEqual(decodeReference(raw,'other'),[]);const out=decodeReference(raw,'2026-10-08T16:27:00Z',new Set(['pga:2026:1']));assert.equal(out.length,1);assert.equal(out[0].round_data[0].sg_total,2);assert.deepEqual(decodeReference(raw,'2026-10-08T10:00:00Z'),[]);assert.deepEqual(decodeReference(raw,'2026-10-09T12:00:00Z'),[]);
});
test('filtered shot metrics weight individual opportunities, preserve zeros and orient low-is-good against fixed reference',()=>{
 const es=sliceHistory([event()],f());
 const deep={shots:{metrics:[{key:'test',status:'available',value:99}],round_data:[{event_id:es[0].id,round:1,date:'a',metrics:{test:{total:0,n_shots:10}}},{event_id:es[0].id,round:2,date:'b',metrics:{test:{total:30,n_shots:30}}},{event_id:es[0].id,round:3,date:'c',metrics:{test:{total:0,n_shots:10}}},{event_id:'other',round:4,date:'d',metrics:{test:{total:999,n_shots:10}}} ]}};
 const r={metrics:{test:{mean:1,sd:1,higher_is_better:false,n_players:100,cohort_values:[0,1,2]}}};
 const m=filteredShotMetrics(deep,es,r).metrics[0];assert.equal(m.value,.6);assert.equal(m.n_shots,50);assert.equal(m.n_rounds,3);assert.equal(m.reference_z,.4);assert.equal(m.status,'available');
 const empty=filteredShotMetrics(deep,[],r).metrics[0];assert.equal(empty.value,null);assert.equal(empty.status,'unavailable');assert.equal(filteredShotMetrics(null,[],r).available,false);
});
test('SG shape is observed selected-round mean versus fixed trait reference; thin samples do not become full shape',()=>{
 const p={radar:[{key:'sg_ott'}],references:{sg_ott:{mean:99,sd:99}}},ref={schema_version:'observed_category_reference.v1',axes:{sg_ott:{mean:.1,sd:.2}}};const full=filteredSkillProfile(p,sliceHistory([event()],f()),ref);assert.ok(Math.abs(full.radar[0].value-.25)<1e-9);assert.ok(Math.abs(full.radar[0].z-.75)<1e-9);assert.equal(full.radar[0].shrinkage,null);
 const thin=filteredSkillProfile(p,sliceHistory([event()],f({rounds:[3]})),ref);assert.equal(thin.radar[0].n,1);assert.equal(thin.radar[0].z,null);
});
test('mean and median use finite observations and differ with a tail, zero remains valid',()=>{
 const s=summaryWithMean([0,0,9,null,NaN]);assert.equal(s.n,3);assert.equal(s.mean,3);assert.equal(s.median,0);assert.equal(summaryWithMean([]).mean,null);
});

test('player baseline preserves period and population while removing situation predicates without mutating filters',()=>{
 const active=f({window:'all',season:'2026',tour:'pga',rounds:[3],contention:'near',major:true,event:'pga:2026:1',course:'North',search:'Test',difficulty:'hard',strength:'strong'}),saved=structuredClone(active),base=playerBaselineFilters(active);
 assert.equal(hasSituationFilters(active),true);assert.equal(hasSituationFilters(base),false);assert.equal(hasSituationFilters(f()),false);
 assert.deepEqual(active,saved);assert.equal(base.window,'all');assert.equal(base.season,'2026');assert.equal(base.tour,'pga');assert.equal(base.asOf,active.asOf);
 const es=[event(),event('pga:2026:2'),event('kft:2026:1',{tour:'kft'}),event('pga:2025:1',{year:2025})];
 assert.equal(sliceHistory(es,base).length,2);assert.deepEqual(adjustedValues(sliceHistory(es,base),0),[1,2,3,4,1,2,3,4]);
 assert.equal(hasSituationFilters(f({rounds:[]})),true);assert.equal(hasSituationFilters(f({rounds:[4,3,2,1]})),false);
});
test('baseline does not collapse when the situation is empty and shares the fixed SG and shot references',()=>{
 const active=f({event:'unplayed',rounds:[4]}),es=[event()],selected=sliceHistory(es,active),baseline=sliceHistory(es,playerBaselineFilters(active));assert.deepEqual(selected,[]);assert.equal(adjustedValues(baseline,-.2).length,4);
 const p={radar:[{key:'sg_ott'}]},ref={schema_version:'observed_category_reference.v1',axes:{sg_ott:{mean:.1,sd:.2}}};assert.equal(filteredSkillProfile(p,selected,ref).radar[0].z,null);assert.ok(Math.abs(filteredSkillProfile(p,baseline,ref).radar[0].z-.75)<1e-9);
 const d={shots:{metrics:[{key:'test'}],round_data:[1,2,3,4].map(r=>({event_id:es[0].id,round:r,date:'2026-09-01',metrics:{test:{total:r*10,n_shots:10}}}))}},sr={metrics:{test:{mean:2,sd:1}}};
 assert.equal(filteredShotMetrics(d,selected,sr).metrics[0].status,'unavailable');const m=filteredShotMetrics(d,baseline,sr).metrics[0];assert.equal(m.n_shots,40);assert.equal(m.value,2.5);assert.equal(m.reference_z,.5);
});
