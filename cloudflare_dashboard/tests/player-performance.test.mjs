import assert from "node:assert/strict";
import test from "node:test";
import {roundEma,performanceSeries,visiblePerformance} from "../app/player-performance-rules.ts";
test("EMA seeds first span mean, responds once per round, and withholds warmup",()=>{assert.deepEqual(roundEma([1,3,5,7],3),[null,null,3,5]);assert.deepEqual(roundEma([2,2,2],1),[2,2,2]);assert.throws(()=>roundEma([NaN],2));assert.throws(()=>roundEma([1],0));});
test("performance requires adjusted basis, exact dates, finite anchor; sorts rounds and removes duplicates",()=>{
 const event={id:"pga:1",round_data:[{round:2,date:"2026-01-02",sg_total:1,sg_basis:"source_adjusted"},{round:1,date:"2026-01-01",sg_total:0,sg_basis:"source_adjusted"},{round:3,date:"2026-01-03",sg_total:3,sg_basis:"source_adjusted"},{round:4,date:"bad",sg_total:9,sg_basis:"source_adjusted"}]};
 const unknown={id:"x",round_data:[{round:1,date:"2026-01-01",sg_total:100}]};
 const p=performanceSeries([event,unknown,event],-.1);assert.equal(p.length,2);assert.deepEqual(p.map(v=>v.value),[.1,1.1]);assert.equal(performanceSeries([event],null).length,0);assert.equal(performanceSeries([event],0,true).length,3);
});
test("visible range preserves EMA initialized from earlier history and never reweights inactivity",()=>{
 const events=Array.from({length:60},(_,i)=>({id:`e${i}`,round_data:[{round:1,date:new Date(Date.UTC(2020,0,i+1)).toISOString(),sg_total:i,sg_basis:"source_adjusted"}]}));
 const points=performanceSeries(events,0);assert.equal(points[49].ema50,24.5);assert.equal(points[59].n,60);const tail=visiblePerformance(points,"2023-02-01","3y");assert.equal(tail[0].ema20,points.find(p=>p.date===tail[0].date).ema20);assert.equal(visiblePerformance(points,"2030-01-01","3y").length,0);assert.equal(visiblePerformance(points,"2030-01-01","all").length,60);
});
