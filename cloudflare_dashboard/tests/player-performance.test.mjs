import assert from "node:assert/strict";
import test from "node:test";
import {nearestPerformanceIndex,roundSma,performanceSeries,visiblePerformance,normalizeSmas,recentVsLonger,roundBarFill,smaOf,TOTAL_RANGE,MAJOR_FILL,DEFAULT_SMA} from "../app/player-performance-rules.ts";
test("SMA uses equal weights, expires old rounds exactly, and withholds partial windows",()=>{
 assert.deepEqual(roundSma([1,3,5,7,11],3),[null,null,3,5,23/3]);
 assert.deepEqual(roundSma([10,0,0,0],2),[null,5,0,0]);
 assert.deepEqual(roundSma([2,0,-2],1),[2,0,-2]);
 assert.deepEqual(roundSma([],20),[]);assert.deepEqual(roundSma([1,2],3),[null,null]);
 for(const span of [0,-1,1.5,NaN])assert.throws(()=>roundSma([1],span));
 assert.throws(()=>roundSma([NaN],2));assert.throws(()=>roundSma([Infinity],2));
});
test("performance requires adjusted basis, exact dates, finite anchor; sorts rounds and removes duplicates",()=>{
 const event={id:"pga:1",name:"Test Championship",course:"North Course",tour:"pga",round_data:[{round:2,date:"2026-01-02",sg_total:1,sg_basis:"source_adjusted"},{round:1,date:"2026-01-01",sg_total:0,sg_basis:"source_adjusted"},{round:3,date:"2026-01-03",sg_total:3,sg_basis:"source_adjusted"},{round:4,date:"bad",sg_total:9,sg_basis:"source_adjusted"}]};
 const unknown={id:"x",round_data:[{round:1,date:"2026-01-01",sg_total:100}]};
 const p=performanceSeries([event,unknown,event],-.1,false);assert.equal(p.length,2);assert.equal(performanceSeries([event],-.1).length,3,"R3/R4 included by default");assert.equal(p[0].eventName,"Test Championship");assert.equal(p[0].course,"North Course");assert.equal(p[0].tour,"pga");assert.deepEqual(p.map(v=>v.value),[.1,1.1]);assert.equal(performanceSeries([event],null).length,0);assert.equal(performanceSeries([event],0,true).length,3);
});
test("visible range preserves trailing SMA history and never reweights inactivity",()=>{
 const events=Array.from({length:60},(_,i)=>({id:`e${i}`,round_data:[{round:1,date:new Date(Date.UTC(2020,0,i+1)).toISOString(),sg_total:i,sg_basis:"source_adjusted"}]}));
 const points=performanceSeries(events,0);assert.equal(points[49].sma50,24.5);assert.equal(points[59].n,60);assert.equal(points[59].sma50,34.5);assert.equal(points[59].sma20,49.5);const tail=visiblePerformance(points,"2023-02-01","3y");assert.equal(tail[0].sma20,points.find(p=>p.date===tail[0].date).sma20);assert.equal(visiblePerformance(points,"2030-01-01","3y").length,0);assert.equal(visiblePerformance(points,"2030-01-01","all").length,60);
});

test("SMA ignores calendar gaps and changes only when eligible observations arrive",()=>{
 const values=Array.from({length:55},(_,i)=>i%7-3);
 const make=(time)=>values.map((value,i)=>({id:`e${i}`,round_data:[{round:1,date:new Date(time(i)).toISOString(),sg_total:value,sg_basis:"source_adjusted"}]}));
 const dense=performanceSeries(make(i=>Date.UTC(2020,0,i+1)),0),gapped=performanceSeries(make(i=>Date.UTC(2020+Math.floor(i/10),0,i%10+1)),0);
 assert.deepEqual(dense.map(p=>[p.sma20,p.sma50]),gapped.map(p=>[p.sma20,p.sma50]));
 assert.equal(gapped.at(-1).sma20,values.slice(-20).reduce((a,b)=>a+b,0)/20);
 assert.equal(gapped.at(-1).sma50,values.slice(-50).reduce((a,b)=>a+b,0)/50);
});

test("hover snaps to real observation dates across gaps, bounds and ties",()=>{
 const p=[{time:100},{time:200},{time:1000}];
 assert.equal(nearestPerformanceIndex([],200),-1);assert.equal(nearestPerformanceIndex(p,NaN),-1);
 for(const [time,index] of [[0,0],[100,0],[150,0],[151,1],[200,1],[600,1],[601,2],[1000,2],[1200,2]])assert.equal(nearestPerformanceIndex(p,time),index);
 assert.equal(nearestPerformanceIndex([{time:1}],999),0);
 assert.equal(nearestPerformanceIndex([{time:1},{time:1},{time:2}],1),0);
});

test("100 and 200-round SMAs use exact trailing windows; default shown line is the 50",()=>{
 const events=Array.from({length:210},(_,i)=>({id:`e${i}`,round_data:[{round:1,date:new Date(Date.UTC(2020,0,i+1)).toISOString(),sg_total:i,sg_basis:"source_adjusted"}]}));
 const p=performanceSeries(events,0);
 assert.equal(p[98].sma100,null);assert.equal(p[99].sma100,49.5);assert.equal(p[198].sma200,null);assert.equal(p[199].sma200,99.5);assert.equal(p[209].sma100,159.5);assert.equal(p[209].sma200,109.5);
 assert.equal(smaOf(p[209],100),159.5);assert.deepEqual(DEFAULT_SMA,[50]);
 assert.deepEqual(normalizeSmas(undefined),[50]);assert.deepEqual(normalizeSmas([]),[50]);assert.deepEqual(normalizeSmas([200,20,7]),[20,200]);
 assert.equal(recentVsLonger(p[209],[50]),null);assert.equal(recentVsLonger(p[209],[100,200]),null);
 assert.deepEqual(recentVsLonger(p[209],[50,20]),{window:20,diff:15});
});
test("majors are flagged from the event and colored gold; other rounds use the Total diverging scale",()=>{
 const rd=(round,sg)=>({round,date:`2026-01-0${round}`,sg_total:sg,sg_basis:"source_adjusted"});
 const p=performanceSeries([{id:"m",major:true,round_data:[rd(1,0)]},{id:"n",round_data:[rd(1,0)]}],0);
 assert.deepEqual(p.map(v=>v.major),[true,false]);
 const m=performanceSeries([{id:"m",major:true,round_data:[rd(3,-3)]}],0)[0],n=performanceSeries([{id:"n",round_data:[rd(1,-3)]}],0)[0];
 assert.equal(m.major,true);assert.equal(n.major,false);
 assert.equal(roundBarFill(m),MAJOR_FILL);
 assert.match(roundBarFill(n),/--negative/);assert.match(roundBarFill({value:TOTAL_RANGE*2,major:false}),/--positive\) 85%/);
 assert.equal(TOTAL_RANGE,3);assert.doesNotMatch(roundBarFill({value:0,major:false}),/--positive|--negative/);
});
