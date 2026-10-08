import assert from "node:assert/strict";
import test from "node:test";
import {gzipSync,gunzipSync} from "node:zlib";
const worker=(await import("../dist/server/index.js")).default;
const handleOverridesApi=(request,env)=>worker.fetch(request,{...env,ASSETS:{fetch:async()=>new Response("nf",{status:404})}},{waitUntil(){},passThroughOnException(){}});

test("compressed profile reads preserve exactly one gzip layer and manual Workers encoding",async()=>{
  const bytes=gzipSync('{"profiles":{"17639":{"name":"Blair"}}}');
  const NativeResponse=globalThis.Response;const observed=[];
  globalThis.Response=class extends NativeResponse{constructor(body,init){super(body,init);observed.push(init);}};
  try{
    const bucket={get:async()=>({body:bytes,httpEtag:'"dossier"',writeHttpMetadata(h){h.set("content-encoding","gzip");}})};
    const request=new Request("https://golf.example/api/golfprice/player_profiles/deep-hash/batch_0001.json");
    const result=await handleOverridesApi(request,{DASHBOARD_DATA:bucket});
    assert.equal(result.status,200);assert.equal(result.headers.get("content-encoding"),"gzip");
    assert.equal(observed.at(-1).encodeBody,"manual");
    assert.equal(JSON.parse(gunzipSync(Buffer.from(await result.arrayBuffer()))).profiles["17639"].name,"Blair");
    const head=await handleOverridesApi(new Request(request.url,{method:"HEAD"}),{DASHBOARD_DATA:bucket});
    assert.equal(await head.text(),"");assert.equal(head.headers.get("content-encoding"),"gzip");
    assert.equal(observed.at(-1).encodeBody,"manual");
    const plain={get:async()=>({body:'{"count":1}',httpEtag:'"catalog"',writeHttpMetadata(){}})};
    const normal=await handleOverridesApi(request,{DASHBOARD_DATA:plain});
    assert.equal(observed.at(-1).encodeBody,undefined);assert.deepEqual(await normal.json(),{count:1});
    const review=await handleOverridesApi(new Request("https://golf.example/api/golfprice/player_profiles/dossier-review.json"),{DASHBOARD_DATA:plain});
    assert.equal(review.headers.get("cache-control"),"no-cache");
  }finally{globalThis.Response=NativeResponse;}
});
