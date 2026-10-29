import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import worker from '../worker/index.js';

function bucket(){
  const values=new Map();let version=0;
  return {async get(key){const row=values.get(key);return row?{etag:row.etag,json:async()=>JSON.parse(row.body)}:null;},
    async put(key,body,options){if(options?.onlyIf&&values.get(key)?.etag!==options.onlyIf.etagMatches)return null;const etag=String(++version);values.set(key,{body,etag});return {etag};},
    async delete(key){values.delete(key);}};
}
const sample=()=>({schema:1,generatedAt:new Date().toISOString(),status:'RUNNING',total:1,completed:0,failed:0,pending:0,completedActivations:50,totalActivations:10000,elapsedSeconds:10,etaSeconds:1990,etaLowSeconds:1000,etaHighSeconds:3000,untimedJobs:0,runs:[{gpu:0,status:'RUNNING',dataset:'cifar10',partition:'iid',seed:2025,sigma:.01,condition:'attack_noise_defense_rla_cart',nodes:100,rounds:100,localSteps:5,attackStart:20,completedRounds:0,completedActivations:50,totalActivations:10000,accuracy:null,elapsedSeconds:10,etaSeconds:1990,etaSource:'measured',phase:'TRAINING'}]});
const request=(path,method='GET',data,token)=>new Request('https://monitor.example'+path,{method,headers:token?{authorization:'Bearer '+token}:{},body:data===undefined?undefined:JSON.stringify(data)});

test('public viewers can read progress but cannot update or delete it',async()=>{
  const env={BUCKET:bucket()};const created=await worker.fetch(request('/api/feeds','POST',sample()),env);assert.equal(created.status,201);
  const config=await created.json();assert.match(config.writeToken,/^[a-f0-9]{64}$/);const path='/api/feeds/'+config.feedId;
  const read=await (await worker.fetch(request(path),env)).json();assert.equal(read.snapshot.total,1);
  assert.equal(JSON.stringify(read).includes(config.writeToken),false);assert.equal('tokenHash' in read,false);
  assert.equal((await worker.fetch(request(path,'PUT',sample()),env)).status,401);
  assert.equal((await worker.fetch(request(path,'DELETE'),env)).status,401);
  const next=sample();next.completedActivations=100;
  assert.equal((await worker.fetch(request(path,'PUT',next,config.writeToken),env)).status,200);
  assert.equal((await (await worker.fetch(request(path),env)).json()).snapshot.completedActivations,100);
  next.generatedAt='2020-01-01T00:00:00Z';assert.equal((await worker.fetch(request(path,'PUT',next,config.writeToken),env)).status,409);
  assert.equal((await worker.fetch(request(path,'DELETE',undefined,config.writeToken),env)).status,200);
  assert.equal((await worker.fetch(request(path),env)).status,404);
});

test('private path/error fields cannot be uploaded into a public feed',async()=>{
  const env={BUCKET:bucket()};const bad=sample();bad.output='/home/private/secret';
  assert.equal((await worker.fetch(request('/api/feeds','POST',bad),env)).status,400);
  const row=sample();row.runs[0].error='private traceback';
  assert.equal((await worker.fetch(request('/api/feeds','POST',row),env)).status,400);
  assert.equal((await worker.fetch(request('/api/feeds/../../plan.json'),env)).status,404);
});

test('dashboard has a complete browser script and no remote asset dependencies',async()=>{
  const response=await worker.fetch(request('/'),{});const html=await response.text();assert.equal(response.status,200);
  assert.match(html,/Live configurations/);assert.match(html,/No training controls/);assert.match(html,/Updates delayed/);
  const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(source);
  assert.equal(/<script[^>]+src=/.test(html),false);
});
