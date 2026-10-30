import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import worker from '../worker/index.js';
import {PAGE} from '../worker/page.js';

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
  const response=await worker.fetch(request('/'),{});assert.equal(response.status,410);
  assert.match(await response.text(),/retired/);const html=PAGE;
  assert.match(html,/Live configurations/);assert.match(html,/No training controls/);assert.match(html,/Updates delayed/);
  const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(source);
  assert.equal(/<script[^>]+src=/.test(html),false);
});

test('GitHub Pages can read live data without exposing write access',async()=>{
  const env={BUCKET:bucket()};const config=await (await worker.fetch(request('/api/feeds','POST',sample()),env)).json();
  const url='https://monitor.example/api/feeds/'+config.feedId;
  const read=await worker.fetch(new Request(url,{headers:{origin:'https://ojaswi03.github.io'}}),env);
  assert.equal(read.status,200);assert.equal(read.headers.get('access-control-allow-origin'),'https://ojaswi03.github.io');
  assert.equal((await worker.fetch(new Request(url,{headers:{origin:'https://other.example'}}),env)).headers.get('access-control-allow-origin'),null);
  assert.equal((await worker.fetch(request('/api/feeds/'+config.feedId,'PUT',sample()),env)).status,401);
});

test('temporary feed expires, cannot be revived, and returns readable CORS errors',async()=>{
  const env={BUCKET:bucket()};const create=request('/api/feeds','POST',sample());create.headers.set('x-monitor-lease-seconds','45');
  const config=await (await worker.fetch(create,env)).json();const path='/api/feeds/'+config.feedId;
  let record=await (await env.BUCKET.get('feeds/'+config.feedId)).json();
  assert.equal(record.leaseSeconds,45);assert.ok(Date.parse(record.expiresAt)>Date.now());
  assert.equal((await worker.fetch(request(path,'PUT',sample(),config.writeToken),env)).status,200);
  record=await (await env.BUCKET.get('feeds/'+config.feedId)).json();record.expiresAt='2000-01-01T00:00:00Z';
  await env.BUCKET.put('feeds/'+config.feedId,JSON.stringify(record));
  const read=await worker.fetch(new Request('https://monitor.example'+path,{headers:{origin:'https://ojaswi03.github.io'}}),env);
  assert.equal(read.status,410);assert.equal(read.headers.get('access-control-allow-origin'),'https://ojaswi03.github.io');
  assert.equal((await worker.fetch(request(path,'PUT',sample(),config.writeToken),env)).status,410);
  assert.equal((await worker.fetch(request(path,'DELETE',undefined,config.writeToken),env)).status,200);
  const missing=await worker.fetch(new Request('https://monitor.example'+path,{headers:{origin:'https://ojaswi03.github.io'}}),env);
  assert.equal(missing.status,404);assert.equal(missing.headers.get('access-control-allow-origin'),'https://ojaswi03.github.io');
});

function browser(){
  class Element{
    constructor(){this.children=[];this.style={};this.classList={toggle(){},add(){}};this.content='';}
    set textContent(value){this.content=String(value);this.children=[];}
    get textContent(){return this.content+this.children.map(c=>c.textContent).join('');}
    append(child){this.children.push(child);}
    replaceChildren(){this.children=[];this.content='';}
    addEventListener(){}
  }
  const elements=new Map();
  const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();}};
  const context=vm.createContext({document,URL,location:{href:'https://ojaswi03.github.io/PaperMerge/'},
    navigator:{clipboard:{writeText:async()=>{}}},setInterval(){},Date,AbortSignal});
  new vm.Script(PAGE.match(/<script>([\s\S]*?)<\/script>/)[1]).runInContext(context);
  return {context,elements};
}

test('browser table headings and cell values match the wide terminal monitor',()=>{
  const headings=[...PAGE.matchAll(/<th>(.*?)<\/th>/g)].map(m=>m[1]);
  assert.deepEqual(headings,['GPU','STATUS','DATA/SPLIT','SEED','SIGMA','ENV','CONDITION','ROUND','ACTIVATIONS','ACC','ELAPSED','ETA']);
  const {context,elements}=browser();const snapshot=sample();
  snapshot.runs[0]={...snapshot.runs[0],dataset:'mnist',partition:'dirichlet',condition:'SHARP+ACDS / attack+noise',
    completedRounds:56,completedActivations:5660,accuracy:.8952,elapsedSeconds:1257,etaSeconds:964};
  context.render({snapshot,receivedAt:new Date().toISOString()});
  assert.deepEqual(elements.get('runs').children[0].children.map(c=>c.textContent),
    ['G0','RUNNING','mnist/nonIID','2025','0.01','attack+noise','SHARP+ACDS','56/100','5660/10000','89.52%','00:20:57','00:16:04']);
  assert.match(PAGE,/<h1>CART mechanism comparison<\/h1>/);
});

test('CART and ACDS remain distinguishable and old feeds preserve their meaning',()=>{
  const {context,elements}=browser();const snapshot=sample();
  snapshot.runs=[{...snapshot.runs[0],condition:'SHARP+CART / attack+noise'},
    {...snapshot.runs[0],condition:'SHARP+ACDS / clean',sigma:0,status:'INTERRUPTED'},
    {...snapshot.runs[0],condition:'attack_noise_defense_rla_cart'}];
  context.render({snapshot,receivedAt:new Date().toISOString()});
  const cells=elements.get('runs').children.map(row=>row.children.map(c=>c.textContent));
  assert.equal(cells[0][6],'SHARP+CART');assert.equal(cells[1][6],'SHARP+ACDS');
  assert.equal(cells[1][5],'clean');assert.equal(cells[1][1],'STOPPED');
  assert.equal(cells[2][6],'both+CART');assert.equal(cells[2][5],'attack+noise');
  assert.equal(cells[0][9],'--');assert.equal(cells[0][2],'cifar10/IID');
  context.endSession();assert.equal(elements.get('runs').children[0].children[0].colSpan,12);
});

test('browser respects the delivered overall ETA and shows finish time',()=>{
  const {context,elements}=browser();const snapshot=sample();snapshot.etaSeconds=10000;
  snapshot.runs=[{...snapshot.runs[0],etaSeconds:1000},{...snapshot.runs[0],etaSeconds:2000}];
  context.render({snapshot,receivedAt:new Date().toISOString()});
  assert.equal(elements.get('eta').textContent,'02:46:40');
  assert.match(elements.get('eta-note').textContent,/Estimated finish/);
  snapshot.pending=1;context.render({snapshot,receivedAt:new Date().toISOString()});
  assert.equal(elements.get('eta').textContent,'02:46:40');
  assert.equal(context.duration(90061),'1d 01:01:01');
});
