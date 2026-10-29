import { PAGE } from './page.js';
const headers={'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'};
const reply=(body,status=200,extra={})=>new Response(JSON.stringify(body),{status,headers:{...headers,...extra}});
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const hex=value=>Array.from(value,n=>n.toString(16).padStart(2,'0')).join('');
const hash=async value=>hex(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))));
async function authorized(request,expected){
  const token=(request.headers.get('authorization')||'').replace(/^Bearer /,'');
  if(!/^[a-f0-9]{64}$/.test(token))return false;
  const actual=await hash(token);let difference=actual.length^expected.length;
  for(let i=0;i<actual.length;i++)difference|=actual.charCodeAt(i)^expected.charCodeAt(i);
  return difference===0;
}
export function validateSnapshot(value){
  if(!value||value.schema!==1||!Array.isArray(value.runs)||value.runs.length>500||!Number.isFinite(Date.parse(value.generatedAt)))throw Error('Invalid snapshot');
  const allowed=new Set(['schema','generatedAt','status','total','completed','failed','pending','completedActivations','totalActivations','elapsedSeconds','etaSeconds','etaLowSeconds','etaHighSeconds','untimedJobs','runs']);
  const fields=new Set(['gpu','status','dataset','partition','seed','sigma','condition','nodes','rounds','localSteps','attackStart','completedRounds','completedActivations','totalActivations','accuracy','elapsedSeconds','etaSeconds','etaSource','phase']);
  function fieldsValid(row,keys){
    if(!row||Object.keys(row).some(k=>!keys.has(k)))throw Error('Unexpected field');
    for(const [key,item] of Object.entries(row)){
      if(key==='runs')continue;
      if(typeof item==='string'&&item.length>120)throw Error('Label too long');
      if(item!==null&&typeof item!=='number'&&typeof item!=='string')throw Error('Invalid value');
      if(typeof item==='number'&&!Number.isFinite(item))throw Error('Invalid number');
    }
  }
  fieldsValid(value,allowed);for(const row of value.runs)fieldsValid(row,fields);
  return value;
}
async function body(request){
  if(Number(request.headers.get('content-length')||0)>131072)throw Error('Payload too large');
  const raw=await request.text();if(new TextEncoder().encode(raw).length>131072)throw Error('Payload too large');
  return validateSnapshot(JSON.parse(raw));
}
export default {async fetch(request,env){
  const url=new URL(request.url);
  if(url.pathname==='/'&&request.method==='GET')return new Response('This monitor address has been retired. Open the GitHub Pages monitor link for your campaign.',{status:410,headers:{'content-type':'text/plain; charset=utf-8','cache-control':'no-store','x-robots-tag':'noindex, nofollow'}});
  if(url.pathname==='/robots.txt')return new Response('User-agent: *\nDisallow: /\n');
  if(!url.pathname.startsWith('/api/'))return reply({error:'Not found'},404);
  if(!env.BUCKET)return reply({error:'Progress storage unavailable'},503);
  try{
    if(url.pathname==='/api/feeds'&&request.method==='POST'){
      const snapshot=await body(request);const id=crypto.randomUUID();const token=hex(crypto.getRandomValues(new Uint8Array(32)));
      await env.BUCKET.put('feeds/'+id,JSON.stringify({tokenHash:await hash(token),receivedAt:new Date().toISOString(),snapshot}));
      return reply({feedId:id,writeToken:token,viewUrl:url.origin+'/?feed='+id},201);
    }
    const match=url.pathname.match(/^\/api\/feeds\/([^/]+)$/);
    if(!match||!uuid.test(match[1]))return reply({error:'Not found'},404);
    const key='feeds/'+match[1];const object=await env.BUCKET.get(key);if(!object)return reply({error:'Feed not found'},404);
    const record=await object.json();
    if(request.method==='GET'){
      const cors=request.headers.get('origin')==='https://ojaswi03.github.io'
        ?{'access-control-allow-origin':'https://ojaswi03.github.io','vary':'Origin'}:{};
      return reply({receivedAt:record.receivedAt,snapshot:record.snapshot},200,cors);
    }
    if(!['PUT','DELETE'].includes(request.method))return reply({error:'Method not allowed'},405);
    if(!await authorized(request,record.tokenHash))return reply({error:'Update credential required'},401);
    if(request.method==='DELETE'){await env.BUCKET.delete(key);return reply({deleted:true});}
    const snapshot=await body(request);
    if(Date.parse(snapshot.generatedAt)<Date.parse(record.snapshot.generatedAt))return reply({error:'Older update rejected'},409);
    const updated={...record,receivedAt:new Date().toISOString(),snapshot};
    const saved=await env.BUCKET.put(key,JSON.stringify(updated),{onlyIf:{etagMatches:object.etag}});
    if(saved===null)return reply({error:'Concurrent update; retry'},409);
    return reply({receivedAt:updated.receivedAt});
  }catch{return reply({error:'Progress update could not be accepted'},400);}
}};
