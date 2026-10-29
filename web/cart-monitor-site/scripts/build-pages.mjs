import {mkdir,writeFile} from 'node:fs/promises';
import {PAGE} from '../worker/page.js';
const destination=new URL('../../../_site/',import.meta.url);
await mkdir(destination,{recursive:true});
await writeFile(new URL('index.html',destination),PAGE);
await writeFile(new URL('.nojekyll',destination),'');
console.log('Built static monitor in '+destination.pathname);
