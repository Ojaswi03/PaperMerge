import {readFile,writeFile} from 'node:fs/promises';
const root=new URL('../',import.meta.url);
const page=(await readFile(new URL('worker/page.js',root),'utf8')).replace('export const PAGE','const PAGE');
const worker=await readFile(new URL('worker/index.js',root),'utf8');
await writeFile(new URL('dist/server/index.js',root),worker.replace("import { PAGE } from './page.js';",page));
