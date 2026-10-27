import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const mod=process.env.OVR_NODE_MODULES || '/Users/wangzipu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require=createRequire(path.join(mod,'package.json'));
const {chromium}=require('playwright');
let executable=process.env.OVR_CHROMIUM || chromium.executablePath();
try{await fs.access(executable)}catch{executable='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'}
const browser=await chromium.launch({headless:true,executablePath:executable});
try{
 const context=await browser.newContext({offline:true,viewport:{width:1440,height:1100}});
 const page=await context.newPage();const errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto(pathToFileURL(path.join(root,'artifacts/OVR_SHARED100_CASES.html')).href);
 if(await page.locator('#cases option').count()!==6)throw new Error('Expected six rule-selected cases');
 if(!(await page.locator('#status').innerText()).includes('HUMAN REVIEW PENDING'))throw new Error('Missing review disclosure');
 for(let i=0;i<6;i++){
  await page.locator('#cases').selectOption(String(i));
  if(await page.locator('.model').count()!==2)throw new Error('Expected two real model columns');
  if(await page.locator('pre').evaluateAll(xs=>xs.some(x=>!x.textContent.trim())))throw new Error('Empty saved response');
  await page.locator('#image').evaluate(img=>img.decode());
  await page.locator('#reveal').click();
  if(!(await page.locator('#answer').isVisible()))throw new Error('Reference toggle failed');
 }
 await page.locator('#cases').selectOption('0');
 const before=await page.locator('#qtext').innerText();await page.locator('#next').click();
 if(before===await page.locator('#qtext').innerText())throw new Error('Navigation failed');
 await page.locator('#prev').click();
 await page.screenshot({path:path.join(root,'.build/shared100-desktop.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1))throw new Error('Mobile overflow');
 await page.screenshot({path:path.join(root,'.build/shared100-mobile.png'),fullPage:true});
 if(errors.length||network.length)throw new Error(JSON.stringify({errors,network}));
 const report={status:'PASS',offline:true,cases:6,models:2,all_embedded_images:true,all_saved_responses_present:true,navigation:true,reference_toggle:true,mobile_overflow:false,page_errors:errors,http_requests:network,human_content_review:'PENDING'};
 await fs.writeFile(path.join(root,'results/reports/cloud_vllm_shared1008192_v1/pilot/offline_demo_validation.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
}finally{await browser.close()}
