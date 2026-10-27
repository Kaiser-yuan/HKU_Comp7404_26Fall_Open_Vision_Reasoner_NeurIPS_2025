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
 await page.goto(pathToFileURL(path.join(root,'demo/index.html')).href);
 await page.locator('#image').waitFor();
 if(await page.locator('#cases option').count()!==6)throw new Error('Expected six initial benchmark cases');
 if(!(await page.locator('#status').innerText()).includes('NOT RUN'))throw new Error('Initial state must disclose no inference');
 if(await page.locator('.model').count()!==3)throw new Error('Missing model columns');
 const before=await page.locator('#qtext').innerText();
 await page.locator('#next').click();
 if(before===await page.locator('#qtext').innerText())throw new Error('Case navigation did not change question');
 await page.locator('#prev').click();
 await page.locator('#reveal').click();
 if(!(await page.locator('#answer').isVisible()))throw new Error('Reference toggle failed');
 const images=await page.locator('img').evaluateAll(xs=>xs.every(i=>i.complete && i.naturalWidth>0));
 if(!images)throw new Error('Embedded image did not load offline');
 await page.screenshot({path:path.join(root,'.build/demo-desktop.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1))throw new Error('Mobile horizontal overflow');
 await page.screenshot({path:path.join(root,'.build/demo-mobile.png'),fullPage:true});
 if(errors.length || network.length)throw new Error(JSON.stringify({errors,network}));
 const report={status:'PASS',offline:true,cases:6,models:3,embedded_images:true,navigation:true,reference_toggle:true,mobile_overflow:false,page_errors:errors,http_requests:network,model_inference:'NOT RUN'};
 await fs.writeFile(path.join(root,'results/reports/demo_validation.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
}finally{await browser.close()}
