import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';

// Portable via environment overrides; defaults use the verified Codex bundled runtime.
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const moduleRoot=process.env.OVR_NODE_MODULES || '/Users/wangzipu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES=moduleRoot;
const require=createRequire(path.join(moduleRoot,'package.json'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const skill=process.env.OVR_PRESENTATIONS_SKILL || '/Users/wangzipu/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python=process.env.OVR_RUNTIME_PYTHON || '/Users/wangzipu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);

await fs.mkdir(path.join(root,'.build/poster'),{recursive:true});
await fs.mkdir(path.join(root,'output'),{recursive:true});
const W=841*96/25.4,H=1189*96/25.4;
const p=Presentation.create({slideSize:{width:W,height:H}}),s=p.slides.add();
const C={bg:'#F6F5EF',ink:'#15323C',teal:'#007F79',muted:'#52666B',orange:'#AD5B24',line:'#B8C7C3'};
s.background.fill=C.bg;
let seq=0;
function text(name,body,x,y,w,h,size=44,color=C.ink,bold=false){
 const box=s.shapes.add({geometry:'textbox',name:`${++seq}-${name}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 box.text=body;box.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'left',verticalAlignment:'top',insets:0,autoFit:'none',wrap:'square'};return box;
}
function line(y){s.shapes.add({geometry:'rect',name:`rule-${y}`,position:{left:140,top:y,width:W-280,height:2},fill:C.line,line:{fill:'none',width:0}})}
function diagram(x,title,body){
 s.shapes.add({geometry:'rect',name:`method-${title}`,position:{left:x,top:1200,width:840,height:250},fill:'#FFFFFF',line:{fill:C.teal,width:3}});
 text('stage-title',title,x+34,1235,772,80,46,C.teal,true);
 text('stage-body',body,x+34,1325,772,105,39,C.ink);
}

text('eyebrow','COMPUTATIONAL INTELLIGENCE  /  NEURIPS RESEARCH SHOWCASE',140,115,W-280,60,32,C.teal,true);
text('title','Open Vision Reasoner',140,220,W-280,160,134,C.ink,true);
text('subtitle','Transferring linguistic cognitive behavior for visual reasoning',145,405,W-290,95,52,C.muted);
text('draft','PRE-EVALUATION DRAFT  ·  MODEL INFERENCE NOT YET RUN',145,535,W-290,70,35,C.orange,true);
line(650);
text('rq-label','01  /  RESEARCH QUESTION',145,710,2800,65,36,C.teal,true);
text('question','Can reasoning learned in language\nhelp models reason over images?',140,805,2850,235,83,C.ink,true);
text('method-label','02  /  TWO-STAGE POST-TRAINING',145,1090,2850,75,45,C.teal,true);
diagram(145,'Qwen2.5-VL-7B','A pretrained vision-language\nstarting checkpoint');
diagram(1165,'Text-only cold start','Supervised reasoning traces;\nupdate the language module');
diagram(2185,'Multimodal RL','PPO on mixed-modality tasks;\nreward final-answer correctness');
text('arrow-1','→',1020,1250,120,130,85,C.teal,true);
text('arrow-2','→',2040,1250,120,130,85,C.teal,true);
text('method-caveat','The reward checks correctness, not reflective wording. The paper studies how useful patterns\nappear in generated reasoning; these patterns are not direct measurements of visual attention.',145,1510,2850,150,43,C.muted);
line(1730);
text('paper-label','03  /  PAPER-REPORTED EVIDENCE',145,1790,1380,80,43,C.teal,true);
text('study-label','04  /  OUR INDEPENDENT REEVALUATION',1690,1790,1350,80,43,C.teal,true);
text('chart-label','MathVision accuracy (%)',150,1910,1300,65,40,C.muted);
const chart=s.charts.add('bar',{
 position:{left:145,top:2010,width:1350,height:570},
 categories:['Qwen2.5-VL-7B','OVR-7B'],
 series:[{name:'Paper-reported accuracy',values:[25.5,51.8],fill:C.teal}],
 hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:120},
 xAxis:{textStyle:{fontSize:36,fill:C.ink},line:{fill:C.line,width:1}},
 yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:32,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:42,bold:true,fill:C.ink}},
 chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0}
});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('paper-source','Source: Wei et al., NeurIPS 2025, Table 3.\nThis comparison includes both post-training stages.',150,2620,1350,140,36,C.muted);
text('protocol-big','3,040 questions\n3 checkpoints\n9,120 planned responses',1690,1925,1300,285,64,C.ink,true);
text('protocol-body','Base / ColdStart / RL\nSame questions and image processing\nGreedy decoding; 32,768 output-token cap\nOfficial evaluator; saved raw responses',1690,2260,1310,265,42,C.ink);
text('own-result-status','OUR RESULTS: NOT RUN\nPaper values are not our measurements.',1690,2600,1320,145,40,C.orange,true);
line(2830);
text('example-label','05  /  IMAGE-GROUNDED TASK',145,2890,1400,70,43,C.teal,true);
text('analysis-label','06  /  BEHAVIOR AND ERROR ANALYSIS',1690,2890,1350,70,43,C.teal,true);
const pic=await fs.readFile(path.join(root,'data/images/6b86b273ff34fce19d6b804e.png'));
s.images.add({blob:pic.buffer.slice(pic.byteOffset,pic.byteOffset+pic.byteLength),contentType:'image/png',alt:'MathVision item 1: a train sequence with a missing number',fit:'contain',position:{left:145,top:3000,width:1350,height:230}});
text('example-question','Which number replaces the question mark?',150,3270,1350,115,43,C.ink,true);
text('example-attribution','MathVision test, item 1. Benchmark illustration only;\nno model response is shown for this item.',150,3415,1360,130,36,C.muted);
text('behaviors','Visual reflection\nDivide-and-conquer\nVisual verification\nGoal-driven visual tracing',1690,3000,1300,300,47,C.ink,true);
text('manual-study','Predefined: 60 questions, 180 trajectories.\nA second reviewer checks 20 questions.\nTrack improvements, regressions and failures.',1690,3340,1310,210,40,C.muted);
line(3630);
text('takeaway-label','TAKEAWAY TO TEST',145,3690,2850,65,36,C.teal,true);
text('takeaway','Does multimodal RL improve correctness\nand change image-linked reasoning behavior?',145,3790,2860,195,65,C.ink,true);
text('limits','Interpretation limits: output text does not establish internal attention or causal mechanisms.\nOne benchmark cannot establish general visual intelligence. Exact paper-score parity is not assumed.',145,4015,2850,140,38,C.muted);
text('references','References: Wei et al. (2025), OVR, arXiv:2507.05255 / NeurIPS 2025.\nWang et al. (2024), MathVision, NeurIPS Datasets and Benchmarks. Original task image credited to MathVision.',145,4200,2850,110,30,C.muted);
text('disclosure','AI assistance: OpenAI Codex supported code, drafting and poster preparation. Student verification pending.\nThree-person course group · Author names, group ID and exhibition date pending.',145,4340,2850,95,28,C.muted);
s.speakerNotes.text='Preparation draft, not a completed experiment. Sources: https://papers.nips.cc/paper_files/paper/2025/hash/06d86daa7001b7ea755ac4c80dc93497-Abstract-Conference.html (Table 3: baseline 25.5%, OVR 51.8%). https://github.com/mathllm/MATH-V ; dataset MathLLMs/MathVision, pinned revision in configs/protocol.json, item 1. Method diagram is an editable explanatory reconstruction. No group inference results exist yet. Names/registration and manual verification pending.';
const stage=path.join(root,'.build/poster'),candidate=path.join(stage,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const revision=process.argv[2] || 'v3';
const finalPath=path.join(root,`output/OVR_A0_PRE_EVALUATION_DRAFT_${revision}.pptx`);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','30276000,42804000','--validate-heading-fit'],
 explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[1],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(stage,`${revision}.validation.json`)});
const preview=await p.export({slide:s,format:'png',scale:0.4});
await fs.writeFile(path.join(stage,'poster-preview.png'),new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({pptx:finalPath,validationPassed:true,sha256:result.finalSha256},null,2));
