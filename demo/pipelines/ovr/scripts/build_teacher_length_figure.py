"""Scientific figures for the frozen teacher/student length supplement."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1];NS='teacher_student_lengths_v1'
p=ROOT/'results/reports'/NS;report=json.loads((p/'comparison.json').read_text());assert report['complete']
rows=[json.loads(x) for x in (p/'student_lengths.jsonl').read_text().splitlines()];teacher=[json.loads(x) for x in (p/'teacher_lengths.jsonl').read_text().splitlines()]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.4),gridspec_kw={'width_ratios':[1,1.05]})
by={g['group']:g for g in report['groups']};groups=['teacher','coldstart','rl'];colors=['#818B96','#316C9D','#CC7344'];means=[by[g]['reasoning_observed']['mean'] for g in groups];medians=[by[g]['reasoning_observed']['median'] for g in groups]
bars=ax.bar(range(3),means,color=colors,width=.6,alpha=.9,zorder=2)
bars[2].set_hatch('///')
for i,m in enumerate(medians):ax.hlines(m,i-.12,i+.12,color='black',lw=2,zorder=4,label='Median' if i==0 else None)
upper=max(means+medians)*1.25
ax.set_ylim(0,upper);ax.set_ylabel('Observed reasoning tokens');ax.set_title('A  Training reference and visual outputs',loc='left',weight='bold')
ax.set_xticks(range(3),['Teacher\ntext traces\nn='+str(len(teacher)),'ColdStart\nvisual questions\nn=20','RL\nvisual questions\nn=20'])
for i,m in enumerate(means):ax.text(i,max(m,medians[i])+upper*.035,('≥' if i==2 else '')+f'{m:,.0f}',ha='center',va='bottom',weight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y/1000:g}k'));ax.grid(axis='y',alpha=.2,zorder=0);ax.legend(frameon=False,fontsize=8,loc='upper right')
a={r['question_id']:r for r in rows if r['group']=='coldstart'};b={r['question_id']:r for r in rows if r['group']=='rl'}
closed=[q for q in a if b[q]['reasoning_right_censored'] is False];capped=[q for q in a if b[q]['reasoning_right_censored'] is True]
bx.plot([0,17500],[0,17500],color='#9CA3AA',linestyle='--',lw=1,zorder=0)
bx.scatter([a[q]['reasoning_tokens_observed'] for q in closed],[b[q]['reasoning_tokens_observed'] for q in closed],s=35,color=colors[1],label='Both reasoning sections closed (n=16)')
bx.scatter([a[q]['reasoning_tokens_observed'] for q in capped],[b[q]['reasoning_tokens_observed'] for q in capped],s=42,color=colors[2],marker='^',label='RL truncated, lower bound (n=4)')
for q in capped:
 x=a[q]['reasoning_tokens_observed'];y=b[q]['reasoning_tokens_observed'];bx.annotate('',xy=(x,y+600),xytext=(x,y+80),arrowprops={'arrowstyle':'->','color':colors[2],'lw':1})
bx.set(xlim=(0,17500),ylim=(0,17500),xlabel='ColdStart reasoning tokens',ylabel='RL observed reasoning tokens');bx.set_title('B  Same 20 visual questions',loc='left',weight='bold')
bx.xaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y/1000:g}k'));bx.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y/1000:g}k'));bx.grid(alpha=.15);bx.legend(frameon=False,fontsize=7.5,loc='lower right')
longer=sum(b[q]['reasoning_tokens_observed']>a[q]['reasoning_tokens_observed'] for q in a);bx.text(.03,.91,f'RL longer on {longer}/20\nobserved pairs',transform=bx.transAxes,va='top',fontsize=9)
fig.subplots_adjust(bottom=.26,top=.9,wspace=.37)
fig.text(.06,.08,'Teacher traces and student answers use different tasks and generation conditions; the comparison is descriptive.',fontsize=9)
fig.text(.06,.035,'The same Qwen tokenizer is used. Think delimiters and final answers are excluded. Four RL traces are censored at 16k output tokens.',fontsize=8.5)
output=ROOT/'output';stem=output/'OVR_teacher_coldstart_rl_reasoning_lengths'
for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=240,bbox_inches='tight')
print(stem)
