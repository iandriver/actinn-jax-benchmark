from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
r=Path(__file__).resolve().parents[2];out=r/'docs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
order=['pbmc','lung_intra','liver_intra','blood_gut_intra','liver_cross','brain_intra','brain_cluster_intra','lung_cross']
labels=['PBMC · 8 labels','Lung within · 46','Liver within · 38','Blood/gut · 86','Liver across studies · 38','Brain subclasses · 24','Brain clusters · 151','Lung across datasets* · CL']
methods=['scarches','linear-anova-pca','actinn-jax'];names=['scArches (weighted neighbors)','Linear baseline','ACTINN-JAX'];colors=['#009E73','#0072B2','#D55E00'];markers=['D','s','^']
dirs=['q2_matched_baselines','q3_baselines_rest','q3_brain_subclass','q3_brain_cluster','scarches_fix_pbmc','scarches_fix_lung_cross','scarches_fix_rest','scarches_fix_blood_gut']
x=pd.concat([pd.read_csv(r/'results'/d/'results.csv') for d in dirs]);x['total_s']=x.fit_s+x.predict_s;x['memory_gb']=x.peak_mem_mb/1000
fig,axs=plt.subplots(1,4,figsize=(10.4,6.7),sharey=True,gridspec_kw={'width_ratios':[1,1,1.1,1]})
for j,(metric,title) in enumerate([('accuracy','A  Label agreement'),('macro_f1','B  Macro-F1'),('total_s','C  Fit + predict (s)'),('memory_gb','D  Peak memory (GB)')]):
 ax=axs[j]
 for i,ds in enumerate(order):
  for k,m in enumerate(methods):
   if ds=='lung_cross' and metric=='macro_f1':continue
   display_metric='ontology_concordance' if ds=='lung_cross' and metric=='accuracy' else metric
   v=x[(x.dataset==ds)&(x.method==m)][display_metric];y=i+(k-1)*.17
   ax.plot([v.min(),v.max()],[y,y],color=colors[k],lw=1.7)
   ax.scatter(v.mean(),y,c=colors[k],marker=markers[k],s=29,zorder=3,label=names[k] if j==0 and i==0 else None)
 ax.axhspan(6.55,7.5,color='#eeeeee',zorder=0);ax.set_title(title,fontsize=11,loc='left',pad=12);ax.set_ylim(7.5,-.6);ax.grid(axis='x',alpha=.18);ax.tick_params(axis='y',length=0)
 if j==0:ax.set_xlim(.6,1.01);ax.set_xticks([.6,.7,.8,.9,1.])
 elif j==1:
  ax.set_xlim(.5,1.01);ax.set_xticks([.5,.7,.9,1.]);ax.text(.75,7,'Not reported',ha='center',va='center',fontsize=9,color='#555555')
 elif j==2:ax.set_xscale('log');ax.set_xlim(.65,1100);ax.set_xticks([1,10,100,1000]);ax.set_xticklabels(['1','10','100','1,000'])
 else:ax.set_xlim(0,8.5);ax.set_xticks([0,2,4,6,8])
axs[0].set_yticks(range(8));axs[0].set_yticklabels(labels,fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.58,.985),fontsize=10)
fig.subplots_adjust(left=.245,right=.99,top=.87,bottom=.15,wspace=.22)
fig.text(.245,.055,'Means of three repeats; bars show repeat ranges, not donor uncertainty.\n* Shaded lung row: Cell Ontology concordance (same term or lineage); macro-F1 not reported.',fontsize=9,color='#404040')
for ext in ['png','pdf','svg']:fig.savefig(out/f'fig_matched_annotation.{ext}',dpi=260,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.9));ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,text,fc,color='#202020',fontsize=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',lw=.8,edgecolor='#858585',facecolor=fc));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=color)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':'#555555','lw':1.3})
ax.text(.05,4.7,'Reference-based annotation and integrated atlas mapping',fontsize=15,weight='bold')
box(.1,2.08,1.65,.8,'Query\nexpression','#f3f3f3');arrow(1.8,2.5,2.12,3.7);arrow(1.8,2.5,2.12,1.45)
box(2.2,3.1,3.45,1.3,'Saved label classifier\nACTINN-JAX / linear baseline\nFrozen prediction for later queries','#e6f0f8',fontsize=9);arrow(5.73,3.75,6.1,3.75)
box(6.2,3.1,3.55,1.3,'Cell labels and confidence\nEvaluate label accuracy, coverage,\ntime and memory','#e6f0f8',fontsize=9)
box(2.2,.7,3.45,1.5,'Atlas model + query adaptation\nArchMap / scArches workflow\nPretrained reference can be reused','#e7f4ee',fontsize=9);arrow(5.73,1.45,6.1,1.45)
box(6.2,.7,3.55,1.5,'Shared embedding + labels\nAlso evaluate batch mixing and\nbiological conservation','#e7f4ee',fontsize=9)
ax.text(.1,.12,'This study compares annotation outputs. An ArchMap platform or integration comparison remains unmeasured.',fontsize=9,color='#404040')
for ext in ['png','pdf','svg']:fig.savefig(out/f'fig_mapping_outputs.{ext}',dpi=260,bbox_inches='tight')
plt.close(fig)
print('Two figures generated from verified CSVs and documented mapping outputs.')
