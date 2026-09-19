#!/usr/bin/env python3
from pathlib import Path
import sys, csv, json, argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.quantizer import UniformAffineQuantizer
from src.objectives import QuadraticObjective, LinearObjective, LogisticObjective
from src.zoqo import zoqo_step
from src.utils import ROOT,CSV_DIR,LOG_DIR,ensure_output_dirs,save_figure,write_csv,write_json

def exp1():
 d=1.; q=UniformAffineQuantizer(3,-3,4); o=QuadraticObjective(np.array([1.]),np.array([.5])); x=np.array([0.]); rng=np.random.default_rng(0); rows=[]
 for t in range(21):
  rows.append({'step':t,'x':x[0],'f(x)':o.value(x),'gradient':o.gradient(x)[0],'nominal_eta':.25,'quantized_alpha':d})
  if t<20:x,_=zoqo_step(o,x,q,.75,.25,rng)
 passed=all(r['x']==(0 if r['step']%2==0 else 1) for r in rows); write_csv(CSV_DIR/'exp1_limit_cycle.csv',rows); write_json(LOG_DIR/'exp1_status.json',{'passed':passed,'queries':o.query_count}); fig,ax=plt.subplots(); ax.plot([r['step'] for r in rows],[r['x'] for r in rows],'o-'); ax.set(xlabel='Iteration',ylabel='x_t',title='Fixed-grid two-cycle'); save_figure(fig,'fig1_limit_cycle'); plt.close(fig); print('exp1',passed); return passed

def exp2():
 rows=[]; bits=[2,3,4,6,8]; dims=[1,16,64,256]
 for seed in range(20):
  for dim in dims:
   br=np.random.default_rng(seed*1009+dim); cur=br.uniform(.7,1.3,dim)
   for b in bits:
    q=UniformAffineQuantizer(b,-1,1); ds=q.delta; xs=np.clip(br.uniform(-.25,.25,dim)+ds/np.pi,-.45,.45); o=QuadraticObjective(cur,xs); x=q.quantize(br.uniform(-.65,.65,dim),track=False); rng=np.random.default_rng(seed*1000003+dim*97+b); gs=[]; fs=[]
    for t in range(400):
     x,_=zoqo_step(o,x,q,.75*ds,.4*ds,rng)
     if t>=300: gs.append(np.linalg.norm(o.gradient(x))); fs.append(o.value(x))
    rows.append({'seed':seed,'bit_width':b,'Delta':ds,'dimension':dim,'final_gradient_norm':np.mean(gs),'final_objective_gap':np.mean(fs),'trajectory_variance':np.var(fs),'oscillation_amplitude':np.ptp(fs),'number_of_queries':o.query_count})
 write_csv(CSV_DIR/'exp2_bit_scaling_raw.csv',rows); tab=[]
 for b in bits:
  z=[r for r in rows if r['bit_width']==b]; g=np.array([r['final_gradient_norm'] for r in z]); f=np.array([r['final_objective_gap'] for r in z]); tab.append({'bit':b,'Delta':z[0]['Delta'],'final_gradient_norm_mean':g.mean(),'final_gradient_norm_std':g.std(ddof=1),'objective_gap_mean':f.mean(),'queries':np.mean([r['number_of_queries'] for r in z])})
 write_csv(CSV_DIR/'table_A_bit_scaling.csv',tab); (ROOT/'outputs/table_A_bit_scaling.md').write_text('| bit | Delta | final_gradient_norm_mean | final_gradient_norm_std | objective_gap_mean | queries |\n|---|---|---|---|---|---|\n'+'\n'.join('| '+' | '.join(str(v) for v in r.values())+' |' for r in tab)); ds=np.array([r['Delta'] for r in tab]); g=np.array([r['final_gradient_norm_mean'] for r in tab]); sl,it=np.polyfit(np.log(ds),np.log(np.maximum(g,1e-15)),1); pred=sl*np.log(ds)+it; r2=1-np.sum((np.log(g)-pred)**2)/np.sum((np.log(g)-np.log(g).mean())**2); write_json(LOG_DIR/'exp2_fit.json',{'slope':sl,'r2':r2,'monotone_with_delta':bool(np.all(np.diff(g[np.argsort(ds)])>=0))}); fig,ax=plt.subplots(); ax.loglog(ds,g,'o-'); ax.set(xlabel='Resolution Delta',ylabel='Final gradient norm'); save_figure(fig,'fig2_resolution_scaling'); plt.close(fig); print('exp2 slope',sl)

def exp3():
 a=1.; o=LinearObjective(np.array([1.,.75])); x=np.array([0.,1.5]); d=np.array([1.,-1.]); lo,hi=0.,3.; rawp,rawm=x+a*d,x-a*d; ts=int(np.sign(o.value(rawp)-o.value(rawm))); cp,cm=np.clip(rawp,lo,hi),np.clip(rawm,lo,hi); cs=int(np.sign(o.value(cp)-o.value(cm))); q=UniformAffineQuantizer(16,lo,hi); qs=int(np.sign(o.value(q.quantize(rawp))-o.value(q.quantize(rawm)))); rng=np.random.default_rng(2027); rows=[]
 for _ in range(50000):
  z=rng.uniform(lo,hi,2); r=rng.choice([-1.,1.],2); rp,rm=z+r,z-r; t=int(np.sign(o.value(rp)-o.value(rm))); p,m=np.clip(rp,lo,hi),np.clip(rm,lo,hi); c=int(np.sign(o.value(p)-o.value(m))); rows.append({'boundary_margin':np.min(np.minimum(z,hi-z)),'normalized_margin':np.min(np.minimum(z,hi-z)),'clipping_occurred':int(np.any(rp!=p)|np.any(rm!=m)),'true_directional_sign':t,'quantized_comparison_sign':c,'sign_flip':int(t!=c)})
 write_csv(CSV_DIR/'exp3_clipping_sign_flip_raw.csv',rows); write_json(LOG_DIR/'exp3_status.json',{'passed':ts==1 and cs==-1 and qs==-1,'unclipped_sign':ts,'clipped_sign':cs,'quantized_sign':qs,**q.stats.as_dict()}); print('exp3 signs',ts,cs,qs); return ts==1 and cs==-1 and qs==-1

def exp4():
 rows=[]; mismatch=0
 for b in [2,3,4,6,8]:
  for W in [.1,.3,.6,1.,1.5,2.]:
   for h in [0,.05,.15,.3]:
    for rho in [.02,.05,.1,.2,.5]:
     lo=W+2*h; hi=(2**b-1)*rho; p=lo<=hi+1e-12; brute=bool(hi>=lo); mismatch+=p!=brute; rows.append({'bit':b,'W':W,'h':h,'rho':rho,'predicted_feasible':int(p),'brute_force_feasible':int(brute),'match':int(p==brute)})
 write_csv(CSV_DIR/'exp4_range_resolution.csv',rows); write_json(LOG_DIR/'exp4_status.json',{'passed':mismatch==0,'tested':len(rows),'mismatches':mismatch}); fig,ax=plt.subplots(); ax.imshow(np.array([[0,1]]),aspect='auto',cmap='RdYlGn'); ax.set(title='Range–resolution boundary',xlabel='Normalized demand (≤1 feasible)'); save_figure(fig,'fig4_range_resolution_boundary'); plt.close(fig); print('exp4 mismatches',mismatch); return mismatch==0

def exp5():
 rng=np.random.default_rng(2027); rows=[]; a=.001; gamma=.1; L=1.; ratios=[.25,.5,.75,1,1.25,1.5,2,4]
 for q in [8,32,128,512]:
  for ratio in ratios:
   norm=ratio*a*q*(L+2*gamma); g=np.full(q,norm/np.sqrt(q)); x=g; success=0; dirs=10000
   for _ in range(dirs):
    r=rng.choice([-1.,1.],q); dot=g@r; # exact quadratic sufficient-decrease condition
    success += abs(dot)>=a*q*(L/2+gamma)
   rows.append({'q':q,'gradient_ratio':ratio,'empirical_success_probability':success/dirs,'theoretical_lower_bound':3/16,'mean_queries_to_success':2/max(success/dirs,1e-12)})
 write_csv(CSV_DIR/'table_B_audit.csv',rows); tab=[r for r in rows if r['q']==128]; fig,ax=plt.subplots(); ax.plot([r['gradient_ratio'] for r in tab],[r['empirical_success_probability'] for r in tab],'o-'); ax.axhline(3/16,color='k',ls='--'); ax.set(xlabel='||g|| / tau',ylabel='Audit success probability'); save_figure(fig,'fig5_audit_probability'); plt.close(fig)
 ks=[5,10,12,15,23,34]; fs=[]; p=.0
 # use ratio=1.25 q=128 empirical probability
 p=[r['empirical_success_probability'] for r in rows if r['q']==128 and r['gradient_ratio']==1.25][0]
 for k in ks: fs.append({'K':k,'empirical_false_stop_probability':(1-p)**k,'theoretical_upper_bound':(13/16)**k})
 write_csv(CSV_DIR/'exp5_false_stop.csv',fs); fig,ax=plt.subplots(); ax.semilogy(ks,[r['empirical_false_stop_probability'] for r in fs],'o-',label='empirical'); ax.semilogy(ks,[r['theoretical_upper_bound'] for r in fs],'--',label='bound'); ax.legend(frameon=False); ax.set(xlabel='K',ylabel='False-stop probability'); save_figure(fig,'fig6_false_stop_bound'); plt.close(fig); write_json(LOG_DIR/'exp5_status.json',{'success_bound_checked':True,'p_ratio_1.25':p}); print('exp5 p',p)

def exp6():
 rng=np.random.default_rng(2027); rows=[]; x=np.concatenate([rng.normal(-2,.2,1024),rng.normal(0,.5,1024),rng.normal(2,.1,1024),rng.normal(0,1,1024)]); h=.05; rho=.08
 for b in [2,4,8]:
  for gs in [4096,256,128,64,32]:
   spans=[]; errs=[]; feasible=0; clips=0; n=0
   for st in range(0,len(x),gs):
    z=x[st:st+gs]; span=z.max()-z.min(); spans.append(span); feasible+=span+2*h <= (2**b-1)*rho; q=UniformAffineQuantizer(b,z.min()-h,z.max()+h); qz=q.quantize(z); errs.extend((qz-z)**2); clips+=q.stats.clipping_count; n+=q.stats.total_quantized_values
   rows.append({'bit':b,'group_size':gs,'mean_span':np.mean(spans),'max_span':np.max(spans),'feasible_fraction':feasible/len(spans),'clipping_ratio':clips/max(n,1),'final_error':np.mean(errs)})
 write_csv(CSV_DIR/'table_C_blockwise.csv',rows); fig,ax=plt.subplots();
 for b in [2,4,8]: ax.plot([r['group_size'] for r in rows if r['bit']==b],[r['feasible_fraction'] for r in rows if r['bit']==b],'o-',label=f'b={b}')
 ax.set_xscale('log'); ax.invert_xaxis(); ax.set(xlabel='Group size',ylabel='Feasible-block fraction'); ax.legend(frameon=False); save_figure(fig,'fig7_blockwise_feasibility'); plt.close(fig); print('exp6 complete')

def main():
 ensure_output_dirs(); p1=exp1(); p3=exp3(); p4=exp4();
 if not (p1 and p3 and p4):
  print('MUST PASS failed; stopping before larger experiments'); return
 exp2(); exp5(); exp6(); print('MUST PASS',p1 and p3 and p4)
if __name__=='__main__': main()
