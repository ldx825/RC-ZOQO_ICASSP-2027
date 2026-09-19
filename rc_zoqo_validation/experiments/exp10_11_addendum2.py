#!/usr/bin/env python3
"""RC-ZOQO Addendum II: curvature-weighted floor and support-aware audit."""
from pathlib import Path
import sys, csv, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42  # IEEE-safe TrueType embedding (no Type 3)
matplotlib.rcParams['ps.fonttype'] = 42

from src.utils import ROOT, ensure_output_dirs, load_config, write_csv, write_json

OUT = ROOT / 'outputs' / 'addendum2'
FIG = OUT / 'figures'
CSV = OUT / 'csv'
LOG = OUT / 'logs'

def dirs():
 for p in (OUT,FIG,CSV,LOG): p.mkdir(parents=True, exist_ok=True)

def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight'); fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight'); plt.close(fig)

def exp10(rng):
 dims=[16,64,256]; deltas=[.005,.01,.03,.1,.3]; spectra={
  'isotropic':lambda d:np.ones(d),
  'logspace_0.1_10':lambda d:np.geomspace(.1,10,d),
  'logspace_0.01_100':lambda d:np.geomspace(.01,100,d),
 }
 rows=[]; fixed=[]
 for name,maker in spectra.items():
  for d in dims:
   eig=maker(d); tr=float(eig.sum()); tr2=float(np.square(eig).sum())
   for delta in deltas:
    phase=rng.uniform(-delta/2,delta/2,size=(5000,d)); gap=.5*np.sum(eig*phase*phase,axis=1); g2=np.sum(np.square(eig)*phase*phase,axis=1)
    empirical_gap=float(gap.mean()); theory_gap=delta**2*tr/24; empirical_g2=float(g2.mean()); theory_g2=delta**2*tr2/12
    rows.append({'spectrum':name,'dimension':d,'Delta':delta,'samples':5000,'trace_H':tr,'trace_H2':tr2,'empirical_objective_gap':empirical_gap,'theory_objective_gap':theory_gap,'objective_relative_error':abs(empirical_gap/theory_gap-1),'empirical_gradient_norm_squared':empirical_g2,'theory_gradient_norm_squared':theory_g2,'gradient_squared_relative_error':abs(empirical_g2/theory_g2-1),'empirical_rms_gradient':np.sqrt(empirical_g2),'theory_rms_gradient':np.sqrt(theory_g2)})
 # Fixed instances demonstrate non-monotone individual floors.
 for i,val in enumerate([.013,.041,.077,.143]):
  for delta in deltas:
   phase=np.full(16,val); e=delta*np.rint(phase/delta)-phase; fixed.append({'instance':i,'x_star_value':val,'Delta':delta,'objective_gap':.5*np.sum(e*e),'gradient_norm':np.linalg.norm(e)})
 write_csv(CSV/'table_F_curvature_floor.csv',rows); write_csv(CSV/'exp10_fixed_instances.csv',fixed)
 fig,ax=plt.subplots(figsize=(6.2,4));
 for name in spectra:
  sub=sorted([r for r in rows if r['spectrum']==name and r['dimension']==64],key=lambda z:z['Delta']); ax.loglog([r['Delta'] for r in sub],[r['empirical_rms_gradient'] for r in sub],'o-',label=name)
 ax.set(xlabel='Resolution Delta',ylabel='RMS gradient at nearest grid',title='Curvature-weighted phase-averaged floor'); ax.grid(alpha=.25,which='both'); ax.legend(frameon=False,fontsize=8); save(fig,'fig11_curvature_weighted_floor')
 max_obj=max(r['objective_relative_error'] for r in rows); max_g2=max(r['gradient_squared_relative_error'] for r in rows); med=max(np.median([r['objective_relative_error'] for r in rows]),np.median([r['gradient_squared_relative_error'] for r in rows])); fit=[]
 for name in spectra:
  sub=[r for r in rows if r['spectrum']==name and r['dimension']==64]; sl=np.polyfit(np.log([r['Delta'] for r in sub]),np.log([r['empirical_rms_gradient'] for r in sub]),1)[0]; fit.append({'spectrum':name,'dimension':64,'log_log_rms_slope':float(sl)})
 passed=med<.03 and max_obj<.08 and max_g2<.08 and all(abs(x['log_log_rms_slope']-1)<.05 for x in fit); write_json(LOG/'exp10_status.json',{'passed':bool(passed),'median_relative_error':float(med),'max_objective_relative_error':float(max_obj),'max_gradient_squared_relative_error':float(max_g2),'fits':fit}); print('Experiment 10:',passed,'median',med,'max',max(max_obj,max_g2)); return passed

def support_success(q,k,m,norm,a,C,rng,trials=100000):
 # S intersection with the fixed k nonzero gradient coordinates is hypergeometric.
 overlap=rng.hypergeometric(k,q-k,m,size=trials); signed=2*rng.binomial(overlap,.5)-overlap; threshold=a*m*C; return float(np.mean(np.abs(signed)*norm/np.sqrt(k)>=threshold))

def exp11(rng):
 L=1.; gamma=.25; a=.01; C=gamma+L/2; qs=[128,512]; ks_by_q={q:[1,4,16,64,q] for q in qs}; ms=[1,2,4,8,16,32,64,128,256,512]; rows=[]; violations=[]
 for q in qs:
  for k in ks_by_q[q]:
   nu=1./k
   for m in [x for x in ms if x<=q]:
    norm=np.sqrt(2.)*a*C*np.sqrt(m*q); theta=a*a*m*q*C*C/(norm*norm); A=(m/q)/(nu+3*(m-1)/(q-1)*(1-nu)); p_lb=(1-theta)**2*A; p_emp=support_success(q,k,m,norm,a,C,rng)
    se=np.sqrt(max(p_emp*(1-p_emp),1e-15)/100000); violation=p_emp+3*se<p_lb
    rows.append({'q':q,'k':k,'m':m,'nu':nu,'effective_dimension':k,'gradient_norm':norm,'theta_m':theta,'A_m':A,'empirical_success_probability':p_emp,'theoretical_lower_bound':p_lb,'signal_scale_ratio':norm/(a*C*np.sqrt(m*q)),'three_sigma_violation':int(violation)})
    if violation: violations.append(rows[-1])
 write_csv(CSV/'table_G_support_audit.csv',rows)
 # Dedicated all-configuration bound plot required by the addendum.
 fig,ax=plt.subplots(figsize=(6.2,4));
 for q in qs:
  sub=[r for r in rows if r['q']==q and r['k']==q]
  ax.plot([r['m'] for r in sub],[r['empirical_success_probability'] for r in sub],'o-',ms=3,label=f'empirical q={q}')
  ax.plot([r['m'] for r in sub],[r['theoretical_lower_bound'] for r in sub],'--',lw=1,label=f'bound q={q}')
 ax.set(xscale='log',xlabel='Support size m',ylabel='Audit success probability',title='Support-size audit bound (diffuse gradients)'); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); save(fig,'fig12_support_size_audit')
 # Diffuse/concentrated panels, with all q represented.
 fig,ax=plt.subplots(figsize=(6.2,4));
 for q in qs:
  for k,style in [(q,'o-'),(1,'s--')]:
   sub=sorted([r for r in rows if r['q']==q and r['k']==k],key=lambda z:z['m']); ax.plot([r['m'] for r in sub],[r['empirical_success_probability'] for r in sub],style,ms=3,label=f'q={q}, k={k}')
 ax.set(xscale='log',xlabel='Support size m',ylabel='Audit success probability',title='Diffuse vs concentrated gradients'); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); save(fig,'fig13_diffuse_vs_concentrated_support')
 # A_m monotonicity phase transition.
 phase=[]; fig,ax=plt.subplots(figsize=(6.2,4));
 for q in qs:
  threshold=3/(q+2)
  for k in ks_by_q[q]:
   nu=1/k; sub=sorted([r for r in rows if r['q']==q and r['k']==k],key=lambda z:z['m']); vals=[r['A_m'] for r in sub]; observed='decreasing' if np.all(np.diff(vals)<=1e-12) else ('increasing' if np.all(np.diff(vals)>=-1e-12) else 'non-monotone'); predicted='decreasing' if nu<threshold else 'increasing'; phase.append({'q':q,'k':k,'nu':nu,'threshold_3_over_q_plus_2':threshold,'predicted_direction':predicted,'observed_direction':observed,'match':int(predicted==observed)}); 
   if k in (1,16,q): ax.plot([r['m'] for r in sub],vals,'o-',ms=3,label=f'q={q}, k={k}')
 ax.set(xscale='log',xlabel='m',ylabel='A_m',title='Support-size phase transition'); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8); save(fig,'fig14_support_phase_transition'); write_csv(CSV/'exp11_phase_transition.csv',phase)
 # Diffuse closed form sanity check.
 diffuse=[]
 for q in qs:
  for m in [x for x in ms if x<=q]:
   r=next(x for x in rows if x['q']==q and x['k']==q and x['m']==m); closed=(1-r['theta_m'])**2*m/(3*m-2); diffuse.append({'q':q,'m':m,'empirical_lower_bound':r['theoretical_lower_bound'],'diffuse_closed_form':closed,'absolute_difference':abs(r['theoretical_lower_bound']-closed)})
 write_csv(CSV/'exp11_diffuse_corollary.csv',diffuse)
 passed=not violations and all(x['match'] for x in phase); write_json(LOG/'exp11_status.json',{'passed':bool(passed),'configurations':len(rows),'three_sigma_violations':len(violations),'violation_examples':violations[:5],'phase_transition_all_match':all(x['match'] for x in phase)}); print('Experiment 11:',passed,'configs',len(rows),'violations',len(violations)); return passed

def main():
 dirs(); rng=np.random.default_rng(20270910); p10=exp10(rng); p11=exp11(rng); print('Addendum II:',p10 and p11)
if __name__=='__main__': main()
