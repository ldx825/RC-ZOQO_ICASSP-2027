#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.objectives import LogisticObjective
from src.quantizer import UniformAffineQuantizer
from src.zoqo import zoqo_step
from src.rc_zoqo import rc_step
from src.utils import CSV_DIR,LOG_DIR,ensure_output_dirs,write_csv,write_json

def main():
 ensure_output_dirs(); rng=np.random.default_rng(2027); n,d=800,32; X=rng.normal(size=(n,d)); wtrue=rng.normal(size=d); y=np.where(X@wtrue+rng.normal(scale=.5,size=n)>=0,1.,-1.); o=LogisticObjective(X,y,l2=.01); rows=[]
 for b in [2,4,8]:
  for method in ['full_precision','fixed_zoqo','clipping_aware','resolution_audit','rc_zoqo']:
   q=UniformAffineQuantizer(b,-2,2); w=np.zeros(d); start_q=o.query_count; accepted=rejected=0; clipped=0; sign_mismatch=0; audit_queries=0; rr=np.random.default_rng(1000+b*17+len(method))
   for t in range(300):
    if method=='full_precision':
     # Baseline is still zeroth-order: no gradient is consulted.
     w2,step=zoqo_step(o,w,UniformAffineQuantizer(24,-100,100),.08,.03,rr)
    elif method in ('clipping_aware','rc_zoqo'):
     w2,info=rc_step(o,w,q,.08,.03,rr,clipping_aware=True); clipped+=info['clipped']; accepted+=info['resolved']; rejected+=not info['resolved']
    else:
     w2,step=zoqo_step(o,w,q,.08,.03,rr); clipped+=step.clipping_occurred; accepted+=step.comparison!=0
     if method=='resolution_audit' and t%10==0:
      # Audit uses only function values and has no gradient/L input.
      from src.audit import run_audit
      ok,k,cand,val=run_audit(o,w,q.delta,.001,12,rr); audit_queries+=2*k
      if ok: w2=q.quantize(cand)
      else: rejected+=1
    w=w2
   rows.append({'bit':b,'method':method,'objective':o.value(w),'classification_accuracy':o.accuracy(w),'gradient_norm_evaluation_only':np.linalg.norm(o.gradient(w)),'query_count':o.query_count-start_q,'accepted_update_count':accepted,'rejected_unresolved_count':rejected,'clipping_ratio':q.stats.clipping_ratio,'sign_mismatch_ratio':sign_mismatch/max(accepted,1),'audit_query_overhead':audit_queries,'wall_clock_time':'not_recorded'})
 write_csv(CSV_DIR/'exp7_logistic.csv',rows); write_json(LOG_DIR/'exp7_status.json',{'synthetic_dataset':True,'rows':len(rows)}); print('exp7 complete')
if __name__=='__main__': main()
