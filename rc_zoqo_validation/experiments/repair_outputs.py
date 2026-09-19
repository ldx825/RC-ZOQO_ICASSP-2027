from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import CSV_DIR,save_figure,write_csv
rows=list(csv.DictReader((CSV_DIR/'exp3_clipping_sign_flip_raw.csv').open()))
m=np.array([float(r['normalized_margin']) for r in rows]); f=np.array([int(r['sign_flip']) for r in rows]); edges=np.linspace(0,1.5,21); centers=.5*(edges[:-1]+edges[1:]); out=[]
for l,u,c in zip(edges[:-1],edges[1:],centers):
 mask=(m>=l)&(m<u); out.append({'normalized_margin':c,'sign_flip_probability':float(f[mask].mean()) if mask.any() else float('nan'),'count':int(mask.sum())})
write_csv(CSV_DIR/'exp3_clipping_sign_flip_binned.csv',out); fig,ax=plt.subplots(figsize=(5.2,3.4)); ax.plot(centers,[r['sign_flip_probability'] for r in out],'o-',ms=3); ax.axvline(1,color='k',ls='--',lw=1); ax.set(xlabel='Normalized boundary margin / probe radius',ylabel='Sign-flip probability',ylim=(-.02,1.02)); ax.grid(alpha=.25); save_figure(fig,'fig3_clipping_sign_flip'); plt.close(fig)
