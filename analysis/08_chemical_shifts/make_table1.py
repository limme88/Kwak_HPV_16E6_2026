"""Recompute pooled chemical-shift RMSD and Pearson r for manuscript Table 1.
Input: sequence-matched duration-weighted experimental residual CSV.
Outputs: table_values.csv and pooled_shift_pairs.csv.
The original statistical calculation is retained; Word formatting is omitted.
"""
from pathlib import Path
import csv,math,json,hashlib
import numpy as np

# Configuration: repository-relative input and output paths.
REPO = Path(__file__).resolve().parents[2]
source = REPO / 'data/chemical_shifts/experimental_residuals.csv'
B = REPO / 'results/chemical_shifts/table1'
B.mkdir(parents=True, exist_ok=True)
rows=[r for r in csv.DictReader(source.open()) if r['ensemble']=='duration_weighted' and r['mask_primary']=='True']
assert len(rows)==558
assert all(r['sequence_match']=='True' and r['domain'] in ('E6N','E6C') for r in rows)
assert len({(r['domain'],r['source_row_id'],r['atom']) for r in rows})==len(rows)
atoms=['CA','HN','N','CB'];stats=[]
for atom in atoms:
    rr=[r for r in rows if r['atom']==atom]
    x=np.array([float(r['shift_ppm']) for r in rr]);y=np.array([float(r['predicted_mean_ppm']) for r in rr])
    rms=float(np.sqrt(np.mean((x-y)**2)));cor=float(np.corrcoef(x,y)[0,1])
    xc=x-x.mean();yc=y-y.mean()
    assert abs(cor-float(np.sum(xc*yc)/np.sqrt(np.sum(xc**2)*np.sum(yc**2))))<1e-12
    ss=0;counts={}
    for domain in ('E6N','E6C'):
        dr=[r for r in rr if r['domain']==domain];counts[domain]=len(dr)
        ss+=sum((float(r['shift_ppm'])-float(r['predicted_mean_ppm']))**2 for r in dr)
    assert abs(rms-math.sqrt(ss/len(rr)))<1e-12
    stats.append(dict(nucleus=atom,RMSD_ppm=rms,Pearson_r=cor,n_assignments=len(rr),E6N_n=counts['E6N'],E6C_n=counts['E6C']))
for name,table in [('table_values.csv',stats),('pooled_shift_pairs.csv',rows)]:
    with (B/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

