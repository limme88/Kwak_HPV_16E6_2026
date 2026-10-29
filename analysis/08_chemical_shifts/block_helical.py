#!/usr/bin/env python3
"""Estimate per-residue blocking errors of native DSSP H occupancies.
Inputs: framewise DSSP arrays and matching saved input metadata.
Outputs: per-run/combined blocking tables and diagnostic plots.
Supports the unresolved uncertainty assessment underlying Figure 2a.
"""
from pathlib import Path
import argparse,collections,csv,json,hashlib,sys
import numpy as np
# Configuration: metadata are downloaded alongside the native DSSP inputs.
B = Path(__file__).resolve().parents[2] / 'trajectories/chemical_shifts'
from pyblock import blocking as pb

def write(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def hashfile(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def estimate(x):
    x=np.asarray(x,float);n=len(x)
    assert n>=64 and np.isfinite(x).all()
    stats=pb.reblock(x);levels=[]
    for s in stats:
        levels.append(dict(k=int(s.block),block_frames=2**s.block,nblocks=int(s.ndata),
            retained_frames=int(s.ndata*2**s.block),omitted_frames=int(n-s.ndata*2**s.block),
            mean=float(s.mean),sem=float(s.std_err),error_on_sem=float(s.std_err_err)))
    r=dict(status='no_optimum',sem=None,candidate_sem=None,block_frames=None,nblocks=None,
           regular_mean=float(x.mean()),prefix_mean=None,tail_omission_difference=None,early_late_difference=float(x[n//2:].mean()-x[:n//2].mean()))
    if np.ptp(x)==0:
        r['status']='unresolved_no_observed_transition';return r,levels
    k=pb.find_optimal_block(n,stats)[0]
    if not np.isfinite(k):return r,levels
    k=int(k);s=levels[k]
    r.update(candidate_sem=s['sem'],block_frames=s['block_frames'],nblocks=s['nblocks'],prefix_mean=s['mean'],tail_omission_difference=float(x.mean()-s['mean']))
    # The existing project's conservative screens; these are not universal guarantees.
    if k+1>=len(levels) or levels[k+1]['nblocks']<32:r['status']='insufficient_blocks';return r,levels
    higher=levels[k+1]
    if abs(s['sem']-higher['sem'])>2*np.hypot(s['error_on_sem'],higher['error_on_sem']):
        r['status']='no_adjacent_level_plateau';return r,levels
    m=x[:s['retained_frames']].reshape(-1,s['block_frames']).mean(axis=1);mid=len(m)//2
    diff=float(m[mid:].mean()-m[:mid].mean())
    se=float(np.sqrt(m[mid:].var(ddof=1)/len(m[mid:])+m[:mid].var(ddof=1)/len(m[:mid])))
    if abs(diff)>3*se:r['status']='early_late_drift';return r,levels
    if abs(r['tail_omission_difference'])>s['sem']:
        r['status']='tail_omission_sensitive';return r,levels
    r.update(status='estimated_for_stated_window',sem=s['sem'])
    return r,levels

def weights(t):
    d=np.diff(t);assert np.isfinite(t).all() and (d>0).all()
    w=np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2];return w/w.sum()

def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    prov=json.loads((B/'inputs/dssp_provenance.json').read_text())
    saved={(int(r['residue']),r['definition']):r for r in csv.DictReader(open(B/'inputs/dssp_combined.csv'))}
    a=json.loads((B/'inputs/summary_provenance.json').read_text())['weights']
    rows=[];alllevels=[];metadata=[]
    for rr in prov['runs']:
        name=rr['run'];p=source/f'{name}.npz'
        assert hashfile(p)==rr['codes_sha256'],f'Framewise DSSP file identity differs: {p}'
        z=np.load(p,allow_pickle=False);codes=z['codes'].astype('U2');t=z['times_ps']
        assert codes.shape==(rr['frames'],151) and np.array_equal(z['residues'],np.arange(1,152))
        assert set(np.unique(codes))<=set('HBEGITS ')
        # Same small-interval handling used in the prior native SPARTA+ blocking analysis.
        keep=np.r_[True,np.diff(t)>50.];rt=t[keep];spacing=float(np.median(np.diff(rt)))
        assert np.max(abs(np.diff(rt)-spacing))<.03 and abs(spacing-100.002)<.03
        x=(codes=='H').astype(float);fullmean=weights(t)@x;regular=x[keep]
        metadata.append(dict(run=name,full_frames=len(t),blocking_frames=len(rt),short_interval_frames_removed=int((~keep).sum()),spacing_ps=spacing,duration_us=(t[-1]-t[0])/1e6))
        for i in range(151):
            r,levels=estimate(regular[:,i]);delta=float(fullmean[i]-r['regular_mean'])
            if r['sem'] is not None and abs(delta)>max(1e-5,.1*r['sem']):
                r.update(status='weighted_mean_vs_regular_mean_sensitive',sem=None)
            rows.append(dict(run=name,residue=i+1,definition='alpha_H',full_mean=float(fullmean[i]),n_regular=len(rt),cadence_ps=spacing,weighted_minus_regular=delta,**r))
            for level in levels:alllevels.append(dict(run=name,residue=i+1,block_ns=level['block_frames']*spacing/1000,**level))
        print(f'{name}: blocking complete',flush=True)
    lookup={(r['run'],r['residue']):r for r in rows};combined=[]
    for i in range(1,152):
        rs=[lookup[(name,i)] for name in a]
        mean=sum(a[r['run']]*r['full_mean'] for r in rs)
        assert abs(mean-float(saved[(i,'alpha_H')]['mean']))<1e-10
        ok=all(r['status']=='estimated_for_stated_window' for r in rs)
        sem=float(np.sqrt(sum((a[r['run']]*r['sem'])**2 for r in rs))) if ok else None
        combined.append(dict(residue=i,definition='alpha_H',mean=mean,sem=sem,status='estimated_for_stated_window' if ok else 'unresolved',
                             unresolved_runs=';'.join(r['run']+':'+r['status'] for r in rs if r['status']!='estimated_for_stated_window')))
    write(out/'blocking_per_run.csv',rows);write(out/'blocking_levels.csv',alllevels);write(out/'blocking_combined.csv',combined)
    (out/'BLOCKING_REVIEW.json').write_text(json.dumps(dict(runs=metadata,within_run_status=dict(collections.Counter(r['status'] for r in rows)),
        combined_status=dict(collections.Counter(r['status'] for r in combined)),pyblock_commit='a293b5b75dfa862def45a077824aa8df08bcbf46',
        pyblock_module_sha256=hashfile(B/'vendor/pyblock_blocking.py'),mean='Original all-native-frame physical-time quadrature',
        sem='Pointwise conditional within-run SEM from pyblock on nearly regular native cadence; fixed duration-weight variance propagation',
        approximation='Short-interval saved endpoints excluded only from blocking. Difference between weighted and regular means reported and screened.',
        windows='Full production interval; no fit-driven equilibration exclusion. Drift-sensitive series unresolved.',
        required_review='Inspect block curves/window behavior, especially E6HL and C-terminal tail; screens do not certify equilibrium.',
        excluded_error_sources=['between-run heterogeneity','force-field differences including July','shared starting-structure bias','Delta2D forward-model error','experimental uncertainty'],
        no_estimate='No event, no optimum, too few blocks, unstable levels or drift do not justify a zero SEM or a substitute range band.'),indent=2))
    diagnostics(out,rows,alllevels)

def diagnostics(out,rows,levels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    selected=[15,44,66,69,72,73,76,88,116,138,140,145]
    with PdfPages(out/'blocking_diagnostics.pdf') as pdf:
        for name in dict.fromkeys(r['run'] for r in rows):
            fig,axs=plt.subplots(4,3,figsize=(10,10),layout='constrained')
            for ax,i in zip(axs.flat,selected):
                ls=[r for r in levels if r['run']==name and r['residue']==i]
                res=next(r for r in rows if r['run']==name and r['residue']==i)
                ax.errorbar([r['block_ns'] for r in ls],[r['sem'] for r in ls],yerr=[r['error_on_sem'] for r in ls],fmt='.-',lw=.8)
                if res['block_frames']:ax.axvline(res['block_frames']*res['cadence_ps']/1000,color='orange')
                ax.set(xscale='log',xlabel='Block duration (ns)',ylabel='SEM',title=f'{i}: {res["status"]}')
                ax.title.set_fontsize(7)
            fig.suptitle(name);pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dssp-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.dssp_dir,args.out)
