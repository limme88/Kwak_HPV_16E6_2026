#!/usr/bin/env python3
"""Run experimental Delta2D, native-frame DSSP, and residue-resolved comparison.
Inputs: experimental assignments, sequence, native trajectory archive and Delta2D.
Outputs: predictions, DSSP occupancies, comparison tables and diagnostic plots.
Supplies upstream inputs for Figure 2; use make_figure2.py for final plotting.
"""
import argparse, collections, csv, gzip, hashlib, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

# Configuration: download raw assignments and archive metadata into this directory.
BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / 'trajectories/chemical_shifts/inputs'
RUNS = ['native_20250414', 'native_20250505', 'native_20250710', 'native_20250520']
THREE = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))
NUCLEI = ['HA','CA','CB','CO','N','HN']
DEFINITIONS = {'alpha_H':'H', 'beta_E':'E', 'helix_HGI':'HGI', 'beta_EB':'EB'}

def digest(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def table(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))

def write_table(p, rows):
    assert rows, f'No rows: {p}'
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def dump(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n')

def sequence():
    return ''.join(x.strip() for x in (INPUT/'apo16e6_151aa.fasta').read_text().splitlines() if not x.startswith('>'))

def fresh(p): p.mkdir(parents=True,exist_ok=False)

def prepare(out):
    fresh(out)
    wt=sequence(); assert len(wt)==151
    raw=table(INPUT/'experimental_all_rows.csv'); assert len(raw)==1689
    nseq=re.search(r'_Entity.Polymer_seq_one_letter_code\s*\n;\s*\n(.*?)\n;', (INPUT/'17967.str').read_text(), re.S).group(1)
    nseq=''.join(nseq.split()); assert len(nseq)==84 and nseq[:2]=='GA'
    cseq=list(wt[79:])
    for r in [80,97,111,140]: cseq[r-80]='S'
    cseq='GAM'+''.join(cseq)
    # Preserve full declared E6N polymer; also calculate the paper's 1-80 boundary.
    variants=[('E6N','E6N',-1,nseq,80), ('E6C','E6C',77,cseq,151),
              ('E6N_end80','E6N',-1,nseq[:82],80)]
    allmap=[]; provenance=[]
    for name,domain,start,seq,display_end in variants:
        rows=[r for r in raw if r['domain']==domain]
        by=collections.defaultdict(list)
        for r in rows: by[int(r['author_residue'])].append(r)
        assert all(start<=i<start+len(seq) for i in by)
        shifts=['#NUM AA HA CA CB CO N HN']
        for index,aa in enumerate(seq,1):
            author=start+index-1
            rs=by[author]
            assert all(THREE[r['experimental_resname']]==aa for r in rs), (name,author,aa)
            md=author if (1<=author<=80 if domain=='E6N' else 80<=author<=151) else None
            vals={n:[] for n in NUCLEI}
            for r in rs:
                atom={'H':'HN','C':'CO'}.get(r['experimental_atom'],r['experimental_atom'])
                if atom not in vals: continue
                if aa=='G' and atom in ('HA','CB'): continue # ignored by upstream Delta2D
                if aa=='P' and atom in ('N','HN'): continue # ignored by upstream Delta2D
                if r['ambiguity_code']!='1': continue
                v=float(r['shift_ppm']); assert np.isfinite(v) and 0<v<900
                vals[atom].append((v,r['source_row_id']))
            for atom,vs in vals.items():
                assert len(vs)<=1, (name,author,atom,'duplicate assignment: resolve explicitly')
                if vs: provenance.append(dict(variant=name,author_residue=author,atom=atom,shift_ppm=vs[0][0],source_row_id=vs[0][1]))
            shifts.append(f'{index} {aa} '+' '.join(f'{vals[n][0][0]:.6f}' if vals[n] else '0' for n in NUCLEI))
            match=md is not None and aa==wt[md-1]
            mutations=[47,80] if domain=='E6N' else [80,97,111,140]
            allmap.append(dict(variant=name,domain=domain,delta_index=index,author_residue=author,
                md_residue=md if md is not None else '',experimental_aa=aa,md_aa=wt[md-1] if md else '',
                sequence_match=int(match),input_nuclei=sum(bool(v) for v in vals.values()),
                mutation_neighbor_2=int(any(abs(author-m)<=2 for m in mutations)),
                construct_edge_2=int(author<(1 if domain=='E6N' else 80)+2 or author>display_end-2),
                E6HL=int(md is not None and 62<=md<=79)))
        (out/f'{name}.tab').write_text('\n'.join(shifts)+'\n')
    write_table(out/'residue_map.csv',allmap); write_table(out/'shift_provenance.csv',provenance)
    dump(out/'preparation.json',dict(source_rows=len(raw),sequences={x[0]:x[3] for x in variants},
        source_hashes={p.name:digest(p) for p in INPUT.iterdir() if p.is_file()},
        missing='0 is the upstream input sentinel only; output missing values are blank/NaN',
        nuclei_policy='Unambiguous experimental shifts only; no inferred glycine HA; no invented values',
        cysteine='C denotes non-disulfide cysteine; Zn coordination is not encoded as disulfide X',
        E6N_boundary='Declared -1..82 primary; -1..80 sensitivity. Neither silently joins E6C.'))

def nmr(out, prepared, binary, database):
    fresh(out)
    mapping=table(prepared/'residue_map.csv'); result=[]
    for variant in ['E6N','E6C','E6N_end80']:
        rawout=out/f'{variant}.raw.txt'
        command=[str(binary.resolve()),'-file',str((prepared/f'{variant}.tab').resolve()),
                 '-shifty','-pH','neutral','-fres','1','-out',str(rawout.resolve())]
        env=dict(os.environ,CAMDBV3=str(database.resolve()))
        proc=subprocess.run(command,capture_output=True,text=True,env=env,cwd=out)
        (out/f'{variant}.stdout.log').write_text(proc.stdout)
        (out/f'{variant}.stderr.log').write_text(proc.stderr)
        if proc.returncode or not rawout.exists() or '#DONE!' not in rawout.read_text():
            raise RuntimeError(f'Delta2D failed; inspect {out}')
        found={}; text=rawout.read_text()
        assert '#num' in text and 'Helix      Beta       Coil       PPII' in text
        for line in text.splitlines():
            if not line.strip() or line.startswith('#'): continue
            parts=line.split(); assert len(parts)==7, line
            i=int(parts[0]); assert i not in found
            pp=np.array([float(x) for x in parts[2:6]])
            assert np.isfinite(pp).all() and (pp>=0).all() and (pp<=1).all()
            assert abs(pp.sum()-1)<.0021, line # starred outputs rounded to 3 decimals
            found[i]=(parts[1],pp,'*' in parts[6])
        relevant=[m for m in mapping if m['variant']==variant]
        assert set(found)<=set(int(m['delta_index']) for m in relevant)
        for m in relevant:
            f=found.get(int(m['delta_index']))
            if f: assert f[0]==m['experimental_aa']
            # A zero-input residue never becomes an observed NMR datum, even if software fills it.
            observed=int(m['input_nuclei'])>0
            status='missing' if not f else ('neighbor_inferred' if f[2] else 'direct')
            primary=bool(f and not f[2] and observed and m['sequence_match']=='1')
            result.append(dict(**m,status=status,primary_eligible=int(primary),
                helix=f[1][0] if f else '',beta=f[1][1] if f else '',
                coil=f[1][2] if f else '',ppii=f[1][3] if f else ''))
    write_table(out/'delta2d_profiles.csv',result)
    dump(out/'nmr_provenance.json',dict(binary_sha256=digest(binary),database_commit='3c390df3a5a934dd9fc9d49932bafe8821334812',
        settings='-shifty -pH neutral -fres 1; upstream built-in neighbor averaging retained',
        reliability='direct >=3 usable nuclei after upstream range filtering; * neighbor-inferred; missing retained',
        counts={v:dict(collections.Counter(r['status'] for r in result if r['variant']==v)) for v in ['E6N','E6C','E6N_end80']}))

def time_weights(t):
    t=np.asarray(t,dtype=float); assert len(t)>1 and np.isfinite(t).all() and (np.diff(t)>0).all()
    dt=np.diff(t); w=np.empty(len(t)); w[0]=dt[0]/2; w[-1]=dt[-1]/2
    w[1:-1]=(dt[:-1]+dt[1:])/2
    assert np.isclose(w.sum(),t[-1]-t[0]); return w

def dssp(out,root,chunk):
    import mdtraj as md
    fresh(out); specification=json.loads((INPUT/'LARGE_FILES_ON_HPC.json').read_text())
    manifest=root/'exports/frame_manifest.csv'
    assert digest(manifest)==specification['authoritative_frame_manifest']['sha256'], 'Native manifest changed: audit before proceeding'
    times={r:[] for r in RUNS}
    with open(manifest,newline='') as f:
        for row in csv.DictReader(f):
            run=row['native_run_id']; assert run in times
            assert int(row['run_frame_index_0'])==len(times[run])
            assert row['retained_status']=='retained' and row['production_status']=='production'
            times[run].append(float(row['run_elapsed_time_ps']))
    reports=[]
    for run in RUNS:
        info=next(x for x in specification['archive']['runs'] if x['run_id']==run)
        top=root/'exports'/run/f'{run}_protein_H_Zn.pdb'
        traj=root/'exports'/run/f'{run}_protein_H_Zn.xtc'
        for p in [top,traj]:
            expected=next(v['sha256'] for k,v in info['files'].items() if Path(k).name==p.name)
            assert digest(p)==expected, f'Coordinate identity changed: {p}'
        topology=md.load_topology(str(top)); assert topology.n_atoms==2549
        # Native archive groups heavy atoms before H: avoid duplicate hydrogen-only residues.
        # MDTraj DSSP uses backbone heavy atoms and reconstructs its DSSP H geometry.
        protein=topology.select('protein and not element H'); pt=topology.subset(protein)
        assert pt.n_atoms==1279 and pt.n_residues==151 and pt.n_chains==1
        assert [r.resSeq for r in pt.residues]==list(range(1,152))
        assert ''.join(THREE[r.name] for r in pt.residues)==sequence()
        for res in pt.residues:
            assert {'N','CA','C','O'} <= {a.name for a in res.atoms}
        frames=len(times[run]); assert frames==info['nframes']
        t=np.array(times[run]); time_weights(t)
        codes=np.empty((frames,151),dtype='S1'); offset=0
        for tr in md.iterload(str(traj),top=str(top),atom_indices=protein,chunk=chunk):
            # Continuity audit: avoid DSSP across an imaged/broken backbone.
            ca=tr.topology.select('name CA')
            distance=np.linalg.norm(tr.xyz[:,ca[1:],:]-tr.xyz[:,ca[:-1],:],axis=-1)
            assert distance.min()>.20 and distance.max()<.50, 'Broken/improperly imaged protein: audit native export'
            c=md.compute_dssp(tr,simplified=False)
            assert c.shape==(len(tr),151) and set(np.unique(c))<=set('HBEGITS ')
            codes[offset:offset+len(tr)]=c.astype('S1'); offset+=len(tr)
        assert offset==frames
        np.savez_compressed(out/f'{run}.npz',codes=codes,times_ps=t,residues=np.arange(1,152))
        reports.append(dict(run=run,frames=frames,duration_us=float((t[-1]-t[0])/1e6),
            pdb_sha256=digest(top),xtc_sha256=digest(traj),codes_sha256=digest(out/f'{run}.npz')))
        print(f'{run}: DSSP complete, {frames} frames',flush=True)
    dump(out/'dssp_provenance.json',dict(mdtraj=md.__version__,numpy=np.__version__,runs=reports,
        definitions=DEFINITIONS,imaging='Native archive; adjacent CA distances checked; no alignment needed',
        atom_selection='1279 protein heavy atoms selected by element from unchanged 2549-atom native archives; avoids duplicated hydrogen-only residue records',
        timebase='Exact native per-run elapsed times; merged XTC time NEVER used',
        pooling='All four runs, including July, as a descriptive duration-weighted mixture'))

def summarize(out, dssp_dir):
    fresh(out); rows=[]; duration=[]; profiles=[]
    for run in RUNS:
        data=np.load(dssp_dir/f'{run}.npz',allow_pickle=False)
        assert np.array_equal(data['residues'],np.arange(1,152))
        codes=data['codes'].astype('U2'); t=data['times_ps']; w=time_weights(t)
        assert codes.shape==(len(t),151) and set(np.unique(codes))<=set('HBEGITS ')
        duration.append(w.sum()); p=[]
        for name,letters in DEFINITIONS.items():
            values=np.isin(codes,list(letters)); mean=np.average(values,axis=0,weights=w); p.append(mean)
            # Window sensitivity is descriptive; no independent-frame SEM.
            for r in range(151):
                halves=[]
                for mask in [t<=t[0]+(t[-1]-t[0])/2,t>t[0]+(t[-1]-t[0])/2]:
                    halves.append(float(np.average(values[mask,r],weights=w[mask])))
                rows.append(dict(run=run,residue=r+1,definition=name,occupancy=float(mean[r]),
                    first_half=halves[0],second_half=halves[1],duration_us=w.sum()/1e6))
        profiles.append(p)
    a=np.array(profiles); weights=np.array(duration)/sum(duration)
    combined=np.einsum('r,rdn->dn',weights,a); pooled=[]
    for d,name in enumerate(DEFINITIONS):
        for r in range(151):
            pooled.append(dict(residue=r+1,definition=name,mean=combined[d,r],run_min=a[:,d,r].min(),
                run_max=a[:,d,r].max(),equal_run_mean=a[:,d,r].mean()))
    write_table(out/'dssp_per_run.csv',rows); write_table(out/'dssp_combined.csv',pooled)
    dump(out/'summary_provenance.json',dict(weights=dict(zip(RUNS,weights)),
        estimator='Per-run trapezoid integration in physical time, combined by elapsed duration',
        band='Minimum to maximum of four run means; descriptive spread, NOT SEM or confidence interval',
        uncertainty='Sampling uncertainty not estimated here. No equilibrium-convergence claim.'))

def plot(out,nmr_dir,summary_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fresh(out)
    nmrrows=table(nmr_dir/'delta2d_profiles.csv'); mdrows=table(summary_dir/'dssp_combined.csv')
    wt=sequence(); paired=[]; metrics=[]
    styles={'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
            'pdf.fonttype':42,'svg.fonttype':'none','axes.titlesize':10}
    for broad in [False,True]:
        with plt.rc_context(styles):
            fig,axes=plt.subplots(4,1,figsize=(7.2,9.4),layout='constrained')
            for k,(domain,state) in enumerate([('E6N','helix'),('E6N','beta'),('E6C','helix'),('E6C','beta')]):
                ax=axes[k]; lo,hi=(1,80) if domain=='E6N' else (80,151)
                x=np.arange(lo,hi+1)
                definition=('helix_HGI' if broad else 'alpha_H') if state=='helix' else ('beta_EB' if broad else 'beta_E')
                nd={int(r['md_residue']):r for r in nmrrows if r['variant']==domain and r['md_residue']}
                dd={int(r['residue']):r for r in mdrows if r['definition']==definition}
                y=np.array([float(nd[r][state]) if nd[r]['primary_eligible']=='1' else np.nan for r in x])
                inferred=np.array([float(nd[r][state]) if nd[r]['status']=='neighbor_inferred' and nd[r]['sequence_match']=='1' and int(nd[r]['input_nuclei'])>0 else np.nan for r in x])
                m=np.array([float(dd[r]['mean']) for r in x]); low=np.array([float(dd[r]['run_min']) for r in x]); high=np.array([float(dd[r]['run_max']) for r in x])
                ax.fill_between(x,low,high,color='#D55E00',alpha=.17,linewidth=0)
                ax.plot(x,m,color='#D55E00',lw=1.4,ls='-',zorder=3)
                ax.plot(x,y,color='black',lw=1.2,ls='-',marker='o',ms=1.7,zorder=4)
                ax.plot(x,inferred,ls='none',marker='o',mfc='white',mec='.4',ms=3,zorder=5)
                if domain=='E6N':
                    ax.axvspan(61.5,79.5,color='#0072B2',alpha=.06,zorder=0)
                for r in x:
                    ax.text(r,1.035,wt[r-1],ha='center',va='bottom',fontsize=6.2,
                            transform=ax.get_xaxis_transform(),color='#0072B2' if 62<=r<=79 else '.2')
                label='Helical propensity' if state=='helix' else 'β-structure propensity'
                ax.set(xlim=(lo-.6,hi+.6),ylim=(0,1),ylabel='Fraction helix' if state=='helix' else 'Fraction β-structure',xlabel='Residue')
                ax.set_title(f'{chr(65+k)}   {domain}: {label}',loc='left',pad=23,fontweight='bold')
                ax.set_yticks(np.linspace(0,1,6)); ax.set_xticks(np.arange(((lo+9)//10)*10,hi+1,10)); ax.grid(axis='y',color='.88',lw=.6)
                mask=np.isfinite(y)
                if not broad:
                    for i,r in enumerate(x):
                        paired.append(dict(domain=domain,residue=int(r),state=state,dssp_definition=definition,
                            nmr=y[i] if mask[i] else '',nmr_status=nd[r]['status'],sequence_match=nd[r]['sequence_match'],
                            md_mean=m[i],md_run_min=low[i],md_run_max=high[i]))
                    metrics.append(dict(domain=domain,state=state,n=int(mask.sum()),
                        rmsd=float(np.sqrt(np.mean((m[mask]-y[mask])**2))) if mask.any() else '',
                        mean_MD_minus_NMR=float(np.mean(m[mask]-y[mask])) if mask.any() else ''))
            handles=[Line2D([],[],color='k',marker='o',ms=2,label='NMR (Delta2D)'),
                Line2D([],[],color='#D55E00',label='MD (DSSP; all four runs)'),
                Patch(facecolor='#D55E00',alpha=.17,label='Range across runs'),
                Line2D([],[],color='.4',mfc='white',marker='o',ls='none',label='Delta2D neighbor-inferred')]
            fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
            name='comparison_grouped_DSSP' if broad else 'comparison_alpha_beta'
            for ext in ['pdf','svg','png']: fig.savefig(out/f'{name}.{ext}',dpi=600,facecolor='white')
            plt.close(fig)
    write_table(out/'paired_source_data.csv',paired); write_table(out/'descriptive_agreement.csv',metrics)
    (out/'caption.txt').write_text('Residue-resolved secondary-structure propensities of E6N and E6C. Black lines and points show Delta2D estimates from experimental chemical shifts of isolated engineered domains; orange lines show DSSP occupancies from four full-length apo-16E6 MD runs, including July, integrated in physical time and combined by duration. The orange band spans the minimum and maximum run means (descriptive spread, not a confidence interval). Primary panels use DSSP H for alpha helix and E for extended strand; the grouped sensitivity figure uses H/G/I and E/B. Open gray markers identify Delta2D neighbor-inferred predictions; these are excluded from agreement statistics. Missing predictions and direct sequence mismatches are gaps in the black line. Letters show the MD sequence; blue shading/letters indicate E6HL residues 62–79. Delta2D uses its own built-in neighbor averaging; no additional smoothing was applied. Estimates reflect isolated-domain constructs and their measurement conditions, and do not directly establish full-length linker disorder or kinetics. July has audited Zn-site parameter differences; the combined MD curve is a descriptive mixture, not a verified common-Hamiltonian equilibrium estimate.\n')
    dump(out/'figure_provenance.json',dict(matplotlib=matplotlib.__version__,numpy=np.__version__,
        dimensions_inches=[7.2,9.4],dpi=600,publisher_compliance='Not assessed; general manuscript export',
        nmr_sha256=digest(nmr_dir/'delta2d_profiles.csv'),dssp_sha256=digest(summary_dir/'dssp_combined.csv')))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['prepare','nmr','dssp','summarize','plot']:
        s=sub.add_parser(cmd); s.add_argument('--out',type=Path,required=True)
        if cmd=='nmr':
            s.add_argument('--prepared',type=Path,required=True); s.add_argument('--binary',type=Path,required=True); s.add_argument('--database',type=Path,required=True)
        if cmd=='dssp': s.add_argument('--native-root',type=Path,required=True); s.add_argument('--chunk',type=int,default=500)
        if cmd=='summarize': s.add_argument('--dssp-dir',type=Path,required=True)
        if cmd=='plot': s.add_argument('--nmr-dir',type=Path,required=True); s.add_argument('--summary-dir',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.out)
    elif a.command=='nmr': nmr(a.out,a.prepared,a.binary,a.database)
    elif a.command=='dssp': dssp(a.out,a.native_root,a.chunk)
    elif a.command=='summarize': summarize(a.out,a.dssp_dir)
    elif a.command=='plot': plot(a.out,a.nmr_dir,a.summary_dir)
if __name__=='__main__': main()
