#!/usr/bin/env python3
"""Reproduce the residue-resolved helicity and C-alpha comparison in Figure 2.
Inputs: processed experimental/predicted shift CSVs, helicity CSV and FASTA.
Outputs: standalone and paired PDF, SVG, PNG figures, source CSV and metadata.
Scientific calculations and drawing parameters are preserved from the final source.
"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Configuration: all inputs and outputs are relative to the repository root.
REPO = Path(__file__).resolve().parents[2]
COMP = REPO / 'data/chemical_shifts'
HELIX = COMP / 'helical_source_data.csv'
FASTA = Path(__file__).resolve().parent / 'inputs/apo16e6_151aa.fasta'
B = REPO / 'results/chemical_shifts/figure2'
B.mkdir(parents=True, exist_ok=True)
seq=''.join(s.strip() for s in FASTA.read_text().splitlines() if not s.startswith('>'))
exp=list(csv.DictReader((COMP/'experimental_all_rows.csv').open()))
pred=list(csv.DictReader((COMP/'predicted_profiles.csv').open()))
pr={(r['ensemble'],int(r['residue'])):r for r in pred if r['atom']=='CA'}
er={(r['domain'],int(float(r['md_residue']))):r for r in exp if r['sparta_atom']=='CA' and r['md_residue']}
assert len(er)==sum(r['sparta_atom']=='CA' and bool(r['md_residue']) for r in exp)
runs=['native_20250414','native_20250505','native_20250710','native_20250520']
data=[]
for domain,start,stop in [('E6N',1,80),('E6C',80,151)]:
    for i in range(start,stop+1):
        e=er.get((domain,i));p=pr.get(('duration_weighted',i))
        assert p is not None
        parts=[pr[(run,i)] for run in runs]
        run_weights=np.array([float(r['duration_us']) for r in parts]);run_weights/=run_weights.sum()
        mean=float(p['predicted_mean_ppm'])
        assert abs(sum(a*float(r['predicted_mean_ppm']) for a,r in zip(run_weights,parts))-mean)<1e-9
        value=float(e['shift_ppm']) if e else None
        data.append(dict(domain=domain,residue=i,experimental_Ca_ppm=value if e else '',
            MD_Ca_ppm=mean,experimental_minus_MD_ppm=value-mean if e else '',
            experimental_resname=e['experimental_resname'] if e else '',
            md_resname=e['md_resname'] if e else '',sequence_match=e['sequence_match'] if e else '',
            source_row_id=e['source_row_id'] if e else '',source_file=e['source_file'] if e else '',
            status='available' if e else 'missing_experimental_CA',
            ensemble='duration_weighted',nframes=p['nframes']))

# Compare against the prior sequence-matched comparison and the manuscript table.
old={(r['domain'],int(r['residue'])):r for r in csv.DictReader((COMP/'experimental_residuals.csv').open())
     if r['ensemble']=='duration_weighted' and r['atom']=='CA'}
matched_metrics={}
for d in ('E6N','E6C'):
    rr=[r for r in data if r['domain']==d and r['sequence_match']=='True' and r['status']=='available']
    for r in rr:
        assert abs(r['experimental_minus_MD_ppm']+float(old[(d,r['residue'])]['residual_ppm']))<1e-10
    rmse=math.sqrt(np.mean([r['experimental_minus_MD_ppm']**2 for r in rr]))
    metric=next(r for r in csv.DictReader((COMP/'comparison_metrics.csv').open())
                if r['ensemble']=='duration_weighted' and r['atom']=='CA' and r['domain']==d and r['mask']=='primary')
    assert abs(rmse-float(metric['RMSE_ppm']))<1e-10
    matched_metrics[d]=dict(n=len(rr),RMSE_ppm=rmse)
gaps=[(r['domain'],r['residue']) for r in data if r['status']!='available']
assert gaps==[('E6C',124),('E6C',146)]
with (B/'figure_source_data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

orange='#D55E00';benchmark=.92
helix=list(csv.DictReader(HELIX.open()))
def frame(ax,show_sequence=True):
    ax.set_xlim(.4,151.6);ax.set_xticks(range(10,151,10));ax.set_xlabel('Residue');ax.grid(False)
    ax.tick_params(axis='both',labelsize=12)
    if not show_sequence:
        return
    for start,end,color in [(62,79,'#FFE45C'),(140,151,'#C5A3E6')]:
        ax.add_patch(Rectangle((start-.5,1.025),end-start+1,.08,transform=ax.get_xaxis_transform(),
                     facecolor=color,edgecolor='none',clip_on=False))
    for i,aa in enumerate(seq,1):
        ax.text(i,1.06,aa,transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=6.5)

def residual(ax,show_sequence=True):
    for d in ('E6N','E6C'):
        rr=[r for r in data if r['domain']==d]
        x=np.array([r['residue'] for r in rr]);y=np.array([r['experimental_minus_MD_ppm'] if r['status']=='available' else np.nan for r in rr])
        ax.plot(x,y,color=orange,lw=1.4,zorder=3)
        ax.fill_between(x,0,y,color=orange,alpha=.18,lw=0,zorder=2)
    ax.axhline(0,color='.65',lw=.6,zorder=1)
    for v in (-benchmark,benchmark):ax.axhline(v,color='black',lw=1.1,zorder=4)
    lim=math.ceil(max(abs(r['experimental_minus_MD_ppm']) for r in data if r['status']=='available')*2)/2+.2
    ax.set_ylim(-lim,lim)
    ax.set_ylabel(r'$C\alpha_{\mathrm{Exp}}-C\alpha_{\mathrm{MD}}$ (ppm)')
    frame(ax,show_sequence=show_sequence)

def helicity(ax):
    md={}
    for d in ('E6N','E6C'):
        rr=[r for r in helix if r['domain']==d]
        x=[int(r['residue']) for r in rr]
        y=[float(r['nmr']) if r['nmr'] else np.nan for r in rr]
        ax.plot(x,y,color='black',lw=1.4,zorder=4)
        md.update({int(r['residue']):float(r['md_mean']) for r in rr})
    ax.plot(range(1,152),[md[i] for i in range(1,152)],color=orange,lw=1.5,zorder=3)
    ax.set(ylim=(0,1),ylabel='Fraction helix');ax.set_yticks(np.linspace(0,1,6));frame(ax)
    ax.legend(handles=[Line2D([],[],color='black',lw=1.4,label='NMR prediction (Delta2D)'),
                       Line2D([],[],color=orange,lw=1.5,label='MD (DSSP)')],
              loc='lower center',bbox_to_anchor=(.5,1.13),frameon=False,ncol=2,fontsize=9)

handles=[Line2D([],[],color=orange,lw=1.4,label=r'$C\alpha_{\mathrm{Exp}}-C\alpha_{\mathrm{MD}}$'),
         Line2D([],[],color='black',lw=1.1,label='SPARTA+ reference (±0.92 ppm)')]
with plt.rc_context({'font.family':'Helvetica','font.size':12,'axes.labelsize':14,
                     'mathtext.fontset':'custom','mathtext.rm':'Helvetica',
                     'mathtext.it':'Helvetica:italic','mathtext.bf':'Helvetica:bold',
                     'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False}):
    fig,ax=plt.subplots(figsize=(12,3.8),layout='constrained');residual(ax)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    for ext in ('pdf','svg','png'):fig.savefig(B/f'calpha_residual.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    helicity(axs[0]);residual(axs[1],show_sequence=False)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    for ext in ('pdf','svg','png'):fig.savefig(B/f'helicity_and_calpha.{ext}',dpi=600,facecolor='white')
    plt.close(fig)

caption=('Residue-resolved differences between experimental Cα chemical shifts of isolated E6N and E6C and SPARTA+ predictions averaged over the four apo-16E6 MD runs, including July (orange; experimental minus predicted shift). Predictions use the existing physical-time-weighted all-run means, without fitting a chemical-shift referencing offset. Black horizontal lines at ±0.92 ppm indicate the Cα RMS prediction error reported by the SPARTA+ software documentation for an independent validation set of eleven proteins. These lines are a predictor benchmark, not statistical error bars or a confidence interval for the MD ensemble. Orange filling indicates the signed difference from zero and does not represent uncertainty. Sequence backgrounds highlight E6HL 62–79 (yellow) and C-terminal tail 140–151 (purple). Experimental substitution-site values are included (E6N F47R/C80S; E6C C80S/C97S/C111S/C140S), while the displayed sequence is the MD sequence. At those sites the residual also compares different residue identities. Both construct-specific values at residue 80 are retained without averaging or drawing a vertical connection. No experimental Cα shifts are available at E6C R124 or R146; gaps are retained. L151 is included because both experimental and predicted Cα shifts are available. The all-run mean describes the sampled mixture, including July with its audited Zn-site parameter differences.\n')
(B/'caption.txt').write_text(caption)
manifest=dict(ensemble='duration_weighted',runs=runs,weights=run_weights.tolist(),
    sign='experimental - MD',benchmark_ppm=benchmark,benchmark_source='https://spin.niddk.nih.gov/bax-apps/software/SPARTA%2B/',
    benchmark_kind='RMS error for eleven proteins outside the training database',
    includes_mutation_sites=True,comparison_rows=len(data),available_rows=sum(r['status']=='available' for r in data),
    missing_CA=gaps,sequence_matched_QC=matched_metrics,all_run_means_recomputed=True,
    all_available_Ca_RMSE_ppm={d:math.sqrt(np.mean([r['experimental_minus_MD_ppm']**2 for r in data if r['domain']==d and r['status']=='available'])) for d in ('E6N','E6C')},
    no_fitted_offset=True,no_error_band=True,fill='signed residual area to zero',
    font_family='Helvetica',axis_title_pt=14,tick_label_pt=12,combined_sequence_strips=1,
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [COMP/'experimental_all_rows.csv',COMP/'predicted_profiles.csv',HELIX,FASTA]},
    validation='PASS: duration weighting, old residual sign, sequence-matched manuscript RMSEs, missingness and mapping')
(B/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['available_rows','missing_CA','sequence_matched_QC','all_available_Ca_RMSE_ppm','validation']},indent=2))
