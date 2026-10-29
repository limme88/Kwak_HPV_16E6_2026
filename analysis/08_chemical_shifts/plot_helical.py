#!/usr/bin/env python3
"""Build the intermediate two-domain helical propensity panel.
Inputs: paired helical source CSV, FASTA, and optional blocking estimates.
Outputs: plots, plotted source data, caption and metadata.
Intermediate for Figure 2a; make_figure2.py produces the final combined layout.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
# Configuration: use the mutation-inclusive final paired input table.
REPO = Path(__file__).resolve().parents[2]
B = REPO / 'data/chemical_shifts'
FASTA = Path(__file__).resolve().parent / 'inputs/apo16e6_151aa.fasta'

def draw(out,blocking=None):
    out.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(open(B/'paired_source_data.csv')))
    seq=''.join(x.strip() for x in FASTA.read_text().splitlines() if not x.startswith('>'))
    errors={} if blocking is None else {int(r['residue']):r for r in csv.DictReader(open(blocking))}
    if blocking is not None:
        assert set(errors)==set(range(1,152))
        assert {r['definition'] for r in errors.values()}=={'alpha_H'}
    source=[]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.grid':False}):
        fig,axs=plt.subplots(2,1,figsize=(7.2,5.6),layout='constrained')
        for k,domain in enumerate(['E6N','E6C']):
            ax=axs[k];rr=[r for r in rows if r['domain']==domain and r['state']=='helix']
            x=np.array([int(r['residue']) for r in rr]);m=np.array([float(r['md_mean']) for r in rr])
            y=np.array([float(r['nmr']) if r['nmr'] else np.nan for r in rr])
            se=np.full(len(x),np.nan)
            if errors:
                for i,r in enumerate(x):
                    assert abs(float(errors[r]['mean'])-m[i])<1e-10
                    if errors[r]['status']=='estimated_for_stated_window':se[i]=float(errors[r]['sem'])
                ax.fill_between(x,np.maximum(0,m-se),np.minimum(1,m+se),color='#D55E00',alpha=.22,lw=0)
            ax.plot(x,y,color='black',lw=1.4,marker=None,zorder=4)
            ax.plot(x,m,color='#D55E00',lw=1.5,marker=None,zorder=3)
            # Highlight sequence backgrounds only, with readable black amino-acid letters.
            start,end,color=(62,79,'#FFE45C') if domain=='E6N' else (140,151,'#C5A3E6')
            ax.add_patch(Rectangle((start-.5,1.03),end-start+1,.075,
                transform=ax.get_xaxis_transform(),facecolor=color,edgecolor='none',clip_on=False))
            for r in x:ax.text(r,1.055,seq[r-1],transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=6.5,color='black')
            ax.set_title(f'{chr(65+k)}   {domain} helical propensity',loc='left',pad=28,fontweight='bold')
            ax.set(xlim=(x[0]-.6,x[-1]+.6),ylim=(0,1),xlabel='Residue',ylabel='Fraction helix')
            ax.set_yticks(np.linspace(0,1,6));ax.set_xticks(np.arange(((x[0]+9)//10)*10,x[-1]+1,10));ax.grid(False)
            for i,r in enumerate(x):source.append(dict(domain=domain,residue=int(r),nmr=y[i] if np.isfinite(y[i]) else '',md_mean=m[i],sem=se[i] if np.isfinite(se[i]) else '',status=errors[r]['status'] if errors else 'blocking_pending'))
        handles=[Line2D([],[],color='black',lw=1.4,label='NMR prediction (Delta2D)'),Line2D([],[],color='#D55E00',lw=1.5,label='MD (DSSP)')]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
        if blocking is None:fig.suptitle('LAYOUT PREVIEW — blocking error estimates pending',fontsize=9,color='#8F4400')
        for ext in ['pdf','svg','png']:fig.savefig(out/f'helical_propensity.{ext}',dpi=600,facecolor='white')
        plt.close(fig)
    with open(out/'figure_source_data.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    caption=('Helical propensities inferred from experimental chemical shifts of isolated E6N and E6C using Delta2D (black) and alpha-helical DSSP H occupancies from four native MD runs, including July (orange). MD means retain full native physical-time weighting. Sequence backgrounds highlight E6HL residues 62–79 in yellow and the C-terminal tail 140–151 in purple. Experimental prediction gaps and mutation exclusions are retained. ')
    if blocking:
        count=sum(r['status']=='estimated_for_stated_window' for r in errors.values())
        caption+=('Orange shading is mean ± one pointwise blocking SEM, propagated from independent runs with fixed duration weights. Bands are shown only where every run has an accepted error estimate; lack of shading at unresolved positions does not imply zero uncertainty. Blocking SEMs describe within-run sampling under stated assumptions, excluding between-run differences, force-field and predictor uncertainty. Displayed limits are clipped to the physical range 0–1; untruncated errors are in the source CSV. ')
    else:caption+='This is a layout preview. Statistical error bands have not yet been calculated; no substitute range band is displayed. '
    (out/'caption.txt').write_text(caption+'\n')
    (out/'figure_manifest.json').write_text(json.dumps(dict(blocking_source=str(blocking) if blocking else None,
        input_hash=hashlib.sha256((B/'paired_source_data.csv').read_bytes()).hexdigest(),dssp_definition='H',
        figure_size_inches=[7.2,5.6],status='blocking_pending' if blocking is None else 'review_blocking_coverage_before_publication',grid=False,tail=[140,151],E6HL=[62,79]),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--blocking',type=Path)
    a=p.parse_args();draw(a.out,a.blocking)
