# -*- coding: utf-8 -*-
"""Build the final figure from BTL-model and inter-observer results.

Run: python build_final_figure.py
Requires numpy and matplotlib. No model fitting or statistics are recomputed.
Inputs remain in their existing project locations, read only.
Outputs: four individual PNG panels, one final PNG and a vector PDF.

a-c: posterior means and 95% HDIs, matching Bayesian_summary.py.
d: exact agreement for ALL repeated pair types, with 95% pair-bootstrap CIs.
The interval types differ; retain this distinction in the manuscript caption.
"""
from pathlib import Path
import csv
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BAYES = PROJECT / 'results' / 'BTL_model' / 'Bayesian_summary_values.csv'
AGREEMENT = PROJECT / 'results' / 'inter_observer_all_pairs' / '01_reliability_summary.csv'
OUTPUT = PROJECT / 'figures' / 'final'
DIMENSIONS = ('Novelty', 'Significance', 'Testability')
COLORS = {'Novelty': '#0072B2', 'Significance': '#E69F00', 'Testability': '#CC79A7'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Keep temporary font-cache files out of the figure folder.
    with tempfile.TemporaryDirectory(prefix='figure_font_cache_') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        from matplotlib.font_manager import findfont, FontProperties
        import numpy as np

        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BAYES, AGREEMENT)}
        with BAYES.open(encoding='utf-8-sig', newline='') as f:
            bayes = list(csv.DictReader(f))
        with AGREEMENT.open(encoding='utf-8-sig', newline='') as f:
            agreement = {r['Dimension']: r for r in csv.DictReader(f)}
        assert len(bayes) == 9 and set(agreement) == set(DIMENSIONS)

        # Exact font configuration from the existing Bayesian_summary.py.
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 10,
            'xtick.labelsize': 9, 'ytick.labelsize': 10,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False,
        })
        print('Font:', findfont(FontProperties(family='Arial'), fallback_to_default=False))

        def frame(ax, title, letter):
            ax.set_title(title, loc='left', fontsize=10, fontweight='semibold', pad=10)
            ax.text(-.24, 1.09, letter, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='bottom', ha='left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(.8)
            ax.spines['bottom'].set_linewidth(.8)
            ax.tick_params(axis='both', length=3, width=.8, color='0.25', pad=4)

        def posterior(ax, parameter, xlabel, title, letter, limits, ticks):
            frame(ax, title, letter)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='0.92', linewidth=.6)
            if parameter == 'mu_LLM':
                ax.axvspan(-.1, .1, color='0.65', alpha=.12, linewidth=0, zorder=0)
                for bound in (-.1, .1):
                    ax.axvline(bound, color='0.55', linestyle=(0,(1.5,2.5)), linewidth=.8)
                ax.axvline(0, color='0.25', linestyle=(0,(4,3)), linewidth=1)
            for y, dim in zip((2,1,0), DIMENSIONS):
                rows=[r for r in bayes if r['Parameter']==parameter and r['Dimension']==dim]
                assert len(rows)==1
                row=rows[0]
                mean, low, high=[float(row[k]) for k in ('Posterior_mean','HDI_95_lower','HDI_95_upper')]
                assert low <= mean <= high
                ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt='o',
                            markersize=5.8,markerfacecolor=COLORS[dim],
                            markeredgecolor='white',markeredgewidth=.8,
                            ecolor=COLORS[dim],elinewidth=1.7,capsize=3,
                            capthick=1.1,zorder=3)
            ax.set_xlim(*limits)
            ax.set_xticks(ticks)
            ax.set_ylim(-.55,2.55)
            ax.set_yticks([2,1,0], labels=DIMENSIONS)
            ax.set_xlabel(xlabel, labelpad=6)
            if parameter == 'tie_probability':
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))

        def panel_a(ax):
            posterior(ax,'mu_LLM',r'$\mu_{\mathrm{LLM}}$','Overall source effect','a',(-.25,.25),[-.2,-.1,0,.1,.2])

        def panel_b(ax):
            posterior(ax,'sigma_LLM',r'$\sigma_{\mathrm{LLM}}$','Between-expert heterogeneity','b',(0,.45),[0,.1,.2,.3,.4])

        def panel_c(ax):
            posterior(ax,'tie_probability',r'$P(\mathrm{Tie}\mid\Delta=0)$','Baseline tie probability','c',(0,.3),[0,.1,.2,.3])

        def panel_d(ax):
            frame(ax,'Inter-observer agreement','d')
            values=np.array([float(agreement[d]['Exact_agreement_percent']) for d in DIMENSIONS])
            low=np.array([float(agreement[d]['Agreement_CI_lower_percent']) for d in DIMENSIONS])
            high=np.array([float(agreement[d]['Agreement_CI_upper_percent']) for d in DIMENSIONS])
            assert np.all(low<=values) and np.all(values<=high)
            ax.set_axisbelow(True)
            ax.grid(axis='y',color='0.92',linewidth=.6)
            for i,(y,dim) in enumerate(zip((2,1,0),DIMENSIONS)):
                ax.errorbar(values[i],y,xerr=[[values[i]-low[i]],[high[i]-values[i]]],
                            fmt='o',markersize=5.8,markerfacecolor=COLORS[dim],
                            markeredgecolor='white',markeredgewidth=.8,
                            ecolor=COLORS[dim],elinewidth=1.7,capsize=3,
                            capthick=1.1,zorder=3)
            ax.set_xlim(0,60)
            ax.set_xticks([0,20,40,60])
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
            ax.set_ylim(-.55,2.55)
            ax.set_yticks([2,1,0],labels=DIMENSIONS)
            ax.set_xlabel('Exact agreement',labelpad=6)

        panels=[panel_a,panel_b,panel_c,panel_d]
        names=['panel_a_overall_source_effect','panel_b_between_expert_heterogeneity',
               'panel_c_baseline_tie_probability','panel_d_inter_observer_agreement']
        for draw,name in zip(panels,names):
            fig,ax=plt.subplots(figsize=(4.8,3.1))
            fig.subplots_adjust(left=.24,right=.97,bottom=.22,top=.78)
            draw(ax)
            fig.savefig(OUTPUT/(name+'.png'),dpi=600,facecolor='white')
            plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(9.6,6.1))
        fig.subplots_adjust(left=.12,right=.98,bottom=.11,top=.89,wspace=.48,hspace=.70)
        for ax,draw in zip(axes.flat,panels):
            draw(ax)
        fig.savefig(OUTPUT/'final_human_llm_expert_evaluation.png',dpi=600,facecolor='white')
        fig.savefig(OUTPUT/'final_human_llm_expert_evaluation.pdf',facecolor='white')
        plt.close(fig)
        for path,digest in hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        print('Input values unchanged. Panels a-c: 95% HDIs; panel d: 95% bootstrap CIs.')
        for name in names:
            print(OUTPUT/(name+'.png'))
        print(OUTPUT/'final_human_llm_expert_evaluation.png')
        print(OUTPUT/'final_human_llm_expert_evaluation.pdf')


if __name__ == '__main__':
    main()
