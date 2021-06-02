# /usr/bin/env python3

import os
import sys
import argparse
from glob import glob

import pandas as pd
import numpy as np

from utilities import overlap

# initialization
np.set_printoptions(linewidth=250)  # , threshold=5000, suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = 200
pd.options.display.width = 250
pd.options.display.max_columns = 25
pd.options.display.min_rows = 20

# creating argument parser
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--viewpoints', default='./viewpoints/vp_info.tsv', type=str, help='Path to the VP information file')
arg_parser.add_argument('--expr_indices', default='-1', type=str, help='Limits processing to specific experiment indices (sep=",")')
arg_parser.add_argument('--input_dir', default='./outputs/proximity-enrichments/')
arg_parser.add_argument('--output_dir', default='./outputs/merged-enrichments/')
arg_parser.add_argument('--search_pattern', default='{ExprID}_*_bw{BinWidth:0.0f}kb_*.topbins.tsv.gz', type=str)
arg_parser.add_argument('--bin_widths', default='5e3,75e3', type=str)
# arg_parser.add_argument('--gs_widths', default='0.75', type=str)
# arg_parser.add_argument('--n_steps', default='31', type=str)
arg_parser.add_argument('--min_enrichment', default=5.0, type=float)
arg_parser.add_argument('--neighborhood_width', default=1e6, type=float)
inp_args = arg_parser.parse_args()
inp_args.bin_widths = [int(float(bw)) for bw in inp_args.bin_widths.split(',')]
# inp_args.gs_widths = [float(gs) for gs in inp_args.gs_widths.split(',')]
# inp_args.n_steps = [int(ns) for ns in inp_args.n_steps.split(',')]

# load vp info data
print('Loading experiment infos from: {:s}'.format(inp_args.viewpoints))
vp_infos = pd.read_csv(inp_args.viewpoints, sep='\t')
assert len(np.unique(vp_infos['expr_id'])) == vp_infos.shape[0]
vp_infos['row_index'] = range(vp_infos.shape[0])
if inp_args.expr_indices == '-1':
    inp_args.expr_indices = range(len(vp_infos))
else:
    print('Pre-selection of experiments: {:s}'.format(inp_args.expr_indices))
    inp_args.expr_indices = [int(ei) for ei in inp_args.expr_indices.split(',')]
vp_infos = vp_infos.loc[inp_args.expr_indices].reset_index(drop=True)
print('{:,d} experiments will be considered.'.format(len(vp_infos)))

# merging enrichments
for ei, (expr_idx, vp_info) in enumerate(vp_infos.iterrows()):

    # Get VP information
    print('[{:4d}/{:4d}] --> Merging enrichments found in {:s}'.format(ei + 1, len(vp_infos), vp_info['expr_id']))

    # loop over parameters
    enrichments = []
    for bw_idx, bin_width in enumerate(inp_args.bin_widths):
        print('\tbin_width={:3.0f}kb: '.format(bin_width / 1e3), end='')

        # find the proper bin enrichment file
        file_pattern = os.path.join(
            inp_args.input_dir,
            inp_args.search_pattern.format(ExprID=vp_info['expr_id'], BinWidth=bin_width / 1e3)
        )
        enrichment_files = glob(os.path.expanduser(file_pattern))
        assert len(enrichment_files) == 1, 'Non-unique enrichment file is found when searching for: {:s}\n'.format(file_pattern) + \
                                           'Correct the "search pattern"'
        # if not os.path.isfile(enrichment_fpath):
        #     n_miss += 1
        #     warnings.warn('Could not find {:s}'.format(enrichment_fpath))
        #     continue

        # load the enrichment file
        bin_pd = pd.read_csv(enrichment_files[0], delimiter='\t')
        if bin_pd.shape[0] == 0:
            print('Enrichment file is empty: {:s}'.format(enrichment_files[0]))
            continue

        # select enriched bins
        is_sig = bin_pd['#cpt_zscr'] >= inp_args.min_enrichment
        sig_pd = bin_pd.loc[is_sig].reset_index(drop=True)
        print('#bins: {:4d} loaded, {:4d} enriched '.format(bin_pd.shape[0], sig_pd.shape[0]), end='')
        del bin_pd  # bin_pd.loc[is_sig] bin_pd.loc[~is_sig]
        if len(sig_pd) == 0:
            print()
            continue

        # marking neighbor bins
        enrich_crd = sig_pd[['chr', 'pos', 'pos']].values
        enrich_crd[:, 2] += bin_width
        nei_idxs = np.arange(len(sig_pd))
        for ci in range(len(sig_pd)):
            has_ol = overlap(enrich_crd[ci], enrich_crd, offset=inp_args.neighborhood_width)
            if np.sum(has_ol) > 1:
                is_sel = np.isin(nei_idxs, nei_idxs[has_ol])  # sig_pd.loc[has_ol]
                nei_idxs[is_sel] = np.min(nei_idxs[is_sel])  # sig_pd.loc[is_sel]
        sig_pd['nei_idx'] = nei_idxs
        del enrich_crd

        # merging neighbor bins
        sig_pd = sig_pd.sort_values(by='#cpt_zscr', ascending=False).reset_index(drop=True)
        nei_grp = sig_pd.groupby(by='nei_idx', sort=False)
        for rank_idx, (nei_idx, nei_pd) in enumerate(nei_grp):
            itm_crd = [nei_pd['chr'].iat[0], np.min(nei_pd['pos']), np.max(nei_pd['pos']) + bin_width]
            itm_pd = vp_info[['expr_id', 'vp_chr', 'vp_pos', 'vp_gene', 'sample_id']].copy()
            itm_pd['bin_width'] = bin_width
            itm_pd['enrich_rank'] = rank_idx + 1
            itm_pd['#nei_merged'] = nei_pd.shape[0]
            itm_pd['enrich_chr'] = itm_crd[0]
            itm_pd['enrich_pos'] = nei_pd.nlargest(1, '#cpt_zscr')['pos'].values[0]
            itm_pd['enrich_beg'] = itm_crd[1]
            itm_pd['enrich_end'] = itm_crd[2]

            for scr_name in ['#cpt_zscr']:  # 'cvrg_zscr', 'ccpt_zscr', 'shfl_zscr', '#cpt_zscr', 'cmb_zscr'
                # itm_pd[scr_name + '_max'] = nei_pd[scr_name].max()
                # itm_pd[scr_name + '_med'] = nei_pd[scr_name].median()
                itm_pd[scr_name + '_90p'] = nei_pd[scr_name].quantile(0.9)
            # for scr_name in ['#cpt_pval', '#cpt_qval', 'cvrg_pval', 'cmb_pval', 'cmb_qval']:
            #     itm_pd[scr_name + '_min'] = nei_pd[scr_name].min()
            #     itm_pd[scr_name + '_med'] = nei_pd[scr_name].median()
            #     itm_pd[scr_name + '_10p'] = nei_pd[scr_name].quantile(0.1)
            enrichments.append(itm_pd.copy())
        print('-> merged to {:2d} enrichments'.format(nei_grp.ngroups))
    enrichments = pd.DataFrame(enrichments, index=range(len(enrichments)))

    # store the merged enrichments
    out_fpath = os.path.join(
        inp_args.output_dir,
        '{:s}_merged-enrichments.tsv.gz'.format(vp_info['expr_id'])
    )
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    enrichments.to_csv(out_fpath, sep='\t', na_rep='nan', index=False)
    print('\tMerged enrichments are stored in: {:s}'.format(out_fpath))

