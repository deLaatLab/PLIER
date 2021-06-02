# /usr/bin/env python3

import os
import argparse

import pandas as pd
import numpy as np

from utilities import overlap

# initialization
np.set_printoptions(linewidth=250)  # , threshold=5000, suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = 500

# creating argument parser
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--viewpoints', default='./viewpoints/vp_info.tsv', type=str, help='Path to the VP information file')
arg_parser.add_argument('--expr_indices', default='-1', type=str, help='Limits processing to specific experiment indices (sep=",")')
arg_parser.add_argument('--input_dir', default='./outputs/merged-enrichments/')
arg_parser.add_argument('--output_dir', default='./outputs/significant_calls/')
arg_parser.add_argument('--significance_threshold', default=8.0, type=float)
arg_parser.add_argument('--significance_threshold_cis', default=16.0, type=float)
arg_parser.add_argument('--bin_widths', default='5e3,75e3')
arg_parser.add_argument('--min_n_binw', default=2, type=int)
arg_parser.add_argument('--min_n_vp', default=3, type=int)
# arg_parser.add_argument('--gs_widths', default='0.75')
# arg_parser.add_argument('--n_steps', default='31')
arg_parser.add_argument('--enrichment_score', default='#cpt_zscr_90p', type=str)
arg_parser.add_argument('--neighborhood_width', default=5e6, type=float)
inp_args = arg_parser.parse_args()
inp_args.bin_widths = sorted([int(float(b)) for b in inp_args.bin_widths.split(',')])
# inp_args.gaus_widths = [float(g) for g in inp_args.gaus_widths.split(',')]

# load vp info data
print('Loading experiment infos from: {:s}'.format(inp_args.viewpoints))
vp_infos = pd.read_csv(inp_args.viewpoints, sep='\t')
assert len(np.unique(vp_infos['expr_id'])) == vp_infos.shape[0], 'Non-unique experiment ID is found.'
assert len(np.unique(vp_infos['sample_id'])) == 1, 'Non-unique "sample_id" is found in the viewpoint file: {:s}'.format(inp_args.viewpoints)
vp_infos['row_index'] = range(vp_infos.shape[0])
if inp_args.expr_indices == '-1':
    inp_args.expr_indices = range(len(vp_infos))
else:
    print('Pre-selection of experiments: {:s}'.format(inp_args.expr_indices))
    inp_args.expr_indices = [int(ei) for ei in inp_args.expr_indices.split(',')]
vp_infos = vp_infos.loc[inp_args.expr_indices].reset_index(drop=True)
print('{:,d} experiments will be considered.'.format(len(vp_infos)))

# selecting significant enrichments
calls = []
for ei, (expr_idx, vp_info) in enumerate(vp_infos.iterrows()):

    # load enrichment file
    enrichment_fpath = os.path.join(
        inp_args.input_dir,
        '{:s}_merged-enrichments.tsv.gz'.format(vp_info['expr_id'])
    )
    enrichments = pd.read_csv(enrichment_fpath, sep='\t')
    print('\t[{:2d}/{:d}] Loading enrichments in: {:s}'.format(ei + 1, len(vp_infos), enrichment_fpath))

    # filtering enrichments
    is_sel = enrichments['bin_width'].isin(inp_args.bin_widths)
    enrichments = enrichments.loc[is_sel].reset_index(drop=True)
    enrichments = enrichments.sort_values(by=inp_args.enrichment_score, ascending=False).reset_index(drop=True)

    # finding overlapping calls across bin_widths/Gaussian_widths/n_steps
    enrich_crd = enrichments[['enrich_chr', 'enrich_beg', 'enrich_end']].values
    ovl_idxs = np.arange(len(enrich_crd))
    for ci in range(len(enrich_crd)):
        has_ol = overlap(enrich_crd[ci], enrich_crd, offset=inp_args.neighborhood_width)
        if np.sum(has_ol) > 1:
            is_in = np.isin(ovl_idxs, ovl_idxs[has_ol])  # clc_pd.loc[has_ol]
            ovl_idxs[is_in] = np.min(ovl_idxs[is_in])  # clc_pd.loc[is_in]
    enrichments['ovl_idx'] = np.unique(ovl_idxs, return_inverse=True)[1]
    del enrich_crd, ovl_idxs

    # select significant calls: The overlap is determined, we dont need insignificant calls anymore
    is_cis = enrichments['vp_chr'] == enrichments['enrich_chr']
    is_sig = (is_cis & (enrichments[inp_args.enrichment_score] >= inp_args.significance_threshold_cis)) | \
             (~is_cis & (enrichments[inp_args.enrichment_score] >= inp_args.significance_threshold))
    enrichments = enrichments.loc[is_sig].reset_index(drop=True)
    del is_cis, is_sig

    # combine across scales
    for params_idx, params_pd in enrichments.groupby(by=['ovl_idx'], sort=False):

        # select calls multi-width significance
        bw_grp = params_pd.groupby('bin_width')
        if bw_grp.ngroups < inp_args.min_n_binw:
            continue
        repr_call = bw_grp.get_group(np.max(params_pd['bin_width'])).nlargest(1, inp_args.enrichment_score).iloc[0]

        call = vp_info[['expr_id', 'vp_chr', 'vp_be', 'vp_en', 'vp_gene', 'sample_id']].copy()
        call['call_chr'] = repr_call['enrich_chr']
        call['call_pos'] = repr_call['enrich_pos']
        call['call_beg'] = int(params_pd['enrich_beg'].min())
        call['call_end'] = int(params_pd['enrich_end'].max())
        call['call_score'] = repr_call[inp_args.enrichment_score].round(2)
        call['bin_widths'] = ';'.join(['{:0.0f}'.format(x / 1e3) for x in np.unique(params_pd['bin_width'])]) + 'k'
        call['#bin_widths'] = bw_grp.ngroups
        calls.append(call.copy())
        del call

# select enriched calls over multiple bin_widths
significant_calls = pd.DataFrame(calls, index=range(len(calls)))
is_sig = significant_calls['#bin_widths'] >= inp_args.min_n_binw
significant_calls = significant_calls.loc[is_sig].reset_index(drop=True)

# =======================
# checking for amplification events
# mark overlapping calls
if len(significant_calls) > 0:
    significant_calls['sv_type'] = ''
    call_crd = significant_calls[['call_chr', 'call_beg', 'call_end']].values
    rgn_idxs = np.arange(call_crd.shape[0])
    for ci in range(call_crd.shape[0]):
        has_ol = overlap(call_crd[ci], call_crd, offset=1e6)
        if np.sum(has_ol) > 1:
            is_in = np.isin(rgn_idxs, rgn_idxs[has_ol])  # vp_calls.loc[has_ol]
            rgn_idxs[is_in] = np.min(rgn_idxs[is_in])  # vp_calls.loc[is_in]
    significant_calls['call_idx'] = np.unique(rgn_idxs, return_inverse=True)[1]
    del call_crd, rgn_idxs

    # mark nearby VPs
    vp_crd = significant_calls[['vp_chr', 'vp_be', 'vp_en']].values
    vp_idx = np.arange(vp_crd.shape[0])
    for vi in range(vp_crd.shape[0]):
        has_ol = overlap(vp_crd[vi], vp_crd, offset=2e6)
        if np.sum(has_ol) > 1:
            is_in = np.isin(vp_idx, vp_idx[has_ol])  # vp_calls.loc[has_ol]
            vp_idx[is_in] = np.min(vp_idx[is_in])  # vp_calls.loc[is_in]
    significant_calls['vp_idx'] = np.unique(vp_idx, return_inverse=True)[1]
    del vp_crd, vp_idx
    # significant_calls = significant_calls.sort_values(by=['call_idx', 'vp_gene']).reset_index(drop=True)

    # find amplification events
    for rgn_idx, rgn_pd in significant_calls.groupby('call_idx'):
        n_vp = len(np.unique(rgn_pd['vp_idx']))
        if n_vp >= inp_args.min_n_vp:
            significant_calls.loc[rgn_pd.index, 'sv_type'] = 'amplification'
    significant_calls.drop(columns=['call_idx', 'vp_idx'], inplace=True)

# storing the output
out_fpath = os.path.join(
    inp_args.output_dir,
    '{:s}_significant-calls.tsv.gz'.format(vp_infos['sample_id'].iat[0])
)
os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
significant_calls.to_csv(out_fpath, sep='\t', na_rep='nan', index=False)
print('{:d} calls detected and stored in: {:s}'.format(len(significant_calls), out_fpath))


