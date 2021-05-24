#! /usr/bin/env python3

import os
import sys
import argparse
from time import time

import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from sklearn.isotonic import IsotonicRegression

from utilities import get_chr_info, get_re_info, get_vp_info, load_dataset, perform_sigtest, OnlineStats, overlap


def apply_isotonic(arr):
    n_krn = len(gus_krn)
    arr_smt = np.convolve(np.hstack([np.repeat(arr[0], n_krn * 2), arr]), gus_krn, mode='same')[n_krn * 2:]
    # plt.plot(range(len(arr_smt)), arr_smt, alpha=0.5, label='Smooth')
    ir = IsotonicRegression(y_min=0, increasing=False)
    # plt.plot(range(len(arr_smt)), bin_prd, alpha=0.5, label='Pred')
    return ir.fit_transform(range(len(arr_smt)), arr_smt)


# initialization
np.set_printoptions(linewidth=250)  # , threshold=5000, suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = 200
pd.options.display.width = 250
pd.options.display.max_columns = 25
n_bg_splt = 1000
background_methods = ['cpt-shfl', 'cpt-swap']


arg_parser = argparse.ArgumentParser(description='Computing significance of the proximity-frequencies')
arg_parser.add_argument('--expr_index', required=True, type=int, help='Index of the experiment that PLIER runs on')
arg_parser.add_argument('--viewpoints', default='./viewpoints/vp_info.tsv', type=str, help='Path to the VP information file')
arg_parser.add_argument('--dataset_dir', default='./datasets/')
arg_parser.add_argument('--output_id', help='An ID used for output (i.e., the PDF plot and bin enrichment scores)', default=None, type=str)
arg_parser.add_argument('--output_dir', default='./outputs/04_proximity-enrichments/', help='output dir')
arg_parser.add_argument('--plot_dir', default='./plots/04_proximity-enrichments/', help='plots dir')
arg_parser.add_argument('--min_as_captured', help='Minimum coverage to be considered as captured.', default=1, type=float)
arg_parser.add_argument('--gs_width', help='Width of applied Gaussian kernel.', default=0.75, type=float)
arg_parser.add_argument('--bin_width', help='Width of bins used to discretize each chromosome.', default=75000, type=float)
arg_parser.add_argument('--roi_width', help='Width of region around VP where significant bins are ignored.', default=3e6, type=float)
arg_parser.add_argument('--n_epoch', help='Number of random permutation.', default=1000, type=int)
arg_parser.add_argument('--n_step', help='Number of bins per gaussian.', default=31, type=int)
arg_parser.add_argument('--n_topbins', help='Number of top-enriched bins to be stored in the output.', default=3000, type=int)
arg_parser.add_argument('--down_sample', help='Used for downsampling experiments', default=None, type=str)
arg_parser.add_argument('--store_all_enrichments', action='store_true')
arg_parser.add_argument('--draw_plot', action='store_true')
arg_parser.add_argument('--overwrite', action='store_true')
# parser_obj.add_argument('--no_sanity_checks', action='store_true')
inp_args = arg_parser.parse_args()

# Load vp info
vp_info = get_vp_info(inp_args.expr_index, vp_info_path=inp_args.viewpoints)

# define the output name, change it if requested
if inp_args.output_id is None:
    inp_args.output_id = \
        '{:s}_gw{:0.2f}_bw{:0.0f}kb_'.format(vp_info['expr_id'], inp_args.gs_width, inp_args.bin_width / 1e3) + \
        'cl{:0.0f}-X_roi-w{:0.1f}m_'.format(inp_args.min_as_captured, inp_args.roi_width / 1e6) + \
        'nstep{:d}_nepoch{:0.1f}k'.format(inp_args.n_step, inp_args.n_epoch / 1e3)
plot_fpath = os.path.join(inp_args.plot_dir, inp_args.output_id + '.pdf')
out_fpath_all = os.path.join(inp_args.output_dir, inp_args.output_id + '.allbins.tsv.gz')
out_fpath_top = os.path.join(inp_args.output_dir, inp_args.output_id + '.topbins.tsv.gz')
if not inp_args.overwrite and os.path.isfile(out_fpath_top):
    print('[i] Output file is found. Stopping the execution: {:s}'.format(plot_fpath))
    exit(0)

# reporting the stats
print('Running the translocation pipeline using:\n' +
      '\tExperiment id: {:s}\n'.format(vp_info['expr_id']) +
      '\tBin width: {:0.0f}\n'.format(inp_args.bin_width) +
      '\tGaussian width: {:0.2f}\n'.format(inp_args.gs_width) +
      '\t#epoch: {:d}\n'.format(inp_args.n_epoch) +
      '\t#steps: {:d}\n'.format(inp_args.n_step))

# Make gaussian kernel
gus_krn = norm(loc=0, scale=inp_args.gs_width).pdf(np.linspace(-3, 3, inp_args.n_step))
gus_krn = gus_krn / np.sum(gus_krn)
# plt.plot((np.arange(args.n_step) - args.n_step // 2) * args.bin_width, gus_krn, label=(args.gs_width, args.n_step, args.bin_width))

# get chromosome info
chr_lst = get_chr_info(vp_info['genome'], property='chr_name')
chr_size = get_chr_info(vp_info['genome'], property='chr_size')
n_chr = len(chr_lst)

# load RE positions
print('Loading 1st ResEnz: {:s}'.format(vp_info['res_enzyme']))
re1_pos_lst = get_re_info(re_name=vp_info['res_enzyme'], property='pos', genome=vp_info['genome'])
if ('second_cutter' in vp_info) and (isinstance(vp_info['second_cutter'], str)):
    print('Loading 2nd ResEnz: {:s}'.format(vp_info['second_cutter']))
    re2_pos_lst = get_re_info(re_name=vp_info['second_cutter'], property='pos', genome=vp_info['genome'])
else:
    re2_pos_lst = [np.empty(0, dtype=int)] * n_chr

# load data
data_pd = load_dataset(vp_info, target_field='frg_np', verbose=True, data_path=inp_args.dataset_dir)
data = data_pd[['chr', 'pos', '#read']].values.astype('int32')
del data_pd
vp_info['#rd_all'] = np.sum(data[:, 2])

# Downsampling, if requested
if inp_args.down_sample:
    # TODO: Adding other types of downsampling, such as #captures
    assert inp_args.down_sample[:4] == 'nmap'
    n_map = int(float(inp_args.down_sample[4:]))
    print('Downsampling #mapped: From {:,d} mapped fragment to {:0,.0f} fragment.'.format(np.sum(data[:, 2]), n_map))
    idx_set = np.repeat(np.arange(data.shape[0]), data[:, 2])
    ds_set = np.random.choice(idx_set, size=n_map, replace=False)
    # Note: Here, we are only downsampling covered restriction fragments, empty restriction fragments are not selected
    del idx_set
    rf_uid, rf_frq = np.unique(ds_set, return_counts=True)
    data[:, 2] = 0
    data[rf_uid, 2] = rf_frq
    data = data[data[:, 2] > 0, :]
    del ds_set, rf_uid, rf_frq, n_map

# Calculate general statistics
is_cis = data[:, 0] == vp_info['vp_chr']
is_vp = is_cis & \
        (data[:, 1] >= vp_info['vp_be']) & \
        (data[:, 1] <= vp_info['vp_en'])
is_lcl = is_cis & ~is_vp
vp_info['#rf_cis'] = np.sum( is_cis)
vp_info['#rf_trs'] = np.sum(~is_cis)
vp_info['#rd_ivp'] = np.sum(data[is_vp, 2])
vp_info['#rd_nvp'] = np.sum(data[~is_vp, 2])
vp_info['#rd_cis'] = np.sum(data[is_cis, 2])
vp_info['#rd_trs'] = np.sum(data[~is_cis, 2])
vp_info['#rd%_T/C'] = vp_info['#rd_trs'] * 1e2 / vp_info['#rd_cis']
vp_info['#cpt_lcl'] = np.sum(data[is_lcl, 2] >= inp_args.min_as_captured)
vp_info['#cpt_cis'] = np.sum(data[is_cis, 2] >= inp_args.min_as_captured)
vp_info['#cpt_trs'] = np.sum(data[~is_cis, 2] >= inp_args.min_as_captured)
vp_info['#cpt_nvp'] = np.sum(data[:, 2] >= inp_args.min_as_captured)
vp_info['cpt%_L/C'] = vp_info['#cpt_lcl'] * 1e2 / vp_info['#cpt_cis']
vp_info['cpt%_L/A'] = vp_info['#cpt_lcl'] * 1e2 / vp_info['#cpt_nvp']
vp_info['cpt%_C/T'] = vp_info['#cpt_cis'] * 1e2 / vp_info['#cpt_trs']
del is_cis
# if (not inp_args.no_sanity_checks) and (vp_info['#rf_trs'] <= 100):
#     print('[w] #trans frag-ends is {:,d}. This run is ignored.'.format(vp_info['#rf_trs']))
#     exit(0)
# if (not inp_args.no_sanity_checks) and (vp_info['#rd_nvp'] < 1000):
#     print('[w] Only {:0.0f} (non-vp) reads are mapped. Stopping ...'.format(vp_info['#rd_nvp']))
#     exit(0)
print('Stats of loaded data are:')
for stat_name in ['#rd_all', '#rd_ivp', '#rd_nvp', '#rd_cis', '#rd_trs', '#cpt_lcl', '#cpt_cis', '#cpt_trs']:
    print('\t{:s}: {:,d}'.format(stat_name, vp_info[stat_name]))
print('\tcpt%_L/A: {:0,.1f}%'.format(vp_info['cpt%_L/A']))

# Calculate res-frag coverage of each bin
print('Calculating res-frag coverage for each bin')
bin_info = pd.DataFrame()
for chr_nid in np.unique(data[:, 0]):
    is_sel = data[:, 0] == chr_nid
    data_sel = data[is_sel, :]
    print('{:s}, '.format(chr_lst[chr_nid - 1]), end='')

    # get RE pos
    res_sites = np.unique(np.hstack([
        re1_pos_lst[chr_nid - 1],
        re2_pos_lst[chr_nid - 1]
    ]))

    # get capture efficiency
    if chr_nid == vp_info['vp_chr']:
        for lcl_w in [100e3, 200e3, 500e3]:
            col_name = 'cpt%_{:0.0f}kb'.format(lcl_w / 1e3)
            n_lcl_rf = np.sum(
                ((res_sites < vp_info['vp_be']) |
                 (res_sites > vp_info['vp_en'])) &
                (res_sites > vp_info['vp_be'] - lcl_w) &
                (res_sites < vp_info['vp_en'] + lcl_w)
            )
            is_cpt = ((data_sel[:, 1] < vp_info['vp_be']) |
                      (data_sel[:, 1] > vp_info['vp_en'])) & \
                     ((data_sel[:, 1] > vp_info['vp_be'] - lcl_w) &
                      (data_sel[:, 1] < vp_info['vp_en'] + lcl_w)) & \
                     (data_sel[:, 2] >= inp_args.min_as_captured)
            vp_info[col_name] = np.sum(is_cpt) * 1e2 / n_lcl_rf
        del is_cpt

    # Make bin list
    edge_lst = np.arange(0, chr_size[chr_nid - 1] + inp_args.bin_width, inp_args.bin_width, dtype=np.int)
    n_bin = len(edge_lst) - 1
    if n_bin < inp_args.n_step:
        continue

    # partition fragments to bins
    fe_gidx = np.searchsorted(edge_lst, data_sel[:, 1], side='right') - 1

    # Adding results to list
    bin_chr = pd.DataFrame()
    bin_chr['chr'] = np.tile(chr_nid, n_bin)
    bin_chr['pos'] = edge_lst[:-1]
    bin_chr['#restriction'] = np.bincount(np.searchsorted(edge_lst, res_sites, side='right') - 1, minlength=n_bin)
    bin_chr['#read'] = np.bincount(fe_gidx, weights=data_sel[:, 2], minlength=n_bin).astype(np.int32)
    if chr_nid == vp_info['vp_chr']:
        bin_chr['coverage'] = bin_chr['#read'] * 1e6 / float(vp_info['#rd_cis'])
    else:
        bin_chr['coverage'] = bin_chr['#read'] * 1e6 / float(vp_info['#rd_trs'])
    bin_chr['#capture'] = np.bincount(fe_gidx, minlength=n_bin)

    # append the data
    bin_info = bin_info.append(bin_chr, ignore_index=True)
    del bin_chr
print()

# calculate background distribution
print('Calculating the background')
bgp_edges = {}
bgp_freq = {}
bgp_stat = {}
is_cis = bin_info['chr'] == vp_info['vp_chr']
vp_info['avg_cvg'] = np.mean(bin_info.loc[~ is_cis, '#read'])
vp_info['avg_cpt'] = np.mean(bin_info.loc[~ is_cis, '#capture'])
for chr_nid in np.unique(bin_info['chr']):
    is_sel = bin_info['chr'] == chr_nid
    bin_type = 'cis' if chr_nid == vp_info['vp_chr'] else 'trans'
    print('Convolving {:5s} ({:5s}), '.format(chr_lst[chr_nid-1], bin_type) +
          '#res={:7,.1f}k, '.format(np.sum(bin_info.loc[is_sel, '#restriction']) / 1e3) +
          '#read={:5,.1f}k, '.format(np.sum(bin_info.loc[is_sel, '#read']) / 1e3) +
          '#cpt={:5.0f} '.format(np.sum(bin_info.loc[is_sel, '#capture'])), end='')

    # Compute score significance
    cur_time = time()
    for background_method in background_methods:
        nonzero_only = background_method.endswith('swap')
        if background_method.startswith('cpt-'):
            source_colname = '#capture'
        else:
            assert background_method == 'cvrg'
            source_colname = 'coverage'
        if chr_nid == vp_info['vp_chr']:

            # find probed area
            bin_pos = bin_info.loc[is_sel, 'pos'].values
            bin_obs = bin_info.loc[is_sel, source_colname].values.copy()
            is_prob = (bin_pos >= vp_info['vp_be'] - inp_args.bin_width - 250e3) & \
                      (bin_pos <= vp_info['vp_en'] + inp_args.bin_width + 250e3)
            pdx_be, pdx_en = np.where(is_prob)[0][[0, -1]]

            # correct the profile
            bin_prd = bin_obs.copy()
            bin_prd[:pdx_be] = apply_isotonic(bin_obs[:pdx_be][::-1])[::-1]
            bin_prd[pdx_en:] = apply_isotonic(bin_obs[pdx_en:])

            bin_decay_corrected = np.maximum(bin_obs - bin_prd, 0)

            # plt.close('all')
            # plt.figure()
            # plt.plot(bin_pos, bin_obs, alpha=0.5, label='Observe')
            # plt.plot(bin_pos, bin_prd, alpha=0.5, label='Prediction')
            # plt.plot(bin_pos, bin_decay_corrected, alpha=0.5, label='Corrected')
            # plt.axvline(x=vp_info['vp_be'], linestyle=':', color='k')
            # plt.axvline(x=vp_info['vp_en'], linestyle=':', color='k')
            # plt.xlim([vp_info['vp_be'] - 1e6, vp_info['vp_en'] + 1e6])
            # plt.legend()

            [score_observed, bkgnd_mat] = perform_sigtest(bin_decay_corrected, gus_krn, background=bin_decay_corrected,
                                                          n_epoch=inp_args.n_epoch * 10, nz_only=nonzero_only)
            MAX_BG_PEAK = np.max(bin_decay_corrected)
        else:
            is_bg = bin_info['chr'] != vp_info['vp_chr']
            [score_observed, bkgnd_mat] = perform_sigtest(bin_info.loc[is_sel, source_colname].values, gus_krn,
                                                          background=bin_info.loc[is_bg, source_colname].values,
                                                          n_epoch=inp_args.n_epoch, nz_only=nonzero_only)
            MAX_BG_PEAK = np.max(bin_info.loc[is_bg, source_colname])

        # add to background peak collection
        store_colname = background_method + '_' + bin_type
        peaks = bkgnd_mat.flatten()
        if store_colname not in bgp_edges:
            bgp_edges[store_colname] = np.append(np.linspace(0, MAX_BG_PEAK, num=n_bg_splt - 1), np.inf)
            bgp_freq[store_colname] = np.zeros(n_bg_splt - 1, dtype=np.int64)
            bgp_stat[store_colname] = OnlineStats()
        if np.max(peaks) > bgp_edges[store_colname][-2]:
            is_outbnd = peaks > bgp_edges[store_colname][-2]
            print('\n[w] {:,d} large peaks are found in the "{:s}" background: '.format(np.sum(is_outbnd), store_colname) +
                  'Trimmed to max(observed signal)={:f}.'.format(bgp_edges[store_colname][-2]))
            peaks[is_outbnd] = bgp_edges[store_colname][-2]
            del is_outbnd
        freqs = np.histogram(peaks, bins=bgp_edges[store_colname])[0]
        bgp_freq[store_colname] = bgp_freq[store_colname] + freqs
        bgp_stat[store_colname] = bgp_stat[store_colname].combine(peaks.astype(np.float64))
        print('| {:s}: avg(obs, bg)={:8.04f}, {:8.04f} '.format(background_method, np.mean(score_observed), bgp_stat[store_colname].mean), end='')
        del peaks, freqs

        # Calculate z-score and pvalues
        bkgnd_avg = np.mean(bkgnd_mat, axis=0)
        bkgnd_std = np.std(bkgnd_mat, axis=0)
        del bkgnd_mat
        np.seterr(invalid='ignore')
        bin_zscr = (score_observed - bkgnd_avg) / bkgnd_std
        np.seterr(invalid=None)
        # bin_pval = 1 - norm.cdf(bin_zscr)
        # bin_pval[bin_pval == 0] = MIN_PVAL
        # bin_pval[np.isnan(bin_pval)] = 1

        bin_info.loc[is_sel, background_method + '_obs'] = score_observed
        bin_info.loc[is_sel, background_method + '_avg'] = bkgnd_avg
        bin_info.loc[is_sel, background_method + '_std'] = bkgnd_std
        bin_info.loc[is_sel, background_method + '_bzs'] = bin_zscr
        # bin_info.loc[is_sel, score_name + '_pval'] = bin_pval
    print()
    # print('| took: {:0.1f}s'.format(time() - cur_time))

# Calculate peak compared p-values
# TODO: change how p-values are calculated, currently its difficult to get very small p-values, i.e. #peaks is small
for bin_type in ['cis', 'trans']:
    if bin_type == 'cis':
        is_sel = bin_info['chr'] == vp_info['vp_chr']
    else:
        is_sel = bin_info['chr'] != vp_info['vp_chr']
    n_bin = np.sum(is_sel)
    for background_method in background_methods:
        col_name = background_method + '_' + bin_type
        n_bg_peak = np.sum(bgp_freq[col_name])
        print('Calculating p-values for "{:7s}" using {:0.2f}m background peaks.'.format(col_name, n_bg_peak / 1e6))

        # calculate zscore/pvalues from aggregated distribution
        score_observed = bin_info.loc[is_sel, background_method + '_obs'].values
        bin_info.loc[is_sel, background_method + '_zscr'] = (score_observed - bgp_stat[col_name].mean) / bgp_stat[col_name].std

        failed_frq = np.zeros(n_bin, dtype=np.int64)
        for si in range(1, n_bg_splt):
            if bgp_freq[col_name][si - 1] == 0:
                continue
            is_smaller = score_observed <= bgp_edges[col_name][si]
            failed_frq[is_smaller] += bgp_freq[col_name][si - 1]
        failed_frq[failed_frq == 0] = 1  # To make sure 0 p-values are not produced
        bin_info.loc[is_sel, background_method + '_pval'] = failed_frq / float(n_bg_peak)

        print('\tCorrecting for {:0.0f} bins.'.format(np.sum(is_sel)))
        bin_info.loc[is_sel, background_method + '_qval'] = np.minimum(failed_frq * np.sum(is_sel) / float(n_bg_peak), 1)
        del score_observed, failed_frq, is_smaller

# combine p_values
bin_info['#cpt_zscr'] = np.min(bin_info[['cpt-shfl_zscr', 'cpt-swap_zscr']], axis=1)
bin_info['#cpt_pval'] = np.max(bin_info[['cpt-shfl_pval', 'cpt-swap_pval']], axis=1)
bin_info['#cpt_qval'] = np.max(bin_info[['cpt-shfl_qval', 'cpt-swap_qval']], axis=1)
bin_info['cmb_zscr'] = np.min(bin_info[[scr + '_zscr' for scr in background_methods]], axis=1)
bin_info['cmb_pval'] = np.max(bin_info[[scr + '_pval' for scr in background_methods]], axis=1)
bin_info['cmb_qval'] = np.max(bin_info[[scr + '_qval' for scr in background_methods]], axis=1)
# bin_info['cmb_zscr'] = np.maximum(norm.ppf(1 - bin_info['cmb_qval']), 0)

# correct p-values for multiple testing
# TODO: Correction factor for cis-bins is too strong; there are not many #background to reach small p-values
# bin_info['#cpt_qval'] = np.minimum(bin_info['#cpt_pval'] * bin_info.shape[0], 1)
# bin_info['cmb_qval'] = np.minimum(bin_info['cmb_pval'] * bin_info.shape[0], 1)

####################################################################################################################
# Output all windows
if inp_args.store_all_enrichments:
    os.makedirs(os.path.dirname(out_fpath_all), exist_ok=True)
    bin_info.to_csv(out_fpath_all, sep='\t', na_rep='nan', index=False, compression='gzip')
    print('All bins scores are saved to: {:s}'.format(out_fpath_all))

# Output top windows
is_roi = overlap([vp_info['vp_chr'], vp_info['vp_be'], vp_info['vp_en']],
                 bin_info[['chr', 'pos', 'pos']].values, offset=inp_args.roi_width)
bin_nroi = bin_info.loc[~is_roi].sort_values(by='#cpt_zscr', ascending=False).reset_index(drop=True)[:inp_args.n_topbins]
os.makedirs(os.path.dirname(out_fpath_top), exist_ok=True)
bin_nroi.to_csv(out_fpath_top, sep='\t', na_rep='nan', index=False)
assert bin_nroi['#cpt_zscr'].iat[-1] < 8.0, 'Some "top bins" could be cropped, increase #top_bins that are stored (current={:d}).'.format(inp_args.n_topbins)

# Plotting
if inp_args.draw_plot:
    plt.figure(figsize=(25, 7))
    ax_h = plt.gca()
    ax_h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,.0f}'.format(x)))

    # Plot important areas
    ax_h.add_patch(Rectangle((0, vp_info['vp_chr'] - 1), chr_size[vp_info['vp_chr'] - 1], 1, facecolor='#61c0ff', edgecolor='None', alpha=0.1))
    ax_h.add_patch(Rectangle((vp_info['vp_be'], vp_info['vp_chr'] - 1), vp_info['vp_en'] - vp_info['vp_be'], 1, facecolor='#fab700', edgecolor='None', alpha=1, zorder=100))
    plt.plot([vp_info['vp_pos'], vp_info['vp_pos']], [vp_info['vp_chr'] - 1, vp_info['vp_chr']], color='#0f5bff', linewidth=10, alpha=0.7, solid_capstyle='butt')

    # Mark top windows
    n_marked = 0
    n_sig = np.sum(bin_nroi['#cpt_zscr'] >= 8.0)
    clr_map = [cm.Reds(x) for x in np.linspace(0.8, 0.3, n_sig)]
    for ti in range(n_sig):
        top_bin = bin_nroi.loc[ti]
        if top_bin['#cpt_zscr'] < 8:
            break
        is_nei = (top_bin['chr'] == bin_nroi['chr'].iloc[:ti]) & \
                 (np.abs(top_bin['pos'] - bin_nroi['pos'].iloc[:ti]) < 10e6)
        if np.any(is_nei):
            continue
        plt.plot([top_bin['pos'], top_bin['pos']], [top_bin['chr'] - 1, top_bin['chr']],
                 linewidth=1.5, color=clr_map[n_marked], alpha=1.0, solid_capstyle='butt')
        plt.text(top_bin['pos'], top_bin['chr'] - 0.75, ' {:d},{:d}'.format(n_marked + 1, ti + 1),
                 verticalalignment='center', horizontalalignment='left', fontsize=4)
        n_marked = n_marked + 1

    # Plot profiles
    PROFILE_MAX = {
        '#read': bin_info.loc[bin_info['chr'] != vp_info['vp_chr'], '#read'].max(),
        '#capture': bin_info.loc[bin_info['chr'] != vp_info['vp_chr'], '#capture'].max(),
        '#cpt_zscr': 8.0,
        'cpt-shfl_zscr': 8.0,
        'cpt-swap_zscr': 8.0,
        'cvrg_zscr': 8.0,
        'cmb_zscr': 8.0
    }
    clr_map = ['#fc974a', '#000000', '#04db00']  # '#e2d003', '#a352ff'
    prf_name_lst = ['#read', '#capture', '#cpt_zscr']
    for chr_idx in range(n_chr):
        is_sel = bin_info['chr'] == (chr_idx + 1)
        if np.sum(is_sel) == 0:
            continue
        bin_sel = bin_info.loc[is_sel]

        chr_txt = ''
        plt_h = []
        for col_name in prf_name_lst:
            if col_name in PROFILE_MAX:
                scr_nrm = bin_sel[col_name] / PROFILE_MAX[col_name]
            else:
                scr_nrm = bin_sel[col_name] / np.max(bin_sel[col_name])
            scr_nrm[scr_nrm > 1] = 1
            scr_nrm[scr_nrm < 0] = 0
            plt_h.append(plt.plot(bin_sel['pos'], scr_nrm + chr_idx, color=clr_map[len(plt_h)], alpha=0.2, linewidth=0.2)[0])
            chr_txt += '{:0.1f}, '.format(np.nanmax(bin_sel[col_name]))
        plt.text(chr_size[chr_idx], 0.5 + chr_idx, chr_txt[:-2], fontsize=4, alpha=0.5, ha='right', va='center')
        plt_h[-1].set_alpha(0.7)
        plt_h[-1].set_linewidth(0.6)

        n_cpt = np.sum(bin_sel['#capture'])
        plt.text(0, 0.5 + chr_idx, '#cpt={:0.0f},cv={:0.2f}'.format(n_cpt, n_cpt * 1e7 / chr_size[chr_idx]), fontsize=4,
                 ha='left', va='center')
    plt.legend(plt_h, prf_name_lst, loc='upper right')

    # Final modifications
    plt.yticks(range(n_chr), chr_lst)
    plt.xlim([0, np.max(chr_size)+1e6])
    plt.ylim([-0.5, n_chr])
    plt.title(
        '[{:d}] {:s} (gw={:0.2f})\n'.format(vp_info['row_index'], vp_info['expr_id'], inp_args.gs_width) +
        '#rd: all={:0,.0f}; ivp={:0,.0f}; nvp={:0,.0f}; cis={:0,.0f}k; trans={:0,.0f}k, '.format(vp_info['#rd_all'], vp_info['#rd_ivp'], vp_info['#rd_nvp'], vp_info['#rd_cis'] / 1e3, vp_info['#rd_trs'] / 1e3) +
        '#cpt: cis={:0,.1f}k; trans={:0,.1f}k; '.format(vp_info['#cpt_cis'] / 1e3, vp_info['#cpt_trs'] / 1e3) +
        'avg_trs: cvg={:0.4f}, cpt={:0.4f}\n'.format(vp_info['avg_cvg'], vp_info['avg_cpt']) +
        'cpt(L/C, L/A, C/T): {:0.1f}%; {:0.1f}%; {:0.1f}%\n'.format(vp_info['cpt%_L/C'], vp_info['cpt%_L/A'], vp_info['cpt%_C/T']) +
        'cpt(100kb, 200kb, 500kb) / nvp: {:0.2f}; {:0.2f}; {:0.2f}\n'.format(vp_info['cpt%_100kb'], vp_info['cpt%_200kb'], vp_info['cpt%_500kb']) +
        '#sig={:,d}; max z-scores (ccpt, shfl, combined): {:0.1f}; {:0.1f}; {:0.1f}'.format(n_sig, *bin_nroi[[sn + '_zscr' for sn in background_methods + ['cmb']]].max())
    )
    os.makedirs(os.path.dirname(plot_fpath), exist_ok=True)
    plt.savefig(plot_fpath, bbox_inches='tight')
    print('Plot is saved to: {:s}'.format(plot_fpath))

