#! /usr/bin/env python3
# example run: python3 ./04_compute_enrichment_scores.py

from sys import argv
import argparse
from os import path, makedirs
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
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
n_bg_splt = 1000
score_lst = ['cvrg', 'ccpt', 'shfl']
vp_width = 10e3

parser_obj = argparse.ArgumentParser(description='Computing gaussian smoothed signal of genomw wide 4C')
parser_obj.add_argument('--run_id', help='Run ID of the run.', default='0', type=str)
parser_obj.add_argument('--min_as_captured', help='Minimum coverage to be considered as captured.', default=1, type=float)
parser_obj.add_argument('--gs_width', help='Width of applied Gaussian kernel.', default=0.75, type=float)
parser_obj.add_argument('--bin_width', help='Width of bins used to discretize each chromosome.', default=75000, type=float)
parser_obj.add_argument('--roi_width', help='Width of region around VP where coverage is ignored.', default=3e6, type=float)
parser_obj.add_argument('--n_epoch', help='Number of random permutation.', default=1000, type=int)
parser_obj.add_argument('--n_step', help='Number of bins per gaussian.', default=31, type=int)
parser_obj.add_argument('--down_sample', help='Used for downsampling', default=None, type=int)
parser_obj.add_argument('--method', help='Type of analysis used.', default='v1.0', type=str)
parser_obj.add_argument('--overwrite', action='store_true')
args = parser_obj.parse_args(argv[1:])

# Load vp info
vp_info_lst = []
for run_idx in args.run_id.split(','):
    vp_info_lst.append(get_vp_info(int(run_idx)))
    assert vp_info_lst[0]['genome'] == vp_info_lst[-1]['genome']
    assert vp_info_lst[0]['vp_chr'] == vp_info_lst[-1]['vp_chr']
n_run = len(vp_info_lst)
vp_info = vp_info_lst[0]
run_sid = ','.join([vp_info_lst[i]['run_id'] for i in range(n_run)])

# make output file name
output_dir = '04_sv-caller_01-zscores-per-bin'
output_name = '{:s}_gw{:0.2f}_bw{:03.0f}kb_'.format(run_sid, args.gs_width, args.bin_width / 1e3) + \
              'cl{:0.0f}-X_roi-w{:0.1f}m_'.format(args.min_as_captured, args.roi_width / 1e6) + \
              'nstep{:02d}_nepoch{:02.1f}k_{:s}'.format(args.n_step, args.n_epoch / 1e3, args.method)
if not args.overwrite and path.isfile('./plots/{:s}/{:s}.pdf'.format(output_dir, output_name)):
    print('[w] Output plot is found. Exiting the run: {:s}'.format(output_name))
    exit(0)

# reporting the stats
print('Running the translocation pipeline using:\n' +
      '\tRun indices: {:s}\n'.format(args.run_id) +
      '\tBin width: {:0.0f}\n'.format(args.bin_width) +
      '\tGaussian width: {:0.2f}\n'.format(args.gs_width) +
      '\t#epoch: {:d}\n'.format(args.n_epoch) +
      '\t#steps: {:d}\n'.format(args.n_step) +
      '\tMethod: {:s}\n'.format(args.method))
print('Selected runs ids:')
for ri in range(n_run):
    print('\t{:2d}: {:4d}, {:s}'.format(ri + 1, vp_info_lst[ri]['row_index'], vp_info_lst[ri]['run_id']))

# stop if dataset does not exists
h5_fname = './datasets/{:s}/{:s}.hdf5'.format(vp_info['seqrun_id'], vp_info['original_name'])
if not path.isfile(h5_fname):
    print('[w] No dataset is found: {:s}'.format(h5_fname))
    exit(0)

# Make gaussian kernel
gus_krn = norm(loc=0, scale=args.gs_width).pdf(np.linspace(-3, 3, args.n_step))
gus_krn = gus_krn / np.sum(gus_krn)
# plt.plot((np.arange(args.n_step) - args.n_step // 2) * args.bin_width, gus_krn, label=(args.gs_width, args.n_step, args.bin_width))

# get chromosome info
chr_lst = get_chr_info(vp_info['genome'], property='chr_name')
chr_size = get_chr_info(vp_info['genome'], property='chr_size')
n_chr = len(chr_lst)

# load RE positions
print('Loading 1st ResEnz: {:s}'.format(vp_info['first_cutter']))
re1_pos_lst = get_re_info(re_name=vp_info['first_cutter'], property='pos', genome=vp_info['genome'])
if isinstance(vp_info['second_cutter'], str):
    print('Loading 2nd ResEnz: {:s}'.format(vp_info['second_cutter']))
    re2_pos_lst = get_re_info(re_name=vp_info['second_cutter'], property='pos', genome=vp_info['genome'])
else:
    re2_pos_lst = [np.empty(0)] * n_chr

# load data
rf_pd = load_dataset(vp_info_lst, target_field='frg_np', verbose=True, vp_width=vp_width)
rf_np = rf_pd[['chr', 'pos', '#read']].values.astype('int32')
del rf_pd

# removing prob area reads
# if vp_info['assay_type'] == 'TLCapture':
#     print('[i] Removing res-frags in probed area (+{:0.1f}k): '.format(vp_width / 1e3) +
#           '{:d}:{:,d}-{:,d}'.format(vp_info['vp_chr'], vp_info['vp_be'], vp_info['vp_en']))
#     is_nei = (rf_np[:, 0] == vp_info['vp_chr']) & \
#              (rf_np[:, 1] > vp_info['vp_be'] - vp_width) & \
#              (rf_np[:, 1] < vp_info['vp_en'] + vp_width)
#     rf_np = rf_np[~is_nei, :]

# Downsampling
if args.down_sample:
    print('Downsampling {:,d} reads to {:,d} reads.'.format(np.sum(rf_np[:, 2]), args.down_sample))
    rd_set = np.repeat(np.arange(rf_np.shape[0]), rf_np[:, 2])
    ds_set = np.random.choice(rd_set, size=args.down_sample, replace=False)
    rf_uid, rf_frq = np.unique(ds_set, return_counts=True)
    rf_np[:, 2] = 0
    rf_np[rf_uid, 2] = rf_frq
    del rd_set, ds_set, rf_uid, rf_frq

# ignore coverage on chromosomes
if ('ignore_chr' in vp_info) and (not np.isnan(vp_info['ignore_chr'])):
    print('[w] Ignoring coverage on {:s}'.format(chr_lst[int(vp_info['ignore_chr']) - 1]))
    rf_np = rf_np[~ np.isin(rf_np[:, 0], vp_info['ignore_chr'])]

# Calculate general statistics
is_cis = rf_np[:, 0] == vp_info['vp_chr']
is_lcl = is_cis & \
         (np.abs(rf_np[:, 1] - vp_info['vp_pos']) < 100e3)
vp_info['#rf_cis'] = np.sum( is_cis)
vp_info['#rf_trs'] = np.sum(~is_cis)
vp_info['#rd_nvp'] = np.sum(rf_np[      :, 2])
vp_info['#rd_cis'] = np.sum(rf_np[ is_cis, 2])
vp_info['#rd_trs'] = np.sum(rf_np[~is_cis, 2])
vp_info['#rd%_T/C'] = vp_info['#rd_trs'] * 1e2 / vp_info['#rd_cis']
vp_info['#cpt_lcl'] = np.sum(rf_np[ is_lcl, 2] >= args.min_as_captured)
vp_info['#cpt_cis'] = np.sum(rf_np[ is_cis, 2] >= args.min_as_captured)
vp_info['#cpt_trs'] = np.sum(rf_np[~is_cis, 2] >= args.min_as_captured)
vp_info['#cpt_nvp'] = np.sum(rf_np[      :, 2] >= args.min_as_captured)
vp_info['cpt%_L/C'] = vp_info['#cpt_lcl'] * 1e2 / vp_info['#cpt_cis']
vp_info['cpt%_L/A'] = vp_info['#cpt_lcl'] * 1e2 / vp_info['#cpt_nvp']
vp_info['cpt%_C/T'] = vp_info['#cpt_cis'] * 1e2 / vp_info['#cpt_trs']
del is_cis
if vp_info['#rf_trs'] <= 100:
    print('[w] #trans frag-ends is {:,d}. This run is ignored.'.format(vp_info['#rf_trs']))
    exit(0)
if vp_info['#rd_nvp'] < 1000:
    print('[w] Only {:0.0f} (non-vp) reads are mapped. Stopping ...'.format(vp_info['#rd_nvp']))
    exit(0)
print('Stats of loaded data are:')
for stat_name in ['#rd_nvp', '#rd_cis', '#rd_trs', '#cpt_lcl', '#cpt_cis', '#cpt_trs']:
    print('\t{:s}: {:,d}'.format(stat_name, vp_info[stat_name]))
print('\tcpt%_L/A: {:0,.1f}%'.format(vp_info['cpt%_L/A']))

# Calculate res-frag coverage of each bin
print('Calculating res-frag coverage for each bin')
bin_info = pd.DataFrame()
for chr_nid in np.unique(rf_np[:, 0]):
    is_sel = rf_np[:, 0] == chr_nid
    rf_sel = rf_np[is_sel, :]
    print('{:s}, '.format(chr_lst[chr_nid - 1]), end='')

    # get RE pos
    re_pos = np.unique(np.hstack([
        re1_pos_lst[chr_nid - 1],
        re2_pos_lst[chr_nid - 1]
    ]))

    # get capture efficiency
    if chr_nid == vp_info['vp_chr']:
        is_nei = np.abs(rf_sel[:, 1] - vp_info['vp_pos']) < vp_width
        for lcl_w in [100e3, 200e3, 500e3]:
            col_name = 'cpt%_{:0.0f}kb'.format(lcl_w / 1e3)
            n_lcl_rf = np.sum(
                (np.abs(re_pos - vp_info['vp_pos']) > vp_width) &
                (np.abs(re_pos - vp_info['vp_pos']) < lcl_w))
            is_cpt = (np.abs(rf_sel[:, 1] - vp_info['vp_pos']) > vp_width) & \
                     (np.abs(rf_sel[:, 1] - vp_info['vp_pos']) < lcl_w) & \
                     (rf_sel[:, 1] >= args.min_as_captured)
            vp_info[col_name] = len(np.unique(rf_sel[is_cpt, 1])) * 1e2 / n_lcl_rf
        del is_nei, is_cpt

    # Make bin list
    bin_lst = np.arange(0, chr_size[chr_nid - 1], args.bin_width, dtype=np.int)
    n_bin = len(bin_lst)
    if n_bin < args.n_step:
        continue

    # group fragments according to bin overlaps'
    fe_gidx = np.digitize(rf_sel[:, 1], bin_lst) - 1
    bin_n_rfrg = np.bincount(np.digitize(re_pos, bin_lst) - 1, minlength=n_bin)
    bin_n_read = np.bincount(fe_gidx, weights=rf_sel[:, 2], minlength=n_bin)
    bin_n_rcpt = np.bincount(fe_gidx, minlength=n_bin)
    if np.sum(bin_n_rcpt) < 2:
        continue

    # calculate binomial significance
    if chr_nid == vp_info['vp_chr']:
        bin_cvg = bin_n_read / float(vp_info['#rd_cis'])
    else:
        bin_cvg = bin_n_read / float(vp_info['#rd_trs'])

    # Adding results to list
    bin_cis = pd.DataFrame()
    bin_cis['chr'] = np.tile(chr_nid, n_bin)
    bin_cis['pos'] = bin_lst
    bin_cis['#restriction'] = bin_n_rfrg.astype(np.int32)
    bin_cis['#read'] = bin_n_read.astype(np.int32)
    bin_cis['cvrg'] = bin_cvg * 1e5
    bin_cis['#capture'] = bin_n_rcpt.astype(np.int32)
    bin_info = bin_info.append(bin_cis, ignore_index=True)
    del bin_cis
print()

# get scores
# if args.max_ncapture is None:
#     is_cis = bin_info['chr'] == vp_info['vp_chr']
#     args.max_ncapture = np.max(bin_info.loc[~ is_cis, '#capture'])
#     del is_cis
# print('Maximum #capture per bin is set to: {:0.1f}'.format(args.max_ncapture)
# bin_info['ccpt'] = np.minimum(bin_info['#capture'], MAX_N_CAPTURE)
# bin_info['shfl'] = np.minimum(bin_info['#capture'], MAX_N_CAPTURE)
bin_info['ccpt'] = bin_info['#capture'].copy()
bin_info['shfl'] = bin_info['#capture'].copy()

# removing VP neighborhood coverage
# print('Ignoring coverage from VP neighborhood (+/- {:0.2f}mb)'.format(args.roi_width / 1e6))
# is_nei = (bin_info['chr'] == vp_info['vp_chr']) & \
#          (np.abs(bin_info['pos'] - vp_info['vp_pos']) < args.roi_width)
# bin_info.loc[is_nei, 'ccpt'] = 0
# bin_info.loc[is_nei, 'shfl'] = 0
# del is_nei

# calculate background distribution
print('Calculating the background')
bgp_edge_lst = {}
bgp_freq = {}
bgp_stat = {}
is_cis = bin_info['chr'] == vp_info['vp_chr']
vp_info['avg_cvg'] = np.mean(bin_info.loc[~ is_cis, '#read'])
vp_info['avg_cpt'] = np.mean(bin_info.loc[~ is_cis, '#capture'])
for chr_nid in np.unique(bin_info['chr']):
    is_sel = bin_info['chr'] == chr_nid
    bin_type = 'cis' if chr_nid == vp_info['vp_chr'] else 'trans'
    print('Convolving {:5s} ({:5s}), '.format(chr_lst[chr_nid-1], bin_type) +
          '#res={:7,.1f}, '.format(np.sum(bin_info.loc[is_sel, '#restriction']) / 1e3) +
          '#read={:5,.1f}k, '.format(np.sum(bin_info.loc[is_sel, '#read']) / 1e3) +
          '#cpt={:5.0f} '.format(np.sum(bin_info.loc[is_sel, '#capture'])), end='')

    # Compute score significance
    cur_time = time()
    for score_name in score_lst:
        nz_only = score_name == 'shfl'
        if chr_nid == vp_info['vp_chr']:

            # normalize profile
            bin_pos = bin_info.loc[is_sel, 'pos'].values
            is_prob = (bin_pos >= vp_info['vp_be'] - args.bin_width - 250e3) & \
                      (bin_pos <= vp_info['vp_en'] + args.bin_width + 250e3)
            pdx_be = np.argmax(is_prob)
            pdx_en = len(is_prob) - np.argmax(is_prob[::-1]) - 1
            bin_obs = bin_info.loc[is_sel, score_name].values
            # bin_nrm = np.minimum(bin_obs, np.max(bin_obs[~is_prob]))
            # bin_nrm = bin_nrm * 1e2 / np.sum(bin_nrm)

            bin_prd = bin_obs.copy()
            bin_prd[:pdx_be] = apply_isotonic(bin_obs[:pdx_be][::-1])[::-1]
            bin_prd[pdx_en:] = apply_isotonic(bin_obs[pdx_en:])

            bin_decay_corrected = np.maximum(bin_obs - bin_prd, 0)

            # plt.close('all')
            # plt.figure()
            # plt.plot(bin_pos, bin_obs, alpha=0.5, label='Observe')
            # plt.plot(bin_pos, bin_prd, alpha=0.5, label='Prediction')
            # plt.plot(bin_pos, bin_crd, alpha=0.5, label='Corrected')
            # plt.xlim([vp_info['vp_be'] - 1e6, vp_info['vp_en'] + 1e6])
            # plt.legend()

            [score_observed, bkgnd_mat] = perform_sigtest(bin_decay_corrected, gus_krn, background=bin_decay_corrected,
                                                          n_epoch=args.n_epoch * 10, nz_only=nz_only)
            MAX_BG_PEAK = np.max(bin_decay_corrected)
        else:
            is_bg = bin_info['chr'] != vp_info['vp_chr']
            [score_observed, bkgnd_mat] = perform_sigtest(bin_info.loc[is_sel, score_name].values, gus_krn,
                                                          background=bin_info.loc[is_bg, score_name].values,
                                                          n_epoch=args.n_epoch, nz_only=nz_only)
            MAX_BG_PEAK = np.max(bin_info.loc[is_bg, score_name])

        # add to background peak collection
        col_name = score_name + '_' + bin_type
        peaks = bkgnd_mat.flatten()
        if col_name not in bgp_edge_lst:
            bgp_edge_lst[col_name] = np.append(np.linspace(0, MAX_BG_PEAK, num=n_bg_splt - 1), np.inf)
            bgp_freq[col_name] = np.zeros(n_bg_splt - 1, dtype=np.int64)
            bgp_stat[col_name] = OnlineStats()
        if np.max(peaks) > bgp_edge_lst[col_name][-2]:
            is_outbnd = peaks > bgp_edge_lst[col_name][-2]
            print('\n[w] {:,d} large peaks are found in the "{:s}" background: '.format(np.sum(is_outbnd), col_name) +
                  'Trimmed to max(observed signal)={:f}.'.format(bgp_edge_lst[col_name][-2]))
            peaks[is_outbnd] = bgp_edge_lst[col_name][-2]
            del is_outbnd
        freqs = np.histogram(peaks, bins=bgp_edge_lst[col_name])[0]
        bgp_freq[col_name] = bgp_freq[col_name] + freqs
        bgp_stat[col_name] = bgp_stat[col_name].combine(peaks.astype(np.float64))
        print('| {:s}: avg(obs, bg)={:8.04f}, {:8.04f} '.format(score_name, np.mean(score_observed), bgp_stat[col_name].mean), end='')
        del peaks, freqs

        # Calculate z-score and pvalues
        bkgnd_avg = np.mean(bkgnd_mat, axis=0)
        bkgnd_std = np.std(bkgnd_mat, axis=0)
        del bkgnd_mat
        np.seterr(invalid='ignore')
        bin_zscr = np.divide((score_observed - bkgnd_avg), bkgnd_std)
        np.seterr(invalid=None)
        # bin_pval = 1 - norm.cdf(bin_zscr)
        # bin_pval[bin_pval == 0] = MIN_PVAL
        # bin_pval[np.isnan(bin_pval)] = 1

        bin_info.loc[is_sel, score_name + '_obs'] = score_observed
        bin_info.loc[is_sel, score_name + '_avg'] = bkgnd_avg
        bin_info.loc[is_sel, score_name + '_std'] = bkgnd_std
        bin_info.loc[is_sel, score_name + '_bzs'] = bin_zscr
        # bin_info.loc[is_sel, score_name + '_pval'] = bin_pval
    print('| took: {:0.1f}s'.format(time() - cur_time))

# Calculate peak compared p-values
# TODO: change how p-values are calculated, currently its difficult to get very small p-values, i.e. #peaks is small
for bin_type in ['cis', 'trans']:
    if bin_type == 'cis':
        is_sel = bin_info['chr'] == vp_info['vp_chr']
    else:
        is_sel = bin_info['chr'] != vp_info['vp_chr']
    n_bin = np.sum(is_sel)
    for score_name in score_lst:
        col_name = score_name + '_' + bin_type
        n_bg_peak = np.sum(bgp_freq[col_name])
        print('Calculating p-values for "{:7s}" using {:0.2f}m background peaks.'.format(col_name, n_bg_peak / 1e6))

        # calculate zscore/pvalues from aggregated distribution
        score_observed = bin_info.loc[is_sel, score_name + '_obs'].values
        bin_info.loc[is_sel, score_name + '_zscr'] = (score_observed - bgp_stat[col_name].mean) / bgp_stat[col_name].std

        failed_frq = np.zeros(n_bin, dtype=np.int64)
        for si in range(1, n_bg_splt):
            if bgp_freq[col_name][si - 1] == 0:
                continue
            is_smaller = score_observed <= bgp_edge_lst[col_name][si]
            failed_frq[is_smaller] += bgp_freq[col_name][si - 1]
        failed_frq[failed_frq == 0] = 1  # To make sure 0 p-values are not produced
        bin_info.loc[is_sel, score_name + '_pval'] = failed_frq / float(n_bg_peak)

        print('\tCorrecting for {:0.0f} bins.'.format(np.sum(is_sel)))
        bin_info.loc[is_sel, score_name + '_qval'] = np.minimum(failed_frq * np.sum(is_sel) / float(n_bg_peak), 1)
        del score_observed, failed_frq, is_smaller

# combine p_values
bin_info['#cpt_zscr'] = np.min(bin_info[['ccpt_zscr', 'shfl_zscr']], axis=1)
bin_info['#cpt_pval'] = np.max(bin_info[['ccpt_pval', 'shfl_pval']], axis=1)
bin_info['#cpt_qval'] = np.max(bin_info[['ccpt_qval', 'shfl_qval']], axis=1)
bin_info['cmb_zscr'] = np.min(bin_info[[scr + '_zscr' for scr in score_lst]], axis=1)
bin_info['cmb_pval'] = np.max(bin_info[[scr + '_pval' for scr in score_lst]], axis=1)
bin_info['cmb_qval'] = np.max(bin_info[[scr + '_qval' for scr in score_lst]], axis=1)
# bin_info['cmb_zscr'] = np.maximum(norm.ppf(1 - bin_info['cmb_qval']), 0)

# correct p-values for multiple testing
# TODO: Correction for cis-bins is too strong; there are not many #background to reach small p-values
# bin_info['#cpt_qval'] = np.minimum(bin_info['#cpt_pval'] * bin_info.shape[0], 1)
# bin_info['cmb_qval'] = np.minimum(bin_info['cmb_pval'] * bin_info.shape[0], 1)

####################################################################################################################
# Output all windows
gz_fname = './outputs/{:s}/AllBins_{:s}/{:s}_AllBins.tsv.gz'.format(output_dir, args.method, output_name)
if not path.isdir(path.dirname(gz_fname)):
    makedirs(path.dirname(gz_fname))
bin_info.to_csv(gz_fname, sep='\t', na_rep='nan', index=False, compression='gzip')
print('All bins scores are saved to: {:s}'.format(gz_fname))

# Output top windows
is_nei = overlap([vp_info['vp_chr'], vp_info['vp_be'], vp_info['vp_en']],
                 bin_info[['chr', 'pos', 'pos']].values, offset=args.roi_width)
bin_nnei = bin_info.loc[~is_nei].reset_index(drop=True)
top_bidx = np.argsort(bin_nnei['#cpt_zscr'])[::-1]
tsv_fname = './outputs/{:s}/TopBins_{:s}/{:s}_TopBins.tsv.gz'.format(output_dir, args.method, output_name)
makedirs(path.dirname(tsv_fname), exist_ok=True)
bin_nnei.iloc[top_bidx[:3000]].to_csv(tsv_fname, sep='\t', na_rep='nan', index=False)

# Plotting
plt.figure(figsize=(25, 7))
ax_h = plt.gca()
ax_h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,.0f}'.format(x)))
ax_h.add_patch(Rectangle((0, vp_info['vp_chr'] - 1), chr_size[vp_info['vp_chr'] - 1], 1, facecolor='#61c0ff', edgecolor='None', alpha=0.1))
if n_run > 1:
    ant_h = plt.text(175e6, 7, '\n'.join([vp_info_lst[i]['run_id'] for i in range(n_run)]))

# Plot important areas
plt.plot([vp_info['vp_pos'], vp_info['vp_pos']], [vp_info['vp_chr'] - 1, vp_info['vp_chr']],
         color='#0f5bff', linewidth=6, alpha=0.7, solid_capstyle='butt')
if ('sv_lst' in vp_info) and (~ np.isnan(vp_info['sv_lst'])):
    for sv_crd in vp_info['sv_lst'].split(';'):
        sv_chr, sv_pos = [int(x) for x in sv_crd.split(':')]
        ax_h.add_patch(Rectangle((sv_pos - 3e6, sv_chr - 1), 6e6, 1, color='red', fill=False, linewidth=0.5, alpha=0.8))

# Mark top windows
n_marked = 0
clr_map = [cm.Reds(x) for x in np.linspace(0.8, 0.3, 5)]
for top_rank, top_idx in enumerate(top_bidx):
    top_bin = bin_nnei.iloc[top_idx]
    if (n_marked >= 5) | (top_bin['#cpt_qval'] > 0.05):
        break
    is_nei = (top_bin['chr'] == bin_nnei['chr'].iloc[top_bidx[:top_rank]]) & \
             (np.abs(top_bin['pos'] - bin_nnei['pos'].iloc[top_bidx[:top_rank]]) < 10e6)
    if np.any(is_nei):
        continue
    plt.plot([top_bin['pos'], top_bin['pos']], [top_bin['chr']-1, top_bin['chr']],
             linewidth=1.5, color=clr_map[n_marked], alpha=1.0, solid_capstyle='butt')
    plt.text(top_bin['pos'], top_bin['chr'] - np.random.rand(), ' {:d},{:d}'.format(n_marked+1, top_rank + 1),
             verticalalignment='center', horizontalalignment='left', fontsize=4)
    n_marked = n_marked + 1

# Plot profiles
PROFILE_MAX = {
    'ccpt': 5,
    '#capture': bin_info.loc[bin_info['chr'] != vp_info['vp_chr'], '#capture'].max(),
    'ccpt_zscr': 7.0,
    'shfl_zscr': 7.0,
    'cvrg_zscr': 7.0,
    'cmb_zscr': 7.0
}
clr_map = ['#000000', '#fc974a', '#e2d003', '#a352ff', '#04db00']
prf_name_lst = ['#capture', 'ccpt_zscr', 'shfl_zscr', 'cvrg_zscr', 'cmb_zscr']
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
    '[{:d}] {:s} (gw={:0.2f})\n'.format(vp_info['row_index'], run_sid, args.gs_width) +
    '#rd: nvp={:0,.0f}k; cis={:0,.0f}k; trans={:0,.0f}k, '.format(vp_info['#rd_nvp'] / 1e3, vp_info['#rd_cis'] / 1e3, vp_info['#rd_trs'] / 1e3) +
    '#cpt: cis={:0,.1f}k; trans={:0,.1f}k; '.format(vp_info['#cpt_cis'] / 1e3, vp_info['#cpt_trs'] / 1e3) +
    'avg_trs: cvg={:0.4f}, cpt={:0.4f}\n'.format(vp_info['avg_cvg'], vp_info['avg_cpt']) +
    'cpt(L/C, L/A, C/T): {:0.1f}%; {:0.1f}%; {:0.1f}%\n'.format(vp_info['cpt%_L/C'], vp_info['cpt%_L/A'], vp_info['cpt%_C/T']) +
    'cpt(100kb, 200kb, 500kb) / nvp: {:0.2f}; {:0.2f}; {:0.2f}\n'.format(vp_info['cpt%_100kb'], vp_info['cpt%_200kb'], vp_info['cpt%_500kb']) +
    'max z-scores (ccpt, shfl, combined): {:0.1f}; {:0.1f}; {:0.1f}'.format(*bin_nnei[[sn + '_zscr' for sn in score_lst + ['cmb']]].max())
)
plt_fname = './plots/{:s}/{:s}.pdf'.format(output_dir, output_name)
makedirs(path.dirname(plt_fname), exist_ok=True)
plt.savefig(plt_fname, bbox_inches='tight')
print('Plot is saved to: {:s}'.format(plt_fname))

