#! /usr/bin/env python3
# example run: python3 ./03_calculate_datasetsStatistics.py

import argparse
from sys import argv
from os import path, makedirs

import numpy as np
import pandas as pd

from utilities import get_vp_info, get_chr_info, get_re_info, load_dataset

# Initialization
np.set_printoptions(linewidth=180, threshold=5000)  # , suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 15)
vp_width = 10e3
tad_width = 100e3

# creating argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--vpi_file', default='./vp_info.tsv', type=str, help='VP information file')
parser.add_argument('--ridx_beg', default=-1, type=int)
parser.add_argument('--ridx_end', default=-1, type=int)
args = parser.parse_args(argv[1:])

# limit to specific experiment indices
if args.ridx_beg == -1:
    print('Processing datasets found in: {:s}'.format(args.vpi_file))
    vp_pd = pd.read_csv(args.vpi_file, sep='\t')
    run_idx_lst = range(vp_pd.shape[0])
    del vp_pd
else:
    run_idx_lst = range(args.ridx_beg, args.ridx_end + 1)
n_run = len(run_idx_lst)
print('Calculating stats for {:d} --> {:d} runs'.format(run_idx_lst[0], run_idx_lst[-1]))

# Loop over runs
stat_pd = pd.DataFrame()
re_dict = {}
for ri in range(n_run):

    # Get VP info
    vp_info = get_vp_info(run_idx_lst[ri])
    print('{:4d}/{:4d}: {:s}'.format(ri + 1, n_run, vp_info['run_id']))
    chr_lst = get_chr_info(vp_info['genome'], 'chr_name')
    chr_size = get_chr_info(vp_info['genome'], 'chr_size')
    n_chr = len(chr_lst)

    # get index of raw read stats
    if str(vp_info['source_fastq']) != 'nan':
        stat_fname = './fastqs/demuxed/{:s}/{:s}_stats.tsv'.format(vp_info['seqrun_id'],
                                                                   vp_info['source_fastq'].replace('.fastq.gz', ''))
        if path.isfile(stat_fname):
            dmx_stat = pd.read_csv(stat_fname, sep='\t')
            dmx_idx = np.where(vp_info['original_name'] == dmx_stat['original_name'])[0]
            assert len(dmx_idx) == 1
        else:
            if vp_info['assay_type'] == '4C':
                print('\t[w] Stat file is missing: {:s}'.format(stat_fname))
    else:
        print('\t[w] No source FASTQ file is defined, stats are not loaded')

    # get re information
    re_sid = '{:s}_{:s}'.format(vp_info['genome'], vp_info['first_cutter'])
    if str(vp_info['second_cutter']) != 'nan':
        re_sid += '-{:s}'.format(vp_info['second_cutter'])
    if re_sid not in re_dict:
        print('[i] Loading {:s}'.format(re_sid))
        re_dict[re_sid] = get_re_info(re_name=vp_info['first_cutter'], property='pos', genome=vp_info['genome'])
        if str(vp_info['second_cutter']) != 'nan':
            re2_lst = get_re_info(re_name=vp_info['second_cutter'], property='pos', genome=vp_info['genome'])
            assert n_chr == len(re_dict[re_sid]) == len(re2_lst)
            for ci in range(n_chr):
                re_dict[re_sid][ci] = np.unique(np.hstack([re_dict[re_sid][ci], re2_lst[ci]]))
            del re2_lst
    re_pos = re_dict[re_sid]

    # select cis/nvp restriction enzymes
    is_vp = np.abs(re_pos[vp_info['vp_chr'] - 1] - vp_info['vp_pos']) <= vp_width / 2.0
    re_cis = re_pos[vp_info['vp_chr'] - 1][~ is_vp]
    del is_vp

    # add initial info
    stat_prt = pd.DataFrame({'run_idx': [run_idx_lst[ri]]})
    stat_prt['run_id'] = vp_info['run_id']
    stat_prt['vp_chr'] = vp_info['vp_chr']
    stat_prt['vp_pos'] = vp_info['vp_pos']
    stat_prt['quality_status'] = 'passed'

    # load processed data
    h5_fname = './datasets/{:s}/{:s}.hdf5'.format(vp_info['seqrun_id'], vp_info['original_name'])
    if not path.isfile(h5_fname):
        stat_prt['quality_status'] = 'not_found'
        stat_pd = stat_pd.append(stat_prt, ignore_index=True, sort=False)
        print('[w] No dataset is found for: {:s}'.format(vp_info['run_id']))
        continue

    rf_pd = load_dataset(vp_info, target_field='frg_np', verbose=False)
    rf_all = rf_pd[['chr', 'pos', '#read']].values.astype('int32')
    del rf_pd

    # filter and select res-frg
    is_vp = (rf_all[:, 0] == vp_info['vp_chr']) & \
            (np.abs(rf_all[:, 1] - vp_info['vp_pos']) <= vp_width / 2.0)
    rf_nvp = rf_all[~ is_vp, :].copy()
    del is_vp

    # select cis and trans res-frg
    is_cis = rf_nvp[:, 0] == vp_info['vp_chr']
    rf_cis = rf_nvp[ is_cis].copy()
    rf_trs = rf_nvp[~is_cis].copy()

    # compute statistics
    if 'dmx_stat' in locals():
        stat_prt['dmx_#pass'] = dmx_stat.at[dmx_idx[0], '#read_passed']
        stat_prt['dmx_#fail'] = dmx_stat.at[dmx_idx[0], '#read_failed']
        stat_prt['dmx_pass%'] = stat_prt['dmx_#pass'] * 1e2 / (stat_prt['dmx_#pass'] + stat_prt['dmx_#fail'])
        stat_prt['dmx_bsize'] = dmx_stat.at[dmx_idx[0], 'barcode_size']
        del dmx_stat
    stat_prt['#rf_all'] = len(np.hstack(re_pos))
    stat_prt['#rf_nvp'] = stat_prt['#rf_all'] - (len(re_pos[vp_info['vp_chr'] - 1]) - len(re_cis))
    stat_prt['#rf_cis'] = len(re_cis)
    stat_prt['#rf_trs'] = stat_prt['#rf_all'] - len(re_pos[vp_info['vp_chr'] - 1])
    stat_prt['#rd_all'] = np.sum(rf_all[:, 2])
    stat_prt['#rd_nvp'] = np.sum(rf_nvp[:, 2])
    stat_prt['#rd_cis'] = np.sum(rf_cis[:, 2])
    stat_prt['#rd_trs'] = np.sum(rf_trs[:, 2])
    stat_prt['#cpt_all'] = np.sum(rf_all[:, 2] > 0)
    stat_prt['#cpt_nvp'] = np.sum(rf_nvp[:, 2] > 0)
    stat_prt['#cpt_cis'] = np.sum(rf_cis[:, 2] > 0)
    stat_prt['#cpt_trs'] = np.sum(rf_trs[:, 2] > 0)
    stat_prt['#rd_trs/#rd_cis'] = np.round(stat_prt['#rd_trs'] * 100.0 / stat_prt['#rd_cis'], 2)
    stat_prt['#cpt_cis/#cpt_nvp'] = np.round(stat_prt['#cpt_cis'] * 100.0 / stat_prt['#cpt_nvp'], 4)
    stat_prt['cpt%_cis'] = np.round(stat_prt['#cpt_cis'] * 100.0 / stat_prt['#rf_cis'], 4)

    for roi_width in [10, 25, 50, 100, 200, 500]:
        is_sel = np.abs(rf_cis[:, 1] - vp_info['vp_pos']) <= roi_width * 1e3
        n_sel_map = np.sum(rf_cis[is_sel, 2])
        n_sel_cpt = np.sum(rf_cis[is_sel, 2] > 0)

        width_str = '{:0.0f}kb'.format(roi_width)
        stat_prt['#rf_' + width_str] = np.sum(np.abs(re_cis - vp_info['vp_pos']) <= roi_width * 1e3)
        stat_prt['#rd_' + width_str] = n_sel_map
        stat_prt['#cpt_' + width_str] = n_sel_cpt
        stat_prt['#rd_' + width_str + '/#rd_cis'] = np.round(n_sel_map * 1e2 / stat_prt['#rd_cis'], 3)
        stat_prt['#rd_' + width_str + '/#rd_nvp'] = np.round(n_sel_map * 1e2 / stat_prt['#rd_nvp'], 3)
        stat_prt['#cpt_' + width_str + '/#cpt_cis'] = np.round(n_sel_cpt * 1e2 / stat_prt['#cpt_cis'], 3)
        stat_prt['#cpt_' + width_str + '/#cpt_nvp'] = np.round(n_sel_cpt * 1e2 / stat_prt['#cpt_nvp'], 3)
        stat_prt['cpt%_' + width_str] = np.round(n_sel_cpt * 1e2 / stat_prt['#rf_' + width_str], 3)

    # Compute down sampling effect
    n_map = np.sum(rf_nvp[:, 2])
    n_rf_tad = np.sum(np.abs(rf_cis[:, 1] - vp_info['vp_pos']) <= tad_width)
    rf_idx = np.repeat(np.arange(rf_nvp.shape[0]), rf_nvp[:, 2])
    for draw_size in [1, 3, 5, 7, 10, 15, 25, 50, 100]:
        ds_str = '{:0.0f}k'.format(draw_size)
        if draw_size * 1e3 > n_map:
            is_cpt = 0
        else:
            dwn_idx = np.random.choice(rf_idx, int(draw_size * 1e3), replace=False)
            rf_sub = rf_nvp[dwn_idx, :2]
            cis_cpt_pos = np.unique(rf_sub[rf_sub[:, 0] == vp_info['vp_chr'], 1])
            is_cpt = np.abs(cis_cpt_pos - vp_info['vp_pos']) <= tad_width
        stat_prt['seqSat_' + ds_str] = np.round(np.sum(is_cpt) * 1e2 / n_rf_tad, 2)

    # compute multiplicity of runs
    edge_lst = np.hstack([0, np.logspace(0, 10, 11, base=2).astype(np.int), np.inf])
    n_edge = len(edge_lst)
    for roi_width in [50, 100, 200]:
        is_sel = np.abs(rf_cis[:, 1] - vp_info['vp_pos']) <= roi_width * 1e3
        bin_idx = np.digitize(rf_cis[is_sel, 2], edge_lst) - 1
        mul_frq = np.bincount(bin_idx, minlength=n_edge - 1)
        for ei in range(n_edge - 1):
            stat_prt['mul_{:0.0f},{:0.0f}kb'.format(edge_lst[ei], roi_width)] = mul_frq[ei]

    # check coverage sufficiency
    if (stat_prt.at[0, '#rd_trs'] <= 100) or (stat_prt.at[0, '#rd_all'] < 1000):
        stat_prt['quality_status'] = 'failed'

    # appending the stats
    stat_pd = stat_pd.append(stat_prt, ignore_index=True, sort=False)

# Output results
makedirs('./outputs/', exist_ok=True)
out_name = './outputs/03_datasets_statistics_{:d}-{:d}.tsv'.format(run_idx_lst[0], run_idx_lst[-1])
stat_pd.to_csv(out_name, sep='\t', na_rep='nan', index=False)