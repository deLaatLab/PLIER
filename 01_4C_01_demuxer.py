#! /usr/bin/env python3

import argparse
import json
import sys
from os import path, makedirs
import numpy as np
import pandas as pd
import pysam
import gzip
import re
from matplotlib import pyplot as plt

from Levenshtein import distance

# initialization
np.set_printoptions(linewidth=180, threshold=5000)  # , suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 25)
n_class = 2
n_top = 10

# creating argument parser
parser = argparse.ArgumentParser(description="Demultiplexer script using given primer sequences")
parser.add_argument('--vpi_file', default=None, type=str, help='VP information file')
parser.add_argument('--seqrun', default='VER3386', type=str, help='Limits demuxing to specific sequencing runs')
parser.add_argument('--input_dir', default='./fastqs/raw', type=str, help='input dir')
parser.add_argument('--output_dir', default='./fastqs/demuxed', type=str, help='output dir')
parser.add_argument('--primer_tolerance', default=0, type=int, help='Number of mismatches allowed in primer seq')
parser.add_argument('--barcode_tolerance', default=20, type=float, help='Barcode mismatch tolerance')
parser.add_argument('--lane_index', default=None, type=str, help='Limits trimming to a specific lane indices')
args = parser.parse_args(sys.argv[1:])
with open('./configs.json', 'r') as cnf_fid:
    configs = json.load(cnf_fid)

# load vp info file
if args.vpi_file is None:
    args.vpi_file = '{:s}/{:s}/vp_info.tsv'.format(args.input_dir, args.seqrun)
vp_pd = pd.read_csv(args.vpi_file, sep='\t')
assert vp_pd.shape[0] == len(np.unique(vp_pd['run_id']))
assert vp_pd.shape[0] == len(np.unique(vp_pd['original_name']))
print('{:,d} experiments are loaded from: {:s}'.format(vp_pd.shape[0], args.vpi_file))

# identify selected source file
if args.seqrun:
    print('[i] Limiting demux to following sequencing runs: {:s}'.format(args.seqrun))
    is_sel = np.isin(vp_pd['seqrun_id'], args.seqrun.split(','))
    vp_pd = vp_pd.loc[is_sel].reset_index(drop=True)

# group experiments by sequencing lane
lane_grp = vp_pd.groupby(by='source_fastq', sort=False)
print('{:d} groups are found in the given vp_info file.'.format(lane_grp.ngroups))
for li, (lane_name, lane_indices) in enumerate(lane_grp):
    print('\t{:d}: {:s}'.format(li, lane_name))

if args.lane_index is None:
    args.lane_index = range(lane_grp.ngroups)
else:
    print('[i] Limiting demux to following lane indices: {:s}'.format(args.lane_index))
    args.lane_index = [int(x) for x in args.lane_index.split(',')]

# loop over each source fastq file
for src_idx, (source_filename, expr_pd) in enumerate(lane_grp):
    if src_idx not in args.lane_index:
        continue
    expr_pd = expr_pd.reset_index(drop=True).copy()
    expr_pd['barcode_size'] = expr_pd['vpfe_seq'].str.len() - expr_pd['primer_seq'].str.len()
    n_exp = expr_pd.shape[0]

    # identify the source file
    source_id = re.sub(r'\..*', '', source_filename)
    seqrun_id = np.unique(expr_pd['seqrun_id'])
    assert len(seqrun_id) == 1
    seqrun_id = seqrun_id[0]
    source_fastq = path.join(args.input_dir, seqrun_id, source_filename)
    # if not path.isfile(source_fastq):
    #     print('[w] Source FASTQ file could not be found: {:s}, moving to next source.'.format(source_fastq)
    #     continue
    print('Demultiplexing {:d} experiments from: {:s}'.format(n_exp, source_fastq))

    # create a dictionary of
    prm_seq = expr_pd['primer_seq'].to_list()
    vpf_seq = expr_pd['vpfe_seq'].to_list()
    prm_size = expr_pd['primer_seq'].str.len().values
    vpf_size = expr_pd['vpfe_seq'].str.len().values
    barcode_size = expr_pd['barcode_size'].values
    cls_freq = np.zeros([n_exp, n_class], dtype=int)

    # check uniqueness/non-containment of the primer seqs
    for ei in range(n_exp):
        for ej in range(ei + 1, n_exp):
            assert distance(expr_pd.at[ei, 'primer_seq'], expr_pd.at[ej, 'primer_seq']) > args.primer_tolerance, expr_pd.loc[[ei, ej]]
            assert distance(expr_pd.at[ei, 'vpfe_seq'], expr_pd.at[ej, 'vpfe_seq']) > args.primer_tolerance, expr_pd.loc[[ei, ej]]

    # make file handles
    output_dir = path.join(args.output_dir, seqrun_id)
    makedirs(output_dir, exist_ok=True)
    fid_lst = [[None] * n_class for ei in range(n_exp)]
    unmatched_fname = path.join(output_dir, source_id + '_unmatched.fastq.gz')
    fid_lst.append(gzip.open(unmatched_fname, 'wt'))

    # loop over reads in the current source file
    print('Looping over reads in {:s}'.format(source_fastq))
    n_read = 0
    n_unmatched = 0
    with pysam.FastxFile(source_fastq) as src_fid:
        for rd_idx, read in enumerate(src_fid):
            if np.mod(rd_idx, 2000000) == 0:
                print('\t{:0,.0f} reads are processed'.format(rd_idx))
            n_read += 1

            # check distance to all primers
            hit_lst = []
            for ei in range(n_exp):
                if distance(read.sequence[:prm_size[ei]], prm_seq[ei]) <= args.primer_tolerance:
                    hit_lst.append(ei)
            n_hit = len(hit_lst)
            assert n_hit <= 1, '[e] Multiple experiments matched to:\n{:s}'.format(str(read))

            # categorize read by primer seq
            if n_hit == 0:  # none matched
                out_fid = fid_lst[-1]
                n_unmatched += 1
            else:  # check complete sequence
                mch_idx = hit_lst[0]  # must be only one here
                mm_dist = distance(read.sequence[:vpf_size[mch_idx]], vpf_seq[mch_idx])
                if mm_dist > barcode_size[mch_idx] * (args.barcode_tolerance / 1e2):
                    rd_cls = 1
                    read.name += '_nm{:d}'.format(mm_dist)
                else:
                    rd_cls = 0

                # check if output file needs to be made
                if fid_lst[mch_idx][rd_cls] is None:
                    out_sid = path.join(output_dir, expr_pd.at[mch_idx, 'original_name'])
                    if rd_cls != 0:
                        out_sid += '_failed'
                    fid_lst[mch_idx][rd_cls] = gzip.open(out_sid + '.fastq.gz', 'wt')

                # select proper file handle
                out_fid = fid_lst[mch_idx][rd_cls]
                cls_freq[mch_idx, rd_cls] += 1

            # output the read
            out_fid.write(str(read) + '\n')
    print('\tfinished processing {:0,.0f} reads'.format(n_read))

    # close file handels
    for mch_idx in range(n_exp):
        for rd_cls in range(n_class):
            if fid_lst[mch_idx][rd_cls]:
                fid_lst[mch_idx][rd_cls].close()
    fid_lst[-1].close()

    # save stats
    expr_pd['#read_passed'] = cls_freq[:, 0]
    expr_pd['#read_failed'] = cls_freq[:, 1]
    expr_pd['pass_ratio'] = np.round(cls_freq[:, 0] * 1e2 / np.sum(cls_freq, axis=1), 2)
    stat_fname = path.join(output_dir, source_id + '_stats.tsv')
    expr_pd.to_csv(stat_fname, sep='\t', index=False)

    # plot class frequencies
    plt.figure(figsize=[10, 7])
    cls_err = np.sum(cls_freq[:, 1:], axis=1)
    sort_idx = np.argsort(cls_err)[::-1]
    bar_h = [None] * n_class
    clr_lst = ['#f0e000', '#ff1f1f']
    # x_lim = [0, np.max(cls_freq[:, 1:]) * 1.1 + 1]
    x_lim = [0, 25000]
    n_bar = np.min([n_top, n_exp])
    for bi in range(n_bar):

        # pass ratio
        pass_ratio = float(cls_freq[sort_idx[bi], 0]) / np.sum(cls_freq[sort_idx[bi]])
        bar_h[-1] = plt.barh(bi, pass_ratio * x_lim[1], alpha=0.5, height=1.0, color='#23ee11')
        plt.text(x_lim[1], bi, 'passed={:0.0f}% '.format(pass_ratio * 100), fontsize=8, va='center', ha='right')

        # number of passed/failed reads
        for ci in range(n_class - 1):
            bar_h[ci] = plt.barh(bi, cls_freq[sort_idx[bi], ci + 1], alpha=0.8, height=0.6 / (ci + 1), color=clr_lst[ci])
        bar_txt = '#pass={:,d}, '.format(cls_freq[sort_idx[bi], 0]) + \
                  '#fail={:,d}, '.format(cls_freq[sort_idx[bi], 1]) + \
                  'barcode={:d}bp'.format(barcode_size[sort_idx[bi]])
        plt.text(0, bi, ' ' + bar_txt, fontsize=8, va='center', ha='left')

    plt.yticks(range(n_bar), expr_pd.loc[sort_idx[:n_bar], 'original_name'], fontsize=8)
    plt.xlim(x_lim)
    plt.ylim([-0.5, n_top + 0.5])
    plt.xlabel('#failed reads')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: '{:,.0f}'.format(x)))
    plt.legend(bar_h, ['#read failed', 'passed ratio (%)'])
    plt.title('{:s}\n'.format(source_filename) +
              '#expriment={:d}, #read={:,d}, #unmatched={:,d}, '.format(n_exp, n_read, n_unmatched) +
              'barcode tolerance={:0.0f}%'.format(args.barcode_tolerance))

    # plt.show()
    fig_fname = path.join(output_dir, source_id + '_stats.pdf')
    plt.savefig(fig_fname, bbox_inches='tight')
    plt.close('all')



