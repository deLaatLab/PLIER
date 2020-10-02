#! /usr/bin/env python3
# example run:
# python3 01_TLC_01-demux_by_coords.py --seqrun_id='TLC-test' --patient_id='FXXX' --genome='hg19'

import argparse
import sys
from os import path, makedirs
from copy import deepcopy

import numpy as np
import pandas as pd
import pysam
from matplotlib import pyplot as plt

from utilities import get_chr_info, overlap, get_read

# initialization
np.set_printoptions(linewidth=180, threshold=5000)  # , suppress=True, formatter={'float_kind':'{:0.5f}'.format}
np.set_printoptions(formatter={'float_kind': '{:0.5f}'.format, 'int_kind': '{:,d}'.format})
pd.options.display.max_columns = 30
pd.options.display.max_rows = 100
pd.options.display.min_rows = 20
pd.options.display.width = 210
clr_map = plt.cm.get_cmap('jet', 25)

# creating argument parser
parser = argparse.ArgumentParser(description="Demultiplexer script using given prob info")
parser.add_argument('--seqrun_id', required=True)
parser.add_argument('--patient_id', required=True)
parser.add_argument('--genome', required=True)
parser.add_argument('--seqrun_tag', default='TLC', type=str)
parser.add_argument('--first_cutter', default='NlaIII')
parser.add_argument('--second_cutter', default=None)
parser.add_argument('--prbi_file', default='./probs/prob_info.tsv')
parser.add_argument('--bam_dir', default='./bams/')
parser.add_argument('--dmux_dir', default='./bams/')
parser.add_argument('--bin_w', default=20000, type=int)
parser.add_argument('--multiprob_dmux', action='store_true')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args(sys.argv[1:])
if args.seqrun_tag is None:
    args.seqrun_tag = args.seqrun_id
if args.second_cutter is None:
    args.second_cutter = ''
if args.multiprob_dmux:
    print('[w] Multi-prob is activated: A single fragment can be assigned to multi-probs.')

# get chromosome information
print('Collecting chromosome information from {:s} genome.'.format(args.genome))
chr_lst = get_chr_info(args.genome, property='chr_name')
chr_size = get_chr_info(args.genome, property='chr_size')
n_chr = len(chr_lst)
chr2nid = dict(zip(chr_lst, np.arange(n_chr) + 1))

# load prob info file
prb_pd = pd.read_csv(args.prbi_file, sep='\t')
prb_pd['chr_num'] = prb_pd['chr'].map(chr2nid)
prb_pd = prb_pd.sort_values(by=['chr_num', 'pos_begin']).reset_index(drop=True)
print('{:,d} probs are loaded from: {:s}'.format(prb_pd.shape[0], args.prbi_file))
# prb_pd['covered_area'] = prb_pd['pos_end'] - prb_pd['pos_begin']
# prb_pd.groupby('target_locus').sum().to_csv('./tmp.tsv', sep='\t')

# assign group ids to each prob
# TODO: Fragments mapping between prob-groups of the same locus are lost in this implementation
cur_gidx = 0
prb_pd['group_id'] = -1
for ei in range(prb_pd.shape[0]):
    if (prb_pd.at[ei, 'pos_begin'] > prb_pd.at[cur_gidx, 'pos_begin'] + args.bin_w) or \
            (prb_pd.at[ei, 'chr'] != prb_pd.at[cur_gidx, 'chr']) or \
            (prb_pd.at[ei, 'target_locus'] != prb_pd.at[cur_gidx, 'target_locus']):
        cur_gidx = ei
    prb_pd.at[ei, 'group_id'] = cur_gidx
assert not np.any(prb_pd['group_id'] == -1)
print('{:d} groups of probs are formed.'.format(len(np.unique(prb_pd['group_id']))))

# loop over prob groups and merge
prb_grp = prb_pd.groupby(by=['group_id'])
vpi_pd = pd.DataFrame()
for set_idx, (grp_idx, prb_set) in enumerate(prb_grp):
    print('{:02d}/{:02d} '.format(set_idx + 1, prb_grp.ngroups) +
          '{:s}, {:s}:{:d}-{:d}'.format(*prb_set.iloc[0][['target_locus', 'chr', 'pos_begin', 'pos_end']]))

    # make prob info
    locus_name = np.unique(prb_set['target_locus'])
    vp_chr = np.unique(prb_set['chr'])
    assert len(locus_name) == 1
    assert len(vp_chr) == 1
    locus_name = locus_name[0]
    vp_chr = vp_chr[0]
    vp_be = min(prb_set['pos_begin'])
    vp_en = max(prb_set['pos_end'])

    # construct read id
    vp_pos = int(np.mean([vp_be, vp_en]))
    if len(prb_pd.loc[prb_pd['target_locus'] == locus_name, 'group_id'].unique()) == 1:
        run_id = '{:s}_{:s}'.format(args.patient_id, locus_name)
    else:
        run_id = '{:s}_{:s}-{:0.3f}m'.format(args.patient_id, locus_name, vp_pos / 1e6)
    run_id += '_{:s}'.format(args.seqrun_tag)

    prob_info = pd.DataFrame({'run_id': [run_id]})
    prob_info['vp_chr'] = chr2nid[vp_chr]
    prob_info['vp_pos'] = vp_pos
    prob_info['vp_be'] = vp_be
    prob_info['vp_en'] = vp_en
    prob_info['vp_gene'] = locus_name
    prob_info['patient_id'] = args.patient_id
    prob_info['primer_id'] = '{:s},{:d}:{:0.3f}'.format(locus_name, chr2nid[vp_chr], vp_pos / 1e6)
    prob_info['original_name'] = run_id
    prob_info['genome'] = args.genome
    prob_info['first_cutter'] = args.first_cutter
    prob_info['second_cutter'] = args.second_cutter
    # prob_info['prob_size'] = vp_en - vp_be
    prob_info['source_fastq'] = '{:s}.fastq.gz'.format(run_id)
    prob_info['seqrun_id'] = args.seqrun_id

    vpi_pd = vpi_pd.append(prob_info, ignore_index=True)
n_expr = vpi_pd.shape[0]
assert n_expr == len(vpi_pd['run_id'].unique())
print('{:,d} experiments are formed.'.format(n_expr))
del prob_info

# output prob info
vpi_fname = path.join(args.dmux_dir, args.seqrun_id, '{:s}_vp_info.tsv'.format(args.patient_id))
if not path.isdir(path.dirname(vpi_fname)):
    makedirs(path.dirname(vpi_fname))
print('Exporting VP info file to: {:s}'.format(vpi_fname))
vpi_pd.to_csv(vpi_fname, sep='\t', index=False, compression=None)

# make file handles
source_fname = path.join(args.bam_dir, args.seqrun_id, args.patient_id + '.bam')
if not path.isfile(source_fname):
    print('[e] Source BAM file is not found: {:s}\nHave you mapped your FASTQ files?'.format(source_fname))
    exit(1)
print('Making BAM file handles from: {:s}'.format(source_fname))
with pysam.AlignmentFile(source_fname, 'rb') as source_fid:
    dmux_fids = []
    for ei in range(n_expr):
        dmux_fname = path.join(args.dmux_dir, args.seqrun_id, vpi_pd.at[ei, 'run_id'] + '.bam')
        dmux_fids.append(pysam.AlignmentFile(dmux_fname, 'wb', template=source_fid))
    unmatched_fname = path.join(args.dmux_dir, args.seqrun_id, args.patient_id + '_unmatched.bam')
    dmux_fids.append(pysam.AlignmentFile(unmatched_fname, 'wb', template=source_fid))

# loop over reads in the current source file
n_read = 0
n_unmatched = 0
n_multi_capture = 0
vp_crd = vpi_pd[['vp_chr', 'vp_be', 'vp_en']].values
print('Looping over reads in: {:s}'.format(source_fname))
with pysam.AlignmentFile(source_fname, 'rb') as src_fid:
    for rd_idx, read in enumerate(get_read(src_fid)):
        if np.mod(rd_idx, 1e6) == 0:
            print('\t{:,d} reads are processed'.format(rd_idx))
        n_read += 1

        # check overlap
        hit_vps = {}
        hit_length = np.zeros(n_expr + 1, dtype=int)
        for frg in read:
            frg_crd = [
                chr2nid[frg.reference_name],
                frg.reference_start,
                frg.reference_end
            ]
            is_ol = overlap(frg_crd, vp_crd)
            if any(is_ol):
                expr_idx = np.where(is_ol)[0]
                assert len(expr_idx) == 1, '[e] A single fragment is mapped to multiple experiments!'
                expr_idx = expr_idx[0]
                hit_length[expr_idx] += frg.get_overlap(vp_crd[expr_idx, 1], vp_crd[expr_idx, 2])
                if expr_idx not in hit_vps:  # coloring is based on the first fragment that maps to the VP
                    clr_ratio = float(np.mean(frg_crd[1:]) - vp_crd[expr_idx, 1]) / (vp_crd[expr_idx, 2] - vp_crd[expr_idx, 1])
                    hit_vps[expr_idx] = (
                        '{:d},{:d},{:d}'.format(*[int(x * 255) for x in clr_map(clr_ratio)[:3]]),
                        '{:d}:{:0.3f}'.format(vp_crd[expr_idx, 0], vp_crd[expr_idx, 1] / 1e6)
                    )
        n_hit = len(hit_vps)
        if n_hit == 0:
            n_unmatched += 1
            hit_vps[n_expr] = (
                '0,0,0',
                'None'
            )
        if n_hit > 1:
            n_multi_capture += 1

        # store the read
        hit_id = ','.join([hit[1][1] for hit in hit_vps.items()])
        max_hit_length = np.max(hit_length)
        for expr_idx, (rd_clr, vp_id) in hit_vps.items():
            if (not args.multiprob_dmux) and (hit_length[expr_idx] != max_hit_length):
                continue
            for frg in read:
                out = deepcopy(frg)
                out.query_name = 'np{:d}_{:s}_{:s}'.format(n_hit, hit_id, out.query_name)
                out.tags += [('YC', rd_clr)]
                dmux_fids[expr_idx].write(out)
print('{:,d} reads are demuxed successfuly, and {:,d} are unmatched.'.format(n_read, n_unmatched))
print('{:,d} reads are captured by more than one probes/VPs.'.format(n_multi_capture))

# closing file handles
for fi in range(len(dmux_fids)):
    dmux_fids[fi].close()
