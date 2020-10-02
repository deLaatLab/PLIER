#! /usr/bin/env python3
# example run: python3 ./02_make_dataset.py

import argparse
import sys
from os import path, makedirs
import numpy as np
import pandas as pd
import pysam
from copy import deepcopy
import h5py

from utilities import get_chr_info, get_re_info, overlap

# initialization
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 35)
np.set_printoptions(formatter={'int_kind': '{:,d}'.format})  # 'float_kind': '{:0.5f}'.format,
gap_size = 10

# creating argument parser
parser = argparse.ArgumentParser(description="Collect mapped fragments into a dataset for the given experiment indices")
parser.add_argument('--vpi_file', default='./vp_info.tsv', type=str, help='VP information file')
parser.add_argument('--input_dir', default='./bams', help='input dir')
parser.add_argument('--output_dir', default='./datasets', help='output dir')
parser.add_argument('--expr_index', default='-1', help='Limits mapping to a specific experiment indices')
parser.add_argument('--min_mq', default=1, type=int, help='Minimum mapping quality (MQ) to consider reads as mapped')
args = parser.parse_args(sys.argv[1:])

# load vp info file
vp_pd = pd.read_csv(args.vpi_file, sep='\t')
assert vp_pd.shape[0] == len(np.unique(vp_pd['run_id']))
assert vp_pd.shape[0] == len(np.unique(vp_pd['original_name']))

# limit to specific experiment indices
if args.expr_index == '-1':
    print('Processing {:d} datasets found in: {:s}'.format(len(vp_pd), args.vpi_file))
    args.expr_index = range(vp_pd.shape[0])
else:
    print('Processing is limited to following experiment indices: {:s}'.format(args.expr_index))
    args.expr_index = [int(x) for x in args.expr_index.split(',')]
n_exp = len(args.expr_index)

# loop over experiments
for ei, exp_idx in enumerate(args.expr_index):

    # prepare file names
    vp_info = vp_pd.loc[exp_idx]
    input_bam = '{:s}/{:s}/{:s}.bam'.format(args.input_dir, vp_info['seqrun_id'], vp_info['original_name'])
    output_hdf5 = '{:s}/{:s}/{:s}.hdf5'.format(args.output_dir, vp_info['seqrun_id'], vp_info['original_name'])
    makedirs(path.dirname(output_hdf5), exist_ok=True)
    print('==================')
    print('{:,d}/{:,d} Processing: idx{:d}, {:s}'.format(ei + 1, n_exp, exp_idx, vp_info['run_id']))
    if not path.isfile(input_bam):
        print('[w] Could not find the bam file. Ignoring this run: {:s}'.format(input_bam))
        continue

    # get chr info
    chr_lst = get_chr_info(genome=vp_info['genome'], property='chr_name')
    chr_size = get_chr_info(genome=vp_info['genome'], property='chr_size')
    n_chr = len(chr_lst)
    chr2nid = dict(zip(chr_lst, np.arange(n_chr) + 1))

    # load RE positions
    re1_pos = get_re_info(re_name=vp_info['first_cutter'], property='pos', genome=vp_info['genome'])
    if isinstance(vp_info['second_cutter'], str):
        re2_pos = get_re_info(re_name=vp_info['second_cutter'], property='pos', genome=vp_info['genome'])
    else:
        re2_pos = [np.empty(0)] * n_chr
    re_pos = [np.empty(0) for _ in range(n_chr)]
    for chr_idx in np.arange(n_chr):
        re_pos[chr_idx] = np.vstack([
            np.vstack([re1_pos[chr_idx], np.zeros(len(re1_pos[chr_idx])) + 1]).T,
            np.vstack([re2_pos[chr_idx], np.zeros(len(re2_pos[chr_idx])) + 2]).T,
            np.vstack([0, 1]).T,
            np.vstack([chr_size[chr_idx], 1]).T,
        ])
        re_pos[chr_idx] = re_pos[chr_idx][np.argsort(re_pos[chr_idx][:, 0]), :]

        # keep only unique res-enz
        unq_idx = np.unique(re_pos[chr_idx][:, 0], return_index=True)[1]
        re_pos[chr_idx] = re_pos[chr_idx][unq_idx, :].astype(np.int)
        assert np.all(np.diff(re_pos[chr_idx][:, 0]) > 0)
        del unq_idx
    del re1_pos, re2_pos

    # load the bam file, part by part
    print('\tLoading {:s}'.format(input_bam))
    n_loaded = 0
    n_ignore = 0
    name2idx = {}
    bam_pd = pd.DataFrame(columns=['read_nid', 'map_chr', 'map_start', 'map_end', 'map_strand', 'mq', 'seq_order'], dtype=np.int)
    with pysam.AlignmentFile(input_bam, 'rb') as bam_fid:
        EOF = False
        while not EOF:
            EOF = True
            frags = []
            for frag in bam_fid:
                if (frag.reference_name not in chr_lst) or (frag.mapping_quality == 0):
                    n_ignore += 1
                    continue
                frags.append(deepcopy(frag))
                n_loaded += 1
                if n_loaded % 100000 == 0:
                    EOF = False
                    break

            # assign read identifiers
            for frag in frags:
                if frag.qname not in name2idx:
                    name2idx[frag.qname] = len(name2idx) + 1
            print('\t{:12,d} frags and {:12,d} reads are loaded, '.format(n_loaded, len(name2idx)) +
                  '{:5,d} are ignored.'.format(n_ignore))

            # make/append to dataframe
            part_pd = pd.DataFrame()
            part_pd['read_nid'] = [name2idx[frag.qname] for frag in frags]
            part_pd['map_chr'] = [chr2nid[frag.reference_name] for frag in frags]
            part_pd['map_start'] = [frag.reference_start for frag in frags]
            part_pd['map_end'] = [frag.reference_end for frag in frags]
            part_pd['map_strand'] = [1 - (frag.is_reverse * 2) for frag in frags]
            part_pd['mq'] = [frag.mapping_quality for frag in frags]
            part_pd['seq_order'] = [frag.is_read1 + frag.is_read2 * 2 for frag in frags]
            bam_pd = bam_pd.append(part_pd, ignore_index=True)
            del frags, part_pd
    del name2idx
    if bam_pd.shape[0] == 0:
        print('[w] No fragments are mapped. Ignoring this run'.format(bam_pd.shape[0]))
        continue

    # filter on MQ
    is_mapped = bam_pd['mq'] >= args.min_mq
    print('\t{:,d} frags ({:0.1f}%) will be discarded due to low MQ.'.format(np.sum(~ is_mapped),
                                                                             np.sum(~ is_mapped) * 1e2 / bam_pd.shape[0]))
    bam_pd = bam_pd.loc[is_mapped].reset_index(drop=True)
    del is_mapped

    # sort fragments, by chr, largest map_size
    # TODO: can be better, sort/merge can happen during batch reading from bam file
    #  (care for "edge" fragments at the end of each batch)
    print('\tSorting {:,d} mapped fragments ...'.format(bam_pd.shape[0]))
    srt_idx = np.lexsort([bam_pd['map_start'], bam_pd['map_chr'], bam_pd['read_nid']])
    bam_pd = bam_pd.loc[srt_idx].reset_index(drop=True)
    del srt_idx

    # checks for overlapping fragments (e.g. paired-end data)
    n_frg = bam_pd.shape[0]
    print('\tChecking {:,d} fragment overlap ...'.format(n_frg))
    bam_pd['map_#merge'] = 0
    frg_np = bam_pd[['read_nid', 'map_chr', 'map_start', 'map_end', 'map_#merge']].values
    is_val = np.full(n_frg, fill_value=True)
    fi = 0
    while fi < n_frg - 1:
        if fi % 100000 == 0:
            print('\t{:12,d} fragments are checked for overlap, to be merged.'.format(fi))
        if (frg_np[fi, 0] != frg_np[fi + 1, 0]) or (frg_np[fi, 1] != frg_np[fi + 1, 1]):
            fi += 1
            continue

        # check overlap (ignoring strand)
        fi_be = fi
        while overlap(frg_np[fi_be, 2:4], frg_np[fi + 1:fi + 2, 2:4])[0]:
            fi += 1
            if fi == n_frg - 1:
                break
        if fi_be != fi:
            frg_np[fi_be, 2] = np.min(frg_np[fi_be:fi + 1, 2])
            frg_np[fi_be, 3] = np.max(frg_np[fi_be:fi + 1, 3])
            frg_np[fi_be, 4] = fi - fi_be
            is_val[fi_be + 1:fi + 1] = False
        fi += 1
    print('\t{:,d} fragments are merged.'.format(np.sum(~is_val)))
    bam_pd[['map_start', 'map_end', 'map_#merge']] = frg_np[:, 2:5]
    bam_pd = bam_pd.loc[is_val].reset_index(drop=True)
    del frg_np, is_val, n_frg

    # extending to closest restriction enzyme recognition site
    # TODO: this can be done much more efficiently: searchsort for all fragments at once
    n_rf = bam_pd.shape[0]
    print('\tExtending {:,d} frag-ends to closest restriction enzyme recognition sites'.format(n_rf))
    map_np = bam_pd[['map_chr', 'map_start', 'map_end', 'map_strand']].values
    rf_np = np.zeros([n_rf, 4], dtype=np.int64)
    for fi in range(n_rf):
        if fi % 100000 == 0:
            print('\t{:12,d} frag-ends are checked for extension.'.format(fi))
        chr_idx = map_np[fi, 0] - 1
        if map_np[fi, 3] == 1:
            rf_idx = np.searchsorted(re_pos[chr_idx][:, 0], map_np[fi, 1], side='right') - 1
            if re_pos[chr_idx][rf_idx + 1, 0] - map_np[fi, 1] < gap_size:
                rf_idx += 1
        else:
            rf_idx = np.searchsorted(re_pos[chr_idx][:, 0], map_np[fi, 2], side='right') - 1
            if map_np[fi, 2] - re_pos[chr_idx][rf_idx, 0] < gap_size:
                rf_idx -= 1
        rf_np[fi, 0] = re_pos[chr_idx][rf_idx, 0]
        rf_np[fi, 1] = re_pos[chr_idx][rf_idx, 1]
        rf_np[fi, 2] = re_pos[chr_idx][rf_idx + 1, 0]
        rf_np[fi, 3] = re_pos[chr_idx][rf_idx + 1, 1]
    bam_pd['rf_start'] = rf_np[:, 0]
    bam_pd['rf_end'] = rf_np[:, 2]
    bam_pd['re1_type'] = rf_np[:, 1]
    bam_pd['re2_type'] = rf_np[:, 3]
    del map_np, rf_np, re_pos

    # sort reads
    print('\tSorting {:,d} mapped fragments by read ID ...'.format(bam_pd.shape[0]))
    srt_idx = np.lexsort(bam_pd[['mq', 'map_strand', 'rf_start', 'map_chr', 'read_nid']].values.T)
    bam_pd = bam_pd.loc[srt_idx].reset_index(drop=True)
    del srt_idx

    # merging fragments covering the same restriction fragment (ignoring their strands)
    print('\tMerging closely mapped fragments, scanning {:,d} fragments:'.format(bam_pd.shape[0]))
    n_multi = 0
    ext_np = bam_pd[['read_nid', 'map_chr', 'rf_start', 'rf_end', 'mq', 'map_#merge']].values
    n_rf = ext_np.shape[0]
    is_val = np.full(n_rf, fill_value=True)
    for fi in range(n_rf - 1):
        if fi % 100000 == 0:
            print('\t{:12,d} reads are checked for multi-way mappings.'.format(fi))
        if ext_np[fi, 0] == ext_np[fi + 1, 0]:  # bam_pd.loc[fi:fi+1]
            if (ext_np[fi, 1] == ext_np[fi + 1, 1]) and (ext_np[fi, 2] == ext_np[fi + 1, 2]):
                # assert ext_np[fi, 3] == ext_np[fi + 1, 3]
                ext_np[fi + 1, 4] = np.max(ext_np[fi:fi + 2, 4])
                ext_np[fi + 1, 5] += 1
                is_val[fi] = False
            else:
                n_multi += 1
    if np.any(~is_val):
        bam_pd['mq'] = ext_np[:, 4]
        bam_pd['map_#merge'] = ext_np[:, 5]
        bam_pd = bam_pd.loc[is_val].reset_index(drop=True)
    print('\t\t#multiway={:,d}\n\t\t#mw-merged={:,d}'.format(n_multi, np.sum(~is_val)))
    del ext_np, is_val

    # count number of reads per restriction fragments captures
    print('\tCounting number of reads per restriction fragment using {:,d} reads ...'.format(bam_pd.shape[0]))
    cmb_pd = pd.DataFrame()
    for side in ['start', 'end']:
        if side == 'start':
            is_sel = bam_pd['map_strand'] == 1
        else:
            is_sel = bam_pd['map_strand'] == -1
        if not np.any(is_sel):
            continue
        strnd_pd = bam_pd.loc[is_sel].copy()
        strnd_pd = strnd_pd.sort_values(by='mq', ascending=False).reset_index(drop=True)
        fe_uid, fe_idx, fe_frq = np.unique(strnd_pd[['map_chr', 'rf_' + side]],
                                           axis=0, return_index=True, return_counts=True)
        assert np.array_equal(strnd_pd.loc[fe_idx, ['map_chr', 'rf_' + side]], fe_uid)
        strnd_pd = strnd_pd.loc[fe_idx]
        strnd_pd['#read'] = fe_frq
        cmb_pd = cmb_pd.append(strnd_pd, ignore_index=True)
        del fe_uid, fe_idx, fe_frq, strnd_pd
    bam_pd = cmb_pd.copy()
    bam_pd.drop(columns=['read_nid', 'map_#merge', 'seq_order'], inplace=True)
    del cmb_pd

    # final adjustments
    bam_pd.rename({'map_chr': 'chr'}, axis=1, inplace=True)
    is_fwrd = bam_pd['map_strand'] == 1
    bam_pd['pos'] = -1
    bam_pd.loc[ is_fwrd, 'pos'] = bam_pd.loc[ is_fwrd, 'rf_start']
    bam_pd.loc[~is_fwrd, 'pos'] = bam_pd.loc[~is_fwrd, 'rf_end']
    bam_pd = bam_pd[['chr', 'pos', '#read', 'mq', 'map_start', 'map_end', 'map_strand',
                     'rf_start', 'rf_end', 're1_type', 're2_type']]
    bam_pd = bam_pd.sort_values(by=['chr', 'rf_start', 'pos']).reset_index(drop=True)

    # saving results
    print('\tSaving {:,d} res-frgs and {:,d} reads in: {:s}'.format(bam_pd.shape[0], np.sum(bam_pd['#read']), output_hdf5))
    with h5py.File(output_hdf5, 'w', libver='latest') as h5_fid:
        h5_fid.create_dataset('frg_np', data=bam_pd.values, compression='gzip', compression_opts=5)
        h5_fid.create_dataset('frg_np_header_lst', data=np.array(bam_pd.columns, dtype=h5py.special_dtype(vlen=str)))
        h5_fid.create_dataset('chr_lst', data=np.array(chr_lst, dtype=h5py.special_dtype(vlen=str)))

print('All runs are mapped successfully.')



