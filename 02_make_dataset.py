#! /usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import pandas as pd
import pysam
from copy import deepcopy
import h5py

from utilities import get_chr_info, get_re_info, overlap, get_read

# initialization
pd.options.display.max_columns = 35
pd.options.display.width = 250
np.set_printoptions(formatter={'int_kind': '{:,d}'.format})  # 'float_kind': '{:0.5f}'.format,
gap_size = 10

# creating argument parser
arg_parser = argparse.ArgumentParser(description="Collect mapped fragments into a dataset for the given experiment indices")
arg_parser.add_argument('--expr_indices', default='-1', type=str, help='Limits processing to specific experiment indices (sep=",")')
arg_parser.add_argument('--viewpoints', default='./viewpoints/vp_info.tsv', type=str, help='Path to the VP information file')
arg_parser.add_argument('--input_dir', default='./bams/demuxed/', help='input dir')
arg_parser.add_argument('--output_dir', default='./datasets/', help='output dir')
arg_parser.add_argument('--min_mq', default=1, type=int, help='Minimum mapping quality (MQ) to consider reads as mapped')
inp_args = arg_parser.parse_args()

# load vp info file
print('Loading VP info file from: {:s}'.format(inp_args.viewpoints))
vp_infos = pd.read_csv(inp_args.viewpoints, sep='\t')
assert vp_infos.shape[0] == len(np.unique(vp_infos['expr_id'])), 'Failed: Experiment IDs are not unique!'

# limit to specific experiment indices
if inp_args.expr_indices == '-1':
    print('Processing every {:d} experiments found in: {:s}'.format(len(vp_infos), inp_args.viewpoints))
    inp_args.expr_indices = range(vp_infos.shape[0])
else:
    inp_args.expr_indices = [int(float(x)) for x in inp_args.expr_indices.split(',')]
    print('Processing is limited to the following experiment indices: [{:s}]'.format(', '.join(['{:d}'.format(i) for i in inp_args.expr_indices])))
n_exp = len(inp_args.expr_indices)

# loop over experiments
for ei, exp_idx in enumerate(inp_args.expr_indices):
    if exp_idx >= len(vp_infos):
        break

    # prepare file names
    vp_info = vp_infos.loc[exp_idx]
    input_bam = os.path.join(inp_args.input_dir, vp_info['expr_id'] + '.bam')
    output_hdf5 = os.path.join(inp_args.output_dir, vp_info['expr_id'] + '.hdf5')
    os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
    print('==================')
    print('{:,d}/{:,d} Processing: idx{:d}, {:s}'.format(ei + 1, n_exp, exp_idx, vp_info['expr_id']))
    assert os.path.isfile(input_bam), 'Could not find the bam file: {:s}'.format(input_bam)

    # get chr info
    chr_lst = get_chr_info(genome=vp_info['genome'], property='chr_name')
    chr_size = get_chr_info(genome=vp_info['genome'], property='chr_size')
    n_chr = len(chr_lst)
    chr2nid = dict(zip(chr_lst, np.arange(n_chr) + 1))

    # load RE positions
    re1_pos = get_re_info(re_name=vp_info['res_enzyme'], property='pos', genome=vp_info['genome'])
    if ('second_cutter' in vp_info) and isinstance(vp_info['second_cutter'], str):
        re2_pos = get_re_info(re_name=vp_info['second_cutter'], property='pos', genome=vp_info['genome'])
    else:
        re2_pos = [np.empty(0, dtype=int)] * n_chr
    re_pos = [[] for _ in range(n_chr)]
    for chr_idx in np.arange(n_chr):
        re_pos[chr_idx] = np.vstack([
            np.vstack([re1_pos[chr_idx], np.repeat(1, len(re1_pos[chr_idx]))]).T,
            np.vstack([re2_pos[chr_idx], np.repeat(2, len(re2_pos[chr_idx]))]).T,
            np.vstack([0, 1]).T,
            np.vstack([chr_size[chr_idx], 1]).T,
        ])
        re_pos[chr_idx] = re_pos[chr_idx][np.argsort(re_pos[chr_idx][:, 0]), :]

        # keep only uniquely-positioned res-enz
        unq_idx = np.unique(re_pos[chr_idx][:, 0], return_index=True)[1]
        re_pos[chr_idx] = re_pos[chr_idx][unq_idx, :]
        assert np.all(np.diff(re_pos[chr_idx][:, 0]) > 0)
        del unq_idx
    del re1_pos, re2_pos

    # load the bam file, part by part
    print('\tIterating the alignment file: {:s}'.format(input_bam))
    n_read = 0
    nfrg_loaded = 0
    nfrg_ignore = 0
    qname2nid = {}
    bam_pd = pd.DataFrame(columns=['read_nid', 'map_chr', 'map_start', 'map_end', 'map_strand', 'mq', 'seq_order'], dtype=np.int)
    with pysam.AlignmentFile(input_bam, 'rb') as bam_fid:
        EOF = False
        while not EOF:
            EOF = True
            frags = []
            for rd_idx, read in enumerate(get_read(bam_fid)):
                assert read[0].qname not in qname2nid, 'Alignment file is not queryname sorted!'
                qname2nid[read[0].qname] = len(qname2nid) + 1
                n_read += 1
                for frag in read:
                    if (frag.reference_name not in chr_lst) or (frag.mapping_quality == 0):
                        nfrg_ignore += 1
                        continue
                    frags.append(deepcopy(frag))
                    nfrg_loaded += 1
                if len(frags) > 1e6:
                    print('\tIterated reads={:12,d}; frags={:12,d}; '.format(n_read, nfrg_loaded) +
                          'ignored={:5,d} ...'.format(nfrg_ignore))
                    EOF = False
                    break

            # make/append to dataframe
            part_pd = pd.DataFrame()
            part_pd['read_nid'] = [qname2nid[frag.qname] for frag in frags]
            part_pd['map_chr'] = [chr2nid[frag.reference_name] for frag in frags]
            part_pd['map_start'] = [frag.reference_start for frag in frags]
            part_pd['map_end'] = [frag.reference_end for frag in frags]
            part_pd['map_strand'] = [1 - (frag.is_reverse * 2) for frag in frags]
            part_pd['mq'] = [frag.mapping_quality for frag in frags]
            part_pd['seq_order'] = [frag.is_read1 + frag.is_read2 * 2 for frag in frags]
            assert np.array_equal(bam_pd.columns, part_pd.columns)
            bam_pd = bam_pd.append(part_pd, ignore_index=True)
            del frags, part_pd
    del qname2nid
    # nid2qname = {qname2nid[k]: k for k in qname2nid}
    print('\tFinished with reads={:12,d}; frags={:12,d}; '.format(n_read, nfrg_loaded) +
          'ignored={:5,d} ...'.format(nfrg_ignore))
    if bam_pd.shape[0] == 0:
        print('[w] No fragments are mapped. Ignoring this run'.format(bam_pd.shape[0]))
        continue

    # filter on MQ
    if inp_args.min_mq > 1:
        is_mapped = bam_pd['mq'] >= inp_args.min_mq
        print('\t{:,d} frags ({:0.1f}%) will be discarded due to low MQ.'.format(np.sum(~ is_mapped),
                                                                                 np.sum(~ is_mapped) * 1e2 / bam_pd.shape[0]))
        bam_pd = bam_pd.loc[is_mapped].reset_index(drop=True)
        del is_mapped

    # correct mapping orientations
    assert np.array_equal(np.unique(bam_pd['seq_order']), [1, 2])
    # Orientation is checked in IGV, no need to correct the R2 reads
    # bam_pd.loc[bam_pd['seq_order'] == 2, 'map_strand'] *= -1

    # sort fragments, by chr, largest map_size
    print('\tSorting {:,d} mapped fragments ...'.format(bam_pd.shape[0]))
    # lexsort seems to be faster than pandas: https://stackoverflow.com/questions/55493274/performance-of-sorting-structured-arrays-numpy
    srt_idx = np.lexsort([-bam_pd['mq'], bam_pd['map_start'], bam_pd['map_chr'], bam_pd['read_nid']])
    bam_pd = bam_pd.loc[srt_idx].reset_index(drop=True)
    del srt_idx

    # checks for overlapping fragments (e.g. paired-end data)
    n_frg = bam_pd.shape[0]
    print('\tChecking {:,d} fragment overlap ...'.format(n_frg))
    bam_pd['map_#merge'] = 0
    frg_np = bam_pd[['read_nid', 'map_chr', 'map_start', 'map_end', 'mq', 'map_#merge']].values
    is_val = np.full(n_frg, fill_value=True)
    fi = 0
    # TODO: better merging strategy is to keep top MQs, but that requires pairwise comparison of all fragments => expensive
    while fi < n_frg - 1:
        if fi % 1e6 == 0:
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
            # bam_pd.loc[fi_be:fi]
            # frg_np[fi_be:fi + 1, :]
            frg_np[fi_be, 2] = np.min(frg_np[fi_be:fi + 1, 2])
            frg_np[fi_be, 3] = np.max(frg_np[fi_be:fi + 1, 3])
            frg_np[fi_be, 4] = np.max(frg_np[fi_be:fi + 1, 4])
            frg_np[fi_be, 5] = fi - fi_be
            is_val[fi_be + 1:fi + 1] = False
        fi += 1
    print('\t{:,d} overlapping fragments are merged.'.format(np.sum(~is_val)))
    bam_pd[['map_start', 'map_end', 'mq', 'map_#merge']] = frg_np[:, 2:6]
    bam_pd = bam_pd.loc[is_val].reset_index(drop=True)
    del frg_np, is_val, n_frg

    # extending to closest restriction enzyme recognition site
    # TODO: this can be done much more efficiently: searchsort for all fragments at once
    n_rf = bam_pd.shape[0]
    print('\tExtending {:,d} frag-ends to closest restriction enzyme recognition sites'.format(n_rf))
    map_np = bam_pd[['map_chr', 'map_start', 'map_end', 'map_strand']].values
    rf_np = np.zeros([n_rf, 4], dtype=np.int64)
    for fi in range(n_rf):
        if fi % 1e6 == 0:
            print('\t{:12,d} frag-ends are checked for extension.'.format(fi))
        chr_idx = map_np[fi, 0] - 1
        if map_np[fi, 3] == 1:
            rf_idx = np.searchsorted(re_pos[chr_idx][:, 0], map_np[fi, 1], side='right') - 1
            # re_pos[chr_idx][rf_idx, 0], map_np[fi, 1], re_pos[chr_idx][rf_idx + 1, 0]
            if re_pos[chr_idx][rf_idx + 1, 0] - map_np[fi, 1] < gap_size:
                rf_idx += 1
        else:
            rf_idx = np.searchsorted(re_pos[chr_idx][:, 0], map_np[fi, 2], side='right') - 1
            # re_pos[chr_idx][rf_idx, 0], map_np[fi, 2], re_pos[chr_idx][rf_idx + 1, 0]
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
    print('\tMerging fragments covering the same restriction fragment, scanning {:,d} fragments:'.format(bam_pd.shape[0]))
    ext_np = bam_pd[['read_nid', 'map_chr', 'rf_start', 'rf_end', 'mq', 'map_#merge']].values
    n_rf = ext_np.shape[0]
    is_val = np.full(n_rf, fill_value=True)
    for fi in range(n_rf - 1):
        if fi % 1e6 == 0:
            print('\t{:12,d} reads are checked for multi-way mappings.'.format(fi))
        if ext_np[fi, 0] == ext_np[fi + 1, 0]:  # bam_pd.loc[fi:fi+1]
            if (ext_np[fi, 1] == ext_np[fi + 1, 1]) and (ext_np[fi, 2] == ext_np[fi + 1, 2]):
                # assert ext_np[fi, 3] == ext_np[fi + 1, 3]
                ext_np[fi + 1, 4] = np.max(ext_np[fi:fi + 2, 4])
                ext_np[fi + 1, 5] += 1
                is_val[fi] = False
    if np.any(~is_val):
        bam_pd['mq'] = ext_np[:, 4]
        bam_pd['map_#merge'] = ext_np[:, 5]
        bam_pd = bam_pd.loc[is_val].reset_index(drop=True)
    print('\t\t#fragments removed={:,d}'.format(np.sum(~is_val)))
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
    del is_fwrd

    # saving results
    print('\tSaving {:,d} res-frgs and {:,d} reads in: {:s}'.format(bam_pd.shape[0], np.sum(bam_pd['#read']), output_hdf5))
    with h5py.File(output_hdf5, 'w', libver='latest') as h5_fid:
        h5_fid.create_dataset('frg_np', data=bam_pd.values, compression='gzip', compression_opts=5)
        h5_fid.create_dataset('frg_np_header_lst', data=np.array(bam_pd.columns, dtype=h5py.special_dtype(vlen=str)))
        h5_fid.create_dataset('chr_lst', data=np.array(chr_lst, dtype=h5py.special_dtype(vlen=str)))

print('All runs are mapped successfully.')



