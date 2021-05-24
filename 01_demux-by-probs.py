#! /usr/bin/env python3


import argparse
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
pd.options.display.expand_frame_repr = False

# creating argument parser
arg_parser = argparse.ArgumentParser(description="Demultiplexer script using given prob info")
arg_parser.add_argument('--sample_id', required=True)
arg_parser.add_argument('--tissue_id', default=None)
arg_parser.add_argument('--genome', required=True)
arg_parser.add_argument('--res_enzyme', default='NlaIII')
arg_parser.add_argument('--probs', default='./probs/probs.tsv')
arg_parser.add_argument('--input_bam', default=None)
arg_parser.add_argument('--output_dir', default='./bams/demuxed/')
arg_parser.add_argument('--viewpoint_dir', default='./viewpoints/')
arg_parser.add_argument('--multiprob', action='store_true')
inp_args = arg_parser.parse_args()
if inp_args.tissue_id is None:
    inp_args.tissue_id = inp_args.sample_id
if inp_args.input_bam is None:
    inp_args.input_bam = './bams/combined/{:s}.bam'.format(inp_args.sample_id)
if inp_args.multiprob:
    print('[i] Multi-prob is activated: A single fragment can be assigned to multiple experiments!')
if not path.isfile(inp_args.input_bam):
    print('[e] Source BAM file is not found: {:s}\nHave you mapped your FASTQ files?'.format(inp_args.input_bam))
    exit(1)

# get chromosome information
print('Collecting chromosome information from {:s} genome.'.format(inp_args.genome))
chr_lst = get_chr_info(inp_args.genome, property='chr_name')
chr2nid = dict(zip(chr_lst, np.arange(len(chr_lst)) + 1))

# load prob info file
prb_pd = pd.read_csv(inp_args.probs, sep='\t')
prb_pd['chr_num'] = prb_pd['chr'].map(chr2nid)
assert not prb_pd['chr_num'].isna().any(), 'Unknown chromosome found in the prob file: {:s}'.format(inp_args.probs)
prb_pd = prb_pd.sort_values(by=['chr_num', 'pos_begin']).reset_index(drop=True)
print('{:,d} probs coordinates are loaded from: {:s}'.format(prb_pd.shape[0], inp_args.probs))

# loop over prob groups and merge
prb_grp = prb_pd.groupby(by=['target_locus'], sort=False)
vpi_pd = pd.DataFrame()
print('{:d} viewpoints are formed.'.format(prb_grp.ngroups))
for set_idx, (grp_id, prb_set) in enumerate(prb_grp):
    print('{:02d}/{:02d} '.format(set_idx + 1, prb_grp.ngroups) +
          '{:5s} {:11d}  {:11d}; locus:{:>12s}, '.format(*prb_set.iloc[0][['chr', 'pos_begin', 'pos_end', 'target_locus']]))

    # make prob info
    locus_name = np.unique(prb_set['target_locus'])
    vp_chr = np.unique(prb_set['chr'])
    assert len(locus_name) == 1
    assert len(vp_chr) == 1
    locus_name = locus_name[0]
    vp_chr = vp_chr[0]
    vp_be = min(prb_set['pos_begin'])
    vp_en = max(prb_set['pos_end'])

    prob_info = dict()
    prob_info['expr_id'] = '{:s}_{:s}'.format(inp_args.sample_id, locus_name)
    prob_info['vp_chr'] = chr2nid[vp_chr]
    prob_info['vp_pos'] = int((vp_en + vp_be) / 2)
    prob_info['vp_be'] = vp_be
    prob_info['vp_en'] = vp_en
    prob_info['vp_gene'] = locus_name
    prob_info['sample_id'] = inp_args.sample_id
    prob_info['tissue_id'] = inp_args.tissue_id
    prob_info['genome'] = inp_args.genome
    prob_info['res_enzyme'] = inp_args.res_enzyme

    vpi_pd = vpi_pd.append([prob_info], ignore_index=True)
n_expr = vpi_pd.shape[0]
assert n_expr == len(vpi_pd['expr_id'].unique())
del prob_info

# output prob info
vpi_fpath = path.join(inp_args.viewpoint_dir, '{:s}.viewpoints.tsv'.format(inp_args.sample_id))
makedirs(path.dirname(vpi_fpath), exist_ok=True)
print('Exporting VP info file to: {:s}'.format(vpi_fpath))
vpi_pd.to_csv(vpi_fpath, sep='\t', index=False, compression=None)

# make file handles
print('Making BAM file handles from: {:s}'.format(inp_args.input_bam))
makedirs(path.dirname(inp_args.output_dir), exist_ok=True)
with pysam.AlignmentFile(inp_args.input_bam, 'rb') as source_fid:
    out_fids = []
    for ei in range(n_expr):
        out_fpath = path.join(inp_args.output_dir, vpi_pd.at[ei, 'expr_id'] + '.bam')
        out_fids.append(pysam.AlignmentFile(out_fpath, 'wb', template=source_fid))
    unmatched_fpath = path.join(inp_args.output_dir, inp_args.sample_id + '_unmatched.bam')
    out_fids.append(pysam.AlignmentFile(unmatched_fpath, 'wb', template=source_fid))

# loop over reads in the bam source file
n_read = 0
n_unmatched = 0
n_multi_capture = 0
vp_crds = vpi_pd[['vp_chr', 'vp_be', 'vp_en']].values
vp_names = vpi_pd['vp_gene'].values
hit_colors = ['{:0.0f},{:0.0f},{:0.0f}'.format(*c[:3] * 255) for c in plt.cm.get_cmap('jet', 50)(np.linspace(0, 1, 50))]
print('Looping over reads in: {:s}'.format(inp_args.input_bam))
with pysam.AlignmentFile(inp_args.input_bam, 'rb') as src_fid:
    # hint: no need to check continuity (uniqueness) of the read_ids, we will do this on the make_dataset script: better use of memory
    for rd_idx, read in enumerate(get_read(src_fid)):
        if rd_idx % 1e6 == 0:
            print('\t{:,d} reads are processed'.format(rd_idx))
        n_read += 1

        # check overlap with probes/VPs
        hit_vps = {}
        hit_overlap_size = np.zeros(n_expr + 1, dtype=int)
        for frg in read:
            frg_crd = [
                chr2nid[frg.reference_name],
                frg.reference_start,
                frg.reference_end
            ]
            is_ol = overlap(frg_crd, vp_crds)
            if any(is_ol):
                vp_idx = np.where(is_ol)[0]
                assert len(vp_idx) == 1, '[e] A single fragment is mapped to multiple viewpoints!'
                vp_idx = vp_idx[0]

                hit_overlap_size[vp_idx] += frg.get_overlap(vp_crds[vp_idx, 1], vp_crds[vp_idx, 2])
                if vp_idx not in hit_vps:  # coloring is based on the first fragment that maps to the VP
                    clr_ratio = float(np.mean(frg_crd[1:]) - vp_crds[vp_idx, 1]) / (vp_crds[vp_idx, 2] - vp_crds[vp_idx, 1])
                    hit_vps[vp_idx] = {
                        'color': hit_colors[int(clr_ratio * 24)],
                        'coord': '{:s}:{:0.0f}'.format(frg.reference_name, frg.reference_start),
                        'id': '{:s}'.format(vp_names[vp_idx])
                    }
        n_hit = len(hit_vps)
        if n_hit == 0:
            n_unmatched += 1
            hit_vps[n_expr] = {
                'color': '0,0,0',
                'coord': 'None',
                'id': 'None'
            }
        if n_hit > 1:
            n_multi_capture += 1

        # store the read
        hit_id = ','.join([hit_vps[key]['id'] for key in hit_vps.keys()])
        max_hit_length = np.max(hit_overlap_size)
        for hit_key in hit_vps.keys():
            if (not inp_args.multiprob) and (hit_overlap_size[hit_key] != max_hit_length):
                continue
            for frg in read:
                out = deepcopy(frg)
                out.query_name = 'vp:{:s};nvp:{:d};qn:{:s}'.format(hit_id, n_hit, out.query_name)
                out.tags += [('YC', hit_vps[hit_key]['color']),
                             ('VP', hit_vps[hit_key]['coord'])]
                out_fids[hit_key].write(out)
print('{:,d} reads are de-multiplexed successfully, and {:,d} are unmatched.'.format(n_read, n_unmatched))
print('{:,d} reads are captured by more than one probes/VPs.'.format(n_multi_capture))

# closing file handles
for fi in range(len(out_fids)):
    out_fids[fi].close()
print('De-multiplexing is successfully finished.')


