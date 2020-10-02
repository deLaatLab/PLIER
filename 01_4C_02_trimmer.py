#! /usr/bin/env python3

import argparse
import sys
from os import path, makedirs
import numpy as np
import pandas as pd
import pysam
import gzip

from utilities import get_re_info

# initialization
np.set_printoptions(linewidth=250, threshold=5000)  # , suppress=True, formatter={'float_kind':'{:0.5f}'.format}
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 25)

# creating argument parser
parser = argparse.ArgumentParser(description="Trimmer script using given primer sequences")
parser.add_argument('--vpi_file', default='./vp_info.tsv', type=str, help='VP information file')
parser.add_argument('--input_dir', default='./fastqs/demuxed', type=str, help='input dir')
parser.add_argument('--output_dir', default='./fastqs/trimmed', type=str, help='output dir')
parser.add_argument('--expr_index', default='1', type=str, help='Limits trimming to a specific experiment indices')
args = parser.parse_args(sys.argv[1:])

# load vp info file
vp_pd = pd.read_csv(args.vpi_file, sep='\t')
assert vp_pd.shape[0] == len(np.unique(vp_pd['run_id']))
assert vp_pd.shape[0] == len(np.unique(vp_pd['original_name']))

# limit to specific experiment indices
if args.expr_index:
    print('Trimming is limited to following experiment indices: {:s}'.format(args.expr_index))
    args.expr_index = [int(x) for x in args.expr_index.split(',')]
else:
    args.expr_index = range(vp_pd.shape[0])
n_exp = len(args.expr_index)

# loop over experiments
for ei, exp_idx in enumerate(args.expr_index):
    if exp_idx >= vp_pd.shape[0]:
        print('[w] Requested run index {:d} does not exists in {:s} ...'.format(exp_idx, args.vpi_file))
        continue
    print('{:3d}/{:3d}: Trimming: idx{:d}, {:s}'.format(ei + 1, n_exp, exp_idx, vp_pd.at[exp_idx, 'original_name']))

    # prepare file names
    input_fastq = '{:s}/{:s}/{:s}.fastq.gz'.format(args.input_dir, vp_pd.at[exp_idx, 'seqrun_id'], vp_pd.at[exp_idx, 'original_name'])
    output_fastq = '{:s}/{:s}/{:s}.fastq.gz'.format(args.output_dir, vp_pd.at[exp_idx, 'seqrun_id'], vp_pd.at[exp_idx, 'original_name'])
    print('\tReading from: {:s}'.format(input_fastq))
    print('\t Writing  to: {:s}'.format(output_fastq))
    if not path.isfile(input_fastq):
        print('\t[e] Error: Source FASTQ is not found: {:s}'.format(input_fastq))
        continue
    makedirs(path.dirname(output_fastq), exist_ok=True)

    # identify first restriction enzyme sequence
    re1_seq = get_re_info(genome=vp_pd.at[exp_idx, 'genome'], re_name=vp_pd.at[exp_idx, 'first_cutter'], property='seq')
    re1_nnt = len(re1_seq)
    assert re1_seq == vp_pd.at[exp_idx, 'vpfe_seq'][- re1_nnt:], vp_pd.loc[[exp_idx]]
    trm_size = len(vp_pd.at[exp_idx, 'vpfe_seq']) - re1_nnt

    # loop over reads in the current fastq file
    with pysam.FastxFile(input_fastq) as inp_fid, gzip.open(output_fastq, 'wt') as out_fid:
        for rd_idx, read in enumerate(inp_fid):
            if np.mod(rd_idx, 50000) == 0:
                print('\t{:0,.0f} reads are processed'.format(rd_idx))

            # output the trimmed read
            read.name = read.name + '_ntr{:d}'.format(trm_size)
            read.sequence = read.sequence[trm_size:]
            read.quality = read.quality[trm_size:]
            out_fid.write(str(read) + '\n')
    print('\tfinished trimming {:0,.0f} reads'.format(rd_idx + 1))
print('All runs are trimmed.')



