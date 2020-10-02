#! /usr/bin/env python3

import argparse
import sys
from os import path, makedirs
import subprocess
import json

import pandas as pd

# initialization
with open('./configs.json', 'r') as cnf_fid:
    configs = json.load(cnf_fid)

# creating argument parser
parser = argparse.ArgumentParser(description="Mapper script using given experiment indices")
parser.add_argument('--vpi_file', default='./vp_info.tsv', type=str, help='VP information file')
parser.add_argument('--input_dir', default='./fastqs/trimmed', type=str, help='input dir')
parser.add_argument('--output_dir', default='./bams', type=str, help='output dir')
parser.add_argument('--expr_index', default='1', type=str, help='Limits mapping to a specific experiment indices')
parser.add_argument('--n_thread', default=1, type=int, help='Number of threads used for mapping')
args = parser.parse_args(sys.argv[1:])

# load vp info file
vp_pd = pd.read_csv(args.vpi_file, sep='\t')

# limit to specific experiment indices
if args.expr_index:
    print('Mapping is limited to following experiment indices: {:s}'.format(args.expr_index))
    args.expr_index = [int(x) for x in args.expr_index.split(',')]
else:
    args.expr_index = range(vp_pd.shape[0])
n_exp = len(args.expr_index)

# loop over experiments
for ei, exp_idx in enumerate(args.expr_index):
    if vp_pd.at[exp_idx, 'assay_type'] != '4C':
        print('[w] Expriment ignored, not 4C: idx{:d}, {:s}'.format(exp_idx, vp_pd.at[exp_idx, 'original_name']))
        continue

    # prepare file names
    input_fastq = '{:s}/{:s}/{:s}.fastq.gz'.format(args.input_dir, vp_pd.at[exp_idx, 'seqrun_id'], vp_pd.at[exp_idx, 'original_name'])
    output_fastq = '{:s}/{:s}/{:s}.bam'.format(args.output_dir, vp_pd.at[exp_idx, 'seqrun_id'], vp_pd.at[exp_idx, 'original_name'])
    makedirs(path.dirname(output_fastq), exist_ok=True)
    print('==================')
    print('{:04d}/{:d} Mapping to: idx{:d}, {:s}'.format(ei + 1, n_exp, exp_idx, output_fastq))

    # run the command
    index_path = path.expanduser(configs['bwa_index'][vp_pd.at[exp_idx, 'genome']])
    cmd = ['bwa', 'mem',
           '-t', str(args.n_thread),
           '-k', '12',
           '-A', '2',
           '-B', '3',
           '{:s}'.format(index_path),
           input_fastq]
    print('Running: {:s}'.format(' '.join(cmd)))
    with open(output_fastq, 'wb') as bam_fid:
        mapper = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        sorter = subprocess.Popen(('samtools', 'sort', '-n'), stdin=mapper.stdout, stdout=subprocess.PIPE)
        filter = subprocess.Popen(('samtools', 'view', '-q', '1', '-b', '-'), stdin=sorter.stdout, stdout=bam_fid)
        mapper.stdout.close()  # https://stackoverflow.com/questions/7391689/closing-stdout-of-piped-python-subprocess
        sorter.stdout.close()
        mapper.wait()
        sorter.wait()
        filter.wait()
    if not (mapper.returncode == sorter.returncode == filter.returncode == 0):
        raise Exception('[e] Mapping failed with a non-zero status ({:d})'.format(mapper.returncode))

print('All runs are mapped successfully.')



