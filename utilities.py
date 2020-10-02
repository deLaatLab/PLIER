
import json

import numpy as np
import pandas as pd


def overlap(que_item, ref_lst, include_ref_left=False, include_ref_right=False, offset=0):
    if isinstance(que_item, list):
        que_item = np.array(que_item)
    que_dim = que_item.shape[0]
    [n_ref, ref_dim] = np.shape(ref_lst)
    if (que_dim != ref_dim) or (que_item.ndim != 1):
        raise ValueError('Query or reference are inconsistent')

    crd_ind = 0
    has_ovl = np.ones(n_ref, dtype=bool)
    if que_dim == 4:  # Orientation
        has_ovl = que_item[3] == ref_lst[:, 3]
    if que_dim >= 3:  # Chromosome
        has_ovl = np.logical_and(has_ovl, que_item[0] == ref_lst[:, 0])
        crd_ind = 1
    if include_ref_left:
        lft_ovl = ref_lst[:, crd_ind] <= (que_item[crd_ind + 1] + offset)
    else:
        lft_ovl = ref_lst[:, crd_ind] <  (que_item[crd_ind + 1] + offset)
    if include_ref_right:
        rgh_ovl = ref_lst[:, crd_ind + 1] >= (que_item[crd_ind] - offset)
    else:
        rgh_ovl = ref_lst[:, crd_ind + 1] >  (que_item[crd_ind] - offset)
    has_ovl = np.logical_and(has_ovl, np.logical_and(lft_ovl, rgh_ovl))
    return has_ovl


def get_chr_info(genome, property='chr_name'):
    chr_details = dict({
        'hg19': dict({
            'chr_name': [
                'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
                'chrX', 'chrY', 'chrM'
            ],
            'chr_size': [
                249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747,
                135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566,
                155270560, 59373566, 16571
            ]
        }),
        'mm9': dict({
            'chr_name': [
                'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                'chrX', 'chrY', 'chrM'
            ],
            'chr_size': [
                197195432, 181748087, 159599783, 155630120, 152537259, 149517037, 152524553, 131738871, 124076172,
                129993255, 121843856, 121257530, 120284312, 125194864, 103494974, 98319150, 95272651, 90772031, 61342430,
                166650296, 15902555, 16299
            ]
        }),
        'mm10': dict({
            'chr_name': [
                'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                'chrX', 'chrY', 'chrM'
            ],
            'chr_size': [
                195471971, 182113224, 160039680, 156508116, 151834684, 149736546, 145441459, 129401213, 124595110,
                130694993, 122082543, 120129022, 120421639, 124902244, 104043685, 98207768, 94987271, 90702639, 61431566,
                171031299, 91744698, 16299,
            ]
        })
    })
    return chr_details[genome][property]


def get_re_info(re_name='DpnII', property='seq', genome=None, re_path='./renzs/'):
    from os import path

    re_details = dict({
        'DpnII': dict({'seq': 'GATC'}),
        'MboI': dict({'seq': 'GATC'}),
        'Csp6I': dict({'seq': 'GTAC'}),
        'NlaIII': dict({'seq': 'CATG'}),
        'XbaI': dict({'seq': 'TCTAGA'}),
        'BamHI': dict({'seq': 'GGATCC'}),
        'SacI': dict({'seq': 'GAGCTC'}),
        'PstI': dict({'seq': 'CTGCAG'}),
        'HindIII': dict({'seq': 'AAGCTT'})
    })

    if property == 'pos':
        re_fname = path.join(re_path, '{:s}_{:s}.npz'.format(genome, re_name))
        if not path.isfile(re_fname):
            extract_re_positions(genome=genome, re_name_lst=[re_name], output_fname=re_fname)
        re_data = np.load(re_fname, allow_pickle=True)
        assert np.array_equal(re_data['chr_lst'], get_chr_info(genome=genome, property='chr_name'))
        assert re_data['genome'] == genome
        return re_data['pos_lst']
    else:
        return re_details[re_name][property]


def showprogress(iter, n_iter, n_step=10, output_format='{:1.0f}%, '):
    iter = iter + 1
    if ((iter % (n_iter / float(n_step))) - ((iter - 1) % (n_iter / float(n_step))) < 0) or (n_iter / float(n_step) <= 1):
        if iter != n_iter:
            print(output_format.format(iter * 100 / n_iter)),
        else:
            print(output_format.format(iter * 100 / n_iter))


def accum_array(group_idx, arr, func=None, default_value=None, min_n_group=None, rebuild_index=False):
    """groups a by indices, and then applies func to each group in turn.
    e.g. func example: [func=lambda g: g] or [func=np.sum] or None for speed
    based on https://github.com/ml31415/numpy-groupies
    """

    if rebuild_index:
        group_idx = np.unique(group_idx.copy(), return_inverse=True)[1]
    if not min_n_group:
        min_n_group = np.max(group_idx) + 1

    counts = np.bincount(group_idx, minlength=min_n_group)
    order_group_idx = np.argsort(group_idx, kind='mergesort')

    if isinstance(arr, np.ndarray):
        groups = np.split(arr[order_group_idx], np.cumsum(counts)[:-1], axis=0)
    else:  # If arr is a Pandas DataFrame
        groups = np.split(arr.loc[order_group_idx,:], np.cumsum(counts)[:-1], axis=0)

    if func:
        ret = [default_value] * min_n_group
        for i, grp in enumerate(groups):
            if len(grp) > 0:
                ret[i] = func(grp)
        return ret
    else:
        return groups


def mapEx(list_idx, arr, func=None, default_value=None):
    ret = [default_value] * len(list_idx)
    for i, set_idx in enumerate(list_idx):
        if len(set_idx) > 0:
            ret[i] = func(arr[set_idx])
    return ret


def color_stdout(message, color_name='red'):
    END = '\033[0m'
    clr_dict = {
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'dark_cyan': '\033[36m',
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }

    return clr_dict[color_name] + message + END


def flatten(nested_lst):
    out_lst = []
    for item in nested_lst:
        if isinstance(item, list):
            out_lst.extend(flatten(item))
        else:
            out_lst.append(item)
    return out_lst


def group_by_bin_overlap(reads, bins):
    assert reads.ndim == 1
    assert bins.shape[1] == 2
    bin_w = bins[0, 1] - bins[0, 0]
    n_bin = bins.shape[0]
    n_read = reads.shape[0]
    assert np.unique(bins[:, 1] - bins[:, 0]) == bin_w
    assert np.array_equal(bins[1:, 0] - bin_w / 2, bins[:-1, 0])
    # assert np.max(reads) <= np.max(bins) # this code does not work if bins are smaller than maximum coordinate ###
    # make sure bins are sorted

    # calculate edge overlaps
    edge_lst = np.unique(bins)
    ovl_nid = np.digitize(reads, edge_lst) - 1
    ovl_nid[reads > np.max(edge_lst)] = -1

    # filling groups
    groups = [None] * n_bin
    idx_be = 0
    idx_en = 0
    idx_lst = np.arange(n_read)
    for bi in range(n_bin):
        while edge_lst[idx_be] < bins[bi, 0]:
            idx_be += 1
        while edge_lst[idx_en + 1] < bins[bi, 1]:
            idx_en += 1
        ovl_be = np.searchsorted(ovl_nid, idx_be, side='left')
        ovl_en = np.searchsorted(ovl_nid, idx_en, side='right')
        groups[bi] = idx_lst[ovl_be:ovl_en]

        # test
        # has_ol = np.where((edge_lst >= bins[bi, 0]) & (edge_lst < bins[bi, 1]))[0]
        # assert np.array_equal(np.where(np.isin(ovl_nid, has_ol))[0], groups[bi])

    # test
    # for bi in range(n_bin):
    #     is_sel = (reads >= bins[bi, 0]) & (reads < bins[bi, 1])
    #     assert np.array_equal(np.where(is_sel)[0], groups[bi])

    return groups


def get_marker_lst():
    mrk_lst = ['.', ',', 'o', 'v', '^', '<', '>',
               '1', '2', '3', '4', '8', 's', 'p', 'P',
               '*', 'h', 'H', '+', 'x', 'X', 'D', 'd',
               '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return mrk_lst


def extract_re_positions(genome, re_name_lst, output_fname=None, ref_fasta=None):
    from os import path, makedirs
    import pysam
    import re

    # Initialization
    chr_lst = get_chr_info(genome=genome, property='chr_name')
    chr2idx = dict(zip(chr_lst, np.arange(len(chr_lst))))

    if output_fname is None:
        output_fname = './renzs/{:s}_{:s}.npz'.format(genome, '-'.join(re_name_lst))
    if path.isfile(output_fname):
        print('[w] Restriction enzyme file exists: ' + output_fname)
        return
    if not path.isdir(path.dirname(output_fname)):
        makedirs(path.dirname(output_fname))
    if ref_fasta is None:
        with open('./configs.json', 'r') as cnf_fid:
            ref_fasta = json.load(cnf_fid)['reference_path'][genome]
    print('Searching in the reference genome defined in: ' + ref_fasta)

    # get re sequences
    seq_lst = []
    for re_name in re_name_lst:
        seq_lst.append(get_re_info(genome=genome, re_name=re_name, property='seq'))
    re_regex = '|'.join(seq_lst)

    # Loop over chromosomes
    re_pos_lst = [None] * len(chr_lst)
    re_type_lst = [None] * len(chr_lst)
    chr_observed = [None] * len(chr_lst)
    with pysam.FastxFile(path.expanduser(ref_fasta)) as ref_fid:
        print('Scanning chromosomes for restriction recognition sequences: {:s}'.format(', '.join(seq_lst)))
        for chr_ind, chr in enumerate(ref_fid):
            if not chr.name in chr_lst:
                print('\t{:s} is ignored,'.format(chr.name))
                continue
            print('\t{:s},'.format(chr.name))

            re_pos = []
            re_type = []
            for frg in re.finditer(re_regex, chr.sequence, re.IGNORECASE):
                re_pos.append(frg.start() + 1)
                re_type.append(seq_lst.index(str.upper(frg.group())) + 1)
            re_pos_lst[chr2idx[chr.name]] = np.array(re_pos, dtype=np.uint32)
            re_type_lst[chr2idx[chr.name]] = np.array(re_type, dtype=np.uint32)
            chr_observed[chr2idx[chr.name]] = chr.name
        assert np.array_equal(chr_lst, chr_observed), '[e] Inconsistent reference genome!'
        print()

    # save the result
    np.savez(output_fname, pos_lst=re_pos_lst, type_lst=re_type_lst, chr_lst=chr_observed, genome=genome, scan_regex=re_regex)


def stable_unique(arr, return_index=False, axis=None):
    unq, inv, idx = np.unique(arr, return_index=True, return_inverse=True, axis=axis)
    srt_idx = np.argsort(inv)
    if return_index:
        inv[srt_idx] = range(len(inv))
        return unq[srt_idx], inv[idx]
    else:
        return unq[srt_idx]


def get_fasta_sequence(genome, chromosome, pos_start, pos_end):
    from urllib.request import urlopen
    from xml.etree import ElementTree

    message = 'http://genome.ucsc.edu/cgi-bin/das/{:s}/dna?segment={:s}:{:d},{:d}'.format(
            genome, chromosome, pos_start, pos_end)
    response_xml = urlopen(message)
    html = response_xml.read()  # I'm going to assume a safe XML here
    response_tree = ElementTree.fromstring(html)
    return response_tree[0][0].text.replace('\n', '').replace('\r', '')


def seq_complement(seq):
    trans_tbl = str.maketrans('TCGAtcga', 'AGCTagct')
    return seq.translate(trans_tbl)


def get_vp_info(run_id):
    vpi_lst = pd.read_csv('./vp_info.tsv', delimiter='\t', comment='#')  # , skip_blank_lines=False
    if isinstance(run_id, str):
        run_id = np.where(vpi_lst['run_id'] == run_id)[0]
        assert len(run_id) == 1
        run_id = int(run_id[0])
    vp_info = vpi_lst.iloc[run_id].to_dict()
    vp_info['row_index'] = run_id
    if not vp_info:
        raise Exception('VP information could not be found.')
    return vp_info


def load_dataset(vp_info_lst, target_field='frg_np', data_path='./datasets', verbose=True, vp_width=None,
                 load_cols=('chr', 'pos', '#read')):
    import h5py
    import gc

    if not isinstance(vp_info_lst, list):
        vp_info_lst = [vp_info_lst]

    # loop over runs
    if verbose:
        print('Loading data from:')
    for vp_idx, vp_info in enumerate(vp_info_lst):
        tsv_fname = '{:s}/{:s}/{:s}.hdf5'.format(data_path, vp_info['seqrun_id'], vp_info['original_name'])
        if verbose:
            print('\t#{:d}: [{:d}: {:s}]: {:s}'.format(vp_idx + 1, vp_info['row_index'], vp_info['run_id'], tsv_fname))
            if (vp_info_lst[0]['vp_chr'] != vp_info['vp_chr']) or (np.abs(vp_info_lst[0]['vp_pos'] - vp_info['vp_pos']) > 1e6):
                print('[w] Viewpoint is far away compared to others runs being loaded.')

        # load from hdf5
        with h5py.File(tsv_fname, 'r') as h5_fid:
            header_lst = list(h5_fid[target_field + '_header_lst'][()])
            frg_prt = pd.DataFrame(h5_fid[target_field][()], columns=header_lst)

        # only preserve requested columns
        if load_cols:
            frg_prt = frg_prt[list(load_cols)]

        # remove vp frag-ends?
        if vp_width is not None:
            print('\tRemoving frag-ends closer than {:0,.0f}bp to viewpoint.'.format(vp_width / 2.0))
            is_nei = (frg_prt['chr'] == vp_info['vp_chr']) & \
                     (np.abs(frg_prt['pos'] - vp_info['vp_pos']) < vp_width / 2.0)
            frg_prt = frg_prt.loc[~is_nei].reset_index(drop=True)

        # report stats
        if verbose:
            is_cis = frg_prt['chr'] == vp_info['vp_chr']
            is_lcl = is_cis & \
                     (np.abs(frg_prt['pos'] - vp_info['vp_pos']) < 100e3)
            print('\tData stats are:')
            print('\t\t #fe_cis: {:0,.0f}'.format(np.sum(frg_prt.loc[ is_cis, '#read'])))
            print('\t\t #fe_trs: {:0,.0f}'.format(np.sum(frg_prt.loc[~is_cis, '#read'])))

        if vp_idx == 0:
            frg_cmb = frg_prt.copy()
            del frg_prt
        else:
            agr_hlst = ['chr', 'pos']
            if np.array_equal(frg_cmb[agr_hlst], frg_prt[agr_hlst]):
                print('\t[i] Identical restriction sites detected. Direct addition of coverages...')
                frg_cmb['#read'] += frg_prt['#read']
                del frg_prt
            else:
                print('\t[i] Diverse restriction sites are detected. Aggregation ...')
                frg_cmb = frg_cmb.append(frg_prt.copy(), ignore_index=True, sort=False)
                del frg_prt

                # aggregation
                rf_inv, rf_idx = np.unique(frg_cmb[agr_hlst], axis=0, return_inverse=True, return_index=True)[1:]
                rf_nrd = np.bincount(rf_idx, weights=frg_cmb['#read'], minlength=len(rf_inv))
                frg_cmb = frg_cmb.loc[rf_inv, :].reset_index(drop=True)
                frg_cmb['#read'] = rf_nrd
                del rf_inv, rf_idx, rf_nrd
                gc.collect()
        if verbose:
            print('\tCurrent memory usage: {:0,.2f}GB'.format(frg_cmb.memory_usage().sum() / 1e9))

    if verbose and (len(vp_info_lst) != 1):
        print('Final memory usage: {:0,.2f}GB'.format(frg_cmb.memory_usage().sum() / 1e9))
    return frg_cmb.copy()


def perform_sigtest(observed, smoothing_kernel, background=None, n_epoch=1000, nz_only=False, replacement=False):
    if (observed.ndim != 1) or (background.ndim != 1):
        raise Exception('Inconsistant data are provided.')
    n_obs = len(observed)
    if background is None:
        background = observed.copy()
    if len(background) < len(observed):
        replacement = True

    # Calculate observed
    observed_smoothed = np.convolve(observed, smoothing_kernel, mode='same')

    # Calculate background
    bkgnd_mat = np.zeros([n_epoch, n_obs])
    if nz_only:
        inz_obs = observed > 0
        n_obs = np.sum(inz_obs)
        drawn_array = observed.copy()
        background = background[background > 0].copy()
        for ei in range(n_epoch):
            drawn_array[inz_obs] = np.random.choice(background, n_obs, replace=replacement)
            bkgnd_mat[ei, :] = np.convolve(drawn_array, smoothing_kernel, mode='same')
    else:
        for ei in range(n_epoch):
            drawn_array = np.random.choice(background, n_obs, replace=replacement)
            bkgnd_mat[ei, :] = np.convolve(drawn_array, smoothing_kernel, mode='same')

    # observed_smoothed[observed_smoothed < 0] = 0
    # bkgnd_mat[bkgnd_mat < 0] = 0
    return observed_smoothed, bkgnd_mat


def get_from_hpc(server_name, source_file, target_dir, verbose=False):
    import os
    import subprocess

    src_fname = server_name + ':' + source_file
    tar_path = os.path.expanduser(target_dir)
    if verbose:
        print('[i] Downloading: {:s}'.format(os.path.basename(src_fname)))
        print('\tFrom: {:s}'.format(src_fname))
        print('\t  To: {:s}'.format(tar_path))
    cmd_str = ['rsync', src_fname, tar_path]
    proc_dl = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # for out_line in iter(proc_dl.stdout.readline, b''):
    #     print(out_line.rstrip())
    stdout, stderr = proc_dl.communicate()
    return proc_dl, stdout, stderr


def str2crd(str_crd, genome='hg19'):
    chr_lst = get_chr_info(genome=genome, property='chr_name')
    n_chr = len(chr_lst)
    chr2nid = dict(zip(chr_lst, range(1, n_chr + 1)))

    items = str_crd.split(':')
    if len(items) == 1:
        return [chr2nid[items[0]]]
    else:
        return [chr2nid[items[0]]] + [float(x) for x in items[1].split('-')]


def slice_by_index(index, arr, assert_sorted=True, do_copy=True):
    def get_item(items, copy):
        if copy:
            return items.copy()
        else:
            return items

    n_row = index.shape[0]
    assert n_row == arr.shape[0]
    if assert_sorted:
        assert np.all(np.diff(index) >= 0)

    # looping over elements
    idx_be = 0
    for idx_en in range(1, n_row):
        if index[idx_be] != index[idx_en]:
            yield get_item(arr[idx_be:idx_en], copy=do_copy)
            idx_be = idx_en
    yield get_item(arr[idx_be:], copy=do_copy)


class OnlineStats(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    citation: Welford, B. P. (1962). doi:10.2307/1266577
    from: https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    example:
        from _utilities import OnlineStats
        ostat = OnlineStats(ddof=0)
        for item in item_lst:
            bkg_ostat.include(item)
        assert np.allclose(ostat.mean, np.mean(item_lst))
        assert np.allclose(ostat.std, np.std(item_lst))
    """

    def __init__(self, items=None, ddof=0):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if items is not None:
            for item in items:
                self.include(item)

    def include(self, item):
        delta = item - self.mean
        self.n += 1
        self.mean += delta / self.n
        self.M2 += delta * (item - self.mean)

    def combine(self, arr):
        if isinstance(arr, OnlineStats):
            arr_n, arr_mean, arr_m2 = arr.n, arr.mean, arr.M2
        else:
            arr_n, arr_mean, arr_m2 = len(arr), np.mean(arr), np.var(arr) * (len(arr) - self.ddof)
        ostat_cmb = OnlineStats(ddof=self.ddof)
        ostat_cmb.n = self.n + arr_n
        delta_mean = arr_mean - self.mean
        ostat_cmb.mean = (self.n * self.mean + arr_n * arr_mean) / ostat_cmb.n
        ostat_cmb.M2 = self.M2 + arr_m2 + (delta_mean * delta_mean) * self.n * arr_n / ostat_cmb.n
        return ostat_cmb

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


def get_read(input_fid):
    read = []
    for frg in input_fid:
        if len(read) == 0:
            read = [frg]
        else:
            if read[0].query_name == frg.query_name:
                read.append(frg)
            else:
                yield read
                read = [frg]
    yield read


def number_to_si(value, format='{:0.2f}'):
    # inspired by: https://stackoverflow.com/questions/10969759/python-library-to-convert-between-si-unit-prefixes
    _prefix = {'y': 1e-24,  # yocto
               'z': 1e-21,  # zepto
               'a': 1e-18,  # atto
               'f': 1e-15,  # femto
               'p': 1e-12,  # pico
               'n': 1e-9,  # nano
               'u': 1e-6,  # micro
               'm': 1e-3,  # mili
               'c': 1e-2,  # centi
               'd': 1e-1,  # deci
               'k': 1e3,  # kilo
               'M': 1e6,  # mega
               'G': 1e9,  # giga
               'T': 1e12,  # tera
               'P': 1e15,  # peta
               'E': 1e18,  # exa
               'Z': 1e21,  # zetta
               'Y': 1e24,  # yotta
               }
    np.searchsorted(_prefix[:, 1], value)