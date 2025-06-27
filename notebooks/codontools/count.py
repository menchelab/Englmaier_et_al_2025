import logging

import itertools as it
import pandas as pd
import numpy as np

from scipy.stats import binomtest
from statsmodels.stats.multitest import fdrcorrection
from typing import Optional, Callable, Any, Union
from . import concurrency


logging.basicConfig(
    format = '%(threadName)s: %(asctime)s-%(levelname)s-%(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    level = logging.INFO
)


def count_codons(
    transcript: tuple[str, str],
    codons_of_interest: Optional[list] = None
) -> tuple[str, dict[str, int]]:
    """
    counts the occurences of all possible codons

    :param transcript:          tuple of strings; the transcript from which the CDS is derived
                                and the DNA sequence of CDS
    :param codons_of_interest:  list of codons to count in the CDS

    :return:                    transcript_id, dictionary with codons as keys and codon counts as values
    """
    transcript_id, cds = transcript
    if not codons_of_interest:
        codon_counts = {
            ''.join(codon): 0 for codon in it.product(*['ACGT'] * 3)
        }
    
    else:
        codon_counts = {codon: 0 for codon in codons_of_interest}
        
    for i in range(0, len(cds), 3):
        codon = cds[i: i+3]
        
        if not codon in codon_counts:
            continue
            
        codon_counts[codon] += 1
    
    return transcript_id, codon_counts


def per_cds(
    count_func: Callable, 
    cdss: pd.Series, 
    n_processes: int = 1, 
    **kwargs
) -> dict[str, Any]:
    """
    execute count_func per sequence in cds_frame

    :param count_func:  Callable that takes a DNA sequence as string
    :param cdss:        pandas.Series with 'transcript_id' as keys and 'sequence' as values
    :param n_processes: number of processes to use for computation
    :param **kwargs:    any keyword arguments to pass to count_func

    :return:            dictionary with keys as in 'transcript_id' and values as return of count_func
    """
    if n_processes > 1:
        kwargs['n_processes'] = n_processes
        map_func = concurrency.mmap

    else:
        map_func = concurrency.smap

    codon_count_dict = map_func(
        count_func, 
        cdss.items(), 
        **kwargs
    )

    return codon_count_dict


def count_codons_per_cds(cdss: pd.Series, n_processes: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    count codons for all coding sequences in the given series

    :param cdss:        pandas.Series with 'transcript_id' as keys and 'sequence' as values
    :param n_processes: number of processes to use

    :return:            tuple of pandas.DataFrames with codon counts and codon frequencies per transcript
    """
    codon_count_dict = per_cds(
        count_codons, 
        cdss, 
        n_processes
    )
    codon_counts = pd.DataFrame.from_dict(
        codon_count_dict,
        orient = 'index'
    )
    return codon_counts


def count_codons_per_window(
    transcript: tuple[str, str],
    window_size: int, 
    codons_of_interest: list[str]
) -> tuple[str, dict[int, dict[str, int]]]:
    """
    compute the number of codons_of_interest in sliding windows of size window_size over cds

    :param transcript:          tuple of strings; the transcript from which the CDS is derived
                                and the DNA sequence of CDS
    :param window_size:         size of the sliding window to compute counts in
    :param codons_of_interest:  list of nucleotide triplets to count in sliding windows

    :return:                    dictionary of dictionaries containing codon counts per sequence window
    """
    transcript_id, cds = transcript
    def slide_window(sequence, window_size, n_windows):
        for i in range(n_windows):
            window_start = i * 3
            window_end = window_start + window_size * 3
            yield i, sequence[window_start: window_end]

    n_codons = len(cds)//3
    n_windows = n_codons - window_size + 1
    window_count_dict = {}
    for sliding_window in slide_window(cds, window_size, n_windows):
        i, window_counts = count_codons(
            sliding_window, 
            codons_of_interest
        )
        if sum(window_counts.values()) == 0:
            continue

        window_count_dict[i] = window_counts

    window_count_frame = pd.DataFrame.from_dict(
        window_count_dict,
        orient = 'index'
    )
    return transcript_id, window_count_frame
    

# needs multithreading to speed up computation
def count_windows_per_cds(
    cdss: pd.Series, 
    window_size: int, 
    codons_of_interest: list[str],
    n_processes: int = 1
) -> pd.DataFrame:
    """
    compute codon_of_interest counts in sliding windows over all CDS in cds_frame

    :param cdss:                pandas.Series with 'transcript_id' as keys and 'sequence' as values
    :param window_size:         size of the sliding window to use
    :param codons_of_interest:  list of nucleotide triplets to count in sliding windows
    :param n_processes:         number of processes to use

    :return:                    multi indexed pandas.DataFrame containing codon counts per sliding window per CDS
    """
    window_count_dicts = per_cds(
        count_codons_per_window, 
        cdss, 
        n_processes = n_processes,
        window_size = window_size,
        codons_of_interest = codons_of_interest,
    )
    return pd.concat(window_count_dicts.values(), keys = window_count_dicts.keys())


def estimate_codon_distribution(cdss: pd.Series, n_processes: int = 1) -> pd.Series:
    """
    computes the empirical distribution of codon counts (i.e. occurrences of codons in the whole translatome)

    :param cdss:        pandas.Series with 'transcript_id' as keys and 'sequence' as values

    :return:            frequencies of codons in the translatome
    """
    codon_counts = count_codons_per_cds(cdss, n_processes)
    sum_over_cds = codon_counts.sum(axis = 0)
    return sum_over_cds / sum_over_cds.sum()


def window_significance(
    id_and_counts: tuple[str, pd.Series], 
    window_size: int, 
    p: float, 
    fdr: float
) -> pd.DataFrame:
    """
    computes the binomial p-Value for all sliding windows over a given CDS 
    according to how likely it is to find X or more codons of interest in a given window.

    :param id_and_counts:               transcript_id and pandas.Series containing total counts of 
                                        codons of interest for a range of sliding windows
    :param window_size:                 size of the sliding window used
    :param p:                           the probability of seeing a codon of interest (see estimate_codon_distribution)
    :param fdr:                         significance level

    :return:                            pandas.DataFrame with p-Value and BH-adjusted p-Value per slidnig window
    """
    transcript_id, total_counts_per_window = id_and_counts
    pvalues_dict = {
        # cast count to int as new binomtest does not allow floats that are actually int
        i: binomtest(int(count), window_size, p, alternative = 'greater').pvalue
        for i, count
        in total_counts_per_window.reset_index(level = 0, drop = True).items()
    }
    pvalues = pd.DataFrame.from_dict(
        pvalues_dict,
        orient = 'index',
        columns = ['pval']
    )
    signif, padj = fdrcorrection(
        pvalues.pval,
        alpha = fdr
    )
    pvalues['padj'] = padj
    pvalues['signif'] = signif
    return transcript_id, pvalues


# needs multithreading to speed up computation
def compute_window_significance_per_cds(
    window_counts_frame: pd.DataFrame, 
    window_size: int, 
    p: float, 
    fdr: float,
    n_processes: int = 1
) -> pd.DataFrame:
    """
    computes binomial p-Values for a range of sliding windows over a range of CDSs
    according to how likely it is to find X or more codons of interest in a given window.

    :param window_counts_frame:     multilevel-indexed pandas.DataFrame containing 
                                    codons of interest counts per sliding window per CDS
                                    (see count_windows_per_cds for more details)
    :param window_size:             size of the sliding window used to compute window_counts_frame
    :param p:                       probability of seeing a codon of interest in the translatome
                                    (see estimate_codon_distribution)
    :param fdr:                     significance level
    :param n_processes:             number of processes to use

    :return:                        pandas.DataFrame containing binomial statistics for each window per CDS
    """
    total_counts_per_window = window_counts_frame.sum(axis = 1)
    total_counts_per_window.name = 'count'
    kwargs = {
        'window_size': window_size,
        'p': p,
        'fdr': fdr
    }
    if n_processes > 1:
        kwargs['n_processes'] = n_processes
        map_func = concurrency.mmap

    else:
        map_func = concurrency.smap

    pvalue_frame_dict = map_func(
        window_significance,
        total_counts_per_window.groupby(level = 0),
        **kwargs
    )
    
    pvalue_frame = pd.concat(
        pvalue_frame_dict.values(), 
        keys = pvalue_frame_dict.keys()
    )
    return pd.concat([total_counts_per_window, pvalue_frame], axis = 1)


def calculate_cluster_extents(
    window_stats: pd.DataFrame, 
    window_size: int
) -> dict[str, Union[int, list[tuple[int, int]]]]:
    """
    calculates the total length of all significant windows in a CDS in nucleotides

    :param window_stats:    pandas.DataFrame containing the computed p-Values for each sliding window
                            (see compute_window_significance_per_cds)
    :param window_size:     length of the sliding window used to compute the stats

    :return:                dictionary containing total length of clusters as well as coordinates of cluster stretches
    """
    window_stats = window_stats.reset_index(level = 0, drop = True)
    if not window_stats.signif.any():
        return {'cluster_length': 0, 'cluster_coords': []}

    def consecutive_windows(window_index):
        diff = np.diff(window_index)
        split_idx = ~(diff <= window_size)
        return np.split(window_index, np.where(split_idx)[0] + 1)
    
    total_significant_length = 0
    cluster_stretches = []
    for window_cluster in consecutive_windows(
        window_stats[window_stats.signif].index
    ):
        start, end = window_cluster[[0, -1]]
        # start, end and window_size are in codons
        cluster_length = (end - start + window_size) * 3
        cluster_stretches.append(
            (start * 3, start * 3 + cluster_length)
        )
        total_significant_length += cluster_length


    return {'cluster_length': cluster_length, 'cluster_coords': cluster_stretches}


def compute_cluster_score_per_cds(
    cdss: pd.Series, 
    codons_of_interest: list[str], 
    window_size: int, 
    fdr: float,
    n_processes: int = 1
) -> pd.DataFrame:
    """
    compute the cluster score (normalized length of windows significantly enriched in codons_of_interest)
    for each CDS in cds_frame.

    :param cdss:                pandas.Series with 'transcript_id' as keys and 'sequence' as values
    :param codons_of_interest:  list of nucleotide triplets to count in sliding windows
    :param window_size:         size of the sliding window used to compute window_counts_frame
    :param fdr:                 significance level

    :return:                    pandas.DataFrame containing cluster score, cluster extent and cluster coordinates as well as the CDS
    """
    logging.info('computing codon frequencies')
    codon_distribution = estimate_codon_distribution(
        cdss,
        n_processes
    )
    p = codon_distribution[codons_of_interest].sum()

    logging.info('counting codons of interest per window per CDS')
    window_counts = count_windows_per_cds(
        cdss, 
        window_size, 
        codons_of_interest,
        n_processes
    )

    logging.info('computing window significanes')
    window_stats = compute_window_significance_per_cds(
        window_counts,
        window_size,
        p,
        fdr,
        n_processes
    )

    logging.info('calulating cluster lengths')
    cluster_extents_dict = {
        transcript_id: calculate_cluster_extents(
            stats_per_window, 
            window_size
        )
        for transcript_id, stats_per_window
        in window_stats.groupby(level = 0)
    }
    cluster_extents = pd.DataFrame.from_dict(
        cluster_extents_dict,
        orient = 'index'
    )
    sequence_lengths = cdss.str.len()
    sequence_lengths.name = 'sequence_length'
    cluster_scores = pd.concat(
        [cluster_extents, sequence_lengths, cdss],
        axis = 1
    )
    cluster_scores['cluster_score'] = cluster_scores.cluster_length / cluster_scores.sequence_length * 1000
    return cluster_scores
