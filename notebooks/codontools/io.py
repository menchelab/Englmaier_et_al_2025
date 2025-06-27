import re
import gzip

import pandas as pd

from functools import reduce
from typing import Union, Optional
from os import PathLike
from io import TextIOWrapper


PARSERS = dict(
    gene_symbol = re.compile('gene_symbol:(.+)'),
    gene_id = re.compile('gene:(.+)'),
    transcript_biotype = re.compile('transcript_biotype:(.+)'),
    gene_biotype = re.compile('gene_biotype:(.+)')
)

RESULT_KEYS = [
    'transcript_biotype', 
    'gene_biotype', 
    'gene_id', 
    'gene_symbol', 
    'transcript_id'
]
def parse_keyvalue(keyvalue: str) -> Union[tuple[str, str], tuple[None, None]]:
    """
    tries to extract the key value pairs of interest

    :param keyvalue:    key, value pair as string

    :return:            
    """
    for k, parser in PARSERS.items():
        m = parser.match(keyvalue)
        if m:
            return k, m.groups()[0]
        
    return None, None


def parse_cds_header(cds_header: str) -> dict[str, str]:
    """
    parses the header of a sequence in the CDS FASTA file

    :param cds_header:  CDS sequence header as string

    :return:            dictionary of all relevant key, value pairs (see PARSERS)
    """
    keyvalues = cds_header.split()
    result = {k: None for k in RESULT_KEYS}
    result['transcript_id'] = keyvalues.pop(0)[1:]
    for keyvalue in keyvalues:
        key, value = parse_keyvalue(keyvalue)
        if not key:
            continue
            
        result[key] = value
    
    return result


def parse_cds_file(cds_file: TextIOWrapper) -> pd.DataFrame:
    """
    parses a CDS sequence FASTA file and returnd the data as pandas.DataFrame

    :param cds_file:    filehandle of the CDS sequence FASTA to parse

    :return:            pandas.DataFrame containing the parsed infos
    """
    # get first cds_header here for cleaner code
    cds = parse_cds_header(
        cds_file.readline()
    )
    coding_sequences = {}
    sequence_parts = []
    for line in cds_file:
        if line.startswith('>'):
            cds['sequence'] = ''.join(sequence_parts)
            coding_sequences[cds['transcript_id']] = cds
            
            cds = parse_cds_header(line)
            sequence_parts = []
            continue
        
        sequence_parts.append(line.rstrip())
    
    return pd.DataFrame.from_dict(coding_sequences, orient = 'index')


def read_cds_fasta(filename: Union[PathLike, str], filter_dict: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """
    parses a CDS sequence FASTA file and returnd the data as pandas.DataFrame

    :param filename:    filehandle of the CDS sequence FASTA to parse
    :param filter_dict: dictionary containing dataframe columns as key and a desired value to filter this column for as value
                        (useful to filter for specific biotypes of CDS)

    :return:            pandas.DataFrame containing the parsed infos
    """
    if filename.endswith('gz'):
        fileopen = gzip.open
        mode = 'rt'

    else:
        fileopen = open
        mode = 'r'

    with fileopen(filename, mode) as cds_file:
        cds_frame = parse_cds_file(cds_file)
    
    if filter_dict:
        index = reduce(
            lambda x, y: x & y,
            [
                cds_frame[column] == value 
                for column, value 
                in filter_dict.items()
            ]
        )
        cds_frame = cds_frame.loc[index, :]
    
    return cds_frame


def write_cluster_score_frame(
    cluster_score_frames: dict[int, pd.DataFrame], 
    gene_symbol_mapping: pd.DataFrame,
    output_file: Union[PathLike, str]
) -> None:
    """
    writes a table containing all the cluster scores for each window size annotated by gene_symbol

    :param cluster_score_frames:    dictionary containing the computed scores for each window size 
                                    (see compute_cluster_score_per_cds)
    :param gene_symbol_mapping:     dataframe the column "gene_symbol" and "transcript_id" as index 
                                    (matching the transcript_ids of the cluster_score_frames)

    :return:                        None
    """
    frames = []
    for window_size, df in cluster_score_frames.items():
        df = df.loc[:, ['cluster_score']]
        df.columns = [f'cluster_score_w{window_size}']
        frames.append(df)
        
    cluster_score_frame = pd.concat(frames, axis = 1)
    cluster_score_frame = cluster_score_frame.merge(
        gene_symbol_mapping,
        right_index = True,
        left_index = True,
        how = 'inner'
    )
    cluster_score_frame = cluster_score_frame.drop_duplicates()
    cluster_score_frame.to_csv(
        output_file,
        sep = '\t'
    )