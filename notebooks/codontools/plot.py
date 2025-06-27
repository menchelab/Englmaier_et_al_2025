import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


from os import PathLike
from Bio import Seq
from typing import Union


mpl.rcParams['pdf.fonttype'] = 42


def slice_sequence(
    sequence: str, 
    cluster_coords: list[tuple[int, int]]
) -> list[tuple[str, bool]]:
    """
    slices a string into pieces according to start, end coordinates in cluster_coords

    :param sequence:        string to slice
    :param cluster_coords:  list of start, end coordinate tuples

    :return:                list of slice, is_cluster tuples where is_cluster denotes if the slice is a window cluster
    """
    sequence_length = len(sequence)
    slices = []
    prev_end = 0
    for slice_start, slice_end in cluster_coords:
        slices.extend(
            [
                (sequence[prev_end: slice_start], False),
                (sequence[slice_start: slice_end], True)
            ]
        )
        prev_end = slice_end
        
    if prev_end < sequence_length:
        slices.append((sequence[prev_end: sequence_length], False))
    
    return slices
    
        
def get_text_extents(
    text: mpl.text.Text, 
    renderer: mpl.backend_bases.RendererBase, 
    inverse_ax_transform: mpl.transforms.Transform
) -> tuple[float, float, float, float]:
    """
    uses inverse plt.Axes transform to compute the extent of the bounding box
    of the text object

    :param text:                    Text object returned by ax.text
    :param renderer:                canvas renderer returned by fig.get_renderer
    :param inverse_ax_transform:    inverse transform of the Axes object

    :return:                        x, y, width and height of the text bounding box
    """
    bb = text.get_window_extent(renderer = renderer)
    transformed_bb = inverse_ax_transform.transform_bbox(bb)
    width, height = transformed_bb.width, transformed_bb.height
    x, y = transformed_bb.x0, transformed_bb.y0
    return x, y, width, height


def plot_string(
    string: str, 
    xpos: float,
    ypos: float, 
    ax: plt.Axes, 
    **kwargs
) -> float:
    """
    plots the given string and returns the x position of the righthand side of the bounding box

    :param string:  string to add to the Axes
    :param xpos:    x coordinate of the string 
    :param ypos:    y coordinate of the string
    :param ax:      Axes object to add the string to
    :**kwargs:      any keyword arguments passed to ax.text

    :return:        x coordinate of the righthand side of the bounding box of the added text
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    inv_ax_trans = ax.transData.inverted()
    txt = ax.text(
        xpos,
        ypos,
        string,
        **kwargs
    )
    _, _, width, _ = get_text_extents(txt, renderer, inv_ax_trans)
    return xpos + width


def plot_individual_codons(
    sequence: str, 
    xpos: float, 
    ypos: float, 
    ax: plt.Axes, 
    append_separator: bool = True, 
    **kwargs
) -> float:
    """
    plots the given nucelotide sequence as separate codons including their
    one-letter aminoacid translation

    :param sequence:            string to generate the codon plot for
    :param xpos:                x coordinate for the first codon
    :param ypos:                y coordinate of the whole string
    :param ax:                  Axes object to plot the string in
    :param append_separator:    if to append the separator character to the end of the string
                                False for any sting that terminates the sequence, True otherwise
    :**kwargs:                  any keyword arguments to pass to ax.text
    """
    codon_list = [sequence[i: i+3] for i in range(0, len(sequence), 3)]
    codon_suffix = '|' if append_separator else ''
    prev_xpos = xpos
    y_offset = 0.4
    xpos = plot_string(
        '|'.join(codon_list) + codon_suffix,
        prev_xpos,
        ypos - y_offset,
        ax,
        va = 'bottom',
        **kwargs
    )
    
    aminoacid_list = list(Seq.translate(sequence))
    aminoacid_suffix = '' * 2 if append_separator else ''
    plot_string(
        ' ' + (' ' * 3).join(aminoacid_list) + aminoacid_suffix,
        prev_xpos,
        ypos + y_offset,
        ax,
        va = 'top',
        **kwargs
    )
        
    return xpos


def plot_clusters_n_largest(
    cluster_score_frame: pd.DataFrame, 
    n: int
) -> tuple[plt.Figure, plt.Axes]:
    """
    plot the cluster annotated codon sequence of the n CDS with the largest cluster score

    :param cluster_score_frame:     pandas.DataFrame with columns 'cluster_score', 'cluster_coords', 'gene_symbol' and 'sequence'
    :param n:                       number of top ranked CDS to plot

    :return:                        plt.Figure, plt.Axes of the generated plot
    """
    n_largest = cluster_score_frame.nlargest(n, 'cluster_score')

    fig, ax = plt.subplots()
    ax.set_ylim(n)
    ypos = 0
    for _, cds in n_largest.iterrows():
        ypos += 2 * n // 10
        sequence_slices = slice_sequence(
            cds.sequence,
            cds.cluster_coords,
        )
        plot_string(
            cds.gene_symbol,
            0,
            ypos,
            ax
        )
        plot_string(
            '{:.5f}'.format(cds.cluster_score),
            1,
            ypos,
            ax,
            ha = 'right'
        )
        xpos = 1.5
        for i, (seq_slice, is_cluster) in enumerate(sequence_slices):
            xpos = plot_individual_codons(
                seq_slice,
                xpos,
                ypos,
                ax,
                True if i < len(sequence_slices) - 1 else False,
                c = 'r' if is_cluster else 'k',
                fontfamily = 'monospace'
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for pos in ['right', 'left', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)

    ax.axis('off')
    return fig, ax