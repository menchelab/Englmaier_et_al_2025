import multiprocessing as mp

from functools import partial
from typing import Callable, Iterable, Any


def mmap(func: Callable, data: Iterable, n_processes: int, **kwargs) -> dict[str, Any]:
    """
    concurrent equivalent of map

    :param func:            function to map to data concurrently
    :param data:            Iterable to process, compatible with func
    :param n_processes:     number of processes to use
    :**kwargs:              any keyword arguments that need to be passed to func

    :return:                dictionary of results with id as key and result as value
    """
    func = partial(func, **kwargs)
    with mp.Pool(n_processes) as p:
        result = {k: v for k, v in p.imap(func, data)}
    
    return result


def smap(func: Callable, data: Iterable, **kwargs) -> dict[str, Any]:
    """
    sequential equivalent of map

    :param func:            function to map to data
    :param data:            Iterable to process, compatible with func
    :**kwargs:              any keyword arguments that need to be passed to func

    :return:                dictionary of results with id as key and result as value
    """
    func = partial(func, **kwargs)
    return {k: v for k, v in map(func, data)}