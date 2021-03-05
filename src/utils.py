# utils.py

import config

def logger(func):
    """this is a decorator for logging a specific function and running it
    use with @logger before a function definitions if needs to be logged
    Args:
        func : a first class function

    Returns:
        : wrapped function return value
    """
    import logging
    logging.basicConfig(filename=f"{func.__name__}.log", level = logging.INFO)
    def wrapper(*args, **kwargs):
        """adds the functionality of logging for then func
        Returns:
            executed func
        """
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')  
        return func(*args, **kwargs)
    return wrapper

def timer(func):
    """a decorator to time the execution of func
    use with @timer before a function definitions if needs to be timed

    Args:
        func : a first class function

    Returns:
         wrapped function return value
    """
    import time
    def wrapper(*args, **kwargs):
        """adds the functionality of timing the exectable func

        Returns:
            executed func
        """
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time() - t0
        print(f'{func.__name__} elapsed {t1} sec')
        return ret
    return wrapper


def spread(arr):
    """spreads a list 

    Args:
        arr : a list of list

    Returns:
        opened up list of lists
    """
    ret = []
    for i in arr:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret

def deep_flatten(xs):
    """flattens everyhitng there is a list of lists in anay form

    Args:
        xs : all the elements

    Returns:
        : flattened data
    """
    flat_list = []
    [flat_list.extend(deep_flatten(x)) for x in xs] if isinstance(xs, list) else flat_list.append(xs)
    return flat_list

def chunk(list, size):
    """chunks a list to portions of size

    Args:
        list : list of elemetns
        size : chunk size

    Returns:
        : list of chunks
    """
    return [list[i:i+size] for i in range(0,len(list), size)]

def difference(a, b):
    """difference between two collection, no need to convert to set

    Args:
        a : [first collection]
        b : [second collection]

    Returns:
        the list of the remaining elements 
    """
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return list(comparison)

def merge_dictionaries(a, b):
    """given two dictionaries a and b merge them together

    Args:
        a : first dictionary
        b : second dictionary

    Returns:
        the merged dictionary
    """
    return {**a, **b}


def to_dictionary(keys, values):
    """given a key and values convert them to dictionary

    Args:
        keys : list of keys
        values : list of values associated with the keys

    Returns:
        dictionary of keys:values
    """
    return dict(zip(keys, values))

def most_frequent(list):
    """return the most frequent element of a list

    Args:
        list : list of elements

    Returns:
        the most frequent item
    """
    return max(set(list), key = list.count)