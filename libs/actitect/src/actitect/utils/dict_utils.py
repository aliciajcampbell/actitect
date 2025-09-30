import logging

logger = logging.getLogger(__name__)

__all__ = ['concat_dicts_with_times', 'fmt_dict', 'dict_minus_key', 'compare_dicts', 'subdict', 'nested_dict_iterator']


def concat_dicts_with_times(dict_a, dict_b):
    result = {}
    for key in set(dict_a.keys()).union(dict_b.keys()):
        value_a = dict_a.get(key)
        value_b = dict_b.get(key)

        if value_a is None and value_b is None:
            result[key] = None
        elif value_a is None:
            result[key] = value_b
        elif value_b is None:
            result[key] = value_a
        else:
            result[key] = value_a + value_b if 'time' in key else value_b
    return result


def fmt_dict(input_dict, digits=2):
    """ Returns a new dictionary with the values formatted to a specified number of decimal places
    if they are numbers, or left unchanged if they are strings.

    Parameters:
    input_dict (dict): The dictionary whose values need to be formatted.
    digits (int): The number of decimal places to format the values to.

    Returns:
    dict: A new dictionary with formatted values."""
    formatted_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, (int, float)):
            formatted_dict[k] = round(v, digits)
        elif isinstance(v, str):
            formatted_dict[k] = v
        else:
            formatted_dict[k] = v
    return formatted_dict


def dict_minus_key(in_dict, keys_to_remove):
    """ Returns a copy of in_dict without the key(s) specified in keys_to_remove.
    :param in_dict:
    :param keys_to_remove:
    :return:
    """
    return {key: in_dict[key] for key in in_dict if key not in keys_to_remove}


def compare_dicts(d1, d2, path=""):
    """Recursively compare two dictionaries and return differences."""
    differences = {}
    for key in d1.keys() | d2.keys():  # Union of keys
        if key not in d1:
            differences[path + key] = ("<MISSING>", d2[key])
        elif key not in d2:
            differences[path + key] = (d1[key], "<MISSING>")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            sub_diffs = compare_dicts(d1[key], d2[key], path + key + ".")
            differences.update(sub_diffs)
        elif d1[key] != d2[key]:
            differences[path + key] = (d1[key], d2[key])
    return differences


def subdict(original: dict, keys_to_keep: list):
    """ Returns a new sub-dictionary containing only the specified keys from the original dictionary.

    Parameters:
        original (dict): The original dictionary to filter.
        keys_to_keep (list): A list of keys to retain in the new dictionary.

    Returns:
        dict: A new dictionary with only the specified keys.
    """
    missing_keys = [k for k in keys_to_keep if k not in original]
    if missing_keys:
        raise KeyError(f"The following keys are missing in the original dictionary: {missing_keys}")

    return {k: original[k] for k in keys_to_keep}


def nested_dict_iterator(nested_dict: dict, iteration_depth: int, current_depth=1, keys=None):
    """ A utility function to iterate over nested dictionaries up to a specified depth.
    Parameters:
        :param nested_dict: (dict) The nested dictionary to iterate over.
        :param iteration_depth: (int) The depth of keys to return in the iterator.
        :param current_depth: (int) The current depth during recursion.
        :param keys: (list) The accumulated keys so far.
    Yields:
        tuple: A tuple where the first element is a list of keys at the specified depth
               and the second element is the value at the next level. """
    keys = keys or []
    if current_depth < iteration_depth:
        if isinstance(nested_dict, dict):
            for k, v in nested_dict.items():
                yield from nested_dict_iterator(v, iteration_depth, current_depth + 1, keys + [k])
    else:
        yield keys, nested_dict
