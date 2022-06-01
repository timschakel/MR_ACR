import os
import json
import datetime

"""
Changelog:
  20160514: python3
  20160421: renamed plugin to module
"""


def _checkString(name, value):
    value = value.encode("ascii", "xmlcharrefreplace").decode('ascii')
    if len(value) > 100:
        error = "({}) longer than 100 chars: {}".format(name, value)
        raise ValueError(error)
    return value


def _checkBool(name, value):
    if not isinstance(value, bool):
        error = "({}) expected type 'bool', received {}".format(
            name, type(value))
        raise TypeError(error)
    return '1' if value else '0'


def _checkDateTime(name, value):
    if not isinstance(value, datetime.datetime):
        error = "({}) expected type 'datetime', received {}".format(
            name, type(value))
        raise TypeError(error)
    return datetime.datetime.strftime(value, '%Y-%m-%d %H:%M:%S')


class ModuleResults(object):
    """Object to add and store results generated by the modules.

    Results are added using the public class methods:
        addBool, addFloat, addChar and addObject

    For each of these methods, the arguments are:
        description, value

        `description` and `value` are required.

    The SingleResult objects are retrieved by iterating over this object.
    """

    def __init__(self, out_path):
        """Instantiate self._results list: stores the SingleResult objects.
        """
        self._results = []
        self._out_path = out_path

    def __iter__(self):
        """Return an iterator over the self._results list."""
        return iter(self._results)

    def __len__(self):
        return len(self._results)

    def write(self):
        result_json_list = [result.getDict() for result in self._results]
        if len(result_json_list):
            with open(self._out_path, 'w') as f:
                json.dump(result_json_list, f)

    def _addResult(self, category, name, value, val_equal=None, val_min=None, val_max=None, val_low=None, val_high=None,
                   val_period=None):
        """Instantiate a SingleResult object and append to self._results list.
        """

        self._results.append(
            SingleResult(category, name, value, val_equal, val_min, val_max, val_low, val_high, val_period))

    def addDateTime(self, name, value, val_equal=None, val_period=None):
        """Add string representation of `value` (datetime) to results.
        Raise exception if `value` is not of type 'datetime'.
        """
        value = _checkDateTime(name, value)

        if val_equal is not None:
            val_equal = _checkDateTime(name + " constraint equal", val_equal)
        limits = [val_equal, None, None, None, None, val_period]
        self._addResult('datetime', name, value, *limits)

    def addBool(self, name, value, val_equal=None):
        """Add '0' or '1' to results depending on `value` (bool).
        Raise exception if `value` is not of type 'bool'.
        """
        value = _checkBool(name, value)

        if val_equal is not None:
            val_equal = _checkBool(name + " constraint equal", val_equal)

        self._addResult('bool', name, value, val_equal)

    def addFloat(self, name, value, val_equal=None, val_min=None, val_max=None, val_low=None, val_high=None):
        """Add string representation of `value` (float) to results.
        Raise exception if `value` can not be cast to float.
        """
        value = str(float(value))

        limits = [val_equal, val_min, val_max, val_low, val_high]
        for i, v in enumerate(limits):
            if v is not None:
                limits[i] = str(float(v))

        self._addResult('float', name, value, *limits)

    def addString(self, name, value, val_equal=None):
        """Add `value` (str) to results.
        Raise exception if `value` is longer than 100 (limited by WAD)
        """
        value = _checkString(name, value)

        if val_equal is not None:
            val_equal = _checkString(name + " constraint equal", val_equal)

        self._addResult('string', name, value, val_equal)

    def addObject(self, name, value):
        """Add `value` (str) to results if it represents an accessible filepath.
        Raise exception if the file cannot be accessed.
        """
        value = os.path.abspath(value)
        open(value).close()  # Test if file can be accessed

        self._addResult('object', name, value)


class SingleResult(object):
    """Object to store a single module result."""

    def __init__(self, category, name, value, val_equal, val_min, val_max, val_low, val_high, val_period):
        self.category = category
        self.val = value
        self.name = name
        self.val_equal = val_equal
        self.val_min = val_min
        self.val_max = val_max
        self.val_low = val_low
        self.val_high = val_high
        self.val_period = val_period

    def __repr__(self):
        """Return a human-readable string"""
        return '{}: {}'.format(self.name, self.val)

    def getDict(self):
        attributes = ('category', 'name', 'val',
                      'val_equal', 'val_min', 'val_max', 'val_low', 'val_high', 'val_period')

        return {attr: getattr(self, attr) for attr in attributes if getattr(self, attr) is not None}
