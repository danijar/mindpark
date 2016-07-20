import traceback
import errno
import functools
import os
import sys


def ensure_directory(directory):
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def clamp(value, min_, max_):
    return max(min_, min(value, max_))


def merge_dicts(*mappings):
    merged = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def sum_dicts(*mappings):
    summed = {}
    for key, value in mappings.items():
        if key not in summed:
            summed[key] = value
        else:
            summed[key] = summed[key] + value
    return summed


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def color_stack_trace(type_, value, trace):
    text = ''.join(traceback.format_exception(type_, value, trace))
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import TerminalFormatter
        lexer = get_lexer_by_name('pytb', stripall=True)
        formatter = TerminalFormatter()
        sys.stderr.write(highlight(text, lexer, formatter))
    except Exception:
        sys.stderr.write(text)
        sys.stderr.write('Failed to colorize the traceback.')
