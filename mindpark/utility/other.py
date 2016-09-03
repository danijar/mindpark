import errno
import functools
import os
import re
import sys
import threading
import traceback
import yaml
from mindpark.utility.attrdict import use_attrdicts


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


def get_subdirs(directory):
    subdirs = os.listdir(directory)
    subdirs = [os.path.join(directory, x) for x in subdirs]
    subdirs = [x for x in subdirs if os.path.isdir(x)]
    return sorted(subdirs)


def color_stack_trace():

    def excepthook(type_, value, trace):
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

    sys.excepthook = excepthook
    setup_thread_excepthook()


def setup_thread_excepthook():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540

    Call once from the main thread before creating any threads.
    """
    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_original(self, *args, **kwargs)
        run_original = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                run_original(*args2, **kwargs2)
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


def natural_sorted(collection, key=lambda x: x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    natural_key = lambda x: [convert(y) for y in re.split('([0-9]+)', key(x))]
    return sorted(collection, key=natural_key)


def flatten(collection):
    if collection == []:
        return collection
    if isinstance(collection[0], list):
        return flatten(collection[0]) + flatten(collection[1:])
    return collection[:1] + flatten(collection[1:])


def dump_yaml(data, *path):
    def convert(obj):
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if not k.startswith('_')}
            return {convert(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if isinstance(obj, type):
            return obj.__name__
        return obj
    filename = os.path.join(*path)
    ensure_directory(os.path.dirname(filename))
    with open(filename, 'w') as file_:
        yaml.safe_dump(convert(data), file_, default_flow_style=False)


def print_headline(*message, style='-', minwidth=40):
    message = ' '.join(message)
    width = max(minwidth, len(message))
    print('\n' + style * width)
    print(message)
    print(style * width + '\n', flush=True)


def read_yaml(*path):
    path = os.path.join(*path)
    with open(path) as file_:
        return use_attrdicts(yaml.load(file_))
