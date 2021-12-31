__version__ = '0.0.1'

__dev__ = """
mkinit ~/code/shitspotter/shitspotter/__init__.py -w
"""


__submodules__ = {
    'util': ['open_shit_coco'],
}

from shitspotter import util

from shitspotter.util import (open_shit_coco,)

__all__ = ['open_shit_coco', 'util']
