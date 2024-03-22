# 1. import pymo.so
import sys
pyversion = tuple(sys.version_info)
PYMO_LIB_PATH = [
    '/alg-secure-data/lib_superacme/' + f'py{pyversion[0]}{pyversion[1]}',
    '/alg-data/project/lib_superacme/' + f'py{pyversion[0]}{pyversion[1]}'
]
sys.path.extend(PYMO_LIB_PATH)
try:
    import libpymo
except ImportError:
    from mmrazor.utils import get_placeholder
    libpymo = get_placeholder('Make sure you provided the right access path.')

from .cle import apply_cross_layer_equalization

__all__ = ['apply_cross_layer_equalization']
