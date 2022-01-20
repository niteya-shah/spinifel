import os

from pathlib import Path
from inspect import stack, getmodule


# Location of this module -- all FQMN's will be from here, or a parent
PARENT = Path(__file__).parent


def get_parent(n, parent_path=PARENT):
    """
    get_parent(n, parent_path=PARENT)

    Returns n-th order parent directory of `parent_path`.
    """
    # parent_path = PARENT
    for i in range(n):
        parent_path = parent_path.parent
    return parent_path


def to_fqmn(path):
    """
    to_fqmn(path)

    Convert `path` to a Fully-Qalified-Module-Name. E.g.:
    `module/submodule/component.py`
    is converted to:
    `module.submodule.component`
    """
    module = str(path).replace(os.sep, '.')
    # remove the `.py` extension (if it has one)
    if module.endswith('.py'):
        return module[:-3]
    # don't remove non-py extensions
    return module


def is_relative(module_path, parent_path):
    """
    is_relative(module_path, parent_path)

    Checks if `module_path` starts with `parent_path`
    """
    return str(module_path).startswith(str(parent_path))
 

def to_relative(module_path, parent_path):
    """
    to_relative(module_path, parent_path)

    Returns the relative path from `parent_path` to `module_path` if they are
    relative to one-another. If not, do nothing.
    """
    if is_relative(module_path, parent_path):
        return module_path.relative_to(parent_path)
    # don't change non-related paths
    return module_path
 


def fully_qualified_module_name(n_parent=2):
    """
    fully_qualified_module_name(n_parent=2)

    Return the fully qualified module name (FQM) corresponding to `__file__`
    relative to this file's location. It will detect the file from which it is
    called and generates as FQMN from this. If the calling `__file__` is not in
    the same tree as this function, then simply the string representation of
    `__file__` is returned.

    Use `n_parent` to increase the number of parents that are considere the root
    of the module tree. Default is `n_pattern=2` as this module lives under
    `utils` but we want module names to be resolved from `spinfiel`.
    """
    frame  = stack()[1]
    module = getmodule(frame[0])

    module_path = Path(module.__file__)
    parent_path = get_parent(n_parent)
    
    if is_relative(module_path, parent_path):
        return to_fqmn(to_relative(module_path, parent_path))

    return str(module_path)