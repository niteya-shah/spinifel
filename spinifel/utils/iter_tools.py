import PyNVTX as nvtx


@nvtx.annotate("utils/iter_tools.py", is_prefix=True)
def prod(iterable):
    """Return product of elements of iterable."""
    if not iterable:
        return 0
    accumulator = 1
    for element in iterable:
        accumulator *= element
    return accumulator