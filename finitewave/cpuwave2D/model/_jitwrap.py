# finitewave/models/_jitwrap.py
from numba import njit

def wrap_calc(ops):
    jitted = {}
    for name in dir(ops):
        if name.startswith(("calc_", "rhs_")):
            fn = getattr(ops, name)
            if callable(fn):
                jitted[name] = njit(cache=True)(fn)
    return jitted
