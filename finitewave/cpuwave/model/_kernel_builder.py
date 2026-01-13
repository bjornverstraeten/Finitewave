from functools import lru_cache
from numba import njit, prange
from finitewave.core.model.ionic_kernel_generator import IonicKernelGenerator


def _freeze_observers(observers):
    if not observers:
        return tuple()
    return tuple((o["name"], o["expr"]) for o in observers)

def merge_set(base: list[str], extra: tuple[str, ...]) -> list[str]:
    seen = set(base)
    for x in extra:
        if x not in seen:
            base.append(x)
            seen.add(x)
    return base

@lru_cache(maxsize=64)
def _build_cached(gen_cls, dimensions, scalar_params, array_params, observers_key, glb_key):
    gen = gen_cls()
    gen.dimensions = dimensions
    gen.scalars = merge_set(list(gen.scalars), scalar_params)
    gen.arrays  = merge_set(list(gen.arrays),  array_params)
    gen.observers = [{"name": n, "expr": e} for (n, e) in observers_key]

    src = gen.generate_cpu_numba()

    loc = {}
    glb = { # dict of injected globals (calc_*, etc.)
        "njit": njit,
        "prange": prange,
        **dict(glb_key),        
    }
    exec(src, glb, loc)

    return loc[gen.kernel_func_name()], src

def build_kernel(gen: IonicKernelGenerator, glb: dict, dimensions: int,
                 scalar_params=(), array_params=(), observers=()):
    """
    gen: *instance* used only to pass class + defaults; build uses its class.
    glb: injected globals for exec (calc_* etc.) — must be stable for caching
    """
    observers_key = _freeze_observers(observers)

    # make globals hashable for caching
    # simplest V0: disable caching based on glb (or use ids)
    glb_key = tuple(sorted(glb.items(), key=lambda kv: kv[0]))

    fn, src = _build_cached(
        gen.__class__,
        int(dimensions),
        tuple(scalar_params),
        tuple(array_params),
        observers_key,
        glb_key,
    )
    return fn, src
