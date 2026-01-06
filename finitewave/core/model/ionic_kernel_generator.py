import re
import warnings

class IonicKernelGenerator:    
    """
    Base generator for model ionic kernels.

    Conventions:
    - arrays: names passed as array arguments (e.g., u, v, gating variables, current fields)
    - scalars: names passed as scalar arguments (e.g., parameters)
    - observers: list of dicts: {"name": <arg_name>, "expr": <code>}
      where expr is injected at the end of the per-cell loop body.

    Observer notes:
    - This is advanced instrumentation; expr must be numba-friendly and race-safe.
    - No dynamic append / allocation in parallel kernels.
    """

    def __init__(self):
        self.arrays = ["u"]
        self.scalars = []
        self.observers = []
        self.dimensions = 2

        self.names = ["u"]

    def _indexing(self, name):
        if name in self.arrays:
            if self.dimensions == 2:
                return f"{name}[i, j]"
            elif self.dimensions == 3:
                return f"{name}[i, j, k]"
            else:
                raise ValueError("Unsupported number of dimensions")
        else:
            return name
        
    def _raw_indexing(self):
        if self.dimensions == 2:
            return "[i, j]"
        elif self.dimensions == 3:
            return "[i, j, k]"
        else:
            raise ValueError("Unsupported number of dimensions")
        
    def _normalize_observers(self):
        ident = re.compile(r"^[A-Za-z_]\w*$")

        if not self.observers:
            return [], ""

        if not isinstance(self.observers, (list, tuple)):
            raise TypeError("observers must be a list of dicts with keys: 'name', 'expr'.")

        seen = set()
        args = []
        lines = []

        for idx, obs in enumerate(self.observers):
            if not isinstance(obs, dict):
                raise TypeError(f"Observer #{idx} must be a dict.")
            if "name" not in obs or "expr" not in obs:
                raise ValueError(f"Observer #{idx} must have keys 'name' and 'expr'.")

            name = obs["name"]
            expr = obs["expr"]

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Observer #{idx}: 'name' must be a non-empty string.")
            name = name.strip()

            if not ident.match(name):
                raise ValueError(
                    f"Observer #{idx}: invalid name '{name}'. Must be a valid Python identifier."
                )

            if name in seen:
                raise ValueError(f"Duplicate observer name '{name}'.")
            seen.add(name)

            if not isinstance(expr, str) or not expr.strip():
                raise ValueError(f"Observer '{name}': 'expr' must be a non-empty string.")
            expr = expr.strip()

            # warnings (cheap heuristics)
            if "append(" in expr or ".append(" in expr:
                warnings.warn(
                    f"Observer '{name}': 'append' in expr is unsafe in numba-parallel kernels. "
                    f"Use preallocated arrays like {name}[step, ...] = value.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if "import " in expr:
                warnings.warn(
                    f"Observer '{name}': imports in expr are not allowed/expected.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if name not in expr:
                warnings.warn(
                    f"Observer '{name}': expr does not reference its output buffer '{name}'. "
                    f"Did you forget to write into it?",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if "=" not in expr and not expr.lstrip().startswith("if "):
                warnings.warn(
                    f"Observer '{name}': expr has no '='; it may not store anything.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            args.append(name)
            lines.append(expr)

        return args, "\n        ".join(lines) # 8 spaces for indentation

    def generate_observers(self) -> str:
        _, code = self._normalize_observers()
        return code

    def kernel_func_name(self) -> str:
        return "ionic_kernel"

    def kernel_base_args(self) -> list[str]:
        # common arguments: output, indexes, dt, (optional step)
        base = ["u_new", "indexes", "dt"]
        if self.include_step:
            base.append("step")
        # then arrays + scalars
        base.extend(self.arrays)
        base.extend(self.scalars)
        # then observers
        obs_args, _ = self._normalize_observers()
        base.extend(obs_args)
        return base

    def generate_loop_header(self) -> str:
        if self.dimensions == 2:
            return """\
    n_j = u_new.shape[1]
    for idx in prange(indexes.shape[0]):
        ii = indexes[idx]
        i = ii // n_j
        j = ii % n_j
"""
        # 3D placeholder; you can override in subclasses
        return """\
    n_k = u_new.shape[2]
    n_j = u_new.shape[1]
    for idx in prange(indexes.shape[0]):
        ii = indexes[idx]
        i = ii // (n_j * n_k)
        rem = ii % (n_j * n_k)
        j = rem // n_k
        k = rem % n_k
"""

    def generate_body(self) -> str:
        """
        Subclasses must override this to generate the per-cell body BEFORE observers.
        Must end with state updates.
        """
        raise NotImplementedError

    def generate_cpu_numba(self) -> str:
        args = ", ".join(self.kernel_base_args())
        loop = self.generate_loop_header()
        body = self.generate_body()
        obs = self.generate_observers()

        src = f"""
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def {self.kernel_func_name()}({args}):
{loop}
{body}
        {obs}
"""
        return src