
class AlievPanfilovKernel:
    def __init__(self):
        self.arrays = ["u", "v"]
        self.scalars = ["a", "k", "mu1", "mu2", "eps"]
        self.observers = []
        self.dimensions = 2

        self.names = ["u", "v", "a", "k", "mu1", "mu2", "eps"]

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
        
    def generate_observers(self):
        return ""

    def generate_cpu_numba(self):
        src = f"""
from numba import njit, prange
@njit(parallel=True, fastmath=True)
def ionic_kernel_2d(u_new, indexes, dt, u, v, a, k, mu1, mu2, eps, {", ".join(self.observers)}):
    n_j = u_new.shape[1] 
    for idx in prange(indexes.shape[0]):
        ii = indexes[idx]
        i = ii // n_j
        j = ii %  n_j
        # 3d

        u_idx   = {self._indexing("u")}
        v_idx   = {self._indexing("v")}
        a_idx   = {self._indexing("a")}
        k_idx   = {self._indexing("k")}
        mu1_idx = {self._indexing("mu1")}
        mu2_idx = {self._indexing("mu2")}
        eps_idx = {self._indexing("eps")}
        
    
        {self._indexing("v")} += dt*calc_dv(v_idx, u_idx, a_idx, k_idx, eps_idx, mu1_idx, mu2_idx)
        u_new{self._raw_indexing()} += dt * calc_rhs(u_idx, v_idx, a_idx, k_idx)
        
        {self.generate_observers()}

"""
        return src