import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import netket as nk
import netket.experimental as nkx
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from netket.operator.spin import sigmax,sigmay,sigmaz
import scipy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc
from netket.experimental.observable import Renyi2EntanglementEntropy
from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from typing import Any, Callable, Sequence
from functools import partial
from jax.experimental import sparse
from itertools import combinations
from itertools import permutations
from scipy.special import comb

# Create hilbert space
N = 16
N_f = N//2
hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)

# Set up Mini-SYK Hamiltonian
RNG = np.random.default_rng(0)
J = RNG.normal(0, 1/np.sqrt(N), size=comb(N,2,exact=True))
V = RNG.normal(0, 1/np.sqrt(N), size=comb(N,2,exact=True))
hamiltonian = 0
for i, pair in zip(range(comb(N,2,exact=True)), combinations(range(N), 2)):
    hamiltonian += J[i] * cdag(hi, pair[0]) * c(hi, pair[1]) + J[i] * cdag(hi, pair[1]) * c(hi, pair[0])
    hamiltonian += V[i] * nc(hi, pair[0]) * nc(hi, pair[1]) + V[i] * nc(hi, pair[1]) * nc(hi, pair[0])

# Diagonalize with SciPy
hamiltonian_sparse = hamiltonian.to_sparse()
eig_vals, eig_vecs = eigsh(hamiltonian_sparse, k=2, which="SA")
e_gs = eig_vals[0]
print("Exact ground state energy:", e_gs)

# Jastrow-Slater architecture
class LogNeuralJastrowSlater(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    hidden_units: int

    @nn.compact
    def __call__(self, n):

        @partial(jnp.vectorize, signature='(n)->()')
        def log_wf(n):
            #Bare Slater Determinant (N x Nf matrix of the orbital amplitudes)
            M = self.param('M', default_kernel_init, (self.hilbert.n_orbitals, self.hilbert.n_fermions,), float)

            #Construct the Neural Jastrow
            J = nn.Dense(self.hidden_units, param_dtype=float)(n)
            J = jax.nn.tanh(J)
            J = J.sum()

            # Find the positions of the occupied orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            # Select the N rows of M corresponding to the occupied orbitals, obtaining the Nf x Nf slater matrix
            A = M[R]
            # compute the (log) determinant and add the Jastrow
            # (when exponentiating this becomes a product of the slater and jastrow terms)
            return nk.jax.logdet_cmplx(A)+J

        return log_wf(n)

# Backflow-Slater architecture
class LogNeuralBackflow(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    hidden_units: int
    hidden_layers: int

    @nn.compact
    def __call__(self, n):

        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            #Bare Slater Determinant (N x Nf matrix of the orbital amplitudes)
            M = self.param('M', default_kernel_init, (self.hilbert.n_orbitals, self.hilbert.n_fermions,), float)

            # Construct the Backflow. Takes as input strings of $N$ occupation numbers, outputs an $N x Nf$ matrix
            # that modifies the bare orbitals.
            for i in range(self.hidden_layers):
                F = nn.Dense(self.hidden_units, param_dtype=float)(n)
                F = jax.nn.tanh(F)
            # last layer, outputs N x Nf values
            F = nn.Dense(self.hilbert.n_orbitals * self.hilbert.n_fermions, param_dtype=float)(F)
            # reshape into M and add
            M += F.reshape(M.shape)


            #Find the positions of the occupied, backflow-modified orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            A = M[R]
            return nk.jax.logdet_cmplx(A)

        return log_sd(n)

# Pick an architecture
model = LogNeuralJastrowSlater(hi, N)
# model = LogNeuralBackflow(hi, N, 1)

# Define all-to-all graph for sampler
graph = nk.graph.Graph([(i,j) for i in range(N) for j in range(N) if i!=j])

# Pick a sampler
# sampler = nk.sampler.ExactSampler(hi)
sampler = nk.sampler.MetropolisExchange(hi, graph=graph)

optimizer = nk.optimizer.Adam(learning_rate=0.01)

# Pick FullSum or MCMC
# vstate = nk.vqs.FullSumState(hi, model)
vstate = nk.vqs.MCState(sampler, model, n_samples=2**11)

# Tune diag_shift
preconditioner = nk.optimizer.SR(diag_shift=1e-2)

# Use preconditioner or not
# gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate)
gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=preconditioner)

logger=nk.logging.RuntimeLog()

# Run optimization
gs.run(n_iter=200, out=logger)

# Display results
sd_energy = vstate.expect(hamiltonian)
error = abs((sd_energy.mean - e_gs) / e_gs)
print(f"Optimized energy : {sd_energy}")
print(f"Exact energy : {e_gs}")
print(f"Relative error : {error}")
print(f"Tau Corr : {sd_energy.tau_corr}")
print(f"Fraction Accepted : {vstate.sampler_state.n_accepted/vstate.sampler_state.n_steps}")
plt.plot(logger.data['Energy']['iters'], logger.data['Energy']['Mean'])
plt.hlines([e_gs], xmin=0, xmax=max(logger.data['Energy']['iters']), color='black', label="Exact")
plt.xlabel("Time steps")
plt.ylabel("Energy")
plt.show();
