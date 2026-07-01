# Standard libraries
from copy import copy

# Local modules
from global_helpers import smart_round
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from operators import electric_n_direction, magnetic_term_n, particle_n_hamiltonian, charge_n_hamiltonian

# Third-party libraries
from scipy.sparse.linalg import eigsh
import numpy as np

def matrixfree_statevector_solver(hamiltonian : Hamiltonian):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")

    H = hamiltonian.to_linear_operator()

    v0 = np.ones(H.shape[0], dtype = np.complex128)
    v0 = v0 / np.linalg.norm(v0)

    eigenvalues, eigenvectors = eigsh(H, k = 1, which='SA', v0 = v0)

    groundstate_energy = eigenvalues[0].real
    groundstate = eigenvectors[:, 0]
    return groundstate_energy, groundstate


def matrixfree_statevector_solver_k(hamiltonian : Hamiltonian, k_ = 1):
    if not isinstance(hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian must be a hamiltonian_class.Hamiltonian, you have entered a {type(hamiltonian)}")

    H = hamiltonian.to_linear_operator()

    v0 = np.ones(H.shape[0], dtype = np.complex128)
    v0 = v0 / np.linalg.norm(v0)

    eigenvalues, eigenvectors = eigsh(H, k = k_, which='SA', v0 = v0)

    return eigenvalues.real, eigenvectors