# Standard libraries
import time

# Local modules
from lattice_class import Lattice
from observables import qed_hamiltonian

from solvers.dense import solve_and_observe_dense
from solvers.sparse import solve_and_observe_sparse
from solvers.noiseless import solve_and_observe_noiseless
from solvers.noisy import solve_and_observe_noisy

from config import vqe_parameters, SPSA_parameters

from global_helpers import dict_print

# Third-party libraries
import numpy as np

parameters = {
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'charge_weight': 10000.0
}

E_0s = np.linspace(0.0, 5.0, 10)

print("- Sparse - ")
particle_numbers = []
for e_0 in E_0s:
    lattice = Lattice(
        L_x = 3,
        L_y = 2,
        n_g = 2,
        dynamical_links_list = [((0, 0), 1), ((1, 0), 2)],
        charge_site = (0, 0),
        anticharge_site = (2, 1),
        background_field = [e_0, 0.0]
    )

    hamiltonian = qed_hamiltonian(parameters, lattice)

    results1 = solve_and_observe_sparse(hamiltonian, lattice)
    dict_print(results1)
    particle_numbers.append(results1['particle_number_total'])

print(E_0s)
print(particle_numbers)

print("- Noiseless - ")
particle_numbers = []
for e_0 in E_0s:
    lattice = Lattice(
        L_x = 3,
        L_y = 2,
        n_g = 2,
        dynamical_links_list = [((0, 0), 1), ((1, 0), 2)],
        charge_site = (0, 0),
        anticharge_site = (2, 1),
        background_field = [e_0, 0.0]
    )

    hamiltonian = qed_hamiltonian(parameters, lattice)

    results1 = solve_and_observe_noiseless(hamiltonian, lattice, vqe_parameters, SPSA_parameters)
    dict_print(results1)
    particle_numbers.append(results1['particle_number_total'])

print(E_0s)
print(particle_numbers)

