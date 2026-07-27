# Standard libraries

# Local modules
from lattice_class import Lattice
from solvers.sparse import solve_and_observe_sparse
from observables import qed_hamiltonian
from operators import mass_term_n
from global_helpers import dict_print

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

qed_parameters = {
    'm' : 3.0, 
    'g' : 3.0,  
    'a' : 1.0, 
    'charge_weight' : 20.0
}

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 4,
    'dynamical_links_list' : [((0, 0), 1), ((1, 0), 2)],# ((2, 0), 2)],#  ((0, 1), 1), ((1, 1), 2), ((2, 1), 2)],
    'charge_site' : (),
    'anticharge_site' : (),
    'background_field' : [0.0, 0.0]
}

dict_print(lattice_parameters)

lattice = Lattice(
    L_x = lattice_parameters['L_x'],
    L_y = lattice_parameters['L_y'],
    n_g = lattice_parameters['n_g'],
    dynamical_links_list = lattice_parameters['dynamical_links_list'],
    charge_site = lattice_parameters['charge_site'],
    anticharge_site = lattice_parameters['anticharge_site'],
    background_field = lattice_parameters['background_field']
)

chi_op_sparse = sum([mass_term_n(lattice, n).to_sparse_matrix() 
                     for n in range(lattice.n_fermion_qubits)])

gs = np.linspace(0.3, 3.0, 10)
E0s = np.linspace(0.0, 10.0, 10)
energies = []
for e0 in E0s:
    print("-"*10)
    print("BEF:", e0)
    lattice = Lattice(
        L_x = lattice_parameters['L_x'],
        L_y = lattice_parameters['L_y'],
        n_g = lattice_parameters['n_g'],
        dynamical_links_list = lattice_parameters['dynamical_links_list'],
        charge_site = lattice_parameters['charge_site'],
        anticharge_site = lattice_parameters['anticharge_site'],
        background_field = [e0, e0]
    )
    hamiltonian = qed_hamiltonian(qed_parameters, lattice, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1)
    results = solve_and_observe_sparse(hamiltonian, lattice)
    dict_print(results)

    max_electric = max(np.abs(np.array(list(results['electric_field_dict'].values()))))
    print("max_electric:", max_electric)
    energies.append(max_electric)

plt.figure(figsize = (10, 8), dpi = 100)
plt.plot(E0s, energies)
plt.show()

