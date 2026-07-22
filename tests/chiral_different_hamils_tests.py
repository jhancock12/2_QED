# Standard libraries

# Local modules
from global_helpers import dict_print
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from observables import *
from operators import mass_term_n
from solvers.sparse import sparse_statevector_solver
from plotter import nice_scatter_plotter

# Third-party libraries
import numpy as np

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1), ((1, 0), 2)],# ((2, 0), 2)],#  ((0, 1), 1), ((1, 1), 2), ((2, 1), 2)],
    'charge_site' : (),
    'anticharge_site' : (),
    'background_field' : [0.0, 0.0]
}

lattice = Lattice(
    L_x = lattice_parameters['L_x'],
    L_y = lattice_parameters['L_y'],
    n_g = lattice_parameters['n_g'],
    dynamical_links_list = lattice_parameters['dynamical_links_list'],
    charge_site = lattice_parameters['charge_site'],
    anticharge_site = lattice_parameters['anticharge_site'],
    background_field = lattice_parameters['background_field']
)

qed_parameters = {
    'm' : 1.0, 
    'g' : 3.0,  
    'a' : 1.0, 
    'charge_weight' : 20.0
}

chi_op_sparse = sum([mass_term_n(lattice, n).to_sparse_matrix() 
                     for n in range(lattice.n_fermion_qubits)])

names = {"mass_" : mass_hamiltonian,
          "electric_" : electric_hamiltonian, 
          "magnetic_" : magnetic_hamiltonian, 
          "kinetic_" : kinetic_hamiltonian}

lambdas = np.linspace(-1.0, 1.0, 30)
ms = lambdas * qed_parameters['g'] ** 2 

results = {}
results_list = []

charge_ = charge_hamiltonian(qed_parameters, lattice)

counter = 0
for name0 in names:
    for name1 in names:
        if name0 != name1:
            results[name0 + name1] = []
            results_list.append([])
            for m in ms:
                qed_parameters['m'] = m

                temp_hamiltonian = Hamiltonian(lattice.n_qubits)
                temp_hamiltonian.add_hamiltonian(names[name0](qed_parameters, lattice, 1))
                temp_hamiltonian.add_hamiltonian(names[name1](qed_parameters, lattice, 1))
                temp_hamiltonian.add_hamiltonian(charge_)

                groundstate_energy, groundstate = sparse_statevector_solver(temp_hamiltonian)
                chiral_condensate = np.real(np.vdot(groundstate, chi_op_sparse @ groundstate))
                results[name0 + name1].append(chiral_condensate)
                results_list[counter].append(chiral_condensate)
            counter += 1

results["full"] = []
results_list.append([])
for m in ms:
    qed_parameters['m'] = m
    full_hamiltonian = qed_hamiltonian(qed_parameters, lattice)
    groundstate_energy, groundstate = sparse_statevector_solver(full_hamiltonian)
    chiral_condensate = np.real(np.vdot(groundstate, chi_op_sparse @ groundstate))
    results["full"].append(chiral_condensate)
    results_list[counter].append(chiral_condensate)

print("lambdas:", lambdas.tolist())
dict_print(results)

label_x = r"$\lambda$"
label_y = r"$\mathcal{C}$"
data_x_line = lambdas
data_y_line = results_list

labels_line = list(results.keys())
label_save_title = "temp"

nice_scatter_plotter(
    data_x=None, data_y=None, data_y_errors=None,
    data_x_line=data_x_line, data_y_line=data_y_line,
    label_x=label_x, label_y=label_y, label_title="",
    save=True, label_save_title=label_save_title, marker="-x",
    labels=None, labels_line=labels_line,
    log_x_scale=False, log_y_scale=False,
    same_color=False, square = False
) 