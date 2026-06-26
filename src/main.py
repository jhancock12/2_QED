# Standard libraries

# Local modules
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice
from solvers.sparse import sparse_statevector_solver
from observables import qed_hamiltonian
from operators import mass_term_n

from plotter import nice_scatter_plotter

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

qed_parameters = {
    'm' : 3.0, 
    'g' : 3.0, 
    'a' : 1.0, 
    'charge_weight' : 1000.0
}

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
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

chi_op_sparse = sum([mass_term_n(lattice, n).to_sparse_matrix() 
                     for n in range(lattice.n_fermion_qubits)])
gs = [0.3, 1.0, 3.0]
ms = np.linspace(-5.0, 5.0, 4)
As = [0.5, 1.0, 2.0]

print("gs =", gs)
print("ms =", ms.tolist())

for a in As:
    print("="*10)
    print("a =", a)
    print("="*10)

    qed_parameters['a'] = a

    Chiral_condensates = []

    for g in gs: 
        qed_parameters['g'] = g
        chiral_condensates = []
        for m in ms:
            qed_parameters['m'] = m
            hamiltonian = qed_hamiltonian(qed_parameters, lattice)
            groundstate_energy, groundstate = sparse_statevector_solver(hamiltonian)
            chi = np.real(np.vdot(groundstate, chi_op_sparse @ groundstate)) / lattice.n_fermion_qubits
            chiral_condensates.append(chi)
        Chiral_condensates.append(chiral_condensates)


    print("ccs =", Chiral_condensates)

data_x = []
data_y = []
data_y_err = None
data_x_line = ms
data_y_line = Chiral_condensates
log_x_scale = False 
log_y_scale = False

label_x = r"$m$"
label_y = r"$\chi$"
label_title = ""
labels = []
labels_line = [r"$g = "+str(g)+r"$" for g in gs]
same_color = True
square = True

label_save_title = "temp"
save = True

nice_scatter_plotter(data_x = data_x, data_y = data_y, data_y_errors= data_y_err, data_x_line = data_x_line, data_y_line = data_y_line,
                     label_x = label_x, label_y = label_y, label_title = label_title, 
                     save = save, label_save_title = label_save_title, marker = "x", labels = labels, labels_line = labels_line,
                     log_x_scale = log_x_scale, log_y_scale = log_y_scale)