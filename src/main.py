# Standard libraries
import copy 

# Local modules
from lattice_class import Lattice
from solvers.sparse import sparse_statevector_solver, observes_reduced_from_statevector
from observables import qed_hamiltonian
from operators import mass_term_n
from global_helpers import dict_print
from plotter import nice_scatter_plotter

# Third-party libraries
import numpy as np

qed_parameters = {
    'm' : 3.0, 
    'g' : 3.0,  
    'a' : 1.0, 
    'charge_weight' : 1000.0
}

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 3,
    'dynamical_links_list' : [((0, 0), 1), ((1, 0), 2)], # ((2, 0), 2), ((0, 1), 1), ((1, 1), 2), ((2, 1), 2)],
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

g_fixed = 3.0
xs = [1.0, 2.0]
lambdas = np.linspace(-1.0, 1.0, 11)

Chiral_condensates_by_x = []

total = len(xs) * len(lambdas)

Energies = []
Gaps = []
Zero_points = [] 
counter = 1
for x in xs:
    a = x / g_fixed**2
    qed_parameters['a'] = a
    qed_parameters['g'] = g_fixed
    energies_ = []
    gaps = []
    chiral_condensates = []
    for lam in lambdas:
        m = lam * g_fixed**2
        qed_parameters['m'] = m
        hamiltonian = qed_hamiltonian(qed_parameters, lattice, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1)
        gs_energy, gs_vector = sparse_statevector_solver(hamiltonian)
        chi = np.real(np.vdot(gs_vector, chi_op_sparse @ gs_vector)) / lattice.n_fermion_qubits
        if len(chiral_condensates) > 0:
            if abs(chi) < 1e-12 or chiral_condensates[-1] * chi < 0:
                lam_prev = lambdas[len(chiral_condensates) - 1]
                chi_prev = chiral_condensates[-1]
                lam_cross = lam_prev - chi_prev * (lam - lam_prev) / (chi - chi_prev)
                Zero_points.append(lam_cross)
        results = observes_reduced_from_statevector(gs_vector, lattice)
        chiral_condensates.append(chi)
        energies_.append(gs_energy)

        print(round((counter / total) * 100, 5), "% done")
        counter += 1

    Chiral_condensates_by_x.append(chiral_condensates)
    Energies.append(energies_)
    

Chiral_chis = []
lambdas_chis = []

gap = lambdas[1] - lambdas[0]

for x_k in range(len(xs)):
    chiral_chis = []
    for lam_k in range(1, len(lambdas) - 1):
        chiral_chi = (Chiral_condensates_by_x[x_k][lam_k + 1] - Chiral_condensates_by_x[x_k][lam_k - 1]) / (2 * gap)
        chiral_chis.append(abs(chiral_chi))
    Chiral_chis.append(chiral_chis)

print("="*10)
print("="*10)
print("xs =", xs)
print("lambdas =", lambdas.tolist())
print("Zero_points =", Zero_points)
print("Energies =", Energies)
print("Chiral_condensates_by_x=", Chiral_condensates_by_x)
print("Chiral_chis =", Chiral_chis)


label_x = r"$\lambda$"
label_y = r"\mathcal{C}"
data_x_line = lambdas[1:-1]
data_y_line = Chiral_chis

labels_line = [r"$x = " + str(x) + r"$" for x in xs]
label_save_title = "chiral_chi"
save = True

nice_scatter_plotter(
    data_x=None, data_y=None, data_y_errors=None,
    data_x_line=data_x_line, data_y_line=data_x_line,
    label_x=label_x, label_y=label_y, label_title="",
    save=False, label_save_title=label_save_title, marker="-x",
    labels=None, labels_line=labels_line,
    log_x_scale=False, log_y_scale=False,
    same_color=False, square = False
)