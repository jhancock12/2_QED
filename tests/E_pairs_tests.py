# Standard libraries

# Local modules
from lattice_class import Lattice
from observables import qed_hamiltonian
from solvers.sparse import sparse_statevector_solver, solve_and_observe_sparse
from global_helpers import smart_round

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
    'charge_site' : (0, 0),
    'anticharge_site' : (2, 1),
    'background_field' : [0.0, 0.0]
}

qed_parameters = {
    'm' : 3.0, 
    'g' : 1.0, 
    'a' : 1.0, 
    'charge_weight' : 100.0
}

def delta_E_pairs(E_0, g):
    lattice_parameters = {
        'L_x' : 3,
        'L_y' : 2,
        'n_g' : 2,
        'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
        'charge_site' : (0, 0),
        'anticharge_site' : (2, 1),
        'background_field' : [E_0, E_0]
    }

    lattice_charges_BEF = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                lattice_parameters['background_field'])

    lattice_parameters = {
        'L_x' : 3,
        'L_y' : 2,
        'n_g' : 2,
        'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
        'charge_site' : (0, 0),
        'anticharge_site' : (2, 1),
        'background_field' : [0.0, 0.0]
    }

    lattice_charges_no_BEF = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                lattice_parameters['background_field'])

    lattice_parameters = {
        'L_x' : 3,
        'L_y' : 2,
        'n_g' : 2,
        'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
        'charge_site' : (),
        'anticharge_site' : (),
        'background_field' : [E_0, E_0]
    }

    lattice_no_charges_BEF = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                lattice_parameters['background_field'])

    lattice_parameters = {
        'L_x' : 3,
        'L_y' : 2,
        'n_g' : 2,
        'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
        'charge_site' : (),
        'anticharge_site' : (),
        'background_field' : [0.0, 0.0]
    }

    lattice_no_charges_no_BEF = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                lattice_parameters['background_field'])

    qed_parameters = {
        'm' : 3.0, 
        'g' : g, 
        'a' : 1.0, 
        'charge_weight' : 100.0
    }

    hamiltonian_charges_BEF = qed_hamiltonian(qed_parameters, lattice_charges_BEF)
    hamiltonian_charges_no_BEF = qed_hamiltonian(qed_parameters, lattice_charges_no_BEF)
    hamiltonian_no_charges_BEF = qed_hamiltonian(qed_parameters, lattice_no_charges_BEF)
    hamiltonian_no_charges_no_BEF = qed_hamiltonian(qed_parameters, lattice_no_charges_no_BEF)

    E_charge_BEF, _ = sparse_statevector_solver(hamiltonian_charges_BEF)
    E_charges_no_BEF, _ = sparse_statevector_solver(hamiltonian_charges_no_BEF)
    E_no_charges_BEF, _ = sparse_statevector_solver(hamiltonian_no_charges_BEF)
    E_no_charges_no_BEF, _ = sparse_statevector_solver(hamiltonian_no_charges_no_BEF)

    delta_E_charges = E_charge_BEF - E_charges_no_BEF
    delta_E_vacuum = E_no_charges_BEF - E_no_charges_no_BEF 

    delta_e_pairs = delta_E_charges - delta_E_vacuum

    return delta_e_pairs

gs = np.linspace(0.3, 3.0, 15)
E_0s = np.linspace(0.75, 1.25, 5)

total = len(gs) * len(E_0s)

counter = 0
Scores = []
for e_0 in E_0s:
    scores = []
    for g in gs:
        score = delta_E_pairs(e_0, g)
        scores.append(score)
        counter += 1
        print(smart_round((counter / total) * 100, 4),"% done")
    Scores.append(scores)

labels = [r"$E_0 = "+str(smart_round(e_0, 3))+r"$" for e_0 in E_0s]

plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'lines.markersize': 8,
        'lines.linewidth': 1.5
    })

plt.figure(figsize = (10,6), dpi = 100)
for k in range(len(Scores)):
    scores = Scores[k]
    if E_0s[k] < 1.0:
        colour = "tab:blue"
    elif E_0s[k] == 1.0:
        colour = "tab:red"
    else:
        colour = "tab:orange"

    plt.plot(
        gs,
        scores,
        color = colour,
        label = labels[k]
    )
plt.xlabel(r"$g$")
plt.ylabel(r"$\Delta E_{\rm pairs}$")
plt.legend()
plt.savefig("delta_E_pairs_Es_2" + ".pdf", bbox_inches='tight')
