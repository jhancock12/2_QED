# Standard libraries

# Local modules
from _dont_release import full_runner

# Third-party libraries
import numpy as np
import qiskit_aer

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1), ((1,0), 2)],
    'charge_site' : (0, 0),
    'anticharge_site' : (2, 1),
    'background_field' : [0.0, 0.0]
}

vqe_parameters = {
    'n_fermion_layers' : 2,
    'shots' : 150000,
    'simulator' : qiskit_aer.AerSimulator(),
    'MEM' : False,
    'SV' : False
}

qed_parameters = {
    'm' : 3.0, 
    'g' : 1.0, 
    'a' : 1.0, 
    'charge_weight' : 0.0
}

SPSA_parameters = {
    'max_iters' : 40000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 0.08,
    'c' : 0.03,
    'prints' : False,
    'diagnostics' : False
}

gs = np.linspace(0.3, 3.0, 5)
for g in gs:
    qed_parameters['g'] = g
    print("="*10)
    print("g =",g)
    print("="*10)
    full_runner(lattice_parameters, qed_parameters, vqe_parameters, SPSA_parameters)

# =====
# Standard libraries

# Local modules
from lattice_class import Lattice
from visualization import geometry_from_lattice_scaled, lattice_geometry_plotter, lattice_observable_plotter

from data.qed_results_lists import results_no_quantum_noise, results_no_QEM, results_MEM_only, results_SV_only, results_both, results_both_ZNE, results_sparse, gs

# Third-party libraries
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

parameters = {
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'charge_weight': 10000.0
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

site_positions, link_pairs = geometry_from_lattice_scaled(lattice, sx = 1.0, sy = 1.0)

full_results = {"No_QEM": results_no_QEM, 
                "MEM_only": results_MEM_only, 
                "SV_only": results_SV_only, 
                "MEM_SV": results_both, 
                "MEM_SV_ZNE": results_both_ZNE, 
                "Numerical_diagonalization": results_sparse}

reduced_results = {"No_QEM": results_no_QEM,
                "MEM_SV_ZNE": results_both_ZNE, 
                "Numerical_diagonalization": results_sparse}

for k in range(len(gs)):
    for name in reduced_results:
        results_dict = reduced_results[name][k]

        lattice_observable_plotter(
            lattice,
            results_dict,
            site_positions = site_positions,
            link_pairs = link_pairs,
            link_values_key = 'electric_field_dict',
            charge_values_key = 'charge_field_dict',
            label_title = None,
            save = True,
            label_save_title = name + "_" + str(k),
            figsize = (7, 5)
        )