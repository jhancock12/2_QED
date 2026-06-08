# Standard libraries

# Local modules
from _dont_release import full_runner

# Third-party libraries
import qiskit_aer

lattice_parameters = {
    'L_x' : 2,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1)],
    'charge_site' : (),
    'anticharge_site' : (),
    'background_field' : [0.0, 0.0]
}

vqe_parameters = {
    'n_fermion_layers' : 2,
    'shots' : 10000,
    'simulator' : qiskit_aer.AerSimulator(),
    'MEM' : False,
    'SV' : False
}

qed_parameters = {
    'm' : 3.0, 
    'g' : 1.0, 
    'a' : 1.0, 
    'charge_weight' : 1000.0
}

SPSA_parameters = {
    'max_iters' : 10000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 0.08,
    'c' : 0.03,
    'prints' : False,
    'diagnostics' : False
}

full_runner(lattice_parameters, qed_parameters, vqe_parameters, SPSA_parameters)