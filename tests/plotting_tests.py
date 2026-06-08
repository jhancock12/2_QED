# Standard libraries

# Local modules
from lattice_class import Lattice
from solvers.sparse import solve_and_observe_sparse
from visualization import geometry_from_lattice_scaled, lattice_geometry_plotter, lattice_observable_plotter
from observables import qed_hamiltonian

# Third-party libraries
import numpy as np


parameters = {
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'charge_weight': 10000.0
}

lattice = Lattice(
    L_x = 2,
    L_y = 2,
    n_g = 2,
    dynamical_links_list = [((0, 0), 1), ((1, 0), 2)],
    charge_site = (),
    anticharge_site = (),
    background_field = [0.0, 0.0]
)

site_positions, link_pairs = geometry_from_lattice_scaled(lattice, sx = 1.6, sy = 1.2)

lattice_geometry_plotter(
    lattice,
    site_positions = site_positions,
    link_pairs = link_pairs,
    label_title = None,
    save = True,
    label_save_title = "temp_geometry",
    figsize = (7, 5)
)

hamiltonian = qed_hamiltonian(parameters, lattice)

results_dict = solve_and_observe_sparse(hamiltonian, lattice)

lattice_observable_plotter(
    lattice,
    results_dict,
    site_positions = site_positions,
    link_pairs = link_pairs,
    link_values_key = 'electric_field_dict',
    charge_values_key = 'charge_field_dict',
    label_title = None,
    save = True,
    label_save_title = "temp_observables",
    figsize = (7, 5)
)