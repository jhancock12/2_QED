# Standard libraries
import time

# Local modules
from lattice_class import Lattice
from observables import qed_hamiltonian

from circuits.circuit_class import CircuitForLattice
from circuits.measurer_class import CircuitMeasurer

# Third-party libraries
import numpy as np
import qiskit
import qiskit_aer

# Things that I want in this class:

# Symmetry Verification
# ZNE

parameters = {
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'charge_weight': 10000.0
}

n_fermion_layers = 1
simulator = qiskit_aer.AerSimulator()
shots = 100

lattice = Lattice(
    L_x = 2,
    L_y = 2,
    n_g = 2,
    dynamical_links_list = [((0, 0), 1), ((1, 0), 2)],
    charge_site = (),
    anticharge_site = (),
    background_field = [0.0, 0.0]
)

hamiltonian = qed_hamiltonian(parameters, lattice)
circuit_class = CircuitForLattice(lattice, n_fermion_layers)
circuit_class.build_circuit()
measurer = CircuitMeasurer(circuit_class, hamiltonian, simulator, shots)
tik0 = time.perf_counter()
measurer.build_measurement_error_mitigation()
tok0 = time.perf_counter()


thetas = [0.1]*len(measurer.circuit_class.thetas)
tik1 = time.perf_counter()
measurer.bind_values(thetas)
tok1 = time.perf_counter()

tik2 = time.perf_counter()
expected_value = measurer.expected_value_hamiltonian()
tok2 = time.perf_counter()

print(expected_value)
print("Time to build MEM:", round(tok0-tik0, 6))
print("Time to bind values:", round(tok1-tik1, 6))
print("Time to calculate EV:", round(tok2-tik2, 6))