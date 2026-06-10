# Standard libraries
from copy import copy

# Local modules
from observables import qed_hamiltonian
from circuits.circuit_class import CircuitForLattice
from circuits.measurer_class import CircuitMeasurer
from lattice_class import Lattice
from solvers.noiseless import SPSA_vqe_solver_noiseless

# Third-party libraries
import qiskit_aer
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

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
    'n_fermion_layers' : 1,
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
    'max_iters' : 10000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 0.08,
    'c' : 0.03,
    'prints' : False,
    'diagnostics' : False
}

lattice = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                    lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                    lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                    lattice_parameters['background_field'])

circuit_class = CircuitForLattice(lattice, vqe_parameters['n_fermion_layers'])
# -------------------------------
fake_backend = FakeCairoV2()
noisy_simulator = qiskit_aer.AerSimulator.from_backend(fake_backend)
accepted_strings = lattice.SV_strings()

hamiltonian = qed_hamiltonian(qed_parameters, lattice, mass_multi = 1, electric_multi = 1, magnetic_multi = 1, kinetic_multi = 1)
results = SPSA_vqe_solver_noiseless(hamiltonian, lattice, vqe_parameters, SPSA_parameters)

hamiltonian_electric = qed_hamiltonian(qed_parameters, lattice, mass_multi = 0, electric_multi = 1, magnetic_multi = 0, kinetic_multi = 0)

measurer_noiseless = CircuitMeasurer(circuit_class, hamiltonian, qiskit_aer.AerSimulator(), vqe_parameters['shots'])

measurer_noisy = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])

measurer_noisy_SV_only = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
measurer_noisy_SV_only.add_SV(accepted_strings)

measurer_noiseless.bind_values(results['final_paras'])
measurer_noisy.bind_values(results['final_paras'])
measurer_noisy_SV_only.bind_values(results['final_paras'])

print("accepted_strings:", accepted_strings)

measurer_noiseless.change_hamiltonian(hamiltonian)
measurer_noisy.change_hamiltonian(hamiltonian)
measurer_noisy_SV_only.change_hamiltonian(hamiltonian)
energy_noiseless = measurer_noiseless.expected_value_hamiltonian_selective_SV()
energy_noisy = measurer_noisy.expected_value_hamiltonian_selective_SV()
energy_noisy_SV = measurer_noisy_SV_only.expected_value_hamiltonian_selective_SV()
print("Full energies:")
print("Noiseless:", energy_noiseless)
print("Noisy:", energy_noisy)
print("SV:", energy_noisy_SV)

measurer_noiseless.change_hamiltonian(hamiltonian_electric)
measurer_noisy.change_hamiltonian(hamiltonian_electric)
measurer_noisy_SV_only.change_hamiltonian(hamiltonian_electric)
electric_energy_noiseless = measurer_noiseless.expected_value_hamiltonian_selective_SV()
electric_energy_noisy = measurer_noisy.expected_value_hamiltonian_selective_SV()
electric_energy_noisy_SV = measurer_noisy_SV_only.expected_value_hamiltonian_selective_SV()
print("Electric energies:")
print("Noiseless:", electric_energy_noiseless)
print("Noisy:", electric_energy_noisy)
print("SV:", electric_energy_noisy_SV)

measurer_classes.append(copy(measurer_classes[-1]))
    names.append(str(names[-1]) + " + ZNE")
    for measurer_class_k in range(len(measurer_classes)):
        for k in range(len(hamiltonians)):
            print(f"Energy of {names[measurer_class_k]} on {hamiltonian_names[k]} Hamiltonian:")
            measurer_classes[measurer_class_k].bind_values(final_thetas)
            measurer_classes[measurer_class_k].change_hamiltonian(hamiltonians[k])
            if "ZNE" in names[measurer_class_k]:
                energy = measurer_classes[measurer_class_k].ZNE_expected_value_hamiltonian()
            else:
                energy = measurer_classes[measurer_class_k].expected_value_hamiltonian_selective_SV()
            print(energy)